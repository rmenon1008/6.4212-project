"""The authors' pretrained Diffusion Policy on the TRM test seeds, as a reference point.

The network and checkpoint are from https://github.com/real-stanford/diffusion_policy
(Chi et al., RSS 2023), set up as in the course's diffusion_policy_pusht_state.ipynb.
Nothing here trains.  Results land in runs/pusht/dp/eval.json, which
`trm_pusht.py report` picks up.  The checkpoint saw all 206 demonstrations; the
TRM saw 185.

    python pusht_dp_baseline.py                   # downloads about 260 MB on first use
    python pusht_dp_baseline.py --assets <folder> # or reuse a folder that has them
"""

import argparse
import collections
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from pusht import PushTEnv
from trm_pusht import DATASET, MAX_STEPS, RUNS, TEST_SEED, load_demonstrations

ASSETS = Path("data/pusht/diffusion")
CHECKPOINT = ("pusht_state_100ep.ckpt", "1mHDr_DEZSdiGo9yecL50BBQYzR8Fjhl_")  # Google Drive id
UPSTREAM_COMMIT = "5ba07ac6661db573af695b419a7947ecb704690f"
OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON = 2, 16, 8
NOW = OBS_HORIZON - 1  # The plan starts at the older observation, so "now" is index 1.
ITERATIONS = 100

# On ARM CPUs, NNPACK convolutions are pathological for batches (see the notebook).
torch.backends.nnpack.set_flags(False)


def fetch_assets(assets=ASSETS):
    """The checkpoint and the authors' network source, downloaded once into `assets`."""
    assets = Path(assets)
    assets.mkdir(parents=True, exist_ok=True)
    checkpoint = assets / CHECKPOINT[0]
    if not checkpoint.is_file():
        import gdown

        gdown.download(id=CHECKPOINT[1], output=str(checkpoint), quiet=False)
    root = assets / f"diffusion_policy-{UPSTREAM_COMMIT}"
    if not (root / "diffusion_policy").is_dir():
        url = f"https://github.com/real-stanford/diffusion_policy/archive/{UPSTREAM_COMMIT}.zip"
        with urllib.request.urlopen(url) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        prefix = f"diffusion_policy-{UPSTREAM_COMMIT}/diffusion_policy/"
        names = [n for n in archive.namelist() if n.startswith(prefix) and n.endswith(".py")]
        archive.extractall(assets, names)
    return checkpoint, root


class DiffusionPlanner:
    """The pretrained network and its sampler, normalized as it was trained."""

    def __init__(self, assets=ASSETS, device="cpu", dataset=DATASET, seed=0):
        checkpoint, root = fetch_assets(assets)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

        self.device = torch.device(device)
        self.net = ConditionalUnet1D(
            input_dim=2, global_cond_dim=5 * OBS_HORIZON, kernel_size=5, cond_predict_scale=True
        )
        self.net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.net.to(self.device).eval()
        self.scheduler = DDPMScheduler(
            num_train_timesteps=ITERATIONS,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        demos = load_demonstrations(dataset)
        self.stats = {k: (demos[k].min(0), demos[k].max(0)) for k in ("state", "action")}
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    @torch.no_grad()
    def sample(self, history):
        """Reverse diffusion for a batch of (OBS_HORIZON, 5) histories; plans in pixels."""
        low, high = self.stats["state"]
        cond = torch.as_tensor((history - low) / (high - low) * 2 - 1, dtype=torch.float32)
        cond = cond.flatten(start_dim=1).to(self.device)
        x = torch.randn(
            (len(history), PRED_HORIZON, 2), generator=self.generator, device=self.device
        )
        self.scheduler.set_timesteps(ITERATIONS)
        for k in self.scheduler.timesteps:
            noise = self.net(sample=x, timestep=k, global_cond=cond)
            x = self.scheduler.step(noise, k, x, generator=self.generator).prev_sample
        low, high = self.stats["action"]
        return (x.cpu().numpy() + 1) / 2 * (high - low) + low

    def parameters(self):
        return sum(p.numel() for p in self.net.parameters())


def latency_ms(planner, state, repeats=3):
    """Median time for one plan from one observation, after a warm-up plan."""
    history = np.repeat(state[None, None], OBS_HORIZON, axis=1)
    times = []
    for _ in range(repeats + 1):
        start = time.perf_counter()
        planner.sample(history)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times[1:]))


def evaluate(planner, episodes):
    """Best coverage per test episode, executing 8 actions per plan."""
    envs = [PushTEnv(seed=TEST_SEED + i) for i in range(episodes)]
    histories = [
        collections.deque([e.observation()] * OBS_HORIZON, maxlen=OBS_HORIZON) for e in envs
    ]
    best = np.zeros(len(envs))
    step = 0
    while step < MAX_STEPS:
        actions = planner.sample(np.array([np.stack(h) for h in histories]))
        for t in range(NOW, NOW + ACTION_HORIZON):
            for i, env in enumerate(envs):
                obs, c, _ = env.step(actions[i, t])
                histories[i].append(obs)
                best[i] = max(best[i], c)
            step += 1
            if step == MAX_STEPS:
                break
        print(f"step {step}: mean coverage {best.mean():.3f}", flush=True)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--output", type=Path, default=RUNS / "dp")
    args = parser.parse_args()
    planner = DiffusionPlanner(args.assets, args.device, args.dataset)
    # Latency is one plan for one observation on the CPU, as for the TRM.
    cpu = DiffusionPlanner(args.assets, "cpu", args.dataset)
    latency = latency_ms(cpu, load_demonstrations(args.dataset)["state"][0])
    print(f"latency: {latency:.0f} ms per plan", flush=True)
    best = evaluate(planner, args.episodes)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "coverage": best.tolist(),
        "latency_ms": latency,
        "iterations": ITERATIONS,
        "parameters": planner.parameters(),
    }
    (args.output / "eval.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main()
