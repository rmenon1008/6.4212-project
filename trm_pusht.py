"""Tiny recursive policy on state-based Push-T: a CPU go/no-go test.

Asks three questions of one trained model: is it a working policy, do more
recursions give better actions, and does per-state halting beat simply picking a
smaller fixed depth?  The model follows trm_pusht_architecture.tex (MVP setting
n=2, T=1); halting is added after training and leaves the model unchanged.

    python trm_pusht.py check          # harness and model sanity checks
    python trm_pusht.py run --smoke    # one epoch, end to end
    python trm_pusht.py run            # full experiment and report
"""

import argparse
import copy
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pusht import PushTEnv, coverage

DATASET = Path("data/pusht/pusht_cchi_v7_replay.zarr.zip")
DATASET_DRIVE_ID = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq"
RUNS = Path("runs/pusht")

OBS_DIM = 12
PRED_HORIZON = 16
ACTION_HORIZON = 8
ACTION_DIM = PRED_HORIZON * 2
MAX_STEPS = 300
TEST_SEED, DEV_SEED = 100000, 200000
DEPTHS = (1, 2, 4, 8, 16)
GATE_DEPTHS = (1, 2, 4, 8)  # Depths within the training range.
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)  # Convergence thresholds, from held-out deltas.
Q_PROBS = (0.1, 0.3, 0.5, 0.7, 0.9)


@dataclass
class TrainingConfig:
    """Training and model hyperparameters container."""

    arch: str = "trm"  # "trm", or "mlp" for the parameter-matched baseline.
    hidden_dim: int = 128
    n: int = 2  # Latent updates per cycle.
    T: int = 1  # Cycles per supervision step; the first T-1 run without gradient.
    n_sup: int = 8
    epochs: int = 100
    batch_size: int = 256
    lr: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    val_fraction: float = 0.1
    seed: int = 0
    threads: int = 2


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


######################################################################
## Data
######################################################################


def features(states):
    """(..., 5) environment states to (..., 6): positions in [-1, 1], sin, cos."""
    states = np.asarray(states, dtype=np.float32)
    return np.concatenate(
        [states[..., :4] / 256 - 1, np.sin(states[..., 4:5]), np.cos(states[..., 4:5])], axis=-1
    )


def to_pixels(actions):
    return (actions + 1) * 256


def load_demonstrations(path=DATASET):
    """The demonstrations: state (N, 5), action (N, 2), and episode_ends."""
    import zarr

    path = Path(path)
    if not path.is_file():
        import gdown

        path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=DATASET_DRIVE_ID, output=str(path), quiet=False)
    root = zarr.open(zarr.storage.ZipStore(path, mode="r"), mode="r")
    return {
        "state": root["data"]["state"][:],
        "action": root["data"]["action"][:],
        "episode_ends": root["meta"]["episode_ends"][:],
    }


def make_samples(demos, episodes):
    """Pair (s[t-1], s[t]) with the chunk a[t:t+16], so index 0 is the action for now.

    The start repeats s[0], as a rollout does; the end repeats the last action.
    """
    ends = demos["episode_ends"]
    starts = np.r_[0, ends[:-1]]
    obs, chunks = [], []
    for start, end in zip(starts[episodes], ends[episodes]):
        t = np.arange(start, end)
        previous = np.maximum(t - 1, start)
        history = [features(demos["state"][previous]), features(demos["state"][t])]
        obs.append(np.concatenate(history, -1))
        future = np.minimum(t[:, None] + np.arange(PRED_HORIZON), end - 1)
        chunks.append((demos["action"][future] / 256 - 1).reshape(len(t), ACTION_DIM))
    return (
        torch.from_numpy(np.concatenate(obs)).float(),
        torch.from_numpy(np.concatenate(chunks)).float(),
    )


def split_samples(demos, val_fraction=0.1):
    """Hold out the last episodes for open-loop error."""
    count = len(demos["episode_ends"])
    split = int((1 - val_fraction) * count)
    return make_samples(demos, np.arange(split)), make_samples(demos, np.arange(split, count))


######################################################################
## Models
######################################################################


class ResidualBlock(nn.Module):
    """RB(h) = LN(h + W2 GELU(W1 h)), post-norm so that repeated use stays bounded."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.w2 = nn.Linear(4 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h):
        return self.norm(h + self.w2(F.gelu(self.w1(h))))


class TRMPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim, self.n, self.T = config.hidden_dim, config.n, config.T
        self.f_per_step = config.T * (config.n + 1)
        self.embed = nn.Linear(OBS_DIM, config.hidden_dim)
        self.f = nn.Sequential(ResidualBlock(config.hidden_dim), ResidualBlock(config.hidden_dim))
        self.head = nn.Linear(config.hidden_dim, ACTION_DIM)

    def init_state(self, batch):
        zeros = torch.zeros(batch, self.hidden_dim)
        return zeros, zeros.clone()

    def cycle(self, x, y, z):
        for _ in range(self.n):
            z = self.f(x + y + z)
        return self.f(y + z), z

    def supervision_step(self, obs, y, z):
        x = self.embed(obs)
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self.cycle(x, y, z)
        return self.cycle(x, y, z)

    def unroll(self, obs, depth):
        """Every intermediate answer: chunks (depth, B, 32) and states (depth, B, d)."""
        y, z = self.init_state(len(obs))
        states = []
        for _ in range(depth):
            y, z = self.supervision_step(obs, y, z)
            states.append(y)
        states = torch.stack(states)
        return self.head(states), states


class MLPPolicy(nn.Module):
    """The same embedding, f and head, applied once: recursion removed, parameters kept."""

    f_per_step = 1

    def __init__(self, config):
        super().__init__()
        self.embed = nn.Linear(OBS_DIM, config.hidden_dim)
        self.f = nn.Sequential(ResidualBlock(config.hidden_dim), ResidualBlock(config.hidden_dim))
        self.head = nn.Linear(config.hidden_dim, ACTION_DIM)

    def forward(self, obs):
        return self.head(self.f(self.embed(obs)))

    def unroll(self, obs, depth=1):
        states = self.f(self.embed(obs))[None]
        return self.head(states), states


def build_policy(config):
    return {"trm": TRMPolicy, "mlp": MLPPolicy}[config.arch](config)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


######################################################################
## Training
######################################################################


def open_loop_error(model, obs, chunks, depth):
    """Mean absolute error in pixels at each depth."""
    with torch.no_grad():
        predicted, _ = model.unroll(obs, depth)
    return ((predicted - chunks).abs().mean((1, 2)) * 256).tolist()


def train(config, output, dataset=DATASET):
    """Deep supervision: L1, an optimizer step, and a detach at every supervision step."""
    torch.set_num_threads(config.threads)
    seed_everything(config.seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (train_obs, train_chunks), (val_obs, val_chunks) = split_samples(
        load_demonstrations(dataset), config.val_fraction
    )
    model = build_policy(config)
    ema = copy.deepcopy(model).requires_grad_(False)
    params, ema_params = list(model.parameters()), list(ema.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    batches = len(train_obs) // config.batch_size
    steps_per_batch = config.n_sup if config.arch == "trm" else 1
    total = config.epochs * batches * steps_per_batch

    def schedule(step):
        if step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        progress = (step - config.warmup_steps) / max(1, total - config.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    def update(loss):
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, config.grad_clip)
        optimizer.step()
        scheduler.step()
        torch._foreach_lerp_(ema_params, params, 1 - config.ema_decay)

    depth = config.n_sup if config.arch == "trm" else 1
    log_every = max(1, config.epochs // 10)
    history, start = [], time.time()
    for epoch in range(config.epochs):
        order = torch.randperm(len(train_obs))
        first = last = 0.0
        for b in range(batches):
            index = order[b * config.batch_size : (b + 1) * config.batch_size]
            obs, target = train_obs[index], train_chunks[index]
            if config.arch == "trm":
                y, z = model.init_state(len(obs))
                for step in range(config.n_sup):
                    y, z = model.supervision_step(obs, y, z)
                    loss = F.l1_loss(model.head(y), target)
                    update(loss)
                    y, z = y.detach(), z.detach()
                    first += loss.item() if step == 0 else 0.0
                last += loss.item()
            else:
                loss = F.l1_loss(model(obs), target)
                update(loss)
                first += loss.item()
                last += loss.item()
        if (epoch + 1) % log_every == 0 or epoch + 1 == config.epochs:
            row = {
                "epoch": epoch + 1,
                "minutes": (time.time() - start) / 60,
                "train_l1_px_first_step": first / batches * 256,
                "train_l1_px_last_step": last / batches * 256,
                "val_l1_px_by_depth": open_loop_error(ema, val_obs, val_chunks, depth),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            (output / "history.json").write_text(json.dumps(history, indent=2))
    torch.save({"config": asdict(config), "model": ema.state_dict()}, output / "model.pt")
    return ema


######################################################################
## Closed loop
######################################################################


def rollout(planner, starts, max_steps=MAX_STEPS, action_horizon=ACTION_HORIZON):
    """Run a batch of episodes; each start is a seed, or a state to begin from.

    planner(obs (B, 12), step) returns (actions (B, >=horizon, 2), f_calls (B,)).
    Returns each episode's best coverage and its mean f calls per plan.
    """
    envs = [
        PushTEnv(seed=int(s)) if np.ndim(s) == 0 else PushTEnv(np.asarray(s)) for s in starts
    ]
    current = np.stack([e.observation() for e in envs])
    previous = current.copy()
    best = np.zeros(len(envs))
    calls, step = [], 0
    while step < max_steps:
        obs = np.concatenate([features(previous), features(current)], axis=-1)
        actions, f_calls = planner(torch.from_numpy(obs), step)
        calls.append(f_calls)
        for t in range(min(action_horizon, max_steps - step)):
            following = []
            for i, env in enumerate(envs):
                state, c, _ = env.step(actions[i, t])
                best[i] = max(best[i], c)
                following.append(state)
            previous, current = current, np.stack(following)
            step += 1
    return best, np.mean(calls, axis=0)


def first_stop(stop):
    """Depth at which each sample halts, from a (depth, B) bool; the last step always stops."""
    stop = stop.clone()
    stop[-1] = True
    return stop.float().argmax(0)


def convergence_stop(chunks, epsilon):
    """Stop once the answer moves less than epsilon pixels; needs two answers to compare."""
    delta = (chunks[1:] - chunks[:-1]).abs().mean(-1) * 256
    return torch.cat([torch.zeros_like(delta[:1], dtype=torch.bool), delta < epsilon])


def make_planner(model, depth, stop_rule=None):
    """A policy capped at `depth`, halting per sample when stop_rule(chunks, states) fires."""

    def planner(obs, step):
        with torch.no_grad():
            chunks, states = model.unroll(obs, depth)
            if stop_rule is None:
                used = torch.full((len(obs),), len(chunks) - 1)
            else:
                used = first_stop(stop_rule(chunks, states))
            chosen = chunks[used, torch.arange(len(obs))]
        actions = to_pixels(chosen).reshape(len(obs), PRED_HORIZON, 2).numpy()
        return actions, ((used + 1) * model.f_per_step).numpy().astype(float)

    return planner


def train_q_head(model, obs, chunks, depth, seed):
    """The proposal's Q-head, on a frozen model: is y_s within tolerance of the demo?

    Fit on held-out episodes, because the model's error on its own training
    states understates its error in use.  The tolerance is the median error at
    the final depth.  Returns the head and the tolerance in pixels.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        predicted, states = model.unroll(obs, depth)
    error = (predicted - chunks).abs().mean(-1) * 256
    tolerance = error[-1].median().item()
    labels = (error <= tolerance).float().flatten()
    inputs = states.flatten(0, 1)
    head = nn.Sequential(nn.Linear(states.shape[-1], 64), nn.GELU(), nn.Linear(64, 1))
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3)
    for _ in range(400):
        loss = F.binary_cross_entropy_with_logits(head(inputs).squeeze(-1), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.requires_grad_(False), tolerance


def latency_ms(model, depth, repeats=300):
    obs = torch.randn(1, OBS_DIM)
    with torch.no_grad():
        model.unroll(obs, depth)
        start = time.perf_counter()
        for _ in range(repeats):
            model.unroll(obs, depth)
    return (time.perf_counter() - start) / repeats * 1000


def evaluate(run_dir, dataset=DATASET, episodes=200, threads=1):
    """Every fixed depth and halting rule on dev and test seeds, from the same starts."""
    torch.set_num_threads(threads)
    run_dir = Path(run_dir)
    stored = torch.load(run_dir / "model.pt", weights_only=True)
    config = TrainingConfig(**stored["config"])
    model = build_policy(config)
    model.load_state_dict(stored["model"])
    model.eval()
    _, (val_obs, val_chunks) = split_samples(load_demonstrations(dataset), config.val_fraction)
    report = {"arch": config.arch, "seed": config.seed, "f_per_step": model.f_per_step}
    report["parameters"] = count_parameters(model)

    if config.arch == "mlp":
        configs = {"fixed_1": {"kind": "fixed", "depth": 1, "planner": make_planner(model, 1)}}
        report["open_loop_l1_px"] = open_loop_error(model, val_obs, val_chunks, 1)
        report["latency_ms"] = {"1": latency_ms(model, 1)}
    else:
        cap = config.n_sup
        report["open_loop_l1_px"] = open_loop_error(model, val_obs, val_chunks, max(DEPTHS))
        report["latency_ms"] = {str(k): latency_ms(model, k) for k in DEPTHS}
        with torch.no_grad():
            chunks, _ = model.unroll(val_obs, cap)
        delta = (chunks[1:] - chunks[:-1]).abs().mean(-1) * 256
        epsilons = torch.quantile(delta.flatten(), torch.tensor(QUANTILES)).tolist()
        report["delta_px_by_step"] = delta.mean(1).tolist()
        # How much depth each held-out state needs to land near its final answer.
        gap = (chunks - chunks[-1]).abs().mean(-1) * 256
        report["headroom"] = {
            str(tolerance): np.bincount(
                first_stop(gap <= tolerance).numpy() + 1, minlength=cap + 1
            )[1:].tolist()
            for tolerance in (1, 2, 4)
        }
        q_head, tolerance = train_q_head(model, val_obs, val_chunks, cap, config.seed)
        report["q_tolerance_px"] = tolerance

        configs = {
            f"fixed_{k}": {"kind": "fixed", "depth": k, "planner": make_planner(model, k)}
            for k in DEPTHS
        }
        for i, epsilon in enumerate(epsilons):
            configs[f"convergence_{i}"] = {
                "kind": "convergence",
                "epsilon_px": epsilon,
                "planner": make_planner(model, cap, lambda c, s, e=epsilon: convergence_stop(c, e)),
            }
        for i, probability in enumerate(Q_PROBS):
            configs[f"q_{i}"] = {
                "kind": "q",
                "probability": probability,
                "planner": make_planner(
                    model,
                    cap,
                    lambda c, s, p=probability: torch.sigmoid(q_head(s).squeeze(-1)) > p,
                ),
            }

    report["sets"] = {}
    for name, first_seed in (("test", TEST_SEED), ("dev", DEV_SEED)):
        seeds = np.arange(first_seed, first_seed + episodes)
        results = {}
        for key, entry in configs.items():
            best, calls = rollout(entry["planner"], seeds)
            results[key] = {k: v for k, v in entry.items() if k != "planner"}
            results[key].update(coverage=best.tolist(), f_calls=calls.tolist())
            print(name, key, f"{best.mean():.3f}", f"{calls.mean():.2f}", flush=True)
        report["sets"][name] = results
    (run_dir / "eval.json").write_text(json.dumps(report))
    return report


######################################################################
## Report
######################################################################


def bootstrap_ci(values, resamples=4000, seed=0):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def replay_ceiling(dataset=DATASET):
    """Best coverage when the human actions are replayed through the rollout harness."""
    demos = load_demonstrations(dataset)
    ends = demos["episode_ends"]
    scores = []
    for start, end in zip(np.r_[0, ends[:-1]], ends):
        scores.append(replay_episode(demos, start, end))
    return float(np.mean(scores))


def replay_episode(demos, start, end):
    """Best coverage from replaying one recorded episode's actions, 8 per plan."""
    actions = demos["action"][start:end]

    def planner(obs, step):
        index = np.minimum(step + np.arange(ACTION_HORIZON), len(actions) - 1)
        return actions[index][None], np.zeros(1)

    return rollout(planner, [demos["state"][start]], max_steps=end - start)[0][0]


def seed_mean(evals, split, key, field="coverage"):
    """Per-episode values averaged over training seeds, so episodes stay paired."""
    return np.mean([e["sets"][split][key][field] for e in evals], axis=0)


def fixed_table(evals, split, depths):
    return {k: seed_mean(evals, split, f"fixed_{k}") for k in depths}


def halting_gate(trm, name, count):
    """Pick the threshold on dev seeds, then judge it on test seeds."""
    f_per_step = trm[0]["f_per_step"]
    dev_fixed = fixed_table(trm, "dev", GATE_DEPTHS)
    test_fixed = fixed_table(trm, "test", GATE_DEPTHS)
    dev_best = max(v.mean() for v in dev_fixed.values())
    keys = [f"{name}_{i}" for i in range(count)]
    eligible = [k for k in keys if seed_mean(trm, "dev", k).mean() >= dev_best - 0.02]
    if not eligible:
        return {"rule": name, "passed": False, "reason": "no threshold within 0.02 on dev seeds"}
    key = min(eligible, key=lambda k: seed_mean(trm, "dev", k, "f_calls").mean())
    cov = seed_mean(trm, "test", key).mean()
    calls = seed_mean(trm, "test", key, "f_calls").mean()
    test_best = max(v.mean() for v in test_fixed.values())
    reaching = [k for k in GATE_DEPTHS if test_fixed[k].mean() >= cov]
    depth = min(reaching) if reaching else max(test_fixed, key=lambda k: test_fixed[k].mean())
    baseline = depth * f_per_step
    return {
        "rule": name,
        "threshold": key,
        "coverage": float(cov),
        "f_calls": float(calls),
        "best_fixed": float(test_best),
        "matching_depth": int(depth),
        "matching_f_calls": int(baseline),
        "saving": float(1 - calls / baseline),
        "passed": bool(cov >= test_best - 0.02 and calls <= 0.7 * baseline),
    }


def plot_results(trm, mlp, floor, ceiling, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    plt.rcParams.update(
        {
            "figure.facecolor": surface,
            "axes.facecolor": surface,
            "axes.edgecolor": grid,
            "axes.labelcolor": muted,
            "xtick.color": muted,
            "ytick.color": muted,
            "text.color": ink,
            "axes.grid": True,
            "grid.color": grid,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )

    def note(ax, text, xy, offset, color=muted, size=8):
        ax.annotate(text, xy, xytext=offset, textcoords="offset points", color=color, fontsize=size)

    def reference(ax, value, label):
        ax.axhline(value, color=muted, lw=1, ls="--")
        note(ax, label, (1, value), (0, 3))

    def trained_depth(ax, y):
        ax.axvline(TrainingConfig.n_sup, color=muted, lw=1, ls=":")
        note(ax, "trained to k=8", (TrainingConfig.n_sup, y), (4, 0))

    table = fixed_table(trm, "test", DEPTHS)
    means = [table[k].mean() for k in DEPTHS]
    bounds = np.array([bootstrap_ci(table[k]) for k in DEPTHS])
    mlp_cov = seed_mean(mlp, "test", "fixed_1").mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.fill_between(DEPTHS, bounds[:, 0], bounds[:, 1], color=blue, alpha=0.15, lw=0)
    ax.plot(DEPTHS, means, color=blue, lw=2, marker="o", ms=6, label="TRM")
    ax.set_xscale("log", base=2)
    ax.set_xticks(DEPTHS, [str(k) for k in DEPTHS])
    ax.minorticks_off()
    ax.set(xlabel="Recursion depth k (supervision steps)", ylabel="Mean max coverage", ylim=(0, 1))
    ax.set_title("Closed-loop coverage by depth", loc="left", fontsize=11)
    reference(ax, ceiling, f"Human replay {ceiling:.2f}")
    reference(ax, mlp_cov, f"Same-size MLP {mlp_cov:.2f}")
    reference(ax, floor, f"Do nothing {floor:.2f}")
    note(ax, "TRM", (2, means[1]), (0, 8), ink, 9)
    trained_depth(ax, 0.2)

    ax = axes[1]
    errors = np.mean([e["open_loop_l1_px"] for e in trm], axis=0)
    steps = np.arange(1, len(errors) + 1)
    ax.plot(steps, errors, color=blue, lw=2, marker="o", ms=6, label="TRM")
    trained_depth(ax, 1)
    note(ax, "TRM", (3, errors[2]), (0, -14), ink, 9)
    ax.set(xlabel="Recursion depth k (supervision steps)", ylabel="Held-out L1 error (pixels)")
    ax.set_ylim(bottom=0)
    ax.set_title("Open-loop error by depth", loc="left", fontsize=11)
    mlp_error = np.mean([e["open_loop_l1_px"][0] for e in mlp])
    reference(ax, mlp_error, f"Same-size MLP {mlp_error:.1f}")
    fig.tight_layout()
    fig.savefig(output / "depth_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    f_per_step = trm[0]["f_per_step"]
    calls = [k * f_per_step for k in GATE_DEPTHS]
    fixed = [table[k].mean() for k in GATE_DEPTHS]
    ax.plot(calls, fixed, color=blue, lw=2, marker="o", ms=6, label="Fixed depth")
    for name, count, color, label, marker in (
        ("convergence", len(QUANTILES), orange, "Convergence rule", "s"),
        ("q", len(Q_PROBS), aqua, "Q-head", "^"),
    ):
        keys = [f"{name}_{i}" for i in range(count)]
        points = sorted(
            (seed_mean(trm, "test", k, "f_calls").mean(), seed_mean(trm, "test", k).mean())
            for k in keys
        )
        ax.plot(*zip(*points), color=color, lw=2, marker=marker, ms=6, label=label)
    ax.set(xlabel="Mean f calls per plan", ylabel="Mean max coverage")
    ax.set_title("Halting against fixed depth (test seeds)", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "halting_pareto.png", dpi=160)
    plt.close(fig)


def report(root=RUNS, dataset=DATASET, minutes=None):
    root = Path(root)
    trm = [json.loads(p.read_text()) for p in sorted(root.glob("trm/seed_*/eval.json"))]
    mlp = [json.loads(p.read_text()) for p in sorted(root.glob("mlp/seed_*/eval.json"))]
    if not trm or not mlp:
        raise FileNotFoundError(f"No evaluations under {root}")
    episodes = len(trm[0]["sets"]["test"]["fixed_1"]["coverage"])
    floor = float(
        np.mean(
            [
                coverage(PushTEnv(seed=int(s)).observation()[2:5])
                for s in range(TEST_SEED, TEST_SEED + episodes)
            ]
        )
    )
    ceiling = replay_ceiling(dataset)
    table = fixed_table(trm, "test", DEPTHS)
    mlp_scores = seed_mean(mlp, "test", "fixed_1")
    best_depth = max(GATE_DEPTHS, key=lambda k: table[k].mean())

    best = table[best_depth].mean()
    margin = table[best_depth] - mlp_scores
    margin_ci = bootstrap_ci(margin)
    g1 = bool(best >= 0.60 and margin_ci[1] >= 0)
    gain = table[8] - table[1]
    gain_ci = bootstrap_ci(gain)
    g2 = bool(gain.mean() >= 0.05 and gain_ci[0] > 0)
    halting = [
        halting_gate(trm, "convergence", len(QUANTILES)),
        halting_gate(trm, "q", len(Q_PROBS)),
    ]
    g3 = any(h["passed"] for h in halting)

    def verdict(passed):
        return "PASS" if passed else "FAIL"

    lines = [
        "# TRM on Push-T: go/no-go report",
        "",
        f"{len(trm)} training seeds, {episodes} test episodes, {MAX_STEPS} steps each, "
        f"{trm[0]['parameters']:,} parameters.",
    ]
    timing = root / "minutes.txt"
    if minutes is not None:
        timing.write_text(f"{minutes:.1f}")
    if timing.is_file():
        lines.append(f"Total wall-clock time: {timing.read_text().strip()} minutes.")
    lines += [
        "",
        "## Gates",
        "",
        "| Gate | Result | Evidence |",
        "|---|---|---|",
        f"| G1 policy | {verdict(g1)} | best k={best_depth}: {best:.3f} coverage; "
        f"minus MLP {margin.mean():+.3f} (95% CI {margin_ci[0]:+.3f} to {margin_ci[1]:+.3f}) |",
        f"| G2 recursion | {verdict(g2)} | k=8 minus k=1: {gain.mean():+.3f} "
        f"(95% CI {gain_ci[0]:+.3f} to {gain_ci[1]:+.3f}) |",
        f"| G3 halting | {verdict(g3)} | see halting table |",
        "",
        "## Closed loop (test seeds)",
        "",
        "| Policy | f calls | Coverage | 95% CI | Success ≥0.90 | Success ≥0.95 "
        "| Latency (ms) | Per training seed |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]

    def row(label, scores, calls, latency, evals, key):
        ci = bootstrap_ci(scores)
        raw = np.array([e["sets"]["test"][key]["coverage"] for e in evals])
        per_seed = ", ".join(f"{x:.3f}" for x in raw.mean(1))
        return (
            f"| {label} | {calls} | {scores.mean():.3f} | {ci[0]:.3f}–{ci[1]:.3f} | "
            f"{(raw >= 0.90).mean():.1%} | {(raw >= 0.95).mean():.1%} | {latency:.2f} "
            f"| {per_seed} |"
        )

    for k in DEPTHS:
        latency = np.mean([e["latency_ms"][str(k)] for e in trm])
        calls = k * trm[0]["f_per_step"]
        lines.append(row(f"TRM k={k}", table[k], calls, latency, trm, f"fixed_{k}"))
    latency = np.mean([e["latency_ms"]["1"] for e in mlp])
    lines.append(row("Same-size MLP", mlp_scores, 1, latency, mlp, "fixed_1"))
    lines += [
        f"| Human replay | – | {ceiling:.3f} | – | – | – | – | – |",
        f"| Do nothing | – | {floor:.3f} | – | – | – | – | – |",
    ]

    dp_path = root / "dp" / "eval.json"
    if dp_path.is_file():
        dp = json.loads(dp_path.read_text())
        count = len(dp["coverage"])
        lines.append("")
        lines.append(
            f"Pretrained diffusion policy, first {count} test seeds: "
            f"{np.mean(dp['coverage']):.3f} coverage, {dp['latency_ms']:.0f} ms per plan "
            f"({dp['iterations']} denoising iterations). "
            f"TRM k={best_depth} on those seeds: {table[best_depth][:count].mean():.3f}."
        )

    lines += [
        "",
        "## Halting (threshold chosen on dev seeds, judged on test seeds)",
        "",
        "| Rule | Coverage | f calls | Best fixed | Smallest fixed depth matching "
        "| Saving vs that depth | Result |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for h in halting:
        if "threshold" not in h:
            lines.append(f"| {h['rule']} | – | – | – | – | – | FAIL: {h['reason']} |")
        else:
            lines.append(
                f"| {h['rule']} | {h['coverage']:.3f} | {h['f_calls']:.1f} "
                f"| {h['best_fixed']:.3f} | k={h['matching_depth']} "
                f"({h['matching_f_calls']} calls) | {h['saving']:.0%} | {verdict(h['passed'])} |"
            )
    lines += [
        "",
        "All halting points on test seeds:",
        "",
        "| Threshold | Coverage | f calls |",
        "|---|---:|---:|",
    ]
    for name, count in (("convergence", len(QUANTILES)), ("q", len(Q_PROBS))):
        for i in range(count):
            key = f"{name}_{i}"
            lines.append(
                f"| {key} | {seed_mean(trm, 'test', key).mean():.3f} | "
                f"{seed_mean(trm, 'test', key, 'f_calls').mean():.1f} |"
            )

    errors = np.mean([e["open_loop_l1_px"] for e in trm], axis=0)
    deltas = np.mean([e["delta_px_by_step"] for e in trm], axis=0)
    lines += [
        "",
        "## Open loop and headroom (held-out episodes)",
        "",
        "L1 error in pixels by depth: " + ", ".join(f"{x:.2f}" for x in errors),
        "",
        f"Same-size MLP: {np.mean([e['open_loop_l1_px'][0] for e in mlp]):.2f}",
        "",
        "Mean change in the answer between steps (pixels): "
        + ", ".join(f"{x:.2f}" for x in deltas),
        "",
        "Share of states first within tolerance of the final answer, by step:",
        "",
        "| Tolerance | " + " | ".join(str(i) for i in range(1, 9)) + " |",
        "|---|" + "---:|" * 8,
    ]
    for tolerance in ("1", "2", "4"):
        counts = np.sum([e["headroom"][tolerance] for e in trm], axis=0)
        shares = counts / counts.sum()
        lines.append(f"| {tolerance} px | " + " | ".join(f"{x:.0%}" for x in shares) + " |")
    lines.append("")
    (root / "report.md").write_text("\n".join(lines))
    plot_results(trm, mlp, floor, ceiling, root)
    print("\n".join(lines))


######################################################################
## Checks
######################################################################


def check(dataset=DATASET):
    """Harness and model checks to run before trusting a result."""
    torch.manual_seed(0)
    demos = load_demonstrations(dataset)
    ends = demos["episode_ends"]
    scores = [replay_episode(demos, s, e) for s, e in list(zip(np.r_[0, ends[:-1]], ends))[:40]]
    print(f"demo replay through the harness: mean coverage {np.mean(scores):.3f}")
    assert np.mean(scores) > 0.85

    (obs, chunks), (val_obs, _) = split_samples(demos)
    print(f"samples: {len(obs)} train, {len(val_obs)} held out")
    start = demos["state"][0]
    assert torch.allclose(obs[0, :6], obs[0, 6:]), "the first history should repeat s[0]"
    assert np.allclose(to_pixels(chunks[0, :2]).numpy(), demos["action"][0], atol=1e-3)
    assert np.allclose(to_pixels(chunks[0, 2:4]).numpy(), demos["action"][1], atol=1e-3)
    assert np.allclose(features(start), obs[0, 6:].numpy(), atol=1e-5)

    config = TrainingConfig()
    model = build_policy(config)
    print(f"TRM parameters: {count_parameters(model):,}")
    assert count_parameters(model) == 269_728
    assert count_parameters(build_policy(TrainingConfig(arch="mlp"))) == 269_728
    assert model.f_per_step == 3

    # Depth-s inference must reproduce the state reached by chaining s steps.
    batch = obs[:32]
    chained, (y, z) = [], model.init_state(32)
    for _ in range(config.n_sup):
        y, z = model.supervision_step(batch, y.detach(), z.detach())
        chained.append(y)
    states = model.unroll(batch, config.n_sup)[1]
    assert torch.allclose(torch.stack(chained), states)

    # With T=3 only the last cycle has gradient, so the incoming state gets none.
    deep = build_policy(TrainingConfig(T=3))
    y, z = (s.requires_grad_() for s in deep.init_state(4))
    out, _ = deep.supervision_step(obs[:4], y, z)
    out.sum().backward()
    assert y.grad is None and z.grad is None
    # With T=1 the incoming state is part of the graph, so a detach is what stops it.
    y, z = (s.requires_grad_() for s in model.init_state(4))
    out, _ = model.supervision_step(obs[:4], y, z)
    out.sum().backward()
    assert y.grad is not None
    print("recursion and gradient checks passed")

    stops = torch.tensor([[False, True], [True, True], [False, False]])
    assert first_stop(stops).tolist() == [1, 0]
    assert first_stop(torch.zeros(3, 2, dtype=torch.bool)).tolist() == [2, 2]

    # A healthy learner drives one fixed batch to a small error.
    iterations = 1000
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations * config.n_sup)
    target = chunks[:256]
    for _ in range(iterations):
        y, z = model.init_state(256)
        for _ in range(config.n_sup):
            y, z = model.supervision_step(obs[:256], y, z)
            loss = F.l1_loss(model.head(y), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            y, z = y.detach(), z.detach()
    print(f"one-batch overfit: {loss.item() * 256:.2f} px")
    assert loss.item() * 256 < 3
    print("all checks passed")


######################################################################
## Command line
######################################################################


def run_jobs(commands, logs):
    """Run commands in parallel, each logging to its own file, and fail loudly."""
    jobs = []
    for command, log in zip(commands, logs):
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("w")
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        jobs.append((process, handle, log))
    for job, handle, log in jobs:
        job.wait()
        handle.close()
        if job.returncode:
            raise RuntimeError(f"Job failed; see {log}:\n{log.read_text()[-2000:]}")


def run(args):
    start = time.time()
    root = args.output
    epochs = 1 if args.smoke else args.epochs
    episodes = 5 if args.smoke else args.episodes
    seeds = range(1 if args.smoke else args.seeds)
    base = [sys.executable, __file__]
    commands, logs = [], []
    for seed in seeds:
        for arch, threads, count in (
            ("trm", 2, epochs),
            ("mlp", 1, epochs * TrainingConfig.n_sup),  # Same number of optimizer steps.
        ):
            path = root / arch / f"seed_{seed}"
            commands.append(
                base
                + ["train", "--arch", arch, "--seed", str(seed), "--threads", str(threads)]
                + ["--epochs", str(count), "--n", str(args.n), "--T", str(args.T)]
                + ["--output", str(path), "--dataset", str(args.dataset)]
            )
            logs.append(path / "train.log")
    print(f"training {len(commands)} models", flush=True)
    run_jobs(commands, logs)
    print(f"trained in {(time.time() - start) / 60:.1f} minutes", flush=True)
    commands = [
        base
        + ["evaluate", "--run", str(log.parent), "--episodes", str(episodes)]
        + ["--dataset", str(args.dataset)]
        for log in logs
    ]
    run_jobs(commands, [log.with_name("evaluate.log") for log in logs])
    report(root, args.dataset, (time.time() - start) / 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "train", "evaluate", "report", "run"):
        p = sub.add_parser(command)
        p.add_argument("--dataset", type=Path, default=DATASET)
        if command == "train":
            p.add_argument("--arch", choices=["trm", "mlp"], default="trm")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--threads", type=int, default=TrainingConfig.threads)
            p.add_argument("--output", type=Path, default=RUNS / "trm" / "seed_0")
        if command == "evaluate":
            p.add_argument("--run", type=Path, default=RUNS / "trm" / "seed_0")
        if command in ("evaluate", "run"):
            p.add_argument("--episodes", type=int, default=200)
        if command in ("train", "run"):
            p.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
            p.add_argument("--n", type=int, default=TrainingConfig.n)
            p.add_argument("--T", type=int, default=TrainingConfig.T)
        if command in ("report", "run"):
            p.add_argument("--output", type=Path, default=RUNS)
        if command == "run":
            p.add_argument("--seeds", type=int, default=3)
            p.add_argument("--smoke", action="store_true", help="One epoch and five episodes")
    args = parser.parse_args()
    if args.command == "check":
        check(args.dataset)
    elif args.command == "train":
        config = TrainingConfig(
            arch=args.arch,
            seed=args.seed,
            threads=args.threads,
            epochs=args.epochs,
            n=args.n,
            T=args.T,
        )
        train(config, args.output, args.dataset)
    elif args.command == "evaluate":
        evaluate(args.run, args.dataset, args.episodes)
    elif args.command == "report":
        report(args.output, args.dataset)
    else:
        run(args)


if __name__ == "__main__":
    main()
