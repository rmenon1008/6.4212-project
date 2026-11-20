"""Follow-up analyses of the trained Push-T models: why TRM fails, and what helps.

Each command reads the models and evaluations under runs/pusht (written by
`trm_pusht.py run`), prints a table, and saves JSON under runs/pusht/analysis.
Closed-loop numbers use the evaluator's own rollout with one CPU thread, so
episodes match the logged results exactly.

    python pusht_analysis.py equal-compute   # halting against fixed depth at equal cost
    python pusht_analysis.py routes          # route choice at a state with two good routes
    python pusht_analysis.py failures        # stuck episodes, loops, distance from the demos
    python pusht_analysis.py noise           # random plan offsets, no retraining
    python pusht_analysis.py dp-restart      # diffusion and TRM from TRM's stuck states (slow)
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import shapely.geometry as sg
import torch

import trm_pusht as t
from pusht import AGENT_RADIUS, PushTEnv, apply_transform, coverage, tee_polygons

TEST = np.arange(t.TEST_SEED, t.TEST_SEED + 200)
DEV = np.arange(t.DEV_SEED, t.DEV_SEED + 200)
TRAINING_SEEDS = (0, 1, 2)
POLICIES = (("TRM k=1", "trm", 1), ("TRM k=8", "trm", 8), ("MLP", "mlp", 1))
STUCK, FINISHED = 0.5, 0.9  # Best coverage below STUCK is stuck; at or above FINISHED, done.


######################################################################
## Shared pieces
######################################################################


def load_model(root, arch, seed):
    stored = torch.load(Path(root) / arch / f"seed_{seed}" / "model.pt", weights_only=True)
    model = t.build_policy(t.TrainingConfig(**stored["config"]))
    model.load_state_dict(stored["model"])
    return model.eval()


def load_evals(root, arch):
    return [
        json.loads((Path(root) / arch / f"seed_{i}" / "eval.json").read_text())
        for i in TRAINING_SEEDS
    ]


def logged_coverage(root, arch, seed, depth):
    evaluation = load_evals(root, arch)[seed]
    return np.array(evaluation["sets"]["test"][f"fixed_{depth}"]["coverage"])


def raw_states(features):
    """(..., 6) model features back to environment states (x, y, bx, by, angle)."""
    xy = (features[..., :4] + 1) * 256
    angle = np.mod(np.arctan2(features[..., 4], features[..., 5]), 2 * np.pi)
    return np.concatenate([xy, angle[..., None]], -1)


def logged_rollout(model, depth, starts, offset=None, steps=t.MAX_STEPS):
    """The evaluator's rollout, also returning what the policy saw and planned.

    offset(actions) may return a shifted copy of each plan (the noise control).
    Returns best coverage (B,), states at each replan (R, B, 2, 5) as (previous,
    current), and plans (R, B, 16, 2) in pixels.
    """
    planner, states, plans = t.make_planner(model, depth), [], []

    def logging_planner(obs, step):
        actions, calls = planner(obs, step)
        if offset is not None:
            actions = offset(actions)
        o = obs.numpy()
        states.append(np.stack([raw_states(o[:, :6]), raw_states(o[:, 6:])], 1))
        plans.append(actions.copy())
        return actions, calls

    best, _ = t.rollout(logging_planner, starts, max_steps=steps)
    return best, np.array(states), np.array(plans)


def turns_back(states, plans):
    """(R, B) bool: the executed half of a plan goes out, then heads back (over 120 degrees)."""
    pusher = states[:, :, 1, :2]
    out, back = plans[:, :, 3] - pusher, plans[:, :, 7] - plans[:, :, 3]
    norm = np.linalg.norm(out, axis=-1) * np.linalg.norm(back, axis=-1) + 1e-9
    return (out * back).sum(-1) / norm < -0.5


def gap_to_tee(state):
    """Pixels between the pusher's edge and the T's outline (about 0 means touching)."""
    tee = sg.MultiPolygon([sg.Polygon(p) for p in tee_polygons(state[2:5])])
    return tee.distance(sg.Point(*state[:2])) - AGENT_RADIUS


def reverses(states, plans):
    """(R - 1, B) bool: a plan heads the opposite way to the one before (over 120 degrees)."""
    heading = plans[:, :, :8].mean(2) - states[:, :, 1, :2]
    a, b = heading[:-1], heading[1:]
    norm = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9
    return (a * b).sum(-1) / norm < -0.5


def episode_stats(best, states, plans):
    """Per episode: turn-back and reversal shares, standing still at the end, gap, block motion."""
    half = len(states) // 2
    loop_share = turns_back(states, plans)[half:].mean(0)
    last = states[-13:, :, 1, :2]  # Replan-time positions over about the last 100 steps.
    return {
        "best": best,
        "loop_share": loop_share,
        "reversal_share": reverses(states, plans)[half:].mean(0),
        "looping": loop_share > 0.5,
        "still": np.linalg.norm(last - last.mean(0), axis=-1).max(0) < 15,
        "gap": np.array([gap_to_tee(s) for s in states[-1, :, 1]]),
        "block_moved": np.linalg.norm(np.diff(states[:, :, 1, 2:4], axis=0), axis=-1).sum(0),
    }


class DemoIndex:
    """Nearest training-demo states, and whether those demos agree on where to go next."""

    def __init__(self, dataset=t.DATASET, val_fraction=0.1):
        demos = t.load_demonstrations(dataset)
        ends = demos["episode_ends"]
        split = int((1 - val_fraction) * len(ends))
        train_end = ends[split - 1]
        states = demos["state"][:train_end]
        self.features = t.features(states)
        self.squared = (self.features**2).sum(1)
        episode_end = np.zeros(train_end, int)
        for start, end in zip(np.r_[0, ends[:-1]][:split], ends[:split]):
            episode_end[start:end] = end
        ahead = np.minimum(np.arange(train_end) + 4, episode_end - 1)
        direction = demos["action"][:train_end][ahead] - states[:, :2]
        self.direction = direction / (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9)
        self.held_out = demos["state"][train_end:]

    def nearest(self, states, k=20):
        """Distances (in model-input units) to, and indices of, the k nearest demo states."""
        q = t.features(states)
        d = (q**2).sum(1)[:, None] + self.squared[None] - 2 * q @ self.features.T
        index = np.argsort(d, 1)[:, :k]
        return np.sqrt(np.maximum(np.take_along_axis(d, index, 1), 0)), index

    def describe(self, states):
        """Median distance to the nearest demo, and median agreement of the 20 nearest.

        Agreement is the length of the mean unit direction those demos move over
        their next 4 steps: 1 means they all head the same way, 0 that they cancel.
        """
        d, index = self.nearest(states)
        agreement = np.linalg.norm(self.direction[index].mean(1), axis=1)
        return float(np.median(d[:, 0])), float(np.median(agreement))


def save(root, name, data):
    out = Path(root) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(data, indent=1))
    print(f"\nsaved {out / (name + '.json')}")


def with_ci(values):
    lo, hi = t.bootstrap_ci(values)
    return f"{np.mean(values):+.3f} (95% CI {lo:+.3f} to {hi:+.3f})"


######################################################################
## Commands
######################################################################


def equal_compute(root):
    """Each halting point against the fixed depth of about the same cost, on the same episodes."""
    evals = load_evals(root, "trm")
    fixed = {k: t.seed_mean(evals, "test", f"fixed_{k}") for k in (1, 2, 4, 8)}
    for k in (3, 5, 6, 7):
        runs = [
            t.rollout(t.make_planner(load_model(root, "trm", i), k), TEST)[0]
            for i in TRAINING_SEEDS
        ]
        fixed[k] = np.mean(runs, axis=0)
    print("fixed depth coverage:", {k: round(float(fixed[k].mean()), 3) for k in sorted(fixed)})
    pairs = [
        ("q_0", 3), ("convergence_4", 3), ("q_3", 4), ("convergence_3", 4),
        ("q_4", 5), ("convergence_2", 6), ("convergence_1", 7),
    ]  # fmt: skip
    rows = []
    for key, depth in pairs:
        halting = t.seed_mean(evals, "test", key)
        calls = t.seed_mean(evals, "test", key, "f_calls").mean()
        difference = halting - fixed[depth]
        lo, hi = t.bootstrap_ci(difference)
        rows.append(
            dict(
                point=key,
                f_calls=float(calls),
                coverage=float(halting.mean()),
                fixed_depth=depth,
                fixed_coverage=float(fixed[depth].mean()),
                difference=float(difference.mean()),
                ci=[float(lo), float(hi)],
            )
        )
        print(
            f"{key:14s} {calls:5.1f} calls {halting.mean():.3f} against k={depth} "
            f"({3 * depth} calls) {fixed[depth].mean():.3f}: {with_ci(difference)}"
        )
    fixed_means = {k: float(v.mean()) for k, v in fixed.items()}
    save(root, "equal_compute", dict(fixed=fixed_means, pairs=rows))


def routes(root):
    """Which way TRM goes from the course notebook's two-route state.

    The T is at its starting pose with the pusher 100 px in front of it, between
    it and the goal; demos and diffusion go around either end.  Side is the plan's
    mean x in the T's frame: positive passes the crossbar's right end.
    """
    pose = (219.1, 293.0, np.pi / 4)
    pusher = apply_transform(pose, (0, -100))
    along_bar = np.array([np.cos(pose[2]), np.sin(pose[2])])
    models = {i: load_model(root, "trm", i) for i in TRAINING_SEEDS}
    rows = []
    for velocity in ((0, 0), (100, 0), (-100, 0)):
        current = PushTEnv(np.r_[pusher, pose]).observation()
        previous = PushTEnv(np.r_[pusher - np.asarray(velocity) / 10, pose]).observation()
        obs = torch.from_numpy(np.concatenate([t.features(previous), t.features(current)])[None])
        for i, model in models.items():
            for k in (1, 8):
                with torch.no_grad():
                    plan = t.to_pixels(model.unroll(obs, k)[0][-1]).reshape(16, 2).numpy()
                side = float(((plan - pose[:2]) @ along_bar).mean())
                rows.append(dict(velocity=velocity, training_seed=i, depth=k, side_px=side))
        mine = [r for r in rows if r["velocity"] == velocity]
        sides = "; ".join(
            f"seed {r['training_seed']} k={r['depth']}: {r['side_px']:+.0f}" for r in mine
        )
        print(f"pusher velocity {velocity} px/s -> side in px: {sides}")
    save(root, "routes", rows)


def failures(root, dataset):
    """Stuck episodes for TRM k=1, TRM k=8 and the MLP, and where the loops happen."""
    stats, states_by = {}, {}
    for name, arch, depth in POLICIES:
        per_seed, states_list = [], []
        for i in TRAINING_SEEDS:
            best, states, plans = logged_rollout(load_model(root, arch, i), depth, TEST)
            assert np.allclose(best, logged_coverage(root, arch, i, depth), atol=1e-6)
            per_seed.append(episode_stats(best, states, plans))
            states_list.append(states)
        stats[name] = {k: np.concatenate([s[k] for s in per_seed]) for k in per_seed[0]}
        states_by[name] = np.concatenate(states_list, axis=1)
    print("Pooled over 3 training seeds x 200 test episodes; matches the logged coverage.\n")
    print(
        f"{'':9s}{'coverage':>9s}{'stuck':>8s}{'near miss':>11s}{'finished':>10s}"
        f" | of stuck: {'looping':>8s}{'still':>7s}{'gap px':>8s}"
    )
    summary = {}
    for name, s in stats.items():
        b = s["best"]
        stuck, near, done = b < STUCK, (b >= STUCK) & (b < FINISHED), b >= FINISHED
        r = summary[name] = dict(
            coverage=float(b.mean()),
            stuck=float(stuck.mean()),
            near_miss=float(near.mean()),
            finished=float(done.mean()),
            stuck_looping=float(s["looping"][stuck].mean()),
            stuck_still=float(s["still"][stuck].mean()),
            stuck_gap_px=float(np.median(s["gap"][stuck])),
            stuck_not_looping=int((stuck & ~s["looping"]).sum()),
            loop_share_stuck=float(np.median(s["loop_share"][stuck])),
            loop_share_finished=float(np.median(s["loop_share"][done])),
            reversal_share_stuck=float(np.median(s["reversal_share"][stuck])),
            reversal_share_finished=float(np.median(s["reversal_share"][done])),
        )
        print(
            f"{name:9s}{r['coverage']:9.3f}{r['stuck']:8.1%}{r['near_miss']:11.1%}"
            f"{r['finished']:10.1%} | {'':10s}{r['stuck_looping']:8.0%}{r['stuck_still']:7.0%}"
            f"{r['stuck_gap_px']:8.1f}"
        )
    print("\nshare of executed chunks that turn back (median per episode, second half):")
    for name, r in summary.items():
        stuck_share, finished_share = r["loop_share_stuck"], r["loop_share_finished"]
        print(f"  {name:9s} stuck {stuck_share:.2f}, finished {finished_share:.2f}")
    print("share of consecutive plans that reverse direction (median per episode, second half):")
    for name, r in summary.items():
        stuck_share, finished_share = r["reversal_share_stuck"], r["reversal_share_finished"]
        print(f"  {name:9s} stuck {stuck_share:.2f}, finished {finished_share:.2f}")

    print("\nTRM k=8 by training seed (failed = below 0.90):")
    dp_path = Path(root) / "dp" / "eval.json"
    dp = np.array(json.loads(dp_path.read_text())["coverage"]) if dp_path.is_file() else None
    by_seed = []
    for i, b in zip(TRAINING_SEEDS, stats["TRM k=8"]["best"].reshape(len(TRAINING_SEEDS), -1)):
        row = dict(training_seed=i, failed=int((b < FINISHED).sum()), stuck=int((b < STUCK).sum()))
        text = f"  seed {i}: {row['stuck']} of {row['failed']} failures are stuck"
        if dp is not None:
            stuck_here = b[: len(dp)] < STUCK
            row["stuck_in_dp_seeds"] = int(stuck_here.sum())
            row["dp_finished"] = int((dp[stuck_here] >= FINISHED).sum())
            text += (
                f"; of the {row['stuck_in_dp_seeds']} stuck starts in the first {len(dp)} seeds,"
                f" diffusion finished {row['dp_finished']}"
            )
        by_seed.append(row)
        print(text)

    k1, k8, mlp = (stats[n]["best"] < STUCK for n in ("TRM k=1", "TRM k=8", "MLP"))
    print(
        f"\nsame start and training seed: stuck in both TRM k=8 and MLP {np.sum(k8 & mlp)}, "
        f"TRM only {np.sum(k8 & ~mlp)}, MLP only {np.sum(mlp & ~k8)}"
    )
    print(
        f"TRM k=1 against k=8: stuck in both {np.sum(k1 & k8)}, k=1 only {np.sum(k1 & ~k8)}, "
        f"k=8 only {np.sum(k8 & ~k1)}"
    )

    demos = DemoIndex(dataset)
    places = {"held-out demo states": demos.held_out[::10]}
    for name in ("TRM k=8", "MLP"):
        s, states = stats[name], states_by[name]
        places[f"{name}, stuck end states"] = states[-1, s["best"] < STUCK, 1]
        middle = len(states) // 2
        places[f"{name}, finished, mid-episode"] = states[middle, s["best"] >= FINISHED, 1]
    print(f"\n{'':28s}{'n':>5s}{'nearest demo':>14s}{'20 nearest agree':>18s}")
    where = {}
    for label, group in places.items():
        distance, agreement = demos.describe(group)
        where[label] = dict(n=len(group), nearest_demo=distance, agreement=agreement)
        print(f"{label:28s}{len(group):5d}{distance:14.3f}{agreement:18.2f}")

    escapes = nudge_test(root, stats["TRM k=8"]["best"], states_by["TRM k=8"])
    print(
        f"\nnudge test on TRM k=8 stuck states (pusher moved 20 px in 8 directions): median "
        f"{np.median(escapes):.0f}/8 escape; half or more escape in {np.mean(escapes >= 4):.0%}"
    )
    result = dict(
        summary=summary,
        by_training_seed=by_seed,
        demo_distance=where,
        nudge_escapes=escapes.tolist(),
    )
    save(root, "failures", result)


def nudge_test(root, best, states, distance=20, steps=80):
    """Move the pusher at each stuck end state 20 px in 8 directions and run 80 steps.

    An escape is the pusher getting more than 40 px from where it restarted,
    sampled at each replan.  Returns escapes out of 8 for each stuck episode.
    """
    stuck = np.where(best < STUCK)[0]
    seed_of = stuck // len(TEST)
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    directions = np.c_[np.cos(angles), np.sin(angles)]
    escapes = np.zeros(len(stuck), int)
    for i in TRAINING_SEEDS:
        mine = np.where(seed_of == i)[0]
        if len(mine) == 0:
            continue
        starts = []
        for j in mine:
            end = states[-1, stuck[j], 1]
            for d in directions:
                start = end.copy()
                start[:2] = np.clip(end[:2] + distance * d, 20, 490)
                starts.append(start)
        starts = np.array(starts)
        _, path, _ = logged_rollout(load_model(root, "trm", i), 8, starts, steps=steps)
        moved = np.linalg.norm(path[:, :, 1, :2] - starts[None, :, :2], axis=-1).max(0)
        escapes[mine] = (moved > 2 * distance).reshape(len(mine), len(directions)).sum(1)
    return escapes


def _noise_task(job):
    root, seed, sigma, split = job
    torch.set_num_threads(1)
    rng = np.random.default_rng(1000 * seed + int(10 * sigma) + (0 if split == "test" else 7))

    def offset(actions):
        return actions + rng.normal(0, sigma, (len(actions), 1, 2)) if sigma else actions

    starts = TEST if split == "test" else DEV
    best, _, _ = logged_rollout(load_model(root, "trm", seed), 8, starts, offset)
    return seed, sigma, split, best


def noise(root, levels=(0, 2, 5, 10, 20)):
    """Shift each TRM k=8 plan by a random offset at every replan, without retraining."""
    jobs = [(root, i, s, sp) for sp in ("dev", "test") for s in levels for i in TRAINING_SEEDS]
    with ProcessPoolExecutor(8) as pool:
        results = sorted(pool.map(_noise_task, jobs), key=lambda r: r[0])
    best = {
        (sp, s): np.array([r[3] for r in results if r[1] == s and r[2] == sp])
        for sp in ("dev", "test")
        for s in levels
    }
    logged = [logged_coverage(root, "trm", i, 8) for i in TRAINING_SEEDS]
    assert np.allclose(best[("test", 0)], logged, atol=1e-6)
    rows = []
    for split in ("dev", "test"):
        print(f"\n{split} seeds (3 training seeds x 200 episodes)")
        header = f"{'offset sd':>9s} {'coverage':>9s} {'finished':>9s} {'stuck':>7s}"
        print(f"  {header}   coverage change")
        base = best[(split, 0)]
        for s in levels:
            b = best[(split, s)]
            finished, stuck = float(np.mean(b >= FINISHED)), float(np.mean(b < STUCK))
            rows.append(
                dict(split=split, offset_sd_px=s, coverage=float(b.mean()), finished=finished,
                     stuck=stuck)
            )  # fmt: skip
            change = with_ci(b.mean(0) - base.mean(0))
            print(f"  {s:>7d}px {b.mean():9.3f} {finished:9.1%} {stuck:7.1%}   {change}")
    pick = max(levels[1:], key=lambda s: best[("dev", s)].mean())
    b, base = best[("test", pick)], best[("test", 0)]
    gain = np.mean(b >= FINISHED, axis=0) - np.mean(base >= FINISHED, axis=0)
    print(
        f"\npicked on dev seeds: {pick} px. Test: finished {np.mean(base >= FINISHED):.1%} -> "
        f"{np.mean(b >= FINISHED):.1%}, change {with_ci(gain)}; stuck "
        f"{np.mean(base < STUCK):.1%} -> {np.mean(b < STUCK):.1%}"
    )
    save(root, "noise", dict(levels=rows, picked_px=pick))


def run_from(previous, current, plan, now, steps=t.MAX_STEPS):
    """Run plans from given (previous, current) states; plan(previous, current) -> (B, 16, 2)."""
    envs = [PushTEnv(s) for s in current]
    current = np.stack([e.observation() for e in envs])
    best, done = np.zeros(len(envs)), 0
    while done < steps:
        actions = plan(previous, current)
        for u in range(now, now + t.ACTION_HORIZON):
            following = []
            for k, env in enumerate(envs):
                state, c, _ = env.step(actions[k, u])
                best[k] = max(best[k], c)
                following.append(state)
            previous, current = current, np.stack(following)
            done += 1
            if done == steps:
                break
    return best, current


def dp_restart(root, dataset, assets, device, limit):
    """Restart diffusion and TRM from the exact states where TRM k=8 got stuck."""
    from pusht_dp_baseline import NOW, DiffusionPlanner

    starts = []  # (training seed, previous state, current state)
    for i in TRAINING_SEEDS:
        best, states, _ = logged_rollout(load_model(root, "trm", i), 8, TEST)
        starts += [(i, states[-1, j, 0], states[-1, j, 1]) for j in np.where(best < STUCK)[0]]
    if limit:
        starts = [starts[k] for k in np.linspace(0, len(starts) - 1, limit).astype(int)]
    seeds = np.array([s[0] for s in starts])
    previous, current = np.stack([s[1] for s in starts]), np.stack([s[2] for s in starts])
    print(f"{len(starts)} stuck states (TRM k=8, 3 training seeds)")
    models = {i: load_model(root, "trm", i) for i in TRAINING_SEEDS}

    def trm_plan(prev, cur):
        plans = np.zeros((len(cur), 16, 2))
        for i, model in models.items():
            mine = np.where(seeds == i)[0]
            if len(mine) == 0:
                continue
            obs = np.concatenate([t.features(prev[mine]), t.features(cur[mine])], -1)
            with torch.no_grad():
                chunk = model.unroll(torch.from_numpy(obs), 8)[0][-1]
            plans[mine] = t.to_pixels(chunk).reshape(-1, 16, 2).numpy()
        return plans

    diffusion = DiffusionPlanner(assets, device, dataset)
    trm_best, trm_end = run_from(previous, current, trm_plan, 0)
    dp_best, dp_end = run_from(
        previous, current, lambda p, c: diffusion.sample(np.stack([p, c], 1)), NOW
    )

    demos = DemoIndex(dataset)
    start_coverage = np.array([coverage(s[2:5]) for s in current])
    start_distance = demos.nearest(current, 1)[0][:, 0]
    header = f"{'best coverage':>14s}{'finished':>10s}{'stuck':>8s}{'distance from demos':>21s}"
    print(f"\n{'':20s}{header}")
    rows = (
        ("at the stuck state", start_coverage, current),
        ("TRM restarted", trm_best, trm_end),
        ("diffusion restarted", dp_best, dp_end),
    )
    for label, b, end in rows:
        distance = np.median(demos.nearest(end, 1)[0][:, 0])
        print(
            f"{label:20s}{b.mean():14.3f}{np.mean(b >= FINISHED):10.1%}"
            f"{np.mean(b < STUCK):8.1%}{distance:21.3f}"
        )
    print(f"\ndiffusion minus TRM, paired by state: {with_ci(dp_best - trm_best)}")
    thirds = np.quantile(start_distance, [1 / 3, 2 / 3])
    groups = (
        ("closest third", start_distance <= thirds[0]),
        ("middle third", (start_distance > thirds[0]) & (start_distance <= thirds[1])),
        ("farthest third", start_distance > thirds[1]),
    )
    by_distance = []
    for label, mask in groups:
        dp_rate = float(np.mean(dp_best[mask] >= FINISHED))
        trm_rate = float(np.mean(trm_best[mask] >= FINISHED))
        median_distance = float(np.median(start_distance[mask]))
        by_distance.append(
            dict(group=label, distance=median_distance, diffusion_finished=dp_rate,
                 trm_finished=trm_rate)
        )  # fmt: skip
        print(
            f"  {label:15s} (distance {np.median(start_distance[mask]):.2f}): diffusion "
            f"finished {dp_rate:.0%}, TRM {trm_rate:.0%}"
        )
    result = dict(
        start_coverage=start_coverage.tolist(),
        start_distance=start_distance.tolist(),
        trm=trm_best.tolist(),
        diffusion=dp_best.tolist(),
        by_distance=by_distance,
    )
    save(root, "dp_restart", result)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command", choices=["equal-compute", "routes", "failures", "noise", "dp-restart"]
    )
    parser.add_argument("--runs", type=Path, default=t.RUNS)
    parser.add_argument("--dataset", type=Path, default=t.DATASET)
    parser.add_argument("--assets", type=Path, default=Path("data/pusht/diffusion"))
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="dp-restart: use this many states")
    args = parser.parse_args()
    torch.set_num_threads(1)  # The evaluator's setting; other thread counts change rounding.
    if args.command == "equal-compute":
        equal_compute(args.runs)
    elif args.command == "routes":
        routes(args.runs)
    elif args.command == "failures":
        failures(args.runs, args.dataset)
    elif args.command == "noise":
        noise(args.runs)
    else:
        dp_restart(args.runs, args.dataset, args.assets, args.device, args.limit)


if __name__ == "__main__":
    main()
