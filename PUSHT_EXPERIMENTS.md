# TRM on Push-T: what was done and what it showed

Branch `trm-experimentation-archit`, branched from `main` at `b7f6b20` (first commit).
Work from November 6 and November 20, 2026. Nothing on this branch is committed yet.

**Summary.** A 270k-parameter tiny recursive model (TRM) works as a Push-T policy.
At recursion depth k=8 it reaches 0.746 mean max coverage, against 0.567 for an MLP
of the same size and 0.919 for the pretrained Diffusion Policy (on 50 episodes).
More recursion helps a little: k=8 beats k=1 by 0.052, and gains level off around
k=5. Adaptive halting did not beat simply using a smaller fixed depth. Most of
TRM's failures are not near misses but episodes where it gets stuck in a repeating
out-and-back loop, in states far from any demonstration. Random noise does not free
it, but Diffusion Policy recovers from most of those same states. That makes
diffusion a plausible teacher for retraining TRM on the states it actually visits.

## Contents

1. [Why Push-T](#why-push-t)
2. [What changed in the repo](#what-changed-in-the-repo)
3. [How to reproduce](#how-to-reproduce)
4. [The model](#the-model)
5. [Protocol and terms](#protocol-and-terms)
6. [Experiment 1: go/no-go run](#experiment-1-gono-go-run)
7. [Experiment 2: follow-ups](#experiment-2-follow-ups)
8. [Experiment 3: why TRM fails](#experiment-3-why-trm-fails)
9. [What it tells us](#what-it-tells-us)
10. [Caveats](#caveats)
11. [Open questions and next steps](#open-questions-and-next-steps)

## Why Push-T

The project proposal asks whether a TRM policy with a halting head can match
Diffusion Policy while spending less compute on easy states. Before building that
on LIBERO, the instructor suggested a cheaper test on Push-T: a 2D task where a
round pusher shoves a T-shaped block into a goal. It trains on a laptop CPU in
minutes, has about 200 human demonstrations, and comes with a pretrained Diffusion
Policy from the course problem set. The architecture follows
`trm_pusht_architecture.tex` (not in the repo), using its recommended MVP setting.

## What changed in the repo

| File | Status | What it does |
| --- | --- | --- |
| `pusht.py` | New | Push-T simulator and coverage metric, extracted unchanged from the course notebook `diffusion_policy_pusht_state.ipynb` (pymunk physics, MIT licence) |
| `trm_pusht.py` | New | Data loading, TRM and same-size MLP, deep-supervision training, closed-loop evaluation, both halting rules, report and plots, sanity checks |
| `pusht_dp_baseline.py` | New | The authors' pretrained Diffusion Policy, evaluated on the same test seeds; downloads its checkpoint and network code on first use |
| `pusht_analysis.py` | New | The November 20 analyses: route choice, failure diagnosis, demo distance, noise control, diffusion restarts, halting at equal compute |
| `requirements-pusht.txt` | New | Tested versions for Python 3.11 |
| `README.md` | Edited | Added a Push-T section |
| `PUSHT_EXPERIMENTS.md` | New | This file |
| `trm.py`, `libero.py`, `project.ipynb`, `requirements.txt` | Unchanged | The team's LIBERO pipeline |

The course notebook was used only as a source and is not in the repo. Outputs go to
`runs/` and data to `data/`, both gitignored.

## How to reproduce

All timings are on an Apple M4 laptop (10 CPU cores), using `/opt/homebrew/bin/python3.11`.

| Command | What it produces | Time |
| --- | --- | --- |
| `python trm_pusht.py check` | Harness and model checks | About 20 seconds |
| `python trm_pusht.py run` | 3 seeds each of TRM and MLP, evaluation, `runs/pusht/report.md` and plots | 17 minutes |
| `python trm_pusht.py run --epochs 30 --output runs/pusht_30ep` | The 30-epoch comparison | Not timed cleanly; it shared the machine |
| `python pusht_dp_baseline.py --device mps` | Diffusion on 50 test seeds, `runs/pusht/dp/eval.json` | Not timed cleanly; expect tens of minutes |
| `python trm_pusht.py report` | Report including the diffusion row | Seconds |
| `python pusht_analysis.py routes` | Route choice at the two-route state | Seconds |
| `python pusht_analysis.py equal-compute` | Halting against fixed depth at equal cost | About 1 minute |
| `python pusht_analysis.py failures` | Stuck episodes, loops, demo distance, nudge test | About 2½ minutes |
| `python pusht_analysis.py noise` | Noise control on dev and test seeds | About 1 minute |
| `python pusht_analysis.py dp-restart` | Diffusion and TRM restarted from TRM's stuck states | About 21 minutes on the GPU |

Closed-loop evaluation uses one CPU thread on purpose. Other thread counts change
floating-point rounding, and in a contact simulation that changes a few episodes.
The analyses reuse the evaluator's rollout and assert that they reproduce the
logged results exactly.

## The model

```text
last 2 states (12 numbers) -> linear embedding x (128)
start y = 0, z = 0
one supervision step:  z <- f(x + y + z)  twice,  then  y <- f(y + z)
linear head: y -> 16 future pusher targets (32 numbers)
```

- **Shared network `f`:** two post-norm residual blocks, `LN(h + W2 GELU(W1 h))`,
  width 128. The same `f` does every update.
- **Inputs:** for each of the last 2 states, pusher and block position scaled to
  [-1, 1] (`p/256 - 1`) plus sine and cosine of the block angle.
- **Size:** 269,728 parameters (`f` is 263,936). The same-size MLP applies the same
  embedding, `f` and head once, with no recursion.
- **Training:** each batch runs 8 supervision steps. After every step, the answer
  `y` is decoded, compared with the demonstrated chunk by L1 loss, the weights are
  updated, and `y` and `z` are detached. This deep supervision makes every
  intermediate depth a usable policy. AdamW at 3e-4 with warmup and cosine decay,
  batch 256, gradient clipping at 1.0, EMA of weights at 0.999, 100 epochs. The MLP
  gets 8 times the epochs, so both take the same number of optimizer steps.
- **Inference:** run k supervision steps from zeros, execute the first 8 of the 16
  targets, observe again, restart from zeros.

Compared with `trm.py`, this model has separate answer and memory states, feeds
the observation in at every update, and trains every step rather than only the
last. `trm.py` uses one state, ignores the observation inside its loop by default,
and supervises only its final output, so its intermediate steps are not usable
actions.

## Protocol and terms

- **Data:** 206 human demonstrations (25,650 steps). TRM and MLP train on the first
  185 and hold out the last 21 for open-loop error. The pretrained diffusion
  checkpoint saw all 206.
- **Episodes:** a seeded random start, 300 control steps at 10 Hz. Test seeds start
  at 100000; separate dev seeds at 200000 are used only to choose thresholds.
- **Training seeds:** 3 per model. Intervals are bootstraps over episodes, with each
  episode averaged across the training seeds; they do not capture variation
  between training runs.

| Term | Meaning |
| --- | --- |
| Coverage | Share of the goal T's area covered by the block (1.0 is perfect) |
| Mean max coverage | Each episode's best coverage, averaged over episodes; the main score |
| Finished | Best coverage reached 0.90 |
| Stuck | Best coverage stayed below 0.5 |
| Near miss | Best coverage between 0.5 and 0.90 |
| Depth k | Number of supervision steps run at inference; each makes 3 calls to `f` |
| Turn-back chunk | The 8 executed targets go out, then head back (over 120 degrees) |
| Looping | More than half of an episode's second-half chunks turn back |
| Distance from demos | Distance in model-input units to the nearest training-demo state |
| Demo agreement | For the 20 nearest demo states, length of their mean unit direction over the next 4 steps (1 = all the same way, 0 = cancelling) |

Human demonstrations replayed through the simulator reach 0.900 mean max coverage,
and doing nothing scores 0.081. In a check of the first 40 replays, only 5% reached
the simulator's 0.95 success threshold, so mean max coverage is the main metric
rather than success.

## Experiment 1: go/no-go run

November 6. `python trm_pusht.py run`, 17.0 minutes. Pass bars were set before the run.

| Gate | Bar | Result |
| --- | --- | --- |
| G1: working policy | Coverage at least 0.60 and not below the MLP | **Pass.** k=8 minus MLP +0.179 (95% CI +0.142 to +0.216) |
| G2: recursion helps | k=8 minus k=1 at least +0.05, interval above 0 | **Pass, barely.** +0.052 (+0.027 to +0.078) |
| G3: halting helps | Within 0.02 of the best fixed depth with at least 30% fewer `f` calls | **Fail.** Best saving 23% |

| Policy | `f` calls per plan | Coverage | 95% CI | Finished | Reached 0.95 |
| --- | ---: | ---: | --- | ---: | ---: |
| TRM k=1 | 3 | 0.694 | 0.644–0.742 | 61.3% | 38.2% |
| TRM k=2 | 6 | 0.716 | 0.667–0.764 | 65.7% | 45.8% |
| TRM k=4 | 12 | 0.735 | 0.688–0.779 | 70.5% | 45.2% |
| TRM k=8 | 24 | 0.746 | 0.699–0.790 | 70.2% | 45.8% |
| TRM k=16 (past training depth) | 48 | 0.640 | 0.598–0.682 | 41.3% | 17.3% |
| Same-size MLP | 1 | 0.567 | 0.527–0.608 | 33.5% | 11.7% |

- **Open-loop error barely changes with depth, and gets slightly worse:** held-out
  L1 is 21.9 px at k=1, 22.8 px at k=8 and 26.2 px at k=16, against 23.6 px for the
  MLP. Closed-loop gains with depth are therefore not from matching the demos more
  closely.
- **Halting:** the convergence rule (stop when the plan stops moving) and the
  proposal's Q-head (fitted after training on the frozen model) were tuned on dev
  seeds and judged on test seeds. Both landed on the fixed-depth curve.
- **Latency:** one plan for one observation takes about 0.8 ms at k=8 on an idle
  CPU (1.9 ms as measured while other jobs shared the machine). The robot replans
  every 800 ms, so halting saves almost no real time on Push-T.

## Experiment 2: follow-ups

November 6.

**Diffusion reference** (`pusht_dp_baseline.py`, first 50 test seeds). Diffusion
Policy reaches 0.919 coverage and finishes 88% of episodes; TRM k=8 reaches 0.785 on
the same seeds. Diffusion takes 1,166 ms per plan on the CPU (100 denoising
iterations, 65.4M parameters), so TRM is about 240 times smaller and several hundred
times faster per plan, and clearly worse.

**Halting at equal compute** (`pusht_analysis.py equal-compute`). Adding fixed depths
k=3, 5, 6 and 7 lets every halting point be compared with a fixed depth of about the
same cost, on the same episodes. Coverage by fixed depth is 0.694, 0.716, 0.728,
0.735, 0.746, 0.752, 0.752 and 0.746 for k=1 to 8. Every paired difference between
halting and fixed depth lies within ±0.006, and every 95% interval includes zero
(each spans about ±0.02).

**Training length** (`trm_pusht.py run --epochs 30`). This checks whether the MLP
was simply overtrained at 100 epochs.

| Policy | 30 epochs | 100 epochs |
| --- | ---: | ---: |
| TRM k=1 | 0.602 | 0.694 |
| TRM k=4 | 0.722 | 0.735 |
| TRM k=8 | 0.691 | 0.746 |
| Same-size MLP | 0.449 | 0.567 |
| k=8 minus k=1 | +0.088 | +0.052 |

The MLP was not overtrained: it did worse with less training. Longer training
improved closed-loop coverage for every policy, even though held-out error stopped
improving around epoch 30. The gain from depth shrank with longer training.

## Experiment 3: why TRM fails

November 20. All numbers come from `pusht_analysis.py`, pooled over 3 training seeds
× 200 test episodes unless stated.

**Route choice** (`routes`). At the course notebook's two-route state (T at its
starting pose, pusher between it and the goal), demos and diffusion go around either
end. TRM commits to one side rather than blending: with the pusher at rest, its
plans pass 37 to 85 px to one side of the T's centre line, and 2 of 3 training
seeds go right. When the two-frame history shows the pusher moving sideways, every
seed goes the way it was moving.

**Failure breakdown** (`failures`).

| | TRM k=1 | TRM k=8 | MLP |
| --- | ---: | ---: | ---: |
| Coverage | 0.694 | 0.746 | 0.567 |
| Stuck | 27.5% | 22.3% | 43.0% |
| Near miss | 11.2% | 7.5% | 23.5% |
| Finished | 61.3% | 70.2% | 33.5% |
| Stuck episodes that are looping | 86% | 81% | 69% |
| Stuck episodes standing still at the end | 85% | 80% | 73% |
| Median gap between pusher and T when stuck | 19.0 px | 9.4 px | 17.8 px |

- **Most failures are stuck, not near misses.** For TRM k=8, 47 of 58 failures are
  stuck for training seed 0, 42 of 63 for seed 1, and 45 of 58 for seed 2.
- **Stuck episodes loop.** In every policy, the median stuck episode has all of its
  second-half chunks turning back, against 11–21% in finished episodes. The pusher
  goes out and comes back without moving the block, the model sees nearly the same
  input, and being deterministic it makes the same plan again.
- **It is not flip-flopping between replans.** In the median stuck episode, no
  consecutive plans reverse direction, against 11–17% in finished episodes. The
  loop happens inside each executed chunk.
- **The MLP fails the same way, more often.** About a third of its stuck episodes
  are not loops; there the pusher keeps moving without getting the block anywhere.
- **Depth reduces looping a little.** Going from k=1 to k=8 frees 59 stuck episodes
  and creates 28 new ones. Most of TRM's advantage over the MLP is already present
  at k=1.
- **These are not simply hard starts.** On the TRM k=8 stuck starts that fall within
  the 50 seeds diffusion was run on, diffusion finished 11 of 13 (seed 0), 6 of 6
  (seed 1) and 7 of 8 (seed 2).

**Where the loops happen** (`failures`).

| States | Distance from demos | Demo agreement |
| --- | ---: | ---: |
| Held-out demo states (baseline) | 0.105 | 0.84 |
| TRM k=8, where stuck episodes end | 0.341 | 0.78 |
| MLP, where stuck episodes end | 0.266 | 0.72 |
| TRM k=8, finished episodes, mid-episode | 0.039 | 0.64 |
| MLP, finished episodes, mid-episode | 0.060 | 0.63 |

Stuck states are about three times farther from any demo than held-out demo states.
The demos nearest them agree on direction, so the loops are not caused by averaging
demos that disagree. The policy has drifted somewhere the demos never covered. The
MLP gets stuck closer to the data, which suggests it breaks down sooner as it drifts.

**Nudge test** (`failures`). Moving the pusher 20 px in each of 8 directions from
a stuck state lets it escape (move more than 40 px within 80 steps) in a median of
2 directions. Loop states pull the policy back.

**Noise control** (`noise`). Shifting every plan by a random offset, with no
retraining:

| Offset (std per axis) | Coverage | Finished | Stuck |
| --- | ---: | ---: | ---: |
| None | 0.746 | 70.2% | 22.3% |
| 2 px | 0.756 | 71.0% | 21.2% |
| 5 px | 0.758 | 72.7% | 21.5% |
| 10 px | 0.747 | 62.3% | 20.0% |
| 20 px | 0.672 | 24.8% | 23.5% |

The level chosen on dev seeds (5 px) raises the finished rate by 2.5 points, with
a 95% interval from −0.3 to +5.7, and barely changes the stuck rate. Larger offsets
spoil the final alignment. Simple randomness does not explain diffusion's advantage.

**Diffusion restarted from TRM's stuck states** (`dp-restart`). Each of the 134
states where TRM k=8 was stuck, with its two-frame history, is restarted under
diffusion and under TRM for 300 more steps.

| | Best coverage | Finished | Stuck | Distance from demos at the end |
| --- | ---: | ---: | ---: | ---: |
| At the stuck state | 0.028 | 0% | 100% | 0.341 |
| TRM restarted | 0.065 | 4.5% | 95.5% | 0.323 |
| Diffusion restarted | 0.700 | 63.4% | 26.1% | 0.046 |

Diffusion recovers most of them and steers back to states like those in successful
episodes. It is not immune to unfamiliar states: it finishes 84% of the closest
third, 61% of the middle third and 44% of the farthest third, against 13%, 0% and 0%
for TRM. Restarting from rest does not break TRM's loops, so the recovery is not an
artifact of the restart.

## What it tells us

1. **TRM is a working, very cheap policy,** and clearly better than an MLP of the
   same size. Its gap to diffusion is mostly stuck episodes, not imprecise pushing.
2. **Recursion's benefit is real but small,** and it shrinks with longer training.
   Most of TRM's advantage over the MLP is already present at k=1.
3. **Adaptive halting has little to save here.** Depth adds about 0.05 coverage in
   total, halting matches fixed depth at equal compute, and a plan takes about a
   millisecond anyway.
4. **The failure is drift into unfamiliar states, not multimodality.** TRM commits
   at the two-route state, and the demos near its loop states agree on direction.
   Being deterministic, it then repeats a plan that changes nothing.
5. **Noise does not fix it, but diffusion recovers from it.** That points to
   training TRM on the states it actually visits, with diffusion as the teacher.

## Caveats

- Intervals cover episode-to-episode variation, not variation between training runs.
- The TRM and MLP latencies were measured while other jobs shared the CPU.
- The gate bars were proposed for this test, not standard thresholds. G1's depth
  (k=8) was the best of k=1 to 8 on the test seeds.
- The Q-head was fitted after training, on a frozen model; a jointly trained one is
  untested.
- Diffusion differs from TRM in more than sampling: 65.4M parameters, a different
  architecture, and 21 more demonstrations. Why it handles unfamiliar states better
  is untested.
- "Looping", "distance from demos" and "demo agreement" are measures defined for this
  analysis, not standard ones.
- Push-T uses exact state, not images, so none of this transfers to LIBERO without
  testing.

## Open questions and next steps

- **Diffusion as a teacher (recommended next):** run TRM, record the states it
  visits, have diffusion label them, retrain TRM on demos plus labels, and repeat.
  Labeling about 5,000 states should take roughly 20 minutes on the GPU.
- **Sampling or size?** Diffusion with fixed starting noise would show whether its
  randomness or its capacity explains its recovery.
- **Recursion or depth?** A deeper MLP without shared weights would separate the two.
- **Multimodal sampling (IMLE):** an IMLE-trained TRM was proposed to handle routes
  that split. The diagnosis above weakens that motivation, so it now comes after
  the diffusion-teacher experiment.
- The proposal document and results page written alongside this work are private
  claude.ai artifacts. The proposal predates Experiment 3's later findings, so this
  file is the current account.
