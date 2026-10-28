# TRM on LIBERO with Drake and Panda

| File | Purpose |
|---|---|
| `libero.py` | Scene conversion, Drake environment, controller, downloads, human replay |
| `trm.py` | Dataset, original model, training, learned-policy evaluation |
| `project.ipynb` | Experiment settings, execution, plots, videos, conclusions |
| `requirements.txt` | Pinned runtime and notebook dependencies |
| `README.md` | Setup and reproduction instructions |

Use Python **3.12** on macOS or Linux. Install the pinned environment with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
uv pip sync requirements.txt
source .venv/bin/activate
python -m ipykernel install --user --name trm-drake-panda --display-name "TRM Drake Panda"
jupyter lab project.ipynb
```

Select the **TRM Drake Panda** kernel. Set paths, task IDs, episode counts, seed,
device, and training parameters in the notebook's settings cell, then run cells
in order. The notebook runs its own experiments and writes checkpoints, training
history, reports, figures, and videos under `runs/project/`.

The default training configuration matches the original notebook's
`main_pipeline`: 256 hidden units, 16 shared MLP updates, trainable ResNet18 with
512 visual features, frozen CLIP ViT-L/14, 30 epochs, batch size 512, AdamW at
1e-4 with weight decay 0.1, and loss `0.7 * MSE + 0.3 * L1`. Adjust batch size
for available memory. Full training requires substantial compute.

The same experiments can run from the terminal:

```bash
python libero.py download
python libero.py replay --tasks 0 3 4 --episodes 1 --video
python trm.py train --batch-size 32
python trm.py evaluate --tasks 0 3 4 --episodes 5 --video
```

Use `python libero.py --help` or `python trm.py --help` for options. To replay
all 500 human demonstrations, use `python libero.py replay --episodes 50`.
Downloads occupy approximately 6 GB, with assets in `external/LIBERO` and
recordings in `data/LIBERO/libero_spatial`. ResNet18 and CLIP weights download
on first model construction. The data and asset revisions are pinned in
`libero.py`; the Python dependency graph is pinned in `requirements.txt`.

Policy observations and training images are **128×128**. Videos use a separate
**1024×1024** camera at 20 fps. The model's original encoder preprocessing is
retained. MuJoCo compiles MJCF and maps initial coordinates; Drake integrates
every rollout with 0.25 ms physics substeps, source floating-joint damping,
and 2 ms gripper updates. Policy actions run at 20 Hz. Contact and rendering
can differ from the original simulator.

The loader retains the original HDF5 demo order, per-task 80/20 split,
`obs[t]` / `actions[t]` pairing, augmentation, and train-only action statistics.
Evaluation uses the first requested official initial states per task and a
600-step horizon. Checkpoints include configuration and action statistics;
checkpoints from `trm_on_libero.py` remain compatible. Training writes
`model.history.json` beside `model.pt` for plotting.

Sources: [original TRM notebook](https://github.com/fraadap/Tiny-Recursive-Models-For-Robotics/blob/50a65eac79e78fa3e0bf84b1faee05a41a952c16/trm_on_libero.ipynb),
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01),
and [human demonstrations](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/f13aa24a3da8c43c7225569f28c562979fa0e35a).
LIBERO code is MIT licensed; demonstrations are CC BY 4.0. Credit: Bo Liu,
Yifeng Zhu, Chongkai Gao, Yan Du, and Animesh Garg, *LIBERO: Benchmarking Knowledge
Transfer for Lifelong Robot Learning*, NeurIPS 2023.
