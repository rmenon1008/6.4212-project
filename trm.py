"""The original TRM model, native human dataset, training, and policy evaluation.

Model equations and parameter names follow fraadap/Tiny-Recursive-Models-For-Robotics
at 50a65eac79e78fa3e0bf84b1faee05a41a952c16 (trm_on_libero.ipynb).
TrainingConfig defaults match its main_pipeline training invocation.
"""

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from transformers import CLIPTextModel, CLIPTokenizer

from libero import ASSETS, DATA, NAMES, LiberoEnv, load_initial_states


@dataclass
class TrainingConfig:
    """Training and model hyperparameters container."""

    lr: float = 0.0001
    hidden_dim: int = 256
    num_recursions: int = 16
    epochs: int = 30
    batch_size: int = 512
    weight_decay: float = 0.1
    grad_clip: float | None = 1.0
    early_stop_patience: int | None = 10
    save_path: str = "runs/model.pt"
    freeze_backbone: bool = False
    augmentation: bool = True
    dropout: float = 0.1
    encoder_dropout: float = 0.1
    use_text_prompts: bool = True
    text_encoder_name: str = "openai/clip-vit-large-patch14"
    train_text_encoder: bool = False
    text_dropout: float = 0.1
    double_visual_features: bool = True
    use_attention: bool = False
    attention_fusion: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PretrainedVisualEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        freeze_backbone: bool = True,
        dropout: float = 0.1,
        double_visual_features: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.adapter = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim * 2)
            if double_visual_features
            else nn.Linear(512, hidden_dim),
        )
        self.ln = nn.LayerNorm(hidden_dim * (2 if double_visual_features else 1))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        features = self.backbone(x).flatten(start_dim=1)
        output = self.adapter(features)
        return output


class PromptEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        model_name: str = "openai/clip-vit-large-patch14",
        trainable: bool = False,
        dropout: float = 0.3,
        max_length: int = 77,
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        self.max_length = min(
            max_length, getattr(self.text_model.config, "max_position_embeddings", max_length)
        )
        if not trainable:
            self.text_model.eval()
            for param in self.text_model.parameters():
                param.requires_grad = False
        self.text_hidden = self.text_model.config.hidden_size
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.text_hidden),
            nn.Linear(self.text_hidden, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self._token_cache: dict[str, dict[str, torch.Tensor]] = {}

    def _tokenize(self, prompt: str) -> dict[str, torch.Tensor]:
        if prompt not in self._token_cache:
            tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            self._token_cache[prompt] = {k: v for k, v in tokens.items()}
        cached = self._token_cache[prompt]
        return {k: v.clone() for k, v in cached.items()}

    def forward(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        if len(prompts) == 0:
            raise ValueError("PromptEncoder received an empty batch of prompts")
        token_batches = [self._tokenize(p) for p in prompts]
        batch = {
            key: torch.cat([tokens[key] for tokens in token_batches], dim=0).to(device)
            for key in token_batches[0]
        }
        outputs = self.text_model(**batch)
        pooled = (
            outputs.pooler_output
            if outputs.pooler_output is not None
            else outputs.last_hidden_state[:, -1, :]
        )
        return self.adapter(pooled)


class RecursiveBlock(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.1, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.dropout = nn.Dropout(dropout)
        else:
            self.attention = None
            self.dropout = None
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, h, x_cond):
        h_in = h.unsqueeze(1)
        if self.use_attention:
            cond = x_cond.unsqueeze(1)
            attn_out, _ = self.attention(h_in, cond, cond)
            attn_out = self.dropout(attn_out)
            x = self.norm(attn_out + self.mlp(attn_out))
        else:
            x = self.norm(h_in + self.mlp(h_in))
        return x.squeeze(1)


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query_proj = nn.Linear(visual_dim, hidden_dim)
        self.key_proj = nn.Linear(text_dim, hidden_dim)
        self.value_proj = nn.Linear(text_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.resid_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, visual_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(visual_feats).unsqueeze(1)
        k = self.key_proj(text_feats).unsqueeze(1)
        v = self.value_proj(text_feats).unsqueeze(1)
        attn_out, _ = self.attn(q, k, v)
        fused = self.resid_norm(q + attn_out)
        fused = self.out_norm(fused + self.ffn(fused))
        return fused.squeeze(1)


class TRMPolicy(nn.Module):
    def __init__(
        self,
        obs_shape=(3, 128, 128),
        action_dim=7,
        hidden_dim=256,
        num_heads=4,
        num_recursions=8,
        dropout=0.1,
        freeze_backbone=True,
        encoder_dropout=0.1,
        use_text_prompts=True,
        text_encoder_name="openai/clip-vit-large-patch14",
        train_text_encoder=False,
        text_dropout=0.1,
        double_visual_features=False,
        use_attention: bool = True,
        attention_fusion: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_recursions = num_recursions
        self.obs_shape = obs_shape
        self.use_text_prompts = use_text_prompts
        self.use_attention = use_attention
        self.attention_fusion = attention_fusion and use_text_prompts
        self.encoder = PretrainedVisualEncoder(
            hidden_dim=hidden_dim,
            freeze_backbone=freeze_backbone,
            dropout=encoder_dropout,
            double_visual_features=double_visual_features,
        )
        visual_dim = hidden_dim * (2 if double_visual_features else 1)
        self.fusion_adapter: nn.Module
        if self.use_text_prompts:
            self.prompt_encoder = PromptEncoder(
                hidden_dim=hidden_dim,
                model_name=text_encoder_name,
                trainable=train_text_encoder,
                dropout=text_dropout,
            )
            if self.attention_fusion:
                self.fusion_adapter = CrossAttentionFusion(
                    visual_dim=visual_dim,
                    text_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            else:
                fusion_in = visual_dim + hidden_dim
                self.fusion_adapter = nn.Sequential(
                    nn.LayerNorm(fusion_in),
                    nn.Linear(fusion_in, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
        else:
            self.prompt_encoder = None
            fusion_in = visual_dim
            if fusion_in != hidden_dim:
                self.fusion_adapter = nn.Sequential(
                    nn.LayerNorm(fusion_in),
                    nn.Linear(fusion_in, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            else:
                self.fusion_adapter = nn.Identity()
        self.recursive_block = RecursiveBlock(
            hidden_dim, num_heads, dropout, use_attention=self.use_attention
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs, prompts: list[str] | None = None, return_all_states=False):
        visual_features = self.encoder(obs)
        if self.use_text_prompts:
            text_features = self.prompt_encoder(prompts, device=obs.device)
            if self.attention_fusion:
                x_cond = self.fusion_adapter(visual_features, text_features)
            else:
                x_cond = self.fusion_adapter(torch.cat([visual_features, text_features], dim=-1))
        else:
            x_cond = self.fusion_adapter(visual_features)
        h = x_cond.clone()
        states = [h] if return_all_states else None
        for t in range(self.num_recursions):
            h = self.recursive_block(h, x_cond)
            if return_all_states:
                states.append(h)
        actions = self.action_head(h)
        if return_all_states:
            return (actions, states)
        return actions


def build_policy_from_config(
    config: TrainingConfig, obs_shape: tuple[int, int, int] = (3, 128, 128)
) -> TRMPolicy:
    return TRMPolicy(
        obs_shape=obs_shape,
        action_dim=7,
        hidden_dim=config.hidden_dim,
        num_recursions=config.num_recursions,
        dropout=config.dropout,
        freeze_backbone=config.freeze_backbone,
        encoder_dropout=config.encoder_dropout,
        use_text_prompts=config.use_text_prompts,
        text_encoder_name=config.text_encoder_name,
        train_text_encoder=config.train_text_encoder,
        text_dropout=config.text_dropout,
        double_visual_features=config.double_visual_features,
        use_attention=getattr(config, "use_attention", True),
        attention_fusion=getattr(config, "attention_fusion", False),
    )


class LIBERODataset(Dataset):
    """Upstream obs[t]/action[t] pairs, with its per-file 80/20 demo split.

    Read pixels lazily rather than copying the entire six-GB dataset into RAM.
    Keep HDF5 group order, raw OpenGL images, and train-only z-score statistics.
    """

    def __init__(self, data_dir=DATA, is_train=True, action_stats=None, augmentation=False):
        self.data, self.samples, self.files = [], [], {}
        self.augmentation = augmentation and is_train
        self.image_size = (128, 128)
        all_actions = []
        for name in NAMES:
            path = Path(data_dir) / (name + "_demo.hdf5")
            with h5py.File(path, "r") as f:
                keys = list(f["data"].keys())[:50]
                split = int(0.8 * len(keys))
                for key in keys[:split] if is_train else keys[split:]:
                    g = f[f"data/{key}"]
                    actions = g["actions"][:].astype(np.float32)
                    images = g["obs/agentview_rgb"]
                    if images.shape != (len(actions), 128, 128, 3) or actions.shape[1:] != (7,):
                        raise ValueError(f"Malformed original demonstration: {path}/{key}")
                    if not np.isfinite(actions).all():
                        raise ValueError(f"Nonfinite human actions: {path}/{key}")
                    index = len(self.data)
                    self.data.append((path, key, actions, name.replace("_", " ")))
                    self.samples.extend((index, t) for t in range(len(actions)))
                    all_actions.append(actions)
        if not self.samples:
            raise ValueError("No demonstrations; run the download command first")
        if action_stats is None:
            actions = np.concatenate(all_actions)
            action_stats = {"mean": actions.mean(0), "std": np.clip(actions.std(0), 0.1, None)}
        self.action_stats = {k: np.asarray(v, dtype=np.float32) for k, v in action_stats.items()}
        print(f"{'train' if is_train else 'val'}: {len(self.data)} demos, {len(self)} frames")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        demo, t = self.samples[index]
        path, key, actions, prompt = self.data[demo]
        if path not in self.files:
            self.files[path] = h5py.File(path, "r")
        image = self.files[path][f"data/{key}/obs/agentview_rgb"][t].astype(np.float32) / 255
        if self.augmentation:
            image = self._augment_obs(image[None])[0]
        action = (actions[t] - self.action_stats["mean"]) / self.action_stats["std"]
        return {
            "observations": torch.from_numpy(image.transpose(2, 0, 1).copy()),
            "actions": torch.from_numpy(action),
            "prompt": prompt,
        }

    def _augment_obs(self, obs):
        # Same brightness, contrast, and random crop as the original notebook.
        import cv2

        if np.random.rand() < 0.5:
            obs = np.clip(obs * np.random.uniform(0.8, 1.2), 0, 1)
        if np.random.rand() < 0.3:
            contrast = np.random.uniform(0.8, 1.2)
            mean = obs.mean(axis=(1, 2), keepdims=True)
            obs = np.clip((obs - mean) * contrast + mean, 0, 1)
        if np.random.rand() < 0.3:
            crop_ratio = np.random.uniform(0.85, 0.95)
            height, width = (int(x * crop_ratio) for x in self.image_size)
            y = np.random.randint(0, 128 - height + 1)
            x = np.random.randint(0, 128 - width + 1)
            obs = np.stack(
                [cv2.resize(img[y : y + height, x : x + width], (128, 128)) for img in obs]
            )
        return obs

    def close(self):
        for f in self.files.values():
            f.close()
        self.files.clear()


def device_for(name):
    return torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(config, data_dir=DATA, device="auto", max_batches=None):
    """Original AdamW behavior cloning loop: 0.7 MSE + 0.3 L1, no scheduler."""
    device = device_for(device)
    train_data = LIBERODataset(data_dir, augmentation=config.augmentation)
    val_data = LIBERODataset(data_dir, is_train=False, action_stats=train_data.action_stats)
    train_loader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_data, batch_size=config.batch_size)
    if not len(train_loader):
        raise ValueError("Batch size exceeds the number of training samples")
    model = build_policy_from_config(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, stale = float("inf"), 0
    history = []
    output = Path(config.save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for epoch in range(config.epochs):
            losses = []
            for training, loader in [(True, train_loader), (False, val_loader)]:
                model.train(training)
                total, batches = 0.0, 0
                for batch in loader:
                    image = batch["observations"].to(device)
                    target = batch["actions"].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with (
                        torch.set_grad_enabled(training),
                        torch.autocast("cuda", enabled=use_amp and training),
                    ):
                        predicted = model(image, batch["prompt"])
                        loss = 0.7 * F.mse_loss(predicted, target) + 0.3 * F.l1_loss(
                            predicted, target
                        )
                    if training:
                        scaler.scale(loss).backward()
                        if config.grad_clip:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    total += loss.item()
                    batches += 1
                    if max_batches and batches >= max_batches:
                        break
                losses.append(total / batches)
            print(f"epoch {epoch + 1}: train={losses[0]:.5f}, val={losses[1]:.5f}", flush=True)
            history.append({"epoch": epoch + 1, "train_loss": losses[0], "val_loss": losses[1]})
            output.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
            if losses[1] < best:
                best, stale = losses[1], 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best,
                        "config": asdict(config),
                        "action_stats": {k: v.tolist() for k, v in train_data.action_stats.items()},
                    },
                    output,
                )
            else:
                stale += 1
            if config.early_stop_patience and stale >= config.early_stop_patience:
                break
    finally:
        train_data.close()
        val_data.close()
    return best


def evaluate(
    checkpoint="runs/model.pt",
    data_dir=DATA,
    libero_root=ASSETS,
    task_ids=tuple(range(10)),
    episodes=5,
    max_steps=600,
    output="runs/evaluate",
    video=False,
    device="auto",
    action_stats="action_stats.json",
    seed=42,
):
    """Evaluate a learned policy on the first N official initial states per task."""
    task_ids = tuple(task_ids)
    if not task_ids or any(t not in range(10) for t in task_ids) or not 1 <= episodes <= 50:
        raise ValueError("Choose task IDs 0–9 and 1–50 official starts per task")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    seed_everything(seed)
    device = device_for(device)
    stored = torch.load(checkpoint, map_location=device, weights_only=True)
    config = TrainingConfig(
        **{k: v for k, v in stored["config"].items() if k in TrainingConfig.__dataclass_fields__}
    )
    policy = build_policy_from_config(config).to(device).eval()
    policy.load_state_dict(stored["model_state_dict"])
    stats = stored.get("action_stats")
    if stats is None:
        stats = json.loads(Path(action_stats).read_text())
    mean, std = (torch.tensor(stats[k], device=device) for k in ("mean", "std"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for task_id in task_ids:
        starts = load_initial_states(libero_root, task_id)
        for index in range(episodes):
            env = LiberoEnv(data_dir, task_id, "demo_0", libero_root, video=video)
            writer = None
            try:
                obs = env.reset(starts[index])
                if video:
                    writer = imageio.get_writer(
                        output / f"task_{task_id:02d}_{index:02d}.mp4",
                        fps=20,
                        codec="libx264",
                        macro_block_size=1,
                    )
                success = False
                for step in range(max_steps):
                    if writer:
                        writer.append_data(obs["video_rgb"])
                    # Match the native LIBERO OpenGL pixels used for training.
                    image = torch.from_numpy(obs["agentview_rgb"][::-1].copy())
                    image = image.permute(2, 0, 1).float()[None].to(device) / 255
                    with torch.inference_mode():
                        action = (
                            (policy(image, [NAMES[task_id].replace("_", " ")])[0] * std + mean)
                            .cpu()
                            .numpy()
                        )
                    obs, _, done, info = env.step(action)
                    success |= done
                    if done:
                        break
                if writer:
                    writer.append_data(obs["video_rgb"])
                row = {
                    "task_id": task_id,
                    "episode": index,
                    "success": success,
                    "final_success": info["success"],
                    "steps": step + 1,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
                report = {
                    "engine": "Drake",
                    "physics_time_step": env.time_step,
                    "mode": "policy",
                    "checkpoint": str(checkpoint),
                    "seed": seed,
                    "success_rate": float(np.mean([r["success"] for r in rows])),
                    "episodes": rows,
                }
                (output / "results.json").write_text(json.dumps(report, indent=2))
            finally:
                if writer:
                    writer.close()
                env.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--data", type=Path, default=DATA)
        p.add_argument("--checkpoint", type=Path, default=Path("runs/model.pt"))
        p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
        p.add_argument("--seed", type=int, default=42)
        if command == "train":
            p.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
            p.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
            p.add_argument("--max-batches", type=int, help="Limit each split for a smoke run")
        else:
            p.add_argument("--assets", type=Path, default=ASSETS)
            p.add_argument(
                "--tasks", nargs="+", type=int, choices=range(10), default=list(range(10))
            )
            p.add_argument("--episodes", type=int, choices=range(1, 51), default=5)
            p.add_argument("--max-steps", type=int, default=600)
            p.add_argument("--output", type=Path, default=Path("runs/evaluate"))
            p.add_argument(
                "--video",
                action="store_true",
                help="Save 1024x1024 video; policy input stays 128x128",
            )
            p.add_argument(
                "--action-stats",
                type=Path,
                default=Path("action_stats.json"),
                help="Sidecar statistics for an original upstream checkpoint",
            )
    args = parser.parse_args()
    if args.command == "train":
        if (
            args.epochs < 1
            or args.batch_size < 1
            or (args.max_batches is not None and args.max_batches < 1)
        ):
            parser.error("epochs, batch size, and max batches must be positive")
        seed_everything(args.seed)
        train(
            TrainingConfig(
                epochs=args.epochs, batch_size=args.batch_size, save_path=str(args.checkpoint)
            ),
            args.data,
            args.device,
            args.max_batches,
        )
    else:
        evaluate(
            checkpoint=args.checkpoint,
            data_dir=args.data,
            libero_root=args.assets,
            task_ids=args.tasks,
            episodes=args.episodes,
            max_steps=args.max_steps,
            output=args.output,
            video=args.video,
            device=args.device,
            action_stats=args.action_stats,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
