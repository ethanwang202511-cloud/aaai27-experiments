"""Experiment 1: Cross-Backbone Validation.

Tests whether ITA's frozen-features-plus-token-conditioning paradigm
generalizes across 5 vision foundation models, or depends on DINOv2's
uniquely strong patch-level representations.

Backbones: DINOv2-base, DINOv1-base, MAE-base, CLIP-ViT-B/16, EVA-02-B/16
Species: 3 held-out (fox=close, elephant=mid, chimpanzee=far)
Seeds: 10 per condition
Controls: frozen-backbone + full decoder fine-tune (no ITA) per backbone

Methodology from EXPERIMENT_PLAN.md Experiment 1:
  - For each backbone: extract features, train ITA with identical
    hyperparameters (768-dim token, cross-attention decoder, AdamW lr=1e-3,
    100 epochs, 10 seeds).
  - Also run full decoder fine-tuning control.
  - Report: RMSE_norm, PCK@0.05/0.10/0.20 with 95% CI.

Usage:
    # Smoke test
    python run_cross_backbone.py \
        --cache-dir data/cache/multi_backbone \
        --val-cache-dir data/cache/multi_backbone_val \
        --species fox --backbones dinov2 --n-seeds 1 --n-epochs 10

    # Full experiment
    python run_cross_backbone.py \
        --cache-dir data/cache/multi_backbone \
        --val-cache-dir data/cache/multi_backbone_val \
        --out-dir results/cross_backbone \
        --device cuda
"""

from __future__ import annotations
import argparse
import copy
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cross_backbone")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKBONES = ["dinov2", "dino", "mae", "clip", "eva02"]

BACKBONE_PATCH_COUNTS = {
    "dinov2": 256,   # 14x14 patches at 224x224
    "dino":   196,   # 16x16 patches
    "mae":    196,
    "clip":   196,
    "eva02":  196,
}

BACKBONE_GRID_SIZES = {
    "dinov2": 16,
    "dino":   14,
    "mae":    14,
    "clip":   14,
    "eva02":  14,
}

SPECIES = ["fox", "elephant", "chimpanzee"]
METHODS = ["ita_token_only", "full_decoder_ft"]

HIDDEN_DIM = 768

@dataclass
class ExpConfig:
    n_epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    n_seeds: int = 10
    k_shot: int = 5
    n_adapt_steps: int = 50
    adapt_lr: float = 1.0
    out_size: int = 224
    decoder_layers: int = 2
    decoder_dim: int = 384
    n_keypoints: int = 17
    n_cross_attn_layers: int = 2


# ---------------------------------------------------------------------------
# Lightweight decoder (re-implemented to avoid circular imports)
# ---------------------------------------------------------------------------

class CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 6) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        out, _ = self.attn(q_norm, kv_norm, kv_norm, need_weights=False)
        q = q + out
        q = q + self.mlp(self.norm2(q))
        return q


class PoseDecoderFlexible(nn.Module):
    """Identity-conditioned coord-regression decoder that handles variable patch counts.

    Same architecture as the original PoseDecoder but accepts variable-length
    patch sequences (196 or 256 patches depending on backbone).
    """

    def __init__(
        self,
        in_dim: int = 768,
        n_keypoints: int = 17,
        max_patches: int = 256,
        n_layers: int = 2,
        decoder_dim: int = 384,
        n_cross_attn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.n_keypoints = n_keypoints
        self.max_patches = max_patches

        self.in_proj = nn.Linear(in_dim, decoder_dim)
        # Learnable positional embeddings up to max_patches
        self.pos_emb = nn.Parameter(torch.zeros(max_patches, decoder_dim))
        nn.init.normal_(self.pos_emb, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=6, dim_feedforward=4 * decoder_dim,
            dropout=0.0, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.id_proj = nn.Linear(in_dim, decoder_dim)
        self.id_pos = nn.Parameter(torch.zeros(1, decoder_dim))
        nn.init.normal_(self.id_pos, std=0.02)

        self.kp_queries = nn.Parameter(torch.zeros(n_keypoints, decoder_dim))
        nn.init.normal_(self.kp_queries, std=0.02)

        self.cross_attn_blocks = nn.ModuleList(
            [CrossAttnBlock(decoder_dim, heads=6) for _ in range(n_cross_attn_layers)]
        )

        self.coord_head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 2),
        )

    def forward(
        self,
        patch_features: torch.Tensor,  # (B, N, in_dim) -- N can vary
        id_token: torch.Tensor,        # (B, in_dim)
    ) -> torch.Tensor:
        B, N, _ = patch_features.shape
        x = self.in_proj(patch_features) + self.pos_emb[:N].unsqueeze(0)
        ctx = self.encoder(x)

        id_ctx = self.id_proj(id_token).unsqueeze(1) + self.id_pos.unsqueeze(0)
        ctx_with_id = torch.cat([ctx, id_ctx], dim=1)

        q = self.kp_queries.unsqueeze(0).expand(B, -1, -1)
        for block in self.cross_attn_blocks:
            q = block(q, ctx_with_id)
        coords = self.coord_head(q)
        return torch.sigmoid(coords)


# ---------------------------------------------------------------------------
# Feature cache dataset (simplified, standalone)
# ---------------------------------------------------------------------------

class CachedFeatures(Dataset):
    """Load precomputed features from .npy + meta.json."""

    def __init__(
        self,
        cache_dir: Path,
        species_subset: set[str] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        meta = json.loads((self.cache_dir / "meta.json").read_text())
        shape_info = json.loads((self.cache_dir / "features.shape.json").read_text())
        self._features = np.load(self.cache_dir / "features.npy", mmap_mode="r")
        self.records = meta["records"]
        self.n_patches = shape_info["shape"][1]
        self.hidden_dim = shape_info["shape"][2]

        # Filter by species
        self.indices = []
        for i, r in enumerate(self.records):
            if species_subset and r["species_name"] not in species_subset:
                continue
            self.indices.append(i)
        self.indices = np.array(self.indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        ridx = int(self.indices[i])
        r = self.records[ridx]
        feat = torch.from_numpy(np.array(self._features[ridx])).float()
        return {
            "feature": feat,
            "identity_id": r["identity_id"],
            "keypoints_xy": torch.tensor(r["keypoints_xy"]).float(),
            "vis": torch.tensor(r["vis"]).float(),
            "bbox_diag_orig": float(r["bbox_diag_orig"]),
            "species_name": r["species_name"],
            "crop_side": float(r["crop_side"]),
        }


def collate_cached(batch: list[dict]) -> dict:
    return {
        "feature": torch.stack([b["feature"] for b in batch]),
        "identity_id": torch.tensor([b["identity_id"] for b in batch], dtype=torch.long),
        "keypoints_xy": torch.stack([b["keypoints_xy"] for b in batch]),
        "vis": torch.stack([b["vis"] for b in batch]),
        "bbox_diag_orig": torch.tensor([b["bbox_diag_orig"] for b in batch]),
        "species_name": [b["species_name"] for b in batch],
        "crop_side": torch.tensor([b["crop_side"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Metrics (standalone reimplementation)
# ---------------------------------------------------------------------------

def compute_metrics(
    pred_xy: np.ndarray,    # (B, K, 2) in crop coords
    gt_xy: np.ndarray,      # (B, K, 2)
    vis: np.ndarray,        # (B, K)
    crop_side: np.ndarray,  # (B,)
    bbox_diag: np.ndarray,  # (B,)
    out_size: int = 224,
) -> dict:
    scale = crop_side / out_size
    diff = pred_xy - gt_xy
    dist = np.linalg.norm(diff, axis=-1)           # (B, K) in crop px
    dist_orig = dist * scale[:, None]               # in original px
    dist_orig = np.where(vis > 0, dist_orig, np.nan)
    norm = bbox_diag[:, None]
    dist_norm = dist_orig / norm
    flat = dist_norm[~np.isnan(dist_norm)]
    if flat.size == 0:
        return {"rmse_norm": float("nan"), "pck@0.05": float("nan"),
                "pck@0.10": float("nan"), "pck@0.20": float("nan")}
    return {
        "rmse_norm": float(np.sqrt((flat ** 2).mean())),
        "pck@0.05": float((flat < 0.05).mean()),
        "pck@0.10": float((flat < 0.10).mean()),
        "pck@0.20": float((flat < 0.20).mean()),
    }


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, ci: float = 0.95, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.array([values[rng.integers(0, n, size=n)].mean() for _ in range(n_resamples)])
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci) / 2))
    return float(values.mean()), lo, hi


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def coord_loss(pred_unit, gt_xy, vis, out_size=224):
    target = gt_xy / out_size
    per_kp = ((pred_unit - target) ** 2).sum(dim=-1)
    denom = vis.sum().clamp_min(1.0)
    return (per_kp * vis).sum() / denom


def train_ita(
    decoder: PoseDecoderFlexible,
    id_token: nn.Parameter,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ExpConfig,
    device: str,
    method: str = "ita_token_only",
) -> dict:
    """Train decoder + id_token on cached features.

    method='ita_token_only': only token is trainable (768 params).
    method='full_decoder_ft': full decoder is trainable (no token).
    """
    decoder = decoder.to(device)

    if method == "ita_token_only":
        # Freeze decoder, only train token
        for p in decoder.parameters():
            p.requires_grad_(False)
        id_token.requires_grad_(True)
        trainable = [id_token]
    elif method == "full_decoder_ft":
        # Train full decoder, freeze token
        for p in decoder.parameters():
            p.requires_grad_(True)
        id_token.requires_grad_(False)
        trainable = list(decoder.parameters())
    else:
        raise ValueError(f"Unknown method: {method}")

    n_trainable = sum(p.numel() for p in trainable)
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)

    best_metrics = {"rmse_norm": float("inf")}

    for epoch in range(cfg.n_epochs):
        decoder.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            feat = batch["feature"].to(device)
            kp = batch["keypoints_xy"].to(device)
            vis = batch["vis"].to(device)
            B = feat.shape[0]

            tok = id_token.unsqueeze(0).expand(B, -1)
            pred = decoder(feat, tok)
            loss = coord_loss(pred, kp, vis, cfg.out_size)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == cfg.n_epochs - 1:
            decoder.eval()
            all_pred, all_gt, all_vis, all_crop, all_diag = [], [], [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    feat = batch["feature"].to(device)
                    B = feat.shape[0]
                    tok = id_token.detach().unsqueeze(0).expand(B, -1)
                    pred = decoder(feat, tok).cpu()
                    pred_xy = pred * cfg.out_size
                    all_pred.append(pred_xy.numpy())
                    all_gt.append(batch["keypoints_xy"].numpy())
                    all_vis.append(batch["vis"].numpy())
                    all_crop.append(batch["crop_side"].numpy())
                    all_diag.append(batch["bbox_diag_orig"].numpy())

            metrics = compute_metrics(
                np.concatenate(all_pred), np.concatenate(all_gt),
                np.concatenate(all_vis), np.concatenate(all_crop),
                np.concatenate(all_diag), cfg.out_size,
            )
            if metrics["rmse_norm"] < best_metrics["rmse_norm"]:
                best_metrics = metrics

            if (epoch + 1) % 20 == 0:
                logger.info(
                    "  epoch %3d/%d  loss=%.4e  rmse=%.4f  pck@0.05=%.3f",
                    epoch + 1, cfg.n_epochs,
                    epoch_loss / max(n_batches, 1),
                    metrics["rmse_norm"], metrics["pck@0.05"],
                )

    return {
        "best_metrics": best_metrics,
        "n_trainable": n_trainable,
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_one_condition(
    backbone_name: str,
    species: str,
    method: str,
    seed: int,
    cfg: ExpConfig,
    cache_dir: Path,
    val_cache_dir: Path,
    device: str,
) -> dict:
    """Run one (backbone, species, method, seed) condition."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_patches = BACKBONE_PATCH_COUNTS[backbone_name]

    # Load cached features
    train_ds = CachedFeatures(cache_dir / backbone_name, species_subset={species})
    val_ds = CachedFeatures(val_cache_dir / backbone_name, species_subset={species})

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.warning("No data for %s/%s (train=%d, val=%d)", backbone_name, species,
                       len(train_ds), len(val_ds))
        return {"status": "skipped", "reason": "no data"}

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_cached, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_cached,
    )

    # Build decoder
    decoder = PoseDecoderFlexible(
        in_dim=HIDDEN_DIM,
        n_keypoints=cfg.n_keypoints,
        max_patches=n_patches,
        n_layers=cfg.decoder_layers,
        decoder_dim=cfg.decoder_dim,
        n_cross_attn_layers=cfg.n_cross_attn_layers,
    )

    # Initialize identity token
    id_token = nn.Parameter(torch.randn(HIDDEN_DIM) * 0.02)

    result = train_ita(
        decoder, id_token, train_loader, val_loader,
        cfg, device, method,
    )

    return {
        "backbone": backbone_name,
        "species": species,
        "method": method,
        "seed": seed,
        "n_trainable": result["n_trainable"],
        **result["best_metrics"],
        "status": "ok",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Experiment 1: Cross-Backbone Validation")
    p.add_argument("--cache-dir", type=Path, required=True,
                   help="Root dir with per-backbone train feature caches.")
    p.add_argument("--val-cache-dir", type=Path, required=True,
                   help="Root dir with per-backbone val feature caches.")
    p.add_argument("--out-dir", type=Path, default=Path("results/cross_backbone"))
    p.add_argument("--backbones", type=str, default=",".join(BACKBONES))
    p.add_argument("--species", nargs="+", default=SPECIES)
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    cfg = ExpConfig(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_seeds=args.n_seeds,
    )

    backbone_names = [b.strip() for b in args.backbones.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total = len(backbone_names) * len(args.species) * len(args.methods) * args.n_seeds
    done = 0

    for backbone in backbone_names:
        for species in args.species:
            for method in args.methods:
                for seed in range(args.n_seeds):
                    logger.info(
                        "[%d/%d] backbone=%s species=%s method=%s seed=%d",
                        done + 1, total, backbone, species, method, seed,
                    )
                    result = run_one_condition(
                        backbone, species, method, seed,
                        cfg, args.cache_dir, args.val_cache_dir, device,
                    )
                    all_results.append(result)
                    done += 1

                    # Save incrementally
                    out_file = args.out_dir / "cross_backbone_results.json"
                    with open(out_file, "w") as f:
                        json.dump(all_results, f, indent=2)

    # Summary statistics
    summary = {}
    for backbone in backbone_names:
        summary[backbone] = {}
        for species in args.species:
            summary[backbone][species] = {}
            for method in args.methods:
                rows = [
                    r for r in all_results
                    if r["backbone"] == backbone
                    and r["species"] == species
                    and r["method"] == method
                    and r.get("status") == "ok"
                ]
                if not rows:
                    continue
                rmses = np.array([r["rmse_norm"] for r in rows])
                pcks05 = np.array([r["pck@0.05"] for r in rows])
                pcks10 = np.array([r["pck@0.10"] for r in rows])

                rmse_mean, rmse_lo, rmse_hi = bootstrap_ci(rmses)
                pck05_mean, pck05_lo, pck05_hi = bootstrap_ci(pcks05)
                pck10_mean, pck10_lo, pck10_hi = bootstrap_ci(pcks10)

                summary[backbone][species][method] = {
                    "n_runs": len(rows),
                    "n_trainable": rows[0]["n_trainable"],
                    "rmse_norm": {"mean": rmse_mean, "ci95": [rmse_lo, rmse_hi]},
                    "pck@0.05": {"mean": pck05_mean, "ci95": [pck05_lo, pck05_hi]},
                    "pck@0.10": {"mean": pck10_mean, "ci95": [pck10_lo, pck10_hi]},
                }

    summary_file = args.out_dir / "cross_backbone_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done. Results: %s | Summary: %s", out_file, summary_file)
    logger.info("Total runs: %d (ok: %d)",
                len(all_results),
                sum(1 for r in all_results if r.get("status") == "ok"))


if __name__ == "__main__":
    main()
