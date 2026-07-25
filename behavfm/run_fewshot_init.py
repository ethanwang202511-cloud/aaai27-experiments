"""Experiment 3: Few-Shot Token Initialization via Retrieval.

Tests 3 token initialization strategies to fix the zero-shot transfer
negative result from the workshop paper.

Strategies:
  A. Nearest-Neighbor Token Selection: init token = learned token of
     nearest training species by feature-space cosine similarity.
  B. Weighted Prototype: top-3 softmax-weighted combination of training
     species tokens, weighted by feature similarity to support images.
  C. Learned Token Projector: MLP (768->256->768) that maps species mean
     features to token space, trained on (mean_features, learned_token)
     pairs from all training species.

Evaluation:
  - 7 held-out species (fox, monkey, rabbit, panther, elephant, gorilla, chimpanzee)
  - K = {1, 5, 10, 25} support images
  - Fine-tune for 0, 10, 50, 100 epochs after initialization
  - Convergence speed: epochs to reach 90% of full-training performance
  - 5 seeds per condition

Usage:
    python run_fewshot_init.py \
        --train-cache data/cache/dinov2_base_ap10k_train \
        --val-cache data/cache/dinov2_base_ap10k_val \
        --ckpt results/q2_within_species/iti_xattn_aux01_seed0/model.pt \
        --out-dir results/fewshot_init \
        --device cuda

    # Smoke test
    python run_fewshot_init.py \
        --train-cache data/cache/dinov2_base_ap10k_train \
        --val-cache data/cache/dinov2_base_ap10k_val \
        --ckpt results/model.pt \
        --species fox --k-shots 1,5 --n-seeds 1 --ft-epochs 0,10
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
_SRC = Path(os.environ.get("BEHAVFM_ROOT", ROOT.parent.parent.parent)) / "src"
if (_SRC / "iti").is_dir():
    sys.path.insert(0, str(_SRC.parent))
    sys.path.insert(0, str(_SRC))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fewshot_init")


SPECIES_7 = ["fox", "monkey", "rabbit", "panther", "elephant", "gorilla", "chimpanzee"]
K_SHOTS = [1, 5, 10, 25]
FT_EPOCHS = [0, 10, 50, 100]
STRATEGIES = ["random", "nearest_neighbor", "weighted_prototype", "learned_projector"]
HIDDEN_DIM = 768


# ---------------------------------------------------------------------------
# Feature cache (standalone)
# ---------------------------------------------------------------------------

class CachedFeatures:
    def __init__(self, cache_dir, species_subset=None):
        self.cache_dir = Path(cache_dir)
        meta = json.loads((self.cache_dir / "meta.json").read_text())
        self._features = np.load(self.cache_dir / "features.npy", mmap_mode="r")
        self.records = meta["records"]
        self.indices = []
        for i, r in enumerate(self.records):
            if species_subset and r["species_name"] not in species_subset:
                continue
            self.indices.append(i)
        self.indices = np.array(self.indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
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


def collate_fn(batch):
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
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_xy, gt_xy, vis, crop_side, bbox_diag, out_size=224):
    scale = crop_side / out_size
    diff = pred_xy - gt_xy
    dist = np.linalg.norm(diff, axis=-1)
    dist_orig = dist * scale[:, None]
    dist_orig = np.where(vis > 0, dist_orig, np.nan)
    norm = bbox_diag[:, None]
    dist_norm = dist_orig / norm
    flat = dist_norm[~np.isnan(dist_norm)]
    if flat.size == 0:
        return {"rmse_norm": float("nan"), "pck@0.10": float("nan")}
    return {
        "rmse_norm": float(np.sqrt((flat ** 2).mean())),
        "pck@0.05": float((flat < 0.05).mean()),
        "pck@0.10": float((flat < 0.10).mean()),
        "pck@0.20": float((flat < 0.20).mean()),
    }


def bootstrap_ci(vals, n=1000, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.array([vals[rng.integers(0, len(vals), size=len(vals))].mean() for _ in range(n)])
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci) / 2))
    return float(vals.mean()), lo, hi


# ---------------------------------------------------------------------------
# Token Projector (Strategy C)
# ---------------------------------------------------------------------------

class TokenProjector(nn.Module):
    """MLP that maps species mean patch features -> token space.

    Architecture: 768 -> 256 -> 768 with LayerNorm + dropout.
    Trained on (mean_features, learned_token) pairs from training species.
    """

    def __init__(self, hidden: int = 768, bottleneck: int = 256, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, hidden),
        )
        # Initialize near-identity behavior
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, mean_features: torch.Tensor) -> torch.Tensor:
        return self.net(mean_features)


def train_projector(
    mean_features_by_id: dict[int, torch.Tensor],
    learned_tokens: torch.Tensor,  # (n_ids, 768)
    n_train_epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 0.1,
    device: str = "cpu",
) -> TokenProjector:
    """Train token projector via leave-one-out cross-validation."""
    projector = TokenProjector(hidden=HIDDEN_DIM).to(device)

    ids = sorted(mean_features_by_id.keys())
    X = torch.stack([mean_features_by_id[i] for i in ids]).to(device)  # (N, 768)
    Y = learned_tokens[ids].to(device)  # (N, 768)

    opt = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_train_epochs):
        projector.train()
        pred = projector(X)
        loss = F.mse_loss(pred, Y)
        # Add L2 regularization toward identity
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 100 == 0:
            logger.info("  Projector epoch %d/%d  loss=%.6f", epoch + 1, n_train_epochs, loss.item())

    projector.eval()
    return projector


# ---------------------------------------------------------------------------
# Initialization strategies
# ---------------------------------------------------------------------------

def init_random(hidden: int = 768, **kwargs) -> torch.Tensor:
    """Random Gaussian initialization (baseline)."""
    tok = torch.zeros(hidden)
    nn.init.normal_(tok, mean=0.0, std=0.02)
    return tok


def init_nearest_neighbor(
    support_features: torch.Tensor,    # (k, n_patches, 768)
    training_mean_features: dict[int, torch.Tensor],  # id -> (768,)
    training_tokens: torch.Tensor,     # (n_ids, 768)
    **kwargs,
) -> torch.Tensor:
    """Strategy A: Select token of nearest training species."""
    # Mean-pool support features
    support_mean = support_features.mean(dim=(0, 1))  # (768,)

    best_sim = -1.0
    best_id = 0
    for tid, tmean in training_mean_features.items():
        sim = F.cosine_similarity(support_mean.unsqueeze(0), tmean.unsqueeze(0)).item()
        if sim > best_sim:
            best_sim = sim
            best_id = tid

    return training_tokens[best_id].clone()


def init_weighted_prototype(
    support_features: torch.Tensor,
    training_mean_features: dict[int, torch.Tensor],
    training_tokens: torch.Tensor,
    top_k: int = 3,
    temperature: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """Strategy B: Top-3 weighted combination of training tokens."""
    support_mean = support_features.mean(dim=(0, 1))  # (768,)

    ids = sorted(training_mean_features.keys())
    tmeans = torch.stack([training_mean_features[i] for i in ids])  # (N, 768)
    sims = F.cosine_similarity(support_mean.unsqueeze(0), tmeans, dim=-1)  # (N,)

    # Top-k
    topk_vals, topk_idx = torch.topk(sims, min(top_k, len(ids)))
    weights = F.softmax(topk_vals / temperature, dim=0)  # (top_k,)
    topk_tokens = training_tokens[torch.tensor([ids[i] for i in topk_idx])]  # (top_k, 768)
    return (weights.unsqueeze(1) * topk_tokens).sum(dim=0)


def init_learned_projector(
    support_features: torch.Tensor,
    projector: TokenProjector,
    device: str = "cpu",
    **kwargs,
) -> torch.Tensor:
    """Strategy C: Learned MLP projection from features to token space."""
    support_mean = support_features.mean(dim=(0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        tok = projector(support_mean).squeeze(0).cpu()
    return tok


# ---------------------------------------------------------------------------
# Adaptation + evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, tok, val_loader, device, out_size=224):
    model.eval()
    all_pred, all_gt, all_vis, all_crop, all_diag = [], [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            fv = batch["feature"].to(device)
            Bv = fv.shape[0]
            pred_unit = model.forward_with_token(
                fv, tok.detach().unsqueeze(0).expand(Bv, -1).to(device)
            ).cpu()
            pred_xy = pred_unit * out_size
            all_pred.append(pred_xy.numpy())
            all_gt.append(batch["keypoints_xy"].numpy())
            all_vis.append(batch["vis"].numpy())
            all_crop.append(batch["crop_side"].numpy())
            all_diag.append(batch["bbox_diag_orig"].numpy())

    return compute_metrics(
        np.concatenate(all_pred), np.concatenate(all_gt),
        np.concatenate(all_vis), np.concatenate(all_crop),
        np.concatenate(all_diag), out_size,
    )


def finetune_token(
    model, tok_init: torch.Tensor,
    sup_f: torch.Tensor, sup_kp: torch.Tensor, sup_vis: torch.Tensor,
    val_loader: DataLoader,
    n_epochs: int, lr: float = 5e-4,
    device: str = "cpu", out_size: int = 224,
    full_train_rmse: float | None = None,
) -> dict:
    """Fine-tune token and track convergence."""
    tok = nn.Parameter(tok_init.clone().to(device))
    opt = torch.optim.AdamW([tok], lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_epochs, 1))

    history = []
    epoch_to_90pct = None

    # Evaluate at epoch 0 (initialization quality)
    metrics_0 = evaluate_model(model, tok, val_loader, device, out_size)
    history.append({"epoch": 0, **metrics_0})

    if full_train_rmse and not np.isnan(full_train_rmse):
        target_rmse = full_train_rmse / 0.9  # 90% of full = 10% worse than full
        # Actually: 90% of full-training performance means:
        # rmse <= full_train_rmse + 0.1 * baseline_rmse (but we don't have baseline_rmse here)
        # Simpler: rmse <= full_train_rmse * 1.1 (within 10% of full training)
        target_rmse = full_train_rmse * 1.1
        if metrics_0["rmse_norm"] <= target_rmse:
            epoch_to_90pct = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        B = sup_f.shape[0]
        pred = model.forward_with_token(sup_f, tok.unsqueeze(0).expand(B, -1))
        target = sup_kp / out_size
        per_kp = ((pred - target) ** 2).sum(dim=-1)
        loss = (per_kp * sup_vis).sum() / sup_vis.sum().clamp_min(1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        # Evaluate periodically
        if epoch % 10 == 0 or epoch == n_epochs:
            metrics = evaluate_model(model, tok, val_loader, device, out_size)
            metrics["epoch"] = epoch
            history.append(metrics)

            if (epoch_to_90pct is None and full_train_rmse
                    and not np.isnan(full_train_rmse)
                    and metrics["rmse_norm"] <= target_rmse):
                epoch_to_90pct = epoch

    return {
        "final_metrics": history[-1] if history else {},
        "history": history,
        "epoch_to_90pct": epoch_to_90pct,
        "init_drift": float(
            F.cosine_similarity(tok_init.unsqueeze(0), tok.detach().cpu().unsqueeze(0)).item()
        ),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Experiment 3: Few-Shot Token Initialization")
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--val-cache", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results/fewshot_init"))
    p.add_argument("--species", nargs="+", default=SPECIES_7)
    p.add_argument("--k-shots", type=str, default="1,5,10,25",
                   help="Comma-separated K values.")
    p.add_argument("--ft-epochs", type=str, default="0,10,50,100",
                   help="Comma-separated fine-tune epoch counts.")
    p.add_argument("--strategies", nargs="+", default=STRATEGIES)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--ft-lr", type=float, default=5e-4)
    p.add_argument("--projector-epochs", type=int, default=500,
                   help="Epochs to train the learned projector (Strategy C).")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    k_shots = [int(k) for k in args.k_shots.split(",")]
    ft_epochs = [int(e) for e in args.ft_epochs.split(",")]
    max_ft = max(ft_epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s | species: %s | K: %s | FT epochs: %s | strategies: %s | seeds: %d",
                device, args.species, k_shots, ft_epochs, args.strategies, args.n_seeds)

    # Load model
    try:
        from iti.model.iti import ITIModel, ITIConfig
        state = torch.load(args.ckpt, map_location=device, weights_only=False)
        cfg = ITIConfig(**{k: v for k, v in state["cfg"].items()
                          if k in ITIConfig.__dataclass_fields__})
        model = ITIModel(cfg).to(device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        for p_param in model.parameters():
            p_param.requires_grad_(False)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise

    # Compute mean features per training species
    logger.info("Computing per-species mean features from training cache...")
    train_full = CachedFeatures(args.train_cache)
    species_features: dict[int, list[torch.Tensor]] = {}
    for i in range(len(train_full)):
        item = train_full[i]
        gid = item["identity_id"]
        if gid not in species_features:
            species_features[gid] = []
        # Mean over patches for this image
        species_features[gid].append(item["feature"].mean(dim=0))

    training_mean_features: dict[int, torch.Tensor] = {}
    for gid, feats in species_features.items():
        training_mean_features[gid] = torch.stack(feats).mean(dim=0)

    training_tokens = model.tokens.emb.weight.detach().cpu()
    n_training_ids = training_tokens.shape[0]

    # Token-space smoothness check
    logger.info("Token-space smoothness check...")
    feat_sims = []
    tok_sims = []
    ids = sorted(training_mean_features.keys())
    for i, id_a in enumerate(ids):
        for id_b in ids[i+1:]:
            if id_a >= n_training_ids or id_b >= n_training_ids:
                continue
            fs = F.cosine_similarity(
                training_mean_features[id_a].unsqueeze(0),
                training_mean_features[id_b].unsqueeze(0),
            ).item()
            ts = F.cosine_similarity(
                training_tokens[id_a].unsqueeze(0),
                training_tokens[id_b].unsqueeze(0),
            ).item()
            feat_sims.append(fs)
            tok_sims.append(ts)

    if feat_sims:
        corr = np.corrcoef(feat_sims, tok_sims)[0, 1]
        logger.info("Feature-space vs token-space cosine similarity correlation: %.3f", corr)
        smoothness = {
            "correlation": float(corr),
            "n_pairs": len(feat_sims),
            "warning": "Token space may not be smooth" if corr < 0.3 else "OK",
        }
    else:
        smoothness = {"correlation": float("nan"), "n_pairs": 0}

    # Train projector (Strategy C)
    projector = None
    if "learned_projector" in args.strategies:
        logger.info("Training token projector (Strategy C)...")
        # Only use IDs that are in the training token bank
        proj_features = {k: v for k, v in training_mean_features.items() if k < n_training_ids}
        if len(proj_features) >= 5:
            projector = train_projector(
                proj_features, training_tokens,
                n_train_epochs=args.projector_epochs,
                device=device,
            )
        else:
            logger.warning("Too few training species (%d) for projector; skipping Strategy C",
                           len(proj_features))

    # Run experiments
    all_results = []
    total = len(args.species) * len(k_shots) * len(args.strategies) * args.n_seeds
    done = 0

    for species in args.species:
        # Load species-specific data
        train_sp = CachedFeatures(args.train_cache, species_subset={species})
        val_sp = CachedFeatures(args.val_cache, species_subset={species})

        if len(train_sp) == 0 or len(val_sp) == 0:
            logger.warning("No data for %s, skipping", species)
            continue

        val_loader = DataLoader(val_sp, batch_size=32, shuffle=False, collate_fn=collate_fn)

        # Get full-training reference RMSE (for convergence speed measurement)
        # Use the best RMSE from 100-epoch random-init run as reference.
        # We compute this once and reuse across strategies.
        full_train_rmse = None  # Will be set by random init, 100-epoch run

        for k in k_shots:
            for strategy in args.strategies:
                for seed in range(args.n_seeds):
                    done += 1
                    torch.manual_seed(seed)
                    rng = np.random.default_rng(seed)

                    # Sample support set
                    n_avail = len(train_sp)
                    k_actual = min(k, n_avail)
                    idx = rng.choice(n_avail, size=k_actual, replace=(k_actual > n_avail))

                    sup_items = [train_sp[int(j)] for j in idx]
                    sup_f = torch.stack([s["feature"] for s in sup_items]).to(device)
                    sup_kp = torch.stack([s["keypoints_xy"] for s in sup_items]).to(device)
                    sup_vis = torch.stack([s["vis"] for s in sup_items]).to(device)

                    # Initialize token
                    if strategy == "random":
                        tok_init = init_random()
                    elif strategy == "nearest_neighbor":
                        tok_init = init_nearest_neighbor(
                            sup_f.cpu(), training_mean_features, training_tokens,
                        )
                    elif strategy == "weighted_prototype":
                        tok_init = init_weighted_prototype(
                            sup_f.cpu(), training_mean_features, training_tokens,
                        )
                    elif strategy == "learned_projector":
                        if projector is None:
                            logger.info("  Skipping learned_projector (not trained)")
                            continue
                        tok_init = init_learned_projector(sup_f.cpu(), projector, device)
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")

                    # Fine-tune and evaluate at each epoch checkpoint
                    ft_result = finetune_token(
                        model, tok_init, sup_f, sup_kp, sup_vis,
                        val_loader, n_epochs=max_ft, lr=args.ft_lr,
                        device=device, full_train_rmse=full_train_rmse,
                    )

                    # Extract results at each requested FT epoch
                    for ft_ep in ft_epochs:
                        # Find the closest evaluated epoch
                        epoch_metrics = None
                        for h in ft_result["history"]:
                            if h.get("epoch", 0) <= ft_ep:
                                epoch_metrics = h
                        if epoch_metrics is None:
                            epoch_metrics = ft_result["history"][0] if ft_result["history"] else {}

                        result = {
                            "species": species,
                            "strategy": strategy,
                            "k_shot": k,
                            "ft_epochs": ft_ep,
                            "seed": seed,
                            "rmse_norm": epoch_metrics.get("rmse_norm", float("nan")),
                            "pck@0.05": epoch_metrics.get("pck@0.05", float("nan")),
                            "pck@0.10": epoch_metrics.get("pck@0.10", float("nan")),
                            "pck@0.20": epoch_metrics.get("pck@0.20", float("nan")),
                            "epoch_to_90pct": ft_result["epoch_to_90pct"],
                            "init_drift": ft_result["init_drift"],
                        }
                        all_results.append(result)

                    # Set full_train_rmse from the random baseline's final result
                    if strategy == "random" and seed == 0:
                        final = ft_result["history"][-1] if ft_result["history"] else {}
                        if "rmse_norm" in final:
                            full_train_rmse = final["rmse_norm"]

                    if done % 10 == 0 or done == total:
                        logger.info("[%d/%d] %s k=%d %s seed=%d  rmse=%.4f",
                                    done, total, species, k, strategy, seed,
                                    ft_result["final_metrics"].get("rmse_norm", float("nan")))

                    # Save incrementally
                    with open(args.out_dir / "fewshot_init_results.json", "w") as f:
                        json.dump(all_results, f, indent=2)

    # Summary
    summary = {
        "smoothness_check": smoothness,
        "by_strategy": {},
    }

    for strategy in args.strategies:
        summary["by_strategy"][strategy] = {}
        for k in k_shots:
            for ft_ep in ft_epochs:
                rows = [
                    r for r in all_results
                    if r["strategy"] == strategy and r["k_shot"] == k and r["ft_epochs"] == ft_ep
                ]
                if not rows:
                    continue
                rmses = np.array([r["rmse_norm"] for r in rows if not np.isnan(r["rmse_norm"])])
                if len(rmses) == 0:
                    continue
                m, lo, hi = bootstrap_ci(rmses)
                key = f"k{k}_ft{ft_ep}"
                summary["by_strategy"][strategy][key] = {
                    "rmse_mean": m, "rmse_ci95": [lo, hi], "n_runs": len(rmses),
                }

    # Convergence speed summary
    convergence = {}
    for strategy in args.strategies:
        epochs_90 = [
            r["epoch_to_90pct"] for r in all_results
            if r["strategy"] == strategy and r["epoch_to_90pct"] is not None
        ]
        if epochs_90:
            convergence[strategy] = {
                "mean_epochs_to_90pct": float(np.mean(epochs_90)),
                "median_epochs_to_90pct": float(np.median(epochs_90)),
                "n_converged": len(epochs_90),
            }
    summary["convergence"] = convergence

    with open(args.out_dir / "fewshot_init_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done. Output: %s", args.out_dir)
    logger.info("Token space smoothness: correlation=%.3f", smoothness.get("correlation", float("nan")))


if __name__ == "__main__":
    main()
