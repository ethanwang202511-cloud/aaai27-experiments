"""Experiment 3: Combinatorial dormancy on Norman 0/2 split with 3-way
stratification.

For double perturbations (g1, g2), stratify test pairs by how many genes
are seen in training singles:
  - 0/2: both genes unseen (standard GEARS 0/2 test set)
  - 1/2: one gene seen in training singles, one unseen
  - 2/2: both genes seen in training singles (easy control)

Compounding ratio = gap(0/2) / (2 * gap(0/1)), where
  gap = learned_rho - pop_mean_rho.
If > 1, dormancy compounds super-linearly when both genes are unseen.

For the 0/1 single-gene dormancy baseline, we hold out individual genes
from training singles and evaluate on those genes' single-perturbation
cells. This provides the per-gene gap needed for the denominator.

Models tested: CPA minimal, CPG, GEARS.

Usage:
  python run_combinatorial_dormancy.py --model cpa --seeds 0,1,2 --device cpu
  python run_combinatorial_dormancy.py --model all --seeds 0 --device cuda:0
"""
from __future__ import annotations

import argparse, json, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
INTERVENEFM_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "InterveneFM"
sys.path.insert(0, str(INTERVENEFM_ROOT))

from src.audit import get_top_deg_indices, predict_under_mode, run_audit
from src.cpa_cpg import CPGConfig, CPGModel, compute_gene_identity_table
from src.cpa_minimal import CPAConfig, CPAMinimal
from src.data_norman import (
    PerturbSeqDataset, build_gene_vocab, build_split, load_norman,
)

RESULTS_DIR = INTERVENEFM_ROOT / "results" / "exp3_combinatorial_dormancy"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_X_rows(adata, rows: np.ndarray) -> np.ndarray:
    X = adata.X
    if hasattr(X, "toarray"):
        return np.asarray(X[rows].toarray()).astype(np.float32)
    return np.asarray(X[rows]).astype(np.float32)


def _resolve_train_pert_genes(
    adata, split: Dict, vocab: Dict[str, int],
) -> Tuple[set, List[int], List[str]]:
    """Return (gene_set, 1-indexed vocab ids, perturbed cell ids) for training."""
    obs = adata.obs
    train_set = set(split["train_cells"])
    pert_genes_set: set = set(split.get("train_single_genes", []))
    pg, np_arr = obs["pert_genes"].values, obs["n_pert"].values
    perturbed_ids: List[str] = []
    for ci, cid in enumerate(obs.index):
        if cid in train_set and np_arr[ci] >= 1:
            for g in pg[ci]:
                if g in vocab:
                    pert_genes_set.add(g)
            perturbed_ids.append(cid)
    indices = sorted(vocab[g] for g in pert_genes_set if g in vocab)
    return pert_genes_set, indices, perturbed_ids


def _train_pert_mean(adata, cell_ids: List[str], device: str) -> Optional[torch.Tensor]:
    if not cell_ids:
        return None
    c2r = {c: i for i, c in enumerate(adata.obs.index)}
    rows = np.array([c2r[c] for c in cell_ids if c in c2r])
    if len(rows) == 0:
        return None
    return torch.from_numpy(_get_X_rows(adata, rows).mean(0)).to(device)


# ---------------------------------------------------------------------------
# Single-gene 0/1 split
# ---------------------------------------------------------------------------

def build_single_gene_01_split(adata, test_frac: float = 0.2, seed: int = 0) -> Dict:
    """Hold out individual genes from training singles for 0/1 evaluation."""
    rng = np.random.default_rng(seed)
    obs = adata.obs
    np_arr, pg_arr = obs["n_pert"].values, obs["pert_genes"].values

    singles: set = set()
    for ci in range(len(obs)):
        if np_arr[ci] == 1:
            singles.update(pg_arr[ci])
    singles_list = sorted(singles)
    n_test = max(1, int(round(test_frac * len(singles_list))))
    test_genes = set(singles_list[i] for i in rng.choice(len(singles_list), n_test, replace=False))
    train_single_genes = singles - test_genes

    ctrl_cells = list(obs.index[np_arr == 0])
    train_cells = list(ctrl_cells)
    test_cells: List[str] = []
    for ci, cid in enumerate(obs.index):
        if np_arr[ci] == 0:
            continue
        elif np_arr[ci] == 1:
            (test_cells if pg_arr[ci][0] in test_genes else train_cells).append(cid)
        else:
            train_cells.append(cid)

    print(f"[single_01] {len(test_genes)} test / {len(train_single_genes)} train genes, "
          f"{len(train_cells)} train / {len(test_cells)} test cells")
    return dict(train_cells=train_cells, test_cells=test_cells, ctrl_cells=ctrl_cells,
                test_genes=sorted(test_genes), train_single_genes=sorted(train_single_genes),
                withheld_single_genes=sorted(test_genes), kind="0/1")


def _audit_single_01(
    model, adata, split_01: Dict, vocab: Dict[str, int],
    train_pert_gene_indices: List[int], train_perturbed_ids: List[str],
    n_per_gene: int = 200, deg_k: int = 200, seed: int = 0, device: str = "cpu",
    is_cpg: bool = False,
) -> pd.DataFrame:
    """Audit single-gene 0/1 test genes for CPA or CPG."""
    rng = np.random.default_rng(seed)
    c2r = {c: i for i, c in enumerate(adata.obs.index)}
    ctrl_rows = np.array([c2r[c] for c in split_01["ctrl_cells"]])
    ctrl_X = _get_X_rows(adata, ctrl_rows)

    if is_cpg:
        t_idx = torch.tensor(train_pert_gene_indices, dtype=torch.long, device=device)
        t_pad = torch.cat([t_idx.unsqueeze(1), torch.zeros_like(t_idx.unsqueeze(1))], dim=1)
        with torch.no_grad():
            pool = model.pert_encoder(t_pad)
    else:
        pool = model.pert_encoder.embed.weight.data[train_pert_gene_indices].clone()
    tpm = _train_pert_mean(adata, train_perturbed_ids, device)

    pg_arr = adata.obs["pert_genes"].values
    rows_out: List[Dict] = []
    for tg in split_01["test_genes"]:
        mask = np.array([(len(g) == 1 and g[0] == tg) for g in pg_arr])
        pr = np.where(mask)[0]
        if len(pr) < 5:
            continue
        pX = _get_X_rows(adata, pr)
        deg = get_top_deg_indices(pX, ctrl_X, k=deg_k)
        obs_top = (pX.mean(0) - ctrl_X.mean(0))[deg]
        basal = torch.from_numpy(_get_X_rows(adata, rng.choice(ctrl_rows, n_per_gene, replace=True))).to(device)
        pidx = torch.tensor([[vocab.get(tg, 0), 0]] * n_per_gene, dtype=torch.long, device=device)
        for mode in ("learned", "mean", "zero", "random", "identity", "pop_mean"):
            xh = predict_under_mode(model, basal, pidx, mode, pool, rng, train_pert_mean_x=tpm).cpu().numpy()
            rho = spearmanr((xh.mean(0) - ctrl_X.mean(0))[deg], obs_top).statistic
            rows_out.append(dict(gene=tg, mode=mode,
                                 DE_Spearman=float(rho) if not np.isnan(rho) else np.nan))
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# CPA minimal
# ---------------------------------------------------------------------------

def _train_model(adata, split, vocab, model, epochs, batch_size, lr, device, seed):
    """Generic training loop for CPA/CPG."""
    ds = PerturbSeqDataset(adata, split["train_cells"], split["ctrl_cells"], vocab, max_perts=2, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    params = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    for epoch in range(epochs):
        model.train(); ep_loss = ep_n = 0
        for basal, target, pidx in loader:
            basal, target, pidx = basal.to(device), target.to(device), pidx.to(device)
            loss = F.mse_loss(model(basal, pidx), target)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * basal.shape[0]; ep_n += basal.shape[0]
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch+1}/{epochs}  mse={ep_loss/ep_n:.4f}")
    model.eval()
    return model


def run_cpa_stratified(adata, vocab, split_kind, seed=0, epochs=20,
                       device="cpu", n_per_pair=200, batch_size=128) -> Tuple[pd.DataFrame, Dict]:
    t0 = time.time()
    print(f"\n[cpa] split={split_kind} seed={seed}")
    split = build_split(adata, kind=split_kind, seed=seed)
    cfg = CPAConfig(n_genes=adata.n_vars, n_pert_genes=len(vocab), z_dim=64, pert_dim=32, hidden=256)
    model = CPAMinimal(cfg).to(device)
    print(f"  [cpa] params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    _train_model(adata, split, vocab, model, epochs, batch_size, 3e-4, device, seed)

    _, tpgi, tpci = _resolve_train_pert_genes(adata, split, vocab)
    df = run_audit(model, adata, split["test_pairs"], split["ctrl_cells"], vocab,
                   tpgi, n_per_pair=n_per_pair, seed=seed, train_perturbed_cell_ids=tpci)
    df["split_kind"] = split_kind; df["seed"] = seed; df["model"] = "cpa"
    print(f"  [cpa] done: {len(df)} rows in {time.time()-t0:.1f}s")
    return df, split


def run_cpa_single_01(adata, vocab, seed=0, epochs=20, device="cpu",
                      n_per_gene=200, batch_size=128) -> pd.DataFrame:
    t0 = time.time()
    print(f"\n[cpa] single 0/1 seed={seed}")
    split = build_single_gene_01_split(adata, seed=seed)
    cfg = CPAConfig(n_genes=adata.n_vars, n_pert_genes=len(vocab), z_dim=64, pert_dim=32, hidden=256)
    model = CPAMinimal(cfg).to(device)
    _train_model(adata, split, vocab, model, epochs, batch_size, 3e-4, device, seed)
    _, tpgi, tpci = _resolve_train_pert_genes(adata, split, vocab)
    df = _audit_single_01(model, adata, split, vocab, tpgi, tpci,
                          n_per_gene=n_per_gene, seed=seed, device=device, is_cpg=False)
    df["split_kind"] = "0/1"; df["seed"] = seed; df["model"] = "cpa"
    print(f"  [cpa] single 0/1 done: {len(df)} rows in {time.time()-t0:.1f}s")
    return df


# ---------------------------------------------------------------------------
# CPG
# ---------------------------------------------------------------------------

def _cpg_audit_doubles(model, adata, split, vocab, tpgi, tpci,
                       n_per_pair=200, deg_k=200, seed=0, device="cpu") -> pd.DataFrame:
    """6-mode audit on double-perturbation test pairs for CPG."""
    rng = np.random.default_rng(seed)
    c2r = {c: i for i, c in enumerate(adata.obs.index)}
    ctrl_rows = np.array([c2r[c] for c in split["ctrl_cells"]])
    ctrl_X = _get_X_rows(adata, ctrl_rows)

    t_idx = torch.tensor(tpgi, dtype=torch.long, device=device)
    t_pad = torch.cat([t_idx.unsqueeze(1), torch.zeros_like(t_idx.unsqueeze(1))], dim=1)
    with torch.no_grad():
        pool = model.pert_encoder(t_pad)
    tpm = _train_pert_mean(adata, tpci, device)
    pg_arr = adata.obs["pert_genes"].values
    rows: List[Dict] = []

    for g1, g2 in split["test_pairs"]:
        mask = np.array([(set(g) == {g1, g2}) and len(g) == 2 for g in pg_arr])
        pr = np.where(mask)[0]
        if len(pr) < 5:
            continue
        pX = _get_X_rows(adata, pr)
        deg = get_top_deg_indices(pX, ctrl_X, k=deg_k)
        obs_delta = pX.mean(0) - ctrl_X.mean(0)
        basal = torch.from_numpy(_get_X_rows(adata, rng.choice(ctrl_rows, n_per_pair, replace=True))).to(device)
        pidx = torch.tensor([[vocab.get(g1, 0), vocab.get(g2, 0)]] * n_per_pair,
                            dtype=torch.long, device=device)
        for mode in ("learned", "mean", "zero", "random", "identity", "pop_mean"):
            xh = predict_under_mode(model, basal, pidx, mode, pool, rng,
                                    train_pert_mean_x=tpm).cpu().numpy()
            pd_ = xh.mean(0) - ctrl_X.mean(0)
            rho = spearmanr(pd_[deg], obs_delta[deg]).statistic
            rows.append(dict(pair=f"{g1}_{g2}", mode=mode,
                             DE_Spearman=float(rho) if not np.isnan(rho) else np.nan,
                             n_obs_pert=len(pr)))
    return pd.DataFrame(rows)


def run_cpg_stratified(adata, adata_full, vocab, split_kind, seed=0, epochs=20,
                       device="cpu", n_per_pair=200, batch_size=128) -> Tuple[pd.DataFrame, Dict]:
    t0 = time.time()
    print(f"\n[cpg] split={split_kind} seed={seed}")
    split = build_split(adata, kind=split_kind, seed=seed)
    git = compute_gene_identity_table(adata_full, vocab, split["ctrl_cells"], d_id=64, seed=seed)
    cfg = CPGConfig(n_genes=adata.n_vars, n_pert_genes=len(vocab), gene_id_dim=64,
                    z_dim=64, pert_dim=32, hidden=256, pert_mlp_hidden=128)
    model = CPGModel(cfg, git).to(device)
    print(f"  [cpg] trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    _train_model(adata, split, vocab, model, epochs, batch_size, 3e-4, device, seed)
    _, tpgi, tpci = _resolve_train_pert_genes(adata, split, vocab)
    df = _cpg_audit_doubles(model, adata, split, vocab, tpgi, tpci,
                            n_per_pair=n_per_pair, seed=seed, device=device)
    df["split_kind"] = split_kind; df["seed"] = seed; df["model"] = "cpg"
    print(f"  [cpg] done: {len(df)} rows in {time.time()-t0:.1f}s")
    return df, split


def run_cpg_single_01(adata, adata_full, vocab, seed=0, epochs=20,
                      device="cpu", n_per_gene=200, batch_size=128) -> pd.DataFrame:
    t0 = time.time()
    print(f"\n[cpg] single 0/1 seed={seed}")
    split = build_single_gene_01_split(adata, seed=seed)
    git = compute_gene_identity_table(adata_full, vocab, split["ctrl_cells"], d_id=64, seed=seed)
    cfg = CPGConfig(n_genes=adata.n_vars, n_pert_genes=len(vocab), gene_id_dim=64,
                    z_dim=64, pert_dim=32, hidden=256, pert_mlp_hidden=128)
    model = CPGModel(cfg, git).to(device)
    _train_model(adata, split, vocab, model, epochs, batch_size, 3e-4, device, seed)
    _, tpgi, tpci = _resolve_train_pert_genes(adata, split, vocab)
    df = _audit_single_01(model, adata, split, vocab, tpgi, tpci,
                          n_per_gene=n_per_gene, seed=seed, device=device, is_cpg=True)
    df["split_kind"] = "0/1"; df["seed"] = seed; df["model"] = "cpg"
    print(f"  [cpg] single 0/1 done: {len(df)} rows in {time.time()-t0:.1f}s")
    return df


# ---------------------------------------------------------------------------
# GEARS shared helpers
# ---------------------------------------------------------------------------

def _parse_gears_condition(c: str) -> List[str]:
    return [p for p in c.split("+") if p and p != "ctrl"]


@dataclass
class _GearsAuditCtx:
    """Bundles the state needed for GEARS audit after training."""
    model: object                      # GEARS nn.Module
    pert_data: object                  # PertData
    pert_names: List[str]
    train_conditions: List[str]
    test_conditions: List[str]
    train_single_gene_set: set
    train_pert_indices: List[int]
    pert_global_emb_learned: torch.Tensor
    ctrl_mean: np.ndarray
    obs_delta_per_cond: Dict[str, np.ndarray]
    tpm_delta: np.ndarray
    device: str


def _build_gears_ctx(seed: int, epochs: int, hidden_size: int,
                     batch_size: int, device: str) -> _GearsAuditCtx:
    """Train GEARS and prepare all audit state."""
    from gears import GEARS, PertData

    data_dir = str(INTERVENEFM_ROOT / "data" / "gears_pertdata")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    pd_ = PertData(data_dir)
    pd_.load(data_name="norman")
    pd_.prepare_split(split="simulation", seed=seed)
    pd_.get_dataloader(batch_size=batch_size, test_batch_size=128)

    gm = GEARS(pd_, device=device)
    gm.model_initialize(hidden_size=hidden_size)
    print(f"  [gears] params: {sum(p.numel() for p in gm.model.parameters())/1e6:.2f}M")
    gm.train(epochs=epochs, lr=1e-3)

    model = gm.model; model.eval()
    pn = list(pd_.pert_names) if hasattr(pd_, "pert_names") else list(gm.pert_list)
    train_conds = pd_.set2conditions.get("train", [])
    test_conds = pd_.set2conditions.get("test", [])

    tpgs: set = set()
    tsgs: set = set()
    for c in train_conds:
        genes = _parse_gears_condition(c)
        tpgs.update(genes)
        if len(genes) == 1:
            tsgs.add(genes[0])
    tpi = [i for i, n in enumerate(pn) if n in tpgs]

    with torch.no_grad():
        idx_all = torch.arange(model.num_perts, device=device)
        emb = model.pert_emb(idx_all)
        for layer in model.sim_layers:
            emb = layer(emb, model.G_sim, model.G_sim_weight)
        emb = emb.detach()

    # Ground truth
    ad = pd_.adata; X = ad.X
    ctrl_mask = (ad.obs["condition"] == "ctrl").values

    def _dense(mask):
        return np.asarray(X[mask].toarray()).astype(np.float32) if hasattr(X, "toarray") \
            else np.asarray(X[mask]).astype(np.float32)

    ctrl_mean = _dense(ctrl_mask).mean(0)
    obs_delta: Dict[str, np.ndarray] = {}
    for c in test_conds:
        m = (ad.obs["condition"] == c).values
        if m.sum() >= 5:
            obs_delta[c] = _dense(m).mean(0) - ctrl_mean

    tpm_mask = np.zeros(ad.n_obs, dtype=bool)
    for c in train_conds:
        tpm_mask |= (ad.obs["condition"] == c).values
    tpm_mask &= ~ctrl_mask
    tpm_delta = _dense(tpm_mask).mean(0) - ctrl_mean

    return _GearsAuditCtx(model=model, pert_data=pd_, pert_names=pn,
                          train_conditions=train_conds, test_conditions=test_conds,
                          train_single_gene_set=tsgs, train_pert_indices=tpi,
                          pert_global_emb_learned=emb, ctrl_mean=ctrl_mean,
                          obs_delta_per_cond=obs_delta, tpm_delta=tpm_delta, device=device)


def _gears_run_with_override(ctx: _GearsAuditCtx, override: torch.Tensor) -> Dict[str, np.ndarray]:
    """Forward pass with monkey-patched pert_emb + identity GNN."""
    model = ctx.model

    class FakeEmb(torch.nn.Module):
        def __init__(self, ov): super().__init__(); self.ov = ov
        def forward(self, idx): return self.ov

    class IdGNN(torch.nn.Module):
        def forward(self, x, *a, **k): return x

    orig_emb = model.pert_emb
    orig_layers = list(model.sim_layers)
    model.pert_emb = FakeEmb(override)
    for i in range(len(orig_layers)):
        model.sim_layers[i] = IdGNN()
    try:
        preds: Dict[str, List[np.ndarray]] = {}
        with torch.no_grad():
            tl_key = next((k for k in ("test_loader", "test") if k in ctx.pert_data.dataloader),
                          list(ctx.pert_data.dataloader.keys())[-1])
            for batch in ctx.pert_data.dataloader[tl_key]:
                batch = batch.to(ctx.device)
                pred = model(batch)
                pred = pred[0] if isinstance(pred, tuple) else pred
                if hasattr(batch, "pert"):
                    pnp = pred.detach().cpu().numpy()
                    for i, c in enumerate(batch.pert):
                        preds.setdefault(c, []).append(pnp[i])
        return {c: np.stack(v).mean(0) for c, v in preds.items()}
    finally:
        model.pert_emb = orig_emb
        for i, l in enumerate(orig_layers):
            model.sim_layers[i] = l


def _gears_audit_all_modes(ctx: _GearsAuditCtx, conditions: List[str],
                           deg_k: int, seed: int) -> List[Dict]:
    """Run 4 model-based + 2 model-free audit modes. Returns list of row dicts."""
    rng = np.random.default_rng(seed)
    num_perts = ctx.model.num_perts
    emb = ctx.pert_global_emb_learned
    tpi = ctx.train_pert_indices
    pool = emb[torch.tensor(tpi, dtype=torch.long, device=ctx.device)]
    mean_row = pool.mean(0, keepdim=True)

    overrides = {
        "learned": emb,
        "mean": mean_row.expand(num_perts, -1).clone(),
        "zero": torch.zeros_like(emb),
        "random": emb[torch.tensor(rng.choice(tpi, num_perts, replace=True),
                                   dtype=torch.long, device=ctx.device)],
    }

    results: List[Dict] = []
    for mode, ov in overrides.items():
        preds = _gears_run_with_override(ctx, ov)
        for c in conditions:
            if c not in preds or c not in ctx.obs_delta_per_cond:
                continue
            od = ctx.obs_delta_per_cond[c]
            pd_ = preds[c] - ctx.ctrl_mean
            deg = np.argsort(-np.abs(od))[:deg_k]
            rho = spearmanr(pd_[deg], od[deg]).statistic
            results.append(dict(condition=c, mode=mode,
                                DE_Spearman=float(rho) if not np.isnan(rho) else np.nan))

    for c in conditions:
        if c not in ctx.obs_delta_per_cond:
            continue
        od = ctx.obs_delta_per_cond[c]
        deg = np.argsort(-np.abs(od))[:deg_k]
        rho_pm = spearmanr(ctx.tpm_delta[deg], od[deg]).statistic
        results.append(dict(condition=c, mode="pop_mean",
                            DE_Spearman=float(rho_pm) if not np.isnan(rho_pm) else np.nan))
        rho_id = spearmanr(np.zeros_like(od[deg]), od[deg]).statistic
        results.append(dict(condition=c, mode="identity",
                            DE_Spearman=float(rho_id) if not np.isnan(rho_id) else np.nan))
    return results


# ---------------------------------------------------------------------------
# GEARS: public runners
# ---------------------------------------------------------------------------

def run_gears_stratified(seed: int = 0, epochs: int = 20, hidden_size: int = 64,
                         batch_size: int = 32, deg_k: int = 200,
                         device: str = "cuda:0") -> pd.DataFrame:
    """Train GEARS, audit, and post-hoc stratify doubles into 0/2, 1/2, 2/2."""
    t0 = time.time()
    print(f"\n[gears] stratified doubles seed={seed}")
    ctx = _build_gears_ctx(seed, epochs, hidden_size, batch_size, device)

    # Audit only double-perturbation test conditions
    double_conds = [c for c in ctx.test_conditions if len(_parse_gears_condition(c)) == 2]
    rows = _gears_audit_all_modes(ctx, double_conds, deg_k, seed)
    df = pd.DataFrame(rows)

    strata = []
    for c in df["condition"]:
        genes = _parse_gears_condition(c)
        n_in = sum(1 for g in genes if g in ctx.train_single_gene_set) if len(genes) == 2 else -1
        strata.append(f"{n_in}/2" if n_in >= 0 else "other")
    df["stratum"] = strata
    df = df.rename(columns={"condition": "pair"})
    df["seed"] = seed; df["model"] = "gears"
    print(f"  [gears] doubles done: {len(df)} rows in {time.time()-t0:.1f}s")
    return df


def run_gears_single_01(seed: int = 0, epochs: int = 20, hidden_size: int = 64,
                        batch_size: int = 32, deg_k: int = 200,
                        device: str = "cuda:0") -> pd.DataFrame:
    """Train GEARS, audit single-gene test conditions where gene is unseen."""
    t0 = time.time()
    print(f"\n[gears] single 0/1 seed={seed}")
    ctx = _build_gears_ctx(seed, epochs, hidden_size, batch_size, device)

    unseen = [c for c in ctx.test_conditions
              if len(_parse_gears_condition(c)) == 1
              and _parse_gears_condition(c)[0] not in ctx.train_single_gene_set]
    if not unseen:
        print("  [gears] WARNING: no unseen single-gene test conditions.")
        return pd.DataFrame()

    rows = _gears_audit_all_modes(ctx, unseen, deg_k, seed)
    df = pd.DataFrame(rows)
    df["gene"] = df["condition"].apply(lambda c: _parse_gears_condition(c)[0])
    df = df.drop(columns=["condition"])
    df["split_kind"] = "0/1"; df["seed"] = seed; df["model"] = "gears"
    print(f"  [gears] single 0/1 done: {len(df)} rows in {time.time()-t0:.1f}s")
    return df


# ---------------------------------------------------------------------------
# Compounding ratio and aggregation
# ---------------------------------------------------------------------------

def _compute_gap(df: pd.DataFrame, id_col: str = "pair") -> pd.Series:
    piv = df.pivot_table(index=id_col, columns="mode", values="DE_Spearman")
    if "learned" not in piv.columns or "pop_mean" not in piv.columns:
        return pd.Series(dtype=float)
    return (piv["learned"] - piv["pop_mean"]).dropna()


def _bootstrap_ci(vals: np.ndarray, n_boot: int = 5000, seed: int = 42
                  ) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    boots = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(n_boot)])
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def compute_compounding_ratio(gap_02: np.ndarray, gap_01: np.ndarray,
                              n_boot: int = 5000, seed: int = 42) -> Dict:
    """ratio = mean(gap_02) / (2 * mean(gap_01)); >1 means super-linear compounding."""
    rng = np.random.default_rng(seed)
    if len(gap_02) == 0 or len(gap_01) == 0:
        return dict(ratio=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_02=len(gap_02), n_01=len(gap_01))
    m02, m01 = gap_02.mean(), gap_01.mean()
    ratio = m02 / (2 * m01) if abs(m01) > 1e-9 else np.nan
    br = []
    for _ in range(n_boot):
        b02 = rng.choice(gap_02, len(gap_02), replace=True).mean()
        b01 = rng.choice(gap_01, len(gap_01), replace=True).mean()
        br.append(b02 / (2 * b01) if abs(b01) > 1e-9 else np.nan)
    br = np.array(br); br = br[~np.isnan(br)]
    lo = float(np.percentile(br, 2.5)) if len(br) else np.nan
    hi = float(np.percentile(br, 97.5)) if len(br) else np.nan
    return dict(ratio=float(ratio), ci_lo=lo, ci_hi=hi,
                mean_gap_02=float(m02), mean_gap_01=float(m01),
                n_02=len(gap_02), n_01=len(gap_01))


def aggregate_results(double_dfs: List[pd.DataFrame], single_dfs: List[pd.DataFrame],
                      out_dir: Path) -> pd.DataFrame:
    if not double_dfs and not single_dfs:
        print("[aggregate] no results"); return pd.DataFrame()

    df_d = pd.concat(double_dfs, ignore_index=True) if double_dfs else pd.DataFrame()
    df_s = pd.concat(single_dfs, ignore_index=True) if single_dfs else pd.DataFrame()
    if len(df_d): df_d.to_csv(out_dir / "all_double_audit.csv", index=False)
    if len(df_s): df_s.to_csv(out_dir / "all_single01_audit.csv", index=False)

    models = set()
    if len(df_d): models |= set(df_d["model"].unique())
    if len(df_s): models |= set(df_s["model"].unique())
    summary_rows: List[Dict] = []

    for mn in sorted(models):
        print(f"\n{'='*50}\n  Model: {mn}\n{'='*50}")
        gaps: Dict[str, float] = {}
        for sk in ("0/2", "1/2", "2/2"):
            sub = df_d[(df_d["model"] == mn) & (df_d["split_kind"] == sk)] if len(df_d) else pd.DataFrame()
            if len(sub) == 0:
                print(f"  {sk}: no data"); continue
            gap = _compute_gap(sub)
            m, lo, hi = _bootstrap_ci(gap.values)
            print(f"  {sk}: n={len(gap)}, gap={m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            summary_rows.append(dict(model=mn, stratum=sk, n=len(gap), mean_gap=m, ci_lo=lo, ci_hi=hi))
            gaps[sk] = m

        sub01 = df_s[df_s["model"] == mn] if len(df_s) else pd.DataFrame()
        gap_01 = _compute_gap(sub01, "gene") if len(sub01) else pd.Series(dtype=float)
        if len(gap_01):
            m01, lo01, hi01 = _bootstrap_ci(gap_01.values)
            print(f"  0/1: n={len(gap_01)}, gap={m01:+.4f} [{lo01:+.4f}, {hi01:+.4f}]")
            summary_rows.append(dict(model=mn, stratum="0/1", n=len(gap_01),
                                     mean_gap=m01, ci_lo=lo01, ci_hi=hi01))

        # Compounding ratio
        gap02_vals = _compute_gap(df_d[(df_d["model"]==mn) & (df_d["split_kind"]=="0/2")]) \
            if len(df_d) and "0/2" in set(df_d[df_d["model"]==mn]["split_kind"]) else pd.Series(dtype=float)
        if len(gap02_vals) and len(gap_01):
            cr = compute_compounding_ratio(gap02_vals.values, gap_01.values)
            print(f"  Compounding ratio: {cr['ratio']:.3f} [{cr['ci_lo']:.3f}, {cr['ci_hi']:.3f}]")
            summary_rows.append(dict(model=mn, stratum="compounding_ratio",
                                     n=cr["n_02"], mean_gap=cr["ratio"],
                                     ci_lo=cr["ci_lo"], ci_hi=cr["ci_hi"]))

        # Monotonic test
        ordered = [(sk, gaps[sk]) for sk in ("2/2", "1/2", "0/2") if sk in gaps]
        if len(ordered) >= 2:
            mono = all(ordered[i][1] <= ordered[i+1][1] for i in range(len(ordered)-1))
            print(f"  Monotonic (2/2 <= 1/2 <= 0/2): {mono}")

    sdf = pd.DataFrame(summary_rows)
    if len(sdf):
        sdf.to_csv(out_dir / "stratified_summary.csv", index=False)
        print(f"\n[saved] {out_dir / 'stratified_summary.csv'}")
    return sdf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Exp 3: Combinatorial dormancy")
    ap.add_argument("--model", default="cpa", choices=["cpa", "cpg", "gears", "all"])
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hvg", type=int, default=2000)
    ap.add_argument("--max_cells", type=int, default=60000)
    ap.add_argument("--n_per_pair", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--skip_singles", action="store_true",
                    help="Skip single-gene 0/1 (no compounding ratio).")
    ap.add_argument("--gears_hidden", type=int, default=64)
    ap.add_argument("--gears_batch_size", type=int, default=32)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    models = [args.model] if args.model != "all" else ["cpa", "cpg", "gears"]
    print(f"[exp3] models={models}, seeds={seeds}, epochs={args.epochs}, device={args.device}")
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    need_data = any(m in ("cpa", "cpg") for m in models)
    adata = adata_full = vocab = None
    if need_data:
        print("\n[data] Loading Norman 2019...")
        adata = load_norman(n_top_hvg=args.hvg, max_cells=args.max_cells, seed=0)
        vocab = build_gene_vocab(adata)
        print(f"[data] adata={adata.shape}, vocab={len(vocab)}")
        if "cpg" in models:
            import scanpy as sc
            from src.data_norman import parse_pert
            af = sc.read_h5ad(INTERVENEFM_ROOT / "data" / "norman_2019.h5ad")
            af.obs["nperts"] = af.obs["nperts"].astype(int)
            af.obs["pert_genes"] = af.obs["perturbation"].astype(str).apply(parse_pert).values
            af.obs["n_pert"] = [len(x) for x in af.obs["pert_genes"]]
            sc.pp.normalize_total(af, target_sum=1e4); sc.pp.log1p(af)
            adata_full = af
            print(f"[data] full-panel for CPG: {adata_full.shape}")

    all_d: List[pd.DataFrame] = []
    all_s: List[pd.DataFrame] = []

    for seed in seeds:
        np.random.seed(seed); torch.manual_seed(seed)

        if "cpa" in models:
            for sk in ("0/2", "1/2", "2/2"):
                df, _ = run_cpa_stratified(adata, vocab, sk, seed=seed, epochs=args.epochs,
                                           device=args.device, n_per_pair=args.n_per_pair,
                                           batch_size=args.batch_size)
                df.to_csv(RESULTS_DIR / f"cpa_{sk.replace('/','')}_seed{seed}.csv", index=False)
                all_d.append(df)
            if not args.skip_singles:
                df01 = run_cpa_single_01(adata, vocab, seed=seed, epochs=args.epochs,
                                         device=args.device, n_per_gene=args.n_per_pair,
                                         batch_size=args.batch_size)
                df01.to_csv(RESULTS_DIR / f"cpa_01_seed{seed}.csv", index=False)
                all_s.append(df01)

        if "cpg" in models:
            for sk in ("0/2", "1/2", "2/2"):
                df, _ = run_cpg_stratified(adata, adata_full, vocab, sk, seed=seed,
                                           epochs=args.epochs, device=args.device,
                                           n_per_pair=args.n_per_pair, batch_size=args.batch_size)
                df.to_csv(RESULTS_DIR / f"cpg_{sk.replace('/','')}_seed{seed}.csv", index=False)
                all_d.append(df)
            if not args.skip_singles:
                df01 = run_cpg_single_01(adata, adata_full, vocab, seed=seed,
                                         epochs=args.epochs, device=args.device,
                                         n_per_gene=args.n_per_pair, batch_size=args.batch_size)
                df01.to_csv(RESULTS_DIR / f"cpg_01_seed{seed}.csv", index=False)
                all_s.append(df01)

        if "gears" in models:
            df_g = run_gears_stratified(seed=seed, epochs=args.epochs,
                                        hidden_size=args.gears_hidden,
                                        batch_size=args.gears_batch_size, device=args.device)
            if len(df_g):
                df_g.to_csv(RESULTS_DIR / f"gears_strata_seed{seed}.csv", index=False)
                for st in ("0/2", "1/2", "2/2"):
                    sub = df_g[df_g["stratum"] == st].copy()
                    if len(sub):
                        sub["split_kind"] = st
                        all_d.append(sub)
            if not args.skip_singles:
                df_g01 = run_gears_single_01(seed=seed, epochs=args.epochs,
                                             hidden_size=args.gears_hidden,
                                             batch_size=args.gears_batch_size, device=args.device)
                if len(df_g01):
                    df_g01.to_csv(RESULTS_DIR / f"gears_01_seed{seed}.csv", index=False)
                    all_s.append(df_g01)

    print(f"\n{'='*50}\n  AGGREGATION\n{'='*50}")
    aggregate_results(all_d, all_s, RESULTS_DIR)

    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(dict(models=models, seeds=seeds, epochs=args.epochs, hvg=args.hvg,
                       max_cells=args.max_cells, n_per_pair=args.n_per_pair,
                       device=args.device, gears_hidden=args.gears_hidden), f, indent=2)

    print(f"\n[exp3] total: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")
    print(f"[exp3] results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
