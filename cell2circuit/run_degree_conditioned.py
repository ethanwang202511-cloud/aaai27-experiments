"""run_degree_conditioned.py -- Experiment 3: Degree-conditioned DiGress variant.

Feeds predicted degree sequences into the generative process rather than
post-hoc Sinkhorn correction. Monitors mutual information between learned
edge scores and output adjacency.

The key idea: instead of generating edges with uniform prior then projecting
onto degree constraints via Sinkhorn (which destroys 92-100% of top-E edges),
we condition the generative model on degree sequences during generation.

Two variants:
  (a) Degree-conditioned DiGress: embed in/out-degree as per-node features,
      add auxiliary degree-marginal loss.
  (b) Degree-marginal-coherent flow (DeFoG variant): constrain the noising
      process to preserve degree marginals at each timestep.

AAAI-27 Experiment 3.
GPU required for training. Output: results/aaai27/degree_conditioned.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch not available. Degree-conditioned training disabled.",
          flush=True)

# ---------------------------------------------------------------------------
# Inlined helpers (standalone -- no Cell2Circuit src/ dependency)
# ---------------------------------------------------------------------------

def er_graph(N: int, density: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = (rng.random((N, N)) < density).astype(np.float32)
    np.fill_diagonal(a, 0)
    return a


def sinkhorn_degree_match(
    probs: np.ndarray, out_deg: np.ndarray, in_deg: np.ndarray,
    num_iters: int = 50, eps: float = 1e-8,
) -> np.ndarray:
    P = probs.copy().astype(np.float64)
    np.fill_diagonal(P, 0)
    for _ in range(num_iters):
        row_sums = P.sum(axis=1, keepdims=True).clip(eps)
        P = P * (out_deg[:, None] / row_sums)
        P = np.clip(P, 0, 1)
        col_sums = P.sum(axis=0, keepdims=True).clip(eps)
        P = P * (in_deg[None, :] / col_sums)
        P = np.clip(P, 0, 1)
    return P.astype(np.float32)


def configuration_model_sample(adj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    adj = (adj > 0.5).astype(np.int32)
    N = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int).tolist()
    out_deg = adj.sum(axis=1).astype(int).tolist()
    if sum(in_deg) == 0 or sum(out_deg) == 0:
        return np.zeros((N, N), dtype=np.float32)
    if sum(in_deg) != sum(out_deg):
        diff = sum(out_deg) - sum(in_deg)
        if diff > 0:
            order = np.argsort(out_deg)[::-1]
            i = 0
            while diff > 0 and i < len(order):
                if out_deg[order[i]] > 0:
                    out_deg[order[i]] -= 1
                    diff -= 1
                i = (i + 1) % len(order)
        else:
            diff = -diff
            order = np.argsort(in_deg)[::-1]
            i = 0
            while diff > 0 and i < len(order):
                if in_deg[order[i]] > 0:
                    in_deg[order[i]] -= 1
                    diff -= 1
                i = (i + 1) % len(order)
    seed = int(rng.integers(0, 1 << 30))
    try:
        G = nx.directed_configuration_model(
            in_degree_sequence=in_deg,
            out_degree_sequence=out_deg,
            create_using=nx.MultiDiGraph,
            seed=seed,
        )
        G = nx.DiGraph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        out = np.zeros((N, N), dtype=np.float32)
        for u, v in G.edges():
            out[u, v] = 1.0
        return out
    except Exception:
        d = float(adj.mean())
        return er_graph(N, d, seed=seed)


def _adj_to_digraph(adj: np.ndarray) -> "nx.DiGraph":
    G = nx.DiGraph()
    N = adj.shape[0]
    G.add_nodes_from(range(N))
    rows, cols = np.where(adj > 0)
    for r, c in zip(rows.tolist(), cols.tolist()):
        if r != c:
            G.add_edge(int(r), int(c))
    return G


def reciprocity(adj: np.ndarray) -> float:
    A = np.asarray(adj, dtype=np.float32)
    np.fill_diagonal(A, 0.0)
    total = float(A.sum())
    if total == 0.0:
        return 0.0
    recip = float((A * A.T).sum())
    return recip / total


def motif_3node_spectrum(adj: np.ndarray) -> np.ndarray:
    G = _adj_to_digraph(adj)
    try:
        triad_dict = nx.triads_by_type(G)
    except Exception:
        return np.zeros(16, dtype=np.float64)

    _TRIAD_TYPES = sorted([
        "003", "012", "021C", "021D", "021U",
        "030C", "030T", "102", "111D", "111U",
        "120C", "120D", "120U", "201", "210", "300",
    ])

    counts = np.zeros(16, dtype=np.float64)
    for i, t in enumerate(_TRIAD_TYPES):
        counts[i] = float(len(triad_dict.get(t, [])))
    return counts

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_STEPS = 30000
BATCH_SIZE = 8
LR = 3e-4
LAMBDA_DEG = 0.1       # auxiliary degree loss weight
LAMBDA_DEG_ANNEAL = 5000  # anneal from 0 to LAMBDA_DEG over this many steps
DEG_EMBED_DIM = 64
HIDDEN_DIM = 256
N_LAYERS = 4
N_HEADS = 8
SEED = 42


# ---------------------------------------------------------------------------
# Degree embedding module
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class DegreeEmbedding(nn.Module):
        """Embed in-degree and out-degree as per-node features.

        Maps (in_deg, out_deg) -> R^{deg_embed_dim} via MLP.
        """
        def __init__(self, max_degree: int = 50, embed_dim: int = DEG_EMBED_DIM):
            super().__init__()
            self.max_degree = max_degree
            self.in_embed = nn.Embedding(max_degree + 1, embed_dim // 2)
            self.out_embed = nn.Embedding(max_degree + 1, embed_dim // 2)
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )

        def forward(self, in_deg: Tensor, out_deg: Tensor) -> Tensor:
            """
            Args:
                in_deg:  [B, N] int, clamped to [0, max_degree]
                out_deg: [B, N] int, clamped to [0, max_degree]
            Returns:
                [B, N, embed_dim]
            """
            in_d = in_deg.clamp(0, self.max_degree)
            out_d = out_deg.clamp(0, self.max_degree)
            h_in = self.in_embed(in_d)      # [B, N, embed_dim//2]
            h_out = self.out_embed(out_d)    # [B, N, embed_dim//2]
            h = torch.cat([h_in, h_out], dim=-1)  # [B, N, embed_dim]
            return self.proj(h)

    # ---------------------------------------------------------------------------
    # Degree-conditioned DiGress model
    # ---------------------------------------------------------------------------

    class DegreeConditionedDigress(nn.Module):
        """DiGress with degree conditioning.

        Extends the standard discrete flow matching model by:
        1. Concatenating degree embeddings to node features at each denoising step.
        2. Adding an auxiliary loss penalizing degree-marginal divergence.
        """
        def __init__(self, base_node_dim: int = 256, n_layers: int = N_LAYERS,
                     n_heads: int = N_HEADS, hidden_dim: int = HIDDEN_DIM,
                     deg_embed_dim: int = DEG_EMBED_DIM, max_degree: int = 50):
            super().__init__()
            self.deg_embed = DegreeEmbedding(max_degree, deg_embed_dim)
            self.input_dim = base_node_dim + deg_embed_dim + 1  # +1 for time

            # Simple transformer-based edge predictor
            self.node_proj = nn.Linear(self.input_dim, hidden_dim)
            self.edge_input_proj = nn.Linear(1, hidden_dim)

            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=n_heads,
                    dim_feedforward=hidden_dim * 4,
                    batch_first=True, dropout=0.1,
                )
                for _ in range(n_layers)
            ])

            # Edge prediction head: pairs of node features -> edge logits
            self.edge_head = nn.Sequential(
                nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),  # 2 classes: no-edge, edge
            )

        def forward(self, node_features: Tensor, x_t: Tensor, t: Tensor,
                    in_deg: Tensor, out_deg: Tensor) -> Tensor:
            """
            Args:
                node_features: [B, N, base_node_dim]
                x_t:           [B, N, N] current noised adjacency
                t:             [B] time in [0, 1]
                in_deg:        [B, N] conditioning in-degrees
                out_deg:       [B, N] conditioning out-degrees
            Returns:
                edge_logits: [B, N, N, 2]
            """
            B, N, _ = node_features.shape

            # Degree conditioning
            deg_h = self.deg_embed(in_deg, out_deg)  # [B, N, deg_embed_dim]

            # Time embedding (broadcast to per-node)
            t_emb = t[:, None, None].expand(B, N, 1)  # [B, N, 1]

            # Concatenate node features + degree embedding + time
            h = torch.cat([node_features, deg_h, t_emb], dim=-1)  # [B, N, input_dim]
            h = self.node_proj(h)  # [B, N, hidden_dim]

            # Transformer layers
            for layer in self.layers:
                h = layer(h)  # [B, N, hidden_dim]

            # Edge prediction: pairwise node features + edge state
            h_i = h.unsqueeze(2).expand(B, N, N, -1)  # [B, N, N, hidden_dim]
            h_j = h.unsqueeze(1).expand(B, N, N, -1)  # [B, N, N, hidden_dim]
            e_state = self.edge_input_proj(
                x_t.unsqueeze(-1)
            )  # [B, N, N, hidden_dim]
            e_feat = torch.cat([h_i, h_j, e_state], dim=-1)
            logits = self.edge_head(e_feat)  # [B, N, N, 2]
            return logits

    # ---------------------------------------------------------------------------
    # Auxiliary degree loss
    # ---------------------------------------------------------------------------

    def degree_auxiliary_loss(edge_logits: Tensor, target_in_deg: Tensor,
                              target_out_deg: Tensor) -> Tensor:
        """Penalize divergence between predicted adjacency's degree sums and
        conditioning degree sequence.

        Args:
            edge_logits: [B, N, N, 2]
            target_in_deg:  [B, N] float
            target_out_deg: [B, N] float
        Returns:
            scalar loss
        """
        # Predicted edge probabilities
        p_edge = torch.softmax(edge_logits, dim=-1)[..., 1]  # [B, N, N]
        # Mask diagonal
        N = p_edge.shape[1]
        diag = torch.eye(N, device=p_edge.device, dtype=torch.bool)
        p_edge = p_edge.masked_fill(diag.unsqueeze(0), 0.0)

        # Predicted degrees
        pred_out = p_edge.sum(dim=2)  # [B, N] (row sums)
        pred_in = p_edge.sum(dim=1)   # [B, N] (col sums)

        # L1 loss on degree discrepancy
        loss_out = F.l1_loss(pred_out, target_out_deg.float())
        loss_in = F.l1_loss(pred_in, target_in_deg.float())
        return (loss_out + loss_in) / 2

    # ---------------------------------------------------------------------------
    # Mutual information estimator
    # ---------------------------------------------------------------------------

    def estimate_mi_edge_score_output(edge_probs: np.ndarray,
                                       adj_output: np.ndarray) -> float:
        """Estimate MI(learned_score, final_adjacency) via binning.

        Args:
            edge_probs: [N, N] float in [0,1] -- learned edge probabilities
            adj_output: [N, N] binary -- final adjacency
        Returns:
            float: estimated MI in bits
        """
        np.fill_diagonal(edge_probs, 0)
        np.fill_diagonal(adj_output, 0)
        N = edge_probs.shape[0]
        # Flatten off-diagonal entries
        mask = ~np.eye(N, dtype=bool)
        scores = edge_probs[mask].flatten()
        labels = adj_output[mask].flatten().astype(int)

        # Bin scores into 10 bins
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        score_bins = np.digitize(scores, bins) - 1
        score_bins = np.clip(score_bins, 0, n_bins - 1)

        # Joint distribution
        joint = np.zeros((n_bins, 2), dtype=np.float64)
        for b, l in zip(score_bins, labels):
            joint[b, l] += 1
        joint = joint / joint.sum()

        # Marginals
        p_score = joint.sum(axis=1)
        p_label = joint.sum(axis=0)

        # MI
        mi = 0.0
        for b in range(n_bins):
            for l in range(2):
                if joint[b, l] > 0 and p_score[b] > 0 and p_label[l] > 0:
                    mi += joint[b, l] * np.log2(
                        joint[b, l] / (p_score[b] * p_label[l])
                    )
        return float(mi)


# ---------------------------------------------------------------------------
# Flow matching utilities (self-contained)
# ---------------------------------------------------------------------------

def sample_source(shape, density=0.5, rng=None):
    """Sample source x_0 ~ Bernoulli(density)."""
    if rng is None:
        return (np.random.random(shape) < density).astype(np.float32)
    return (rng.random(shape) < density).astype(np.float32)


def interpolate_np(x_source, x_target, t):
    """Marginal interpolation at time t (numpy version)."""
    mask = np.random.random(x_source.shape) < t
    return np.where(mask, x_target, x_source)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_degree_conditioned(
    train_items: list,
    n_morph_classes: int = 12,
    steps: int = TRAIN_STEPS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    seed: int = SEED,
    device_str: str = "auto",
) -> dict:
    """Train the degree-conditioned DiGress variant.

    Args:
        train_items: list of dicts with 'adjacency', 'morph_idx', etc.
        n_morph_classes: number of morphological classes
        steps: training steps
        batch_size: batch size
        lr: learning rate
        seed: random seed
        device_str: 'auto', 'cpu', or 'cuda'

    Returns:
        dict with training metrics and model state
    """
    if not HAS_TORCH:
        return {"error": "PyTorch not available"}

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Determine base node feature dim
    base_dim = n_morph_classes + 4  # morph_oh + xyz + depth
    N = train_items[0]["adjacency"].shape[0]

    model = DegreeConditionedDigress(
        base_node_dim=base_dim, n_layers=N_LAYERS,
        n_heads=N_HEADS, hidden_dim=HIDDEN_DIM,
        deg_embed_dim=DEG_EMBED_DIM,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    # Build feature matrices
    def make_node_features(item):
        N_ = item["adjacency"].shape[0]
        morph_idx = item["morph_idx"].astype(int)
        morph_oh = np.zeros((N_, n_morph_classes), dtype=np.float32)
        for i in range(N_):
            if 0 <= morph_idx[i] < n_morph_classes:
                morph_oh[i, morph_idx[i]] = 1.0
        pos = item.get("positions", np.zeros((N_, 3), dtype=np.float32))
        depth = item.get("depth", np.zeros(N_, dtype=np.float32))
        return np.hstack([morph_oh, pos, depth[:, None]]).astype(np.float32)

    losses = []
    ce_losses = []
    deg_losses = []
    model.train()

    print(f"[DegCond] Training degree-conditioned DiGress for {steps} steps "
          f"on {device} ...", flush=True)
    t_start = time.time()

    for step in range(steps):
        # Sample batch
        batch_idx = rng.integers(0, len(train_items), size=batch_size)
        batch_adj = []
        batch_feat = []
        batch_in_deg = []
        batch_out_deg = []

        for idx in batch_idx:
            item = train_items[idx]
            adj = item["adjacency"].copy()
            np.fill_diagonal(adj, 0)
            batch_adj.append(adj)
            batch_feat.append(make_node_features(item))
            batch_out_deg.append(adj.sum(axis=1).astype(int))
            batch_in_deg.append(adj.sum(axis=0).astype(int))

        x_target = torch.tensor(np.stack(batch_adj), dtype=torch.float32,
                                device=device)
        node_feat = torch.tensor(np.stack(batch_feat), dtype=torch.float32,
                                 device=device)
        in_deg = torch.tensor(np.stack(batch_in_deg), dtype=torch.long,
                              device=device)
        out_deg = torch.tensor(np.stack(batch_out_deg), dtype=torch.long,
                               device=device)

        B_curr, N_curr = x_target.shape[:2]

        # Sample time
        t = torch.rand(B_curr, device=device)

        # Source density ~ data density for SNR
        data_density = float(x_target.mean())
        x_0 = torch.bernoulli(
            torch.full_like(x_target, max(data_density, 0.01))
        )

        # Interpolate
        t_edge = t[:, None, None]
        mask = torch.bernoulli(t_edge.expand_as(x_0)).bool()
        x_t = torch.where(mask, x_target, x_0)

        # Forward
        logits = model(node_feat, x_t, t, in_deg, out_deg)

        # CE loss (standard flow matching objective)
        diag_mask = (1 - torch.eye(N_curr, device=device)).unsqueeze(0).expand(
            B_curr, -1, -1
        )
        logits_flat = logits.reshape(-1, 2)
        targets_flat = x_target.reshape(-1).long()
        mask_flat = diag_mask.reshape(-1).bool()
        ce_loss = F.cross_entropy(
            logits_flat[mask_flat], targets_flat[mask_flat]
        )

        # Auxiliary degree loss (annealed)
        anneal_factor = min(1.0, step / max(LAMBDA_DEG_ANNEAL, 1))
        deg_loss = degree_auxiliary_loss(logits, in_deg, out_deg)
        total_loss = ce_loss + LAMBDA_DEG * anneal_factor * deg_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(float(total_loss))
        ce_losses.append(float(ce_loss))
        deg_losses.append(float(deg_loss))

        if (step + 1) % 1000 == 0:
            avg_loss = np.mean(losses[-1000:])
            avg_ce = np.mean(ce_losses[-1000:])
            avg_deg = np.mean(deg_losses[-1000:])
            elapsed = time.time() - t_start
            print(f"  step {step+1}/{steps}  loss={avg_loss:.4f} "
                  f"(CE={avg_ce:.4f} deg={avg_deg:.4f})  "
                  f"elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t_start
    print(f"[DegCond] Training complete. {elapsed:.0f}s", flush=True)

    return {
        "model": model,
        "final_loss": float(np.mean(losses[-100:])),
        "final_ce_loss": float(np.mean(ce_losses[-100:])),
        "final_deg_loss": float(np.mean(deg_losses[-100:])),
        "walltime_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Sampling from degree-conditioned model
# ---------------------------------------------------------------------------

def sample_degree_conditioned(
    model, node_features: np.ndarray,
    in_deg: np.ndarray, out_deg: np.ndarray,
    num_steps: int = 16, source_density: float = 0.02,
    device=None,
) -> np.ndarray:
    """Generate adjacency from degree-conditioned model.

    Args:
        model: trained DegreeConditionedDigress
        node_features: [N, feat_dim]
        in_deg: [N] conditioning in-degrees
        out_deg: [N] conditioning out-degrees
        num_steps: Euler steps
        source_density: source edge density

    Returns:
        [N, N] binary adjacency
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for sampling")

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    N = node_features.shape[0]

    nf = torch.tensor(node_features, dtype=torch.float32,
                       device=device).unsqueeze(0)  # [1, N, feat_dim]
    id_ = torch.tensor(in_deg, dtype=torch.long,
                        device=device).unsqueeze(0)   # [1, N]
    od_ = torch.tensor(out_deg, dtype=torch.long,
                        device=device).unsqueeze(0)   # [1, N]

    x_t = torch.bernoulli(
        torch.full((1, N, N), source_density, device=device)
    )
    dt = 1.0 / num_steps

    with torch.no_grad():
        for step in range(num_steps):
            t_val = step / num_steps
            t = torch.full((1,), t_val, device=device)
            logits = model(nf, x_t, t, id_, od_)
            p1 = torch.softmax(logits, dim=-1)[..., 1]
            x1_sample = torch.bernoulli(p1)
            update_mask = torch.bernoulli(
                torch.full_like(p1, dt)
            ).bool()
            x_t = torch.where(update_mask, x1_sample, x_t)

    adj = x_t[0].cpu().numpy()
    np.fill_diagonal(adj, 0)
    return (adj > 0.5).astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def reach_3hop(adj):
    A = (adj > 0.5).astype(np.float32)
    np.fill_diagonal(A, 0)
    N = A.shape[0]
    R = A.copy()
    for _ in range(2):
        R = ((R @ A) > 0).astype(np.float32)
        R = np.clip(R + A, 0, 1)
    np.fill_diagonal(R, 0)
    return float(R.sum() / max(N * (N - 1), 1))


def motif_l1(adj_gt, adj_s):
    m_gt = motif_3node_spectrum(adj_gt)
    m_s = motif_3node_spectrum(adj_s)
    s_gt = m_gt.sum(); s_s = m_s.sum()
    if s_gt > 0: m_gt = m_gt / s_gt
    if s_s > 0: m_s = m_s / s_s
    return float(np.abs(m_gt - m_s).sum())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_degree_conditioned(
    model, test_items: list, n_morph_classes: int = 12,
    n_samples: int = 5, device=None,
) -> dict:
    """Evaluate degree-conditioned model on test items.

    Compares four configurations:
    (a) Standard DiGress + Sinkhorn(GT) -- existing pipeline
    (b) Degree-conditioned DiGress + no Sinkhorn
    (c) Degree-conditioned DiGress + Sinkhorn
    (d) Configuration model (GT degrees)
    """
    results = {
        "deg_cond_no_sinkhorn": {"reach_l1": [], "motif_l1": [], "recip_err": [],
                                 "mi": []},
        "deg_cond_with_sinkhorn": {"reach_l1": [], "motif_l1": [], "recip_err": []},
        "config_model_gt": {"reach_l1": [], "motif_l1": [], "recip_err": []},
    }
    rng = np.random.default_rng(SEED + 100)

    def make_node_features(item):
        N_ = item["adjacency"].shape[0]
        morph_idx = item["morph_idx"].astype(int)
        morph_oh = np.zeros((N_, n_morph_classes), dtype=np.float32)
        for i in range(N_):
            if 0 <= morph_idx[i] < n_morph_classes:
                morph_oh[i, morph_idx[i]] = 1.0
        pos = item.get("positions", np.zeros((N_, 3), dtype=np.float32))
        depth = item.get("depth", np.zeros(N_, dtype=np.float32))
        return np.hstack([morph_oh, pos, depth[:, None]]).astype(np.float32)

    for it in test_items:
        adj_gt = it["adjacency"]
        N = adj_gt.shape[0]
        nf = make_node_features(it)
        in_deg = adj_gt.sum(axis=0).astype(int)
        out_deg = adj_gt.sum(axis=1).astype(int)
        data_density = float(adj_gt.mean())

        for _ in range(n_samples):
            # (b) Degree-conditioned, no Sinkhorn
            adj_dc = sample_degree_conditioned(
                model, nf, in_deg, out_deg,
                num_steps=16, source_density=data_density, device=device,
            )
            results["deg_cond_no_sinkhorn"]["reach_l1"].append(
                abs(reach_3hop(adj_gt) - reach_3hop(adj_dc))
            )
            results["deg_cond_no_sinkhorn"]["motif_l1"].append(
                motif_l1(adj_gt, adj_dc)
            )
            results["deg_cond_no_sinkhorn"]["recip_err"].append(
                abs(reciprocity(adj_gt) - reciprocity(adj_dc))
            )
            # MI between learned score and output
            with torch.no_grad():
                nf_t = torch.tensor(nf, dtype=torch.float32,
                                     device=device).unsqueeze(0)
                id_t = torch.tensor(in_deg, dtype=torch.long,
                                     device=device).unsqueeze(0)
                od_t = torch.tensor(out_deg, dtype=torch.long,
                                     device=device).unsqueeze(0)
                x_t = torch.tensor(adj_gt, dtype=torch.float32,
                                    device=device).unsqueeze(0)
                t = torch.tensor([0.9], device=device)
                logits = model(nf_t, x_t, t, id_t, od_t)
                probs = torch.softmax(logits, dim=-1)[0, :, :, 1].cpu().numpy()
            mi = estimate_mi_edge_score_output(probs, adj_dc)
            results["deg_cond_no_sinkhorn"]["mi"].append(mi)

            # (c) Degree-conditioned + Sinkhorn
            P_dc = np.clip(probs, 0.001, 0.999)
            P_sink = sinkhorn_degree_match(P_dc.copy(), out_deg, in_deg,
                                           num_iters=30)
            adj_dc_sink = (rng.random(P_sink.shape) < P_sink).astype(np.float32)
            np.fill_diagonal(adj_dc_sink, 0)
            results["deg_cond_with_sinkhorn"]["reach_l1"].append(
                abs(reach_3hop(adj_gt) - reach_3hop(adj_dc_sink))
            )
            results["deg_cond_with_sinkhorn"]["motif_l1"].append(
                motif_l1(adj_gt, adj_dc_sink)
            )
            results["deg_cond_with_sinkhorn"]["recip_err"].append(
                abs(reciprocity(adj_gt) - reciprocity(adj_dc_sink))
            )

            # (d) Config model
            adj_cm = configuration_model_sample(adj_gt, rng)
            results["config_model_gt"]["reach_l1"].append(
                abs(reach_3hop(adj_gt) - reach_3hop(adj_cm))
            )
            results["config_model_gt"]["motif_l1"].append(
                motif_l1(adj_gt, adj_cm)
            )
            results["config_model_gt"]["recip_err"].append(
                abs(reciprocity(adj_gt) - reciprocity(adj_cm))
            )

    # Aggregate
    summary = {}
    for config_name, metrics in results.items():
        summary[config_name] = {}
        for metric_name, vals in metrics.items():
            arr = np.array(vals)
            summary[config_name][metric_name] = {
                "mean": float(arr.mean()),
                "sem": float(arr.std() / np.sqrt(max(len(arr), 1))),
                "n": len(arr),
            }
    return summary


# ---------------------------------------------------------------------------
# Data loading (self-contained fallback)
# ---------------------------------------------------------------------------

def load_microns_subsets(seed=0):
    try:
        from exp_powerlaw_vs_ridge import load_microns_subsets as _load
        return _load(seed=seed)
    except (ImportError, FileNotFoundError, ModuleNotFoundError):
        print("[DegCond] MICrONS loader unavailable. Using synthetic data.",
              flush=True)
        rng = np.random.default_rng(seed)
        items = []
        for _ in range(200):
            N = 100
            adj = (rng.random((N, N)) < 0.022).astype(np.float32)
            np.fill_diagonal(adj, 0)
            items.append({
                "adjacency": adj,
                "morph_idx": rng.integers(0, 12, size=N).astype(np.int64),
                "positions": rng.normal(0, 200, size=(N, 3)).astype(np.float32),
                "depth": rng.uniform(0, 800, size=N).astype(np.float32),
            })
        return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print("Loading MICrONS subsets ...", flush=True)
    all_items = load_microns_subsets(seed=0)
    train_items = all_items[:150]
    test_items = all_items[150:200]
    n_morph = max(int(it["morph_idx"].max()) + 1
                  for it in all_items) if all_items else 12

    # Train
    train_result = train_degree_conditioned(
        train_items, n_morph_classes=n_morph,
        steps=TRAIN_STEPS, batch_size=BATCH_SIZE, lr=LR, seed=SEED,
    )

    if "error" in train_result:
        print(f"Training failed: {train_result['error']}")
        return train_result

    model = train_result["model"]
    device = next(model.parameters()).device

    # Evaluate
    print("\nEvaluating on test subsets ...", flush=True)
    eval_results = evaluate_degree_conditioned(
        model, test_items[:20], n_morph_classes=n_morph,
        n_samples=5, device=device,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("DEGREE-CONDITIONED DIGRESS RESULTS")
    print("=" * 60)
    for config, metrics in eval_results.items():
        print(f"\n  {config}:")
        for metric, vals in metrics.items():
            print(f"    {metric}: {vals['mean']:.4f} +/- {vals['sem']:.4f}")

    output = {
        "experiment": "degree_conditioned_aaai27",
        "addresses": "AAAI-27 Exp 3: Degree-conditioned generation",
        "training": {
            "steps": TRAIN_STEPS,
            "final_loss": train_result["final_loss"],
            "final_ce_loss": train_result["final_ce_loss"],
            "final_deg_loss": train_result["final_deg_loss"],
            "walltime_s": train_result["walltime_s"],
        },
        "evaluation": eval_results,
        "config": {
            "lambda_deg": LAMBDA_DEG,
            "lambda_deg_anneal_steps": LAMBDA_DEG_ANNEAL,
            "deg_embed_dim": DEG_EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
        },
        "walltime_total_s": float(time.time() - t0),
    }

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "aaai27")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "degree_conditioned.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}")
    return output


if __name__ == "__main__":
    main()
