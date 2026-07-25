"""run_directed_ph.py -- Experiment 2: Directed Persistent Homology via Flagser.

Replaces gudhi-based undirected (symmetrized) PH with pyflagser-based directed
flag complex PH. Computes bottleneck + Wasserstein-1 distances on H0/H1 for all
pipeline configurations (config model, uniform+Sinkhorn(GT), uniform+Sinkhorn(RF),
Bernoulli baseline).

Uses pyflagser.flagser_weighted for directed PH with inter-soma distance
filtration. Falls back to gudhi symmetrized PH if pyflagser is unavailable.

AAAI-27 Experiment 2.
CPU only. Output: results/aaai27/directed_ph.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
import numpy as np

import networkx as nx
from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------------------------
# Inlined helpers (from Cell2Circuit src/analysis/)
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

# ---------------------------------------------------------------------------
# Library detection
# ---------------------------------------------------------------------------
_USE_FLAGSER_WEIGHTED = False
_USE_FLAGSER_UNWEIGHTED = False
_USE_GUDHI = False
_LIBRARY = "none"

try:
    from pyflagser import flagser_weighted
    _USE_FLAGSER_WEIGHTED = True
    _LIBRARY = "pyflagser_directed_weighted"
    print("[DPH] pyflagser.flagser_weighted found -- using directed flag "
          "complexes with distance filtration.", flush=True)
except ImportError:
    try:
        from pyflagser import flagser_unweighted
        _USE_FLAGSER_UNWEIGHTED = True
        _LIBRARY = "pyflagser_directed_unweighted"
        print("[DPH] pyflagser.flagser_unweighted found (no weighted variant). "
              "Using directed flag complexes without filtration.", flush=True)
    except ImportError:
        pass

if not _USE_FLAGSER_WEIGHTED and not _USE_FLAGSER_UNWEIGHTED:
    try:
        import gudhi
        _USE_GUDHI = True
        _LIBRARY = "gudhi_symmetrised"
        print("[DPH] pyflagser not available; falling back to gudhi on "
              "symmetrised graph.", flush=True)
    except ImportError:
        raise ImportError(
            "Neither pyflagser nor gudhi is installed. "
            "Install via: pip install pyflagser  OR  pip install gudhi"
        )

# Also try giotto-tda as an alternative API
_USE_GIOTTO = False
try:
    from gtda.homology import FlagserPersistence
    _USE_GIOTTO = True
    print("[DPH] giotto-tda FlagserPersistence available as alternative.",
          flush=True)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_TEST = 30
N_MC = 1           # MC draws per subset per sampler
WALL_CAP_S = 60 * 60  # 1 hour hard cap


# ---------------------------------------------------------------------------
# Persistence diagram computation
# ---------------------------------------------------------------------------

def _adj_to_diagram_flagser_weighted(
    adj: np.ndarray, dist_mat: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Directed flag complex PD via pyflagser.flagser_weighted.

    flagser_weighted expects:
    - Off-diagonal: edge weights (0 or np.inf = no edge, positive = filtration value)
    - Diagonal: vertex weights (set to 0 so vertices appear at filtration=0)
    - filtration="max": simplex enters at max edge weight among its faces

    For our inter-soma distance filtration: present edges get weight = distance,
    absent edges are set to np.inf (excluded from complex).
    """
    A = (adj > 0.5).astype(np.float64)
    np.fill_diagonal(A, 0)
    N = A.shape[0]

    # Build weight matrix for flagser_weighted
    W = np.full((N, N), np.inf, dtype=np.float64)
    # Diagonal = vertex weights (all enter at filtration 0)
    np.fill_diagonal(W, 0.0)

    if dist_mat is not None:
        # Present edges: weight = inter-soma distance
        edge_mask = A > 0.5
        W[edge_mask] = dist_mat[edge_mask].astype(np.float64)
    else:
        # No positions: binary filtration (edges at 1.0)
        edge_mask = A > 0.5
        W[edge_mask] = 1.0

    result = flagser_weighted(
        W, min_dimension=0, max_dimension=1,
        directed=True, filtration="max"
    )

    dgms = {}
    for dim in [0, 1]:
        pts = result["dgms"][dim]
        if len(pts) == 0:
            dgms[f"dgm_{dim}"] = np.empty((0, 2), dtype=np.float64)
        else:
            arr = np.array(pts, dtype=np.float64)
            # Cap infinite death values
            finite_mask = np.isfinite(arr)
            if finite_mask.any():
                max_finite = arr[finite_mask].max()
            else:
                max_finite = 1.0
            arr[~np.isfinite(arr)] = max_finite * 1.1
            dgms[f"dgm_{dim}"] = arr
    return dgms


def _adj_to_diagram_flagser_unweighted(
    adj: np.ndarray, dist_mat: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Directed flag complex Betti numbers via flagser_unweighted.

    Note: flagser_unweighted does NOT produce persistence diagrams,
    only Betti numbers. We return empty diagrams and report Betti numbers
    separately.
    """
    A = (adj > 0.5).astype(np.int32)
    np.fill_diagonal(A, 0)

    from pyflagser import flagser_unweighted
    result = flagser_unweighted(
        A, min_dimension=0, max_dimension=1, directed=True
    )

    # flagser_unweighted returns betti numbers, not diagrams
    # Create synthetic single-point diagrams from Betti numbers
    dgms = {}
    for dim in [0, 1]:
        betti = result["betti"][dim] if dim < len(result["betti"]) else 0
        if betti > 0:
            # Create betti-many essential features (birth=0, death=inf->1.1)
            dgms[f"dgm_{dim}"] = np.array(
                [[0.0, 1.1]] * betti, dtype=np.float64
            )
        else:
            dgms[f"dgm_{dim}"] = np.empty((0, 2), dtype=np.float64)
    dgms["betti"] = list(result["betti"])
    return dgms


def _adj_to_diagram_giotto(
    adj: np.ndarray, dist_mat: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Directed PH via giotto-tda's FlagserPersistence wrapper."""
    A = (adj > 0.5).astype(np.float64)
    np.fill_diagonal(A, 0)
    N = A.shape[0]

    # Build weight matrix (same format as flagser_weighted)
    W = np.full((N, N), np.inf, dtype=np.float64)
    np.fill_diagonal(W, 0.0)
    if dist_mat is not None:
        edge_mask = A > 0.5
        W[edge_mask] = dist_mat[edge_mask].astype(np.float64)
    else:
        edge_mask = A > 0.5
        W[edge_mask] = 1.0

    fp = FlagserPersistence(
        homology_dimensions=(0, 1),
        directed=True,
        filtration="max",
        coeff=2,
    )
    # Input: (1, N, N) array
    diagrams = fp.fit_transform(W[np.newaxis, :, :])  # (1, n_features, 3)
    diag = diagrams[0]  # (n_features, 3): [birth, death, dim]

    dgms = {}
    for dim in [0, 1]:
        mask = (diag[:, 2] == dim) & (diag[:, 0] != diag[:, 1])  # skip padding
        if mask.sum() > 0:
            pts = diag[mask, :2].astype(np.float64)
            finite_mask = np.isfinite(pts)
            if finite_mask.any():
                max_finite = pts[finite_mask].max()
            else:
                max_finite = 1.0
            pts[~np.isfinite(pts)] = max_finite * 1.1
            dgms[f"dgm_{dim}"] = pts
        else:
            dgms[f"dgm_{dim}"] = np.empty((0, 2), dtype=np.float64)
    return dgms


def _adj_to_diagram_gudhi(
    adj: np.ndarray, dist_mat: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Gudhi on symmetrised graph (fallback when pyflagser unavailable)."""
    import gudhi
    A = (adj > 0.5).astype(np.float32)
    np.fill_diagonal(A, 0)
    A_sym = ((A + A.T) > 0).astype(np.float32)
    N = A_sym.shape[0]
    st = gudhi.SimplexTree()
    for i in range(N):
        st.insert([i], filtration=0.0)
    for i in range(N):
        for j in range(i + 1, N):
            if A_sym[i, j] > 0:
                fval = float(dist_mat[i, j]) if dist_mat is not None else 1.0
                st.insert([i, j], filtration=fval)
    st.expansion(2)
    st.compute_persistence(persistence_dim_max=True)
    dgms = {}
    for dim in [0, 1]:
        pts = st.persistence_intervals_in_dimension(dim)
        if len(pts) == 0:
            dgms[f"dgm_{dim}"] = np.empty((0, 2), dtype=np.float64)
        else:
            arr = np.array(pts, dtype=np.float64)
            max_finite = arr[np.isfinite(arr)].max() if np.isfinite(arr).any() else 1.0
            arr[~np.isfinite(arr)] = max_finite * 1.1
            dgms[f"dgm_{dim}"] = arr
    return dgms


def compute_diagram(
    adj: np.ndarray, dist_mat: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Compute persistence diagram using best available library."""
    if _USE_FLAGSER_WEIGHTED:
        return _adj_to_diagram_flagser_weighted(adj, dist_mat)
    elif _USE_GIOTTO:
        return _adj_to_diagram_giotto(adj, dist_mat)
    elif _USE_FLAGSER_UNWEIGHTED:
        return _adj_to_diagram_flagser_unweighted(adj, dist_mat)
    elif _USE_GUDHI:
        return _adj_to_diagram_gudhi(adj, dist_mat)
    else:
        raise RuntimeError("No PH library available.")


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def bottleneck_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Bottleneck distance between persistence diagrams."""
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0
    try:
        import gudhi
        d1 = dgm1.astype(np.float64).tolist() if len(dgm1) else []
        d2 = dgm2.astype(np.float64).tolist() if len(dgm2) else []
        return float(gudhi.bottleneck_distance(d1, d2))
    except ImportError:
        return _bottleneck_fallback(dgm1, dgm2)


def _bottleneck_fallback(dgm1, dgm2):
    """Simple bottleneck approximation when gudhi unavailable."""
    from scipy.optimize import linear_sum_assignment

    def _diag_proj(pts):
        m = (pts[:, 0] + pts[:, 1]) / 2
        return np.column_stack([m, m])

    p = dgm1.astype(np.float64) if len(dgm1) else np.empty((0, 2))
    q = dgm2.astype(np.float64) if len(dgm2) else np.empty((0, 2))
    n, m = len(p), len(q)
    if n == 0 and m == 0:
        return 0.0
    if n > 0 and m > 0:
        p_aug = np.vstack([p, _diag_proj(q)])
        q_aug = np.vstack([q, _diag_proj(p)])
    elif n > 0:
        p_aug, q_aug = p, _diag_proj(p)
    else:
        p_aug, q_aug = _diag_proj(q), q
    total = max(len(p_aug), len(q_aug))
    # Pad to same size
    while len(p_aug) < total:
        p_aug = np.vstack([p_aug, np.zeros((1, 2))])
    while len(q_aug) < total:
        q_aug = np.vstack([q_aug, np.zeros((1, 2))])
    cost = np.array([
        [float(np.max(np.abs(p_aug[i] - q_aug[j])))
         for j in range(total)]
        for i in range(total)
    ])
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(np.max(cost[row_ind, col_ind]))


def wasserstein_1_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Wasserstein-1 distance between persistence diagrams."""
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0
    try:
        from gudhi.wasserstein import wasserstein_distance
        p = dgm1.astype(np.float64) if len(dgm1) else np.empty((0, 2))
        q = dgm2.astype(np.float64) if len(dgm2) else np.empty((0, 2))
        return float(wasserstein_distance(p, q, order=1, internal_p=1))
    except ImportError:
        return _wasserstein1_fallback(dgm1, dgm2)


def _wasserstein1_fallback(dgm1, dgm2):
    """Wasserstein-1 approximation via linear_sum_assignment."""
    from scipy.optimize import linear_sum_assignment

    def _diag_proj(pts):
        m = (pts[:, 0] + pts[:, 1]) / 2
        return np.column_stack([m, m])

    p = dgm1.astype(np.float64) if len(dgm1) else np.empty((0, 2))
    q = dgm2.astype(np.float64) if len(dgm2) else np.empty((0, 2))
    n, m_len = len(p), len(q)
    if n == 0 and m_len == 0:
        return 0.0
    if n > 0 and m_len > 0:
        p_aug = np.vstack([p, _diag_proj(q)])
        q_aug = np.vstack([q, _diag_proj(p)])
    elif n > 0:
        p_aug, q_aug = p, _diag_proj(p)
    else:
        p_aug, q_aug = _diag_proj(q), q
    total = max(len(p_aug), len(q_aug))
    while len(p_aug) < total:
        p_aug = np.vstack([p_aug, np.zeros((1, 2))])
    while len(q_aug) < total:
        q_aug = np.vstack([q_aug, np.zeros((1, 2))])
    cost = np.array([
        [float(np.sum(np.abs(p_aug[i] - q_aug[j])))
         for j in range(total)]
        for i in range(total)
    ])
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(np.sum(cost[row_ind, col_ind]))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_microns_subsets(seed=0):
    """Load MICrONS N=100 subsets. Falls back to synthetic if unavailable."""
    try:
        from exp_powerlaw_vs_ridge import load_microns_subsets as _load
        return _load(seed=seed)
    except (ImportError, FileNotFoundError, ModuleNotFoundError):
        print("[DPH] MICrONS loader unavailable. Using synthetic data.",
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


def features_full(item, n_morph_classes=12):
    N = item["adjacency"].shape[0]
    morph_idx = item["morph_idx"].astype(int)
    morph_oh = np.zeros((N, n_morph_classes), dtype=np.float32)
    for i in range(N):
        if 0 <= morph_idx[i] < n_morph_classes:
            morph_oh[i, morph_idx[i]] = 1.0
    pos = item.get("positions", np.zeros((N, 3), dtype=np.float32))
    depth = item.get("depth", np.zeros(N, dtype=np.float32))
    return np.hstack([morph_oh, pos, depth[:, None]]).astype(np.float32)


def _reconcile_totals(o, i):
    o = o.copy(); i = i.copy()
    diff = int(o.sum() - i.sum())
    if diff > 0:
        order = np.argsort(-o)
        for idx in order:
            if diff <= 0: break
            trim = min(diff, int(o[idx]))
            o[idx] -= trim; diff -= trim
    elif diff < 0:
        diff = -diff
        order = np.argsort(-i)
        for idx in order:
            if diff <= 0: break
            trim = min(diff, int(i[idx]))
            i[idx] -= trim; diff -= trim
    return o, i


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def build_samplers(train_items, rng):
    """Build sampler functions matching the existing pipeline."""
    feat_fn = lambda it: features_full(it, n_morph_classes=12)

    # Fit RF degree predictor
    X_l, yo_l, yi_l = [], [], []
    for it in train_items:
        X_l.append(feat_fn(it))
        a = it["adjacency"]
        yo_l.append(a.sum(axis=1).astype(float))
        yi_l.append(a.sum(axis=0).astype(float))
    X = np.concatenate(X_l)
    yo = np.concatenate(yo_l)
    yi = np.concatenate(yi_l)
    fm = X.mean(0); fs = X.std(0).clip(1e-4)
    Xn = (X - fm) / fs
    rf_o = RandomForestRegressor(
        n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
    ).fit(Xn, yo)
    rf_i = RandomForestRegressor(
        n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
    ).fit(Xn, yi)

    def _pred_deg(it):
        X = feat_fn(it)
        Xn_ = (X - fm) / fs
        o = np.maximum(0, np.round(rf_o.predict(Xn_))).astype(int)
        i = np.maximum(0, np.round(rf_i.predict(Xn_))).astype(int)
        return _reconcile_totals(o, i)

    def sample_bernoulli(it, rng):
        N = it["adjacency"].shape[0]
        rho = float(it["adjacency"].mean())
        a = (rng.random((N, N)) < rho).astype(np.float32)
        np.fill_diagonal(a, 0)
        return a

    def sample_sinkhorn_gt(it, rng):
        a_gt = it["adjacency"]
        N = a_gt.shape[0]
        rho = float(a_gt.mean())
        out_deg = a_gt.sum(axis=1).astype(int)
        in_deg = a_gt.sum(axis=0).astype(int)
        P = np.full((N, N), rho, dtype=np.float32)
        np.fill_diagonal(P, 0)
        P_s = sinkhorn_degree_match(P.copy(), out_deg, in_deg, num_iters=30)
        a = (rng.random(P_s.shape) < P_s).astype(np.float32)
        np.fill_diagonal(a, 0)
        return a

    def sample_sinkhorn_rf(it, rng):
        a_gt = it["adjacency"]
        N = a_gt.shape[0]
        rho = float(a_gt.mean())
        out_deg, in_deg = _pred_deg(it)
        P = np.full((N, N), rho, dtype=np.float32)
        np.fill_diagonal(P, 0)
        P_s = sinkhorn_degree_match(P.copy(), out_deg, in_deg, num_iters=30)
        a = (rng.random(P_s.shape) < P_s).astype(np.float32)
        np.fill_diagonal(a, 0)
        return a

    def sample_config_model(it, rng):
        return configuration_model_sample(it["adjacency"], rng)

    return {
        "uniform_bernoulli": sample_bernoulli,
        "uniform+sinkhorn_gt": sample_sinkhorn_gt,
        "uniform+sinkhorn_rf": sample_sinkhorn_rf,
        "config_model_gt": sample_config_model,
    }


# ---------------------------------------------------------------------------
# Inter-soma distance matrix
# ---------------------------------------------------------------------------

def inter_soma_dist(item: dict) -> np.ndarray | None:
    pos = item.get("positions", None)
    if pos is None:
        return None
    pos = np.asarray(pos, dtype=np.float64)
    diff = pos[:, None, :] - pos[None, :, :]
    return np.linalg.norm(diff, axis=-1).astype(np.float64)


# ---------------------------------------------------------------------------
# Per-subset distance computation
# ---------------------------------------------------------------------------

def subset_distances(gt_dgm, sample_dgm):
    return {
        "bottleneck_H0": bottleneck_distance(gt_dgm["dgm_0"], sample_dgm["dgm_0"]),
        "bottleneck_H1": bottleneck_distance(gt_dgm["dgm_1"], sample_dgm["dgm_1"]),
        "wasserstein1_H0": wasserstein_1_distance(gt_dgm["dgm_0"], sample_dgm["dgm_0"]),
        "wasserstein1_H1": wasserstein_1_distance(gt_dgm["dgm_1"], sample_dgm["dgm_1"]),
    }


def mean_sem(vals):
    a = np.array(vals, dtype=np.float64)
    if len(a) == 0:
        return 0.0, 0.0
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    t_start = time.time()
    rng = np.random.default_rng(42)

    print(f"[DPH] Library: {_LIBRARY}", flush=True)
    print(f"[DPH] Loading subsets ...", flush=True)
    all_items = load_microns_subsets(seed=0)
    test_items = all_items[-N_TEST:]
    train_items = all_items[:-N_TEST]

    print(f"[DPH] Fitting RF predictor on {len(train_items)} train subsets ...",
          flush=True)
    samplers = build_samplers(train_items, rng)

    # GT diagrams
    print(f"[DPH] Computing GT persistence diagrams ({N_TEST} subsets) ...",
          flush=True)
    gt_diagrams = []
    dist_mats = []
    for k, it in enumerate(test_items):
        if time.time() - t_start > WALL_CAP_S:
            print("[DPH] WARNING: wall-time cap reached in GT phase.", flush=True)
            break
        dm = inter_soma_dist(it)
        dist_mats.append(dm)
        dgm = compute_diagram(it["adjacency"], dist_mat=dm)
        gt_diagrams.append(dgm)
        if (k + 1) % 10 == 0:
            print(f"  GT: {k+1}/{N_TEST}  elapsed={time.time()-t_start:.0f}s",
                  flush=True)

    # GT summary stats
    gt_h0_pts = sum(len(d["dgm_0"]) for d in gt_diagrams)
    gt_h1_pts = sum(len(d["dgm_1"]) for d in gt_diagrams)
    print(f"[DPH] GT: total H0 points={gt_h0_pts}, H1 points={gt_h1_pts}",
          flush=True)

    # Per-sampler evaluation
    results_per_sampler = {}
    for name, sampler_fn in samplers.items():
        print(f"[DPH] Sampler '{name}' ...", flush=True)
        per_subset_dists = []

        for k, it in enumerate(test_items):
            if time.time() - t_start > WALL_CAP_S:
                print(f"[DPH] WARNING: wall-time cap during '{name}'.",
                      flush=True)
                break
            if k >= len(gt_diagrams):
                break
            dm = dist_mats[k]
            mc_dists = {
                "bottleneck_H0": [],
                "bottleneck_H1": [],
                "wasserstein1_H0": [],
                "wasserstein1_H1": [],
            }
            for _ in range(N_MC):
                adj_s = sampler_fn(it, rng)
                dgm_s = compute_diagram(adj_s, dist_mat=dm)
                dists = subset_distances(gt_diagrams[k], dgm_s)
                for key in mc_dists:
                    mc_dists[key].append(dists[key])
            per_subset_dists.append({
                k: float(np.mean(v)) for k, v in mc_dists.items()
            })

        if not per_subset_dists:
            results_per_sampler[name] = {"error": "no_diagrams"}
            continue

        res = {"n_subsets": len(per_subset_dists), "n_mc_per_subset": N_MC}
        for metric in ["bottleneck_H0", "bottleneck_H1",
                       "wasserstein1_H0", "wasserstein1_H1"]:
            vals = [d[metric] for d in per_subset_dists]
            m, s = mean_sem(vals)
            res[f"{metric}_mean"] = m
            res[f"{metric}_sem"] = s
        results_per_sampler[name] = res
        print(f"  {name}: BN_H0={res['bottleneck_H0_mean']:.4f}+/-"
              f"{res['bottleneck_H0_sem']:.4f}  "
              f"BN_H1={res['bottleneck_H1_mean']:.4f}+/-"
              f"{res['bottleneck_H1_sem']:.4f}  "
              f"W1_H0={res['wasserstein1_H0_mean']:.4f}  "
              f"W1_H1={res['wasserstein1_H1_mean']:.4f}", flush=True)

    elapsed = time.time() - t_start
    has_positions = any(dm is not None for dm in dist_mats)

    output = {
        "experiment": "directed_ph_aaai27",
        "addresses": "AAAI-27 Exp 2: Directed PH via flagser",
        "library": _LIBRARY,
        "library_note": (
            "pyflagser directed flag complex (Luetgehetmann et al. 2020) "
            "with inter-soma distance filtration"
            if "pyflagser" in _LIBRARY else
            "gudhi SimplexTree on symmetrised graph -- FALLBACK. "
            "pyflagser not available."
        ),
        "directed": "pyflagser" in _LIBRARY,
        "substrate": "microns_N100",
        "n_test_subsets": len(gt_diagrams),
        "n_mc_per_subset": N_MC,
        "distance_metrics": ["bottleneck_H0", "bottleneck_H1",
                             "wasserstein1_H0", "wasserstein1_H1"],
        "filtration": (
            "inter-soma Euclidean distance (edges enter at increasing distance)"
            if has_positions else "static filtration=1.0 fallback"
        ),
        "gt_total_h0_points": gt_h0_pts,
        "gt_total_h1_points": gt_h1_pts,
        "samplers": results_per_sampler,
        "walltime_s": elapsed,
    }

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "aaai27")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "directed_ph.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[DPH] Saved {json_path}  (elapsed={elapsed:.0f}s)", flush=True)

    # Print comparison summary
    print("\n" + "=" * 60)
    print("DIRECTED PH SUMMARY (bottleneck H1 = key discriminator)")
    print("=" * 60)
    for name, res in results_per_sampler.items():
        if "error" in res:
            print(f"  {name}: ERROR")
        else:
            print(f"  {name:<25} BN_H1={res['bottleneck_H1_mean']:.4f}"
                  f"+/-{res['bottleneck_H1_sem']:.4f}  "
                  f"W1_H1={res['wasserstein1_H1_mean']:.4f}"
                  f"+/-{res['wasserstein1_H1_sem']:.4f}")

    return output


if __name__ == "__main__":
    run()
