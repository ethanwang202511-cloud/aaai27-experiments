"""Experiment 2: Pretrained gene embeddings for the CPG perturbation encoder.

Tests whether pretrained gene embeddings (GenePT, Geneformer, Gene2Vec, ESM-2,
scGPT) improve on truncated-SVD features as gene-identity inputs to the CPG
(Conditional Perturbation Generator) fix for embedding dormancy.

The CPG replaces nn.Embedding with MLP(gene_identity[g]) where gene_identity
is a precomputed d_id-dim vector. The standard pipeline uses truncated SVD over
control-cell expression columns (64-dim). This experiment replaces that with
each of several pretrained embedding sources, PCA-reduced to a common
dimensionality, and measures gap closure relative to the CPA baseline.

Feature sets tested:
  1. svd          -- baseline truncated SVD (from existing compute_gene_identity_table)
  2. genept       -- GenePT (3072-dim text-embedding-3-large, PCA to d_id)
  3. geneformer   -- Geneformer (256-dim word embeddings, PCA to d_id)
  4. gene2vec     -- Gene2Vec (200-dim, PCA to d_id)
  5. esm2         -- ESM-2 (1280-dim mean-pooled protein embeddings, PCA to d_id)
  6. scgpt        -- scGPT (512-dim gene embeddings, PCA to d_id)
  7. svd_genept   -- SVD + GenePT concatenated (128-dim total, PCA to d_id)

Protocol: train CPG with each feature set on Replogle K562 and RPE1, 3 seeds
each. Report per-run audit CSV + summary table comparing gap closure.

Usage:
  python run_pretrained_embeddings.py --feature_set genept --dataset k562 --seeds 0,1,2
  python run_pretrained_embeddings.py --feature_set all --dataset both --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import svds
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
INTERVENEFM_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "InterveneFM"
sys.path.insert(0, str(INTERVENEFM_ROOT))

from src.cpa_cpg import CPGModel, CPGConfig, CPGPertEncoder, compute_gene_identity_table
from src.data_replogle import (
    load_replogle,
    build_split_replogle,
    build_gene_vocab_replogle,
    DATA as K562_DATA,
)
from src.data_replogle_rpe1 import load_replogle_rpe1, DATA as RPE1_DATA
from src.data_norman import PerturbSeqDataset
from src.audit import predict_under_mode, get_top_deg_indices

RESULTS_DIR = INTERVENEFM_ROOT / "results" / "exp2_pretrained_embeddings"

ALL_FEATURE_SETS = [
    "svd", "genept", "geneformer", "gene2vec", "esm2", "scgpt", "svd_genept",
]

# Paths to pretrained embedding caches (populated by load functions).
# Users should set PRETRAINED_EMB_DIR env var or place files under
# InterveneFM/data/pretrained_embeddings/.
PRETRAINED_EMB_DIR = Path(
    os.environ.get("PRETRAINED_EMB_DIR",
                    str(INTERVENEFM_ROOT / "data" / "pretrained_embeddings"))
)


# ---------------------------------------------------------------------------
# Embedding loaders: each returns Dict[gene_name, np.ndarray(float32)]
# ---------------------------------------------------------------------------

def load_genept_embeddings() -> Dict[str, np.ndarray]:
    """Load GenePT embeddings (3072-dim from text-embedding-3-large).

    Expects a pickle or npy file at PRETRAINED_EMB_DIR/genept/.
    Supported formats:
      - gene_embeddings_3072.pkl  (dict: gene_name -> np.array)
      - GenePT_gene_embedding_ada_text_3072.npy + gene_names.txt
    """
    emb_dir = PRETRAINED_EMB_DIR / "genept"
    pkl_path = emb_dir / "gene_embeddings_3072.pkl"
    npy_path = emb_dir / "GenePT_gene_embedding_ada_text_3072.npy"
    names_path = emb_dir / "gene_names.txt"

    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        if isinstance(raw, dict):
            return {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}
        raise ValueError(f"Unexpected pickle format in {pkl_path}")

    if npy_path.exists() and names_path.exists():
        mat = np.load(npy_path).astype(np.float32)
        with open(names_path) as f:
            names = [line.strip() for line in f if line.strip()]
        assert len(names) == mat.shape[0], (
            f"gene_names ({len(names)}) vs embedding rows ({mat.shape[0]}) mismatch"
        )
        return {name: mat[i] for i, name in enumerate(names)}

    # Try any pkl/npy in the directory
    for p in sorted(emb_dir.glob("*.pkl")):
        with open(p, "rb") as f:
            raw = pickle.load(f)
        if isinstance(raw, dict):
            print(f"[genept] loaded from {p.name}")
            return {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}

    raise FileNotFoundError(
        f"GenePT embeddings not found in {emb_dir}. "
        "Download from github.com/yiqunchen/GenePT and place .pkl or .npy files there."
    )


def load_geneformer_embeddings() -> Dict[str, np.ndarray]:
    """Extract gene embeddings from the Geneformer pretrained model.

    Loads word_embeddings.weight from ctheodoris/Geneformer (256-dim).
    Requires the Geneformer token-to-gene mapping file.
    """
    cache_path = PRETRAINED_EMB_DIR / "geneformer" / "gene_embeddings.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    try:
        from transformers import BertForMaskedLM
    except ImportError:
        raise ImportError(
            "transformers required for Geneformer embedding extraction. "
            "Install with: pip install transformers"
        )

    model_name = "ctheodoris/Geneformer"
    print(f"[geneformer] loading model from {model_name}...")
    model = BertForMaskedLM.from_pretrained(model_name)
    weights = model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy().astype(np.float32)

    # Load token-to-gene mapping
    token_map_path = PRETRAINED_EMB_DIR / "geneformer" / "token_to_gene.json"
    if not token_map_path.exists():
        token_map_path = PRETRAINED_EMB_DIR / "geneformer" / "token_dictionary.pkl"
    if token_map_path.suffix == ".json":
        with open(token_map_path) as f:
            token_to_gene = json.load(f)
    elif token_map_path.suffix == ".pkl":
        with open(token_map_path, "rb") as f:
            token_to_gene = pickle.load(f)
    else:
        raise FileNotFoundError(
            f"Geneformer token mapping not found at {token_map_path}. "
            "Provide token_to_gene.json or token_dictionary.pkl."
        )

    # Build gene -> embedding dict. token_to_gene maps ensembl_id -> token_idx
    # or gene_name -> token_idx depending on version. We try to map to symbols.
    result: Dict[str, np.ndarray] = {}
    for key, idx in token_to_gene.items():
        if isinstance(idx, int) and 0 <= idx < weights.shape[0]:
            gene_sym = str(key).upper()
            result[gene_sym] = weights[idx]

    print(f"[geneformer] extracted {len(result)} gene embeddings (dim={weights.shape[1]})")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result


def load_gene2vec_embeddings() -> Dict[str, np.ndarray]:
    """Load Gene2Vec pretrained embeddings (200-dim).

    Expects gene2vec_dim_200_iter_9.txt in PRETRAINED_EMB_DIR/gene2vec/.
    Format: one header line, then each line is "GENE_NAME val1 val2 ... val200".
    """
    emb_dir = PRETRAINED_EMB_DIR / "gene2vec"
    txt_path = emb_dir / "gene2vec_dim_200_iter_9.txt"
    if not txt_path.exists():
        # Try any .txt file
        candidates = list(emb_dir.glob("*.txt"))
        if not candidates:
            raise FileNotFoundError(
                f"Gene2Vec embeddings not found in {emb_dir}. "
                "Download from github.com/jingcheng-du/Gene2vec."
            )
        txt_path = candidates[0]

    result: Dict[str, np.ndarray] = {}
    with open(txt_path) as f:
        header = f.readline()  # skip header (num_genes dim)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            gene_name = parts[0].upper()
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            result[gene_name] = vec

    print(f"[gene2vec] loaded {len(result)} gene embeddings (dim={len(next(iter(result.values())))})")
    return result


def load_esm2_embeddings() -> Dict[str, np.ndarray]:
    """Load ESM-2 protein embeddings (1280-dim mean-pooled).

    Expects a precomputed cache at PRETRAINED_EMB_DIR/esm2/gene_embeddings.pkl
    (dict: gene_symbol -> np.array of shape (1280,)).

    To generate the cache:
    1. Map gene symbols to UniProt canonical sequences.
    2. Run esm2_t33_650M_UR50D inference on each sequence.
    3. Mean-pool over sequence length to get a (1280,) vector per gene.
    4. Save as pickle dict.
    """
    cache_path = PRETRAINED_EMB_DIR / "esm2" / "gene_embeddings.pkl"
    if not cache_path.exists():
        npy_path = PRETRAINED_EMB_DIR / "esm2" / "gene_embeddings.npy"
        names_path = PRETRAINED_EMB_DIR / "esm2" / "gene_names.txt"
        if npy_path.exists() and names_path.exists():
            mat = np.load(npy_path).astype(np.float32)
            with open(names_path) as f:
                names = [line.strip() for line in f if line.strip()]
            return {name: mat[i] for i, name in enumerate(names)}
        raise FileNotFoundError(
            f"ESM-2 embeddings not found at {cache_path}. "
            "Precompute mean-pooled ESM-2 representations and save as pickle dict. "
            "See docstring for instructions."
        )

    with open(cache_path, "rb") as f:
        raw = pickle.load(f)
    result = {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}
    print(f"[esm2] loaded {len(result)} gene embeddings (dim={len(next(iter(result.values())))})")
    return result


def load_scgpt_embeddings() -> Dict[str, np.ndarray]:
    """Load scGPT gene embeddings (512-dim).

    Expects a precomputed cache at PRETRAINED_EMB_DIR/scgpt/gene_embeddings.pkl
    or a checkpoint + gene list from which we extract encoder.embedding.weight.
    """
    cache_path = PRETRAINED_EMB_DIR / "scgpt" / "gene_embeddings.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            raw = pickle.load(f)
        result = {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}
        print(f"[scgpt] loaded {len(result)} gene embeddings (dim={len(next(iter(result.values())))})")
        return result

    # Try loading from checkpoint
    ckpt_path = PRETRAINED_EMB_DIR / "scgpt" / "best_model.pt"
    gene_list_path = PRETRAINED_EMB_DIR / "scgpt" / "gene_list.txt"
    if ckpt_path.exists() and gene_list_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "encoder.embedding.weight" in state:
            weights = state["encoder.embedding.weight"].numpy().astype(np.float32)
        elif "model" in state and "encoder.embedding.weight" in state["model"]:
            weights = state["model"]["encoder.embedding.weight"].numpy().astype(np.float32)
        else:
            # Try finding the right key
            emb_keys = [k for k in state.keys() if "embedding" in k.lower() and "weight" in k.lower()]
            if emb_keys:
                weights = state[emb_keys[0]].numpy().astype(np.float32)
            else:
                raise KeyError(f"Cannot find embedding weights in checkpoint. Keys: {list(state.keys())[:20]}")
        with open(gene_list_path) as f:
            genes = [line.strip() for line in f if line.strip()]
        result = {g: weights[i] for i, g in enumerate(genes) if i < weights.shape[0]}
        print(f"[scgpt] extracted {len(result)} gene embeddings (dim={weights.shape[1]})")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        return result

    raise FileNotFoundError(
        f"scGPT embeddings not found at {cache_path}. "
        "Download from github.com/bowang-lab/scGPT and extract encoder.embedding.weight."
    )


# ---------------------------------------------------------------------------
# Gene identity table builder from pretrained embeddings
# ---------------------------------------------------------------------------

def build_gene_identity_table_from_embeddings(
    embeddings: Dict[str, np.ndarray],
    vocab: Dict[str, int],
    d_id: int = 64,
    seed: int = 0,
    label: str = "pretrained",
) -> torch.Tensor:
    """Map vocab entries to pretrained embedding rows, PCA-reduce to d_id.

    Returns a (n_perts+1, d_id) float32 tensor with row 0 = zeros (pad/control).
    Genes in vocab but not in the embedding dict get zero vectors (with warning).

    Args:
        embeddings: dict mapping gene names (uppercase) to raw embedding vectors.
        vocab: gene name -> 1-indexed perturbation index.
        d_id: target dimensionality after PCA.
        seed: random seed for PCA.
        label: name for logging.

    Returns:
        Tensor of shape (max(vocab.values()) + 1, d_id).
    """
    n_perts = max(vocab.values()) if vocab else 0

    # Collect raw embeddings for genes that are in both vocab and embedding dict
    resolved_genes: List[str] = []
    resolved_indices: List[int] = []
    raw_vecs: List[np.ndarray] = []
    missing_genes: List[str] = []

    for gname, pidx in vocab.items():
        gname_upper = gname.upper()
        if gname_upper in embeddings:
            resolved_genes.append(gname)
            resolved_indices.append(pidx)
            raw_vecs.append(embeddings[gname_upper])
        elif gname in embeddings:
            resolved_genes.append(gname)
            resolved_indices.append(pidx)
            raw_vecs.append(embeddings[gname])
        else:
            missing_genes.append(gname)

    n_resolved = len(resolved_genes)
    n_missing = len(missing_genes)
    print(f"[gene_id_{label}] {n_resolved}/{len(vocab)} pert genes resolved; "
          f"{n_missing} missing (will get zero vectors)")

    if n_resolved == 0:
        print(f"[gene_id_{label}] WARNING: no genes resolved. Returning zero table.")
        return torch.zeros(n_perts + 1, d_id)

    raw_mat = np.stack(raw_vecs, axis=0)  # (n_resolved, raw_dim)
    raw_dim = raw_mat.shape[1]
    print(f"[gene_id_{label}] raw embedding dim={raw_dim}, target d_id={d_id}")

    # PCA reduce to d_id
    effective_d = min(d_id, raw_mat.shape[0], raw_mat.shape[1])
    if effective_d < d_id:
        print(f"[gene_id_{label}] clamping PCA components to {effective_d} "
              f"(min of d_id={d_id}, n_genes={raw_mat.shape[0]}, raw_dim={raw_dim})")

    pca = PCA(n_components=effective_d, random_state=seed)
    reduced = pca.fit_transform(raw_mat)  # (n_resolved, effective_d)
    explained = pca.explained_variance_ratio_.sum()
    print(f"[gene_id_{label}] PCA explained variance: {explained:.3f} "
          f"(top-5 components: {pca.explained_variance_ratio_[:5].round(3)})")

    # Build the table
    table = np.zeros((n_perts + 1, d_id), dtype=np.float32)
    for i, pidx in enumerate(resolved_indices):
        table[pidx, :effective_d] = reduced[i]

    return torch.from_numpy(table)


def build_svd_genept_concat_table(
    adata_full,
    vocab: Dict[str, int],
    ctrl_cell_ids: List[str],
    genept_embeddings: Dict[str, np.ndarray],
    d_id: int = 64,
    seed: int = 0,
) -> torch.Tensor:
    """Concatenate SVD (64-dim) + GenePT (64-dim) = 128-dim, then PCA to d_id.

    This tests whether combining expression-derived and text-derived features
    is complementary.
    """
    # Get SVD table (full dim, not PCA-reduced)
    svd_dim = 64
    svd_table = compute_gene_identity_table(
        adata_full, vocab, ctrl_cell_ids, d_id=svd_dim, seed=seed,
    )
    svd_np = svd_table.numpy()  # (n_perts+1, svd_dim)

    # Get GenePT table (64-dim PCA)
    genept_table = build_gene_identity_table_from_embeddings(
        genept_embeddings, vocab, d_id=svd_dim, seed=seed, label="genept_for_concat",
    )
    genept_np = genept_table.numpy()  # (n_perts+1, svd_dim)

    # Concatenate (skip row 0 = pad)
    concat = np.concatenate([svd_np, genept_np], axis=1)  # (n_perts+1, 128)
    print(f"[svd_genept] concatenated shape: {concat.shape}")

    # PCA the non-zero rows down to d_id
    nonzero_mask = np.any(concat != 0, axis=1)
    nonzero_indices = np.where(nonzero_mask)[0]
    if len(nonzero_indices) == 0:
        return torch.zeros(concat.shape[0], d_id)

    nonzero_data = concat[nonzero_indices]
    effective_d = min(d_id, nonzero_data.shape[0], nonzero_data.shape[1])
    pca = PCA(n_components=effective_d, random_state=seed)
    reduced = pca.fit_transform(nonzero_data)
    print(f"[svd_genept] PCA {concat.shape[1]} -> {effective_d}, "
          f"explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    table = np.zeros((concat.shape[0], d_id), dtype=np.float32)
    for i, idx in enumerate(nonzero_indices):
        table[idx, :effective_d] = reduced[i]

    return torch.from_numpy(table)


# ---------------------------------------------------------------------------
# Full-panel loaders (no HVG restriction, for gene-identity computation)
# ---------------------------------------------------------------------------

def load_replogle_full_panel(
    dataset: str,
    max_cells: int | None = None,
    seed: int = 0,
):
    """Load Replogle K562 or RPE1 without HVG restriction."""
    data_path = K562_DATA if dataset == "k562" else RPE1_DATA
    default_max = 60000 if dataset == "k562" else 80000
    if max_cells is None:
        max_cells = default_max

    adata = sc.read_h5ad(data_path)
    if "nperts" in adata.obs.columns:
        adata.obs["nperts"] = adata.obs["nperts"].astype(int)

    pert_genes = []
    for p in adata.obs["perturbation"]:
        s = str(p).strip()
        if s == "control" or "NegCtrl" in s or s == "unassigned":
            pert_genes.append([])
        else:
            base = s.split(".")[0].split(";")[0]
            pert_genes.append([base])
    adata.obs["pert_genes"] = np.array(pert_genes, dtype=object)
    adata.obs["n_pert"] = [len(x) for x in adata.obs["pert_genes"]]

    if max_cells is not None and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        keep = rng.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[np.sort(keep)].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


# ---------------------------------------------------------------------------
# Training + audit for a single (feature_set, dataset, seed) run
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    feature_set: str
    dataset: str  # "k562" or "rpe1"
    seed: int
    epochs: int = 40
    batch_size: int = 128
    lr: float = 3e-4
    gene_id_dim: int = 64
    n_per_gene: int = 200
    hvg: int = 2000
    max_cells: int | None = None
    max_test_genes: int = 80
    device: str = "cpu"


def build_gene_identity_for_feature_set(
    feature_set: str,
    vocab: Dict[str, int],
    adata_full,
    ctrl_cell_ids: List[str],
    d_id: int = 64,
    seed: int = 0,
) -> torch.Tensor:
    """Dispatch to the appropriate gene-identity-table builder."""
    if feature_set == "svd":
        return compute_gene_identity_table(adata_full, vocab, ctrl_cell_ids, d_id=d_id, seed=seed)
    elif feature_set == "genept":
        embs = load_genept_embeddings()
        return build_gene_identity_table_from_embeddings(embs, vocab, d_id=d_id, seed=seed, label="genept")
    elif feature_set == "geneformer":
        embs = load_geneformer_embeddings()
        return build_gene_identity_table_from_embeddings(embs, vocab, d_id=d_id, seed=seed, label="geneformer")
    elif feature_set == "gene2vec":
        embs = load_gene2vec_embeddings()
        return build_gene_identity_table_from_embeddings(embs, vocab, d_id=d_id, seed=seed, label="gene2vec")
    elif feature_set == "esm2":
        embs = load_esm2_embeddings()
        return build_gene_identity_table_from_embeddings(embs, vocab, d_id=d_id, seed=seed, label="esm2")
    elif feature_set == "scgpt":
        embs = load_scgpt_embeddings()
        return build_gene_identity_table_from_embeddings(embs, vocab, d_id=d_id, seed=seed, label="scgpt")
    elif feature_set == "svd_genept":
        genept_embs = load_genept_embeddings()
        return build_svd_genept_concat_table(
            adata_full, vocab, ctrl_cell_ids, genept_embs, d_id=d_id, seed=seed,
        )
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")


def run_single(cfg: RunConfig) -> pd.DataFrame:
    """Train CPG with the specified feature set and run the dormancy audit.

    Returns a DataFrame with per-(test_gene, mode) audit rows.
    """
    tag = f"{cfg.feature_set}_{cfg.dataset}_seed{cfg.seed}"
    print(f"\n{'='*72}")
    print(f"[run] {tag}")
    print(f"{'='*72}")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    t0 = time.time()

    # ---- Load full-panel (for gene-identity computation) ----
    adata_full = load_replogle_full_panel(cfg.dataset, max_cells=cfg.max_cells, seed=cfg.seed)
    print(f"[load] {cfg.dataset} full-panel: {adata_full.shape} ({time.time()-t0:.1f}s)")

    # ---- HVG-restrict for training ----
    ctrl_mask = (adata_full.obs["n_pert"] == 0).values
    ctrl_only = adata_full[ctrl_mask].copy()
    sc.pp.highly_variable_genes(
        ctrl_only, n_top_genes=cfg.hvg, subset=False, flavor="seurat", batch_key=None,
    )
    hvg_mask = ctrl_only.var["highly_variable"].values
    adata = adata_full[:, hvg_mask].copy()
    print(f"[hvg] training adata: {adata.shape}")

    # ---- Vocab + split ----
    vocab = build_gene_vocab_replogle(adata)
    split = build_split_replogle(adata, test_frac_genes=0.2, seed=cfg.seed)
    print(f"[split] train={len(split['train_cells'])}, test={len(split['test_cells'])}, "
          f"ctrl={len(split['ctrl_cells'])}, test_genes={len(split['test_genes'])}")

    # Cap test genes for tractability
    if len(split["test_genes"]) > cfg.max_test_genes:
        rng_cap = np.random.default_rng(cfg.seed)
        split["test_genes"] = list(rng_cap.choice(
            np.array(split["test_genes"], dtype=object),
            size=cfg.max_test_genes, replace=False,
        ))
        keep_genes = set(split["test_genes"])
        n_pert_arr = adata.obs["n_pert"].values
        pg_arr = adata.obs["pert_genes"].values
        new_test = []
        new_train = []
        for ci, cell_id in enumerate(adata.obs.index):
            if n_pert_arr[ci] == 0:
                new_train.append(cell_id)
            elif n_pert_arr[ci] == 1 and pg_arr[ci][0] in keep_genes:
                new_test.append(cell_id)
            else:
                new_train.append(cell_id)
        split["test_cells"] = new_test
        split["train_cells"] = new_train
        print(f"[split] capped: test_genes={len(split['test_genes'])}, "
              f"train={len(new_train)}, test={len(new_test)}")

    # ---- Build gene-identity table ----
    print(f"[gene_id] building {cfg.feature_set} gene-identity table (d_id={cfg.gene_id_dim})...")
    gene_id_table = build_gene_identity_for_feature_set(
        cfg.feature_set, vocab, adata_full, split["ctrl_cells"],
        d_id=cfg.gene_id_dim, seed=cfg.seed,
    )
    print(f"[gene_id] table shape: {gene_id_table.shape}")
    # Check coverage: fraction of non-zero rows
    nonzero_rows = (gene_id_table.abs().sum(dim=1) > 0).sum().item()
    print(f"[gene_id] non-zero rows: {nonzero_rows}/{gene_id_table.shape[0]} "
          f"({100*nonzero_rows/gene_id_table.shape[0]:.1f}%)")
    del adata_full  # free memory

    # ---- Model ----
    model_cfg = CPGConfig(
        n_genes=adata.n_vars,
        n_pert_genes=len(vocab),
        gene_id_dim=cfg.gene_id_dim,
        z_dim=64,
        pert_dim=32,
        hidden=256,
        pert_mlp_hidden=128,
    )
    device = torch.device(cfg.device)
    model = CPGModel(model_cfg, gene_id_table).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {n_params/1e6:.2f}M")

    # ---- Training ----
    train_ds = PerturbSeqDataset(
        adata, split["train_cells"], split["ctrl_cells"],
        vocab, max_perts=2, seed=cfg.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=1e-5,
    )

    train_log: List[Dict] = []
    for epoch in range(cfg.epochs):
        model.train()
        ep_loss = 0.0
        ep_n = 0
        for basal, target, pidx in train_loader:
            basal, target, pidx = basal.to(device), target.to(device), pidx.to(device)
            x_hat = model(basal, pidx)
            loss = F.mse_loss(x_hat, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * basal.shape[0]
            ep_n += basal.shape[0]
        avg = ep_loss / ep_n
        if (epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1:
            print(f"[train] epoch {epoch+1}/{cfg.epochs} mse={avg:.4f} ({time.time()-t0:.1f}s)")
        train_log.append({"epoch": epoch + 1, "mse": avg, "elapsed_s": time.time() - t0})

    # Save training log
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_log).to_csv(RESULTS_DIR / f"training_log_{tag}.csv", index=False)

    # Save model checkpoint
    torch.save(
        {"state_dict": model.state_dict(), "cfg": model_cfg.__dict__, "vocab": vocab},
        RESULTS_DIR / f"model_{tag}.pt",
    )

    # ---- Audit ----
    model.eval()
    model = model.to("cpu")  # audit on CPU for simplicity
    obs = adata.obs
    train_set = set(split["train_cells"])
    pg = obs["pert_genes"].values
    n_pert_arr = obs["n_pert"].values

    # Collect train pert gene indices
    train_pert_genes_set: set = set()
    for ci, cell_id in enumerate(obs.index):
        if cell_id in train_set:
            for g in pg[ci]:
                if g in vocab:
                    train_pert_genes_set.add(g)
    train_pert_gene_indices = sorted(vocab[g] for g in train_pert_genes_set)
    print(f"[audit] train pert genes: {len(train_pert_gene_indices)}")

    # Collect train perturbed cells (for pop_mean baseline)
    train_perturbed_cell_ids: List[str] = []
    for ci, cell_id in enumerate(obs.index):
        if cell_id in train_set and n_pert_arr[ci] >= 1:
            train_perturbed_cell_ids.append(cell_id)
    print(f"[audit] train perturbed cells: {len(train_perturbed_cell_ids)}")

    rng = np.random.default_rng(cfg.seed)

    # Build train embed pool via the CPG MLP (not nn.Embedding indexing)
    train_idx_t = torch.tensor(train_pert_gene_indices, dtype=torch.long)
    train_idx_padded = torch.cat(
        [train_idx_t.unsqueeze(1), torch.zeros_like(train_idx_t.unsqueeze(1))], dim=1,
    )
    with torch.no_grad():
        train_embed_pool = model.pert_encoder(train_idx_padded)
    print(f"[audit] train_embed_pool shape: {train_embed_pool.shape}")

    # Pre-extract expression matrices
    cell_to_row = {c: i for i, c in enumerate(obs.index)}
    ctrl_rows = np.array([cell_to_row[c] for c in split["ctrl_cells"]])
    X = adata.X
    is_sparse = hasattr(X, "toarray")

    def get_X(rows: np.ndarray) -> np.ndarray:
        if is_sparse:
            return np.asarray(X[rows].toarray()).astype(np.float32)
        return np.asarray(X[rows]).astype(np.float32)

    ctrl_X = get_X(ctrl_rows)
    train_pert_rows = np.array([cell_to_row[c] for c in train_perturbed_cell_ids])
    tpm = get_X(train_pert_rows).mean(0)
    train_pert_mean = torch.from_numpy(tpm)

    # Per-test-gene audit (Replogle = single perturbations, 0/1 split)
    rows_out: List[Dict] = []
    for tg in split["test_genes"]:
        mask = np.array([(len(g) == 1 and g[0] == tg) for g in pg])
        pert_rows = np.where(mask)[0]
        if len(pert_rows) < 5:
            continue
        pert_X = get_X(pert_rows)
        deg_idx = get_top_deg_indices(pert_X, ctrl_X, k=200)
        obs_pert_mean = pert_X.mean(0)
        obs_delta_full = obs_pert_mean - ctrl_X.mean(0)
        obs_delta_top = obs_delta_full[deg_idx]

        basal_rows = rng.choice(ctrl_rows, size=cfg.n_per_gene, replace=True)
        basal = torch.from_numpy(get_X(basal_rows))
        idx = vocab.get(tg, 0)
        pert_idx = torch.tensor([[idx, 0]] * cfg.n_per_gene, dtype=torch.long)

        with torch.no_grad():
            for mode in ("learned", "mean", "zero", "random", "identity", "pop_mean"):
                x_hat = predict_under_mode(
                    model, basal, pert_idx, mode, train_embed_pool, rng,
                    train_pert_mean_x=train_pert_mean,
                ).numpy()
                pred_delta = x_hat.mean(0) - ctrl_X.mean(0)
                pred_delta_top = pred_delta[deg_idx]
                rho = spearmanr(pred_delta_top, obs_delta_top).statistic
                num = np.dot(
                    pred_delta - pred_delta.mean(),
                    obs_delta_full - obs_delta_full.mean(),
                )
                denom = (
                    np.linalg.norm(pred_delta - pred_delta.mean())
                    * np.linalg.norm(obs_delta_full - obs_delta_full.mean())
                    + 1e-12
                )
                pearson = float(num / denom)
                mae = float(np.mean(np.abs(pred_delta_top - obs_delta_top)))
                mse = float(((x_hat - obs_pert_mean[None]) ** 2).mean())
                rows_out.append({
                    "feature_set": cfg.feature_set,
                    "dataset": cfg.dataset,
                    "seed": cfg.seed,
                    "test_gene": tg,
                    "mode": mode,
                    "DE_Spearman": float(rho) if not np.isnan(rho) else np.nan,
                    "Pearson_delta_full": pearson,
                    "mae_delta_top200": mae,
                    "mse_to_pert_mean": mse,
                    "n_obs_pert": int(len(pert_rows)),
                })

    df = pd.DataFrame(rows_out)
    out_csv = RESULTS_DIR / f"audit_{tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    # Summary
    if len(df) > 0:
        summary = df.groupby("mode").agg(
            DE_Spearman_mean=("DE_Spearman", "mean"),
            DE_Spearman_median=("DE_Spearman", "median"),
            DE_Spearman_sem=("DE_Spearman", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
            Pearson_full_mean=("Pearson_delta_full", "mean"),
            n_genes=("test_gene", "count"),
        ).round(4)
        print(f"\n=== {tag} audit summary ===")
        print(summary.to_string())
        summary.to_csv(RESULTS_DIR / f"audit_summary_{tag}.csv")

        # Gap (learned - pop_mean)
        piv = df.pivot_table(
            index="test_gene", columns="mode", values="DE_Spearman",
        ).dropna(subset=["learned", "pop_mean"])
        if len(piv) > 0:
            gap = piv["learned"] - piv["pop_mean"]
            rng_boot = np.random.default_rng(0)
            boot = np.array([
                rng_boot.choice(gap.values, size=len(gap), replace=True).mean()
                for _ in range(2000)
            ])
            print(f"\n[gap] {tag}: n={len(gap)}, mean={gap.mean():+.4f}, "
                  f"95% CI [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]")

    elapsed = time.time() - t0
    print(f"[done] {tag} in {elapsed:.1f}s")
    return df


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------

def aggregate_results(all_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate across seeds and produce summary comparison table.

    Returns a DataFrame with one row per (feature_set, dataset, mode) showing
    mean and SEM of DE_Spearman across seeds.
    """
    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(RESULTS_DIR / "all_runs_combined.csv", index=False)

    # Per-seed summary: average across test genes within each seed
    seed_summary = combined.groupby(
        ["feature_set", "dataset", "seed", "mode"]
    ).agg(
        DE_Spearman_mean=("DE_Spearman", "mean"),
        Pearson_full_mean=("Pearson_delta_full", "mean"),
        n_genes=("test_gene", "nunique"),
    ).reset_index()

    # Aggregate across seeds
    final = seed_summary.groupby(["feature_set", "dataset", "mode"]).agg(
        DE_Spearman=("DE_Spearman_mean", "mean"),
        DE_Spearman_sem=("DE_Spearman_mean", lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0),
        Pearson_full=("Pearson_full_mean", "mean"),
        n_seeds=("seed", "nunique"),
        n_genes_per_seed=("n_genes", "mean"),
    ).reset_index().round(4)

    print("\n" + "=" * 80)
    print("EXPERIMENT 2 SUMMARY: Pretrained Embeddings for CPG")
    print("=" * 80)
    print(final.to_string(index=False))

    final.to_csv(RESULTS_DIR / "exp2_summary.csv", index=False)

    # Compute gap closure relative to CPA baseline for each feature set
    gap_rows: List[Dict] = []
    for (fs, ds), grp in combined.groupby(["feature_set", "dataset"]):
        piv = grp.pivot_table(
            index=["seed", "test_gene"], columns="mode", values="DE_Spearman",
        )
        if "learned" not in piv.columns or "pop_mean" not in piv.columns:
            continue
        piv = piv.dropna(subset=["learned", "pop_mean"])
        if len(piv) == 0:
            continue

        gap_per_gene = piv["learned"] - piv["pop_mean"]
        mean_gap = gap_per_gene.mean()
        # Bootstrap CI
        rng_boot = np.random.default_rng(42)
        boot = np.array([
            rng_boot.choice(gap_per_gene.values, size=len(gap_per_gene), replace=True).mean()
            for _ in range(2000)
        ])
        gap_rows.append({
            "feature_set": fs,
            "dataset": ds,
            "gap_mean": round(mean_gap, 4),
            "gap_ci_lo": round(float(np.percentile(boot, 2.5)), 4),
            "gap_ci_hi": round(float(np.percentile(boot, 97.5)), 4),
            "n_obs": len(gap_per_gene),
        })

    if gap_rows:
        gap_df = pd.DataFrame(gap_rows)
        print("\n--- Gap (learned - pop_mean) by feature set ---")
        print(gap_df.to_string(index=False))
        gap_df.to_csv(RESULTS_DIR / "exp2_gap_closure.csv", index=False)

    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Experiment 2: pretrained gene embeddings for CPG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--feature_set", type=str, default="svd",
        choices=ALL_FEATURE_SETS + ["all"],
        help="Which embedding feature set to use (default: svd). "
             "'all' runs all feature sets sequentially.",
    )
    ap.add_argument(
        "--dataset", type=str, default="k562",
        choices=["k562", "rpe1", "both"],
        help="Which Replogle dataset to use (default: k562).",
    )
    ap.add_argument(
        "--seeds", type=str, default="0,1,2",
        help="Comma-separated list of random seeds (default: 0,1,2).",
    )
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gene_id_dim", type=int, default=64)
    ap.add_argument("--hvg", type=int, default=2000)
    ap.add_argument("--n_per_gene", type=int, default=200)
    ap.add_argument("--max_cells", type=int, default=None)
    ap.add_argument("--max_test_genes", type=int, default=80)
    ap.add_argument(
        "--device", type=str, default="cpu",
        help="Device for training (cpu/cuda/mps).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    feature_sets = ALL_FEATURE_SETS if args.feature_set == "all" else [args.feature_set]
    datasets = ["k562", "rpe1"] if args.dataset == "both" else [args.dataset]

    n_runs = len(feature_sets) * len(datasets) * len(seeds)
    print(f"[exp2] {n_runs} total runs: "
          f"feature_sets={feature_sets}, datasets={datasets}, seeds={seeds}")
    print(f"[exp2] results -> {RESULTS_DIR}")

    all_dfs: List[pd.DataFrame] = []
    run_idx = 0

    for fs in feature_sets:
        for ds in datasets:
            for seed in seeds:
                run_idx += 1
                print(f"\n[exp2] === Run {run_idx}/{n_runs} ===")
                run_cfg = RunConfig(
                    feature_set=fs,
                    dataset=ds,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    gene_id_dim=args.gene_id_dim,
                    n_per_gene=args.n_per_gene,
                    hvg=args.hvg,
                    max_cells=args.max_cells,
                    max_test_genes=args.max_test_genes,
                    device=args.device,
                )
                try:
                    df = run_single(run_cfg)
                    all_dfs.append(df)
                except FileNotFoundError as e:
                    print(f"[exp2] SKIPPING {fs}/{ds}/seed{seed}: {e}")
                except Exception as e:
                    print(f"[exp2] ERROR in {fs}/{ds}/seed{seed}: {e}")
                    import traceback
                    traceback.print_exc()

    # Final aggregation
    if all_dfs:
        aggregate_results(all_dfs)
    else:
        print("[exp2] No runs completed successfully.")


if __name__ == "__main__":
    main()
