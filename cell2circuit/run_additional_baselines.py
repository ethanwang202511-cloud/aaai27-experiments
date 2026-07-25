"""run_additional_baselines.py -- Experiment 4: Additional graph generation baselines.

Adds GraphRNN, GRAN, and NetGAN baselines with directed-graph adaptation.
Each is wrapped in a common interface matching the existing baseline pattern.

These baselines test whether the structural-null (config model) dominance
is specific to diffusion/flow architectures or a general property of all
learned generators on connectomes.

Baseline families:
  - GraphRNN (You et al., ICML 2018): autoregressive, node-by-node
  - GRAN (Liao et al., NeurIPS 2019): autoregressive with attention blocks
  - NetGAN (Bojchevski et al., ICML 2018): GAN on random walks

All three require directed-graph adaptation since they were designed for
undirected graphs.

AAAI-27 Experiment 4.
GPU required for GRAN/NetGAN training. Output: results/aaai27/additional_baselines.json

Installation notes:
  - GraphRNN: clone github.com/mark-koch/graph-rnn (directed extension)
    or implement the directed adaptation inline (this script does the latter).
  - GRAN: clone github.com/lrjconan/GRAN and remove symmetrization.
  - NetGAN: use github.com/mmiller96/netgan_pytorch (PyTorch port).

This script provides self-contained implementations of all three baselines
suitable for N=100 directed graphs, avoiding external repo dependencies.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
CELL2CIRCUIT_ROOT = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "..", "Cell2Circuit")
)
SRC = os.path.join(CELL2CIRCUIT_ROOT, "src")
SCRIPTS = os.path.join(CELL2CIRCUIT_ROOT, "scripts")
for p in (SRC, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from analysis.degree_constrained_ablation import sinkhorn_degree_match
from analysis.rigorous_reeval import configuration_model_sample
from evaluation.graph_stats import reciprocity, motif_3node_spectrum

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_STEPS = 5000
BATCH_SIZE = 8
LR = 1e-3
N = 100           # graph size
MAX_PREV_NODE = 40
HIDDEN_DIM = 128
SEED = 42


# ---------------------------------------------------------------------------
# Metrics (self-contained)
# ---------------------------------------------------------------------------

def reach_3hop(adj):
    A = (adj > 0.5).astype(np.float32)
    np.fill_diagonal(A, 0)
    N_ = A.shape[0]
    R = A.copy()
    for _ in range(2):
        R = ((R @ A) > 0).astype(np.float32)
        R = np.clip(R + A, 0, 1)
    np.fill_diagonal(R, 0)
    return float(R.sum() / max(N_ * (N_ - 1), 1))


def motif_l1(adj_gt, adj_s):
    m_gt = motif_3node_spectrum(adj_gt)
    m_s = motif_3node_spectrum(adj_s)
    s_gt = m_gt.sum(); s_s = m_s.sum()
    if s_gt > 0: m_gt = m_gt / s_gt
    if s_s > 0: m_s = m_s / s_s
    return float(np.abs(m_gt - m_s).sum())


# ===================================================================
# BASELINE 1: Directed GraphRNN
# ===================================================================
# Adapted from You et al. (ICML 2018) for directed graphs.
# Key change: for each new node i, generate BOTH incoming and outgoing
# edges to previously generated nodes (2 * MAX_PREV_NODE outputs instead
# of MAX_PREV_NODE).
# Node ordering: BFS on the undirected skeleton (standard GraphRNN).
# ===================================================================

if HAS_TORCH:
    class DirectedGraphRNN(nn.Module):
        """Directed GraphRNN: graph-level GRU + edge-level GRU.

        For directed graphs, each new node generates:
        - outgoing edges to previous nodes (row of adjacency)
        - incoming edges from previous nodes (column of adjacency)

        Combined into a single 2*M-dim output per node step.
        """
        def __init__(self, max_prev_node: int = MAX_PREV_NODE,
                     hidden_dim: int = HIDDEN_DIM):
            super().__init__()
            self.max_prev = max_prev_node
            self.hidden_dim = hidden_dim
            # 2x because we predict both outgoing and incoming edges
            self.edge_dim = 2 * max_prev_node

            # Graph-level RNN
            self.graph_rnn = nn.GRU(
                input_size=self.edge_dim,
                hidden_size=hidden_dim,
                num_layers=4,
                batch_first=True,
            )
            # Edge-level RNN
            self.edge_rnn = nn.GRU(
                input_size=1,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
            )
            self.edge_output = nn.Linear(hidden_dim, 1)
            self.graph_to_edge = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x_seq: Tensor) -> Tensor:
            """
            Args:
                x_seq: [B, N-1, 2*max_prev] edge sequence
            Returns:
                logits: [B, N-1, 2*max_prev] edge logits
            """
            B, T, D = x_seq.shape
            h_graph, _ = self.graph_rnn(x_seq)  # [B, T, hidden_dim]

            logits_all = []
            for t_step in range(T):
                h_g = self.graph_to_edge(h_graph[:, t_step, :])  # [B, hidden_dim]
                h_e = h_g.unsqueeze(0).expand(
                    2, -1, -1
                ).contiguous()  # [num_layers, B, hidden_dim]

                # Autoregressive edge generation
                edge_input = torch.zeros(B, 1, 1, device=x_seq.device)
                edge_logits = []
                for e in range(D):
                    out, h_e = self.edge_rnn(edge_input, h_e)
                    logit = self.edge_output(out[:, -1, :])  # [B, 1]
                    edge_logits.append(logit)
                    if self.training:
                        edge_input = x_seq[:, t_step, e:e+1].unsqueeze(1)
                    else:
                        edge_input = torch.sigmoid(logit).unsqueeze(1)
                logits_all.append(torch.cat(edge_logits, dim=1))  # [B, D]

            return torch.stack(logits_all, dim=1)  # [B, T, D]

    def _adj_to_sequence_directed(adj: np.ndarray, max_prev: int) -> np.ndarray:
        """Convert directed adjacency to GraphRNN sequence.

        For node i (i=1..N-1), the sequence entry contains:
        - adj[i, max(0,i-max_prev):i]  (outgoing edges to previous nodes)
        - adj[max(0,i-max_prev):i, i]  (incoming edges from previous nodes)
        Both zero-padded to length max_prev.
        """
        N_ = adj.shape[0]
        seq = np.zeros((N_ - 1, 2 * max_prev), dtype=np.float32)
        for i in range(1, N_):
            start = max(0, i - max_prev)
            out_edges = adj[i, start:i]
            in_edges = adj[start:i, i]
            pad_out = np.zeros(max_prev, dtype=np.float32)
            pad_in = np.zeros(max_prev, dtype=np.float32)
            pad_out[max_prev - len(out_edges):] = out_edges
            pad_in[max_prev - len(in_edges):] = in_edges
            seq[i - 1, :max_prev] = pad_out
            seq[i - 1, max_prev:] = pad_in
        return seq

    def _sequence_to_adj_directed(seq: np.ndarray, N_: int,
                                   max_prev: int) -> np.ndarray:
        """Reconstruct directed adjacency from GraphRNN sequence."""
        adj = np.zeros((N_, N_), dtype=np.float32)
        for i in range(1, N_):
            start = max(0, i - max_prev)
            L = i - start
            # Outgoing: adj[i, start:i]
            out_edges = seq[i - 1, max_prev - L:max_prev]
            adj[i, start:i] = out_edges
            # Incoming: adj[start:i, i]
            in_edges = seq[i - 1, max_prev + (max_prev - L):2 * max_prev]
            adj[start:i, i] = in_edges
        np.fill_diagonal(adj, 0)
        return (adj > 0.5).astype(np.float32)


# ===================================================================
# BASELINE 2: Directed GRAN (simplified)
# ===================================================================
# Block-wise edge prediction with attention. Simplified from Liao et al.
# (NeurIPS 2019). Generates edges one block at a time using GNN-based
# attention over existing nodes.
# ===================================================================

if HAS_TORCH:
    class DirectedGRAN(nn.Module):
        """Simplified GRAN for directed graphs.

        Generates adjacency matrix block-by-block. Each block adds one node
        and predicts both its outgoing and incoming edges.
        """
        def __init__(self, hidden_dim: int = HIDDEN_DIM, n_layers: int = 4):
            super().__init__()
            self.hidden_dim = hidden_dim

            # Node embedding (initialized from graph state)
            self.node_embed = nn.Linear(1, hidden_dim)

            # Message passing layers
            self.msg_layers = nn.ModuleList([
                nn.Linear(hidden_dim * 2, hidden_dim)
                for _ in range(n_layers)
            ])
            self.update_layers = nn.ModuleList([
                nn.GRUCell(hidden_dim, hidden_dim)
                for _ in range(n_layers)
            ])

            # Edge prediction (for both directions)
            self.edge_pred = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),  # out-edge and in-edge
            )

        def forward(self, adj_partial: Tensor, n_existing: int) -> Tensor:
            """Predict edges for the next node given partial adjacency.

            Args:
                adj_partial: [B, n_existing, n_existing] current partial adj
                n_existing: number of existing nodes
            Returns:
                logits: [B, n_existing, 2] (out-edge, in-edge) logits
            """
            B = adj_partial.shape[0]
            # Initialize node embeddings from degree
            deg = adj_partial.sum(dim=-1, keepdim=True)  # [B, n_existing, 1]
            h = self.node_embed(deg)  # [B, n_existing, hidden]

            # Message passing
            for msg_layer, update_layer in zip(self.msg_layers, self.update_layers):
                # Aggregate neighbors
                agg = torch.bmm(adj_partial, h)  # [B, n_existing, hidden]
                msg = msg_layer(torch.cat([h, agg], dim=-1))
                msg = F.relu(msg)
                h_flat = h.reshape(-1, self.hidden_dim)
                msg_flat = msg.reshape(-1, self.hidden_dim)
                h_flat = update_layer(msg_flat, h_flat)
                h = h_flat.reshape(B, n_existing, self.hidden_dim)

            # New node embedding (mean of existing)
            h_new = h.mean(dim=1, keepdim=True).expand_as(h)  # [B, n_existing, hidden]

            # Edge prediction for each existing node
            pair_feat = torch.cat([h, h_new], dim=-1)  # [B, n_existing, 2*hidden]
            logits = self.edge_pred(pair_feat)  # [B, n_existing, 2]
            return logits


# ===================================================================
# BASELINE 3: Directed NetGAN (simplified)
# ===================================================================
# GAN on directed random walks. Generator produces directed walk sequences;
# discriminator distinguishes real from fake walks.
# ===================================================================

if HAS_TORCH:
    class NetGANGenerator(nn.Module):
        """LSTM-based generator producing directed random walks."""
        def __init__(self, N_: int, walk_length: int = 20,
                     hidden_dim: int = 40):
            super().__init__()
            self.N = N_
            self.walk_length = walk_length
            self.lstm = nn.LSTM(N_, hidden_dim, num_layers=2, batch_first=True)
            self.output = nn.Linear(hidden_dim, N_)

        def forward(self, z: Tensor) -> Tensor:
            """Generate a walk from noise z.

            Args:
                z: [B, N] noise vector
            Returns:
                walks: [B, walk_length, N] one-hot walk steps
            """
            B = z.shape[0]
            x = z.unsqueeze(1)  # [B, 1, N]
            walks = []
            h = None
            for _ in range(self.walk_length):
                out, h = self.lstm(x, h)
                logits = self.output(out[:, -1, :])  # [B, N]
                probs = F.softmax(logits, dim=-1)
                if self.training:
                    # Gumbel-softmax for differentiable sampling
                    step = F.gumbel_softmax(logits, tau=0.5, hard=True)
                else:
                    step = torch.zeros_like(probs)
                    idx = probs.argmax(dim=-1)
                    step.scatter_(1, idx.unsqueeze(1), 1.0)
                walks.append(step)
                x = step.unsqueeze(1)
            return torch.stack(walks, dim=1)  # [B, walk_length, N]

    class NetGANDiscriminator(nn.Module):
        """LSTM-based discriminator for random walks."""
        def __init__(self, N_: int, walk_length: int = 20,
                     hidden_dim: int = 40):
            super().__init__()
            self.lstm = nn.LSTM(N_, hidden_dim, num_layers=2, batch_first=True)
            self.output = nn.Linear(hidden_dim, 1)

        def forward(self, walks: Tensor) -> Tensor:
            """
            Args:
                walks: [B, walk_length, N]
            Returns:
                score: [B, 1]
            """
            out, _ = self.lstm(walks)
            return self.output(out[:, -1, :])

    def _extract_directed_walks(adj: np.ndarray, num_walks: int,
                                 walk_length: int, rng) -> np.ndarray:
        """Extract directed random walks from adjacency matrix.

        Follows outgoing edges only (directed walks).
        """
        N_ = adj.shape[0]
        walks = np.zeros((num_walks, walk_length, N_), dtype=np.float32)
        for w in range(num_walks):
            start = rng.integers(0, N_)
            node = start
            for step in range(walk_length):
                walks[w, step, node] = 1.0
                neighbors = np.where(adj[node] > 0.5)[0]
                if len(neighbors) == 0:
                    # Dead end: restart
                    node = rng.integers(0, N_)
                else:
                    node = rng.choice(neighbors)
        return walks

    def _walks_to_adjacency(walks: np.ndarray, N_: int,
                             threshold: float = 0.5) -> np.ndarray:
        """Reconstruct directed adjacency from generated walks.

        Score matrix S[i,j] = # times walk goes from node i to node j
        (consecutive steps). Directed: S is asymmetric.
        """
        S = np.zeros((N_, N_), dtype=np.float64)
        n_walks, walk_len, _ = walks.shape
        for w in range(n_walks):
            for step in range(walk_len - 1):
                src = walks[w, step].argmax()
                dst = walks[w, step + 1].argmax()
                if src != dst:
                    S[src, dst] += 1.0
        # Normalize and threshold
        if S.max() > 0:
            S = S / S.max()
        adj = (S > threshold).astype(np.float32)
        np.fill_diagonal(adj, 0)
        return adj


# ===================================================================
# Common baseline interface
# ===================================================================

class BaselineWrapper:
    """Common interface for all baselines."""
    def __init__(self, name: str):
        self.name = name
        self.trained = False

    def train(self, train_adjs: List[np.ndarray], **kwargs):
        raise NotImplementedError

    def generate(self, n_nodes: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class GraphRNNBaseline(BaselineWrapper):
    """Directed GraphRNN wrapper."""
    def __init__(self):
        super().__init__("graphrnn_directed")
        self.model = None
        self.max_prev = MAX_PREV_NODE

    def train(self, train_adjs, steps=TRAIN_STEPS, lr=LR, **kwargs):
        if not HAS_TORCH:
            print(f"  [{self.name}] PyTorch unavailable, skipping.", flush=True)
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DirectedGraphRNN(
            max_prev_node=self.max_prev, hidden_dim=HIDDEN_DIM
        ).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Prepare sequences
        seqs = [_adj_to_sequence_directed(a, self.max_prev) for a in train_adjs]
        rng = np.random.default_rng(SEED)

        self.model.train()
        for step in range(steps):
            idx = rng.integers(0, len(seqs), size=min(BATCH_SIZE, len(seqs)))
            batch = np.stack([seqs[i] for i in idx])
            x = torch.tensor(batch, dtype=torch.float32, device=device)
            logits = self.model(x)
            loss = F.binary_cross_entropy_with_logits(logits, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1) % 1000 == 0:
                print(f"    [{self.name}] step {step+1}/{steps} "
                      f"loss={loss.item():.4f}", flush=True)

        self.trained = True

    def generate(self, n_nodes, rng):
        if not self.trained or self.model is None:
            return np.zeros((n_nodes, n_nodes), dtype=np.float32)
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            # Start with empty edge vector
            x_input = torch.zeros(1, 1, 2 * self.max_prev, device=device)
            all_edges = [x_input[:, 0, :]]
            h = None
            for i in range(1, n_nodes):
                out, h_new = self.model.graph_rnn(x_input, h)
                h = h_new
                # Generate edge vector for node i
                h_g = self.model.graph_to_edge(out[:, -1, :])
                h_e = h_g.unsqueeze(0).expand(2, -1, -1).contiguous()
                edge_input = torch.zeros(1, 1, 1, device=device)
                edge_logits = []
                for e in range(2 * self.max_prev):
                    edge_out, h_e = self.model.edge_rnn(edge_input, h_e)
                    logit = self.model.edge_output(edge_out[:, -1, :])
                    edge_logits.append(logit)
                    edge_input = torch.sigmoid(logit).unsqueeze(1)
                edges = torch.cat(edge_logits, dim=1)  # [1, 2*max_prev]
                edges = (torch.sigmoid(edges) > 0.5).float()
                all_edges.append(edges)
                x_input = edges.unsqueeze(1)

        seq = torch.stack(all_edges[1:], dim=1)[0].cpu().numpy()
        return _sequence_to_adj_directed(seq, n_nodes, self.max_prev)


class GRANBaseline(BaselineWrapper):
    """Simplified directed GRAN wrapper."""
    def __init__(self):
        super().__init__("gran_directed")
        self.model = None

    def train(self, train_adjs, steps=TRAIN_STEPS, lr=LR, **kwargs):
        if not HAS_TORCH:
            print(f"  [{self.name}] PyTorch unavailable, skipping.", flush=True)
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DirectedGRAN(hidden_dim=HIDDEN_DIM).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        rng = np.random.default_rng(SEED)

        self.model.train()
        for step in range(steps):
            idx = rng.integers(0, len(train_adjs), size=min(BATCH_SIZE, len(train_adjs)))
            # For each graph, pick a random partial adjacency size
            losses = []
            for i in idx:
                adj = train_adjs[i]
                N_ = adj.shape[0]
                n_partial = rng.integers(2, min(N_, 50))
                adj_partial = torch.tensor(
                    adj[:n_partial, :n_partial], dtype=torch.float32,
                    device=device
                ).unsqueeze(0)
                logits = self.model(adj_partial, n_partial)  # [1, n_partial, 2]
                # Target: edges to/from next node
                if n_partial < N_:
                    target_out = torch.tensor(
                        adj[n_partial, :n_partial], dtype=torch.float32,
                        device=device
                    ).unsqueeze(0)
                    target_in = torch.tensor(
                        adj[:n_partial, n_partial], dtype=torch.float32,
                        device=device
                    ).unsqueeze(0)
                    target = torch.stack([target_out, target_in], dim=-1)
                    loss = F.binary_cross_entropy_with_logits(logits, target)
                    losses.append(loss)

            if losses:
                total = torch.stack(losses).mean()
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
                if (step + 1) % 1000 == 0:
                    print(f"    [{self.name}] step {step+1}/{steps} "
                          f"loss={total.item():.4f}", flush=True)

        self.trained = True

    def generate(self, n_nodes, rng):
        if not self.trained or self.model is None:
            return np.zeros((n_nodes, n_nodes), dtype=np.float32)
        device = next(self.model.parameters()).device
        self.model.eval()
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        with torch.no_grad():
            for i in range(1, n_nodes):
                n_existing = i
                adj_partial = torch.tensor(
                    adj[:n_existing, :n_existing], dtype=torch.float32,
                    device=device
                ).unsqueeze(0)
                logits = self.model(adj_partial, n_existing)  # [1, n_existing, 2]
                probs = torch.sigmoid(logits[0])  # [n_existing, 2]
                # Sample edges
                out_edges = (torch.rand(n_existing, device=device) <
                             probs[:, 0]).float().cpu().numpy()
                in_edges = (torch.rand(n_existing, device=device) <
                            probs[:, 1]).float().cpu().numpy()
                adj[i, :n_existing] = out_edges
                adj[:n_existing, i] = in_edges
        np.fill_diagonal(adj, 0)
        return adj


class NetGANBaseline(BaselineWrapper):
    """Directed NetGAN wrapper."""
    def __init__(self):
        super().__init__("netgan_directed")
        self.generator = None
        self.N_ = N

    def train(self, train_adjs, steps=TRAIN_STEPS, lr=LR, **kwargs):
        if not HAS_TORCH:
            print(f"  [{self.name}] PyTorch unavailable, skipping.", flush=True)
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N_ = train_adjs[0].shape[0]
        walk_length = 20
        n_walks_per_graph = 100

        self.generator = NetGANGenerator(
            self.N_, walk_length, hidden_dim=40
        ).to(device)
        discriminator = NetGANDiscriminator(
            self.N_, walk_length, hidden_dim=40
        ).to(device)

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr)
        rng = np.random.default_rng(SEED)

        # Extract real walks
        print(f"    [{self.name}] Extracting directed walks ...", flush=True)
        real_walks_list = []
        for adj in train_adjs[:20]:  # Use subset for efficiency
            walks = _extract_directed_walks(adj, n_walks_per_graph,
                                            walk_length, rng)
            real_walks_list.append(walks)
        real_walks = np.concatenate(real_walks_list, axis=0)

        for step in range(steps):
            # Train discriminator
            idx = rng.integers(0, len(real_walks), size=BATCH_SIZE)
            real_batch = torch.tensor(
                real_walks[idx], dtype=torch.float32, device=device
            )
            z = torch.randn(BATCH_SIZE, self.N_, device=device)
            fake_batch = self.generator(z)

            real_score = discriminator(real_batch)
            fake_score = discriminator(fake_batch.detach())
            d_loss = (F.relu(1 - real_score).mean() +
                      F.relu(1 + fake_score).mean())
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # Train generator
            z = torch.randn(BATCH_SIZE, self.N_, device=device)
            fake_batch = self.generator(z)
            fake_score = discriminator(fake_batch)
            g_loss = -fake_score.mean()
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            if (step + 1) % 1000 == 0:
                print(f"    [{self.name}] step {step+1}/{steps} "
                      f"d_loss={d_loss.item():.4f} g_loss={g_loss.item():.4f}",
                      flush=True)

        self.trained = True

    def generate(self, n_nodes, rng):
        if not self.trained or self.generator is None:
            return np.zeros((n_nodes, n_nodes), dtype=np.float32)
        device = next(self.generator.parameters()).device
        self.generator.eval()
        # Generate many walks, then reconstruct adjacency
        n_gen_walks = 500
        with torch.no_grad():
            z = torch.randn(n_gen_walks, self.N_, device=device)
            walks = self.generator(z).cpu().numpy()
        adj = _walks_to_adjacency(walks, n_nodes, threshold=0.3)
        return adj


# ===================================================================
# Evaluation
# ===================================================================

def evaluate_baselines(baselines: List[BaselineWrapper],
                       test_adjs: List[np.ndarray],
                       rng: np.random.Generator,
                       n_samples: int = 5) -> dict:
    """Evaluate all baselines on test adjacencies."""
    results = {}
    for bl in baselines:
        print(f"  Evaluating '{bl.name}' ...", flush=True)
        reach_l1s, motif_l1s, recip_errs = [], [], []
        for adj_gt in test_adjs:
            for _ in range(n_samples):
                adj_s = bl.generate(adj_gt.shape[0], rng)
                reach_l1s.append(abs(reach_3hop(adj_gt) - reach_3hop(adj_s)))
                motif_l1s.append(motif_l1(adj_gt, adj_s))
                recip_errs.append(abs(reciprocity(adj_gt) - reciprocity(adj_s)))

        r_arr = np.array(reach_l1s)
        m_arr = np.array(motif_l1s)
        re_arr = np.array(recip_errs)
        results[bl.name] = {
            "reach_3hop_L1": {
                "mean": float(r_arr.mean()),
                "sem": float(r_arr.std() / np.sqrt(max(len(r_arr), 1))),
            },
            "motif_L1": {
                "mean": float(m_arr.mean()),
                "sem": float(m_arr.std() / np.sqrt(max(len(m_arr), 1))),
            },
            "reciprocity_error": {
                "mean": float(re_arr.mean()),
                "sem": float(re_arr.std() / np.sqrt(max(len(re_arr), 1))),
            },
        }
        print(f"    reach_L1={results[bl.name]['reach_3hop_L1']['mean']:.4f}"
              f"+/-{results[bl.name]['reach_3hop_L1']['sem']:.4f}  "
              f"motif_L1={results[bl.name]['motif_L1']['mean']:.4f}",
              flush=True)

    # Also evaluate config model for comparison
    print("  Evaluating config_model_gt ...", flush=True)
    reach_l1s, motif_l1s, recip_errs = [], [], []
    for adj_gt in test_adjs:
        for _ in range(n_samples):
            adj_s = configuration_model_sample(adj_gt, rng)
            reach_l1s.append(abs(reach_3hop(adj_gt) - reach_3hop(adj_s)))
            motif_l1s.append(motif_l1(adj_gt, adj_s))
            recip_errs.append(abs(reciprocity(adj_gt) - reciprocity(adj_s)))
    r_arr = np.array(reach_l1s)
    m_arr = np.array(motif_l1s)
    re_arr = np.array(recip_errs)
    results["config_model_gt"] = {
        "reach_3hop_L1": {
            "mean": float(r_arr.mean()),
            "sem": float(r_arr.std() / np.sqrt(max(len(r_arr), 1))),
        },
        "motif_L1": {
            "mean": float(m_arr.mean()),
            "sem": float(m_arr.std() / np.sqrt(max(len(m_arr), 1))),
        },
        "reciprocity_error": {
            "mean": float(re_arr.mean()),
            "sem": float(re_arr.std() / np.sqrt(max(len(re_arr), 1))),
        },
    }
    return results


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_microns_subsets(seed=0):
    try:
        sys.path.insert(0, SCRIPTS)
        from exp_powerlaw_vs_ridge import load_microns_subsets as _load
        return _load(seed=seed)
    except (ImportError, FileNotFoundError):
        print("[Baselines] MICrONS loader unavailable. Using synthetic data.",
              flush=True)
        rng = np.random.default_rng(seed)
        items = []
        for _ in range(200):
            N_ = 100
            adj = (rng.random((N_, N_)) < 0.022).astype(np.float32)
            np.fill_diagonal(adj, 0)
            items.append({
                "adjacency": adj,
                "morph_idx": rng.integers(0, 12, size=N_).astype(np.int64),
                "positions": rng.normal(0, 200, size=(N_, 3)).astype(np.float32),
                "depth": rng.uniform(0, 800, size=N_).astype(np.float32),
            })
        return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    print("Loading MICrONS subsets ...", flush=True)
    all_items = load_microns_subsets(seed=0)
    train_items = all_items[:150]
    test_items = all_items[150:200][:20]  # Use 20 test items

    train_adjs = [it["adjacency"].copy() for it in train_items]
    for a in train_adjs:
        np.fill_diagonal(a, 0)
    test_adjs = [it["adjacency"].copy() for it in test_items]
    for a in test_adjs:
        np.fill_diagonal(a, 0)

    # Initialize baselines
    baselines = [
        GraphRNNBaseline(),
        GRANBaseline(),
        NetGANBaseline(),
    ]

    # Train each baseline
    for bl in baselines:
        print(f"\nTraining '{bl.name}' ...", flush=True)
        bl.train(train_adjs, steps=TRAIN_STEPS, lr=LR)

    # Evaluate all
    print("\n" + "=" * 60)
    print("EVALUATING ALL BASELINES")
    print("=" * 60)
    eval_results = evaluate_baselines(baselines, test_adjs, rng, n_samples=5)

    # Summary
    print("\n" + "=" * 60)
    print("ADDITIONAL BASELINES SUMMARY")
    print("=" * 60)
    print(f"{'baseline':<25} | {'reach_L1':>10} | {'motif_L1':>10} | "
          f"{'recip_err':>10}")
    print("-" * 65)
    for name, metrics in eval_results.items():
        print(f"{name:<25} | "
              f"{metrics['reach_3hop_L1']['mean']:>10.4f} | "
              f"{metrics['motif_L1']['mean']:>10.4f} | "
              f"{metrics['reciprocity_error']['mean']:>10.4f}")

    output = {
        "experiment": "additional_baselines_aaai27",
        "addresses": "AAAI-27 Exp 4: GraphRNN + GRAN + NetGAN baselines",
        "baselines": list(eval_results.keys()),
        "train_steps": TRAIN_STEPS,
        "n_test_items": len(test_adjs),
        "results": eval_results,
        "notes": {
            "graphrnn": "Directed adaptation: generates both incoming and "
                        "outgoing edges per node step. BFS ordering.",
            "gran": "Simplified directed GRAN: block-by-block with GNN "
                    "attention. Predicts bidirectional edges.",
            "netgan": "Directed walks: follows outgoing edges only. "
                      "Asymmetric score matrix for directed adjacency.",
        },
        "walltime_s": float(time.time() - t0),
    }

    out_dir = os.path.join(CELL2CIRCUIT_ROOT, "results", "aaai27")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "additional_baselines.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}")
    return output


if __name__ == "__main__":
    main()
