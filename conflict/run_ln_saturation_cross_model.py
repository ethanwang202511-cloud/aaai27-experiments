#!/usr/bin/env python3
"""
Experiment 2: Cross-Model LayerNorm/RMSNorm Saturation Probe
=============================================================
AAAI-27 submission — ports GPT-2 LN saturation findings to:
  - Gemma-2-2B  (RMSNorm pre+post, 26 layers)
  - Llama-3.1-8B (RMSNorm pre only, 32 layers)

Baseline (GPT-2, Experiment 4b):
  Pearson r(delta_scale_zero, |metric_drop_zero|) = 0.876
  Pearson r(delta_scale_mean, |metric_drop_mean|) = 0.099

Methodology per model, per MLP block L:
  1. Clean forward  -> capture ln_final.hook_scale at last-token position
  2. Zero-ablated   -> hook_mlp_out := 0, capture hook_scale
  3. Mean-ablated   -> hook_mlp_out := dataset mean, capture hook_scale
  4. delta_scale     = |scale_ablated - scale_clean|
  5. metric_drop     = logit_diff_clean - logit_diff_ablated

Analysis: Pearson r, paired t-test (zero vs mean delta_scale).

Usage:
  python run_ln_saturation_cross_model.py --model gemma2
  python run_ln_saturation_cross_model.py --model llama3
  python run_ln_saturation_cross_model.py --model all --n-prompts 200

Note: Llama-3.1-8B requires ~16GB VRAM. Gemma-2-2B fits in ~6GB.
      Runs locally — no Modal, no SLURM.
"""

import argparse
import gc
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy import stats

# ---------------------------------------------------------------------------
# IOI prompt generation
# ---------------------------------------------------------------------------

NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Leo", "Mary", "Nathan", "Olivia", "Peter",
    "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
    "Yolanda", "Zach",
]

PLACES = [
    "store", "park", "library", "cafe", "beach", "museum", "school",
    "hospital", "restaurant", "office", "garden", "market", "theater",
    "gym", "church", "station", "airport", "hotel", "zoo", "bank",
]


@dataclass
class IOIPrompt:
    text: str
    io_name: str   # indirect object — correct answer
    s_name: str    # subject — incorrect answer


def generate_ioi_prompts(n: int, seed: int = 42) -> list[IOIPrompt]:
    """Generate n IOI prompts with the standard template.

    Template: "When {IO} and {S} went to the {place}, {IO} gave a gift to"
    Correct completion: S (the subject receives the gift)
    Wait — standard IOI: IO is repeated, correct answer is IO.
    Actually in standard IOI:
      "When Mary and John went to the store, John gave a drink to"
      Answer: Mary (IO). The repeated name is the Subject (S=John).
      name1 = IO (Mary), name2 = S (John), name3 = S (John, repeated).
      Correct = IO = Mary.

    Revised per standard IOI:
      "When {name1} and {name2} went to the {place}, {name3} gave a gift to"
      where name2 = name3 (S is repeated), correct = name1 (IO).
    """
    rng = random.Random(seed)
    prompts = []
    for _ in range(n):
        io_name, s_name = rng.sample(NAMES, 2)
        place = rng.choice(PLACES)
        text = f"When {io_name} and {s_name} went to the {place}, {s_name} gave a gift to"
        prompts.append(IOIPrompt(text=text, io_name=io_name, s_name=s_name))
    return prompts


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str               # display name
    tl_name: str            # TransformerLens model string
    n_layers: int
    default_batch_size: int
    norm_type: str          # "rmsnorm_pre_post" or "rmsnorm_pre"
    scale_hook: str = "ln_final.hook_scale"


MODEL_CONFIGS = {
    "gemma2": ModelConfig(
        name="Gemma-2-2B",
        tl_name="gemma-2-2b",
        n_layers=26,
        default_batch_size=16,
        norm_type="rmsnorm_pre_post",
    ),
    "llama3": ModelConfig(
        name="Llama-3.1-8B",
        tl_name="meta-llama/Llama-3.1-8B",
        n_layers=32,
        default_batch_size=8,
        norm_type="rmsnorm_pre",
    ),
}


# ---------------------------------------------------------------------------
# Device utilities
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_cache(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_experiment(
    model_key: str,
    prompts: list[IOIPrompt],
    batch_size: Optional[int] = None,
    output_dir: Path = Path("."),
):
    """Run the LN saturation probe for one model."""
    from transformer_lens import HookedTransformer

    cfg = MODEL_CONFIGS[model_key]
    device = get_device()
    bs = batch_size or cfg.default_batch_size

    print(f"\n{'='*70}")
    print(f"  Model: {cfg.name} ({cfg.tl_name})")
    print(f"  Device: {device}")
    print(f"  Prompts: {len(prompts)}, Batch size: {bs}")
    print(f"  Layers: {cfg.n_layers}, Norm: {cfg.norm_type}")
    print(f"{'='*70}\n")

    # ---- Load model ----
    t0 = time.time()
    print(f"Loading {cfg.name}...")
    model = HookedTransformer.from_pretrained(
        cfg.tl_name,
        device=device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ---- Tokenize ----
    print("Tokenizing prompts...")
    texts = [p.text for p in prompts]

    # Get token IDs for IO and S names (first token of each name)
    io_token_ids = []
    s_token_ids = []
    for p in prompts:
        io_toks = model.to_tokens(f" {p.io_name}", prepend_bos=False)[0]
        s_toks = model.to_tokens(f" {p.s_name}", prepend_bos=False)[0]
        io_token_ids.append(io_toks[0].item())
        s_token_ids.append(s_toks[0].item())

    io_token_ids = torch.tensor(io_token_ids, device=device)
    s_token_ids = torch.tensor(s_token_ids, device=device)

    # ---- Phase 1: Clean forward pass — collect scales and logit diffs ----
    print("\nPhase 1: Clean forward passes...")
    clean_scales = []        # (n_prompts,) — ln_final scale at last token
    clean_logit_diffs = []   # (n_prompts,)

    for batch_start in range(0, len(texts), bs):
        batch_texts = texts[batch_start : batch_start + bs]
        batch_io = io_token_ids[batch_start : batch_start + bs]
        batch_s = s_token_ids[batch_start : batch_start + bs]

        scale_cache = {}

        def capture_scale_hook(value, hook):
            scale_cache["scale"] = value.detach().clone()
            return value

        with torch.no_grad():
            logits = model.run_with_hooks(
                batch_texts,
                fwd_hooks=[(cfg.scale_hook, capture_scale_hook)],
            )

        # Last token logits
        # logits shape: (batch, seq, vocab). We need last non-pad token.
        # Since prompts have different lengths after tokenization, we use
        # the model's tokenizer to find sequence lengths.
        tokens = model.to_tokens(batch_texts)
        seq_lens = (tokens != model.tokenizer.pad_token_id).sum(dim=1) if model.tokenizer.pad_token_id is not None else torch.full((tokens.shape[0],), tokens.shape[1], device=device)

        for i in range(len(batch_texts)):
            last_pos = seq_lens[i].item() - 1
            last_logits = logits[i, last_pos]
            io_logit = last_logits[batch_io[i]].item()
            s_logit = last_logits[batch_s[i]].item()
            clean_logit_diffs.append(io_logit - s_logit)

            # Scale at last token position
            # hook_scale shape varies by model; typically (batch, seq, 1) or (batch, seq, 1, 1)
            sc = scale_cache["scale"]
            if sc.dim() == 3:
                clean_scales.append(sc[i, last_pos, 0].item())
            elif sc.dim() == 4:
                clean_scales.append(sc[i, last_pos, 0, 0].item())
            else:
                clean_scales.append(sc[i, last_pos].item())

        clear_cache(device)
        print(f"  Batch {batch_start//bs + 1}/{(len(texts) + bs - 1)//bs} done")

    clean_scales = np.array(clean_scales)
    clean_logit_diffs = np.array(clean_logit_diffs)
    print(f"  Mean clean logit diff: {clean_logit_diffs.mean():.4f}")
    print(f"  Mean clean scale: {clean_scales.mean():.4f}")

    # ---- Phase 2: Compute per-position mean MLP outputs for mean ablation ----
    print("\nPhase 2: Computing mean MLP activations...")
    # We accumulate the mean of hook_mlp_out across prompts, per layer.
    # Shape per layer: (max_seq, d_model).
    # We'll do a running mean.

    tokens_all = model.to_tokens(texts)
    max_seq = tokens_all.shape[1]
    d_model = model.cfg.d_model

    mlp_means = {}  # layer -> (max_seq, d_model)
    counts = 0

    for batch_start in range(0, len(texts), bs):
        batch_texts = texts[batch_start : batch_start + bs]
        batch_tokens = model.to_tokens(batch_texts)
        cur_seq = batch_tokens.shape[1]
        cur_bs = batch_tokens.shape[0]

        names_filter = [f"blocks.{L}.hook_mlp_out" for L in range(cfg.n_layers)]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch_texts,
                names_filter=names_filter,
            )

        for L in range(cfg.n_layers):
            act = cache[f"blocks.{L}.hook_mlp_out"]  # (bs, seq, d_model)
            act_sum = act.sum(dim=0).cpu().float()    # (seq, d_model)

            if L not in mlp_means:
                mlp_means[L] = torch.zeros(cur_seq, d_model)

            # Handle potential sequence length mismatch by padding
            if act_sum.shape[0] < mlp_means[L].shape[0]:
                padded = torch.zeros_like(mlp_means[L])
                padded[:act_sum.shape[0]] = act_sum
                act_sum = padded
            elif act_sum.shape[0] > mlp_means[L].shape[0]:
                old = mlp_means[L]
                mlp_means[L] = torch.zeros(act_sum.shape[0], d_model)
                mlp_means[L][:old.shape[0]] = old * counts

                # Re-scale to undo the mean so we can add
                mlp_means[L][:old.shape[0]] /= max(counts, 1)
                # Actually let's just track sums and divide at the end
                # Restart with sums approach

        counts += cur_bs
        del cache
        clear_cache(device)

    # Simpler approach: recompute means with a second pass tracking sums
    print("  (Recomputing with sum-based approach...)")
    mlp_sums = {}
    total_prompts = 0

    for batch_start in range(0, len(texts), bs):
        batch_texts = texts[batch_start : batch_start + bs]
        names_filter = [f"blocks.{L}.hook_mlp_out" for L in range(cfg.n_layers)]

        with torch.no_grad():
            _, cache = model.run_with_cache(batch_texts, names_filter=names_filter)

        cur_bs = cache[f"blocks.0.hook_mlp_out"].shape[0]

        for L in range(cfg.n_layers):
            act = cache[f"blocks.{L}.hook_mlp_out"].cpu().float()  # (bs, seq, d_model)
            batch_sum = act.sum(dim=0)  # (seq, d_model)

            if L not in mlp_sums:
                mlp_sums[L] = torch.zeros_like(batch_sum)

            # Pad to match sizes
            s1, s2 = mlp_sums[L].shape[0], batch_sum.shape[0]
            if s2 > s1:
                new = torch.zeros(s2, d_model)
                new[:s1] = mlp_sums[L]
                mlp_sums[L] = new
            elif s1 > s2:
                padded = torch.zeros(s1, d_model)
                padded[:s2] = batch_sum
                batch_sum = padded

            mlp_sums[L] += batch_sum

        total_prompts += cur_bs
        del cache
        clear_cache(device)

    # Convert sums to means
    for L in range(cfg.n_layers):
        mlp_sums[L] /= total_prompts  # now these are means

    mlp_mean_acts = mlp_sums  # rename for clarity
    print(f"  Computed mean MLP activations over {total_prompts} prompts")

    # ---- Phase 3: Ablation passes ----
    print("\nPhase 3: Ablation passes...")

    # Results storage: per layer
    results_per_layer = {}

    for L in range(cfg.n_layers):
        t_layer = time.time()
        layer_delta_scale_zero = []
        layer_delta_scale_mean = []
        layer_logit_diff_zero = []
        layer_logit_diff_mean = []

        mean_act = mlp_mean_acts[L].to(device).to(
            torch.float16 if device.type == "cuda" else torch.float32
        )

        for ablation_type in ["zero", "mean"]:
            abl_scales = []
            abl_logit_diffs = []

            for batch_start in range(0, len(texts), bs):
                batch_texts = texts[batch_start : batch_start + bs]
                batch_io = io_token_ids[batch_start : batch_start + bs]
                batch_s = s_token_ids[batch_start : batch_start + bs]

                scale_cache = {}

                def make_ablation_hook(abl_type, mean_activation):
                    def hook_fn(value, hook):
                        if abl_type == "zero":
                            return torch.zeros_like(value)
                        else:
                            # Replace with mean activation (broadcast over batch)
                            seq_len = value.shape[1]
                            ma = mean_activation[:seq_len].unsqueeze(0)
                            return ma.expand_as(value)
                    return hook_fn

                def capture_scale(value, hook):
                    scale_cache["scale"] = value.detach().clone()
                    return value

                ablation_hook = make_ablation_hook(ablation_type, mean_act)

                with torch.no_grad():
                    logits = model.run_with_hooks(
                        batch_texts,
                        fwd_hooks=[
                            (f"blocks.{L}.hook_mlp_out", ablation_hook),
                            (cfg.scale_hook, capture_scale),
                        ],
                    )

                tokens = model.to_tokens(batch_texts)
                seq_lens = (
                    (tokens != model.tokenizer.pad_token_id).sum(dim=1)
                    if model.tokenizer.pad_token_id is not None
                    else torch.full((tokens.shape[0],), tokens.shape[1], device=device)
                )

                for i in range(len(batch_texts)):
                    last_pos = seq_lens[i].item() - 1
                    last_logits = logits[i, last_pos]
                    io_logit = last_logits[batch_io[i]].item()
                    s_logit = last_logits[batch_s[i]].item()
                    abl_logit_diffs.append(io_logit - s_logit)

                    sc = scale_cache["scale"]
                    if sc.dim() == 3:
                        abl_scales.append(sc[i, last_pos, 0].item())
                    elif sc.dim() == 4:
                        abl_scales.append(sc[i, last_pos, 0, 0].item())
                    else:
                        abl_scales.append(sc[i, last_pos].item())

                clear_cache(device)

            abl_scales = np.array(abl_scales)
            abl_logit_diffs = np.array(abl_logit_diffs)

            if ablation_type == "zero":
                layer_delta_scale_zero = np.abs(abl_scales - clean_scales)
                layer_logit_diff_zero = clean_logit_diffs - abl_logit_diffs
            else:
                layer_delta_scale_mean = np.abs(abl_scales - clean_scales)
                layer_logit_diff_mean = clean_logit_diffs - abl_logit_diffs

        results_per_layer[L] = {
            "mean_delta_scale_zero": float(np.mean(layer_delta_scale_zero)),
            "mean_delta_scale_mean": float(np.mean(layer_delta_scale_mean)),
            "mean_metric_drop_zero": float(np.mean(np.abs(layer_logit_diff_zero))),
            "mean_metric_drop_mean": float(np.mean(np.abs(layer_logit_diff_mean))),
            "delta_scale_zero_all": layer_delta_scale_zero.tolist(),
            "delta_scale_mean_all": layer_delta_scale_mean.tolist(),
            "metric_drop_zero_all": layer_logit_diff_zero.tolist(),
            "metric_drop_mean_all": layer_logit_diff_mean.tolist(),
        }

        elapsed = time.time() - t_layer
        print(
            f"  Layer {L:2d}/{cfg.n_layers-1}: "
            f"d_scale_zero={np.mean(layer_delta_scale_zero):.4f}  "
            f"d_scale_mean={np.mean(layer_delta_scale_mean):.4f}  "
            f"|drop_zero|={np.mean(np.abs(layer_logit_diff_zero)):.4f}  "
            f"|drop_mean|={np.mean(np.abs(layer_logit_diff_mean)):.4f}  "
            f"({elapsed:.1f}s)"
        )

    # ---- Phase 4: Aggregate analysis ----
    print(f"\n{'='*70}")
    print(f"  Analysis: {cfg.name}")
    print(f"{'='*70}\n")

    # Vectors across layers: mean delta_scale and mean |metric_drop|
    n_layers = cfg.n_layers
    mean_ds_zero = np.array([results_per_layer[L]["mean_delta_scale_zero"] for L in range(n_layers)])
    mean_ds_mean = np.array([results_per_layer[L]["mean_delta_scale_mean"] for L in range(n_layers)])
    mean_md_zero = np.array([results_per_layer[L]["mean_metric_drop_zero"] for L in range(n_layers)])
    mean_md_mean = np.array([results_per_layer[L]["mean_metric_drop_mean"] for L in range(n_layers)])

    # Pearson correlations (across components/layers)
    r_zero, p_zero = stats.pearsonr(mean_ds_zero, mean_md_zero)
    r_mean, p_mean = stats.pearsonr(mean_ds_mean, mean_md_mean)

    # Paired t-test: delta_scale_zero vs delta_scale_mean across layers
    t_stat, t_pval = stats.ttest_rel(mean_ds_zero, mean_ds_mean)

    print(f"  Pearson r (zero ablation):  r={r_zero:.4f}, p={p_zero:.2e}")
    print(f"  Pearson r (mean ablation):  r={r_mean:.4f}, p={p_mean:.2e}")
    print(f"  GPT-2 baseline:             r_zero=0.876, r_mean=0.099")
    print(f"  Paired t-test (zero vs mean delta_scale): t={t_stat:.4f}, p={t_pval:.2e}")
    print(f"  Mean delta_scale_zero: {mean_ds_zero.mean():.6f}")
    print(f"  Mean delta_scale_mean: {mean_ds_mean.mean():.6f}")

    # ---- Save results ----
    output = {
        "model": cfg.name,
        "tl_name": cfg.tl_name,
        "n_layers": cfg.n_layers,
        "norm_type": cfg.norm_type,
        "n_prompts": len(prompts),
        "device": str(device),
        "summary": {
            "pearson_r_zero": float(r_zero),
            "pearson_p_zero": float(p_zero),
            "pearson_r_mean": float(r_mean),
            "pearson_p_mean": float(p_mean),
            "paired_ttest_t": float(t_stat),
            "paired_ttest_p": float(t_pval),
            "mean_delta_scale_zero": float(mean_ds_zero.mean()),
            "mean_delta_scale_mean": float(mean_ds_mean.mean()),
            "gpt2_baseline_r_zero": 0.876,
            "gpt2_baseline_r_mean": 0.099,
        },
        "per_layer": {
            str(L): {
                "mean_delta_scale_zero": results_per_layer[L]["mean_delta_scale_zero"],
                "mean_delta_scale_mean": results_per_layer[L]["mean_delta_scale_mean"],
                "mean_metric_drop_zero": results_per_layer[L]["mean_metric_drop_zero"],
                "mean_metric_drop_mean": results_per_layer[L]["mean_metric_drop_mean"],
            }
            for L in range(n_layers)
        },
    }

    out_path = output_dir / f"saturation_probe_{model_key}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # Cleanup
    del model
    clear_cache(device)

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp 2: Cross-Model LN/RMSNorm Saturation Probe"
    )
    parser.add_argument(
        "--model",
        choices=["gemma2", "llama3", "all"],
        default="gemma2",
        help="Which model to run (default: gemma2)",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=200,
        help="Number of IOI prompts (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override default batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same dir as script)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = generate_ioi_prompts(args.n_prompts, seed=args.seed)
    print(f"Generated {len(prompts)} IOI prompts")
    print(f"  Example: '{prompts[0].text}' -> IO={prompts[0].io_name}, S={prompts[0].s_name}")

    models_to_run = (
        ["gemma2", "llama3"] if args.model == "all" else [args.model]
    )

    all_results = {}
    for model_key in models_to_run:
        result = run_experiment(
            model_key=model_key,
            prompts=prompts,
            batch_size=args.batch_size,
            output_dir=output_dir,
        )
        all_results[model_key] = result

    # ---- Cross-model comparison ----
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("  Cross-Model Comparison")
        print(f"{'='*70}\n")
        print(f"  {'Model':<20} {'r_zero':>10} {'r_mean':>10} {'t-stat':>10} {'t-pval':>12}")
        print(f"  {'-'*62}")
        print(f"  {'GPT-2 (baseline)':<20} {'0.876':>10} {'0.099':>10} {'—':>10} {'—':>12}")
        for key, res in all_results.items():
            s = res["summary"]
            print(
                f"  {res['model']:<20} "
                f"{s['pearson_r_zero']:>10.4f} "
                f"{s['pearson_r_mean']:>10.4f} "
                f"{s['paired_ttest_t']:>10.4f} "
                f"{s['paired_ttest_p']:>12.2e}"
            )

    # Save combined results
    if len(all_results) > 1:
        combined_path = output_dir / "saturation_probe_combined.json"
        combined = {
            "gpt2_baseline": {"pearson_r_zero": 0.876, "pearson_r_mean": 0.099},
        }
        for key, res in all_results.items():
            combined[key] = res
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\n  Combined results: {combined_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
