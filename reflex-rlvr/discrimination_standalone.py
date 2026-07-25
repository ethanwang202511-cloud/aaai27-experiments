"""Modal-free discrimination probe library for AAAI-27 RLVR experiments.

Provides corruption primitives, vLLM-based scoring helpers, and three
discrimination probes (consistent-corruption, verdict, per-token surprise)
that quantify whether a language model can distinguish correct from
corrupted chain-of-thought solutions.

All heavy dependencies (vLLM, torch, scipy) are imported lazily inside the
functions that need them.  The corruption helpers are pure Python / stdlib.
There are ZERO Modal dependencies in this file.
"""

from __future__ import annotations

import json
import math
import re
import random
from pathlib import Path
from typing import Any

__all__ = [
    # Corruption primitives
    "corrupt_solution_consistent",
    "corrupt_solution_random_answer",
    "shuffle_solution_steps",
    "corrupt_solution_one_step",
    # Data loading
    "load_problems",
    # vLLM helpers
    "create_llm",
    "score_answer",
    "score_verdict_token",
    "verdict_logit",
    # Prompt builders
    "build_cot_prompt",
    "build_verdict_prompt",
    # Probes
    "run_consistent_corruption_probe",
    "run_verdict_probe",
    "run_per_token_surprise_probe",
    # Cost estimation
    "estimate_cost",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DELTAS = [1, 2, 3, 5, 7, -1, -2, -3, -5, -7]

_INTERMEDIATE_RE = re.compile(r"\b(\d+)\b")
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")

_GPU_RATES: dict[str, float] = {
    "H100": 2.39,
    "A100-80GB": 1.19,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nan_to_none(v: float | None) -> float | None:
    """Convert NaN to None for JSON-safe serialization."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Recursively replace NaN values with None in a dict."""
    return {k: _nan_to_none(v) if isinstance(v, float) else v for k, v in row.items()}


def _mean(key: str, per_problem: list[dict[str, Any]]) -> float:
    """Compute mean of *key* across per-problem dicts, skipping NaN."""
    vals = [r[key] for r in per_problem if not math.isnan(r.get(key, float("nan")))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Corruption primitives (pure Python)
# ---------------------------------------------------------------------------


def corrupt_solution_consistent(
    solution: str, answer: str, seed: int = 0
) -> tuple[str, int]:
    """Shift both an intermediate numeric token AND the final \\boxed{N} by
    the same delta, producing a self-consistent but wrong solution.

    Parameters
    ----------
    solution : str
        The original chain-of-thought solution text containing \\boxed{answer}.
    answer : str
        The correct final answer (numeric string).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (new_solution, n_corrupted) : tuple[str, int]
        The corrupted solution and the number of sites modified (0, 1, or 2).
    """
    rng = random.Random(seed)

    try:
        answer_int = int(answer)
    except (ValueError, TypeError):
        return solution, 0

    # Choose a delta that avoids the original answer and non-positive results
    candidates = [d for d in _DELTAS if answer_int + d != answer_int and answer_int + d > 0]
    if not candidates:
        return solution, 0
    delta = rng.choice(candidates)
    new_answer = answer_int + delta
    n_corrupted = 0

    # --- Shift one intermediate number ---
    intermediates = list(_INTERMEDIATE_RE.finditer(solution))
    # Exclude numbers inside \boxed{} from intermediate candidates
    boxed_spans = [m.span(1) for m in _BOXED_RE.finditer(solution)]

    def _in_boxed(m: re.Match) -> bool:
        s, e = m.span()
        return any(bs <= s and e <= be for bs, be in boxed_spans)

    eligible = [m for m in intermediates if not _in_boxed(m)]
    if eligible:
        target = rng.choice(eligible)
        try:
            old_val = int(target.group())
            new_val = old_val + delta
            if new_val > 0:
                solution = solution[: target.start()] + str(new_val) + solution[target.end() :]
                n_corrupted += 1
        except ValueError:
            pass

    # --- Shift the boxed answer ---
    def _replace_boxed(m: re.Match) -> str:
        nonlocal n_corrupted
        n_corrupted += 1
        return f"\\boxed{{{new_answer}}}"

    solution = _BOXED_RE.sub(_replace_boxed, solution, count=1)

    return solution, n_corrupted


def corrupt_solution_random_answer(
    solution: str, answer: str, answer_pool: list[str], seed: int = 0
) -> tuple[str, int]:
    """Swap the final \\boxed{} answer with a random in-distribution answer
    from *answer_pool*, leaving the chain content unchanged.

    Parameters
    ----------
    solution : str
        Original solution text.
    answer : str
        The correct answer currently in the boxed expression.
    answer_pool : list[str]
        Pool of plausible answers to draw from (should be same-distribution).
    seed : int
        Random seed.

    Returns
    -------
    (new_solution, n_random) : tuple[str, int]
        The modified solution and 1 if a swap was made, else 0.
    """
    rng = random.Random(seed)
    candidates = [a for a in answer_pool if str(a) != str(answer)]
    if not candidates:
        return solution, 0

    new_answer = rng.choice(candidates)
    new_solution, n = _BOXED_RE.subn(f"\\\\boxed{{{new_answer}}}", solution, count=1)
    return new_solution, min(n, 1)


def shuffle_solution_steps(solution: str, seed: int = 0) -> str:
    """Split solution on double-newlines, shuffle the paragraphs, and rejoin.

    Parameters
    ----------
    solution : str
        Original solution text.
    seed : int
        Random seed.

    Returns
    -------
    str
        The shuffled solution.
    """
    rng = random.Random(seed)
    paragraphs = solution.split("\n\n")
    rng.shuffle(paragraphs)
    return "\n\n".join(paragraphs)


def corrupt_solution_one_step(solution: str, seed: int = 0) -> str:
    """Change one intermediate numeric token by a random delta.

    Parameters
    ----------
    solution : str
        Original solution text.
    seed : int
        Random seed.

    Returns
    -------
    str
        The corrupted solution with one intermediate number changed.
    """
    rng = random.Random(seed)

    intermediates = list(_INTERMEDIATE_RE.finditer(solution))
    boxed_spans = [m.span(1) for m in _BOXED_RE.finditer(solution)]

    def _in_boxed(m: re.Match) -> bool:
        s, e = m.span()
        return any(bs <= s and e <= be for bs, be in boxed_spans)

    eligible = [m for m in intermediates if not _in_boxed(m)]
    if not eligible:
        return solution

    target = rng.choice(eligible)
    delta = rng.choice(_DELTAS)
    try:
        old_val = int(target.group())
        new_val = old_val + delta
        if new_val <= 0:
            new_val = old_val + abs(delta)
        solution = solution[: target.start()] + str(new_val) + solution[target.end() :]
    except ValueError:
        pass

    return solution


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_problems(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Load problems from a JSONL file, keeping only those with a
    ``"solution"`` field.

    Parameters
    ----------
    jsonl_path : str or Path
        Path to the JSONL file.

    Returns
    -------
    list[dict]
        List of problem dicts that contain a ``"solution"`` key.
    """
    problems: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "solution" in obj:
                problems.append(obj)
    return problems


# ---------------------------------------------------------------------------
# vLLM helpers (lazy imports)
# ---------------------------------------------------------------------------


def create_llm(
    model_id: str,
    seed: int = 1337,
    tensor_parallel_size: int | None = None,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.85,
    **extra_kwargs,
) -> Any:
    """Create a vLLM ``LLM`` instance with standard settings."""
    from vllm import LLM  # type: ignore[import-untyped]

    if tensor_parallel_size is None:
        import torch

        tensor_parallel_size = max(torch.cuda.device_count(), 1)

    return LLM(
        model=model_id,
        seed=seed,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        **extra_kwargs,
    )


def score_answer(
    llm: Any,
    tokenizer: Any,
    prompt: str,
    answer_string: str,
    seed: int = 1337,
) -> float:
    """Score the mean per-token log-probability of *answer_string* given
    *prompt*.

    Uses the prompt-logprobs scoring trick: concatenate ``prompt + answer``,
    request ``prompt_logprobs=0``, and read back the logprobs on the answer
    tokens.

    Parameters
    ----------
    llm : vllm.LLM
        The language model.
    tokenizer : transformers.PreTrainedTokenizer
        The model's tokenizer.
    prompt : str
        The conditioning prefix.
    answer_string : str
        The continuation to score.
    seed : int
        Random seed for sampling params.

    Returns
    -------
    float
        Mean per-token logprob, or ``float('nan')`` if scoring failed.
    """
    from vllm import SamplingParams  # type: ignore[import-untyped]

    full_text = prompt + answer_string
    answer_token_ids = tokenizer.encode(answer_string, add_special_tokens=False)
    n_answer_tokens = len(answer_token_ids)

    if n_answer_tokens == 0:
        return float("nan")

    params = SamplingParams(
        prompt_logprobs=0,
        max_tokens=1,
        temperature=0.0,
        seed=seed,
    )

    outputs = llm.generate([full_text], params)
    if not outputs or not outputs[0].prompt_logprobs:
        return float("nan")

    prompt_logprobs = outputs[0].prompt_logprobs
    # The answer tokens start at position (total_prompt_len - n_answer_tokens)
    prompt_token_ids = tokenizer.encode(full_text, add_special_tokens=False)
    start_idx = len(prompt_token_ids) - n_answer_tokens

    logprobs: list[float] = []
    for i in range(start_idx, len(prompt_token_ids)):
        if i >= len(prompt_logprobs) or prompt_logprobs[i] is None:
            continue
        token_id = prompt_token_ids[i]
        lp_dict = prompt_logprobs[i]
        if token_id in lp_dict:
            lp_value = lp_dict[token_id]
            # Handle both float and Logprob object
            if isinstance(lp_value, float):
                logprobs.append(lp_value)
            elif hasattr(lp_value, "logprob"):
                logprobs.append(lp_value.logprob)

    if not logprobs:
        return float("nan")
    return sum(logprobs) / len(logprobs)


def score_verdict_token(
    llm: Any,
    tokenizer: Any,
    prompt: str,
    token_str: str,
    seed: int = 1337,
) -> float:
    """Score a single verdict token (e.g. ``" YES"`` or ``" NO"``) given
    *prompt*.

    Same mechanism as :func:`score_answer` but specialised for a single
    token.

    Parameters
    ----------
    llm : vllm.LLM
        The language model.
    tokenizer : transformers.PreTrainedTokenizer
        The model's tokenizer.
    prompt : str
        The conditioning prefix.
    token_str : str
        The single token to score (e.g. ``" YES"``).
    seed : int
        Random seed.

    Returns
    -------
    float
        Log-probability of the token, or ``float('nan')``.
    """
    return score_answer(llm, tokenizer, prompt, token_str, seed=seed)


def verdict_logit(
    llm: Any,
    tokenizer: Any,
    prompt: str,
    seed: int = 1337,
) -> float:
    """Return ``logp(" YES") - logp(" NO")`` for *prompt*.

    Parameters
    ----------
    llm : vllm.LLM
        The language model.
    tokenizer : transformers.PreTrainedTokenizer
        The model's tokenizer.
    prompt : str
        The verdict prompt.
    seed : int
        Random seed.

    Returns
    -------
    float
        The log-odds of YES vs NO.
    """
    lp_yes = score_verdict_token(llm, tokenizer, prompt, " YES", seed=seed)
    lp_no = score_verdict_token(llm, tokenizer, prompt, " NO", seed=seed)
    if math.isnan(lp_yes) or math.isnan(lp_no):
        return float("nan")
    return lp_yes - lp_no


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_cot_prompt(problem: str, cot: str | None = None) -> str:
    """Build a chain-of-thought prompt.

    Parameters
    ----------
    problem : str
        The math problem statement.
    cot : str or None
        An optional solution sketch.  If ``None``, a free-generation prompt
        is returned; otherwise a guided-generation prompt.

    Returns
    -------
    str
        The formatted prompt.
    """
    if cot is None:
        return (
            "Solve the following problem. Show your reasoning briefly, "
            "then put the final integer answer in \\boxed{}.\n\n"
            f"Problem: {problem}\n\n"
            "Solution: "
        )
    return (
        "Solve the following problem. Use the provided solution sketch "
        "as guidance. Put the final integer answer in \\boxed{}.\n\n"
        f"Problem: {problem}\n\n"
        f"Solution sketch: {cot}\n\n"
        "Final answer: "
    )


def build_verdict_prompt(problem: str, solution: str) -> str:
    """Build a verdict (grading) prompt.

    Parameters
    ----------
    problem : str
        The math problem statement.
    solution : str
        The proposed solution to grade.

    Returns
    -------
    str
        The formatted verdict prompt.
    """
    return (
        "You are a careful math grader. Read the problem and the "
        "proposed solution, then judge whether the proposed solution "
        "is correct.\n\n"
        f"Problem: {problem}\n\n"
        f"Proposed solution: {solution}\n\n"
        "Is the proposed solution correct? Answer YES or NO.\n\n"
        "Answer:"
    )


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

_GATE_PCT_THRESHOLD = 0.6
_GATE_PVALUE_THRESHOLD = 0.05


def _save_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write a list of dicts to a JSONL file, sanitising NaNs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_sanitize_row(row)) + "\n")


def _wilcoxon_one_sided(x: list[float], y: list[float]) -> float:
    """Wilcoxon signed-rank test, one-sided (x > y).

    Returns the p-value, or 1.0 if the test cannot be computed.
    """
    from scipy.stats import wilcoxon  # type: ignore[import-untyped]

    diffs = [a - b for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    diffs = [d for d in diffs if d != 0.0]
    if len(diffs) < 10:
        return 1.0
    try:
        _, p = wilcoxon(diffs, alternative="greater")
        return float(p)
    except Exception:
        return 1.0


def run_consistent_corruption_probe(
    llm: Any,
    tokenizer: Any,
    problems: list[dict[str, Any]],
    seed: int = 1337,
    save_to: str | Path | None = None,
) -> dict[str, Any]:
    """Run the consistent-corruption discrimination probe.

    For each problem, score the model's per-token logprob under the oracle
    solution vs. a consistently-corrupted solution.  The discrimination
    signal is ``score_oracle - score_corrupted``.

    Parameters
    ----------
    llm : vllm.LLM
        The language model.
    tokenizer : transformers.PreTrainedTokenizer
        The model's tokenizer.
    problems : list[dict]
        Problem dicts, each with ``"problem"``, ``"solution"``, ``"answer"``.
    seed : int
        Random seed.
    save_to : str, Path, or None
        If provided, save per-problem JSONL to this path.

    Returns
    -------
    dict
        Summary with ``mean_discrimination_signal``, ``pct_oracle_above_corrupted``,
        ``wilcoxon_pvalue``, ``gate_passed``, and ``per_problem`` detail.
    """
    per_problem: list[dict[str, Any]] = []
    oracle_scores: list[float] = []
    corrupted_scores: list[float] = []

    for i, prob in enumerate(problems):
        problem_text = prob["problem"]
        solution = prob["solution"]
        answer = str(prob.get("answer", ""))

        corrupted, n_corrupted = corrupt_solution_consistent(solution, answer, seed=seed + i)
        if n_corrupted == 0:
            continue

        prompt_oracle = build_cot_prompt(problem_text)
        prompt_corrupted = build_cot_prompt(problem_text)

        s_oracle = score_answer(llm, tokenizer, prompt_oracle, solution, seed=seed)
        s_corrupt = score_answer(llm, tokenizer, prompt_corrupted, corrupted, seed=seed)

        discrimination = s_oracle - s_corrupt if not (math.isnan(s_oracle) or math.isnan(s_corrupt)) else float("nan")

        row = {
            "idx": i,
            "score_oracle": s_oracle,
            "score_corrupted": s_corrupt,
            "discrimination_signal": discrimination,
            "n_corrupted": n_corrupted,
        }
        per_problem.append(row)
        if not math.isnan(s_oracle):
            oracle_scores.append(s_oracle)
        if not math.isnan(s_corrupt):
            corrupted_scores.append(s_corrupt)

    mean_disc = _mean("discrimination_signal", per_problem)
    valid = [r for r in per_problem if not math.isnan(r["discrimination_signal"])]
    pct_above = (
        sum(1 for r in valid if r["discrimination_signal"] > 0) / len(valid)
        if valid
        else float("nan")
    )
    p_val = _wilcoxon_one_sided(oracle_scores, corrupted_scores)
    gate_passed = (
        not math.isnan(mean_disc)
        and mean_disc > 0
        and not math.isnan(pct_above)
        and pct_above > _GATE_PCT_THRESHOLD
        and p_val < _GATE_PVALUE_THRESHOLD
    )

    if save_to is not None:
        _save_jsonl(per_problem, save_to)

    return {
        "mean_discrimination_signal": _nan_to_none(mean_disc),
        "pct_oracle_above_corrupted": _nan_to_none(pct_above),
        "wilcoxon_pvalue": _nan_to_none(p_val),
        "gate_passed": gate_passed,
        "n_problems": len(per_problem),
        "per_problem": per_problem,
    }


def run_verdict_probe(
    llm: Any,
    tokenizer: Any,
    problems: list[dict[str, Any]],
    seed: int = 1337,
    save_to: str | Path | None = None,
) -> dict[str, Any]:
    """Run the verdict discrimination probe.

    For each problem, compute the verdict logit (YES vs NO) for oracle,
    corrupted, and shuffled solutions.

    Parameters
    ----------
    llm : vllm.LLM
        The language model.
    tokenizer : transformers.PreTrainedTokenizer
        The model's tokenizer.
    problems : list[dict]
        Problem dicts with ``"problem"``, ``"solution"``, ``"answer"``.
    seed : int
        Random seed.
    save_to : str, Path, or None
        Optional JSONL output path.

    Returns
    -------
    dict
        Summary with ``mean_delta_verdict``, ``pct_oracle_above_corrupted``,
        ``wilcoxon_pvalue``, ``gate_passed``.
    """
    per_problem: list[dict[str, Any]] = []
    oracle_logits: list[float] = []
    corrupted_logits: list[float] = []

    for i, prob in enumerate(problems):
        problem_text = prob["problem"]
        solution = prob["solution"]
        answer = str(prob.get("answer", ""))

        corrupted, _ = corrupt_solution_consistent(solution, answer, seed=seed + i)
        shuffled = shuffle_solution_steps(solution, seed=seed + i)

        v_oracle = verdict_logit(llm, tokenizer, build_verdict_prompt(problem_text, solution), seed=seed)
        v_corrupt = verdict_logit(llm, tokenizer, build_verdict_prompt(problem_text, corrupted), seed=seed)
        v_shuffle = verdict_logit(llm, tokenizer, build_verdict_prompt(problem_text, shuffled), seed=seed)

        delta = v_oracle - v_corrupt if not (math.isnan(v_oracle) or math.isnan(v_corrupt)) else float("nan")

        row = {
            "idx": i,
            "verdict_oracle": v_oracle,
            "verdict_corrupted": v_corrupt,
            "verdict_shuffled": v_shuffle,
            "delta_verdict": delta,
        }
        per_problem.append(row)
        if not math.isnan(v_oracle):
            oracle_logits.append(v_oracle)
        if not math.isnan(v_corrupt):
            corrupted_logits.append(v_corrupt)

    mean_delta = _mean("delta_verdict", per_problem)
    valid = [r for r in per_problem if not math.isnan(r["delta_verdict"])]
    pct_above = (
        sum(1 for r in valid if r["delta_verdict"] > 0) / len(valid)
        if valid
        else float("nan")
    )
    p_val = _wilcoxon_one_sided(oracle_logits, corrupted_logits)
    gate_passed = (
        not math.isnan(mean_delta)
        and mean_delta > 0
        and not math.isnan(pct_above)
        and pct_above > _GATE_PCT_THRESHOLD
        and p_val < _GATE_PVALUE_THRESHOLD
    )

    if save_to is not None:
        _save_jsonl(per_problem, save_to)

    return {
        "mean_delta_verdict": _nan_to_none(mean_delta),
        "pct_oracle_above_corrupted": _nan_to_none(pct_above),
        "wilcoxon_pvalue": _nan_to_none(p_val),
        "gate_passed": gate_passed,
        "n_problems": len(per_problem),
        "per_problem": per_problem,
    }


def run_per_token_surprise_probe(
    llm: Any,
    tokenizer: Any,
    problems: list[dict[str, Any]],
    seed: int = 1337,
    save_to: str | Path | None = None,
) -> dict[str, Any]:
    """Run the per-token surprise probe.

    For each problem, compute per-token logprobs for the corrupted solution
    and check whether tokens near corruption sites (+-3 tokens around diff
    sites) show elevated surprise relative to the global average.

    Parameters
    ----------
    llm : vllm.LLM
        The language model.
    tokenizer : transformers.PreTrainedTokenizer
        The model's tokenizer.
    problems : list[dict]
        Problem dicts with ``"problem"``, ``"solution"``, ``"answer"``.
    seed : int
        Random seed.
    save_to : str, Path, or None
        Optional JSONL output path.

    Returns
    -------
    dict
        Summary with ``mean_local_minus_global``, ``pct_positive``,
        ``wilcoxon_pvalue``, ``gate_passed``.
    """
    from vllm import SamplingParams  # type: ignore[import-untyped]

    per_problem: list[dict[str, Any]] = []
    local_surprises: list[float] = []
    global_surprises: list[float] = []
    window_radius = 3

    for i, prob in enumerate(problems):
        problem_text = prob["problem"]
        solution = prob["solution"]
        answer = str(prob.get("answer", ""))

        corrupted, n_corrupted = corrupt_solution_consistent(solution, answer, seed=seed + i)
        if n_corrupted == 0:
            continue

        prompt = build_cot_prompt(problem_text)
        full_text = prompt + corrupted

        # Tokenize to find diff sites
        oracle_tokens = tokenizer.encode(prompt + solution, add_special_tokens=False)
        corrupt_tokens = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        n_prompt = len(prompt_tokens)

        # Find positions where tokens differ (corruption sites)
        diff_positions: set[int] = set()
        min_len = min(len(oracle_tokens), len(corrupt_tokens))
        for j in range(n_prompt, min_len):
            if oracle_tokens[j] != corrupt_tokens[j]:
                diff_positions.add(j)
        # Also flag positions beyond min_len
        for j in range(min_len, max(len(oracle_tokens), len(corrupt_tokens))):
            diff_positions.add(j)

        if not diff_positions:
            continue

        # Expand to window
        window_positions: set[int] = set()
        for pos in diff_positions:
            for offset in range(-window_radius, window_radius + 1):
                p = pos + offset
                if n_prompt <= p < len(corrupt_tokens):
                    window_positions.add(p)

        # Score the corrupted solution
        params = SamplingParams(
            prompt_logprobs=0,
            max_tokens=1,
            temperature=0.0,
            seed=seed,
        )
        outputs = llm.generate([full_text], params)
        if not outputs or not outputs[0].prompt_logprobs:
            continue

        prompt_logprobs = outputs[0].prompt_logprobs

        # Collect per-token surprise (negative logprob = surprise)
        all_surprises: list[float] = []
        local_surp: list[float] = []
        non_local_surp: list[float] = []

        for j in range(n_prompt, len(corrupt_tokens)):
            if j >= len(prompt_logprobs) or prompt_logprobs[j] is None:
                continue
            token_id = corrupt_tokens[j]
            lp_dict = prompt_logprobs[j]
            if token_id in lp_dict:
                lp_value = lp_dict[token_id]
                lp = lp_value.logprob if hasattr(lp_value, "logprob") else float(lp_value)
                surprise = -lp  # higher = more surprising
                all_surprises.append(surprise)
                if j in window_positions:
                    local_surp.append(surprise)
                else:
                    non_local_surp.append(surprise)

        if not all_surprises or not local_surp:
            continue

        global_mean = sum(all_surprises) / len(all_surprises)
        local_mean = sum(local_surp) / len(local_surp)
        delta = local_mean - global_mean

        row = {
            "idx": i,
            "local_surprise": local_mean,
            "global_surprise": global_mean,
            "local_minus_global": delta,
            "n_local_tokens": len(local_surp),
            "n_total_tokens": len(all_surprises),
        }
        per_problem.append(row)
        local_surprises.append(local_mean)
        global_surprises.append(global_mean)

    mean_delta = _mean("local_minus_global", per_problem)
    valid = [r for r in per_problem if not math.isnan(r["local_minus_global"])]
    pct_positive = (
        sum(1 for r in valid if r["local_minus_global"] > 0) / len(valid)
        if valid
        else float("nan")
    )
    p_val = _wilcoxon_one_sided(local_surprises, global_surprises)
    gate_passed = (
        not math.isnan(mean_delta)
        and mean_delta > 0
        and not math.isnan(pct_positive)
        and pct_positive > _GATE_PCT_THRESHOLD
        and p_val < _GATE_PVALUE_THRESHOLD
    )

    if save_to is not None:
        _save_jsonl(per_problem, save_to)

    return {
        "mean_local_minus_global": _nan_to_none(mean_delta),
        "pct_positive": _nan_to_none(pct_positive),
        "wilcoxon_pvalue": _nan_to_none(p_val),
        "gate_passed": gate_passed,
        "n_problems": len(per_problem),
        "per_problem": per_problem,
    }


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def estimate_cost(gpu: str, n_gpus: int, hours: float) -> float:
    """Estimate GPU cost in USD.

    Parameters
    ----------
    gpu : str
        GPU type, one of ``"H100"`` or ``"A100-80GB"``.
    n_gpus : int
        Number of GPUs.
    hours : float
        Duration in hours.

    Returns
    -------
    float
        Estimated cost in USD.

    Raises
    ------
    ValueError
        If the GPU type is not recognised.
    """
    rate = _GPU_RATES.get(gpu)
    if rate is None:
        raise ValueError(f"Unknown GPU type {gpu!r}. Known: {list(_GPU_RATES)}")
    return rate * n_gpus * hours
