"""
Evaluation metrics for the ECG-Expert-QA benchmark.

Metrics:
    - ROUGE-1, ROUGE-2, ROUGE-L   (text overlap)
    - BLEU-1                      (unigram precision + brevity penalty)
    - METEOR                      (unigram F-mean + chunk penalty)
    - BERTScore (P / R / F1)       (semantic similarity)
    - Entity F1                     (for MEE task)
    - Exact-match accuracy          (for simple QA)
"""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any

import numpy as np
from rouge_score import rouge_scorer


TOKEN_PATTERN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word-level tokenizer for lexical overlap metrics."""
    return TOKEN_PATTERN.findall(text.lower())


def _bleu1_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute averaged sentence-level BLEU-1.

    BLEU-1 = BP * p1
    where p1 is modified unigram precision and BP is the brevity penalty.
    """
    bleu1_values: list[float] = []

    for pred, ref in zip(predictions, references):
        cand_tokens = _tokenize(pred)
        ref_tokens = _tokenize(ref)

        c = len(cand_tokens)
        r = len(ref_tokens)
        if c == 0:
            bleu1_values.append(0.0)
            continue

        cand_counts = Counter(cand_tokens)
        ref_counts = Counter(ref_tokens)
        clipped_matches = sum(
            min(count, ref_counts[token]) for token, count in cand_counts.items()
        )
        p1 = clipped_matches / c

        bp = 1.0 if c > r else math.exp(1.0 - (r / c))
        bleu1_values.append(bp * p1)

    return {"bleu1": float(np.mean(bleu1_values))}


def _meteor_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute averaged sentence-level METEOR using exact unigram matches."""

    def _matched_reference_indices(
        cand_tokens: list[str], ref_tokens: list[str]
    ) -> list[int]:
        used = [False] * len(ref_tokens)
        matched_indices: list[int] = []
        for token in cand_tokens:
            for idx, ref_token in enumerate(ref_tokens):
                if not used[idx] and token == ref_token:
                    used[idx] = True
                    matched_indices.append(idx)
                    break
        return matched_indices

    def _chunk_count(matched_indices: list[int]) -> int:
        if not matched_indices:
            return 0
        chunks = 1
        for prev_idx, current_idx in zip(matched_indices, matched_indices[1:]):
            if current_idx != prev_idx + 1:
                chunks += 1
        return chunks

    meteor_values: list[float] = []
    for pred, ref in zip(predictions, references):
        cand_tokens = _tokenize(pred)
        ref_tokens = _tokenize(ref)

        if not cand_tokens or not ref_tokens:
            meteor_values.append(0.0)
            continue

        matched_indices = _matched_reference_indices(cand_tokens, ref_tokens)
        matches = len(matched_indices)
        if matches == 0:
            meteor_values.append(0.0)
            continue

        precision = matches / len(cand_tokens)
        recall = matches / len(ref_tokens)
        f_mean = (10.0 * precision * recall) / (9.0 * precision + recall)

        chunks = _chunk_count(matched_indices)
        penalty = 0.5 * ((chunks / matches) ** 3)
        meteor = f_mean * (1.0 - penalty)
        meteor_values.append(max(0.0, meteor))

    return {"meteor": float(np.mean(meteor_values))}


def _rouge_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute averaged ROUGE-1/2/L F-measures."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
    return {
        "rouge1": float(np.mean(r1)),
        "rouge2": float(np.mean(r2)),
        "rougeL": float(np.mean(rl)),
    }


def _bert_score(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute BERTScore (precision, recall, F1)."""
    try:
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(
            predictions, references,
            lang="en",
            verbose=False,
            rescale_with_baseline=True,
        )
        return {
            "bertscore_precision": float(P.mean()),
            "bertscore_recall": float(R.mean()),
            "bertscore_f1": float(F1.mean()),
        }
    except Exception as e:
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
            "bertscore_error": str(e),
        }


def _entity_f1(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Token-level F1 for entity extraction (MEE task).

    Treats each comma-separated entity (lowered, stripped) as a set member
    and computes micro-averaged precision / recall / F1.
    """
    total_tp, total_fp, total_fn = 0, 0, 0
    for pred, ref in zip(predictions, references):
        pred_set = {e.strip().lower() for e in re.split(r"[,;\n]", pred) if e.strip()}
        ref_set = {e.strip().lower() for e in re.split(r"[,;\n]", ref) if e.strip()}
        tp = len(pred_set & ref_set)
        fp = len(pred_set - ref_set)
        fn = len(ref_set - pred_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"entity_precision": precision, "entity_recall": recall, "entity_f1": f1}


def compute_metrics(
    predictions: list[str],
    references: list[str],
    task: str = "",
) -> dict[str, Any]:
    """Compute all relevant metrics for a task.

    Args:
        predictions: Model outputs.
        references: Gold-standard answers.
        task: Task abbreviation (used to select metric suite).

    Returns:
        Dictionary of metric name → value.
    """
    if not predictions or not references:
        return {"error": "empty predictions or references"}

    results: dict[str, Any] = {"n_samples": len(predictions)}

    # ROUGE for all tasks
    results.update(_rouge_scores(predictions, references))

    # BLEU-1 / METEOR for all tasks
    results.update(_bleu1_scores(predictions, references))
    results.update(_meteor_scores(predictions, references))

    # BERTScore for all tasks
    results.update(_bert_score(predictions, references))

    # Entity F1 specifically for MEE
    if task == "MEE":
        results.update(_entity_f1(predictions, references))

    return results
