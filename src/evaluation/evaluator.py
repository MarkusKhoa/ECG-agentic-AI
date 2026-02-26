"""
Task-level evaluation driver.

Runs the CardioClinician-Agent graph on each sample in a task and collects
predictions, then computes metrics against gold references.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from tqdm import tqdm

from src.agents.base import get_llm
from src.agents.conformal import (
    CalibrationResult,
    ConformalCalibrator,
    make_calibration_generate_factory,
)
from src.agents.coordinator import build_graph
from src.agents.prompts import COORDINATOR_PROMPT, ETHICS_GUARDIAN_PROMPT
from src.config import CFG
from src.dataio.loader import TASK_NAMES, load_task
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


def _run_sample(graph: Any, sample: dict) -> dict[str, Any]:
    """Run a single sample through the agent graph and return full result."""
    state = {
        "task": sample["task"],
        "input": sample["input"],
        "meta": sample.get("meta", {}),
        "agent_output": "",
        "ethics_review": "",
        "needs_revision": False,
        "revision_count": 0,
        "final_answer": "",
        "trace": [],
        "clm_prediction_set": {},
        "clm_reliable_sentences": [],
    }
    try:
        result = graph.invoke(state)
        return {
            "final_answer": result.get("final_answer", ""),
            "clm_prediction_set": result.get("clm_prediction_set", {}),
            "clm_reliable_sentences": result.get("clm_reliable_sentences", []),
            "trace": result.get("trace", []),
        }
    except Exception as e:
        logger.error("Error processing sample: %s", e)
        return {
            "final_answer": f"[ERROR] {e}",
            "clm_prediction_set": {},
            "clm_reliable_sentences": [],
            "trace": [],
        }


# ---------------------------------------------------------------------------
# CLM calibration with disk caching
# ---------------------------------------------------------------------------

def _calibration_cache_path(task: str) -> Path:
    """Deterministic cache path based on (task, model, ε, δ)."""
    key = (
        f"{task}|{CFG.openai_model}|{CFG.clm_epsilon}|{CFG.clm_delta}"
        f"|{CFG.clm_sampling_temperature}"
    )
    digest = hashlib.md5(key.encode()).hexdigest()[:10]
    return CFG.results_dir / f"{task}_clm_cal_{digest}.json"


def _load_cached_calibration(task: str) -> CalibrationResult | None:
    """Load a cached CalibrationResult from disk, if it exists."""
    path = _calibration_cache_path(task)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        result = CalibrationResult(**data)
        logger.info("Loaded cached CLM calibration for %s from %s", task, path)
        return result
    except Exception as e:
        logger.warning("Failed to load CLM cache for %s: %s", task, e)
        return None


def _save_calibration_cache(task: str, result: CalibrationResult) -> None:
    """Persist a CalibrationResult to disk."""
    path = _calibration_cache_path(task)
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    logger.info("Saved CLM calibration for %s to %s", task, path)


def calibrate_clm(
    task: str,
    calibration_data: list[dict[str, str]],
    recalibrate: bool = False,
) -> CalibrationResult | None:
    """Run LTT calibration for a task (or load from cache).

    Args:
        task: Task abbreviation.
        calibration_data: List of dicts with 'input' and 'reference' keys.
        recalibrate: If True, ignore cached results and recalibrate.

    Returns:
        CalibrationResult, or None if calibration is disabled.
    """
    if not CFG.clm_enabled or CFG.clm_calibration_samples <= 0:
        return None

    if not calibration_data:
        logger.warning("No calibration data for %s; skipping CLM calibration.", task)
        return None

    # Check cache first
    if not recalibrate:
        cached = _load_cached_calibration(task)
        if cached is not None:
            return cached

    # Determine system prompt for synthesis (same as synthesise_node uses)
    if task in ("PRTK", "GRA"):
        system_prompt = ETHICS_GUARDIAN_PROMPT
    else:
        system_prompt = COORDINATOR_PROMPT

    llm = get_llm()
    gen_factory = make_calibration_generate_factory(
        llm, system_prompt, temperature=CFG.clm_sampling_temperature,
    )

    calibrator = ConformalCalibrator(
        epsilon=CFG.clm_epsilon,
        delta=CFG.clm_delta,
    )

    logger.info(
        "Running CLM calibration for %s with %d samples...",
        task, len(calibration_data),
    )
    result = calibrator.calibrate(
        generate_fn=gen_factory,
        calibration_data=calibration_data,
        cal_k_max=3,
    )

    _save_calibration_cache(task, result)
    return result


def evaluate_task(
    task: str,
    graph: Any | None = None,
    save: bool = True,
    clm_cal_result: CalibrationResult | None = None,
) -> dict[str, Any]:
    """Evaluate a single task end-to-end.

    Args:
        task: Task abbreviation (e.g. 'EBK', 'CMD').
        graph: Pre-compiled LangGraph. Built fresh if None.
        save: Whether to write results to disk.
        clm_cal_result: Optional calibration result for CLM thresholds.

    Returns:
        Dictionary with metrics and per-sample details.
    """
    logger.info("Evaluating task: %s", task)
    samples = load_task(task)
    logger.info("Loaded %d samples for %s", len(samples), task)

    # Split off calibration samples if calibration is active
    n_cal = CFG.clm_calibration_samples
    if CFG.clm_enabled and n_cal > 0 and clm_cal_result is None:
        cal_data = samples[:n_cal]
        samples = samples[n_cal:]
        logger.info(
            "Split %d calibration + %d evaluation samples for %s",
            len(cal_data), len(samples), task,
        )
        clm_cal_result = calibrate_clm(task, cal_data, recalibrate=CFG.clm_recalibrate)

    if graph is None:
        clm_config = clm_cal_result.to_config() if clm_cal_result else None
        graph = build_graph(clm_config=clm_config)

    predictions: list[str] = []
    references: list[str] = []
    details: list[dict] = []

    clm_set_sizes: list[int] = []
    clm_confidences: list[float] = []
    clm_reliable_fracs: list[float] = []

    for i, sample in enumerate(tqdm(samples, desc=f"  └─ {task} samples", position=1, leave=False)):
        t0 = time.time()
        result = _run_sample(graph, sample)
        elapsed = time.time() - t0

        # Log trace if verbose mode is enabled
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"\n{'='*60}\nSample {i+1}/{len(samples)} - {task}\n{'='*60}")
            logger.debug(f"Input: {sample['input'][:200]}...")
            for step in result.get("trace", []):
                logger.debug(f"  → {step.get('node', 'unknown')}: {step}")
            logger.debug(f"Final answer: {result['final_answer'][:200]}...")
            logger.debug(f"Time: {elapsed:.2f}s\n")

        pred = result["final_answer"]
        predictions.append(pred)
        references.append(sample["reference"])

        # Collect CLM metadata
        clm_meta = result.get("clm_prediction_set", {})
        if clm_meta:
            clm_set_sizes.append(clm_meta.get("set_size", 1))
            clm_confidences.append(clm_meta.get("set_confidence", 0.0))
            reliable = result.get("clm_reliable_sentences", [])
            if reliable:
                n_reliable = sum(1 for c in reliable if c.get("reliable"))
                clm_reliable_fracs.append(n_reliable / len(reliable))

        details.append({
            "index": i,
            "input": sample["input"][:500],
            "reference": sample["reference"][:500],
            "prediction": pred[:500],
            "time_s": round(elapsed, 2),
            "clm": clm_meta,
            "trace": result.get("trace", []),
        })

    metrics = compute_metrics(predictions, references, task=task)

    # Add CLM-specific metrics
    if clm_set_sizes:
        metrics["clm_avg_set_size"] = round(float(np.mean(clm_set_sizes)), 2)
        metrics["clm_avg_confidence"] = round(float(np.mean(clm_confidences)), 4)
        if clm_reliable_fracs:
            metrics["clm_avg_reliable_frac"] = round(
                float(np.mean(clm_reliable_fracs)), 4
            )

    # Add calibration metadata if available
    if clm_cal_result is not None:
        metrics["clm_calibrated"] = clm_cal_result.is_valid
        metrics["clm_lambda_sim"] = clm_cal_result.lambda_sim
        metrics["clm_lambda_qual"] = clm_cal_result.lambda_qual
        metrics["clm_lambda_conf"] = clm_cal_result.lambda_conf
        metrics["clm_cal_coverage"] = round(clm_cal_result.empirical_coverage, 4)

    result = {"task": task, "metrics": metrics, "details": details}

    if save:
        out_path = CFG.results_dir / f"{task}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", out_path)

    return result


def evaluate_all(
    tasks: list[str] | None = None,
    save: bool = True,
) -> dict[str, dict[str, Any]]:
    """Evaluate all (or selected) tasks and produce a summary report.

    Args:
        tasks: List of task abbreviations. Defaults to all tasks.
        save: Whether to write results to disk.

    Returns:
        Dictionary mapping task → results.
    """
    if tasks is None:
        tasks = TASK_NAMES

    all_results: dict[str, dict[str, Any]] = {}

    for task in tqdm(tasks, desc="Overall Progress", position=0, leave=True):
        try:
            # Per-task graph build (supports per-task calibrated λ)
            all_results[task] = evaluate_task(task, graph=None, save=save)
        except Exception as e:
            logger.error("Failed on task %s: %s", task, e)
            all_results[task] = {"task": task, "error": str(e)}

    # Summary table
    summary_rows: list[dict] = []
    for task, res in all_results.items():
        m = res.get("metrics", {})
        row = {
            "task": task,
            "n_samples": m.get("n_samples", 0),
            "rouge1": round(m.get("rouge1", 0), 4),
            "rouge2": round(m.get("rouge2", 0), 4),
            "rougeL": round(m.get("rougeL", 0), 4),
            "bleu1": round(m.get("bleu1", 0), 4),
            "meteor": round(m.get("meteor", 0), 4),
            "bertscore_f1": round(m.get("bertscore_f1", 0), 4),
        }
        if task == "MEE":
            row["entity_f1"] = round(m.get("entity_f1", 0), 4)
        # CLM metrics
        if "clm_avg_set_size" in m:
            row["clm_set_size"] = m["clm_avg_set_size"]
            row["clm_confidence"] = m.get("clm_avg_confidence", 0.0)
            row["clm_reliable_frac"] = m.get("clm_avg_reliable_frac", 0.0)
        summary_rows.append(row)

    if save:
        summary_path = CFG.results_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2, ensure_ascii=False)
        logger.info("Summary saved to %s", summary_path)

        # Also write a human-readable table
        table_path = CFG.results_dir / "summary.txt"
        has_clm = any("clm_set_size" in r for r in summary_rows)
        with open(table_path, "w") as f:
            header = (
                f"{'Task':<8} {'N':>6} {'R-1':>8} {'R-2':>8} {'R-L':>8} "
                f"{'BL-1':>8} {'MET':>8} {'BS-F1':>8}"
            )
            if has_clm:
                header += f" {'CLM-Sz':>7} {'CLM-Cf':>7} {'CLM-Rl':>7}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for row in summary_rows:
                line = (
                    f"{row['task']:<8} {row['n_samples']:>6} "
                    f"{row['rouge1']:>8.4f} {row['rouge2']:>8.4f} "
                    f"{row['rougeL']:>8.4f} {row['bleu1']:>8.4f} "
                    f"{row['meteor']:>8.4f} {row['bertscore_f1']:>8.4f}"
                )
                if has_clm and "clm_set_size" in row:
                    line += (
                        f" {row['clm_set_size']:>7.1f}"
                        f" {row['clm_confidence']:>7.4f}"
                        f" {row.get('clm_reliable_frac', 0.0):>7.4f}"
                    )
                f.write(line + "\n")
        logger.info("Table saved to %s", table_path)

    return all_results
