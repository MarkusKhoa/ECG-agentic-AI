"""
Task-level evaluation driver.

Runs the CardioClinician-Agent graph on each sample in a task and collects
predictions, then computes metrics against gold references.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np

from tqdm import tqdm

from src.agents.coordinator import build_graph
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


def evaluate_task(
    task: str,
    graph: Any | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Evaluate a single task end-to-end.

    Args:
        task: Task abbreviation (e.g. 'EBK', 'CMD').
        graph: Pre-compiled LangGraph. Built fresh if None.
        save: Whether to write results to disk.

    Returns:
        Dictionary with metrics and per-sample details.
    """
    logger.info("Evaluating task: %s", task)
    samples = load_task(task)
    logger.info("Loaded %d samples for %s", len(samples), task)

    if graph is None:
        graph = build_graph()

    predictions: list[str] = []
    references: list[str] = []
    details: list[dict] = []

    clm_set_sizes: list[int] = []
    clm_confidences: list[float] = []
    clm_reliable_fracs: list[float] = []

    for i, sample in enumerate(tqdm(samples, desc=f"[{task}]")):
        t0 = time.time()
        result = _run_sample(graph, sample)
        elapsed = time.time() - t0

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

    graph = build_graph()
    all_results: dict[str, dict[str, Any]] = {}

    for task in tasks:
        try:
            all_results[task] = evaluate_task(task, graph=graph, save=save)
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
