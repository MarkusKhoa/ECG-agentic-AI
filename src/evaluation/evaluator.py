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

from tqdm import tqdm

from src.agents.coordinator import build_graph
from src.config import CFG
from src.dataio.loader import TASK_NAMES, load_task
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


def _run_sample(graph: Any, sample: dict) -> str:
    """Run a single sample through the agent graph and return the final answer."""
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
    }
    try:
        result = graph.invoke(state)
        return result.get("final_answer", "")
    except Exception as e:
        logger.error("Error processing sample: %s", e)
        return f"[ERROR] {e}"


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

    for i, sample in enumerate(tqdm(samples, desc=f"[{task}]")):
        t0 = time.time()
        pred = _run_sample(graph, sample)
        elapsed = time.time() - t0

        predictions.append(pred)
        references.append(sample["reference"])
        details.append({
            "index": i,
            "input": sample["input"][:500],
            "reference": sample["reference"][:500],
            "prediction": pred[:500],
            "time_s": round(elapsed, 2),
        })

    metrics = compute_metrics(predictions, references, task=task)
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
        summary_rows.append({
            "task": task,
            "n_samples": m.get("n_samples", 0),
            "rouge1": round(m.get("rouge1", 0), 4),
            "rouge2": round(m.get("rouge2", 0), 4),
            "rougeL": round(m.get("rougeL", 0), 4),
            "bertscore_f1": round(m.get("bertscore_f1", 0), 4),
            **({"entity_f1": round(m.get("entity_f1", 0), 4)} if task == "MEE" else {}),
        })

    if save:
        summary_path = CFG.results_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2, ensure_ascii=False)
        logger.info("Summary saved to %s", summary_path)

        # Also write a human-readable table
        table_path = CFG.results_dir / "summary.txt"
        with open(table_path, "w") as f:
            header = f"{'Task':<8} {'N':>6} {'R-1':>8} {'R-2':>8} {'R-L':>8} {'BS-F1':>8}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for row in summary_rows:
                line = (
                    f"{row['task']:<8} {row['n_samples']:>6} "
                    f"{row['rouge1']:>8.4f} {row['rouge2']:>8.4f} "
                    f"{row['rougeL']:>8.4f} {row['bertscore_f1']:>8.4f}"
                )
                f.write(line + "\n")
        logger.info("Table saved to %s", table_path)

    return all_results
