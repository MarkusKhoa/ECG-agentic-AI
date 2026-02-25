#!/usr/bin/env python3
"""
CardioClinician-Agent  –  Baseline evaluation on ECG-Expert-QA.

Usage:
    # Evaluate all 11 tasks:
    python -m src.run_baseline

    # Evaluate specific tasks:
    python -m src.run_baseline --tasks EBK CK CMD

    # Limit samples per task (for quick testing):
    MAX_SAMPLES=5 python -m src.run_baseline --tasks EBK

    # Use a HuggingFace local model:
    LLM_BACKEND=huggingface HF_MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct \
        python -m src.run_baseline

Environment:
    OPENAI_API_KEY  – Required when LLM_BACKEND=openai (default).
"""

from __future__ import annotations

import argparse
import logging
import sys
import os

# Ensure project root is on sys.path so `src.*` imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.dataio.loader import TASK_NAMES
from src.evaluation.evaluator import evaluate_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CardioClinician-Agent baseline on ECG-Expert-QA."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=TASK_NAMES,
        help="Task(s) to evaluate.  Defaults to all.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override MAX_SAMPLES (cap per task).  0 = all.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    if args.max_samples is not None:
        CFG.max_samples = args.max_samples

    # Validate API key for OpenAI backend
    if CFG.llm_backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        logging.error(
            "OPENAI_API_KEY not set.  Either export it or switch to "
            "LLM_BACKEND=huggingface."
        )
        sys.exit(1)

    logging.info("=" * 60)
    logging.info("CardioClinician-Agent  –  Baseline Evaluation")
    logging.info("=" * 60)
    logging.info("LLM backend : %s", CFG.llm_backend)
    logging.info("Model       : %s",
                 CFG.openai_model if CFG.llm_backend == "openai" else CFG.hf_model_name)
    logging.info("Temperature : %s", CFG.temperature)
    logging.info("Max tokens  : %s", CFG.max_tokens)
    logging.info("Max samples : %s", CFG.max_samples if CFG.max_samples else "all")
    logging.info("Results dir : %s", CFG.results_dir)
    logging.info("Tasks       : %s", args.tasks or "ALL")
    logging.info("=" * 60)

    results = evaluate_all(tasks=args.tasks, save=True)

    # Print summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Task':<8} {'N':>6}  {'ROUGE-1':>8}  {'ROUGE-2':>8}  {'ROUGE-L':>8}  {'BS-F1':>8}")
    print("-" * 70)
    for task, res in results.items():
        m = res.get("metrics", {})
        if "error" in res:
            print(f"  {task:<8} {'ERR':>6}  {res['error']}")
        else:
            print(
                f"  {task:<8} {m.get('n_samples', 0):>6}  "
                f"{m.get('rouge1', 0):>8.4f}  {m.get('rouge2', 0):>8.4f}  "
                f"{m.get('rougeL', 0):>8.4f}  {m.get('bertscore_f1', 0):>8.4f}"
            )
            if task == "MEE":
                print(f"  {'':>8} {'':>6}  Entity-F1: {m.get('entity_f1', 0):.4f}")
    print("=" * 70)
    print(f"\nDetailed results written to: {CFG.results_dir}/")


if __name__ == "__main__":
    main()
