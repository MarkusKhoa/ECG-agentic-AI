#!/usr/bin/env python3
"""
View detailed agent reasoning traces from evaluation results.

Usage:
    # View traces for a specific task
    python -m src.view_trace --task EBK
    
    # View a specific sample
    python -m src.view_trace --task EBK --sample 0
    
    # View all samples with full details
    python -m src.view_trace --task EBK --all
"""

from __future__ import annotations

import argparse
import json

from src.config import CFG


def format_trace_step(step: dict, indent: int = 2) -> str:
    """Format a single trace step for display."""
    prefix = " " * indent
    node = step.get("node", "unknown")
    
    lines = [f"{prefix}┌─ {node.upper().replace('_', ' ')}"]
    
    for key, value in step.items():
        if key == "node":
            continue
        if isinstance(value, (int, float, bool)):
            lines.append(f"{prefix}│  {key}: {value}")
        elif isinstance(value, str) and len(value) < 100:
            lines.append(f"{prefix}│  {key}: {value}")
        elif isinstance(value, list):
            lines.append(f"{prefix}│  {key}: [{len(value)} items]")
        elif isinstance(value, dict):
            lines.append(f"{prefix}│  {key}: {{{len(value)} keys}}")
    
    lines.append(f"{prefix}└─")
    return "\n".join(lines)


def view_sample_trace(task: str, sample_idx: int) -> None:
    """Display detailed trace for a specific sample."""
    results_file = CFG.results_dir / f"{task}_results.json"
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print(f"   Run evaluation first: python -m src.run_baseline --tasks {task}")
        return
    
    with open(results_file, "r") as f:
        data = json.load(f)
    
    details = data.get("details", [])
    if sample_idx >= len(details):
        print(f"❌ Sample {sample_idx} not found. Task {task} has {len(details)} samples.")
        return
    
    sample = details[sample_idx]
    trace = sample.get("trace", [])
    
    print("=" * 80)
    print(f"AGENT REASONING TRACE - {task} Sample #{sample_idx}")
    print("=" * 80)
    
    print(f"\n📥 INPUT ({len(sample['input'])} chars):")
    print("-" * 80)
    print(sample["input"])
    
    print("\n🤖 AGENT EXECUTION TRACE:")
    print("-" * 80)
    if not trace:
        print("  (No trace data available)")
    else:
        for i, step in enumerate(trace, 1):
            print(f"\nStep {i}:")
            print(format_trace_step(step))
    
    print(f"\n📤 FINAL ANSWER ({len(sample['prediction'])} chars):")
    print("-" * 80)
    print(sample["prediction"])
    
    print(f"\n✅ REFERENCE ({len(sample['reference'])} chars):")
    print("-" * 80)
    print(sample["reference"])
    
    print(f"\n⏱️  Execution time: {sample.get('time_s', 0):.2f}s")
    
    # CLM info if available
    clm = sample.get("clm", {})
    if clm:
        print("\n🎯 CLM INFO:")
        print(f"   Set size: {clm.get('set_size', 'N/A')}")
        print(f"   Confidence: {clm.get('set_confidence', 'N/A')}")
        print(f"   Reliable sentences: {clm.get('n_reliable', 'N/A')}/{clm.get('n_total', 'N/A')}")
    
    print("\n" + "=" * 80)


def view_all_traces(task: str) -> None:
    """Display summary of all traces for a task."""
    results_file = CFG.results_dir / f"{task}_results.json"
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    with open(results_file, "r") as f:
        data = json.load(f)
    
    details = data.get("details", [])
    metrics = data.get("metrics", {})
    
    print("=" * 80)
    print(f"TASK: {task} - {len(details)} samples")
    print("=" * 80)
    
    print("\n📊 METRICS:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    print(f"\n📋 SAMPLE TRACES:")
    print("-" * 80)
    
    for sample in details:
        idx = sample["index"]
        trace = sample.get("trace", [])
        time_s = sample.get("time_s", 0)
        
        # Extract agent path
        agent_path = " → ".join([s.get("node", "?") for s in trace])
        
        print(f"Sample {idx:3d}: {len(trace):2d} steps | {time_s:5.2f}s | {agent_path}")
    
    print("\n" + "=" * 80)
    print(f"💡 View details: python -m src.view_trace --task {task} --sample <N>")


def main() -> None:
    parser = argparse.ArgumentParser(description="View agent reasoning traces")
    parser.add_argument("--task", required=True, help="Task abbreviation (e.g., EBK, CMD)")
    parser.add_argument("--sample", type=int, default=None, help="Sample index to view")
    parser.add_argument("--all", action="store_true", help="View summary of all samples")
    
    args = parser.parse_args()
    
    if args.sample is not None:
        view_sample_trace(args.task, args.sample)
    else:
        view_all_traces(args.task)


if __name__ == "__main__":
    main()
