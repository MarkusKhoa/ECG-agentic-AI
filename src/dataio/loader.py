"""
Unified loader for every ECG-Expert-QA task file.

Each task is normalised into a list of dicts with at least:
    - "input"  : the question / prompt the agent must answer
    - "reference" : the gold-standard answer
    - "task"   : task abbreviation (EBK, CK, …)
    - "meta"   : any extra fields (subject_id, ECG params, …)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import CFG

TASK_NAMES: list[str] = list(CFG.task_files.keys())


# ---------------------------------------------------------------------------
# Per-task normalisers
# ---------------------------------------------------------------------------

def _norm_simple_qa(raw: list[dict], task: str) -> list[dict]:
    """EBK, CK, PP – keys Q/A (or Question/Answer)."""
    out: list[dict] = []
    for item in raw:
        q = item.get("Q") or item.get("Question", "")
        a = item.get("A") or item.get("Answer", "")
        meta = {k: v for k, v in item.items() if k not in ("Q", "A", "Question", "Answer")}
        out.append({"input": q, "reference": a, "task": task, "meta": meta})
    return out


def _norm_cmd(raw: list[dict], task: str) -> list[dict]:
    """Cross-modal diagnosis – note: A and Q are *swapped* in the JSON."""
    out: list[dict] = []
    for item in raw:
        question = item.get("Q", "")
        answer = item.get("A", "")
        ecg_meta = {
            k: item[k]
            for k in (
                "subject_id", "study_id", "cart_id", "ecg_time",
                "report_0", "report_1", "report_3",
                "bandwidth", "filtering",
                "rr_interval", "p_onset", "p_end",
                "qrs_onset", "qrs_end", "t_end",
                "p_axis", "qrs_axis", "t_axis",
            )
            if k in item
        }
        out.append({
            "input": answer,      # the *question* is actually in "A"
            "reference": question, # the *answer* is actually in "Q"
            "task": task,
            "meta": ecg_meta,
        })
    return out


def _norm_cd(raw: list[dict], task: str) -> list[dict]:
    """Complex diagnosis – has Disease, Q, A."""
    out: list[dict] = []
    for item in raw:
        out.append({
            "input": item["Q"],
            "reference": item["A"],
            "task": task,
            "meta": {"disease": item.get("Disease", "")},
        })
    return out


def _norm_multi_turn(raw: list[dict], task: str) -> list[dict]:
    """MROD & LTD – keys subject_id, result, QA (list or dict)."""
    out: list[dict] = []
    for item in raw:
        result = item.get("result", "")
        qa = item.get("QA", {})
        if isinstance(qa, list):
            for turn in qa:
                q = turn.get("Q", "")
                a = turn.get("A", "")
                out.append({
                    "input": f"Patient context:\n{result}\n\nQuestion: {q}",
                    "reference": a,
                    "task": task,
                    "meta": {"subject_id": item.get("subject_id")},
                })
        elif isinstance(qa, dict):
            for key, val in qa.items():
                if isinstance(val, dict) and "Question" in val and "Answer" in val:
                    out.append({
                        "input": f"Patient context:\n{result}\n\nQuestion: {val['Question']}",
                        "reference": val["Answer"],
                        "task": task,
                        "meta": {"subject_id": item.get("subject_id"), "qa_type": key},
                    })
    return out


def _norm_mc(raw: list[dict], task: str) -> list[dict]:
    """Memory correction – Conversation is a dict with multiple phases."""
    out: list[dict] = []
    for item in raw:
        result = item.get("result", "")
        conv = item.get("Conversation", {})
        if isinstance(conv, dict):
            # Build full conversation context then ask model to produce corrected memory
            turns: list[str] = []
            for phase_name, phase_turns in conv.items():
                if isinstance(phase_turns, list):
                    for t in phase_turns:
                        if isinstance(t, dict):
                            for role, text in t.items():
                                turns.append(f"{role}: {text}")
            full_conv = "\n".join(turns)
            out.append({
                "input": (
                    f"Patient context:\n{result}\n\n"
                    f"Conversation:\n{full_conv}\n\n"
                    "Based on the conversation, provide corrected clinical findings "
                    "and updated diagnosis considering the patient's memory corrections."
                ),
                "reference": full_conv,  # full conversation is the reference
                "task": task,
                "meta": {"subject_id": item.get("subject_id")},
            })
    return out


def _norm_mee(raw: list[dict], task: str) -> list[dict]:
    """Medical entity extraction – result text + keywords."""
    out: list[dict] = []
    for item in raw:
        result = item.get("result", "")
        keywords = item.get("keywords", "")
        if isinstance(keywords, list):
            keywords = ", ".join(str(k) for k in keywords)
        out.append({
            "input": (
                f"Extract all medical entities (diagnoses, findings, measurements, "
                f"procedures) from the following clinical text:\n\n{result}"
            ),
            "reference": str(keywords),
            "task": task,
            "meta": {"subject_id": item.get("subject_id")},
        })
    return out


def _norm_gra(raw: list[dict], task: str) -> list[dict]:
    """Generate risk assessment – result context + RA text."""
    out: list[dict] = []
    for item in raw:
        result = item.get("result", "")
        ra = item.get("RA", "")
        out.append({
            "input": (
                f"Patient context:\n{result}\n\n"
                "Generate a comprehensive cardiac risk assessment report for this patient."
            ),
            "reference": str(ra),
            "task": task,
            "meta": {"subject_id": item.get("subject_id")},
        })
    return out


def _norm_prtk(raw: list[dict], task: str) -> list[dict]:
    """Patient's right to know."""
    out: list[dict] = []
    for item in raw:
        q = item.get("Question", "")
        a = item.get("Answer", "")
        out.append({"input": q, "reference": a, "task": task, "meta": {}})
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_NORMALISERS: dict[str, Any] = {
    "EBK": lambda raw: _norm_simple_qa(raw, "EBK"),
    "CK":  lambda raw: _norm_simple_qa(raw, "CK"),
    "CMD": lambda raw: _norm_cmd(raw, "CMD"),
    "CD":  lambda raw: _norm_cd(raw, "CD"),
    "MROD": lambda raw: _norm_multi_turn(raw, "MROD"),
    "MC":  lambda raw: _norm_mc(raw, "MC"),
    "LTD": lambda raw: _norm_multi_turn(raw, "LTD"),
    "MEE": lambda raw: _norm_mee(raw, "MEE"),
    "PP":  lambda raw: _norm_simple_qa(raw, "PP"),
    "PRTK": lambda raw: _norm_prtk(raw, "PRTK"),
    "GRA": lambda raw: _norm_gra(raw, "GRA"),
}


def load_task(task: str) -> list[dict]:
    """Load and normalise a single task by abbreviation."""
    if task not in CFG.task_files:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(CFG.task_files)}")
    fpath = CFG.ecg_expert_qa_dir / CFG.task_files[task]
    if not fpath.exists():
        raise FileNotFoundError(f"Task file not found: {fpath}")
    with open(fpath, "r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = _NORMALISERS[task](raw)
    if CFG.max_samples > 0:
        samples = samples[: CFG.max_samples]
    return samples


def load_all_tasks() -> dict[str, list[dict]]:
    """Load every available task."""
    out: dict[str, list[dict]] = {}
    for task in TASK_NAMES:
        try:
            out[task] = load_task(task)
        except FileNotFoundError:
            print(f"[WARN] Skipping {task}: file not found.")
    return out
