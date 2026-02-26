"""
LangGraph-based multi-agent orchestrator for CardioClinician-Agent.

Implements a state machine where:
    1. The Coordinator routes each sample to the correct specialist agent(s).
    2. The Ethics Guardian runs in parallel on every output.
    3. If the Guardian flags high risk / low confidence → loop back to the
       Diagnostic Reasoner for revision.
    4. The Coordinator synthesises the final answer.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.base import get_llm
from src.agents.conformal import (
    ConformalPredictionSet,
    ConformalSampler,
    SamplerConfig,
    make_openai_generate_fn,
)
from src.agents.prompts import (
    COORDINATOR_PROMPT,
    DATA_VALIDATOR_PROMPT,
    DIAGNOSTIC_REASONER_PROMPT,
    ECG_INTERPRETER_PROMPT,
    ETHICS_GUARDIAN_PROMPT,
    INQUIRY_AGENT_PROMPT,
    PROGNOSIS_AGENT_PROMPT,
    TRIAGE_AGENT_PROMPT,
)
from src.config import CFG
from src.tools.ecg_parser import parse_ecg_parameters
from src.tools.guideline_lookup import lookup_guideline
from src.tools.risk_calculators import cha2ds2_vasc_score, grace_score

logger = logging.getLogger(__name__)

# ── Dynamic invocation sets ──────────────────────────────────────────────
NEEDS_TRIAGE: set[str] = {"CMD", "CD", "MROD", "MC", "LTD", "PP", "GRA"}
NEEDS_VALIDATION: set[str] = {"CMD", "CD", "LTD", "MC", "GRA"}

# ── Task → Agent routing table ───────────────────────────────────────────
TASK_AGENT_MAP: dict[str, str] = {
    "EBK": "ecg_interpreter",
    "CK": "diagnostic_reasoner",
    "CMD": "ecg_interpreter",
    "CD": "diagnostic_reasoner",
    "MROD": "inquiry_agent",
    "MC": "inquiry_agent",
    "LTD": "diagnostic_reasoner",
    "MEE": "diagnostic_reasoner",
    "PP": "prognosis_agent",
    "PRTK": "ethics_guardian",
    "GRA": "ethics_guardian",
}


# ── Graph state ──────────────────────────────────────────────────────────
class AgentState(TypedDict):
    task: str
    input: str
    meta: dict[str, Any]
    agent_output: str
    ethics_review: str
    needs_revision: bool
    revision_count: int
    final_answer: str
    trace: list[dict[str, str]]
    # Triage / Data Validation context (empty string when skipped)
    triage_assessment: str
    data_validation: str
    # CLM fields
    clm_prediction_set: dict[str, Any]
    clm_reliable_sentences: list[dict[str, Any]]


# ── Node functions ───────────────────────────────────────────────────────

def _call_llm(system_prompt: str, user_message: str, llm: Any) -> str:
    """Invoke the LLM with a system + user message pair."""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    response = llm.invoke(messages)
    return response.content


def _enrich_cmd_input(state: AgentState) -> str:
    """For CMD tasks, prepend structured ECG parameters to the input."""
    meta = state.get("meta", {})
    ecg_keys = [
        "rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end",
        "t_end", "p_axis", "qrs_axis", "t_axis",
    ]
    ecg_params = {k: meta[k] for k in ecg_keys if k in meta and meta[k] is not None}

    report_parts = []
    for rk in ("report_0", "report_1", "report_3"):
        if rk in meta and meta[rk]:
            report_parts.append(str(meta[rk]))
    report_text = "; ".join(report_parts)

    if ecg_params:
        parsed = parse_ecg_parameters.invoke({**ecg_params, "report_text": report_text})
        ecg_summary = (
            f"ECG Parameters:\n{json.dumps(parsed['derived_parameters'], indent=2)}\n"
            f"Automated Findings: {parsed['summary']}\n\n"
        )
    else:
        ecg_summary = ""

    if report_text:
        ecg_summary += f"Machine Report: {report_text}\n\n"

    return ecg_summary + state["input"]


def _enrich_with_tools(state: AgentState, raw_input: str) -> str:
    """Optionally call guideline lookup / risk calculators based on keywords."""
    task = state["task"]
    enrichment = ""

    if task in ("CD", "LTD", "GRA", "CMD"):
        lower = raw_input.lower()
        conditions_to_check = [
            "atrial_fibrillation", "stemi", "nstemi", "heart_failure",
            "lvh", "bradycardia", "qt_prolongation", "ventricular_tachycardia",
        ]
        for cond in conditions_to_check:
            if cond.replace("_", " ") in lower or cond in lower:
                guideline = lookup_guideline.invoke({"condition": cond})
                enrichment += f"\n[Guideline – {cond}]: {guideline}\n"
                break

    if task == "GRA":
        lower = raw_input.lower()
        if "atrial fibrillation" in lower or "af" in lower:
            result = cha2ds2_vasc_score.invoke({
                "age": 70, "sex": "male", "chf": False,
                "hypertension": True, "stroke_tia_history": False,
                "vascular_disease": False, "diabetes": False,
            })
            enrichment += f"\n[CHA₂DS₂-VASc Example]: {json.dumps(result)}\n"

    return enrichment


def build_graph() -> StateGraph:
    """Construct and compile the LangGraph state machine."""
    llm = get_llm()

    # ── Helper: build upstream context prefix for specialist agents ──
    def _upstream_context(state: AgentState) -> str:
        """Build a context prefix from triage/validation outputs (empty if skipped)."""
        parts: list[str] = []
        triage = state.get("triage_assessment", "")
        validation = state.get("data_validation", "")
        if triage:
            parts.append(f"[TRIAGE ASSESSMENT]:\n{triage}")
        if validation:
            parts.append(f"[DATA VALIDATION REPORT]:\n{validation}")
        return "\n\n".join(parts) + ("\n\n" if parts else "")

    # ── Node: Router ─────────────────────────────────────────────────
    def route_node(state: AgentState) -> AgentState:
        task = state["task"]
        target = TASK_AGENT_MAP.get(task, "diagnostic_reasoner")
        state["trace"] = state.get("trace", [])
        state["trace"].append({"node": "router", "target": target})
        state["revision_count"] = state.get("revision_count", 0)
        state["needs_revision"] = False
        state["triage_assessment"] = state.get("triage_assessment", "")
        state["data_validation"] = state.get("data_validation", "")
        return state

    # ── Node: Triage Agent (Step-Back Prompting) ─────────────────────
    def triage_node(state: AgentState) -> AgentState:
        out = _call_llm(TRIAGE_AGENT_PROMPT, state["input"], llm)
        state["triage_assessment"] = out
        state["trace"].append({"node": "triage_agent", "output_len": len(out)})
        return state

    # ── Node: Data Validator (Structured Clinical Reasoning) ─────────
    def data_validator_node(state: AgentState) -> AgentState:
        # Include triage context if available so validator knows urgency
        triage = state.get("triage_assessment", "")
        prefix = f"[TRIAGE ASSESSMENT]:\n{triage}\n\n" if triage else ""
        out = _call_llm(DATA_VALIDATOR_PROMPT, prefix + state["input"], llm)
        state["data_validation"] = out
        state["trace"].append({"node": "data_validator", "output_len": len(out)})
        return state

    # ── Node: Inquiry Agent ──────────────────────────────────────────
    def inquiry_node(state: AgentState) -> AgentState:
        ctx = _upstream_context(state)
        out = _call_llm(INQUIRY_AGENT_PROMPT, ctx + state["input"], llm)
        state["agent_output"] = out
        state["trace"].append({"node": "inquiry_agent", "output_len": len(out)})
        return state

    # ── Node: ECG Interpreter ────────────────────────────────────────
    def ecg_interpreter_node(state: AgentState) -> AgentState:
        ctx = _upstream_context(state)
        enriched = _enrich_cmd_input(state) if state["task"] == "CMD" else state["input"]
        tool_context = _enrich_with_tools(state, enriched)
        full_input = ctx + enriched + tool_context
        out = _call_llm(ECG_INTERPRETER_PROMPT, full_input, llm)
        state["agent_output"] = out
        state["trace"].append({"node": "ecg_interpreter", "output_len": len(out)})
        return state

    # ── Node: Diagnostic Reasoner ────────────────────────────────────
    def diagnostic_reasoner_node(state: AgentState) -> AgentState:
        ctx = _upstream_context(state)
        tool_context = _enrich_with_tools(state, state["input"])
        revision_note = ""
        if state.get("needs_revision") and state.get("ethics_review"):
            revision_note = (
                f"\n\n[REVISION REQUESTED by Ethics Guardian]:\n"
                f"{state['ethics_review']}\n"
                f"Please revise your response addressing the above concerns.\n"
            )
        full_input = ctx + state["input"] + tool_context + revision_note
        out = _call_llm(DIAGNOSTIC_REASONER_PROMPT, full_input, llm)
        state["agent_output"] = out
        state["trace"].append({"node": "diagnostic_reasoner", "output_len": len(out)})
        return state

    # ── Node: Prognosis Agent ────────────────────────────────────────
    def prognosis_node(state: AgentState) -> AgentState:
        ctx = _upstream_context(state)
        out = _call_llm(PROGNOSIS_AGENT_PROMPT, ctx + state["input"], llm)
        state["agent_output"] = out
        state["trace"].append({"node": "prognosis_agent", "output_len": len(out)})
        return state

    # ── Node: Ethics Guardian ────────────────────────────────────────
    def ethics_guardian_node(state: AgentState) -> AgentState:
        task = state["task"]
        if task in ("PRTK", "GRA"):
            # Primary handler for these tasks
            ctx = _upstream_context(state)
            tool_context = _enrich_with_tools(state, state["input"])
            full_input = ctx + state["input"] + tool_context
            out = _call_llm(ETHICS_GUARDIAN_PROMPT, full_input, llm)
            state["agent_output"] = out
            state["ethics_review"] = ""
            state["needs_revision"] = False
        else:
            # Review mode: check the agent output for safety/ethics
            review_prompt = (
                f"Review the following clinical response for safety, accuracy, "
                f"and ethical compliance:\n\n{state.get('agent_output', '')}\n\n"
                f"If there are concerns, state them clearly. "
                f"If the response is acceptable, respond with 'APPROVED'."
            )
            review = _call_llm(ETHICS_GUARDIAN_PROMPT, review_prompt, llm)
            state["ethics_review"] = review
            state["needs_revision"] = (
                "APPROVED" not in review.upper()
                and state.get("revision_count", 0) < 1
            )
            if state["needs_revision"]:
                state["revision_count"] = state.get("revision_count", 0) + 1
        state["trace"].append({
            "node": "ethics_guardian",
            "mode": "primary" if task in ("PRTK", "GRA") else "review",
            "needs_revision": state.get("needs_revision", False),
        })
        return state

    # ── CLM sampler (initialised once per graph) ──────────────────────
    clm_sampler = None
    if CFG.clm_enabled:
        clm_sampler = ConformalSampler(SamplerConfig(
            k_max=CFG.clm_k_max,
            sampling_temperature=CFG.clm_sampling_temperature,
        ))
        logger.info(
            "CLM enabled: k_max=%d, temp=%.2f",
            CFG.clm_k_max, CFG.clm_sampling_temperature,
        )

    # ── Node: Final synthesis ────────────────────────────────────────
    def synthesise_node(state: AgentState) -> AgentState:
        agent_out = state.get("agent_output", "")
        ethics_out = state.get("ethics_review", "")
        upstream_ctx = _upstream_context(state)

        # Determine the system prompt + user message for synthesis
        if state["task"] in ("PRTK", "GRA"):
            synth_system = ETHICS_GUARDIAN_PROMPT
            synth_user = state["input"]
            baseline_answer = agent_out
        elif ethics_out and "APPROVED" not in ethics_out.upper():
            synth_system = COORDINATOR_PROMPT
            synth_user = (
                f"{upstream_ctx}"
                f"Synthesise the following specialist output and ethics review "
                f"into a final clinical response:\n\n"
                f"Specialist Output:\n{agent_out}\n\n"
                f"Ethics Review:\n{ethics_out}\n\n"
                f"Provide the final integrated answer."
            )
            baseline_answer = _call_llm(synth_system, synth_user, llm)
        else:
            synth_system = COORDINATOR_PROMPT
            synth_user = upstream_ctx + agent_out
            baseline_answer = agent_out

        # ── Conformal Language Modelling ──────────────────────────────
        if clm_sampler is not None:
            gen_fn = make_openai_generate_fn(
                llm, synth_system, synth_user,
                temperature=CFG.clm_sampling_temperature,
            )
            pred_set: ConformalPredictionSet = clm_sampler.sample(gen_fn)

            if pred_set.candidates:
                state["final_answer"] = pred_set.best_candidate
                state["clm_prediction_set"] = pred_set.to_dict()
                state["clm_reliable_sentences"] = pred_set.reliable_components
                state["trace"].append({
                    "node": "synthesise_clm",
                    "set_size": pred_set.set_size,
                    "set_confidence": round(pred_set.set_confidence, 4),
                    "total_drawn": pred_set.total_samples_drawn,
                    "n_reliable": sum(
                        1 for c in pred_set.reliable_components if c.get("reliable")
                    ),
                })
            else:
                # CLM produced no candidates – fall back to baseline
                state["final_answer"] = baseline_answer
                state["clm_prediction_set"] = {}
                state["clm_reliable_sentences"] = []
                state["trace"].append({"node": "synthesise", "clm": "fallback"})
        else:
            state["final_answer"] = baseline_answer
            state["clm_prediction_set"] = {}
            state["clm_reliable_sentences"] = []
            state["trace"].append({"node": "synthesise"})

        return state

    # ── Build graph ──────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    graph.add_node("router", route_node)
    graph.add_node("triage_agent", triage_node)
    graph.add_node("data_validator", data_validator_node)
    graph.add_node("inquiry_agent", inquiry_node)
    graph.add_node("ecg_interpreter", ecg_interpreter_node)
    graph.add_node("diagnostic_reasoner", diagnostic_reasoner_node)
    graph.add_node("prognosis_agent", prognosis_node)
    graph.add_node("ethics_guardian", ethics_guardian_node)
    graph.add_node("synthesise", synthesise_node)

    graph.set_entry_point("router")

    # ── Dynamic invocation: Router → [Triage?] → [Validator?] → Specialist
    #
    # 3 cost tiers:
    #   Fast          (EBK, CK, MEE, PRTK): router → specialist        (+0 LLM)
    #   Triage        (MROD, PP):           router → triage → specialist (+1 LLM)
    #   Full Clinical (CMD, CD, LTD, MC, GRA): router → triage → validator → specialist (+2 LLM)

    def router_edge(state: AgentState) -> str:
        """Route from router: triage-tier or fast-path to specialist."""
        task = state["task"]
        if task in NEEDS_TRIAGE:
            return "triage_agent"
        return TASK_AGENT_MAP.get(task, "diagnostic_reasoner")

    graph.add_conditional_edges(
        "router",
        router_edge,
        {
            "triage_agent": "triage_agent",
            "inquiry_agent": "inquiry_agent",
            "ecg_interpreter": "ecg_interpreter",
            "diagnostic_reasoner": "diagnostic_reasoner",
            "prognosis_agent": "prognosis_agent",
            "ethics_guardian": "ethics_guardian",
        },
    )

    # ── From triage → data validator (full clinical) or specialist ───
    def triage_edge(state: AgentState) -> str:
        """Route from triage: validation-tier or directly to specialist."""
        task = state["task"]
        if task in NEEDS_VALIDATION:
            return "data_validator"
        return TASK_AGENT_MAP.get(task, "diagnostic_reasoner")

    graph.add_conditional_edges(
        "triage_agent",
        triage_edge,
        {
            "data_validator": "data_validator",
            "inquiry_agent": "inquiry_agent",
            "ecg_interpreter": "ecg_interpreter",
            "diagnostic_reasoner": "diagnostic_reasoner",
            "prognosis_agent": "prognosis_agent",
            "ethics_guardian": "ethics_guardian",
        },
    )

    # ── From data validator → specialist agent ───────────────────────
    def validator_edge(state: AgentState) -> str:
        """Route from data validator to the task's specialist agent."""
        return TASK_AGENT_MAP.get(state["task"], "diagnostic_reasoner")

    graph.add_conditional_edges(
        "data_validator",
        validator_edge,
        {
            "inquiry_agent": "inquiry_agent",
            "ecg_interpreter": "ecg_interpreter",
            "diagnostic_reasoner": "diagnostic_reasoner",
            "prognosis_agent": "prognosis_agent",
            "ethics_guardian": "ethics_guardian",
        },
    )

    # ── From specialist agents → ethics review (except PRTK/GRA) ────
    for agent in ("inquiry_agent", "ecg_interpreter", "diagnostic_reasoner", "prognosis_agent"):
        graph.add_edge(agent, "ethics_guardian")

    # ── From ethics guardian → revision or synthesis ──────────────────
    def ethics_edge(state: AgentState) -> str:
        if state.get("needs_revision", False):
            return "diagnostic_reasoner"
        return "synthesise"

    graph.add_conditional_edges(
        "ethics_guardian",
        ethics_edge,
        {
            "diagnostic_reasoner": "diagnostic_reasoner",
            "synthesise": "synthesise",
        },
    )

    graph.add_edge("synthesise", END)

    return graph.compile()
