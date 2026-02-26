"""
Central configuration for the CardioClinician-Agent baseline.

Environment variables:
    OPENAI_API_KEY      – Required for OpenAI backend.
    LLM_BACKEND         – "openai" (default) or "huggingface".
    OPENAI_MODEL        – e.g. "gpt-4o-mini" (default).
    HF_MODEL_NAME       – e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct".
    ECG_EXPERT_QA_DIR   – Path to the ECG-Expert-QA JSON directory.
    RESULTS_DIR         – Where evaluation outputs are written.
    MAX_SAMPLES         – Cap samples per task (0 = all).  Useful for debugging.
    TEMPERATURE         – LLM temperature (default 0.3).
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


@dataclass
class Config:
    # --- LLM -----------------------------------------------------------------
    llm_backend: str = os.getenv("LLM_BACKEND", "openai")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    hf_model_name: str = os.getenv(
        "HF_MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    temperature: float = float(os.getenv("TEMPERATURE", "0.3"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1024"))

    # --- Data ----------------------------------------------------------------
    ecg_expert_qa_dir: Path = Path(
        os.getenv(
            "ECG_EXPERT_QA_DIR",
            str(_PROJECT_ROOT / "data" / "ecg-expert-qa" / "ECG-Expert-QA"),
        )
    )
    results_dir: Path = Path(
        os.getenv("RESULTS_DIR", str(_PROJECT_ROOT / "results"))
    )
    max_samples: int = int(os.getenv("MAX_SAMPLES", "0"))

    # --- Task file mapping (filename stems) ----------------------------------
    task_files: dict[str, str] = field(default_factory=lambda: {
        "EBK": "ECG Knowledge (Basic Q&A).json",
        "CK": "Cardiology Knowledge (Basic Knowledge Q&A).json",
        "CMD": "Cross modal diagnosis (ECG Text corresponding diagnosis).json",
        "CD": "Complex diagnosis (analysis of difficult diseases).json",
        "MROD": "Medical inquiry, multiple rounds of dialogue between doctors and patients.json",
        "MC": "Memory correction.json",
        "LTD": "Long text diagnosis.json",
        "MEE": "Medical entity extraction.json",
        "PP": "Patient prognosis (prognostic analysis).json",
        "PRTK": "Patient's right to know.json",
        "GRA": "Generate risk assessment.json",
    })

    # --- Conformal Language Modelling (CLM) -----------------------------------
    clm_enabled: bool = bool(int(os.getenv("CLM_ENABLED", "1")))
    clm_epsilon: float = float(os.getenv("CLM_EPSILON", "0.1"))
    clm_delta: float = float(os.getenv("CLM_DELTA", "0.05"))
    clm_k_max: int = int(os.getenv("CLM_K_MAX", "5"))
    clm_sampling_temperature: float = float(os.getenv("CLM_TEMPERATURE", "0.7"))
    clm_calibration_samples: int = int(os.getenv("CLM_CAL_SAMPLES", "0"))

    # --- Safety ---------------------------------------------------------------
    constitutional_prefix: str = (
        "You are a board-certified cardiologist AI assistant. "
        "Think like a real cardiologist: on every clinical case, follow this reasoning loop — "
        "triage first (is this patient unstable?), then focused data acquisition, "
        "validate data quality (distrust raw inputs until verified), "
        "construct a concise problem representation, "
        "generate a ranked differential (most likely, most dangerous, can't-miss), "
        "propose discriminative tests/actions, and reassess whenever new information arrives. "
        "Prioritize non-maleficence, patient autonomy, and justice. "
        "Always express uncertainty when present and recommend confirmatory tests."
    )

    def __post_init__(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)


CFG = Config()
