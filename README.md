# ECG Agentic AI — CardioClinician-Agent Baseline

A multi-agent AI system for cardiological ECG analysis, evaluated on the
[ECG-Expert-QA](https://github.com/MEETI-org/ECG-Expert-QA) benchmark.

## Architecture

```
User Query ──► Chief Coordinator (Router)
                    │
        ┌───────────┴───────────┐
        │                       │
    [FAST PATH]            [TRIAGE PATH]
   (EBK, CK, MEE,         (MROD, PP)
    PRTK)                      │
        │                      ▼
        │              Triage Agent
        │              (urgency, red-flags)
        │                      │
        └──────────┬───────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
   [SPECIALIST]         [VALIDATION PATH]
   Inquiry Agent        (CMD, CD, LTD, MC, GRA)
   ECG Interpreter           │
   Diagnostic Reasoner       ▼
   Prognosis Agent    Data Validator Agent
                      (data quality check)
        │                     │
        └──────────┬──────────┘
                   ▼
        Ethics & Safety Guardian  ◄── parallel review
                   │
                   ▼
            Final Synthesis (CLM)
```

**8 Agents** (LangGraph state-machine orchestration):

1. **Chief Coordinator** – routes tasks, manages revision loop, synthesises final answers
2. **Triage Agent** (Step-Back Prompting) – urgency classification, red-flag screening, clinical framing
3. **Data Validator Agent** (Structured Clinical Reasoning) – data quality assessment, sentinel value detection
4. **Inquiry Agent** – multi-round patient dialogue, memory correction
5. **ECG Interpreter Agent** – cross-modal ECG ↔ text diagnosis, waveform analysis
6. **Diagnostic Reasoner Agent** – complex/long-text diagnosis, differential generation
7. **Prognosis Agent** – prognostic analysis, treatment planning, risk stratification
8. **Ethics & Safety Guardian** – patient rights, risk assessment, ethical review, revision triggers

**Dynamic Invocation** (cost-aware routing):
- **Fast path** (0 extra calls): EBK, CK, MEE, PRTK → Router → Specialist → Synthesis
- **Triage path** (+1 call): MROD, PP → Router → Triage → Specialist → Synthesis
- **Full clinical** (+2 calls): CMD, CD, LTD, MC, GRA → Router → Triage → Validator → Specialist → Synthesis

**Tools**: ECG waveform parser, CHA₂DS₂-VASc / GRACE risk calculators, AHA/ESC guideline lookup.

## Conformal Language Modelling (CLM)

Based on [Quach et al., "Conformal Language Modeling", ICLR 2024](https://arxiv.org/abs/2306.10193).

CLM wraps the final synthesis step with **statistically grounded uncertainty quantification**:

1. **Sample** – draw multiple candidate responses from the LLM at higher temperature
2. **Accept/Reject** – reject near-duplicate candidates (ROUGE-L > λ₁) and low-quality ones (logprob-based quality < λ₂) to keep the set diverse and high-quality
3. **Stop** – halt sampling once set confidence ≥ λ₃ or budget k_max is exhausted
4. **Component Selection** (§4.4) – decompose the best candidate into sentences and measure cross-candidate support to flag **reliable vs. uncertain claims**

The thresholds λ = (λ₁, λ₂, λ₃) can be **calibrated** on a held-out set using the Learn-then-Test (LTT) framework with Bonferroni correction, providing a guarantee:

> *P(prediction set contains ≥1 admissible answer) ≥ 1 − ε*, with confidence 1 − δ.

**Key output fields per sample:**
- `clm_prediction_set.set_size` – number of accepted candidates
- `clm_prediction_set.set_confidence` – stopping-rule confidence
- `clm_reliable_sentences[].reliable` – whether each sentence is independently supported

```
CLM_ENABLED=1          # Enable/disable (default: enabled)
CLM_K_MAX=5            # Sampling budget per query
CLM_TEMPERATURE=0.7    # Temperature for diverse sampling
CLM_EPSILON=0.1        # Target miscoverage rate
CLM_DELTA=0.05         # Confidence level
```

## Baseline Results (gpt-4o-mini, n=20 per task, CLM enabled)

| Task | Description | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-1 | METEOR | CLM Set Size | CLM Confidence | CLM Reliable Frac |
|------|------------|---------|---------|---------|--------|--------|-------------|-----------------|------------------|
| EBK  | ECG Basic Knowledge Q&A | 0.1192 | 0.0520 | 0.0936 | 0.0631 | 0.2936 | 3.0 | 3.0000 | 0.6813 |
| CK   | Cardiology Knowledge Q&A | 0.1612 | 0.0580 | 0.1022 | 0.0846 | 0.3020 | 3.0 | 3.0000 | 0.6277 |
| CMD  | Cross-Modal Diagnosis | 0.1683 | 0.0384 | 0.0933 | 0.0908 | 0.2388 | 3.0 | 3.0000 | 0.5247 |
| CD   | Complex Diagnosis | 0.0915 | 0.0400 | 0.0683 | 0.0459 | 0.2465 | 3.0 | 3.0000 | 0.5810 |
| MROD | Multi-Round Dialogue | 0.1780 | 0.0505 | 0.1072 | 0.0951 | 0.2590 | 3.0 | 3.0000 | 0.6329 |
| MC   | Memory Correction | 0.3102 | 0.0640 | 0.1265 | 0.2019 | 0.1469 | 3.0 | 3.0000 | 0.6436 |
| LTD  | Long-Text Diagnosis | 0.3321 | 0.1187 | 0.1605 | 0.2079 | 0.3695 | 3.0 | 3.0000 | 0.6251 |
| MEE  | Medical Entity Extraction | 0.1782 | 0.0621 | 0.1039 | 0.1362 | 0.1152 | 3.0 | 3.0000 | 0.3644 |
| PP   | Patient Prognosis | 0.1113 | 0.0298 | 0.0753 | 0.0574 | 0.2189 | 3.0 | 3.0000 | 0.6237 |
| PRTK | Patient's Right to Know | 0.2318 | 0.1260 | 0.1905 | 0.1336 | 0.4280 | 3.0 | 3.0000 | 0.4775 |
| GRA  | Generate Risk Assessment | 0.5376 | 0.2120 | 0.2493 | 0.4411 | 0.3219 | 3.0 | 3.0000 | 0.3311 |

## Project Structure

```
src/
├── config.py                 # Central configuration
├── run_baseline.py           # Main entry point
├── dataio/
│   ├── __init__.py
│   └── loader.py             # Unified loader for 11 ECG-Expert-QA tasks
├── agents/
│   ├── __init__.py
│   ├── base.py               # LLM backend factory (OpenAI / HuggingFace)
│   ├── prompts.py            # System prompts per agent
│   ├── conformal.py          # Conformal Language Modelling (CLM)
│   └── coordinator.py        # LangGraph state-machine orchestrator
├── tools/
│   ├── __init__.py
│   ├── ecg_parser.py         # ECG parameter → clinical findings
│   ├── risk_calculators.py   # CHA₂DS₂-VASc, GRACE score
│   └── guideline_lookup.py   # AHA/ESC guideline retrieval
└── evaluation/
    ├── __init__.py
    ├── metrics.py            # ROUGE, BERTScore, Entity F1
    └── evaluator.py          # Task-level evaluation driver
```

## Dependencies

- **LangGraph / LangChain** – agent orchestration
- **OpenAI** – LLM backend (default: gpt-4o-mini)
- **Transformers / PyTorch** – HuggingFace local model fallback
- **rouge-score, bert-score, nltk** – evaluation metrics
- **NumPy, Pandas, SciPy** – data processing
- **WFDB** – ECG signal processing

## Installation

```bash
# Create and activate conda environment
conda create -n ecg-agentic-ai python=3.12
conda activate ecg-agentic-ai
poetry install
```

## Usage

```bash
# Set your OpenAI API key (or add to .env file)
export OPENAI_API_KEY=sk-...

# Run evaluation on all tasks (default 20 samples per task)
MAX_SAMPLES=20 python -m src.run_baseline

# Run specific tasks
python -m src.run_baseline --tasks EBK CK CMD

# Quick smoke test
MAX_SAMPLES=3 python -m src.run_baseline --tasks EBK --verbose

# Use HuggingFace local model
LLM_BACKEND=huggingface python -m src.run_baseline

# Disable CLM (faster, no prediction sets)
python -m src.run_baseline --no-clm --tasks EBK

# Custom CLM budget
python -m src.run_baseline --clm-k-max 8 --tasks EBK
```

Results are written to `results/` as JSON files and a summary table.

## References

- Quach, V., Fisch, A., Schuster, T., Yala, A., Sohn, J.H., Jaakkola, T.S., & Barzilay, R. (2024). *Conformal Language Modeling*. ICLR 2024. [arXiv:2306.10193](https://arxiv.org/abs/2306.10193)
