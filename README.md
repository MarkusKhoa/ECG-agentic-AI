# ECG Agentic AI — CardioClinician-Agent Baseline

A multi-agent AI system for cardiological ECG analysis, evaluated on the
[ECG-Expert-QA](https://github.com/MEETI-org/ECG-Expert-QA) benchmark.

## Architecture

```
User Query ──► Chief Coordinator (Router)
                    │
        ┌───────────┼───────────┬──────────────┐
        ▼           ▼           ▼              ▼
   Inquiry     ECG Interp.  Diagnostic    Prognosis
   Agent       Agent        Reasoner      Agent
        └───────────┼───────────┴──────────────┘
                    ▼
            Ethics & Safety Guardian  ◄── parallel review
                    │
                    ▼
              Final Synthesis
```

**Agents** (LangGraph state-machine orchestration):
- **Inquiry Agent** – multi-round patient dialogue, memory correction
- **ECG Interpreter Agent** – cross-modal ECG ↔ text diagnosis
- **Diagnostic Reasoner Agent** – complex/long-text diagnosis
- **Prognosis Agent** – prognostic analysis, treatment planning
- **Ethics & Safety Guardian** – patient rights, risk assessment, ethical review
- **Chief Coordinator** – routes tasks, synthesises final answers, revision loop

**Tools**: ECG waveform parser, CHA₂DS₂-VASc / GRACE risk calculators, AHA/ESC guideline lookup.

## Baseline Results (gpt-4o-mini, n=20 per task)

| Task | Description | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|------|------------|---------|---------|---------|-------------|
| EBK  | ECG Basic Knowledge Q&A | 0.1254 | 0.0700 | 0.1080 | 0.0664 |
| CK   | Cardiology Knowledge Q&A | 0.1561 | 0.0713 | 0.1045 | 0.0570 |
| CMD  | Cross-Modal Diagnosis | 0.1929 | 0.0620 | 0.1124 | -0.0186 |
| CD   | Complex Diagnosis | 0.1017 | 0.0480 | 0.0776 | 0.0751 |
| MROD | Multi-Round Dialogue | 0.2100 | 0.0812 | 0.1292 | 0.0861 |
| MC   | Memory Correction | 0.3919 | 0.1405 | 0.1917 | -0.0225 |
| LTD  | Long-Text Diagnosis | 0.3281 | 0.1308 | 0.1714 | 0.1011 |
| MEE  | Medical Entity Extraction | 0.2060 | 0.0799 | 0.1273 | -0.2972 |
| PP   | Patient Prognosis | 0.1212 | 0.0331 | 0.0806 | 0.0434 |
| PRTK | Patient's Right to Know | 0.2226 | 0.1219 | 0.1842 | 0.2890 |
| GRA  | Generate Risk Assessment | 0.4865 | 0.2033 | 0.2353 | 0.0015 |

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
```

Results are written to `results/` as JSON files and a summary table.
