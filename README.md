# refine-ai

> Autonomous Data Pipeline & Synthetic Data Agent with Human-in-the-Loop (HITL) Governance.

**refine-ai** is an autonomous CLI data engineering agent built on top of LangGraph, Polars, and Rich. It deterministically profiles dirty datasets, enforces strict governance invariants defined in a `RULES.md` constitution, safely interrupts execution when critical anomalies occur, and resumes with human-approved remediation strategies (statistical imputation, row pruning, or synthetic minority oversampling).

## Key Features

- **Constitutional Governance (`RULES.md`):** Enforces explicit invariants (e.g., null ratio thresholds, Z-score bounds, class imbalance limits).

- **Human-in-the-Loop (HITL) State Machine:** Employs LangGraph's native `interrupt()` mechanism and persistent SQLite checkpointers (`.checkpoints.db`) to pause graph execution and wait for operator intervention.

- **Deterministic Pre-Cleaning:** Safely strips whitespace, enforces type casting, and standardizes categorical aliases without unnecessary human overhead.

- **Local AI Advisory:** Uses local LLM inference via Ollama (`llama3.2` or configurable) with automatic heuristic fallback to provide senior data architect reasoning on detected anomalies.

- **Synthetic Data Engine:** Mitigates severe target class imbalance and replaces numerical anomalies using distribution-preserving synthetic data generation.

- **Audit Logging & Governance Reports:** Exports both cleaned datasets and executive Markdown audit reports tracking row deltas, human decisions, and operational steps.

## Architecture & Workflow

```
[Raw Dataset (.csv)]
        │
        ▼
[Node: Profile Dataset] ───────────► Deterministic missingness & Z-score checks
        │
        ▼
[Node: Deterministic Clean] ───────► Canonicalize aliases & strip whitespace
        │
        ▼
[Node: Evaluate Anomalies] ────────► Checks invariants against RULES.md
        │
        ├── Invariants violated?
        │       │
        │      YES ──► [LangGraph Interrupt] (State saved to SQLite)
        │                     │
        │                     ▼
        │             [CLI / TUI Prompt] ──► Human selects strategy
        │                     │              (DROP / IMPUTE / SYNTHESIZE)
        │                     ▼
        │             [Command(resume=...)] ◄─ Resumes graph
        │       ┌─────────────┘
        ▼       ▼
[Node: Apply Resolutions] ─────────► Imputes or synthesizes minority records
        │
        ▼
[Node: Export & Audit] ────────────► Produces clean CSV + Markdown Audit Report
```

## Project Structure

```
refine-ai/
├── RULES.md                      # Operational governance constitution
├── pyproject.toml                # Project packaging and CLI entrypoint
├── data/
│   ├── generate_dirty_data.py   # Test dataset generator
│   ├── raw/                     # Untouched raw datasets
│   └── processed/               # Clean outputs and audit reports
├── refine/
│   ├── cli.py                   # Typer & Rich interactive CLI
│   ├── graph.py                 # LangGraph StateGraph & interrupt workflow
│   ├── state.py                 # TypedDict pipeline state schema
│   ├── profiler/
│   │   ├── stats.py             # Polars & SciPy statistical profiler
│   │   └── reporters.py         # Rich terminal dashboard
│   └── tools/
│       ├── advisor.py           # Local LLM / Ollama reasoning module
│       ├── cleaner.py           # Deterministic invariant transforms
│       ├── reporter.py          # Markdown governance audit generator
│       ├── synthesizer.py       # Synthetic data generator
│       └── transformer.py       # Human resolution execution engine
└── tests/
    └── test_pipeline.py         # Unit tests
```

## Quickstart

### 1. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/username/refine-ai.git
cd refine-ai

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### 2. (Optional) Start Local LLM

Ensure Ollama is installed and running:

```bash
ollama run llama3.2
```

> [!NOTE]
> If Ollama is not active, refine-ai automatically falls back to its built-in rule heuristic engine without throwing errors.

### 3. Generate Dirty Dataset

```bash
python data/generate_dirty_data.py
```

### 4. Run the Pipeline

```bash
refine run --file data/raw/dirty_customers.csv --output data/processed/clean_customers.csv --thread-id session_001
```

## Example HITL Interaction

When an anomaly triggers an interrupt, the CLI renders an interactive panel:

```
╭────────────────── HUMAN-IN-THE-LOOP (HITL) GOVERNANCE REQUIRED ──────────────────╮
│ Critical anomalies require human governance before pipeline execution.            │
╰───────────────────────────────────────────────────────────────────────────────────╯
╭─ Agent Guidance ─────────────────────────────────────────────────────────────────╮
│ 🧠 Senior Data Architect Reasoning (via RULES.md):                               │
│ • 'salary' contains severe Z-score outliers: Recommend SYNTHETIC_SYNTHESIS.      │
│ • 'churn' exhibits severe class imbalance (<10%): Recommend SYNTHETIC_SYNTHESIS.  │
╰──────────────────────────────────────────────────────────────────────────────────╯

Select strategy for feature 'salary' ([1] DROP, [2] STATISTICAL_IMPUTE, [3] SYNTHETIC_SYNTHESIS) (2): 3
Select strategy for feature 'churn' ([1] DROP, [2] STATISTICAL_IMPUTE, [3] SYNTHETIC_SYNTHESIS) (2): 3
Select strategy for feature 'age' ([1] DROP, [2] STATISTICAL_IMPUTE, [3] SYNTHETIC_SYNTHESIS) (2): 2
```

Upon resolution, the output is saved and a governance audit report is written to `data/processed/clean_customers_audit_report.md`.

## Running Tests

```bash
pytest tests/
```

## License

MIT License.