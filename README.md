# refine-ai `v0.3.2`

[![PyPI](https://img.shields.io/pypi/v/refine-ai.svg?color=blue)](https://pypi.org/project/refine-ai/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/Toprak1yu/refine-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Toprak1yu/refine-ai/actions/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **Autonomous Data Pipeline & Synthetic Data Agent with Human-in-the-Loop (HITL) Governance.**

**refine-ai** is a stateful, autonomous data engineering agent built on top of [LangGraph](https://github.com/langchain-ai/langgraph), [Polars](https://pola.rs/), and [Rich](https://github.com/Textualize/rich). It deterministically profiles arbitrary datasets, enforces strict governance invariants defined in an operational constitution (`RULES.md`), safely pauses execution via native state-machine interrupts when critical anomalies occur, and resumes with human-approved remediation strategies.

---

## Key Features

- **Constitutional Governance (`RULES.md`):** Enforces strict statistical invariants (null ratio thresholds, Z-score bounds, domain range constraints, class imbalance limits).
- **Human-in-the-Loop (HITL) State Machine:** Employs LangGraph's native `interrupt()` mechanism and persistent SQLite checkpointers (`.checkpoints.db`) to pause graph execution and yield control to the human operator.
- **AI-Driven Streaming Output:** Experience live typewriter-style streaming (`--stream / --no-stream`) for operational logs, status steps, and expert guidance.
- **Interactive CLI Wizard (`refine`):** Run `refine` directly without arguments to launch an interactive wizard that prompts for dataset location and launches the pipeline seamlessly.
- **Planned Execution Manifest:** High-visibility audit plan rendered before modifying data. Review all planned pre-cleaning and remediation actions with one-click approval or individual column-level overrides.
- **Dry-Run Mode (`--dry-run`):** Inspect inferred schemas, anomaly profiles, AI recommendations, and planned execution manifests without modifying source data or writing any files to disk.
- **AI Dataset Schema Inference:** Automatically infers column roles (`id`, `target`, `feature`, `ignore`), semantic types (`numerical`, `categorical`, `binary`, `text`), domain bounds, and canonical aliases via local Ollama LLM with intelligent heuristic fallbacks.
- **Pure Column `DROP` Strategy:** Selecting `DROP` removes only the specified anomalous feature column while strictly preserving 100% of rows across all other features.
- **Distribution-Preserving Synthetic Synthesis:** Gaussian sampling for numerical outliers, frequency-weighted sampling for categoricals, and minority oversampling up to 35% balance for imbalanced target classes.
- **Executive Audit Reports:** Automatically produces clean CSVs alongside an executive Markdown audit report (`*_audit_report.md`) detailing row deltas, human decisions, execution timings, and seed telemetry.

---

## Architecture & Workflow

```
[Raw Dataset (.csv)]
        │
        ▼
[Node: Schema Inference] ───────► Detects ID, Target, and Feature roles (LLM / Heuristic)
        │
        ▼
[Node: Profiler] ───────────────► Computes missingness, Z-scores, domain bounds & imbalance
        │
        ▼
[Node: Deterministic Cleaner] ──► Strips whitespace, standardizes aliases & validates types
        │
        ▼
[Node: Evaluator (Advisor)] ────► Synthesizes remediation advice against RULES.md
        │
        ├── Invariants violated?
        │       │
        │      YES ──► [LangGraph Interrupt] (State checkpointed to SQLite)
        │                     │
        │                     ▼
        │             [Execution Manifest Panel]
        │             ├── Approve AI Manifest? [Y/n]
        │             └── (If 'n') Interactive Column Prompts (DROP / IMPUTE / SYNTHESIS / MANUAL)
        │                     │
        │                     ▼
        │             [Command(resume=...)] ◄─ Graph Resumed
        │       ┌─────────────┘
        ▼       ▼
[Node: Transformer] ────────────► Imputes, removes column, or synthesizes minority records
        │
        ▼
[Node: Exporter] ───────────────► Produces Clean CSV + Governance Audit Markdown Report
```

---

## Project Structure

```
refine-ai/
├── RULES.md                      # Operational governance constitution
├── pyproject.toml                # Project packaging, dependencies & entrypoint
├── ruff.toml                     # Linter & code formatter configuration
├── data/
│   ├── generate_dirty_data.py   # Synthetic dirty customer data generator
│   ├── raw/                     # Untouched source datasets
│   └── processed/               # Cleaned datasets and markdown audit reports
├── refine/
│   ├── __init__.py              # Package version (0.3.2)
│   ├── cli.py                   # Typer & Rich interactive CLI interface
│   ├── graph.py                 # LangGraph StateGraph & interrupt workflow
│   ├── logger.py                # Structured console logging utilities
│   ├── schema_inference.py      # AI & heuristic schema inference engine
│   ├── state.py                 # TypedDict pipeline state schema
│   ├── profiler/
│   │   ├── stats.py             # Polars & SciPy statistical profiling engine
│   │   └── reporters.py         # Rich terminal tables & execution manifests
│   └── tools/
│       ├── advisor.py           # Local LLM / Ollama reasoning & rule-engine fallback
│       ├── cleaner.py           # Deterministic invariant pre-cleaning
│       ├── reporter.py          # Markdown governance audit report generator
│       ├── synthesizer.py       # Distribution-preserving synthetic data generator
│       └── transformer.py       # Remediation strategy execution engine
└── tests/
    └── test_pipeline.py         # End-to-end integration & unit test suite
```

---

## Quickstart

### 1. Installation

Install directly from PyPI via `pip` or `pipx`:

```bash
pip install refine-ai
```

Or run directly without installing in your system Python:

```bash
pipx install refine-ai
```

*(For local development from source)*:

```bash
git clone https://github.com/Toprak1yu/refine-ai.git
cd refine-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. (Optional) Local AI Reasoning with Ollama

`refine-ai` supports local LLM reasoning via [Ollama](https://ollama.com/):

```bash
# Pull and start your preferred model (default: qwen2.5-coder:14b or llama3.2)
ollama run qwen2.5-coder:14b
```

> [!NOTE]
> If Ollama is not installed or offline, `refine-ai` automatically and seamlessly falls back to its deterministic rule-based heuristic engine. No crashes, no manual configuration required.

### 3. Generate Sample Dirty Dataset

Create a realistic test dataset containing missing values, negative ages, severe salary outliers, and class imbalance:

```bash
python data/generate_dirty_data.py
```

### 4. Run the Pipeline

You can run `refine` directly for an interactive prompt:

```bash
refine
```

Or execute directly with full options:

```bash
refine run --file data/raw/dirty_customers.csv --output data/processed/clean_customers.csv
```

---

## CLI Command Reference

### `refine` (Default Interactive Wizard)
Running `refine` without arguments prompts for the file path and launches the pipeline:
```bash
refine
```

### `refine run` (Execute Pipeline)
Executes the autonomous data engineering workflow:
```bash
refine run [OPTIONS]

Options:
  -f, --file PATH            Path to the raw CSV dataset [required]
  -o, --output PATH          Path for the clean CSV destination [default: data/processed/clean_<name>.csv]
  -t, --thread-id TEXT       Session identifier for SQLite checkpointer [default: session_001]
  -s, --seed INTEGER         Random seed for reproducibility [default: 42]
  --stream / --no-stream     Toggle typewriter streaming effect [default: --stream]
  --dry-run                  Inspect manifest & recommendations without writing files
  --help                     Show this message and exit
```

### `refine profile` (Standalone Profiler)
Inspects a dataset's missingness, outliers, and class distributions without executing cleaning:
```bash
refine profile data/raw/dirty_customers.csv
```

### `refine status` (Check State)
Checks whether an active SQLite checkpoint exists for the given thread:
```bash
refine status --thread-id session_001
```

### `refine reset` (Clear Checkpoints)
Removes SQLite checkpoint databases (`.checkpoints.db`) to start fresh:
```bash
refine reset
```

### `refine version` (Display Version)
Prints current package version:
```bash
refine version
```

---

## Example Human-in-the-Loop Interaction

When critical anomalies violate `RULES.md`, execution pauses and displays:

1. **Dataset Profile & Anomaly Audit** table.
2. **Senior Data Architect Reasoning** synthesized by local LLM or rule engine:
   ```text
   [Local Rule-Engine Fallback]:

   • 'age' (Invalid Bounds & Statistical Outliers):
     ➜ Recommended Decision: STATISTICAL_IMPUTE
     ➜ Decision Impact: Both out-of-bounds values and statistical outliers are resolved in a single step using median imputation; preserves data integrity and keeps row count constant.

   • 'churn' (Class Imbalance - 8.0%):
     ➜ Recommended Decision: SYNTHETIC_SYNTHESIS
     ➜ Decision Impact: Synthesize records for minority class (1) to reach 35% balance; mitigates model bias and increases total row count.

   • 'salary' (Statistical Outliers):
     ➜ Recommended Decision: SYNTHETIC_SYNTHESIS
     ➜ Decision Impact: Outliers are replaced with realistic Gaussian distribution values; preserves sample variance and bell-curve geometry.
   ```
3. **Planned Execution Manifest** table:
   ```
   📋 PLANNED EXECUTION MANIFEST
   • CLEANER      country    -> Canonicalized aliases to standard values
   • TRANSFORMER  age        -> STATISTICAL_IMPUTE
   • TRANSFORMER  churn      -> SYNTHETIC_SYNTHESIS (Oversampling minority class)
   • TRANSFORMER  salary     -> SYNTHETIC_SYNTHESIS (Gaussian outlier replacement)
   ```
4. **Interactive Prompt**:
   - Press **`Y`** to approve all recommendations instantly.
   - Press **`n`** to enter interactive column-by-column mode (`DROP`, `STATISTICAL_IMPUTE`, `SYNTHETIC_SYNTHESIS`, or `MANUAL_INPUT`).

Upon completion, output files and executive audit documentation (`clean_customers_audit_report.md`) are saved.

---

## Development & Verification

Run the test suite:

```bash
pytest tests/ -v
```

Check formatting and lint rules:

```bash
ruff check .
ruff format --check .
```

---

## License

MIT License. See `LICENSE` for details.