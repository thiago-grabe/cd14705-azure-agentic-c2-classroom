# Agentic Data Analysis Pipeline

An end-to-end, human-in-the-loop data-analysis workflow built with
**Semantic Kernel 1.37** and **Azure OpenAI (gpt-4.1)**. Six specialised agents
across three group chats take a raw CSV and produce a cleaned dataset,
descriptive statistics, a validated visualization, and a structured markdown
report — with a full audit trail of every agent turn.

```
raw CSV → [DataCleaning → DataStatistics → AnalysisChecker] → HUMAN APPROVAL
        → [PythonExecutorAgent + local execution with retries]
        → [ReportGenerator → ReportChecker] → final_report.md
```

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the diagrams and the
design rationale, and **[docs/RUBRIC.md](docs/RUBRIC.md)** for the
criterion-by-criterion conformance matrix.

## Setup

### 1. Azure

Follow **[docs/AZURE_SETUP.md](docs/AZURE_SETUP.md)** to create an Azure AI
Foundry resource and deploy `gpt-4.1`. It covers both the CLI and the portal, and
explains the two failure modes that trip most people up (wrong region, and
confusing the resource with the deployment).

Then:

```bash
cp .env.template .env
# fill in AZURE_OPENAI_KEY and URL
```

`URL` accepts either the resource root (`https://<res>.openai.azure.com/`) or the
full Target URI from the deployment page — `resolve_azure_target()` normalises both.

### 2. Python

Requires **Python 3.13**.

```bash
uv venv --python 3.13
uv pip install --prerelease=allow -r requirements.txt -r requirements-dev.txt -c constraints.txt
```

> **Why `--prerelease=allow`?** semantic-kernel 1.37 depends on
> `azure-ai-agents>=1.2.0b3`, a pre-release. `uv` refuses pre-releases unless
> told otherwise and fails with "your requirements are unsatisfiable"; plain
> `pip` accepts them because the requirement itself pins one.
>
> **Why `-c constraints.txt`?** `requirements.txt` is the course's file, pinning
> only four packages and letting the resolver pick the rest. Today that resolves
> `openai>=3`, which ships against `httpx2`; semantic-kernel 1.37 still imports
> `httpx`, so the install succeeds and then fails at import with
> `ModuleNotFoundError: No module named 'httpx'`. The constraints file pins the
> two transitive packages and leaves `requirements.txt` untouched.

With plain pip (no flag needed): `pip install -r requirements.txt -r requirements-dev.txt -c constraints.txt`.

## Run

```bash
python final.py
```

You will be asked to pick a CSV, shown the cleaning plan, the statistics and the
validation verdict, and asked to approve before anything is written.

### Flags

| Flag | Effect |
|---|---|
| `--csv PATH` | Skip the interactive picker |
| `--auto-approve` | Skip the approval prompts (non-interactive runs) |
| `--approve-report` | Add a second gate before the report phase |
| `--debug` | Echo full agent messages instead of truncating |
| `--max-retries N` | Override the code-execution retry budget (default 10) |
| `--config PATH` | Load an alternative JSON config |

```bash
python final.py --csv data/data-Marketing-1.csv --debug
python final.py --csv data/data-Sensor-1.csv --auto-approve
```

Environment equivalents: `DATA_FILE`, `PIPELINE_DEBUG`,
`AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`.

### Configuration

`config.json` is **optional** — delete it and the pipeline runs on its baked-in
defaults. It layers over `DEFAULT_CONFIG`, and environment variables and CLI
flags layer over that. Agent temperatures and the chats' iteration budgets are
read at import time; retries, approvals and debug are read at run time.

## Outputs

| File | Contents |
|---|---|
| `data-cleaned.json` | Cleaned dataset, removed outliers with reasons, both statistic sets, the validation verdict |
| `artifacts/data_visualization_code.py` | The script that actually executed successfully |
| `artifacts/data_visualization.png` | Original (blue) vs cleaned (green) line chart |
| `artifacts/final_report.md` | The structured report |
| `logs/agent_chat.log` | Every agent turn, plus approval decisions, retries and divergence events |

Verify them:

```bash
python verify_deliverables.py
```

This re-derives the statistics with pandas, re-executes the saved visualization
script in a temp directory, checks the report against every heading in
`specs/Report_Instructions.txt`, and confirms the audit log names all six agents.

## Tests

```bash
uv run pytest          # 126 tests, no Azure calls
uv run ruff check .
```

`tests/test_rubric.py` is a conformance suite: one test per rubric line, so
"meets the rubric" is a command rather than a claim. The rest cover the helpers,
the executor, the termination strategies, and full mocked runs of `main()`
including the retry loop and the approval gate.

## Design notes worth knowing

**The published reference report is arithmetically wrong.** It claims mean
533.19 / median 521.5 / std 37.21 for the cleaned marketing series; the true
values are 527.94 / 518.5 / 38.05. LLMs miscompute quietly, and free-text
hand-offs give nobody a chance to notice. So agents hand off fenced JSON, the
workflow recomputes with pandas, and a mismatch triggers one bounded corrective
round. `tests/test_helpers.py` pins the correct numbers as a regression guard.

**Success is not "the script didn't raise".** The executor deletes the expected
PNG before running, so the file existing afterwards proves the script actually
plotted. A script that runs clean but never calls `savefig` is fed back to the
agent as a failure, with its own stdout attached.

**Datasets differ in shape.** `data-Sensor-2.csv` carries a UTF-8 BOM;
`data-Sensor-1/-2` have a single column and no x axis at all. `profile_dataset()`
is the one place that parses a CSV: it strips the BOM, synthesises an `Index`
axis where there is no date column, and rounds once so the prompt, the JSON, the
generated code and the report all quote identical numbers.

**Prompts are Semantic Kernel templates.** `instructions` are rendered through
`KernelPromptTemplate`, so a literal `{{` raises `TemplateSyntaxError` at the
first invoke. An import-time assertion enforces this permanently.
