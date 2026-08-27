# Rubric conformance matrix

Every rubric criterion, where it is implemented, and the command that proves it.
Static criteria are asserted in `tests/test_rubric.py`; runtime deliverables are
asserted by `verify_deliverables.py`.

```bash
uv run pytest tests/test_rubric.py   # static conformance
uv run pytest                        # everything, no Azure calls
python verify_deliverables.py        # after a live run
```

## Section 1 — Imports, Environment, and Logging

| Criterion | Implementation | Proof |
|---|---|---|
| Correct SK modules imported (`Kernel`, `ChatCompletionAgent`, `AgentGroupChat`, `TerminationStrategy`, `AzureChatCompletion`, `OpenAIChatPromptExecutionSettings`, `KernelArguments`) | `final.py` import block | `test_s1_imports_required_semantic_kernel_symbols` resolves each name and asserts its defining module |
| No `<TODO>` left unimplemented | whole file | `test_s1_no_unimplemented_todo_markers_remain` |
| `API_KEY` / `BASE_URL` from `.env` | `load_dotenv()`, `os.getenv("AZURE_OPENAI_KEY")`, `os.getenv("URL")` | `test_s1_environment_variables_come_from_dotenv` |
| `API_VERSION` set correctly | `"2024-05-01-preview"` | `test_s1_api_version_is_the_specified_value` |
| `agent_logger` captures messages to `logs/agent_chat.log` | `_configure_agent_logger()` | `test_s1_agent_logger_writes_to_agent_chat_log` (name, DEBUG, `propagate=False`, exactly one `FileHandler`) |
| Formatter applied, handler added | `Formatter('%(asctime)s - %(name)s:%(message)s')` | `test_s1_formatter_applied_to_handler` |
| `log_agent_message()` logs role, name, content | helper | `test_s1_log_agent_message_records_role_name_and_content` (+ `name=None` → `*`, + exception path) |

## Section 2 — Helper Functions

| Criterion | Implementation | Proof |
|---|---|---|
| `get_csv_name()` lists and selects | interactive picker with re-prompting | `test_get_csv_name_lists_and_selects`, `test_get_csv_name_reports_an_empty_data_directory` |
| `load_csv_file()` loads correctly | `read_csv(encoding="utf-8-sig")` + flatten | `test_load_csv_file_flattens_headers_and_values`, `..._strips_the_utf8_bom`, `..._returns_empty_string_on_failure` |
| Spec/log loaders handle missing files gracefully | `_read_lines()` → `[]` on `FileNotFoundError`/`OSError` | `test_spec_loaders_return_stripped_non_empty_lines`, `test_loaders_handle_missing_files_gracefully` |
| `PythonExecutor.run()` executes safely | Agg backend, `chdir`, fresh namespace, captured output, PNG post-condition | `tests/test_executor.py` — 13 tests incl. no-plot, namespace isolation, cwd restoration |
| `save_final_report()` saves reports | fence-stripping, dir creation, empty-content refusal | `test_save_final_report_round_trips` and siblings |

## Section 3 — Agent Configuration & Factory

| Criterion | Implementation | Proof |
|---|---|---|
| Prompts for cleaning, stats, analysis, code, report | `AGENT_CONFIG` (6 entries, each commented with role + temperature rationale + the failure mode it prevents) | `test_s3_agent_config_has_all_six_documented_prompts`, `test_s3_spec_files_are_embedded_in_the_auditor_prompts` |
| `create_agent()` builds `ChatCompletionAgent`, `KernelArguments` when settings given | factory | `test_s3_create_agent_wraps_settings_in_kernel_arguments`, `..._without_settings_omits_arguments` |
| Temperatures reasonable for role | 0.1 / 0.7 / 0.5 / 0.2 / 1.0 / 0.2 | `test_s3_live_agents_carry_the_required_temperature` (parametrised, read off the **live agent objects**), `test_s3_temperature_ordering_matches_the_role_rationale` |

## Section 4 — Chat Groups & Workflow Execution

| Criterion | Implementation | Proof |
|---|---|---|
| Three chats with the right agents | `analysis_chat`, `code_chat`, `report_chat` | `test_s4_group_chats_hold_the_right_agents_in_order` (parametrised) |
| Termination halts on approval | `ApprovalTerminationStrategy` scoped to the auditors | `tests/test_strategies.py` — approve / fail-containing-"approved" / REVISE / `None` / empty history / out-of-scope agent |
| CSV data loaded and passed to analysis agents | `main()` steps 1–2 | `test_full_workflow_produces_all_four_deliverables` asserts the seeded prompt carries the real rows |
| Code executed with retry logic | `main()` step 6 | `test_code_execution_retries_until_the_script_works` (3 attempts, tracebacks fed back), `test_retry_loop_gives_up_after_the_budget_and_keeps_the_last_attempt` |
| Reports generated from agent logs | `summarize_logs(load_logs(...))` | `test_full_workflow_...` asserts the report prompt contains the log slice; `test_summarize_logs_caps_lines_then_head_and_tail` |
| Human approval incorporated | `ask_approval()` behind `asyncio.to_thread` | `test_human_rejection_stops_before_any_artifact_is_written`, `test_only_a_literal_yes_proceeds` (4 cases), `test_optional_second_gate_before_the_report` |

## Section 5 — Error Handling, Robustness, Code Quality

| Criterion | Implementation | Proof |
|---|---|---|
| Exceptions handled in logging, CSV reading, code execution, file I/O | try/except at each boundary, degrading to `[]` / `""` / `(False, tb)` / a warning | `test_s5_no_bare_or_silently_swallowed_exception_handlers` (AST scan), plus the per-helper failure tests |
| Directories for artifacts and logs exist | import-time `mkdir(parents=True, exist_ok=True)` | `test_s5_required_directories_exist`, `test_s5_directories_are_created_before_the_log_handler` |
| Files saved to correct paths | `BASE_DIR`-anchored path constants | `test_s5_deliverable_paths_match_the_project_specification`; `verify_deliverables.py` at runtime |
| Modular functions, clear structure | single-responsibility helpers, injectable paths | `ruff check` clean; `test_s5_every_public_helper_has_a_docstring` |
| `<TODO>` placeholders indicate where student work is required | the six starter step markers survive as `Step N - ... [starter marker: ...]` section banners | `test_s5_starter_step_banners_are_preserved_in_order` |
| Agent prompts clearly documented | each `AGENT_CONFIG` entry preceded by a comment giving role, temperature rationale and the defended failure mode | reviewed; see `final.py` `AGENT_CONFIG` |

## Part 3 — Deliverables

| Deliverable | Location | Proof (`verify_deliverables.py`) |
|---|---|---|
| Cleaned data | `data-cleaned.json` | parses; `cleaned[]` populated with `x`/`y` rows; `reference_statistics` equals a fresh pandas computation |
| Working visualization code | `artifacts/data_visualization_code.py` | non-empty, compiles, **re-executes standalone** in a temp cwd and regenerates a PNG |
| Generated plot | `artifacts/data_visualization.png` | exists, PNG magic bytes, >5 KB |
| Final report | `artifacts/final_report.md` | every heading from `Report_Instructions.txt`; no `XXXX-XX-XX`; image reference present; ≥4 agent rows in the workflow table |
| Audit trail | `logs/agent_chat.log` | non-empty; all six agents present; approval decision recorded |

## Stand-out items

| Suggestion | Where |
|---|---|
| Enhanced agent instructions with explicit output contracts | `AGENT_CONFIG` — each prompt states its exact output shape and the failure it prevents |
| Visual workflow diagram | `docs/ARCHITECTURE.md` — flowchart + sequence diagram, termination strategies, data flow |
| Comprehensive testing | 126 tests: helpers, executor, strategies, rubric conformance, mocked end-to-end runs incl. retry and approval paths |
| Logging & traceability | timestamps, agent decisions, approval status, retry attempts and numeric-divergence events; `--debug` mode |
| Enhanced reporting | markdown headings/tables driven by `Report_Instructions.txt`, with the generated chart embedded |
| Error handling & robustness | graceful degradation at every I/O boundary; bounded retries; `preflight()` fails fast on credentials |
| Extendability & modularity | `create_agent()` factory + `AGENT_CONFIG` map; optional `config.json` for temperatures, retries and paths |
| Interactive human-in-the-loop | approval after analysis, optional second gate before reporting, `--auto-approve` for automation |
| Documentation & README | `README.md`, `docs/AZURE_SETUP.md`, `docs/ARCHITECTURE.md`, this matrix |
| Creative enhancement | deterministic pandas cross-check of agent arithmetic with a bounded corrective round — catches the exact defect present in the published reference report |
