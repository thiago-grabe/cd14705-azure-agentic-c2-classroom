"""End-to-end workflow tests driven by stub chats -- zero Azure calls.

Covers rubric Section 4's runtime lines: CSV data reaches the analysis agents,
code is executed with retry logic, the report is built from the agent logs, and
human approval actually gates the pipeline.
"""

import json
from pathlib import Path

import pytest
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

import final

PLOT_SCRIPT = """
import os
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 9, 3], color="blue", label="Original Data")
plt.plot([1, 3], [1, 3], color="green", label="Clean Data")
plt.legend()
os.makedirs("artifacts", exist_ok=True)
plt.savefig("artifacts/data_visualization.png", dpi=80)
plt.close()
"""

CLEANING_REPLY = """### CLEANING PLAN
Rule: zero sentinel removal plus an IQR fence.
- 2025-09-05 -> 0 (zero reading)

### CLEANED DATA
```json
{"data_date": "2025-09-01 to 2025-09-02", "rule": "zero sentinel + IQR fence",
 "removed": [{"x": "2025-09-05", "y": 0, "reason": "zero reading"}],
 "cleaned": [{"x": "2025-09-01", "y": 542}, {"x": "2025-09-02", "y": 489}]}
```"""

STATS_REPLY = """| Statistic | Value |
|---|---|
| Count | 2 |

```json
{"count": 2, "mean": 515.5, "median": 515.5, "std": 37.4767, "min": 489.0, "max": 542.0}
```"""

CHECKER_REPLY = '{"title": "Approved", "reason": "", "descriptive_statistics": {"count": 2}}'
REPORT_REPLY = "# Data Analysis Report\n\n**Data Date:** 2025-09-01 to 2025-09-02\n\nbody"
REPORT_OK = "Approved - the report is complete and consistent with the analysis."


class StubChat:
    """Replays scripted turns, recording the prompts it was seeded with."""

    def __init__(self, script):
        self.script = list(script)
        self.prompts: list[str] = []
        self.invocations = 0
        self.is_complete = False

    async def add_chat_message(self, message):
        self.prompts.append(message.content)

    async def invoke(self):
        turns = self.script[min(self.invocations, len(self.script) - 1)]
        self.invocations += 1
        for name, content in turns:
            yield ChatMessageContent(role=AuthorRole.ASSISTANT, name=name, content=content)


# --- Transcript --------------------------------------------------------------
def test_transcript_keeps_the_latest_message_per_agent():
    transcript = final.Transcript("analysis")
    for name, content in [("DataCleaning", "v1"), ("AnalysisChecker", "fail"),
                          ("DataCleaning", "v2"), ("AnalysisChecker", "pass")]:
        transcript.record(ChatMessageContent(role=AuthorRole.ASSISTANT, name=name, content=content))
    assert transcript.latest("DataCleaning") == "v2"
    assert transcript.latest("AnalysisChecker") == "pass"
    assert transcript.turns("AnalysisChecker") == 2
    assert transcript.latest("Nobody", "fallback") == "fallback"
    assert len(transcript.messages) == 4


def test_transcript_coerces_missing_content_to_empty_string():
    transcript = final.Transcript()
    transcript.record(ChatMessageContent(role=AuthorRole.ASSISTANT, name=None, content=None))
    assert transcript.latest("*") == ""


async def test_run_group_chat_logs_every_message_and_captures_per_agent(tmp_path):
    log_file = tmp_path / "agent_chat.log"
    final._configure_agent_logger(log_file)
    try:
        chat = StubChat([[("DataCleaning", CLEANING_REPLY), ("AnalysisChecker", CHECKER_REPLY)]])
        transcript = await final.run_group_chat(chat, "seed prompt", "analysis")
    finally:
        final._configure_agent_logger(final.AGENT_LOG_PATH)

    assert chat.prompts == ["seed prompt"]
    assert transcript.latest("DataCleaning").startswith("### CLEANING PLAN")
    assert final.is_approved(transcript.latest("AnalysisChecker"))
    text = log_file.read_text(encoding="utf-8")
    assert "DataCleaning" in text and "AnalysisChecker" in text
    assert "seed prompt" in text, "the seeded user prompt belongs in the audit trail"


async def test_run_group_chat_raises_when_a_chat_produces_nothing():
    with pytest.raises(RuntimeError, match="produced no messages"):
        await final.run_group_chat(StubChat([[]]), "seed", "analysis")


# --- Full workflow -----------------------------------------------------------
@pytest.fixture
def workflow(monkeypatch, tmp_path):
    """Point every deliverable at tmp_path and stub out Azure entirely."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr(final, "BASE_DIR", tmp_path)
    monkeypatch.setattr(final, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(final, "CLEANED_JSON_PATH", tmp_path / "data-cleaned.json")
    monkeypatch.setattr(final, "VIZ_CODE_PATH", artifacts / "data_visualization_code.py")
    monkeypatch.setattr(final, "VIZ_IMAGE_PATH", artifacts / "data_visualization.png")
    monkeypatch.setattr(final, "FINAL_REPORT_PATH", artifacts / "final_report.md")

    async def no_preflight():
        return None

    monkeypatch.setattr(final, "preflight", no_preflight)
    return tmp_path


def install_chats(monkeypatch, analysis, code, report):
    monkeypatch.setattr(final, "analysis_chat", analysis)
    monkeypatch.setattr(final, "code_chat", code)
    monkeypatch.setattr(final, "report_chat", report)


async def test_full_workflow_produces_all_four_deliverables(workflow, monkeypatch, solution_dir):
    analysis = StubChat([[("DataCleaning", CLEANING_REPLY), ("DataStatistics", STATS_REPLY),
                          ("AnalysisChecker", CHECKER_REPLY)]])
    code = StubChat([[("PythonExecutorAgent", PLOT_SCRIPT)]])
    report = StubChat([[("ReportGenerator", REPORT_REPLY), ("ReportChecker", REPORT_OK)]])
    install_chats(monkeypatch, analysis, code, report)

    csv = str(solution_dir / "data" / "data-Marketing-1.csv")
    exit_code = await final.main(["--csv", csv, "--auto-approve"])
    assert exit_code == 0

    # Rubric S4: the CSV data actually reached the analysis agents.
    seed = analysis.prompts[0]
    assert "Website_Visits" in seed and "2025-09-01" in seed and "542" in seed
    assert "REFERENCE STATISTICS" in seed

    # Deliverables.
    payload = json.loads((workflow / "data-cleaned.json").read_text(encoding="utf-8"))
    assert payload["cleaned"] == [{"x": "2025-09-01", "y": 542}, {"x": "2025-09-02", "y": 489}]
    assert payload["removed"][0]["reason"] == "zero reading"
    assert payload["reference_statistics"]["count"] == 2
    assert payload["validation"]["status"] == "Approved"
    assert payload["source_file"].endswith("data-Marketing-1.csv")

    assert (workflow / "artifacts" / "data_visualization.png").exists()
    assert "savefig" in (workflow / "artifacts" / "data_visualization_code.py").read_text(encoding="utf-8")
    assert (workflow / "artifacts" / "final_report.md").read_text(encoding="utf-8").startswith(
        "# Data Analysis Report"
    )

    # Rubric S4: the code agent got literal arrays, not a CSV blob to re-parse.
    assert "original_x = " in code.prompts[0] and "cleaned_y = " in code.prompts[0]
    # Rubric S4: the report was generated from the agent logs.
    assert "Agent interaction log" in report.prompts[0]
    assert "DataCleaning" in report.prompts[0]


async def test_human_rejection_stops_before_any_artifact_is_written(workflow, monkeypatch, solution_dir):
    analysis = StubChat([[("DataCleaning", CLEANING_REPLY), ("DataStatistics", STATS_REPLY),
                          ("AnalysisChecker", CHECKER_REPLY)]])
    code = StubChat([[("PythonExecutorAgent", PLOT_SCRIPT)]])
    report = StubChat([[("ReportGenerator", REPORT_REPLY), ("ReportChecker", REPORT_OK)]])
    install_chats(monkeypatch, analysis, code, report)
    monkeypatch.setattr("builtins.input", lambda *_: "no")

    csv = str(solution_dir / "data" / "data-Marketing-1.csv")
    assert await final.main(["--csv", csv]) == 1
    assert not (workflow / "data-cleaned.json").exists()
    assert not (workflow / "artifacts" / "final_report.md").exists()
    assert code.invocations == 0, "the code chat must never run without approval"


@pytest.mark.parametrize("answer", ["no", "", "y", "YES please"])
async def test_only_a_literal_yes_proceeds(workflow, monkeypatch, solution_dir, answer):
    install_chats(
        monkeypatch,
        StubChat([[("DataCleaning", CLEANING_REPLY), ("DataStatistics", STATS_REPLY),
                   ("AnalysisChecker", CHECKER_REPLY)]]),
        StubChat([[("PythonExecutorAgent", PLOT_SCRIPT)]]),
        StubChat([[("ReportGenerator", REPORT_REPLY), ("ReportChecker", REPORT_OK)]]),
    )
    monkeypatch.setattr("builtins.input", lambda *_: answer)
    csv = str(solution_dir / "data" / "data-Marketing-1.csv")
    assert await final.main(["--csv", csv]) == 1


async def test_code_execution_retries_until_the_script_works(workflow, monkeypatch, solution_dir, capsys):
    analysis = StubChat([[("DataCleaning", CLEANING_REPLY), ("DataStatistics", STATS_REPLY),
                          ("AnalysisChecker", CHECKER_REPLY)]])
    code = StubChat([
        [("PythonExecutorAgent", "raise ValueError('first attempt explodes')")],
        [("PythonExecutorAgent", "print('second attempt saves nothing')")],
        [("PythonExecutorAgent", PLOT_SCRIPT)],
    ])
    report = StubChat([[("ReportGenerator", REPORT_REPLY), ("ReportChecker", REPORT_OK)]])
    install_chats(monkeypatch, analysis, code, report)

    csv = str(solution_dir / "data" / "data-Marketing-1.csv")
    assert await final.main(["--csv", csv, "--auto-approve"]) == 0

    assert code.invocations == 3, "one initial generation plus two repair rounds"
    assert "first attempt explodes" in code.prompts[1], "the traceback must reach the agent"
    assert "was not created" in code.prompts[2], "the no-plot diagnosis must reach the agent too"
    assert (workflow / "artifacts" / "data_visualization.png").exists()
    assert "execution attempt 3/" in capsys.readouterr().out


async def test_retry_loop_gives_up_after_the_budget_and_keeps_the_last_attempt(
    workflow, monkeypatch, solution_dir
):
    analysis = StubChat([[("DataCleaning", CLEANING_REPLY), ("DataStatistics", STATS_REPLY),
                          ("AnalysisChecker", CHECKER_REPLY)]])
    code = StubChat([[("PythonExecutorAgent", "raise ValueError('always broken')")]])
    report = StubChat([[("ReportGenerator", REPORT_REPLY), ("ReportChecker", REPORT_OK)]])
    install_chats(monkeypatch, analysis, code, report)

    csv = str(solution_dir / "data" / "data-Marketing-1.csv")
    assert await final.main(["--csv", csv, "--auto-approve", "--max-retries", "3"]) == 1
    assert code.invocations == 3, "3 attempts == 1 generation + 2 repairs, then stop"
    assert report.invocations == 0, "no report without a plot"
    assert "always broken" in (workflow / "artifacts" / "data_visualization_code.py").read_text()


async def test_numeric_divergence_triggers_one_bounded_corrective_round(
    workflow, monkeypatch, solution_dir, capsys
):
    """The published reference report is off by ~1%; the pipeline must notice."""
    wrong_stats = STATS_REPLY.replace('"mean": 515.5', '"mean": 999.0')
    analysis = StubChat([
        [("DataCleaning", CLEANING_REPLY), ("DataStatistics", wrong_stats), ("AnalysisChecker", CHECKER_REPLY)],
        [("DataStatistics", STATS_REPLY), ("AnalysisChecker", CHECKER_REPLY)],
    ])
    install_chats(
        monkeypatch, analysis,
        StubChat([[("PythonExecutorAgent", PLOT_SCRIPT)]]),
        StubChat([[("ReportGenerator", REPORT_REPLY), ("ReportChecker", REPORT_OK)]]),
    )

    csv = str(solution_dir / "data" / "data-Marketing-1.csv")
    assert await final.main(["--csv", csv, "--auto-approve"]) == 0

    assert analysis.invocations == 2, "exactly one corrective round, not a loop"
    assert "REFERENCE STATISTICS (authoritative)" in analysis.prompts[1]
    out = capsys.readouterr().out
    assert "agent statistics disagree" in out
    payload = json.loads((workflow / "data-cleaned.json").read_text(encoding="utf-8"))
    assert payload["agent_statistics"]["mean"] == 515.5, "the corrected stats are the ones stored"


async def test_matching_statistics_skip_the_corrective_round(workflow, monkeypatch, solution_dir):
    analysis = StubChat([[("DataCleaning", CLEANING_REPLY), ("DataStatistics", STATS_REPLY),
                          ("AnalysisChecker", CHECKER_REPLY)]])
    install_chats(
        monkeypatch, analysis,
        StubChat([[("PythonExecutorAgent", PLOT_SCRIPT)]]),
        StubChat([[("ReportGenerator", REPORT_REPLY), ("ReportChecker", REPORT_OK)]]),
    )
    csv = str(solution_dir / "data" / "data-Marketing-1.csv")
    assert await final.main(["--csv", csv, "--auto-approve"]) == 0
    assert analysis.invocations == 1


async def test_optional_second_gate_before_the_report(workflow, monkeypatch, solution_dir):
    answers = iter(["yes", "no"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    install_chats(
        monkeypatch,
        StubChat([[("DataCleaning", CLEANING_REPLY), ("DataStatistics", STATS_REPLY),
                   ("AnalysisChecker", CHECKER_REPLY)]]),
        StubChat([[("PythonExecutorAgent", PLOT_SCRIPT)]]),
        report := StubChat([[("ReportGenerator", REPORT_REPLY), ("ReportChecker", REPORT_OK)]]),
    )
    csv = str(solution_dir / "data" / "data-Marketing-1.csv")
    assert await final.main(["--csv", csv, "--approve-report"]) == 1
    assert report.invocations == 0
    assert (workflow / "artifacts" / "data_visualization.png").exists(), "the plot survives"


async def test_workflow_generalises_to_a_dataset_with_no_date_column(
    workflow, monkeypatch, solution_dir
):
    index_cleaning = CLEANING_REPLY.replace('"2025-09-01", "y": 542', '0, "y": 0.12').replace(
        '"2025-09-02", "y": 489', '1, "y": 0.83'
    ).replace('"x": "2025-09-05", "y": 0', '"x": 4, "y": 24')
    analysis = StubChat([[("DataCleaning", index_cleaning), ("DataStatistics", STATS_REPLY),
                          ("AnalysisChecker", CHECKER_REPLY)]])
    code = StubChat([[("PythonExecutorAgent", PLOT_SCRIPT)]])
    install_chats(monkeypatch, analysis, code,
                  StubChat([[("ReportGenerator", REPORT_REPLY), ("ReportChecker", REPORT_OK)]]))

    csv = str(solution_dir / "data" / "data-Sensor-1.csv")
    assert await final.main(["--csv", csv, "--auto-approve"]) == 0
    assert "X_LABEL = 'Index'" in code.prompts[0]
    assert "X_IS_CATEGORICAL = False" in code.prompts[0]
    payload = json.loads((workflow / "data-cleaned.json").read_text(encoding="utf-8"))
    assert payload["x_label"] == "Index"


async def test_cli_flags_layer_over_the_configuration(monkeypatch):
    monkeypatch.setattr(final, "CONFIG", final.load_config(Path("nonexistent.json")))
    final.apply_cli_overrides(final.parse_args(["--debug", "--max-retries", "4", "--auto-approve"]))
    assert final.CONFIG["debug"] is True
    assert final.CONFIG["max_code_retries"] == 4
    assert final.CONFIG["approvals"]["after_analysis"] is False


async def test_main_does_not_leak_configuration_between_runs(workflow, monkeypatch, solution_dir):
    """A second run in the same process must not inherit the first run's flags."""
    csv = str(solution_dir / "data" / "data-Marketing-1.csv")
    install_chats(
        monkeypatch,
        StubChat([[("DataCleaning", CLEANING_REPLY), ("DataStatistics", STATS_REPLY),
                   ("AnalysisChecker", CHECKER_REPLY)]]),
        StubChat([[("PythonExecutorAgent", PLOT_SCRIPT)]]),
        StubChat([[("ReportGenerator", REPORT_REPLY), ("ReportChecker", REPORT_OK)]]),
    )
    assert await final.main(["--csv", csv, "--auto-approve", "--max-retries", "2"]) == 0
    assert final.CONFIG["approvals"]["after_analysis"] is False

    install_chats(
        monkeypatch,
        StubChat([[("DataCleaning", CLEANING_REPLY), ("DataStatistics", STATS_REPLY),
                   ("AnalysisChecker", CHECKER_REPLY)]]),
        StubChat([[("PythonExecutorAgent", PLOT_SCRIPT)]]),
        report := StubChat([[("ReportGenerator", REPORT_REPLY), ("ReportChecker", REPORT_OK)]]),
    )
    monkeypatch.setattr("builtins.input", lambda *_: "no")
    assert await final.main(["--csv", csv]) == 1, "approval must be required again"
    assert report.invocations == 0
    assert final.CONFIG["max_code_retries"] == 10, "--max-retries must not persist"
