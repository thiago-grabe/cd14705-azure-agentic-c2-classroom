"""Helper-function tests (rubric Section 2 + the parsing utilities)."""

import json

import pytest

import final


# --- Azure endpoint normalisation -------------------------------------------
@pytest.mark.parametrize(
    "raw, endpoint, deployment",
    [
        ("https://r.openai.azure.com/", "https://r.openai.azure.com/", "gpt-4.1"),
        ("https://r.openai.azure.com", "https://r.openai.azure.com/", "gpt-4.1"),
        ("https://r.cognitiveservices.azure.com/openai/deployments/", "https://r.cognitiveservices.azure.com/", "gpt-4.1"),
        ("https://r.openai.azure.com/openai/deployments/my-dep", "https://r.openai.azure.com/", "my-dep"),
        ("https://r.services.ai.azure.com/openai/", "https://r.services.ai.azure.com/", "gpt-4.1"),
        ("r.openai.azure.com", "https://r.openai.azure.com/", "gpt-4.1"),
        ('"https://r.openai.azure.com/"', "https://r.openai.azure.com/", "gpt-4.1"),
    ],
)
def test_resolve_azure_target_normalises_every_documented_url_shape(raw, endpoint, deployment):
    result = final.resolve_azure_target(raw, "gpt-4.1", "2024-05-01-preview")
    assert result["endpoint"] == endpoint
    assert result["deployment_name"] == deployment
    assert "base_url" not in result, "base_url and azure_endpoint are mutually exclusive downstream"


def test_resolve_azure_target_recovers_deployment_and_version_from_a_target_uri():
    result = final.resolve_azure_target(
        "https://r.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview",
        "fallback-deployment",
        "2024-05-01-preview",
    )
    assert result == {
        "endpoint": "https://r.openai.azure.com/",
        "deployment_name": "gpt-4.1",
        "api_version": "2025-01-01-preview",
    }


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_resolve_azure_target_rejects_a_missing_url_with_an_actionable_message(raw):
    with pytest.raises(ValueError, match="URL is not set"):
        final.resolve_azure_target(raw, "gpt-4.1", "2024-05-01-preview")


# --- Spec and log loaders ----------------------------------------------------
def test_spec_loaders_return_stripped_non_empty_lines():
    quality = final.load_quality_instructions("Data_Quality_Instructions.txt")
    reports = final.load_reports_instructions("Report_Instructions.txt")
    assert quality and reports
    assert all(line == line.strip() and line for line in quality + reports)
    assert "# Data Analysis Report" in reports


@pytest.mark.parametrize(
    "loader", [final.load_quality_instructions, final.load_reports_instructions, final.load_logs]
)
def test_loaders_handle_missing_files_gracefully(loader):
    assert loader("definitely-not-here.txt") == []


def test_load_logs_reads_from_the_logs_directory(tmp_path):
    (tmp_path / "sample.log").write_text("first\n\n  second  \n", encoding="utf-8")
    assert final.load_logs("sample.log", tmp_path) == ["first", "second"]


# --- CSV handling ------------------------------------------------------------
def test_load_csv_file_flattens_headers_and_values(marketing_csv):
    flat = final.load_csv_file(marketing_csv)
    assert flat.startswith("Date, Website_Visits, ")
    assert "2025-09-01" in flat and "531" in flat
    assert ", " in flat


def test_load_csv_file_strips_the_utf8_bom(bom_csv):
    flat = final.load_csv_file(bom_csv)
    assert flat.startswith("Sensor Value, ")
    assert "﻿" not in flat


def test_load_csv_file_returns_empty_string_on_failure(tmp_path):
    assert final.load_csv_file(tmp_path / "nope.csv") == ""


def test_get_csv_name_lists_and_selects(monkeypatch, capsys):
    answers = iter(["banana", "99", "1"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    selected = final.get_csv_name()
    out = capsys.readouterr().out
    assert "data-Marketing-1.csv" in out, "the picker must list the available files"
    assert selected.endswith("data-Marketing-1.csv"), "sorted order, first entry"
    assert "Invalid input" in out and "between 1 and" in out, "bad input re-prompts"


def test_get_csv_name_reports_an_empty_data_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="No CSV files"):
        final.get_csv_name(tmp_path)


# --- Dataset profiling -------------------------------------------------------
@pytest.mark.parametrize(
    "name, x_label, y_label, rows, categorical",
    [
        ("data-Marketing-1.csv", "Date", "Website_Visits", 20, True),
        ("data-Sensor-1.csv", "Index", "Sensor Value", 19, False),
        ("data-Sensor-2.csv", "Index", "Sensor Value", 40, False),
        ("data-Sensor-3.csv", "Time", "Sensor Value", 41, False),
    ],
)
def test_profile_dataset_handles_every_csv_shape(solution_dir, name, x_label, y_label, rows, categorical):
    profile = final.profile_dataset(solution_dir / "data" / name)
    assert (profile.x_label, profile.y_label) == (x_label, y_label)
    assert len(profile.x) == len(profile.y) == rows
    assert profile.x_is_categorical is categorical
    assert all(isinstance(v, float) for v in profile.y)


def test_profile_dataset_synthesises_an_index_axis_for_single_column_files(solution_dir):
    """No date column means no dates -- the agents must never invent one."""
    profile = final.profile_dataset(solution_dir / "data" / "data-Sensor-1.csv")
    assert profile.x == list(range(19))
    assert profile.data_date  # falls back to the run date


def test_profile_dataset_truncates_oversized_inputs(tmp_path, capsys):
    csv = tmp_path / "big.csv"
    csv.write_text("Idx,Value\n" + "\n".join(f"{i},{i}" for i in range(50)), encoding="utf-8")
    profile = final.profile_dataset(csv, max_rows=10)
    assert len(profile.y) == 10
    assert "truncating" in capsys.readouterr().out


# --- Reference statistics ----------------------------------------------------
def test_compute_reference_statistics_matches_pandas_on_the_marketing_dataset(solution_dir):
    """Regression guard for the published reference report, which claims
    mean 533.19 / median 521.5 / std 37.21 for this exact cleaned series."""
    profile = final.profile_dataset(solution_dir / "data" / "data-Marketing-1.csv")
    cleaned = [v for v in profile.y if v not in (0.0, 2500.0, 5545.0)]
    stats = final.compute_reference_statistics(cleaned)
    assert stats == {
        "count": 16,
        "mean": 527.9375,
        "median": 518.5,
        "std": 38.0499,
        "min": 488.0,
        "max": 621.0,
    }


def test_compute_reference_statistics_on_empty_input():
    assert final.compute_reference_statistics([]) == {}


def test_statistics_agree_flags_the_reference_reports_error():
    reference = {"count": 16, "mean": 527.9375, "median": 518.5, "std": 38.0499, "min": 488.0, "max": 621.0}
    agree, problems = final.statistics_agree(dict(reference), reference, 0.01)
    assert agree and not problems

    published = {"count": 16, "mean": 533.19, "median": 521.5, "std": 37.21, "min": 488, "max": 621}
    agree, problems = final.statistics_agree(published, reference, 0.001)
    assert not agree
    assert any("mean" in p for p in problems)

    agree, problems = final.statistics_agree(None, reference, 0.01)
    assert not agree and "parseable json" in problems[0]


# --- Text utilities ----------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("print('hi')", "print('hi')"),
        ("```python\nprint('hi')\n```", "print('hi')"),
        ("```\nprint('hi')\n```", "print('hi')"),
        ("Here you go:\n```python\nprint('hi')\n```\nEnjoy!", "print('hi')"),
        ("```py\nx = 1\n```\n```python\nx = 1\ny = 2\nz = 3\n```", "x = 1\ny = 2\nz = 3"),
        ("", ""),
        (None, ""),
    ],
)
def test_strip_code_fences(text, expected):
    assert final.strip_code_fences(text) == expected


def test_extract_json_block_prefers_the_last_valid_object():
    text = 'plan text\n```json\n{"cleaned": [{"x": 1, "y": 2}]}\n```\ntrailing prose'
    assert final.extract_json_block(text) == {"cleaned": [{"x": 1, "y": 2}]}
    assert final.extract_json_block("no json here") is None
    assert final.extract_json_block(None) is None
    assert final.extract_json_block('prose {"a": 1} more prose') == {"a": 1}


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"title": "Approved", "reason": ""}', True),
        ('{"title": "Failed", "reason": "statistics were not approved"}', False),
        ("REVISE\n1. Section 2 is missing the median.", False),
        ("Approved - the report is complete and consistent with the analysis.", True),
        ("The analysis is not approved.", False),
        ("", False),
        (None, False),
    ],
)
def test_is_approved(text, expected):
    assert final.is_approved(text) is expected


def test_summarize_logs_caps_lines_then_head_and_tail():
    assert final.summarize_logs(["a", "b"]) == "a\nb"

    capped = final.summarize_logs(["x" * 5000], max_total=100000, max_line=100)
    assert capped.startswith("x" * 100) and "+4900 chars" in capped

    long = final.summarize_logs([f"line-{i}" for i in range(5000)], max_total=1000, max_line=100)
    assert len(long) < 1200
    assert "characters elided" in long
    assert long.startswith("line-0") and long.rstrip().endswith("line-4999")


# --- Report saving -----------------------------------------------------------
def test_save_final_report_round_trips(tmp_path, capsys):
    target = tmp_path / "nested" / "final_report.md"
    final.save_final_report("# Data Analysis Report\n\nbody", target)
    assert target.read_text(encoding="utf-8") == "# Data Analysis Report\n\nbody\n"
    assert "Report saved to" in capsys.readouterr().out


def test_save_final_report_unwraps_a_fenced_report(tmp_path):
    target = tmp_path / "final_report.md"
    final.save_final_report("```markdown\n# Data Analysis Report\n\nbody\n```", target)
    assert target.read_text(encoding="utf-8").startswith("# Data Analysis Report")


def test_save_final_report_refuses_empty_content(tmp_path, capsys):
    target = tmp_path / "final_report.md"
    final.save_final_report("   ", target)
    assert not target.exists()
    assert "produced no content" in capsys.readouterr().out


# --- Configuration -----------------------------------------------------------
def test_load_config_layers_an_optional_file_over_the_defaults(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"max_code_retries": 3, "temperatures": {"DataCleaning": 0.9}}), encoding="utf-8")
    config = final.load_config(config_file)
    assert config["max_code_retries"] == 3
    assert config["temperatures"]["DataCleaning"] == 0.9
    assert config["temperatures"]["AnalysisChecker"] == 0.2, "unspecified keys keep their defaults"


def test_load_config_survives_a_malformed_file(tmp_path, capsys):
    config_file = tmp_path / "config.json"
    config_file.write_text("{ not json", encoding="utf-8")
    assert final.load_config(config_file)["max_code_retries"] == 10
    assert "ignoring config.json" in capsys.readouterr().out


def test_load_config_without_a_file_returns_the_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("PIPELINE_DEBUG", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    assert final.load_config(tmp_path / "absent.json") == final.DEFAULT_CONFIG


def test_load_config_respects_environment_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
    monkeypatch.setenv("PIPELINE_DEBUG", "true")
    config = final.load_config(tmp_path / "absent.json")
    assert config["deployment_name"] == "gpt-4.1-mini"
    assert config["debug"] is True


def test_load_config_never_shares_mutable_state_with_the_defaults(tmp_path):
    """A shallow copy would share the nested dicts and let any caller rewrite
    the module defaults for every later call."""
    first = final.load_config(tmp_path / "absent.json")
    first["temperatures"]["DataCleaning"] = 0.0
    first["approvals"]["after_analysis"] = False
    assert final.DEFAULT_CONFIG["temperatures"]["DataCleaning"] == 0.7
    assert final.DEFAULT_CONFIG["approvals"]["after_analysis"] is True
    assert final.load_config(tmp_path / "absent.json")["temperatures"]["DataCleaning"] == 0.7


def test_extract_statistics_drills_into_a_checker_verdict():
    """A live run caught DataStatistics answering with the auditor's verdict
    shape; the numbers must still be recovered rather than compared as-is."""
    verdict = json.dumps({
        "title": "Approved",
        "reason": "matches",
        "original_data": [{"x": "2025-09-01", "y": 542.0}],
        "descriptive_statistics": {"count": 16, "mean": 527.9375, "median": 518.5,
                                   "std": 38.0499, "min": 488.0, "max": 621.0},
    })
    assert final.extract_statistics(verdict) == {
        "count": 16, "mean": 527.9375, "median": 518.5, "std": 38.0499, "min": 488.0, "max": 621.0
    }


def test_extract_statistics_reads_a_plain_statistics_block():
    plain = '```json\n{"count": 2, "mean": 515.5, "median": 515.5, "std": 37.4767, "min": 489.0, "max": 542.0}\n```'
    assert final.extract_statistics(plain)["mean"] == 515.5


def test_extract_statistics_rejects_unrelated_json():
    assert final.extract_statistics('{"cleaned": [{"x": 1, "y": 2}]}') is None
    assert final.extract_statistics("no json at all") is None
    assert final.extract_statistics(None) is None


def test_extract_statistics_strips_extraneous_keys():
    noisy = '{"count": 3, "mean": 1.0, "median": 1.0, "std": 0.0, "min": 1.0, "max": 1.0, "commentary": "nice"}'
    assert "commentary" not in final.extract_statistics(noisy)
