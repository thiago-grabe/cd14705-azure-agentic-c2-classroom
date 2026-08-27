"""PythonExecutor tests (rubric S2 'executes Python code safely with run()')."""

from pathlib import Path

import matplotlib
import pytest

import final

PLOT_SCRIPT = """
import os
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 9, 3], color="blue", label="Original Data")
plt.plot([1, 3], [1, 3], color="green", label="Clean Data")
plt.legend()
os.makedirs("artifacts", exist_ok=True)
plt.savefig("artifacts/data_visualization.png", dpi=100, bbox_inches="tight")
plt.close()
"""


@pytest.fixture
def executor(tmp_path):
    return final.PythonExecutor(
        max_attempts=3, workdir=tmp_path, expected_output=tmp_path / "artifacts" / "data_visualization.png"
    )


def test_matplotlib_uses_a_headless_backend():
    """A GUI backend would let a generated plt.show() block the run forever."""
    assert matplotlib.get_backend().lower() == "agg"


def test_run_executes_valid_code_and_produces_the_plot(executor):
    ok, error = executor.run(PLOT_SCRIPT)
    assert (ok, error) == (True, None)
    assert executor.expected_output.exists()
    assert executor.expected_output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_run_strips_markdown_fences_before_compiling(executor):
    ok, error = executor.run(f"```python\n{PLOT_SCRIPT}```")
    assert ok, error


def test_run_reports_a_runtime_error_with_a_traceback(executor):
    ok, error = executor.run("raise ValueError('boom')")
    assert ok is False
    assert "ValueError: boom" in error
    assert "Traceback" in error


def test_run_reports_a_syntax_error_without_executing(executor):
    ok, error = executor.run("def broken(:\n    pass")
    assert ok is False
    assert error.startswith("SyntaxError while compiling")


def test_run_rejects_an_empty_script(executor):
    ok, error = executor.run("")
    assert ok is False
    assert "empty script" in error


def test_run_fails_when_the_script_produces_no_plot(executor):
    """'exec did not raise' is not success: the deliverable has to appear."""
    ok, error = executor.run("print('nothing saved here')")
    assert ok is False
    assert "was not created" in error
    assert "nothing saved here" in error, "captured stdout should reach the agent"


def test_run_does_not_inherit_a_previous_attempts_plot(executor):
    assert executor.run(PLOT_SCRIPT)[0] is True
    ok, error = executor.run("x = 1")
    assert ok is False and "was not created" in error


def test_run_restores_the_working_directory_after_a_failure(executor):
    before = Path.cwd()
    executor.run("raise RuntimeError('x')")
    assert Path.cwd() == before
    executor.run(PLOT_SCRIPT)
    assert Path.cwd() == before


def test_run_pins_relative_paths_to_the_configured_workdir(executor, tmp_path):
    """The agent writes 'artifacts/...' relatively; it must land next to the
    project, not wherever the operator happened to be standing."""
    ok, _ = executor.run(PLOT_SCRIPT)
    assert ok
    assert (tmp_path / "artifacts" / "data_visualization.png").exists()


def test_run_isolates_the_namespace_between_attempts(executor):
    executor.run("LEAKED = 'first attempt'")
    ok, error = executor.run("print(LEAKED)")
    assert ok is False
    assert "NameError" in error


def test_run_treats_a_clean_sys_exit_as_success_and_a_failure_code_as_failure(tmp_path):
    executor = final.PythonExecutor(workdir=tmp_path, check_output=False)
    assert executor.run("import sys\nsys.exit(0)") == (True, None)
    ok, error = executor.run("import sys\nsys.exit(3)")
    assert ok is False and "sys.exit(3)" in error


def test_default_executor_targets_the_project_artifacts_directory():
    executor = final.PythonExecutor()
    assert executor.workdir == final.BASE_DIR
    assert executor.expected_output == final.VIZ_IMAGE_PATH
