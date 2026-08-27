"""Rubric conformance suite.

One test (or small group) per line of the project rubric, so "this meets the
rubric" is a command you run rather than a claim you make. Everything here is
static introspection of the imported module -- no Azure calls.
"""

import ast
import inspect
import logging
from pathlib import Path

import pytest

import final


# ---------------------------------------------------------------------------
# Section 1 - Imports, Environment, and Logging
# ---------------------------------------------------------------------------
def test_s1_imports_required_semantic_kernel_symbols():
    """Rubric S1/Imports: the seven required SK components are imported."""
    expected = {
        "Kernel": "semantic_kernel",
        "ChatCompletionAgent": "semantic_kernel.agents",
        "AgentGroupChat": "semantic_kernel.agents",
        "TerminationStrategy": "semantic_kernel.agents",
        "AzureChatCompletion": "semantic_kernel.connectors.ai.open_ai",
        "OpenAIChatPromptExecutionSettings": "semantic_kernel.connectors.ai.open_ai",
        "KernelArguments": "semantic_kernel.functions",
    }
    for name, module_prefix in expected.items():
        obj = getattr(final, name, None)
        assert obj is not None, f"{name} is not imported in final.py"
        assert inspect.isclass(obj), f"{name} should be a class"
        assert obj.__module__.startswith(module_prefix), (
            f"{name} resolved to {obj.__module__}, expected something under {module_prefix}"
        )


def test_s1_no_unimplemented_todo_markers_remain():
    """Rubric S1: every starter <TODO> block has been replaced."""
    source = Path(final.__file__).read_text(encoding="utf-8")
    assert "<TODO" not in source, "an unimplemented <TODO> placeholder is still present"


def test_s5_starter_step_banners_are_preserved_in_order():
    """Rubric S5/Readability: the six starter step markers survive as banners,
    so a reviewer can map rubric -> code in one pass."""
    source = Path(final.__file__).read_text(encoding="utf-8")
    banners = [
        "TODO: Step 3 - Imports",
        "TODO: Step 2 - Environment Setup",
        "TODO: Step 3 - Kernel",
        "TODO: Step 4 - Supporting Logic",
        "TODO: Step 5 - Build the Agents",
        "TODO: Step 6 - Orchestrate the Main",
    ]
    positions = []
    for banner in banners:
        assert banner in source, f"missing section banner for '{banner}'"
        positions.append(source.index(banner))
    assert positions == sorted(positions), "step banners are out of order"


def test_s1_environment_variables_come_from_dotenv():
    """Rubric S1/Environment: API_KEY and BASE_URL bind from the documented vars."""
    source = Path(final.__file__).read_text(encoding="utf-8")
    assert "load_dotenv(" in source
    assert 'os.getenv("AZURE_OPENAI_KEY")' in source
    assert 'os.getenv("URL")' in source
    assert isinstance(final.API_KEY, str) and final.API_KEY
    assert isinstance(final.BASE_URL, str) and final.BASE_URL


def test_s1_api_version_is_the_specified_value():
    """Rubric S1/Environment: API_VERSION set correctly."""
    assert final.DEFAULT_CONFIG["api_version"] == "2024-05-01-preview"
    assert final.API_VERSION == "2024-05-01-preview"


def test_s1_agent_logger_writes_to_agent_chat_log():
    """Rubric S1/Logging: dedicated logger + FileHandler on logs/agent_chat.log."""
    logger = final.agent_logger
    assert logger.name == "semantic_kernel.agents"
    assert logger.level == logging.DEBUG
    assert logger.propagate is False
    handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(handlers) == 1, "expected exactly one FileHandler on the agent logger"
    # In the test session the handler is re-targeted via AGENT_CHAT_LOG; the
    # production default is asserted separately below.
    assert Path(handlers[0].baseFilename).name.endswith(".log")
    assert final.AGENT_LOG_NAME == "agent_chat.log"
    assert (final.LOGS_DIR / final.AGENT_LOG_NAME).parent == final.LOGS_DIR


def test_s1_formatter_applied_to_handler():
    """Rubric S1/Logging: formatter applied and handler added."""
    handler = next(h for h in final.agent_logger.handlers if isinstance(h, logging.FileHandler))
    assert handler.formatter is not None
    assert handler.formatter._fmt == "%(asctime)s - %(name)s:%(message)s"
    assert handler.level == logging.DEBUG


def test_s1_log_agent_message_records_role_name_and_content(tmp_path):
    """Rubric S1/Logging: log_agent_message() logs role, name and content."""
    from semantic_kernel.contents.chat_message_content import ChatMessageContent
    from semantic_kernel.contents.utils.author_role import AuthorRole

    log_file = tmp_path / "agent_chat.log"
    final._configure_agent_logger(log_file)
    try:
        final.log_agent_message(
            ChatMessageContent(role=AuthorRole.ASSISTANT, name="DataCleaning", content="cleaned 16 rows")
        )
        final.log_agent_message(
            ChatMessageContent(role=AuthorRole.USER, name=None, content="anonymous turn")
        )
        text = log_file.read_text(encoding="utf-8")
    finally:
        final._configure_agent_logger(final.AGENT_LOG_PATH)

    assert "assistant" in text.lower()
    assert "DataCleaning" in text
    assert "cleaned 16 rows" in text
    assert " - *: anonymous turn" in text, "the name=None fallback should render as '*'"


def test_s1_log_agent_message_never_propagates_exceptions(tmp_path):
    """Rubric S5/Error Handling: a bad message must not break the pipeline."""
    log_file = tmp_path / "agent_chat.log"
    final._configure_agent_logger(log_file)
    try:

        class Exploding:
            @property
            def role(self):
                raise RuntimeError("no role for you")

        final.log_agent_message(Exploding())  # must not raise
        assert "Failed to write agent message to log" in log_file.read_text(encoding="utf-8")
    finally:
        final._configure_agent_logger(final.AGENT_LOG_PATH)


# ---------------------------------------------------------------------------
# Section 2 - Helper Functions
# ---------------------------------------------------------------------------
def test_s2_helper_functions_exist_with_expected_signatures():
    """Rubric S2: every named helper is present and callable."""
    for name in (
        "get_csv_name",
        "load_csv_file",
        "load_quality_instructions",
        "load_reports_instructions",
        "load_logs",
        "save_final_report",
    ):
        assert callable(getattr(final, name, None)), f"{name} is missing"
    assert inspect.isclass(final.PythonExecutor)
    assert callable(final.PythonExecutor.run)
    # save_final_report keeps the rubric's documented default destination.
    default = inspect.signature(final.save_final_report).parameters["path"].default
    assert str(default) == "artifacts/final_report.md"


# ---------------------------------------------------------------------------
# Section 3 - Agent Configuration & Factory
# ---------------------------------------------------------------------------
def test_s3_agent_config_has_all_six_documented_prompts():
    """Rubric S3/AGENT_CONFIG: prompts for cleaning, stats, analysis, code and report."""
    expected = {
        "PythonExecutorAgent",
        "DataCleaning",
        "DataStatistics",
        "AnalysisChecker",
        "ReportGenerator",
        "ReportChecker",
    }
    assert set(final.AGENT_CONFIG) == expected
    for name, prompt in final.AGENT_CONFIG.items():
        assert isinstance(prompt, str) and len(prompt) > 200, f"{name}: prompt is too thin"
        assert "{{" not in prompt, f"{name}: '{{{{' is SK template syntax and raises at invoke time"


def test_s3_spec_files_are_embedded_in_the_auditor_prompts():
    """Rubric S3: the checker/report prompts are driven by the spec files."""
    quality_first = final.load_quality_instructions("Data_Quality_Instructions.txt")[0]
    report_heading = "# Data Analysis Report"
    assert quality_first in final.AGENT_CONFIG["AnalysisChecker"]
    assert report_heading in final.AGENT_CONFIG["ReportGenerator"]
    assert report_heading in final.AGENT_CONFIG["ReportChecker"]


def test_s3_create_agent_wraps_settings_in_kernel_arguments():
    """Rubric S3/Agent Factory: ChatCompletionAgent + KernelArguments when settings given."""
    from semantic_kernel.agents import ChatCompletionAgent
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
    from semantic_kernel.functions import KernelArguments

    settings = OpenAIChatPromptExecutionSettings(temperature=0.42)
    agent = final.create_agent("Probe", "probe instructions", final.chat_service, settings)
    assert isinstance(agent, ChatCompletionAgent)
    assert agent.name == "Probe"
    assert agent.instructions == "probe instructions"
    assert isinstance(agent.arguments, KernelArguments)
    assert agent.arguments.execution_settings["default"].temperature == 0.42


def test_s3_create_agent_without_settings_omits_arguments():
    """Rubric S3/Agent Factory: KernelArguments only when settings are provided."""
    agent = final.create_agent("Bare", "bare instructions", final.chat_service)
    assert agent.name == "Bare"
    assert agent.arguments is None or not getattr(agent.arguments, "execution_settings", None)


@pytest.mark.parametrize(
    "attr, name, temperature",
    [
        ("python_agent", "PythonExecutorAgent", 0.1),
        ("cleaning_agent", "DataCleaning", 0.7),
        ("stats_agent", "DataStatistics", 0.5),
        ("checker_agent", "AnalysisChecker", 0.2),
        ("report_agent", "ReportGenerator", 1.0),
        ("report_checker_agent", "ReportChecker", 0.2),
    ],
)
def test_s3_live_agents_carry_the_required_temperature(attr, name, temperature):
    """Rubric S3/Temperature & Behavior, read off the live agent objects so a
    mis-wired factory is caught rather than a matching config dict."""
    agent = getattr(final, attr)
    assert agent.name == name
    assert agent.arguments.execution_settings["default"].temperature == pytest.approx(temperature)


def test_s3_temperature_ordering_matches_the_role_rationale():
    """Rubric S3: low for code generation and auditors, higher for creative roles."""
    temps = final.CONFIG["temperatures"]
    auditors = (temps["AnalysisChecker"], temps["ReportChecker"], temps["PythonExecutorAgent"])
    creative = (temps["DataCleaning"], temps["ReportGenerator"])
    assert max(auditors) < min(creative)


# ---------------------------------------------------------------------------
# Section 4 - Chat Groups
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "chat_attr, agent_names",
    [
        ("analysis_chat", ["DataCleaning", "DataStatistics", "AnalysisChecker"]),
        ("code_chat", ["PythonExecutorAgent"]),
        ("report_chat", ["ReportGenerator", "ReportChecker"]),
    ],
)
def test_s4_group_chats_hold_the_right_agents_in_order(chat_attr, agent_names):
    """Rubric S4/AgentGroupChat Instances."""
    from semantic_kernel.agents import AgentGroupChat

    chat = getattr(final, chat_attr)
    assert isinstance(chat, AgentGroupChat)
    assert [a.name for a in chat.agents] == agent_names


def test_s4_approval_strategies_are_scoped_to_the_auditors():
    """Rubric S4: termination halts on approval, and only the auditor can approve."""
    analysis = final.analysis_chat.termination_strategy
    report = final.report_chat.termination_strategy
    assert isinstance(analysis, final.ApprovalTerminationStrategy)
    assert isinstance(report, final.ApprovalTerminationStrategy)
    assert [a.name for a in analysis.agents] == ["AnalysisChecker"]
    assert [a.name for a in report.agents] == ["ReportChecker"]
    assert analysis.maximum_iterations == final.CONFIG["analysis_max_turns"]
    assert report.maximum_iterations == final.CONFIG["report_max_turns"]


def test_s4_code_chat_spends_exactly_one_llm_call_per_invoke():
    """The single-agent chat must not inherit DefaultTerminationStrategy's five
    iterations, and must be re-enterable for the retry loop."""
    strategy = final.code_chat.termination_strategy
    assert isinstance(strategy, final.SingleTurnTerminationStrategy)
    assert strategy.maximum_iterations == 1
    assert strategy.automatic_reset is True


def test_s4_main_workflow_entry_point_is_async():
    """Rubric S4/Main Workflow."""
    assert inspect.iscoroutinefunction(final.main)


# ---------------------------------------------------------------------------
# Section 5 - Error handling, directories, code quality
# ---------------------------------------------------------------------------
def test_s5_required_directories_exist():
    """Rubric S5/Directory & File Management."""
    for directory in (final.DATA_DIR, final.SPECS_DIR, final.LOGS_DIR, final.ARTIFACTS_DIR):
        assert directory.is_dir(), f"{directory} was not created at import time"


def test_s5_directories_are_created_before_the_log_handler():
    """The starter opens FileHandler('logs/agent_chat.log') before creating logs/,
    which crashes from any other working directory. Assert the fixed ordering."""
    source = Path(final.__file__).read_text(encoding="utf-8")
    assert source.index("_directory.mkdir(") < source.index("logging.FileHandler(")


def test_s5_deliverable_paths_match_the_project_specification():
    """Rubric S5/Files saved to correct paths + Part 3 deliverable table."""
    assert final.CLEANED_JSON_PATH.name == "data-cleaned.json"
    assert final.CLEANED_JSON_PATH.parent == final.BASE_DIR
    assert final.VIZ_CODE_PATH == final.ARTIFACTS_DIR / "data_visualization_code.py"
    assert final.VIZ_IMAGE_PATH == final.ARTIFACTS_DIR / "data_visualization.png"
    assert final.FINAL_REPORT_PATH == final.ARTIFACTS_DIR / "final_report.md"


def test_s5_no_bare_or_silently_swallowed_exception_handlers():
    """Rubric S5/Error Handling: every except clause is typed and does something."""
    tree = ast.parse(Path(final.__file__).read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            offenders.append(f"bare except at line {node.lineno}")
        body = [n for n in node.body if not isinstance(n, ast.Pass)]
        if not body:
            # A bodyless handler is only acceptable where the failure is truly
            # irrelevant; those sites use contextlib.suppress instead.
            offenders.append(f"silent 'except: pass' at line {node.lineno}")
    assert not offenders, offenders


def test_s5_every_public_helper_has_a_docstring():
    """Rubric S5/Readability & Documentation."""
    undocumented = [
        name
        for name, obj in vars(final).items()
        if not name.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and getattr(obj, "__module__", None) == "final"
        and not (obj.__doc__ or "").strip()
    ]
    assert not undocumented, f"missing docstrings: {undocumented}"


def test_s5_chat_service_retries_on_transient_failures():
    """Rubric S5/Error Handling: a low-TPM deployment makes 429s routine, and a
    group chat resends its whole history each turn. The openai client's default
    of 2 retries is raised so Retry-After-aware backoff can absorb them."""
    client = final.chat_service.client
    assert client.max_retries == final.CONFIG["max_request_retries"] >= 4
    assert float(client.timeout) == float(final.CONFIG["request_timeout_seconds"])
    assert str(client.base_url).endswith(f"/openai/deployments/{final.AZURE_TARGET['deployment_name']}/")


def test_s3_datastatistics_prompt_enforces_its_role_boundary():
    """A live run caught DataStatistics answering an instruction addressed to
    the auditor. The prompt now forbids that explicitly."""
    prompt = final.AGENT_CONFIG["DataStatistics"]
    assert "Stay in role" in prompt
    assert "ddof = 1" in prompt
    assert "another agent" in prompt
