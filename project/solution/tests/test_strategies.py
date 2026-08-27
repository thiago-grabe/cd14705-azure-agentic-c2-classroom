"""Termination-strategy tests (rubric S4 'termination strategy halts on approval')."""

import pytest
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

import final


def message(content):
    return ChatMessageContent(role=AuthorRole.ASSISTANT, name="AnalysisChecker", content=content)


@pytest.fixture
def strategy():
    return final.ApprovalTerminationStrategy(agents=[final.checker_agent])


async def test_empty_history_returns_false_instead_of_raising(strategy):
    """Regression guard for the starter's bug: it delegates to
    TerminationStrategy.should_agent_terminate, which is abstract and raises
    NotImplementedError on the first non-approving turn."""
    assert await strategy.should_agent_terminate(final.checker_agent, []) is False


async def test_approval_terminates_the_chat(strategy):
    history = [message('{"title": "Approved", "reason": ""}')]
    assert await strategy.should_agent_terminate(final.checker_agent, history) is True


async def test_a_failing_verdict_containing_the_word_approved_does_not_terminate(strategy):
    history = [message('{"title": "Failed", "reason": "statistics were not approved"}')]
    assert await strategy.should_agent_terminate(final.checker_agent, history) is False


async def test_a_revise_response_does_not_terminate(strategy):
    history = [message("REVISE\n1. Section 2 is missing the median.")]
    assert await strategy.should_agent_terminate(final.checker_agent, history) is False


async def test_none_content_is_handled(strategy):
    history = [ChatMessageContent(role=AuthorRole.ASSISTANT, name="AnalysisChecker", content=None)]
    assert await strategy.should_agent_terminate(final.checker_agent, history) is False


async def test_only_the_scoped_auditor_can_end_the_chat(strategy):
    """should_terminate() short-circuits for out-of-scope agents, which is what
    stops a report body containing the word 'Approved' from self-approving."""
    history = [message('{"title": "Approved"}')]
    assert await strategy.should_terminate(final.checker_agent, history) is True
    assert await strategy.should_terminate(final.cleaning_agent, history) is False


async def test_single_turn_strategy_always_terminates_and_resets():
    strategy = final.SingleTurnTerminationStrategy(agents=[final.python_agent])
    assert strategy.maximum_iterations == 1
    assert strategy.automatic_reset is True
    assert await strategy.should_agent_terminate(final.python_agent, []) is True


def test_analysis_and_report_chats_can_be_re_entered():
    """automatic_reset=True is what keeps a second invoke() from raising
    AgentChatException('Chat is already complete')."""
    assert final.analysis_chat.termination_strategy.automatic_reset is True
    assert final.report_chat.termination_strategy.automatic_reset is True
    assert final.code_chat.termination_strategy.automatic_reset is True


def test_agent_prompts_render_as_semantic_kernel_templates():
    """`instructions` go through KernelPromptTemplate.render(); a literal '{{'
    raises TemplateSyntaxError at the first invoke, long after import."""
    import asyncio

    from semantic_kernel import Kernel
    from semantic_kernel.functions import KernelArguments
    from semantic_kernel.prompt_template import KernelPromptTemplate, PromptTemplateConfig

    async def render(template):
        return await KernelPromptTemplate(
            prompt_template_config=PromptTemplateConfig(template=template)
        ).render(Kernel(), KernelArguments())

    for name, prompt in final.AGENT_CONFIG.items():
        rendered = asyncio.run(render(prompt))
        assert rendered.strip(), f"{name}: prompt rendered empty"
