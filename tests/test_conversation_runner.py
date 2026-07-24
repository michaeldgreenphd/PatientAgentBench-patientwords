# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for conversation runner.

Tests ConversationRunner orchestration with mocked agents.
"""

import asyncio
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from patient_agent_bench.config import AgentSpec, ModelConfig
from patient_agent_bench.runner.experiment_config import ExperimentConfig
from patient_agent_bench.runner.conversation_runner import (
    ConversationRunner,
    ConversationResult,
)
from patient_agent_bench.runner.conversation import Conversation


@pytest.fixture
def mock_bedrock_llm():
    """Mock create_chat_model and create_bedrock_client_with_role to avoid AWS calls."""
    with patch("patient_agent_bench.eval.base_rubric.create_bedrock_client_with_role") as mock_client:
        mock_client.return_value = MagicMock()
        with patch("patient_agent_bench.eval.base_rubric.create_chat_model") as mock_factory:
            mock_llm = MagicMock()
            mock_factory.return_value = mock_llm
            yield mock_llm


@pytest.fixture
def mock_assistant_llm():
    """Mock create_chat_model for assistant agent."""
    with patch("patient_agent_bench.assistant_agent.default_agent.create_chat_model") as mock_factory:
        mock_llm = MagicMock()
        mock_factory.return_value = mock_llm
        yield mock_llm


@pytest.fixture
def mock_user_llm():
    """Mock create_chat_model for user agent."""
    with patch("patient_agent_bench.user_agent.default_agent.create_chat_model") as mock_factory:
        mock_llm = MagicMock()
        mock_factory.return_value = mock_llm
        yield mock_llm


@pytest.fixture
def mock_sandbox_llm_response():
    """Mock LLM response for sandbox generation."""
    return {
        "offices": [
            {
                "id": "office_001",
                "name": "Downtown Clinic",
                "address": "123 Main St",
                "city": "Boston",
                "state": "MA",
                "zip_code": "02101",
                "phone": "555-1234",
                "hours": "Mon-Fri 8AM-6PM",
            }
        ],
        "doctors": [
            {
                "id": "doc_001",
                "name": "Dr. Sarah Johnson",
                "specialty": "Primary Care",
                "credentials": "MD",
                "office_id": "office_001",
            }
        ],
        "medications": [
            {
                "id": "med_001",
                "name": "Lisinopril",
                "dosage": "10mg",
                "frequency": "once daily",
                "status": "active",
                "prescribed_date": "2025-01-01",
                "pharmacy": "CVS",
                "refills_remaining": 3,
            }
        ],
    }


@pytest.fixture
def sandbox_model_config():
    """Sandbox model configuration for tests."""
    return ModelConfig(
        model_id="test-sandbox-model",
        temperature=0.5,
        max_tokens=8192,
    )


@pytest.fixture
def bench_config_with_sandbox(model_config, sandbox_model_config):
    """Experiment configuration with sandbox_model for tests."""
    return ExperimentConfig(
        experiment_id="test_0_0",
        assistant_agent=AgentSpec(model=model_config),
        user_agent=AgentSpec(model=model_config),
        evaluator_models=[ModelConfig(
            model_id="test-model-id",
            temperature=0.0,
            max_tokens=4096,
        )],
        sandbox_model=sandbox_model_config,
        analyzer_model=ModelConfig(
            model_id="test-model-id",
            temperature=0.0,
            max_tokens=4096,
        ),
        assistant_idx=0,
        user_idx=0,
        max_turns=3,
        strip_thinking_content=True,
    )


@pytest.fixture
def mock_sandbox_initialization(mock_sandbox_llm_response):
    """Mock sandbox initialization to avoid LLM calls."""
    with patch(
        "patient_agent_bench.runner.conversation_runner.create_sandbox_llm"
    ) as mock_create_llm:
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_sandbox_llm_response)
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_create_llm.return_value = mock_llm
        yield mock_llm


class TestConversationResult:
    """Tests for ConversationResult dataclass."""

    def test_conversation_result_creation(self, sample_conversation):
        """Test creating a ConversationResult."""
        result = ConversationResult(
            case_id="test_001",
            conversation=sample_conversation,
            user_profile="<profile>Test</profile>",
            scenario="Test scenario",
            num_turns=3,
        )

        assert result.case_id == "test_001"
        assert result.conversation == sample_conversation
        assert result.num_turns == 3
        assert result.evaluation is None

    def test_conversation_result_with_evaluation(
        self, sample_conversation, sample_evaluation_result
    ):
        """Test ConversationResult with evaluation."""
        result = ConversationResult(
            case_id="test_001",
            conversation=sample_conversation,
            user_profile="<profile>Test</profile>",
            scenario="Test scenario",
            num_turns=3,
            evaluation=sample_evaluation_result,
        )

        assert result.evaluation is not None
        assert result.evaluation["aggregate_score"] == 90.0

    def test_conversation_result_to_dict(self, sample_conversation, sample_evaluation_result):
        """Test ConversationResult.to_dict() serialization."""
        result = ConversationResult(
            case_id="test_001",
            conversation=sample_conversation,
            user_profile="<profile>Test</profile>",
            scenario="Test scenario",
            num_turns=3,
            evaluation=sample_evaluation_result,
        )

        result_dict = result.to_dict()

        assert result_dict["case_id"] == "test_001"
        assert isinstance(result_dict["conversation"], list)
        assert len(result_dict["conversation"]) == 5
        assert result_dict["conversation"][0]["type"] == "human"
        assert result_dict["num_turns"] == 3
        assert result_dict["evaluation"]["aggregate_score"] == 90.0


class TestConversationRunnerInit:
    """Tests for ConversationRunner initialization."""

    def test_init_with_experiment(self, bench_config_with_sandbox):
        """Test runner initializes with experiment config."""
        runner = ConversationRunner(experiment=bench_config_with_sandbox)

        assert runner.experiment == bench_config_with_sandbox


class TestConversationRunnerAsync:
    """Tests for async conversation methods."""

    @pytest.mark.asyncio
    async def test_run_conversation_async_basic(
        self,
        bench_config_with_sandbox,
        sample_benchmark_entry,
        mock_sandbox_initialization,
    ):
        """Test running a basic conversation asynchronously."""
        from langchain_core.messages import AIMessage, HumanMessage

        mock_assistant = MagicMock()
        # Return messages list matching the new invoke() signature
        mock_assistant.invoke.return_value = {
            "messages": [
                HumanMessage(content="I need a refill"),
                AIMessage(content="I can help you with that."),
            ],
        }

        mock_user_agent = MagicMock()
        mock_user_agent.start_conversation.return_value = "I need a refill"
        mock_user_agent.respond.return_value = "Thanks!"
        mock_user_agent.is_conversation_complete.side_effect = [False, True]

        runner = ConversationRunner(experiment=bench_config_with_sandbox)

        with patch(
            "patient_agent_bench.runner.conversation_runner.create_assistant_agent_from_spec"
        ) as mock_create_assistant:
            mock_create_assistant.return_value = mock_assistant
            with patch(
                "patient_agent_bench.runner.conversation_runner.create_user_agent_from_spec"
            ) as mock_create_user:
                mock_create_user.return_value = mock_user_agent
                with patch(
                    "patient_agent_bench.runner.conversation_runner."
                    "create_bedrock_client_with_role"
                ) as mock_client:
                    mock_client.return_value = MagicMock()

                    result = await runner.run_conversation_async(
                        sample_benchmark_entry
                    )

        assert result.case_id == sample_benchmark_entry.id
        assert len(result.conversation) > 0
        assert result.num_turns > 0

    @pytest.mark.asyncio
    async def test_run_conversation_async_respects_max_turns(
        self,
        sample_benchmark_entry,
        mock_sandbox_initialization,
        sandbox_model_config,
        model_config,
    ):
        """Test conversation respects max_turns limit from experiment config."""
        from langchain_core.messages import AIMessage, HumanMessage

        # Create experiment config with max_turns=2
        experiment_with_max_turns_2 = ExperimentConfig(
            experiment_id="test_0_0",
            assistant_agent=AgentSpec(model=model_config),
            user_agent=AgentSpec(model=model_config),
            evaluator_models=[model_config],
            sandbox_model=sandbox_model_config,
            analyzer_model=model_config,
            assistant_idx=0,
            user_idx=0,
            max_turns=2,
            strip_thinking_content=True,
        )

        mock_assistant = MagicMock()
        mock_assistant.invoke.return_value = {
            "messages": [
                HumanMessage(content="Start"),
                AIMessage(content="Response"),
            ],
        }

        mock_user_agent = MagicMock()
        mock_user_agent.start_conversation.return_value = "Start"
        mock_user_agent.respond.return_value = "Continue"
        mock_user_agent.is_conversation_complete.return_value = False  # Never complete

        runner = ConversationRunner(experiment=experiment_with_max_turns_2)

        with patch(
            "patient_agent_bench.runner.conversation_runner.create_assistant_agent_from_spec"
        ) as mock_create_assistant:
            mock_create_assistant.return_value = mock_assistant
            with patch(
                "patient_agent_bench.runner.conversation_runner.create_user_agent_from_spec"
            ) as mock_create_user:
                mock_create_user.return_value = mock_user_agent
                with patch(
                    "patient_agent_bench.runner.conversation_runner."
                    "create_bedrock_client_with_role"
                ) as mock_client:
                    mock_client.return_value = MagicMock()

                    result = await runner.run_conversation_async(
                        sample_benchmark_entry
                    )

        # Should stop at max_turns=2
        assert result.num_turns <= 2

    @pytest.mark.asyncio
    async def test_run_conversation_async_creates_sandbox(
        self,
        bench_config_with_sandbox,
        sample_benchmark_entry,
        mock_sandbox_initialization,
    ):
        """Test that run_conversation_async creates and initializes a sandbox."""
        from langchain_core.messages import AIMessage, HumanMessage

        mock_assistant = MagicMock()
        mock_assistant.invoke.return_value = {
            "messages": [
                HumanMessage(content="Start"),
                AIMessage(content="Response"),
            ],
        }

        mock_user_agent = MagicMock()
        mock_user_agent.start_conversation.return_value = "Start"
        mock_user_agent.is_conversation_complete.return_value = True

        runner = ConversationRunner(experiment=bench_config_with_sandbox)

        with patch(
            "patient_agent_bench.runner.conversation_runner.create_assistant_agent_from_spec"
        ) as mock_create_assistant:
            mock_create_assistant.return_value = mock_assistant
            with patch(
                "patient_agent_bench.runner.conversation_runner.create_user_agent_from_spec"
            ) as mock_create_user:
                mock_create_user.return_value = mock_user_agent
                with patch(
                    "patient_agent_bench.runner.conversation_runner.HealthcareSandbox"
                ) as mock_sandbox_class:
                    mock_sandbox = MagicMock()
                    mock_sandbox._initialized = False
                    mock_sandbox_class.return_value = mock_sandbox
                    with patch(
                        "patient_agent_bench.runner.conversation_runner.initialize_sandbox"
                    ) as mock_init_sandbox:
                        mock_init_sandbox.return_value = None
                        with patch(
                            "patient_agent_bench.runner.conversation_runner."
                            "create_bedrock_client_with_role"
                        ) as mock_client:
                            mock_client.return_value = MagicMock()

                            await runner.run_conversation_async(
                                sample_benchmark_entry
                            )

                            # Verify sandbox was created and initialize_sandbox was called
                            mock_sandbox_class.assert_called_once()
                            mock_init_sandbox.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_conversation_async_passes_sandbox_to_registry(
        self,
        bench_config_with_sandbox,
        sample_benchmark_entry,
        mock_sandbox_initialization,
    ):
        """Test that sandbox is passed to create_tool_registry."""
        from langchain_core.messages import AIMessage, HumanMessage

        mock_assistant = MagicMock()
        mock_assistant.invoke.return_value = {
            "messages": [
                HumanMessage(content="Start"),
                AIMessage(content="Response"),
            ],
        }

        mock_user_agent = MagicMock()
        mock_user_agent.start_conversation.return_value = "Start"
        mock_user_agent.is_conversation_complete.return_value = True

        runner = ConversationRunner(experiment=bench_config_with_sandbox)

        with patch(
            "patient_agent_bench.runner.conversation_runner.create_assistant_agent_from_spec"
        ) as mock_create_assistant:
            mock_create_assistant.return_value = mock_assistant
            with patch(
                "patient_agent_bench.runner.conversation_runner.create_user_agent_from_spec"
            ) as mock_create_user:
                mock_create_user.return_value = mock_user_agent
                with patch(
                    "patient_agent_bench.runner.conversation_runner.create_tool_registry"
                ) as mock_create_registry:
                    mock_registry = MagicMock()
                    mock_registry.get_tools.return_value = []
                    mock_create_registry.return_value = mock_registry
                    with patch(
                        "patient_agent_bench.runner.conversation_runner."
                        "create_bedrock_client_with_role"
                    ) as mock_client:
                        mock_client.return_value = MagicMock()

                        await runner.run_conversation_async(
                            sample_benchmark_entry
                        )

                        # Verify create_tool_registry was called with a sandbox
                        mock_create_registry.assert_called_once()
                        call_args = mock_create_registry.call_args
                        assert call_args[0][0] is not None

    @pytest.mark.asyncio
    async def test_run_conversation_async_with_role(
        self,
        bench_config_with_sandbox,
        sample_benchmark_entry,
        mock_sandbox_initialization,
    ):
        """Test that assigned_role is passed to create_bedrock_client_with_role."""
        from langchain_core.messages import AIMessage, HumanMessage

        mock_assistant = MagicMock()
        mock_assistant.invoke.return_value = {
            "messages": [
                HumanMessage(content="Start"),
                AIMessage(content="Response"),
            ],
        }

        mock_user_agent = MagicMock()
        mock_user_agent.start_conversation.return_value = "Start"
        mock_user_agent.is_conversation_complete.return_value = True

        runner = ConversationRunner(experiment=bench_config_with_sandbox)

        test_role = "arn:aws:iam::123456789:role/test-role"

        with patch(
            "patient_agent_bench.runner.conversation_runner.create_assistant_agent_from_spec"
        ) as mock_create_assistant:
            mock_create_assistant.return_value = mock_assistant
            with patch(
                "patient_agent_bench.runner.conversation_runner.create_user_agent_from_spec"
            ) as mock_create_user:
                mock_create_user.return_value = mock_user_agent
                with patch(
                    "patient_agent_bench.runner.conversation_runner."
                    "create_bedrock_client_with_role"
                ) as mock_client:
                    mock_client.return_value = MagicMock()

                    await runner.run_conversation_async(
                        sample_benchmark_entry, assigned_role=test_role
                    )

                    # Verify role was passed to client creation
                    mock_client.assert_called_once_with(test_role)


# =============================================================================
# Property-Based Tests for Configurable Prompts
# =============================================================================

from hypothesis import given, strategies as st, settings, HealthCheck


# Strategy for generating valid prompt names
@st.composite
def valid_prompt_name(draw):
    """Generate a valid prompt file name (without .py extension)."""
    return draw(st.from_regex(r"[a-z][a-z0-9_]{0,29}", fullmatch=True))


class TestConversationRunnerPromptConfigPropertyBased:
    """Property-based tests for prompt configuration in ConversationRunner."""

    @given(
        user_prompt=valid_prompt_name(),
        assistant_prompt=valid_prompt_name(),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    def test_property_prompt_passed_from_config(
        self,
        user_prompt,
        assistant_prompt,
        sample_benchmark_entry,
        mock_sandbox_initialization,
        sandbox_model_config,
        model_config,
    ):
        """
        Property: Verify prompt names are passed from config to agents.

        *For any* experiment configuration with user_prompt and assistant_prompt fields,
        the ConversationRunner SHALL pass these prompt names when creating agents
        via the AgentSpec objects.
        """
        # Create experiment config with custom prompts inside AgentSpec
        experiment = ExperimentConfig(
            experiment_id="test_0_0",
            assistant_agent=AgentSpec(model=model_config, prompt=assistant_prompt),
            user_agent=AgentSpec(model=model_config, prompt=user_prompt),
            evaluator_models=[model_config],
            sandbox_model=sandbox_model_config,
            analyzer_model=model_config,
            assistant_idx=0,
            user_idx=0,
            max_turns=3,
            strip_thinking_content=True,
        )

        # Track the spec passed to create_user_agent_from_spec
        user_agent_calls = []

        def mock_create_user_agent(*args, **kwargs):
            """Mock factory that captures kwargs."""
            user_agent_calls.append(kwargs)
            mock_agent = MagicMock()
            mock_agent.start_conversation.return_value = "Start"
            mock_agent.is_conversation_complete.return_value = True
            return mock_agent

        # Track the spec passed to create_assistant_agent_from_spec
        assistant_agent_calls = []

        def mock_create_assistant_agent(*args, **kwargs):
            """Mock factory that captures kwargs."""
            from langchain_core.messages import AIMessage, HumanMessage
            assistant_agent_calls.append(kwargs)
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {
                "messages": [
                    HumanMessage(content="Start"),
                    AIMessage(content="Response"),
                ],
            }
            return mock_agent

        with patch(
            "patient_agent_bench.runner.conversation_runner.create_assistant_agent_from_spec",
            side_effect=mock_create_assistant_agent,
        ):
            with patch(
                "patient_agent_bench.runner.conversation_runner.create_user_agent_from_spec",
                side_effect=mock_create_user_agent,
            ):
                with patch(
                    "patient_agent_bench.runner.conversation_runner."
                    "create_bedrock_client_with_role"
                ) as mock_client:
                    mock_client.return_value = MagicMock()

                    # Create runner and run conversation
                    runner = ConversationRunner(experiment=experiment)
                    asyncio.run(
                        runner.run_conversation_async(sample_benchmark_entry)
                    )

        # Verify assistant agent was created with spec containing correct prompt
        assert len(assistant_agent_calls) > 0
        assert assistant_agent_calls[0].get("spec").prompt == assistant_prompt

        # Verify user agent was created with spec containing correct prompt
        assert len(user_agent_calls) > 0
        assert user_agent_calls[0].get("spec").prompt == user_prompt


class TestStripThinkingContent:
    """Tests for the _strip_thinking_content method."""

    def test_strip_single_thinking_block(self, bench_config_with_sandbox):
        """Test stripping a single thinking block."""
        runner = ConversationRunner(experiment=bench_config_with_sandbox)
        text = "<thinking>This is my reasoning.</thinking>Here is my response."
        result = runner._strip_thinking_content(text)
        assert result == "Here is my response."

    def test_strip_multiline_thinking_block(self, bench_config_with_sandbox):
        """Test stripping a multiline thinking block."""
        runner = ConversationRunner(experiment=bench_config_with_sandbox)
        text = """<thinking>
Let me think about this.
I need to consider multiple factors.
</thinking>
Here is my response."""
        result = runner._strip_thinking_content(text)
        assert result.strip() == "Here is my response."

    def test_strip_multiple_thinking_blocks(self, bench_config_with_sandbox):
        """Test stripping multiple thinking blocks."""
        runner = ConversationRunner(experiment=bench_config_with_sandbox)
        text = "<thinking>First thought.</thinking>Response 1. <thinking>Second thought.</thinking>Response 2."
        result = runner._strip_thinking_content(text)
        assert result == "Response 1. Response 2."

    def test_no_thinking_block(self, bench_config_with_sandbox):
        """Test text without thinking blocks is unchanged."""
        runner = ConversationRunner(experiment=bench_config_with_sandbox)
        text = "This is a normal response without thinking."
        result = runner._strip_thinking_content(text)
        assert result == text

    def test_empty_thinking_block(self, bench_config_with_sandbox):
        """Test stripping an empty thinking block."""
        runner = ConversationRunner(experiment=bench_config_with_sandbox)
        text = "<thinking></thinking>Response."
        result = runner._strip_thinking_content(text)
        assert result == "Response."

    def test_thinking_at_end(self, bench_config_with_sandbox):
        """Test stripping thinking block at the end."""
        runner = ConversationRunner(experiment=bench_config_with_sandbox)
        text = "Response here.<thinking>Final thoughts.</thinking>"
        result = runner._strip_thinking_content(text)
        assert result == "Response here."
