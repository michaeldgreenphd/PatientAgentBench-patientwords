# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for agent implementations.

Tests AssistantAgent and UserAgent with mocked LLM calls.
"""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from patient_agent_bench.config import AgentSpec, ModelConfig
from patient_agent_bench.assistant_agent.default_agent import (
    DefaultAssistantAgent as AssistantAgent,
    AssistantAgentError,
)
from patient_agent_bench.assistant_agent.default_prompt import SYSTEM_PROMPT as ASSISTANT_SYSTEM_PROMPT
from patient_agent_bench.assistant_agent.registry import create_assistant_agent_from_spec as create_assistant_agent
from patient_agent_bench.user_agent.default_agent import DefaultUserAgent as UserAgent
from patient_agent_bench.user_agent.default_prompt import SYSTEM_PROMPT as USER_SYSTEM_PROMPT
from patient_agent_bench.user_agent.registry import create_user_agent_from_spec as create_user_agent
from tests.conftest import create_mock_tool


# =============================================================================
# AssistantAgent Tests
# =============================================================================

class TestAssistantAgentInit:
    """Tests for AssistantAgent initialization."""

    def test_init_with_model_config(self, model_config, mock_boto3_client, current_datetime):
        """Test agent initializes with model config."""
        # Provide custom tools since default registry requires sandbox
        custom_tools = [create_mock_tool("test_tool")]
        agent = AssistantAgent(
            model_config=model_config,
            current_datetime=current_datetime,
            tools=custom_tools,
        )

        assert agent.model_config == model_config
        assert agent.system_prompt_template == ASSISTANT_SYSTEM_PROMPT
        assert len(agent.tools) > 0

    def test_init_with_custom_tools(self, model_config, mock_boto3_client, current_datetime):
        """Test agent with custom tools."""
        custom_tools = [
            create_mock_tool("custom_tool_1"),
            create_mock_tool("custom_tool_2"),
        ]

        agent = AssistantAgent(
            model_config=model_config,
            current_datetime=current_datetime,
            tools=custom_tools,
        )

        assert len(agent.tools) == 2
        assert agent.tools[0].name == "custom_tool_1"

    def test_init_with_custom_prompt(self, model_config, mock_boto3_client, current_datetime):
        """Test agent with custom system prompt."""
        custom_prompt = "You are a custom assistant. {user_profile}"

        agent = AssistantAgent(
            model_config=model_config,
            current_datetime=current_datetime,
            system_prompt=custom_prompt,
        )

        assert agent.system_prompt_template == custom_prompt

    def test_init_requires_model_id(self, mock_boto3_client, current_datetime):
        """Test that model_id is required."""
        config_without_model_id = ModelConfig(
            model_id="",
            temperature=0.7,
            max_tokens=1024,
        )

        with pytest.raises(ValueError, match="model_config.model_id is required"):
            AssistantAgent(
                model_config=config_without_model_id,
                current_datetime=current_datetime,
            )

    def test_init_requires_current_datetime(self, model_config, mock_boto3_client):
        """Test that current_datetime is required."""
        with pytest.raises(ValueError, match="current_datetime is required"):
            AssistantAgent(
                model_config=model_config,
                current_datetime="",
            )

    def test_get_tools(self, model_config, mock_boto3_client, current_datetime):
        """Test get_tools returns bound tools."""
        # Provide custom tools since default registry requires sandbox
        custom_tools = [create_mock_tool("test_tool")]
        agent = AssistantAgent(
            model_config=model_config,
            current_datetime=current_datetime,
            tools=custom_tools,
        )
        tools = agent.get_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0


class TestAssistantAgentInvoke:
    """Tests for AssistantAgent invoke method."""

    def test_invoke_returns_messages(self, model_config, mock_boto3_client, current_datetime):
        """Test invoke returns result with messages key."""
        agent = AssistantAgent(
            model_config=model_config,
            current_datetime=current_datetime,
            tools=[create_mock_tool("test")],
        )

        # Mock the create_agent to return a mock graph
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="I can help you with that."),
            ]
        }

        with patch("patient_agent_bench.assistant_agent.default_agent.create_agent") as mock_create:
            mock_create.return_value = mock_graph

            result = agent.invoke(
                messages=[HumanMessage(content="Hello")],
                user_profile="<profile>Test</profile>",
            )

        assert "messages" in result
        assert len(result["messages"]) == 2

    def test_invoke_with_chat_history(self, model_config, mock_boto3_client, current_datetime):
        """Test invoke handles full message history."""
        agent = AssistantAgent(
            model_config=model_config,
            current_datetime=current_datetime,
            tools=[create_mock_tool("test")],
        )

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [
                HumanMessage(content="First message"),
                AIMessage(content="First response"),
                HumanMessage(content="Follow up question"),
                AIMessage(content="Follow up response"),
            ]
        }

        with patch("patient_agent_bench.assistant_agent.default_agent.create_agent") as mock_create:
            mock_create.return_value = mock_graph

            result = agent.invoke(
                messages=[
                    HumanMessage(content="First message"),
                    AIMessage(content="First response"),
                    HumanMessage(content="Follow up question"),
                ],
                user_profile="<profile>Test</profile>",
            )

        assert "messages" in result

    def test_invoke_raises_error_on_failure(self, model_config, mock_boto3_client, current_datetime):
        """Test invoke raises AssistantAgentError on failure."""
        agent = AssistantAgent(
            model_config=model_config,
            current_datetime=current_datetime,
            tools=[create_mock_tool("test")],
        )

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = Exception("LLM error")

        with patch("patient_agent_bench.assistant_agent.default_agent.create_agent") as mock_create:
            mock_create.return_value = mock_graph

            with pytest.raises(AssistantAgentError, match="LLM error"):
                agent.invoke(
                    messages=[HumanMessage(content="Test")],
                    user_profile="<profile>Test</profile>",
                )


class TestAssistantAgentFactory:
    """Tests for create_assistant_agent factory function."""

    def test_create_with_defaults(self, model_config, mock_boto3_client, current_datetime):
        """Test factory creates agent with defaults (empty tools without sandbox)."""
        spec = AgentSpec(model=model_config)
        agent = create_assistant_agent(
            spec=spec,
            current_datetime=current_datetime,
        )

        assert isinstance(agent, AssistantAgent)
        # Without sandbox, tools will be empty - this is expected behavior
        assert isinstance(agent.tools, list)

    def test_create_with_tool_registry(self, model_config, mock_boto3_client, current_datetime):
        """Test factory uses provided tool registry."""
        from patient_agent_bench.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(create_mock_tool("registry_tool"))
        spec = AgentSpec(model=model_config)

        agent = create_assistant_agent(
            spec=spec,
            current_datetime=current_datetime,
            tool_registry=registry,
        )

        assert len(agent.tools) == 1
        assert agent.tools[0].name == "registry_tool"


# =============================================================================
# UserAgent Tests
# =============================================================================

class TestUserAgentInit:
    """Tests for UserAgent initialization."""

    def test_init_with_required_params(self, model_config, mock_boto3_client, current_datetime):
        """Test agent initializes with required parameters."""
        agent = UserAgent(
            scenario="Patient needs medication refill",
            user_profile="<profile>John Doe</profile>",
            model_config=model_config,
            current_datetime=current_datetime,
            personality="cooperative",
        )

        assert agent.scenario == "Patient needs medication refill"
        assert agent.user_profile == "<profile>John Doe</profile>"
        assert agent.chat_history == []

    def test_init_formats_system_prompt(self, model_config, mock_boto3_client, current_datetime):
        """Test system prompt is formatted with scenario and profile."""
        agent = UserAgent(
            scenario="Test scenario",
            user_profile="Test profile",
            model_config=model_config,
            current_datetime=current_datetime,
            personality="cooperative",
        )

        assert "Test scenario" in agent.system_prompt
        assert "Test profile" in agent.system_prompt

    def test_init_with_custom_prompt(self, model_config, mock_boto3_client, current_datetime):
        """Test agent with custom system prompt template."""
        custom_prompt = "Custom: {scenario} - {user_profile}"

        agent = UserAgent(
            scenario="S",
            user_profile="P",
            model_config=model_config,
            current_datetime=current_datetime,
            system_prompt=custom_prompt,
            personality="cooperative",
        )

        assert agent.system_prompt == "Custom: S - P"

    def test_init_requires_current_datetime(self, model_config, mock_boto3_client):
        """Test that current_datetime is required."""
        with pytest.raises(ValueError, match="current_datetime is required"):
            UserAgent(
                scenario="Test",
                user_profile="Profile",
                model_config=model_config,
                current_datetime="",
                personality="cooperative",
            )


class TestUserAgentConversation:
    """Tests for UserAgent conversation methods."""

    def test_start_conversation(self, model_config, mock_boto3_client, current_datetime):
        """Test starting a conversation."""
        agent = UserAgent(
            scenario="Test",
            user_profile="Profile",
            model_config=model_config,
            current_datetime=current_datetime,
            personality="cooperative",
        )

        mock_response = MagicMock()
        mock_response.content = "I need help with my prescription"

        with patch.object(agent, 'llm') as mock_llm:
            mock_llm.invoke.return_value = mock_response

            message = agent.start_conversation()

        assert message == "I need help with my prescription"
        assert len(agent.chat_history) == 2  # Opening prompt + response

    def test_start_conversation_raises_on_llm_failure(
        self, model_config, mock_boto3_client, current_datetime
    ):
        """Test start_conversation raises exception on LLM failure.

        Validates: Requirements 3.9
        """
        agent = UserAgent(
            scenario="Test",
            user_profile="Profile",
            model_config=model_config,
            current_datetime=current_datetime,
            personality="cooperative",
        )

        with patch.object(agent, 'llm') as mock_llm:
            mock_llm.invoke.side_effect = Exception("LLM error")

            with pytest.raises(Exception, match="LLM error"):
                agent.start_conversation()

    def test_respond(self, model_config, mock_boto3_client, current_datetime):
        """Test responding to assistant message."""
        agent = UserAgent(
            scenario="Test",
            user_profile="Profile",
            model_config=model_config,
            current_datetime=current_datetime,
            personality="cooperative",
        )

        mock_response = MagicMock()
        mock_response.content = "Yes, that works for me"

        with patch.object(agent, 'llm') as mock_llm:
            mock_llm.invoke.return_value = mock_response

            response = agent.respond("I can schedule that for tomorrow")

        assert response == "Yes, that works for me"
        # Should have assistant message + patient response
        assert len(agent.chat_history) == 2

    def test_respond_error_handling(self, model_config, mock_boto3_client, current_datetime):
        """Test respond handles errors gracefully."""
        agent = UserAgent(
            scenario="Test",
            user_profile="Profile",
            model_config=model_config,
            current_datetime=current_datetime,
            personality="cooperative",
        )

        with patch.object(agent, 'llm') as mock_llm:
            mock_llm.invoke.side_effect = Exception("Error")

            response = agent.respond("Test message")

        assert response == "I'm not sure what to say."

    def test_is_conversation_complete_true(self, model_config, mock_boto3_client, current_datetime):
        """Test conversation completion detection - positive case with drop-off signal."""
        agent = UserAgent(
            scenario="Test",
            user_profile="Profile",
            model_config=model_config,
            current_datetime=current_datetime,
            personality="cooperative",
        )

        agent.chat_history = [
            {"role": "user", "content": "Here's your appointment info."},
            {"role": "assistant", "content": "[DROPPED OFF CONVERSATION]"},
        ]

        assert agent.is_conversation_complete() is True

    def test_is_conversation_complete_false(
        self, model_config, mock_boto3_client, current_datetime
    ):
        """Test conversation completion detection - negative case."""
        agent = UserAgent(
            scenario="Test",
            user_profile="Profile",
            model_config=model_config,
            current_datetime=current_datetime,
            personality="cooperative",
        )

        agent.chat_history = [
            {"role": "user", "content": "I have another question"},
            {"role": "assistant", "content": "what about my meds?"},
        ]

        assert agent.is_conversation_complete() is False

    def test_is_conversation_complete_empty(
        self, model_config, mock_boto3_client, current_datetime
    ):
        """Test conversation completion with empty history."""
        agent = UserAgent(
            scenario="Test",
            user_profile="Profile",
            model_config=model_config,
            current_datetime=current_datetime,
            personality="cooperative",
        )

        assert agent.is_conversation_complete() is False

    def test_get_chat_history(self, model_config, mock_boto3_client, current_datetime):
        """Test getting chat history returns a copy."""
        agent = UserAgent(
            scenario="Test",
            user_profile="Profile",
            model_config=model_config,
            current_datetime=current_datetime,
            personality="cooperative",
        )

        agent.chat_history = [{"role": "user", "content": "Test"}]
        history = agent.get_chat_history()

        assert history == agent.chat_history
        assert history is not agent.chat_history  # Should be a copy

    def test_reset(self, model_config, mock_boto3_client, current_datetime):
        """Test resetting conversation history."""
        agent = UserAgent(
            scenario="Test",
            user_profile="Profile",
            model_config=model_config,
            current_datetime=current_datetime,
            personality="cooperative",
        )

        agent.chat_history = [{"role": "user", "content": "Test"}]
        agent.reset()

        assert not agent.chat_history


class TestUserAgentFactory:
    """Tests for create_user_agent factory function."""

    def test_create_user_agent(self, model_config, mock_boto3_client, current_datetime):
        """Test factory creates agent correctly."""
        spec = AgentSpec(model=model_config)
        agent = create_user_agent(
            spec=spec,
            scenario="Test scenario",
            user_profile="Test profile",
            current_datetime=current_datetime,
            personality="cooperative",
        )

        assert isinstance(agent, UserAgent)
        assert agent.scenario == "Test scenario"

    def test_create_user_agent_with_custom_prompt(
        self, model_config, mock_boto3_client, current_datetime
    ):
        """Test factory creates agent with custom prompt."""
        custom_prompt = "Custom prompt: {scenario}"
        spec = AgentSpec(model=model_config, system_prompt=custom_prompt)
        agent = create_user_agent(
            spec=spec,
            scenario="Test scenario",
            user_profile="Test profile",
            current_datetime=current_datetime,
            personality="cooperative",
        )

        assert isinstance(agent, UserAgent)
        assert "Custom prompt: Test scenario" in agent.system_prompt


# =============================================================================
# User Agent Prompt Tests
# =============================================================================

class TestUserAgentPrompt:
    """Tests for user agent default prompt structure."""

    def test_system_prompt_excludes_seed_intent(self):
        """Test SYSTEM_PROMPT does not contain seed-intent section.

        Validates: Requirements 3.5
        """
        assert "seed-intent" not in USER_SYSTEM_PROMPT
        assert "seed_intent" not in USER_SYSTEM_PROMPT

    def test_system_prompt_excludes_initial_query(self):
        """Test SYSTEM_PROMPT does not contain initial_query placeholder.

        Validates: Requirements 3.5
        """
        assert "initial_query" not in USER_SYSTEM_PROMPT
        assert "{initial_query}" not in USER_SYSTEM_PROMPT
