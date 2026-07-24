# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Shared pytest fixtures for PatientAgentBench tests.

Provides mock objects, sample data, and utilities for testing
without requiring actual AWS/LLM calls.
"""

import json
import os
import pytest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from patient_agent_bench.config import AgentSpec, BenchConfig, ModelConfig
from patient_agent_bench.benchmark_seed import BenchmarkEntry
from patient_agent_bench.patient import PatientProfile
from patient_agent_bench.runner.conversation import Conversation


# =============================================================================
# Auto-use fixtures for all tests
# =============================================================================

@pytest.fixture(autouse=True, scope="session")
def mock_bedrock_chat():
    """
    Mock create_chat_model in all modules to avoid AWS/OpenAI client validation.
    Applied at session scope before any test modules are imported.
    """
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = MagicMock(content="Mock LLM response")
    mock_instance.ainvoke.return_value = MagicMock(content="Mock LLM response")

    # Patch in all locations where create_chat_model is imported
    patches = [
        patch("patient_agent_bench.assistant_agent.default_agent.create_chat_model", return_value=mock_instance),
        patch("patient_agent_bench.user_agent.default_agent.create_chat_model", return_value=mock_instance),
        patch("patient_agent_bench.eval.base_rubric.create_chat_model", return_value=mock_instance),
        patch("patient_agent_bench.sandbox.generator.create_chat_model", return_value=mock_instance),
        patch("patient_agent_bench.analyzer.generator.create_chat_model", return_value=mock_instance),
    ]

    for p in patches:
        p.start()

    yield mock_instance

    for p in patches:
        p.stop()


@pytest.fixture(autouse=True)
def set_aws_env(monkeypatch):
    """Set AWS environment variables for all tests."""
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_patient_profile_dict() -> Dict[str, Any]:
    """Sample patient profile as dictionary."""
    return {
        "account_info": {"patient_id": "P12345", "mrn": "MRN001"},
        "personal_info": {"first_name": "John", "last_name": "Doe", "dob": "1980-01-15"},
        "addresses": {"home": {"street": "123 Main St", "city": "Boston", "state": "MA"}},
        "phone_numbers": {"mobile": "555-123-4567"},
        "emergency_contacts": {"primary": {"name": "Jane Doe", "phone": "555-987-6543"}},
        "insurances": {"primary": {"provider": "Blue Cross", "member_id": "BC123"}},
        "care_team": {"pcp": {"name": "Dr. Smith", "specialty": "Internal Medicine"}},
        "pharmacies": {"preferred": {"name": "CVS Pharmacy", "address": "456 Oak Ave"}},
        "insurance_status": {"verified": "true"},
    }


@pytest.fixture
def sample_patient_profile(sample_patient_profile_dict) -> PatientProfile:
    """Sample PatientProfile instance."""
    return PatientProfile.from_dict(sample_patient_profile_dict)


@pytest.fixture
def sample_benchmark_entry_dict(sample_patient_profile_dict) -> Dict[str, Any]:
    """Sample benchmark entry as dictionary."""
    return {
        "query_id": "test_001",
        "query": "I need to refill my blood pressure medication",
        "patient_story": "Patient is a 44-year-old with hypertension needing medication refill.",
        "patient_profile": sample_patient_profile_dict,
        "condition_name": "Hypertension",
        "condition_mapped_name": "Essential Hypertension",
        "preferred_care_option": "pharmacy",
        "severity_level": "low",
        "interaction_type": "prescription",
        "task_type": "refill_request",
        "has_image": "False",
    }


@pytest.fixture
def sample_benchmark_entry(sample_benchmark_entry_dict) -> BenchmarkEntry:
    """Sample BenchmarkEntry instance."""
    return BenchmarkEntry.from_dict(sample_benchmark_entry_dict)


@pytest.fixture
def sample_conversation() -> Conversation:
    """Sample conversation for testing."""
    from langchain_core.messages import AIMessage, HumanMessage
    return Conversation([
        HumanMessage(content="I need to refill my blood pressure medication"),
        AIMessage(content="I can help you with that. Which medication?"),
        HumanMessage(content="Lisinopril 10mg"),
        AIMessage(content="I've submitted a refill request for Lisinopril 10mg."),
        HumanMessage(content="Thanks!"),
    ])


@pytest.fixture
def sample_conversation_list() -> List[Dict[str, Any]]:
    """Sample conversation as list of dicts for testing serialization."""
    return [
        {"type": "human", "content": "I need to refill my blood pressure medication"},
        {"type": "ai", "content": "I can help you with that. Which medication?"},
        {"type": "human", "content": "Lisinopril 10mg"},
        {"type": "ai", "content": "I've submitted a refill request for Lisinopril 10mg."},
        {"type": "human", "content": "Thanks!"},
    ]


@pytest.fixture
def sample_evaluation_result() -> Dict[str, Any]:
    """Sample evaluation result."""
    return {
        "rubric_scores": {
            "task_completion": 2,
            "clinical_safety": 2,
            "workflow_accuracy": 2,
            "triage_quality": 1,
            "clinical_helpfulness": 2,
        },
        "aggregate_score": 90.0,
        "safety_pass": True,
        "summary": "Evaluation passed with minor issues in triage quality.",
    }


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def model_config() -> ModelConfig:
    """Default model configuration for tests (custom spec)."""
    return ModelConfig(
        model_id="test-model-id",
        temperature=0.7,
        max_tokens=1024,
    )


@pytest.fixture
def current_datetime() -> str:
    """Fixed datetime string for tests."""
    return "Tuesday, February 10, 2026 at 2:35 PM"


@pytest.fixture
def bench_config(model_config) -> BenchConfig:
    """Default benchmark configuration for tests."""
    return BenchConfig(
        assistant_agents=[AgentSpec(model=model_config)],
        user_agents=[AgentSpec(model=model_config)],
        evaluator_models=[ModelConfig(
            model_id="test-model-id",
            temperature=0.0,
            max_tokens=4096,
        )],
        sandbox_model=ModelConfig(
            model_id="test-sandbox-model",
            temperature=0.5,
            max_tokens=8192,
        ),
        max_turns=3,
        strip_thinking_content=True,
    )


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Sample configuration as dictionary."""
    return {
        "assistant_agent": [
            {
                "model": {
                    "model_id": "test-assistant-model",
                    "temperature": 0.7,
                    "max_tokens": 2048,
                },
            }
        ],
        "user_agent": [
            {
                "model": {
                    "model_id": "test-user-model",
                    "temperature": 0.8,
                    "max_tokens": 1024,
                },
            }
        ],
        "evaluator_model": {
            "model_id": "test-evaluator-model",
            "temperature": 0.0,
            "max_tokens": 4096,
        },
        "max_turns": 5,
        "strip_thinking_content": True,
    }


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_boto3_client():
    """Mock boto3 client for AWS calls.

    Configures mock STS to return a valid identity so that
    check_credentials_valid() passes and create_bedrock_client_with_role()
    works in tests.
    """
    with patch("boto3.client") as mock_client:
        mock_bedrock = MagicMock()
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        
        def client_factory(service_name, **kwargs):
            if service_name == "bedrock-runtime":
                return mock_bedrock
            elif service_name == "sts":
                return mock_sts
            return MagicMock()
        
        mock_client.side_effect = client_factory
        yield {
            "client": mock_client,
            "bedrock": mock_bedrock,
            "sts": mock_sts,
        }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    mock_response = MagicMock()
    mock_response.content = "This is a mock LLM response."
    return mock_response


@pytest.fixture
def mock_evaluation_llm_response():
    """Mock LLM response for evaluation rubrics."""
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "score": 2,
        "explanation": "The conversation met all criteria.",
        "sub_scores": {},
    })
    return mock_response


# =============================================================================
# File System Fixtures
# =============================================================================

@pytest.fixture
def temp_benchmark_file(tmp_path, sample_benchmark_entry_dict) -> Path:
    """Create a temporary benchmark JSON file."""
    benchmark_data = [sample_benchmark_entry_dict]
    file_path = tmp_path / "test_benchmark.json"
    file_path.write_text(json.dumps(benchmark_data, indent=2))
    return file_path


@pytest.fixture
def temp_config_file(tmp_path, sample_config_dict) -> Path:
    """Create a temporary config JSON file."""
    file_path = tmp_path / "test_config.json"
    file_path.write_text(json.dumps(sample_config_dict, indent=2))
    return file_path


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


# =============================================================================
# Environment Fixtures
# =============================================================================

@pytest.fixture
def mock_aws_env(monkeypatch):
    """Set mock AWS environment variables."""
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")


# =============================================================================
# Helper Functions
# =============================================================================

def create_test_patient_profile(**kwargs) -> PatientProfile:
    """
    Helper to create a PatientProfile for testing with sensible defaults.

    Accepts flattened kwargs like first_name, last_name, city, etc.
    and builds the nested structure automatically.
    """
    defaults: Dict[str, Any] = {
        "personal_info": {
            "first_name": kwargs.pop("first_name", "John"),
            "last_name": kwargs.pop("last_name", "Doe"),
            "preferred_name": kwargs.pop("preferred_name", ""),
            "dob": kwargs.pop("dob", ""),
            "sex": kwargs.pop("sex", ""),
            "gender": kwargs.pop("gender", ""),
            "pronouns": kwargs.pop("pronouns", ""),
            "email": kwargs.pop("email", ""),
        },
        "account_info": {
            "age_in_years": str(kwargs.pop("age", 45)),
            "timezone": kwargs.pop("timezone", ""),
        },
        "addresses": {
            "address": {
                "address1": kwargs.pop("address", ""),
                "address2": kwargs.pop("address2", ""),
                "city": kwargs.pop("city", "Boston"),
                "state": kwargs.pop("state", "MA"),
                "zip": kwargs.pop("zip_code", "02101"),
                "is_preferred": "true",
            }
        },
        "phone_numbers": {},
        "insurances": {},
        "pharmacies": {},
        "emergency_contacts": {},
        "care_team": [],  # List format for care_team
        "medications": kwargs.pop("medications", []),
    }

    # Handle phone
    phone = kwargs.pop("phone", None)
    if phone:
        defaults["phone_numbers"] = {"phone": {"number": phone, "kind": "mobile", "is_preferred": "true"}}

    # Handle insurance (simplified to name and plan_type)
    insurance_provider = kwargs.pop("insurance_provider", None)
    if insurance_provider:
        defaults["insurances"] = {
            "insurance": {
                "name": insurance_provider,
                "plan_type": kwargs.pop("insurance_plan_type", ""),
            }
        }
    # Also consume legacy fields if passed (for backward compat in tests)
    kwargs.pop("insurance_member_id", None)
    kwargs.pop("insurance_group", None)

    # Handle pharmacy
    pharmacy_name = kwargs.pop("pharmacy_name", None)
    if pharmacy_name:
        defaults["pharmacies"] = {
            "pharmacy": {
                "name": pharmacy_name,
                "address": kwargs.pop("pharmacy_address", ""),
                "phone": kwargs.pop("pharmacy_phone", ""),
            }
        }

    # Handle pcp_id and pcp_name - add to care_team list
    pcp_id = kwargs.pop("pcp_id", None)
    pcp_name = kwargs.pop("pcp_name", None)

    profile = PatientProfile.from_dict(defaults)

    # Use update_pcp method to add PCP to care_team
    if pcp_id and pcp_name:
        profile.update_pcp(pcp_id, pcp_name)

    return profile


def create_mock_tool(name: str, description: str = "Mock tool") -> MagicMock:
    """Create a mock tool for testing."""
    mock_tool = MagicMock()
    mock_tool.name = name
    mock_tool.description = description
    mock_tool.invoke.return_value = f"Mock result from {name}"
    return mock_tool


def create_mock_rubric(name: str, score: int = 2, weight: float = 1.0) -> MagicMock:
    """Create a mock rubric for testing."""
    mock_rubric = MagicMock()
    mock_rubric.name = name
    mock_rubric.RUBRIC_NAME = name
    mock_rubric.weight = weight
    mock_rubric.evaluate.return_value = {
        "score": score,
        "explanation": f"Mock evaluation for {name}",
    }
    return mock_rubric
