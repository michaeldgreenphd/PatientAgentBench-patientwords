# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Experiment configuration for multi-experiment benchmark runs.

Contains the ExperimentConfig dataclass representing a single experiment
with specific model configurations for assistant, user, and evaluator agents.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List

from patient_agent_bench.config import AgentSpec, ModelConfig, parse_model_config


@dataclass
class ExperimentConfig:
    """
    Configuration for a single experiment in a multi-experiment benchmark run.

    An experiment represents a specific combination of assistant and user
    agent configurations. The experiment_id uniquely identifies this
    combination using the format "{assistant_idx}_{user_idx}".

    Each experiment runs all configured evaluator models internally,
    with results aggregated into a merged evaluation.
    """

    experiment_id: str
    assistant_agent: AgentSpec
    user_agent: AgentSpec
    evaluator_models: List[ModelConfig]
    sandbox_model: ModelConfig

    # Indices into the original config lists
    assistant_idx: int
    user_idx: int

    # Conversation settings
    max_turns: int
    strip_thinking_content: bool

    # Analyzer model (for generating evaluation insights)
    analyzer_model: ModelConfig

    # Aggregation method for multi-evaluator results
    aggregation_method: str = "average"

    def __post_init__(self):
        """Validate ExperimentConfig fields."""
        if not self.evaluator_models:
            raise ValueError(
                "evaluator_models must contain at least one evaluator model"
            )

    # --- Backward-compatible properties ---
    # These allow existing code that references the old flat fields
    # (assistant_model, user_model, assistant_prompt, user_prompt)
    # to continue working until those call sites are migrated.

    @property
    def assistant_model(self) -> ModelConfig:
        """Backward-compatible access to assistant model config."""
        return self.assistant_agent.model

    @property
    def user_model(self) -> ModelConfig:
        """Backward-compatible access to user model config."""
        return self.user_agent.model

    @property
    def assistant_prompt(self) -> str:
        """Backward-compatible access to assistant prompt name."""
        return self.assistant_agent.prompt

    @property
    def user_prompt(self) -> str:
        """Backward-compatible access to user prompt name."""
        return self.user_agent.prompt

    def conversation_signature(self) -> str:
        """
        Generate a signature for conversation reuse detection.

        The signature hashes the full AgentSpec for both assistant and user,
        since prompt and agent_class affect conversation output and must be
        included in the signature (not just the model).

        Returns:
            A hex string hash uniquely identifying the assistant+user combination.
        """
        signature_data = {
            "assistant": self.assistant_agent.to_dict(),
            "user": self.user_agent.to_dict(),
        }
        json_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize ExperimentConfig for saving to experiment config file.

        Returns:
            Dictionary representation of the experiment configuration.
        """
        return {
            "experiment_id": self.experiment_id,
            "assistant_agent": self.assistant_agent.to_dict(),
            "user_agent": self.user_agent.to_dict(),
            "evaluator_models": [m.to_dict() for m in self.evaluator_models],
            "sandbox_model": self.sandbox_model.to_dict(),
            "analyzer_model": self.analyzer_model.to_dict(),
            "assistant_idx": self.assistant_idx,
            "user_idx": self.user_idx,
            "aggregation_method": self.aggregation_method,
            "max_turns": self.max_turns,
            "strip_thinking_content": self.strip_thinking_content,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """
        Create ExperimentConfig from a dictionary.

        Supports both new format (evaluator_models list) and legacy format
        (evaluator_model singular + evaluator_idx).

        Args:
            data: Dictionary containing experiment configuration.

        Returns:
            ExperimentConfig instance.
        """
        # Handle evaluator_models (new format) vs evaluator_model (legacy)
        if "evaluator_models" in data:
            evaluator_models = [
                parse_model_config(m) for m in data["evaluator_models"]
            ]
        elif "evaluator_model" in data:
            evaluator_models = [parse_model_config(data["evaluator_model"])]
        else:
            evaluator_models = [ModelConfig()]

        return cls(
            experiment_id=data["experiment_id"],
            assistant_agent=AgentSpec.from_dict(data["assistant_agent"]),
            user_agent=AgentSpec.from_dict(data["user_agent"]),
            evaluator_models=evaluator_models,
            sandbox_model=parse_model_config(data["sandbox_model"]),
            analyzer_model=parse_model_config(data["analyzer_model"]),
            assistant_idx=data["assistant_idx"],
            user_idx=data["user_idx"],
            aggregation_method=data.get("aggregation_method", "average"),
            max_turns=data.get("max_turns", 3),
            strip_thinking_content=data.get("strip_thinking_content", True),
        )
