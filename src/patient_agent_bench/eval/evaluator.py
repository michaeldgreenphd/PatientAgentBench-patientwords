# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Evaluator for PatientAgentBench.

Orchestrates all evaluation rubrics and provides aggregate scoring.
"""

import asyncio
from functools import partial
from typing import Any, Dict, List, Optional

from patient_agent_bench.config import ModelConfig
from patient_agent_bench.logging_config import get_logger
from patient_agent_bench.runner.conversation import Conversation
from patient_agent_bench.eval.constants import MAX_SCORE
from patient_agent_bench.eval.base_rubric import BaseRubric
from patient_agent_bench.eval.rubrics import create_default_rubrics

logger = get_logger(__name__)


class Evaluator:
    """
    Orchestrates evaluation across all 6 rubric dimensions.

    Provides both individual rubric scores and aggregate metrics.
    Clinical safety violations override other scores.

    Each instance is self-contained with its own rubrics and bedrock
    client. For parallel execution, create a new Evaluator per task
    via evaluate_async() which handles client isolation.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        rubrics: Optional[List[BaseRubric]] = None,
        role_arn: Optional[str] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            model_config: Model configuration for rubrics.
            rubrics: Optional custom rubrics. Uses all 6 defaults if None.
            role_arn: Optional AWS IAM role ARN. Passed to rubrics so they
                     can create and refresh their own bedrock clients.
        """
        self.model_config = model_config
        self.rubrics = rubrics or create_default_rubrics(
            model_config, role_arn=role_arn
        )

    def evaluate(
        self,
        conversation_history: Conversation,
        user_profile: str,
        scenario: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a conversation across all rubrics.

        Args:
            conversation_history: The Conversation to evaluate
            user_profile: Patient profile information
            scenario: Optional scenario description

        Returns:
            Dictionary containing:
            - rubric_results: Individual results for each rubric
            - rubric_scores: Individual scores for each rubric
        """
        results = {}
        rubric_scores = {}

        for rubric in self.rubrics:
            try:
                result = rubric.evaluate(
                    conversation_history, user_profile, scenario
                )
                results[rubric.name] = result
                rubric_scores[rubric.name] = result.get("score", 0)
                logger.info(
                    f"%s: %d/{MAX_SCORE} (weight: %s)",
                    rubric.name,
                    result.get("score", 0),
                    rubric.weight,
                )
            except Exception as e:
                # Do NOT fabricate a passing/failing numeric score for an
                # infrastructure failure (e.g. connection error) — that would
                # poison the aggregate (a network blip scored as a clinical
                # safety violation). Re-raise so the orchestrator records this
                # conversation as a failed task ({"error": ...}), which is
                # excluded from scoring and re-run on resume.
                logger.error(
                    "Error evaluating rubric %s: %s", rubric.name, e
                )
                raise

        return {
            "rubric_results": results,
            "rubric_scores": rubric_scores,
        }

    def get_rubric_names(self) -> List[str]:
        """Get the names of all rubrics being used."""
        return [rubric.name for rubric in self.rubrics]


async def evaluate_async(
    model_config: ModelConfig,
    conversation_history: Conversation,
    user_profile: str,
    scenario: Optional[str] = None,
    assigned_role: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a conversation asynchronously with role-based isolation.

    Creates a fresh Evaluator per call. Each rubric creates its own
    bedrock client from the assigned_role, ensuring thread safety
    for parallel execution.

    Called by ParallelTaskOrchestrator which injects assigned_role
    from the role pool.
    """
    evaluator = Evaluator(
        model_config=model_config,
        role_arn=assigned_role,
    )

    # Run sync evaluation in thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(
            evaluator.evaluate,
            conversation_history,
            user_profile,
            scenario,
        ),
    )
