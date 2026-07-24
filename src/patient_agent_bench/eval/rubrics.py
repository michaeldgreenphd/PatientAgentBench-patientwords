# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Rubric registry for PatientAgentBench.

Centralizes rubric class imports and provides helpers for creating
default rubric instances and extracting default weights. Used by
both Evaluator (needs instances) and aggregator (needs weights).
"""

from typing import Dict, List

from patient_agent_bench.eval.base_rubric import BaseRubric
from patient_agent_bench.eval.task_completion import TaskCompletionRubric
from patient_agent_bench.eval.clinical_safety import ClinicalSafetyRubric
from patient_agent_bench.eval.workflow_accuracy import WorkflowAccuracyRubric
from patient_agent_bench.eval.triage_quality import TriageQualityRubric
from patient_agent_bench.eval.clinical_helpfulness import ClinicalHelpfulnessRubric
from patient_agent_bench.eval.conversational_quality import ConversationalQualityRubric

# Ordered list of all rubric classes. Add new rubrics here.
RUBRIC_CLASSES = [
    TaskCompletionRubric,
    ClinicalSafetyRubric,
    WorkflowAccuracyRubric,
    TriageQualityRubric,
    ClinicalHelpfulnessRubric,
    ConversationalQualityRubric,
]


def create_default_rubrics(
    model_config,
    role_arn=None,
) -> List[BaseRubric]:
    """Create instances of all default evaluation rubrics."""
    return [
        cls(model_config, role_arn=role_arn)
        for cls in RUBRIC_CLASSES
    ]


def get_default_rubric_weights() -> Dict[str, float]:
    """
    Return the default rubric weights from class definitions.

    Reads RUBRIC_NAME and WEIGHT class attributes directly,
    no instantiation needed.
    """
    return {cls.RUBRIC_NAME: cls.WEIGHT for cls in RUBRIC_CLASSES}
