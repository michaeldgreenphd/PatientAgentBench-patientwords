# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Attribute selector for benchmark seed generation.

This module provides the AttributeSelector class for selecting attributes
from configured distributions using weighted random selection.
"""

import random
from typing import List, Optional, Tuple

from patient_agent_bench.benchmark_seed.config import DistributionConfig
from patient_agent_bench.benchmark_seed.benchmark_entry import BenchmarkSeed


class AttributeSelector:
    """Selects attributes from configured distributions using generic weighted selection.

    This class provides methods for selecting attribute values according to
    configured probability distributions. It supports reproducible selection
    via an optional random seed parameter.

    Attributes:
        _config: The distribution configuration containing attribute weights
        _rng: Random number generator for reproducible selection
    """

    def __init__(self, config: DistributionConfig, seed: Optional[int] = None):
        """Initialize the AttributeSelector.

        Args:
            config: Distribution configuration with attribute weights
            seed: Optional random seed for reproducibility
        """
        self._config = config
        self._rng = random.Random(seed)

    def weighted_choice(self, items: List[Tuple[str, float]]) -> str:
        """Generic weighted selection from a list of (value, weight) tuples.

        Uses the internal RNG for reproducibility when a seed is set.
        Weights are assumed to be normalized (sum to 1.0).

        Args:
            items: List of (value, weight) tuples

        Returns:
            Selected value string

        Raises:
            ValueError: If items list is empty
        """
        if not items:
            raise ValueError("Cannot select from empty list")

        # Extract values and weights
        values = [item[0] for item in items]
        weights = [item[1] for item in items]

        # Use random.choices with the internal RNG for weighted selection
        # This returns a list, so we take the first element
        selected = self._rng.choices(values, weights=weights, k=1)
        return selected[0]

    def select_task(self) -> Tuple[str, str]:
        """Select task category and subcategory using hierarchical weighted selection.

        First selects category using weighted_choice on category weights,
        then selects subcategory using weighted_choice on that category's subtasks.

        Returns:
            Tuple of (task_category, task_subcategory)
        """
        # Build list of (category, weight) tuples from task definitions
        category_items: List[Tuple[str, float]] = [
            (task.category, task.weight) for task in self._config.tasks
        ]

        # Select category
        selected_category = self.weighted_choice(category_items)

        # Find the selected task definition to get its subtasks
        selected_task = next(
            task for task in self._config.tasks if task.category == selected_category
        )

        # Select subcategory from the selected task's subtasks
        selected_subcategory = self.weighted_choice(selected_task.subtasks)

        return (selected_category, selected_subcategory)

    def create_seed(self) -> BenchmarkSeed:
        """Create a complete benchmark seed with all attributes.

        Uses weighted_choice for: condition, severity, age_group, gender, sex, care_preference
        Uses select_task for hierarchical task selection

        Note: Actual age is determined during LLM enrichment based on age_group.

        Returns:
            BenchmarkSeed with all attributes populated
        """
        condition = self.weighted_choice(self._config.conditions)
        severity = self.weighted_choice(self._config.severity_levels)
        task_cat, task_sub = self.select_task()
        age_group = self.weighted_choice(self._config.age_groups)
        gender = self.weighted_choice(self._config.genders)
        sex = self.weighted_choice(self._config.sexes)
        care_preference = self.weighted_choice(self._config.care_preferences)
        personality = self.weighted_choice(self._config.personalities)

        return BenchmarkSeed(
            condition_name=condition,
            severity_level=severity,
            task_category=task_cat,
            task_subcategory=task_sub,
            age_group=age_group,
            gender=gender,
            sex=sex,
            care_preference=care_preference,
            personality=personality,
        )
