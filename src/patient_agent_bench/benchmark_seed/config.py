# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Distribution configuration loading and validation for benchmark seed generation.

This module handles loading and validation of distribution configuration from JSON files.
It supports configurable probability distributions for conditions, severity levels, tasks,
age groups, genders, and sexes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json


@dataclass
class TaskDefinition:
    """A task category with weighted sub-tasks.

    Attributes:
        category: The task category name (e.g., "prescription", "appointment")
        weight: The weight for selecting this category
        subtasks: List of (subtask_name, weight) tuples for sub-task selection
    """

    category: str
    weight: float
    subtasks: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class DistributionConfig:
    """Configuration for attribute distributions.

    This class holds all the distribution configurations for generating benchmark seeds.
    Each attribute type has a list of (name, weight) tuples where weights are normalized
    to sum to 1.0.

    Attributes:
        conditions: List of (condition_name, weight) tuples
        severity_levels: List of (severity, weight) tuples
        tasks: List of TaskDefinition objects for hierarchical task selection
        age_groups: List of (age_group_name, weight) tuples
        genders: List of (gender, weight) tuples
        sexes: List of (sex, weight) tuples
        care_preferences: List of (care_preference, weight) tuples
    """

    conditions: List[Tuple[str, float]]
    severity_levels: List[Tuple[str, float]]
    tasks: List[TaskDefinition]
    age_groups: List[Tuple[str, float]]
    genders: List[Tuple[str, float]]
    sexes: List[Tuple[str, float]]
    care_preferences: List[Tuple[str, float]]
    personalities: List[Tuple[str, float]]

    @classmethod
    def from_file(cls, path: Path) -> "DistributionConfig":
        """Load configuration from JSON file.

        Args:
            path: Path to the JSON configuration file

        Returns:
            DistributionConfig instance with normalized weights

        Raises:
            FileNotFoundError: If the configuration file does not exist
            ValueError: If the JSON is malformed or missing required fields
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in configuration file {path}: {e}") from e

        # Parse and normalize each attribute type
        conditions = cls._parse_weighted_list(data, "conditions", path)
        severity_levels = cls._parse_weighted_list(data, "severity_levels", path)
        age_groups = cls._parse_weighted_list(data, "age_groups", path)
        genders = cls._parse_weighted_list(data, "genders", path)
        sexes = cls._parse_weighted_list(data, "sexes", path)
        care_preferences = cls._parse_weighted_list(data, "care_preferences", path)
        personalities = cls._parse_weighted_list(data, "personalities", path)

        # Parse tasks with hierarchical structure
        tasks = cls._parse_tasks(data, path)

        config = cls(
            conditions=conditions,
            severity_levels=severity_levels,
            tasks=tasks,
            age_groups=age_groups,
            genders=genders,
            sexes=sexes,
            care_preferences=care_preferences,
            personalities=personalities,
        )

        config.validate()
        return config

    @classmethod
    def _parse_weighted_list(
        cls,
        data: Dict[str, Any],
        field_name: str,
        path: Path,
    ) -> List[Tuple[str, float]]:
        """Parse a list of weighted items from the configuration.

        Args:
            data: The parsed JSON data
            field_name: The name of the field to parse
            path: Path to the config file (for error messages)

        Returns:
            List of (name, weight) tuples with normalized weights

        Raises:
            ValueError: If the field is missing or has invalid format
        """
        if field_name not in data:
            raise ValueError(f"Missing required field '{field_name}' in configuration file {path}")

        items = data[field_name]
        if not isinstance(items, list):
            raise ValueError(f"Field '{field_name}' must be a list in configuration file {path}")

        if len(items) == 0:
            raise ValueError(f"Field '{field_name}' cannot be empty in configuration file {path}")

        result: List[Tuple[str, float]] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Item {i} in '{field_name}' must be an object " f"in configuration file {path}"
                )

            if "name" not in item:
                raise ValueError(
                    f"Item {i} in '{field_name}' missing 'name' field "
                    f"in configuration file {path}"
                )

            name = item["name"]
            # Use weight of 1.0 if not specified (uniform distribution)
            weight = item.get("weight", 1.0)

            if not isinstance(weight, (int, float)):
                raise ValueError(
                    f"Weight for '{name}' in '{field_name}' must be a number "
                    f"in configuration file {path}"
                )

            if weight < 0:
                raise ValueError(
                    f"Weight for '{name}' in '{field_name}' cannot be negative "
                    f"in configuration file {path}"
                )

            result.append((name, float(weight)))

        return cls._normalize_weights(result)

    @classmethod
    def _parse_tasks(cls, data: Dict[str, Any], path: Path) -> List[TaskDefinition]:
        """Parse the hierarchical task definitions from the configuration.

        Args:
            data: The parsed JSON data
            path: Path to the config file (for error messages)

        Returns:
            List of TaskDefinition objects with normalized weights

        Raises:
            ValueError: If tasks are missing or have invalid format
        """
        if "tasks" not in data:
            raise ValueError(f"Missing required field 'tasks' in configuration file {path}")

        tasks_data = data["tasks"]
        if not isinstance(tasks_data, list):
            raise ValueError(f"Field 'tasks' must be a list in configuration file {path}")

        if len(tasks_data) == 0:
            raise ValueError(f"Field 'tasks' cannot be empty in configuration file {path}")

        tasks: List[TaskDefinition] = []
        category_weights: List[Tuple[str, float]] = []

        for i, task_data in enumerate(tasks_data):
            task_def, cat_weight = cls._parse_single_task(task_data, i, path)
            tasks.append(task_def)
            category_weights.append(cat_weight)

        # Normalize category weights and update task definitions
        normalized_category_weights = cls._normalize_weights(category_weights)
        for task, (_, normalized_weight) in zip(tasks, normalized_category_weights):
            task.weight = normalized_weight

        return tasks

    @classmethod
    def _parse_single_task(
        cls,
        task_data: Any,
        index: int,
        path: Path,
    ) -> Tuple[TaskDefinition, Tuple[str, float]]:
        """Parse a single task definition from the configuration.

        Args:
            task_data: The task data from JSON
            index: The index of this task in the list
            path: Path to the config file (for error messages)

        Returns:
            Tuple of (TaskDefinition, (category, weight))

        Raises:
            ValueError: If task data is invalid
        """
        if not isinstance(task_data, dict):
            raise ValueError(f"Task {index} must be an object in configuration file {path}")

        if "category" not in task_data:
            raise ValueError(f"Task {index} missing 'category' field in configuration file {path}")

        category = task_data["category"]
        weight = task_data.get("weight", 1.0)

        if not isinstance(weight, (int, float)):
            raise ValueError(
                f"Weight for task category '{category}' must be a number "
                f"in configuration file {path}"
            )

        if weight < 0:
            raise ValueError(
                f"Weight for task category '{category}' cannot be negative "
                f"in configuration file {path}"
            )

        # Parse subtasks
        subtasks = cls._parse_subtasks(task_data, category, path)

        task_def = TaskDefinition(
            category=category,
            weight=weight,  # Will be normalized later
            subtasks=subtasks,
        )

        return task_def, (category, float(weight))

    @classmethod
    def _parse_subtasks(
        cls,
        task_data: Dict[str, Any],
        category: str,
        path: Path,
    ) -> List[Tuple[str, float]]:
        """Parse subtasks for a task category.

        Args:
            task_data: The task data containing subtasks
            category: The category name (for error messages)
            path: Path to the config file (for error messages)

        Returns:
            List of (subtask_name, normalized_weight) tuples

        Raises:
            ValueError: If subtasks are invalid
        """
        subtasks_data = task_data.get("subtasks", [])
        if not isinstance(subtasks_data, list):
            raise ValueError(
                f"Subtasks for category '{category}' must be a list "
                f"in configuration file {path}"
            )

        if len(subtasks_data) == 0:
            raise ValueError(
                f"Task category '{category}' must have at least one subtask "
                f"in configuration file {path}"
            )

        subtasks: List[Tuple[str, float]] = []
        for j, subtask_data in enumerate(subtasks_data):
            if not isinstance(subtask_data, dict):
                raise ValueError(
                    f"Subtask {j} in category '{category}' must be an object "
                    f"in configuration file {path}"
                )

            if "name" not in subtask_data:
                raise ValueError(
                    f"Subtask {j} in category '{category}' missing 'name' field "
                    f"in configuration file {path}"
                )

            subtask_name = subtask_data["name"]
            subtask_weight = subtask_data.get("weight", 1.0)

            if not isinstance(subtask_weight, (int, float)):
                raise ValueError(
                    f"Weight for subtask '{subtask_name}' in category '{category}' "
                    f"must be a number in configuration file {path}"
                )

            if subtask_weight < 0:
                raise ValueError(
                    f"Weight for subtask '{subtask_name}' in category '{category}' "
                    f"cannot be negative in configuration file {path}"
                )

            subtasks.append((subtask_name, float(subtask_weight)))

        return cls._normalize_weights(subtasks)

    @classmethod
    def _normalize_weights(cls, items: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Normalize weights to sum to 1.0.

        If all weights are zero, uses uniform distribution.

        Args:
            items: List of (name, weight) tuples

        Returns:
            List of (name, normalized_weight) tuples that sum to 1.0
        """
        if not items:
            return items

        total_weight = sum(weight for _, weight in items)

        if total_weight == 0:
            # Use uniform distribution if all weights are zero
            uniform_weight = 1.0 / len(items)
            return [(name, uniform_weight) for name, _ in items]

        return [(name, weight / total_weight) for name, weight in items]

    def validate(self) -> None:
        """Validate configuration completeness.

        Ensures all required fields are present and have valid values.

        Raises:
            ValueError: If configuration is invalid
        """
        self._validate_non_empty_lists()
        self._validate_weight_sums()
        self._validate_tasks()

    def _validate_non_empty_lists(self) -> None:
        """Validate that all required lists are non-empty."""
        if not self.conditions:
            raise ValueError("Configuration must have at least one condition")

        if not self.severity_levels:
            raise ValueError("Configuration must have at least one severity level")

        if not self.tasks:
            raise ValueError("Configuration must have at least one task category")

        if not self.age_groups:
            raise ValueError("Configuration must have at least one age group")

        if not self.genders:
            raise ValueError("Configuration must have at least one gender")

        if not self.sexes:
            raise ValueError("Configuration must have at least one sex")

        if not self.care_preferences:
            raise ValueError("Configuration must have at least one care preference")

    def _validate_weight_sums(self) -> None:
        """Validate that weights sum to approximately 1.0."""
        tolerance = 1e-9

        weight_checks = [
            (self.conditions, "Conditions"),
            (self.severity_levels, "Severity levels"),
            (self.age_groups, "Age groups"),
            (self.genders, "Genders"),
            (self.sexes, "Sexes"),
            (self.care_preferences, "Care preferences"),
        ]

        for items, name in weight_checks:
            weight_sum = sum(w for _, w in items)
            if abs(weight_sum - 1.0) > tolerance:
                raise ValueError(f"{name} weights must sum to 1.0, got {weight_sum}")

        # Validate task category weights
        task_weights_sum = sum(task.weight for task in self.tasks)
        if abs(task_weights_sum - 1.0) > tolerance:
            raise ValueError(f"Task category weights must sum to 1.0, got {task_weights_sum}")

    def _validate_tasks(self) -> None:
        """Validate each task's subtask weights."""
        tolerance = 1e-9

        for task in self.tasks:
            if not task.subtasks:
                raise ValueError(f"Task category '{task.category}' must have at least one subtask")

            subtask_sum = sum(w for _, w in task.subtasks)
            if abs(subtask_sum - 1.0) > tolerance:
                raise ValueError(
                    f"Subtask weights for category '{task.category}' "
                    f"must sum to 1.0, got {subtask_sum}"
                )
