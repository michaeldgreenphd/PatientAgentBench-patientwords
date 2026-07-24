# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Benchmark entry data models for PatientAgentBench.

Contains data classes representing benchmark entries at different stages:
- BenchmarkSeed: Initial seed with selected attributes (pre-enrichment)
- EnrichedBenchmarkEntry: Fully enriched entry ready for serialization
- BenchmarkEntry: Final entry loaded from benchmark JSON files

Also includes utilities for loading benchmark data from JSON files.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from patient_agent_bench.patient.patient_profile import (
    PatientProfile,
    _dict_to_xml,
    _escape_xml,
)
from patient_agent_bench.eval.demographics import derive_age_group, derive_gender_identity

# Re-export for backward compatibility
__all__ = [
    "BenchmarkSeed",
    "EnrichedBenchmarkEntry",
    "BenchmarkEntry",
    "PatientProfile",
    "load_benchmark_entries",
    "_dict_to_xml",
    "_escape_xml",
]


@dataclass
class BenchmarkSeed:
    """Initial benchmark seed with selected attributes.

    Attributes:
        condition_name: Selected medical condition
        severity_level: Selected severity
        task_category: Selected task category (e.g., "prescription")
        task_subcategory: Selected sub-task (e.g., "renewal")
        age_group: Selected age group name (e.g., "young_adult")
        gender: Selected gender
        sex: Selected biological sex
        care_preference: Selected care preference (e.g., "in person visit")
        personality: Personality trait profile (trait -> level mapping)
    """

    condition_name: str
    severity_level: str
    task_category: str
    task_subcategory: str
    age_group: str
    gender: str
    sex: str
    care_preference: str = ""
    personality: str = ""


@dataclass
class EnrichedBenchmarkEntry:
    """A fully enriched benchmark entry ready for serialization.

    This class represents a complete benchmark entry with all fields populated,
    including LLM-generated content (patient_profile, patient_story).

    Attributes:
        seed: The original BenchmarkSeed with selected attributes
        scenario_id: Unique UUID identifier for this entry
        patient_story: LLM-generated clinical narrative
        patient_profile: LLM-generated patient profile dictionary
        task_type: Concatenation of task_category and task_subcategory
        preferred_care_option: Care option (e.g., "Office Visit")
        has_image: Whether the entry has an associated image ("True"/"False")
        scenario_complexity: Complexity classification
    """

    seed: BenchmarkSeed
    scenario_id: str
    patient_story: str
    patient_profile: Dict[str, Any]
    task_type: str
    preferred_care_option: str
    has_image: str
    scenario_complexity: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching sample_benchmark.json format.

        Returns a dictionary with all fields required by BenchmarkEntry.from_dict(),
        ensuring compatibility with the existing benchmark runner.

        Returns:
            Dictionary with all benchmark entry fields
        """
        return {
            "condition_name": self.seed.condition_name,
            "preferred_care_option": self.preferred_care_option,
            "severity_level": self.seed.severity_level,
            "task_type": self.task_type,
            "scenario_id": self.scenario_id,
            "has_image": self.has_image,
            "patient_story": self.patient_story,
            "patient_profile": self.patient_profile,
            "scenario_complexity": self.scenario_complexity,
            "personality": self.seed.personality,
        }


@dataclass
class BenchmarkEntry:
    """A single benchmark entry from the dataset."""

    scenario_id: str
    patient_story: str
    patient_profile: PatientProfile
    condition_name: str = ""
    preferred_care_option: str = ""
    severity_level: str = ""
    task_type: str = ""
    has_image: str = "False"
    scenario_complexity: str = ""
    personality: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkEntry":
        """Create BenchmarkEntry from dictionary (JSON data)."""
        profile_data = data.get("patient_profile", {})
        profile = PatientProfile.from_dict(profile_data) if profile_data else PatientProfile()

        # Support both query_id (legacy) and scenario_id (new), prefer scenario_id
        scenario_id = data.get("scenario_id") or data.get("query_id", "")

        return cls(
            scenario_id=scenario_id,
            patient_story=data.get("patient_story", ""),
            patient_profile=profile,
            condition_name=data.get("condition_name", ""),
            preferred_care_option=data.get("preferred_care_option", ""),
            severity_level=data.get("severity_level", ""),
            task_type=data.get("task_type", ""),
            has_image=data.get("has_image", "False"),
            scenario_complexity=data.get("scenario_complexity", ""),
            personality=data.get("personality", ""),
        )

    @property
    def patient_profile_xml(self) -> str:
        """Get patient profile as XML string."""
        return self.patient_profile.to_xml()

    @property
    def id(self) -> str:
        """Return scenario_id."""
        return self.scenario_id

    @property
    def scenario(self) -> str:
        """Alias for patient_story (used by agents)."""
        return self.patient_story

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata fields as dictionary."""
        return {
            "condition_name": self.condition_name,
            "preferred_care_option": self.preferred_care_option,
            "severity_level": self.severity_level,
            "task_type": self.task_type,
            "has_image": self.has_image,
            "scenario_complexity": self.scenario_complexity,
            "personality": self.personality,
            "age_group": self._derive_age_group(),
            "gender_identity": self._derive_gender_identity(),
        }

    def _derive_age_group(self) -> str:
        """Derive age group from patient profile age."""
        try:
            age = int(self.patient_profile.account_info.age_in_years)
        except (ValueError, AttributeError):
            age = None
        return derive_age_group(age)

    def _derive_gender_identity(self) -> str:
        """Derive gender identity category from sex and gender fields."""
        pi = self.patient_profile.personal_info
        return derive_gender_identity(pi.sex, pi.gender)


def load_benchmark_entries(file_path: str) -> List[BenchmarkEntry]:
    """
    Load benchmark entries from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of BenchmarkEntry objects
    """
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [BenchmarkEntry.from_dict(item) for item in data]
