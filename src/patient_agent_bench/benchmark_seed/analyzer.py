# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Benchmark seed distribution analyzer.

Analyzes generated benchmark entries and produces a summary report
of attribute distributions, comparing actual vs expected distributions.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from patient_agent_bench.benchmark_seed.benchmark_entry import EnrichedBenchmarkEntry

if TYPE_CHECKING:
    from patient_agent_bench.benchmark_seed.benchmark_entry import BenchmarkSeed
from patient_agent_bench.benchmark_seed.config import DistributionConfig
from patient_agent_bench.eval.demographics import derive_gender_identity
from patient_agent_bench.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AttributeDistribution:
    """Distribution statistics for a single attribute type.

    Attributes:
        name: Attribute name (e.g., "conditions", "severity_levels")
        counts: Counter of value occurrences
        total: Total number of entries
        expected_weights: Optional expected weights from config
    """

    name: str
    counts: Counter = field(default_factory=Counter)
    total: int = 0
    expected_weights: Optional[Dict[str, float]] = None

    def add(self, value: str) -> None:
        """Add a value occurrence."""
        self.counts[value] += 1
        self.total += 1

    def get_percentages(self) -> Dict[str, float]:
        """Get percentage distribution."""
        if self.total == 0:
            return {}
        return {k: (v / self.total) * 100 for k, v in self.counts.items()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        percentages = self.get_percentages()
        result: Dict[str, Any] = {
            "total_count": self.total,
            "unique_values": len(self.counts),
            "distribution": {
                k: {"count": self.counts[k], "percentage": round(percentages[k], 2)}
                for k in sorted(self.counts.keys())
            },
        }

        # Add expected vs actual comparison if expected weights available
        if self.expected_weights:
            result["expected_vs_actual"] = self._compare_expected()

        return result

    def _compare_expected(self) -> List[Dict[str, Any]]:
        """Compare actual distribution to expected weights."""
        if not self.expected_weights:
            return []

        percentages = self.get_percentages()
        comparison = []

        # Include all expected values, even if not observed
        all_values = set(self.expected_weights.keys()) | set(self.counts.keys())

        for value in sorted(all_values):
            expected_pct = self.expected_weights.get(value, 0) * 100
            actual_pct = percentages.get(value, 0)
            diff = actual_pct - expected_pct

            comparison.append({
                "value": value,
                "expected_pct": round(expected_pct, 2),
                "actual_pct": round(actual_pct, 2),
                "difference": round(diff, 2),
            })

        return comparison


@dataclass
class BenchmarkSummary:
    """Complete summary of benchmark entry distributions.

    Attributes:
        total_entries: Total number of entries analyzed
        generation_timestamp: When the entries were generated
        distributions: Dictionary of attribute distributions
        metadata: Additional metadata about the generation
    """

    total_entries: int = 0
    generation_timestamp: str = ""
    distributions: Dict[str, AttributeDistribution] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_entries": self.total_entries,
                "generation_timestamp": self.generation_timestamp,
                "analysis_timestamp": datetime.now().isoformat(),
            },
            "metadata": self.metadata,
            "distributions": {
                name: dist.to_dict() for name, dist in self.distributions.items()
            },
        }


class BenchmarkAnalyzer:
    """Analyzes benchmark entries and generates distribution summaries.

    This class processes a list of EnrichedBenchmarkEntry objects and
    produces statistics about the distribution of various attributes.
    """

    def __init__(self, config: Optional[DistributionConfig] = None):
        """Initialize the analyzer.

        Args:
            config: Optional distribution config for expected weight comparison
        """
        self._config = config
        self._expected_weights = self._extract_expected_weights() if config else {}

    def _extract_expected_weights(self) -> Dict[str, Dict[str, float]]:
        """Extract expected weights from config for comparison."""
        if not self._config:
            return {}

        weights: Dict[str, Dict[str, float]] = {}

        # Simple weighted lists
        weights["conditions"] = dict(self._config.conditions)
        weights["severity_levels"] = dict(self._config.severity_levels)
        weights["age_groups"] = dict(self._config.age_groups)
        weights["genders"] = dict(self._config.genders)
        weights["sexes"] = dict(self._config.sexes)
        weights["care_preferences"] = dict(self._config.care_preferences)
        weights["personalities"] = dict(self._config.personalities)

        # Task categories
        weights["task_categories"] = {
            task.category: task.weight for task in self._config.tasks
        }

        # Task types (category_subtask combinations)
        task_types: Dict[str, float] = {}
        for task in self._config.tasks:
            for subtask_name, subtask_weight in task.subtasks:
                task_type = f"{task.category}_{subtask_name}"
                # Combined weight = category_weight * subtask_weight
                task_types[task_type] = task.weight * subtask_weight
        weights["task_types"] = task_types

        return weights

    def analyze(self, entries: List[EnrichedBenchmarkEntry]) -> BenchmarkSummary:
        """Analyze a list of benchmark entries.

        Args:
            entries: List of EnrichedBenchmarkEntry objects to analyze

        Returns:
            BenchmarkSummary with distribution statistics
        """
        summary = BenchmarkSummary(
            total_entries=len(entries),
            generation_timestamp=datetime.now().isoformat(),
        )

        # Initialize distributions
        attr_names = [
            "conditions",
            "severity_levels",
            "task_categories",
            "task_types",
            "age_groups",
            "genders",
            "gender_identities",
            "sexes",
            "care_preferences",
            "scenario_complexity",
            "personalities",
        ]

        for name in attr_names:
            summary.distributions[name] = AttributeDistribution(
                name=name,
                expected_weights=self._expected_weights.get(name),
            )

        # Process each entry
        for entry in entries:
            self._process_entry(entry, summary)

        # Add metadata
        summary.metadata = self._generate_metadata(entries)

        return summary

    def _process_entry(
        self, entry: EnrichedBenchmarkEntry, summary: BenchmarkSummary
    ) -> None:
        """Process a single entry and update distributions."""
        seed = entry.seed

        summary.distributions["conditions"].add(seed.condition_name)
        summary.distributions["severity_levels"].add(seed.severity_level)
        summary.distributions["task_categories"].add(seed.task_category)
        summary.distributions["task_types"].add(entry.task_type)
        summary.distributions["age_groups"].add(seed.age_group)
        summary.distributions["genders"].add(seed.gender or "(not specified)")
        summary.distributions["gender_identities"].add(
            self._derive_gender_identity(seed)
        )
        summary.distributions["sexes"].add(seed.sex)
        summary.distributions["care_preferences"].add(seed.care_preference)
        summary.distributions["scenario_complexity"].add(entry.scenario_complexity)
        summary.distributions["personalities"].add(seed.personality or "unknown")

    @staticmethod
    def _derive_gender_identity(seed: "BenchmarkSeed") -> str:
        """Derive gender identity category from sex and gender."""
        return derive_gender_identity(seed.sex, seed.gender)

    def _generate_metadata(
        self, entries: List[EnrichedBenchmarkEntry]
    ) -> Dict[str, Any]:
        """Generate additional metadata about the entries."""
        if not entries:
            return {}

        # Medication statistics
        med_counts = []
        for entry in entries:
            meds = entry.patient_profile.get("medications", [])
            med_counts.append(len(meds) if isinstance(meds, list) else 0)

        avg_meds = sum(med_counts) / len(med_counts) if med_counts else 0
        min_meds = min(med_counts) if med_counts else 0
        max_meds = max(med_counts) if med_counts else 0

        # Story length statistics
        story_lengths = [len(entry.patient_story) for entry in entries]
        avg_story_len = sum(story_lengths) / len(story_lengths) if story_lengths else 0

        return {
            "medications": {
                "average_count": round(avg_meds, 2),
                "min_count": min_meds,
                "max_count": max_meds,
            },
            "patient_stories": {
                "average_length_chars": round(avg_story_len, 0),
                "min_length_chars": min(story_lengths) if story_lengths else 0,
                "max_length_chars": max(story_lengths) if story_lengths else 0,
            },
        }


def analyze_entries(
    entries: List[EnrichedBenchmarkEntry],
    config: Optional[DistributionConfig] = None,
) -> BenchmarkSummary:
    """Convenience function to analyze benchmark entries.

    Args:
        entries: List of EnrichedBenchmarkEntry objects
        config: Optional distribution config for expected weight comparison

    Returns:
        BenchmarkSummary with distribution statistics
    """
    analyzer = BenchmarkAnalyzer(config)
    return analyzer.analyze(entries)


def write_summary(summary: BenchmarkSummary, path: Path) -> None:
    """Write summary to JSON file.

    Args:
        summary: BenchmarkSummary to write
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info("Wrote distribution summary to %s", path)


def generate_summary_path(entries_path: Path) -> Path:
    """Generate summary file path from entries file path.

    Replaces .json extension with _summary.json.

    Args:
        entries_path: Path to the benchmark entries JSON file

    Returns:
        Path for the summary file
    """
    entries_path = Path(entries_path)
    stem = entries_path.stem
    return entries_path.parent / f"{stem}_summary.json"
