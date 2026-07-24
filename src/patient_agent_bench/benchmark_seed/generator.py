# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Seed generator for benchmark seed generation.

This module provides the SeedGenerator class for generating benchmark seeds
with optional LLM enrichment.

The generator orchestrates:
1. Seed creation using AttributeSelector
2. LLM enrichment using patient generator
3. Output serialization to JSON format compatible with sample_benchmark.json
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from patient_agent_bench.benchmark_seed.config import DistributionConfig
from patient_agent_bench.benchmark_seed.selector import AttributeSelector
from patient_agent_bench.benchmark_seed.benchmark_entry import (
    BenchmarkSeed,
    EnrichedBenchmarkEntry,
)
from patient_agent_bench.logging_config import get_logger
from patient_agent_bench.patient.generator import generate_enrichment

logger = get_logger(__name__)


class SeedGenerator:
    """Generates benchmark seeds with optional LLM enrichment.

    This class orchestrates the generation of benchmark seeds and their
    enrichment with LLM-generated content. It supports:
    - Generating raw seeds without enrichment (generate_seeds)
    - Generating fully enriched entries (generate_enriched)
    - Enriching individual seeds (enrich_seed)

    Attributes:
        _config: Distribution configuration for attribute selection
        _selector: AttributeSelector for creating seeds
        _llm_client: Optional LangChain LLM client for enrichment
    """

    def __init__(
        self,
        config: DistributionConfig,
        llm_client: Optional[Any] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the SeedGenerator.

        Args:
            config: Distribution configuration with attribute weights
            llm_client: Optional LangChain LLM client for enrichment
            seed: Optional random seed for reproducibility
        """
        self._config = config
        self._selector = AttributeSelector(config, seed)
        self._llm_client = llm_client

    def generate_seeds(self, count: int) -> List[BenchmarkSeed]:
        """Generate N benchmark seeds without enrichment.

        Creates the specified number of benchmark seeds by selecting
        attributes from the configured distributions. Each seed is
        independently sampled.

        Args:
            count: Number of seeds to generate

        Returns:
            List of BenchmarkSeed objects
        """
        logger.info("Generating %d benchmark seeds", count)
        seeds = [self._selector.create_seed() for _ in range(count)]
        logger.info("Generated %d seeds successfully", len(seeds))
        return seeds

    async def generate_enriched(self, count: int) -> List[EnrichedBenchmarkEntry]:
        """Generate N fully enriched benchmark entries.

        Creates benchmark seeds and enriches each one with LLM-generated
        content including patient profile, story, and query.

        Args:
            count: Number of enriched entries to generate

        Returns:
            List of EnrichedBenchmarkEntry objects

        Raises:
            ValueError: If LLM client is not configured or enrichment fails
        """
        if self._llm_client is None:
            raise ValueError(
                "LLM client is required for enrichment. "
                "Initialize SeedGenerator with an llm_client parameter."
            )

        logger.info("Generating %d enriched benchmark entries", count)
        seeds = self.generate_seeds(count)

        enriched_entries: List[EnrichedBenchmarkEntry] = []
        for i, seed in enumerate(seeds):
            logger.info(
                "Enriching seed %d/%d: condition=%s, task=%s_%s",
                i + 1,
                count,
                seed.condition_name,
                seed.task_category,
                seed.task_subcategory,
            )
            entry = await self.enrich_seed(seed)
            enriched_entries.append(entry)

        logger.info("Generated %d enriched entries successfully", len(enriched_entries))
        return enriched_entries

    async def enrich_seed(self, seed: BenchmarkSeed) -> EnrichedBenchmarkEntry:
        """Enrich a single seed with LLM-generated content.

        Generates patient profile (with medications), story, and scenario complexity
        in a single LLM call. Also generates a unique scenario_id (UUID) and derives
        task_type by concatenating task_category and task_subcategory.

        Args:
            seed: The benchmark seed to enrich

        Returns:
            EnrichedBenchmarkEntry with all fields populated

        Raises:
            ValueError: If LLM client is not configured or generation fails
        """
        if self._llm_client is None:
            raise ValueError(
                "LLM client is required for enrichment. "
                "Initialize SeedGenerator with an llm_client parameter."
            )

        # Generate unique scenario_id
        scenario_id = str(uuid.uuid4())

        # Derive task_type from category and subcategory
        task_type = f"{seed.task_category}_{seed.task_subcategory}"

        # Generate enrichment data using LLM
        enrichment_data = await generate_enrichment(
            seed=seed,
            llm_client=self._llm_client,
        )

        # Extract generated content
        patient_profile = enrichment_data["patient_profile"]
        patient_story = enrichment_data["patient_story"]
        scenario_complexity = enrichment_data.get("scenario_complexity", "regular")

        # Add empty care_team and pharmacies (generated by sandbox at conversation start)
        patient_profile["care_team"] = {}
        patient_profile["pharmacies"] = {}

        return EnrichedBenchmarkEntry(
            seed=seed,
            scenario_id=scenario_id,
            patient_story=patient_story,
            patient_profile=patient_profile,
            task_type=task_type,
            preferred_care_option=seed.care_preference,
            has_image="False",
            scenario_complexity=scenario_complexity,
        )


def write_benchmark_entries(entries: List[EnrichedBenchmarkEntry], path: Path) -> None:
    """Write enriched entries to JSON file.

    Serializes the list of EnrichedBenchmarkEntry objects to a JSON file
    in the format expected by load_benchmark_entries().

    Args:
        entries: List of enriched benchmark entries to write
        path: Output file path

    Raises:
        PermissionError: If the output path is not writable
        OSError: If there's a disk error during writing
    """
    path = Path(path)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert entries to dictionaries
    data = [entry.to_dict() for entry in entries]

    # Write to file with pretty formatting
    logger.info("Writing %d benchmark entries to %s", len(entries), path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Successfully wrote benchmark entries to %s", path)
