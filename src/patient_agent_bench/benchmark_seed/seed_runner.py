# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Seed runner for benchmark seed generation.

Provides the run_generate_seeds function for the patient-agent-bench CLI.
Generates fully enriched benchmark entries ready for use with the benchmark command.

Supports parallel execution with AWS role pool distribution, similar to
the conversation runner and experiment runner.
"""

import argparse
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from patient_agent_bench.benchmark_seed.config import DistributionConfig
from patient_agent_bench.benchmark_seed.analyzer import (
    analyze_entries,
    generate_summary_path,
    write_summary,
)
from patient_agent_bench.benchmark_seed.generator import (
    EnrichedBenchmarkEntry,
    SeedGenerator,
    write_benchmark_entries,
)
from patient_agent_bench.benchmark_seed.selector import BenchmarkSeed
from patient_agent_bench.config import (
    BenchConfig,
    ModelConfig,
    RolePoolManager,
    create_bedrock_client_with_role,
    create_chat_model,
    ensure_credentials,
)
from patient_agent_bench.logging_config import Colors, get_logger
from patient_agent_bench.patient.generator import generate_enrichment
from patient_agent_bench.runner.parallel import ParallelTaskOrchestrator
from patient_agent_bench.utils.retry import RetryConfig

logger = get_logger(__name__)


def run_generate_seeds(args: argparse.Namespace) -> None:
    """Execute generate-seeds command.

    Generates fully enriched benchmark entries with patient profiles, stories,
    and queries using LLM enrichment. Output is ready for use with the
    benchmark command.

    Uses parallel execution with AWS role pool distribution when max_parallel > 1.

    Args:
        args: Parsed command line arguments
    """
    # Load distribution configuration
    seed_dist_path = Path(args.seed_dist)
    logger.info("Loading seed distribution config from %s", seed_dist_path)

    try:
        dist_config = DistributionConfig.from_file(seed_dist_path)
    except FileNotFoundError as e:
        logger.error("Seed distribution config file not found: %s", e)
        raise SystemExit(1) from e
    except ValueError as e:
        logger.error("Invalid seed distribution config: %s", e)
        raise SystemExit(1) from e

    # Load benchmark configuration for model settings
    bench_config_path = Path(args.config)
    logger.info("Loading benchmark config from %s", bench_config_path)

    try:
        bench_config = BenchConfig.from_file(str(bench_config_path))
    except FileNotFoundError as e:
        logger.error("Benchmark configuration file not found: %s", e)
        raise SystemExit(1) from e
    except ValueError as e:
        logger.error("Invalid benchmark configuration: %s", e)
        raise SystemExit(1) from e

    # Validate AWS credentials for LLM enrichment
    logger.info("Checking AWS credentials...")

    if not ensure_credentials():
        logger.error(
            "Failed to obtain valid AWS credentials. "
            "Please run 'patient-agent-bench auth' or configure credentials."
        )
        raise SystemExit(1)

    logger.info("AWS credentials validated")

    # Load role pool for parallel execution
    role_pool = RolePoolManager.from_env()
    if len(role_pool) > 0:
        logger.info("Loaded %d AWS roles for parallel execution", len(role_pool))
    else:
        logger.info("No role pool configured, using default credentials")

    # Create seed generator (without LLM client - we'll create per-task)
    generator = SeedGenerator(
        config=dist_config,
        llm_client=None,  # Not used for parallel enrichment
        seed=args.seed,
    )

    # Display progress information
    if args.seed is not None:
        logger.info("Using random seed: %d", args.seed)

    # Generate seeds first (no LLM needed)
    logger.info("Generating %d benchmark seeds...", args.count)
    seeds = generator.generate_seeds(args.count)

    # Enrich seeds in parallel
    logger.info(
        "Enriching %d seeds with max_parallel=%d...",
        len(seeds),
        args.max_parallel,
    )

    try:
        entries = asyncio.run(
            _enrich_seeds_parallel(
                seeds=seeds,
                model_config=bench_config.seed_generator_model,
                role_pool=role_pool,
                max_parallel=args.max_parallel,
            )
        )
    except ValueError as e:
        logger.error("Failed to enrich benchmark seeds: %s", e)
        raise SystemExit(1) from e

    # Determine output path
    output_path = args.output
    if output_path is None:
        # Auto-generate filename with timestamp in data/ directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"data/benchmark_{timestamp}.json")

    # Write entries to file
    write_benchmark_entries(entries, output_path)

    # Analyze and write distribution summary
    summary = analyze_entries(entries, dist_config)
    summary_path = generate_summary_path(output_path)
    write_summary(summary, summary_path)

    print(f"\n{Colors.BRIGHT_GREEN}Generated {len(entries)} benchmark entries{Colors.RESET}")
    print(f"{Colors.BRIGHT_GREEN}Output file:{Colors.RESET} {output_path}")
    print(f"{Colors.BRIGHT_GREEN}Summary file:{Colors.RESET} {summary_path}")


async def _enrich_seeds_parallel(
    seeds: List[BenchmarkSeed],
    model_config: ModelConfig,
    role_pool: RolePoolManager,
    max_parallel: int = 1,
) -> List[EnrichedBenchmarkEntry]:
    """Enrich seeds in parallel using role pool distribution.

    Uses ParallelTaskOrchestrator for concurrent execution with
    round-robin role assignment, similar to conversation runner.

    Args:
        seeds: List of BenchmarkSeed objects to enrich
        model_config: Model configuration for LLM enrichment
        role_pool: Pool of AWS ARN roles for load distribution
        max_parallel: Maximum concurrent enrichment tasks

    Returns:
        List of EnrichedBenchmarkEntry objects
    """
    orchestrator = ParallelTaskOrchestrator(
        max_parallel=max_parallel,
        role_pool=role_pool,
        retry_config=RetryConfig(max_retries=3),
    )

    # Build task list
    tasks = [
        (
            f"seed_{i}",
            _enrich_single_seed_async,
            (seed, model_config),
            {},
        )
        for i, seed in enumerate(seeds)
    ]

    # Run all enrichments in parallel
    results = await orchestrator.run_all(tasks)

    # Collect successful results
    entries: List[EnrichedBenchmarkEntry] = []
    failed_count = 0

    for task_result in results:
        if task_result.success and task_result.result is not None:
            entries.append(task_result.result)
        else:
            failed_count += 1
            logger.error(
                "Failed to enrich %s: %s",
                task_result.task_id,
                task_result.error,
            )

    if failed_count > 0:
        logger.warning(
            "Failed to enrich %d/%d seeds",
            failed_count,
            len(seeds),
        )

    return entries


async def _enrich_single_seed_async(
    seed: BenchmarkSeed,
    model_config: ModelConfig,
    assigned_role: Optional[str] = None,
) -> EnrichedBenchmarkEntry:
    """Enrich a single seed asynchronously with role-based credentials.

    Creates an isolated Bedrock client with the assigned role credentials,
    then generates enrichment data using the patient generator.

    Args:
        seed: The benchmark seed to enrich
        model_config: Model configuration for LLM
        assigned_role: AWS ARN role to use (from role pool)

    Returns:
        EnrichedBenchmarkEntry with all fields populated
    """
    # Create bedrock client with assigned role (only needed for Bedrock models)
    bedrock_client = None
    if model_config.requires_bedrock:
        bedrock_client = create_bedrock_client_with_role(assigned_role)

    # Create LLM client using the centralized factory
    llm_client = create_chat_model(model_config, bedrock_client)

    # Generate unique scenario_id
    scenario_id = str(uuid.uuid4())

    # Derive task_type from category and subcategory
    task_type = f"{seed.task_category}_{seed.task_subcategory}"

    # Generate enrichment data using LLM
    enrichment_data = await generate_enrichment(
        seed=seed,
        llm_client=llm_client,
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
