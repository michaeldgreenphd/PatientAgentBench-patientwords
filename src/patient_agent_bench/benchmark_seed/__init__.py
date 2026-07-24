# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Benchmark Seed Generator module for PatientAgentBench.

This module provides programmatic generation of diverse healthcare benchmark cases
with configurable probability distributions. The system operates in two phases:

1. Seed Selection Phase: Selects attribute values (condition, severity, task,
   demographics) from configured distributions.
2. LLM Enrichment Phase: Generates realistic patient stories, queries, and
   profile details using AWS Bedrock.

Exports:
    DistributionConfig: Configuration for attribute distributions loaded from JSON.
    TaskDefinition: A task category with weighted sub-tasks.
    BenchmarkSeed: Initial benchmark seed with selected attributes.
    AttributeSelector: Selects attributes from configured distributions.
    SeedGenerator: Generates benchmark seeds with optional LLM enrichment.
    EnrichedBenchmarkEntry: A fully enriched benchmark entry ready for serialization.
    BenchmarkEntry: Final entry loaded from benchmark JSON files.
    PatientProfile: Patient profile dataclass (re-exported for convenience).
    load_benchmark_entries: Load benchmark entries from JSON file.
"""

from patient_agent_bench.benchmark_seed.config import DistributionConfig, TaskDefinition
from patient_agent_bench.benchmark_seed.selector import AttributeSelector
from patient_agent_bench.benchmark_seed.benchmark_entry import (
    BenchmarkSeed,
    EnrichedBenchmarkEntry,
    BenchmarkEntry,
    PatientProfile,
    load_benchmark_entries,
)
from patient_agent_bench.benchmark_seed.generator import (
    SeedGenerator,
    write_benchmark_entries,
)
from patient_agent_bench.benchmark_seed.analyzer import (
    BenchmarkAnalyzer,
    BenchmarkSummary,
    AttributeDistribution,
    analyze_entries,
    write_summary,
    generate_summary_path,
)

__all__ = [
    "DistributionConfig",
    "TaskDefinition",
    "BenchmarkSeed",
    "AttributeSelector",
    "SeedGenerator",
    "EnrichedBenchmarkEntry",
    "BenchmarkEntry",
    "PatientProfile",
    "load_benchmark_entries",
    "write_benchmark_entries",
    # Analyzer exports
    "BenchmarkAnalyzer",
    "BenchmarkSummary",
    "AttributeDistribution",
    "analyze_entries",
    "write_summary",
    "generate_summary_path",
]
