# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
CLI entry point for PatientAgentBench.

Provides commands for:
- auth: Validate AWS credentials (runs the credential refresh hook)
- generate: Run conversations from benchmark data (no evaluation)
- evaluate: Evaluate already created conversations
- benchmark: End-to-end (generate + evaluate)

Pipeline Architecture:
- Each stage operates on batches: input benchmark -> conversations -> evaluations -> summary
- Results are stored in timestamped output directories: {input_file}_{timestamp}/
- Stages can be run independently or as a full pipeline

Auto-authentication:
- Before running any command that requires AWS credentials, the CLI
  validates credentials and invokes the credential refresh hook if needed.
"""

import argparse
import asyncio
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from patient_agent_bench.analyzer.generator import (
    AnalysisGenerator,
    analyze_experiments_parallel,
)
from patient_agent_bench.benchmark_seed.seed_runner import run_generate_seeds
from patient_agent_bench.config import (
    BenchConfig,
    ModelConfig,
    RolePoolManager,
    create_bedrock_client_with_role,
    refresh_credentials_hook,
    check_credentials_valid,
)
from patient_agent_bench.logging_config import setup_logging, get_logger, Colors
from patient_agent_bench.benchmark_seed import load_benchmark_entries
from patient_agent_bench.runner.experiment_config import ExperimentConfig
from patient_agent_bench.runner.conversation_runner import (
    ConversationResult,
)
from patient_agent_bench.runner.conversation import Conversation
from patient_agent_bench.runner.output_manager import OutputManager
from patient_agent_bench.runner.parallel import ParallelTaskOrchestrator
from patient_agent_bench.eval.evaluator import evaluate_async
from patient_agent_bench.eval.aggregator import (
    add_pass_labels,
    calculate_aggregate_score,
    generate_evaluation_summary,
    merge_evaluations,
)
from patient_agent_bench.utils.retry import RetryConfig

logger = get_logger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """Set up the argument parser with all commands and options."""
    parser = argparse.ArgumentParser(
        prog="patient-agent-bench",
        description="PatientAgentBench - Benchmarking framework for health AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark (generate + evaluate) - results saved to output/{cases}_{timestamp}/
  patient-agent-bench benchmark

  # Generate conversations only - saved to output/{cases}_{timestamp}/conversations.json
  patient-agent-bench generate

  # Evaluate existing conversations from a previous run
  patient-agent-bench evaluate --run-dir output/sample_benchmark_20240101_120000

  # Analyze evaluation results to generate insights
  patient-agent-bench analyze --run-dir output/sample_benchmark_20240101_120000

  # Run specific case
  patient-agent-bench benchmark --case-id refill_001

  # Run with limited cases for quick testing
  patient-agent-bench benchmark --num-cases 3

  # Custom output directory
  patient-agent-bench benchmark --output-dir my_results

  # Run with custom model config (single or multi-experiment)
  patient-agent-bench benchmark --config data/default_config.json
        """
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Common arguments for all commands
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Base output directory (default: output)"
    )
    common_parser.add_argument(
        "--config",
        type=str,
        default="data/default_config.json",
        help="Path to benchmark config JSON file (default: data/default_config.json)"
    )
    common_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)"
    )
    common_parser.add_argument(
        "--log-timestamps",
        action="store_true",
        help="Show timestamps and logger names in logs (auto-enabled for DEBUG level)"
    )
    common_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging (sets level to DEBUG)"
    )
    common_parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum parallel tasks for conversation generation and evaluation (default: 1)"
    )

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        parents=[common_parser],
        help="Generate conversations from benchmark data (no evaluation)"
    )
    generate_parser.add_argument(
        "--cases", "-c",
        type=str,
        default="data/sample_benchmark.json",
        help="Path to test cases JSON file (default: data/sample_benchmark.json)"
    )
    generate_parser.add_argument(
        "--num-cases",
        type=int,
        default=0,
        help="Number of cases to run, 0 for all (default: 0)"
    )
    generate_parser.add_argument(
        "--case-id",
        type=str,
        help="Run specific case by ID"
    )
    generate_parser.add_argument(
        "--run-dir",
        type=str,
        help="Resume into an existing run directory (skips already-generated experiments)"
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        parents=[common_parser],
        help="Evaluate already created conversations"
    )
    evaluate_parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to existing run directory containing experiment subdirectories"
    )

    # Benchmark command (generate + evaluate)
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        parents=[common_parser],
        help="Run full benchmark (generate + evaluate)"
    )
    benchmark_parser.add_argument(
        "--cases", "-c",
        type=str,
        default="data/sample_benchmark.json",
        help="Path to test cases JSON file (default: data/sample_benchmark.json)"
    )
    benchmark_parser.add_argument(
        "--num-cases",
        type=int,
        default=0,
        help="Number of cases to run, 0 for all (default: 0)"
    )
    benchmark_parser.add_argument(
        "--case-id",
        type=str,
        help="Run specific case by ID"
    )
    benchmark_parser.add_argument(
        "--run-dir",
        type=str,
        help="Resume into an existing run directory (skips already-generated experiments)"
    )

    # Generate-seeds command (benchmark seed generation)
    seeds_parser = subparsers.add_parser(
        "generate-seeds",
        parents=[common_parser],
        help="Generate benchmark entries with patient profiles and queries",
    )
    seeds_parser.add_argument(
        "--count", "-n",
        type=int,
        required=True,
        help="Number of benchmark entries to generate",
    )
    seeds_parser.add_argument(
        "--seed-dist",
        type=Path,
        default=Path("data/default_benchmark_seed.json"),
        help="Path to seed distribution config file (default: data/default_benchmark_seed.json)",
    )
    seeds_parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path (default: data/benchmark_{timestamp}.json)",
    )
    seeds_parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Analyze command (generate insights from evaluation results)
    analyze_parser = subparsers.add_parser(
        "analyze",
        parents=[common_parser],
        help="Generate insights from evaluation results"
    )
    analyze_parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to existing run directory containing experiment subdirectories"
    )
    analyze_parser.add_argument(
        "--max-samples",
        type=int,
        default=25,
        help="Maximum samples per score level per rubric (default: 25)"
    )

    # Auth command (manual credential refresh)
    auth_parser = subparsers.add_parser(
        "auth",
        help="Validate AWS credentials (and run the credential refresh hook)"
    )
    auth_parser.add_argument(
        "--role",
        type=str,
        help="IAM role name (optional, for custom credential hooks)"
    )
    auth_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)"
    )
    auth_parser.add_argument(
        "--log-timestamps",
        action="store_true",
        help="Show timestamps and logger names in logs (auto-enabled for DEBUG level)"
    )
    auth_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging (sets level to DEBUG)"
    )

    # Review-annotate command
    review_parser = subparsers.add_parser(
        "review-annotate",
        parents=[common_parser],
        help="Sample conversations for human review and annotation",
    )
    review_parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to benchmark run directory",
    )
    review_parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of conversations to sample (default: 20)",
    )
    review_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling",
    )

    # Align-check command
    align_parser = subparsers.add_parser(
        "align-check",
        parents=[common_parser],
        help="Compute inter-rater agreement between LLM and human scores",
    )
    align_parser.add_argument(
        "--annotation-files",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Paths to pre-merged annotation JSON files (AnnotationFile format). "
            "Each file must contain all annotators' scores on the same conversations. "
            "For separate per-annotator downloads, use --annotation-dir instead."
        ),
    )
    align_parser.add_argument(
        "--annotation-dir",
        type=str,
        default=None,
        help=(
            "Directory containing annotations_*.json files from multiple annotators. "
            "Merges all annotations by case_id into a single file before analysis. "
            "Preferred when annotators downloaded separate JSON files from the review UI."
        ),
    )
    align_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save alignment report JSON",
    )

    return parser


def load_config(args: argparse.Namespace) -> BenchConfig:
    """Load configuration from file or use defaults."""
    if args.config:
        logger.info("Loading config from %s", args.config)
        return BenchConfig.from_file(args.config)
    return BenchConfig.default()


def filter_entries(
    entries: List[Any],
    case_id: Optional[str] = None,
    num_cases: int = 0
) -> List[Any]:
    """Filter benchmark entries based on case_id or num_cases."""
    if case_id:
        entries = [e for e in entries if e.id == case_id]
        if not entries:
            logger.warning("No entry found with case_id: %s", case_id)
    elif num_cases > 0:
        entries = entries[:num_cases]
    return entries


def _evaluate_conversations_parallel(
    model_config: ModelConfig,
    conversations: List[ConversationResult],
    max_parallel: int,
    role_pool: RolePoolManager,
) -> List[Dict[str, Any]]:
    """
    Evaluate conversations in parallel using asyncio.

    Args:
        model_config: Model configuration for the evaluator
        conversations: List of ConversationResult objects
        max_parallel: Maximum concurrent tasks
        role_pool: Pool of AWS ARN roles for distribution

    Returns:
        List of evaluation result dictionaries
    """
    async def _run_parallel():
        orchestrator = ParallelTaskOrchestrator(
            max_parallel=max_parallel,
            role_pool=role_pool,
            retry_config=RetryConfig(),
        )

        # Build task list - skip conversations that already have errors
        tasks: List[tuple] = []
        for conv in conversations:
            if conv.error is None:
                tasks.append((
                    conv.case_id,
                    evaluate_async,
                    (
                        model_config,
                        conv.conversation,
                        conv.user_profile,
                        conv.scenario,
                    ),
                    {},
                ))

        # Run all evaluations in parallel
        results = await orchestrator.run_all(tasks)

        # Build result map for easy lookup
        result_map = {r.task_id: r for r in results}

        # Merge evaluation results with conversation data
        # Serialize to dicts here at the boundary before file write
        evaluations = []
        for conv in conversations:
            conv_dict = conv.to_dict()

            if conv.error is not None:
                # Conversation already had an error, propagate it
                evaluations.append({
                    **conv_dict,
                    "evaluation": {"error": conv.error},
                })
            elif conv.case_id in result_map and result_map[conv.case_id].success:
                # Successful evaluation
                evaluations.append({
                    **conv_dict,
                    "evaluation": result_map[conv.case_id].result,
                })
            else:
                # Evaluation failed
                error = result_map.get(conv.case_id)
                error_msg = error.error if error else "Unknown error"
                evaluations.append({
                    **conv_dict,
                    "evaluation": {"error": error_msg},
                })

        return evaluations

    return asyncio.run(_run_parallel())


def _evaluate_multi_evaluator_parallel(
    evaluator_models: List,
    conversations: List[ConversationResult],
    max_parallel: int,
    role_pool: RolePoolManager,
    aggregation_method: str = "average",
    output_mgr: Optional[OutputManager] = None,
    experiment_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate conversations with multiple evaluator models and merge results.

    For each evaluator model, runs all conversations, adds pass labels,
    saves results incrementally, and merges using the configured aggregation.
    Skips evaluators whose results already exist on disk.

    Args:
        evaluator_models: List of ModelConfig for evaluator models
        conversations: List of ConversationResult objects
        max_parallel: Maximum concurrent tasks
        role_pool: Pool of AWS ARN roles for distribution
        aggregation_method: "average" or "majority_vote"
        output_mgr: OutputManager for incremental saving (optional)
        experiment_id: Experiment ID for incremental saving (optional)

    Returns:
        List of evaluation result dicts with evaluation_0, ..., evaluation_N-1,
        and merged evaluation key.
    """
    num_conversations = len(conversations)

    # Pre-initialize evaluations file if output manager available
    if output_mgr is not None and experiment_id is not None:
        output_mgr.init_evaluations_file(experiment_id, num_conversations)

    # Run evaluation for each evaluator model
    per_evaluator_results: List[List[Dict[str, Any]]] = []
    for e_idx, evaluator_model in enumerate(evaluator_models):
        # Load partial results for this evaluator (skip completed slots)
        existing: List[Optional[Dict[str, Any]]] = [None] * num_conversations
        if output_mgr is not None and experiment_id is not None:
            existing = output_mgr.load_partial_evaluator_results(
                experiment_id, e_idx, num_conversations
            )
            completed = sum(1 for r in existing if r is not None)
            if completed == num_conversations:
                logger.info(
                    "Skipping evaluator %d/%d (%s)"
                    " — all %d results already exist",
                    e_idx + 1,
                    len(evaluator_models),
                    evaluator_model.model or evaluator_model.model_id,
                    num_conversations,
                )
            elif completed > 0:
                logger.info(
                    "Resuming evaluator %d/%d (%s)"
                    " — %d/%d already complete",
                    e_idx + 1,
                    len(evaluator_models),
                    evaluator_model.model or evaluator_model.model_id,
                    completed,
                    num_conversations,
                )

        # Build list of pending conversations
        pending_convs = [
            conv for idx, conv in enumerate(conversations)
            if existing[idx] is None
        ]
        pending_indices = [
            idx for idx in range(num_conversations)
            if existing[idx] is None
        ]

        if pending_convs:
            logger.info(
                "Running evaluator %d/%d (%s) — %d/%d pending",
                e_idx + 1,
                len(evaluator_models),
                evaluator_model.model or evaluator_model.model_id,
                len(pending_convs),
                num_conversations,
            )
            results = _evaluate_conversations_parallel(
                evaluator_model, pending_convs, max_parallel, role_pool
            )

            # Map results back by case_id
            result_by_case = {r.get("case_id"): r for r in results}

            for conv_idx in pending_indices:
                conv = conversations[conv_idx]
                r = result_by_case.get(conv.case_id, {})
                existing[conv_idx] = r.get("evaluation", {})

                # Save incrementally per conversation
                if output_mgr is not None and experiment_id is not None:
                    output_mgr.save_evaluation_at_index(
                        experiment_id, e_idx, conv_idx, existing[conv_idx]
                    )

        # Re-run post-processing on all results so that any weight or
        # threshold changes take effect. The three functions are
        # idempotent (they overwrite the same keys).
        for idx, ev in enumerate(existing):
            if ev is not None and "error" not in ev:
                ev = add_pass_labels(ev)
                ev = calculate_aggregate_score(ev)
                ev = generate_evaluation_summary(ev)
                existing[idx] = ev

        per_evaluator_results.append(existing)  # type: ignore[arg-type]

    # Build final evaluation dicts with individual + merged results
    evaluations = []
    for conv_idx, conv in enumerate(conversations):
        conv_dict = conv.to_dict()

        # Collect individual evaluator results for this conversation
        individual_evals = [
            per_evaluator_results[e_idx][conv_idx]
            for e_idx in range(len(evaluator_models))
        ]

        # Add individual evaluator results
        for e_idx, eval_result in enumerate(individual_evals):
            conv_dict[f"evaluation_{e_idx}"] = eval_result

        # Merge evaluations
        merged = merge_evaluations(individual_evals, method=aggregation_method)
        conv_dict["evaluation"] = merged

        evaluations.append(conv_dict)

    return evaluations


def _output_mgr_from_args(args: argparse.Namespace) -> OutputManager:
    """
    Create an OutputManager from CLI args.

    If --run-dir is provided, reuses the existing directory.
    Otherwise creates a new timestamped directory from --cases.
    """
    run_dir = getattr(args, "run_dir", None)
    if run_dir:
        run_path = Path(run_dir)
        if not run_path.exists():
            logger.error("Run directory not found: %s", run_path)
            sys.exit(1)
        # Extract timestamp from directory name
        dir_name = run_path.name
        parts = dir_name.rsplit("_", 2)
        if len(parts) >= 3:
            input_stem = "_".join(parts[:-2])
            timestamp = f"{parts[-2]}_{parts[-1]}"
        else:
            input_stem = dir_name
            timestamp = dir_name
        return OutputManager(
            input_file=f"{input_stem}.json",
            base_output_dir=str(run_path.parent),
            timestamp=timestamp,
        )
    output_mgr = OutputManager(args.cases, args.output_dir)
    output_mgr.setup()
    return output_mgr


def _exit_if_no_evaluations(summary: Dict[str, Any]) -> None:
    """Fail loudly when a benchmark/evaluate run produced zero real evaluations.

    Every conversation failing (bad credentials, no model access, wrong region,
    etc.) is caught per-task and recorded as an error row, so the pipeline still
    returns a well-formed summary with an all-zeros comparison table. Without
    this guard the CLI would print a green summary and exit 0, masking a run in
    which *nothing* actually succeeded. Only a TOTAL failure exits non-zero;
    partial failures are surfaced as a warning but still exit 0.
    """
    comparison = summary.get("comparison_table", [])
    total_cases = sum(row.get("num_cases", 0) for row in comparison)
    evaluated_cases = sum(row.get("evaluated_cases", 0) for row in comparison)

    if total_cases > 0 and evaluated_cases == 0:
        logger.error(
            "Run FAILED: 0 of %d conversation(s) produced a valid evaluation. "
            "The all-zeros summary above does not reflect real results. Check "
            "the ERROR lines above (common causes: expired/invalid AWS "
            "credentials, missing Bedrock model access, wrong region, or an "
            "invalid API key). Re-run with -v for details.",
            total_cases,
        )
        sys.exit(1)

    if evaluated_cases < total_cases:
        logger.warning(
            "%d of %d conversation(s) failed to evaluate; scores reflect only "
            "the %d that succeeded.",
            total_cases - evaluated_cases,
            total_cases,
            evaluated_cases,
        )


def cmd_generate(args: argparse.Namespace) -> None:
    """
    Handle the generate command.

    Generates conversations for all assistant+user model combinations.
    Saves conversations to experiment subdirectories (skips evaluation).
    Uses ExperimentRunner for incremental save, null placeholders, and resume.
    """
    from patient_agent_bench.runner.experiment_runner import ExperimentRunner

    logger.info("Loading test cases from %s", args.cases)
    entries = load_benchmark_entries(args.cases)
    entries = filter_entries(
        entries, getattr(args, 'case_id', None), getattr(args, 'num_cases', 0)
    )

    if not entries:
        logger.error("No benchmark entries to process")
        sys.exit(1)

    # Setup output manager
    output_mgr = _output_mgr_from_args(args)

    config = load_config(args)

    # Get parallel execution settings
    max_parallel = getattr(args, 'max_parallel', 1)
    role_pool = RolePoolManager.from_env()

    if max_parallel > 1:
        logger.info(
            "Parallel execution enabled: max_parallel=%d, role_pool_size=%d",
            max_parallel,
            len(role_pool),
        )

    # Save run config with CLI params at start
    cli_params = {
        "cases_file": args.cases,
        "num_cases": getattr(args, 'num_cases', 0),
        "case_id": getattr(args, 'case_id', None),
        "max_turns": config.max_turns,
        "max_parallel": max_parallel,
    }
    output_mgr.save_run_config(config, cli_params)
    output_mgr.save_benchmark_cases(args.cases)

    num_experiments = len(config.assistant_agents) * len(config.user_agents)
    logger.info(
        "Generating conversations for %d case(s) with max %d turns, %d experiment(s)",
        len(entries),
        config.max_turns,
        num_experiments,
    )

    experiment_runner = ExperimentRunner(
        config=config,
        output_manager=output_mgr,
        max_parallel=max_parallel,
        role_pool=role_pool,
        generate_only=True,
    )
    results = experiment_runner.run_all(entries)

    print(f"\n{Colors.BRIGHT_GREEN}Generated conversations for {num_experiments} "
          f"model combination(s){Colors.RESET}")
    print(f"{Colors.BRIGHT_GREEN}Run directory:{Colors.RESET} {output_mgr.run_dir}")
    print(f"{Colors.BRIGHT_CYAN}  - run_config.json{Colors.RESET}")
    if num_experiments > 0:
        print(f"{Colors.BRIGHT_CYAN}  - Per-combination subdirectories "
              f"(0_0, 1_0, etc.){Colors.RESET}")

    # Fail loudly if every conversation failed to generate (silent-success
    # guard). In generate-only mode there is no evaluation, so count the
    # conversations that were produced without an error.
    total_convs = 0
    ok_convs = 0
    for exp_result in results.get("experiments", {}).values():
        if "error" in exp_result:
            continue
        for conv in exp_result.get("conversations", []):
            total_convs += 1
            if not conv.get("error"):
                ok_convs += 1
    if total_convs > 0 and ok_convs == 0:
        logger.error(
            "Generation FAILED: 0 of %d conversation(s) generated successfully. "
            "Check the ERROR lines above (common causes: expired/invalid AWS "
            "credentials, missing Bedrock model access, wrong region, or an "
            "invalid API key). Re-run with -v for details.",
            total_convs,
        )
        sys.exit(1)
    if ok_convs < total_convs:
        logger.warning(
            "%d of %d conversation(s) failed to generate.",
            total_convs - ok_convs,
            total_convs,
        )


def cmd_evaluate(args: argparse.Namespace) -> None:
    """
    Handle the evaluate command.

    Evaluates conversations from a previous run directory.
    - Loads experiment configs from experiment subdirectories
    - Runs evaluation with specified evaluator models (from --config or run_config)
    - Saves evaluations to experiment subdirectories
    - Generates cross-experiment summary
    - Supports parallel execution with --max-parallel flag
    """
    if not args.run_dir:
        logger.error("--run-dir must be provided")
        sys.exit(1)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        sys.exit(1)

    # Use evaluator models from --config if provided
    config = load_config(args)

    # Get parallel execution settings
    max_parallel = getattr(args, 'max_parallel', 1)
    role_pool = RolePoolManager.from_env()

    if max_parallel > 1:
        logger.info(
            "Parallel execution enabled: max_parallel=%d, role_pool_size=%d",
            max_parallel,
            len(role_pool),
        )

    # Extract timestamp from run_dir name
    dir_name = run_dir.name
    parts = dir_name.rsplit("_", 2)
    if len(parts) >= 3:
        input_stem = "_".join(parts[:-2])
        timestamp = f"{parts[-2]}_{parts[-1]}"
    else:
        input_stem = dir_name
        timestamp = None

    output_mgr = OutputManager(
        input_file=f"{input_stem}.json",
        base_output_dir=str(run_dir.parent),
        timestamp=timestamp,
    )

    # Save run config for reproducibility
    cli_params = {
        "run_dir": args.run_dir,
        "max_parallel": max_parallel,
    }
    output_mgr.save_run_config(config, cli_params)

    # Find experiment subdirectories
    exp_subdirs = sorted([
        d for d in run_dir.iterdir()
        if d.is_dir() and _is_experiment_dir(d.name)
    ])

    if not exp_subdirs:
        logger.error("No experiment subdirectories found in %s", run_dir)
        sys.exit(1)

    # Load existing experiment configs using ExperimentConfig.from_dict()
    source_experiments: Dict[str, ExperimentConfig] = {}
    for exp_dir in exp_subdirs:
        config_file = exp_dir / "experiment_config.json"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                exp_data = json.load(f)
                source_experiments[exp_dir.name] = ExperimentConfig.from_dict(exp_data)

    # Get unique assistant+user combinations from existing experiments
    unique_combinations = {}
    for exp_id, exp in source_experiments.items():
        key = (exp.assistant_idx, exp.user_idx)
        if key not in unique_combinations:
            unique_combinations[key] = exp_id

    logger.info(
        "Found %d experiment(s), evaluating with %d evaluator model(s)",
        len(unique_combinations),
        len(config.evaluator_models),
    )

    all_results: Dict[str, Dict[str, Any]] = {}
    all_experiments: List[ExperimentConfig] = []

    for (a_idx, u_idx), source_exp_id in unique_combinations.items():
        new_exp_id = f"{a_idx}_{u_idx}"
        source_exp = source_experiments.get(source_exp_id)
        if not source_exp:
            continue

        # Load conversations from source experiment
        source_dir = run_dir / source_exp_id
        conversations_file = source_dir / "conversations.json"
        if not conversations_file.exists():
            logger.warning(
                "conversations.json not found in %s, skipping", source_dir
            )
            continue

        with open(conversations_file, "r", encoding="utf-8") as f:
            conversations_json = json.load(f)

        # Convert JSON dicts to ConversationResult objects
        conversation_results: List[ConversationResult] = []
        for conv_dict in conversations_json:
            conversation_results.append(ConversationResult(
                case_id=conv_dict.get("case_id", ""),
                conversation=Conversation.from_dicts(
                    conv_dict.get("conversation", [])
                ),
                user_profile=conv_dict.get("user_profile", ""),
                scenario=conv_dict.get("scenario", ""),
                num_turns=conv_dict.get("num_turns", 0),
                error=conv_dict.get("error"),
            ))

        # Create new experiment config with all evaluator models
        experiment = ExperimentConfig(
            experiment_id=new_exp_id,
            assistant_agent=source_exp.assistant_agent,
            user_agent=source_exp.user_agent,
            evaluator_models=list(config.evaluator_models),
            sandbox_model=source_exp.sandbox_model,
            analyzer_model=source_exp.analyzer_model,
            assistant_idx=a_idx,
            user_idx=u_idx,
            aggregation_method=config.aggregation_method,
            max_turns=source_exp.max_turns,
            strip_thinking_content=source_exp.strip_thinking_content,
        )
        all_experiments.append(experiment)

        logger.info(
            "Evaluating experiment %s with %d evaluator(s)",
            new_exp_id,
            len(config.evaluator_models),
        )

        # Run multi-evaluator evaluation with aggregation
        evaluations = _evaluate_multi_evaluator_parallel(
            config.evaluator_models,
            conversation_results,
            max_parallel,
            role_pool,
            config.aggregation_method,
            output_mgr=output_mgr,
            experiment_id=new_exp_id,
        )

        # Save results to experiment subdirectory
        # Load case metadata from benchmark file if available
        case_metadata = _load_case_metadata_from_run_dir(run_dir)
        output_mgr.save_experiment_results(
            experiment, conversations_json, evaluations, case_metadata
        )

        all_results[new_exp_id] = {
            "conversations": conversations_json,
            "evaluations": evaluations,
            "experiment_config": experiment.to_dict(),
        }

    # Generate and save cross-experiment summary
    summary = output_mgr.generate_experiments_summary(all_results, all_experiments)
    output_mgr.print_experiments_summary_table(summary)

    print(f"\n{Colors.BRIGHT_GREEN}Evaluations saved to:{Colors.RESET} {output_mgr.run_dir}")
    print(f"{Colors.BRIGHT_CYAN}  - experiments_summary.json{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}  - Per-experiment subdirectories{Colors.RESET}")

    # Fail loudly if every conversation failed to evaluate (silent-success guard).
    _exit_if_no_evaluations(summary)


def _is_experiment_dir(name: str) -> bool:
    """Check if directory name matches experiment ID format (X_Y or legacy X_Y_Z)."""
    parts = name.split("_")
    if len(parts) not in (2, 3):
        return False
    return all(part.isdigit() for part in parts)


def _load_case_metadata_from_run_dir(
    run_dir: Path,
) -> Optional[Dict[str, Dict[str, str]]]:
    """
    Load case metadata from benchmark_cases.json in the run directory.

    This file is copied into the run directory during benchmark/generate
    so that evaluate can access it without depending on the original path.

    Falls back to cli_params.cases_file from run_config.json for
    backward compatibility with older run directories.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Dict mapping case_id to metadata dict, or None if unavailable.
    """
    # Primary: read from local copy
    local_cases = run_dir / "benchmark_cases.json"
    if local_cases.exists():
        try:
            entries = load_benchmark_entries(str(local_cases))
            return {entry.id: entry.metadata for entry in entries}
        except Exception:  # noqa: BLE001
            logger.debug("Could not load benchmark_cases.json from %s", run_dir)

    # Fallback: resolve from run_config.json (backward compat)
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            run_config = json.load(f)

        cases_file = run_config.get("cli_params", {}).get("cases_file")
        if not cases_file:
            return None

        cases_path = Path(cases_file)
        if not cases_path.exists():
            return None

        entries = load_benchmark_entries(str(cases_path))
        return {entry.id: entry.metadata for entry in entries}
    except Exception:  # noqa: BLE001
        logger.debug(
            "Could not load case metadata from %s", config_path
        )
        return None


def cmd_benchmark(args: argparse.Namespace) -> None:
    """
    Handle the benchmark command (generate + evaluate).

    Full pipeline using ExperimentRunner:
    - Generates all experiment combinations (Cartesian product of model configs)
    - Runs conversations with reuse optimization
    - Evaluates all conversations
    - Saves per-experiment results and cross-experiment summary
    """
    from patient_agent_bench.runner.experiment_runner import ExperimentRunner

    logger.info("Loading test cases from %s", args.cases)
    entries = load_benchmark_entries(args.cases)
    entries = filter_entries(
        entries, getattr(args, 'case_id', None), getattr(args, 'num_cases', 0)
    )

    if not entries:
        logger.error("No benchmark entries to process")
        sys.exit(1)

    # Setup output manager
    output_mgr = _output_mgr_from_args(args)

    config = load_config(args)

    # Save run config with CLI params at start
    cli_params = {
        "cases_file": args.cases,
        "num_cases": getattr(args, 'num_cases', 0),
        "case_id": getattr(args, 'case_id', None),
        "max_turns": config.max_turns,
    }
    output_mgr.save_run_config(config, cli_params)
    output_mgr.save_benchmark_cases(args.cases)

    # Calculate total experiments for logging
    num_experiments = (
        len(config.assistant_agents)
        * len(config.user_agents)
    )
    logger.info(
        "Running benchmark with %d case(s), max %d turns, %d experiment(s)",
        len(entries),
        config.max_turns,
        num_experiments,
    )

    # Create ExperimentRunner and run all experiments
    role_pool = RolePoolManager.from_env()
    max_parallel = getattr(args, 'max_parallel', 1)

    if max_parallel > 1:
        logger.info(
            "Parallel execution enabled: max_parallel=%d, role_pool_size=%d",
            max_parallel,
            len(role_pool),
        )

    experiment_runner = ExperimentRunner(
        config=config,
        output_manager=output_mgr,
        max_parallel=max_parallel,
        role_pool=role_pool,
    )
    results = experiment_runner.run_all(entries)

    # Generate and save cross-experiment summary
    experiments = experiment_runner.generate_experiments()
    summary = output_mgr.generate_experiments_summary(
        results.get("experiments", {}), experiments
    )

    # Print summary
    output_mgr.print_experiments_summary_table(summary)

    print(f"\n{Colors.BRIGHT_GREEN}All results saved to:{Colors.RESET} {output_mgr.run_dir}")
    print(f"{Colors.BRIGHT_CYAN}  - run_config.json{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}  - experiments_summary.json{Colors.RESET}")
    if num_experiments > 0:
        print(f"{Colors.BRIGHT_CYAN}  - Per-experiment subdirectories (0_0, etc.){Colors.RESET}")

    # Fail loudly if every conversation failed (silent-success guard).
    _exit_if_no_evaluations(summary)


def cmd_auth(args: argparse.Namespace) -> None:
    """Handle the auth command (validate / refresh AWS credentials)."""
    logger.info("Checking AWS credentials...")

    # Give the (overridable) credential hook a chance to refresh first, then
    # validate whatever credentials are now available via the standard chain.
    refresh_credentials_hook()

    if check_credentials_valid():
        logger.info("✓ AWS credentials are valid")
        print(f"\n{Colors.BRIGHT_GREEN}AWS credentials are valid{Colors.RESET}")
    else:
        logger.error(
            "✗ No valid AWS credentials found. Configure credentials via "
            "environment variables, 'aws configure', or an instance role."
        )
        sys.exit(1)


def cmd_analyze(args: argparse.Namespace) -> None:
    """
    Handle the analyze command.

    Generates insights from evaluation results by clustering explanations
    and identifying patterns across experiments.

    Supports parallel execution with AWS role pool distribution when
    max_parallel > 1, similar to conversation runner and seed generator.
    """
    if not args.run_dir:
        logger.error("--run-dir must be provided")
        sys.exit(1)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        sys.exit(1)

    # Load run config to get analyzer_model
    run_config_file = run_dir / "run_config.json"
    if not run_config_file.exists():
        logger.error("run_config.json not found in %s", run_dir)
        sys.exit(1)

    config = BenchConfig.from_file(str(run_config_file))

    max_samples = getattr(args, 'max_samples', 25)
    max_parallel = getattr(args, 'max_parallel', 1)

    # Find experiment subdirectories
    exp_subdirs = sorted([
        d for d in run_dir.iterdir()
        if d.is_dir() and _is_experiment_dir(d.name)
    ])

    if not exp_subdirs:
        logger.error("No experiment subdirectories found in %s", run_dir)
        sys.exit(1)

    logger.info(
        "Analyzing %d experiment(s) with max_samples=%d, max_parallel=%d",
        len(exp_subdirs),
        max_samples,
        max_parallel,
    )

    # Load role pool for parallel execution
    role_pool = RolePoolManager.from_env()
    if len(role_pool) > 0:
        logger.info("Loaded %d AWS roles for parallel execution", len(role_pool))
    else:
        logger.info("No role pool configured, using default credentials")

    # Collect experiment data for parallel processing
    experiment_data: List[Dict[str, Any]] = []

    for exp_dir in exp_subdirs:
        exp_id = exp_dir.name

        # Load evaluations.json
        evaluations_file = exp_dir / "evaluations.json"
        if not evaluations_file.exists():
            logger.warning("evaluations.json not found in %s, skipping", exp_dir)
            continue

        # Load summary.json
        summary_file = exp_dir / "summary.json"
        if not summary_file.exists():
            logger.warning("summary.json not found in %s, skipping", exp_dir)
            continue

        # Load experiment_config.json to get assistant model name
        config_file = exp_dir / "experiment_config.json"
        assistant_model = "unknown"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                exp_config = json.load(f)
                # Support new format (assistant_agent) and legacy (assistant_model)
                assistant_cfg = exp_config.get("assistant_agent", exp_config.get("assistant_model", {}))
                if "model" in assistant_cfg and isinstance(assistant_cfg["model"], dict):
                    # New AgentSpec format: {"model": {"model": "..."}, ...}
                    assistant_model = assistant_cfg["model"].get("model") or assistant_cfg["model"].get("model_id", "unknown")
                else:
                    # Legacy flat format: {"model": "...", "model_id": "..."}
                    assistant_model = assistant_cfg.get("model") or assistant_cfg.get("model_id", "unknown")

        with open(evaluations_file, "r", encoding="utf-8") as f:
            evaluations = json.load(f)

        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

        experiment_data.append({
            "exp_id": exp_id,
            "exp_dir": exp_dir,
            "assistant_model": assistant_model,
            "evaluations": evaluations,
            "summary": summary,
        })

    if not experiment_data:
        logger.error("No valid experiments found to analyze")
        sys.exit(1)

    # Check for existing analysis.md files (skip logic)
    experiments_to_analyze = []
    experiment_analyses: Dict[str, str] = {}

    for exp in experiment_data:
        analysis_file = exp["exp_dir"] / "analysis.md"
        if analysis_file.exists():
            logger.info(
                "Skipping analysis for %s (analysis.md exists)",
                exp["exp_id"],
            )
            experiment_analyses[exp["exp_id"]] = analysis_file.read_text(
                encoding="utf-8"
            )
        else:
            experiments_to_analyze.append(exp)

    # Run parallel analysis for experiments that need it
    if experiments_to_analyze:
        new_analyses = asyncio.run(
            analyze_experiments_parallel(
                experiment_data=experiments_to_analyze,
                model_config=config.analyzer_model,
                role_pool=role_pool,
                max_parallel=max_parallel,
                max_samples=max_samples,
            )
        )
        experiment_analyses.update(new_analyses)

    # Generate cross-experiment summary
    if len(experiment_analyses) > 0:
        # Load experiments_summary.json
        experiments_summary_file = run_dir / "experiments_summary.json"
        if experiments_summary_file.exists():
            with open(experiments_summary_file, "r", encoding="utf-8") as f:
                experiments_summary = json.load(f)

            logger.info("Generating cross-experiment analysis summary")

            # Create analyzer for summary (uses first role or default)
            summary_role = role_pool.get_role() if len(role_pool) > 0 else None

            analyzer = AnalysisGenerator(
                model_config=config.analyzer_model,
                max_samples_per_score=max_samples,
                role_arn=summary_role,
            )

            summary_md = analyzer.generate_analysis_summary(
                run_dir=str(run_dir),
                experiments_summary=experiments_summary,
                experiment_analyses=experiment_analyses,
                experiment_summaries={
                    exp["exp_id"]: exp["summary"]
                    for exp in experiment_data
                },
            )

            # Save analysis_summary.md to run directory
            summary_file = run_dir / "analysis_summary.md"
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(summary_md)

            logger.info("Saved analysis summary to %s", summary_file)
        else:
            logger.warning(
                "experiments_summary.json not found, skipping cross-experiment summary"
            )

    # Fail loudly if no experiment produced an analysis (silent-success guard).
    # analyze_experiments_parallel swallows per-experiment analyzer failures and
    # only logs them, so without this a run where every analysis failed would
    # still print "Analysis complete" and exit 0.
    if not experiment_analyses:
        logger.error(
            "Analysis FAILED: 0 of %d experiment(s) produced an analysis. "
            "Check the ERROR lines above (common causes: expired/invalid AWS "
            "credentials, missing Bedrock model access, or wrong region). "
            "Re-run with -v for details.",
            len(experiment_data),
        )
        sys.exit(1)

    print(f"\n{Colors.BRIGHT_GREEN}Analysis complete{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}  - Per-experiment: analysis.md in each experiment directory{Colors.RESET}")
    if len(experiment_analyses) > 0:
        print(f"{Colors.BRIGHT_CYAN}  - Summary: analysis_summary.md{Colors.RESET}")


def cmd_review_annotate(args: argparse.Namespace) -> None:
    """Sample conversations for human review and generate annotation HTML."""
    from datetime import datetime as dt

    from patient_agent_bench.review.html_generator import HTMLGenerator
    from patient_agent_bench.review.models import (
        AnnotationFile,
        AnnotationMetadata,
    )
    from patient_agent_bench.review.sampler import ConversationSampler

    run_dir = Path(args.run_dir)
    num_samples = args.num_samples
    seed = args.seed

    # Determine output directory
    # common_parser provides --output-dir with default "output"
    # For review-annotate, default to review/ inside run-dir
    output_dir_arg = getattr(args, "output_dir", None)
    if output_dir_arg and output_dir_arg != "output":
        out_dir = Path(output_dir_arg)
    else:
        out_dir = run_dir / "review"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Sampling %d conversations from %s", num_samples, run_dir)
    sampler = ConversationSampler(run_dir, seed=seed)
    conversations = sampler.sample(num_samples)

    ts = dt.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"annotations_{ts}.json"
    html_filename = f"review_{ts}.html"

    # Build annotation file
    annotation_file = AnnotationFile(
        metadata=AnnotationMetadata(
            source_run_dir=str(run_dir),
            sampling_timestamp=ts,
            sample_count=len(conversations),
            random_seed=seed,
            experiments_sampled=list(sampler.experiment_ids),
        ),
        conversations=conversations,
    )

    json_path = out_dir / json_filename
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(annotation_file.model_dump_json(indent=2))

    # Generate HTML
    gen = HTMLGenerator()
    html_content = gen.generate(conversations, json_filename)
    html_path = out_dir / html_filename
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n{Colors.BRIGHT_GREEN}Review files generated{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}  JSON: {json_path}{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}  HTML: {html_path}{Colors.RESET}")


def cmd_align_check(args: argparse.Namespace) -> None:
    """Compute inter-rater agreement from annotation files."""
    from patient_agent_bench.review.alignment import AlignmentAnalyzer, merge_annotation_dir

    if args.annotation_dir:
        dir_path = Path(args.annotation_dir)
        if not dir_path.is_dir():
            print(f"{Colors.RED}Not a directory: {dir_path}{Colors.RESET}")
            return
        print(f"{Colors.BRIGHT_CYAN}Merging annotations from {dir_path}...{Colors.RESET}")
        merged_path = merge_annotation_dir(dir_path)
        print(f"{Colors.BRIGHT_GREEN}Merged file: {merged_path}{Colors.RESET}")
        annotation_paths = [merged_path]
    elif args.annotation_files:
        annotation_paths = [Path(p) for p in args.annotation_files]
    else:
        print(f"{Colors.RED}Either --annotation-files or --annotation-dir is required.{Colors.RESET}")
        return
    analyzer = AlignmentAnalyzer(annotation_paths)
    report = analyzer.compute_agreement()

    print(AlignmentAnalyzer.format_report(report))

    if args.output:
        out_path = Path(args.output)
        AlignmentAnalyzer.save_report(report, out_path)
        print(
            f"{Colors.BRIGHT_CYAN}Report saved to "
            f"{out_path}{Colors.RESET}"
        )

    # Derive prefix from first annotation file name
    first_ann = annotation_paths[0]
    prefix = first_ann.stem  # e.g. "annotations_20260312_130106_prefilled"
    chart_dir = first_ann.parent

    # Generate alignment charts
    chart_path = chart_dir / f"{prefix}_alignment.png"
    try:
        analyzer.generate_alignment_chart(chart_path)
        print(
            f"{Colors.BRIGHT_CYAN}Alignment charts saved to "
            f"{chart_dir}/{Colors.RESET}"
        )
    except Exception as e:
        logger.warning("Could not generate alignment charts: %s", e)

    # Save report as markdown
    md_path = chart_dir / f"{prefix}_alignment_report.md"
    report_text = AlignmentAnalyzer.format_report(report)
    md_path.write_text(f"```\n{report_text}\n```\n", encoding="utf-8")
    print(
        f"{Colors.BRIGHT_CYAN}Report saved to "
        f"{md_path}{Colors.RESET}"
    )


def main() -> None:
    """Main entry point for the CLI."""
    parser = setup_parser()
    args = parser.parse_args()

    # Determine log level
    if getattr(args, 'verbose', False):
        log_level = "DEBUG"
    else:
        log_level = getattr(args, 'log_level', 'INFO')

    # Determine if timestamps should be shown
    show_timestamps = getattr(args, 'log_timestamps', False) or None  # None = auto

    # Setup logging with colors
    setup_logging(level=log_level, show_timestamps=show_timestamps)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Handle auth command separately (doesn't need credential check)
    if args.command == "auth":
        cmd_auth(args)
        return

    # Credentials are validated lazily, per model channel, rather than up front:
    # a run may use only API-key providers and need no AWS credentials at all.
    # Bedrock clients check on creation, and the retry loop in utils/retry.py
    # calls ensure_credentials() when it sees ExpiredTokenException. Run
    # 'patient-agent-bench auth' to check AWS credentials before a Bedrock run.

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "generate-seeds":
        run_generate_seeds(args)
    elif args.command == "review-annotate":
        cmd_review_annotate(args)
    elif args.command == "align-check":
        cmd_align_check(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
