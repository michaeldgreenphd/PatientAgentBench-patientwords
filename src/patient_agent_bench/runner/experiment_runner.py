# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Experiment Runner for PatientAgentBench.

Orchestrates multi-experiment benchmark execution with support for:
- Cartesian product of assistant × user model configurations
- Multi-evaluator evaluation with aggregation
- Conversation reuse when only evaluator changes
- Per-experiment result storage
- Cross-experiment summary generation
- Parallel execution with role pool distribution
"""

import asyncio
import itertools
from typing import Any, Dict, List, Optional

from patient_agent_bench.config import BenchConfig, RolePoolManager
from patient_agent_bench.eval.aggregator import (
    add_pass_labels,
    calculate_aggregate_score,
    generate_evaluation_summary,
    merge_evaluations,
)
from patient_agent_bench.eval.evaluator import evaluate_async
from patient_agent_bench.logging_config import get_logger
from patient_agent_bench.benchmark_seed import BenchmarkEntry
from patient_agent_bench.runner.experiment_config import ExperimentConfig
from patient_agent_bench.runner.conversation_runner import (
    ConversationRunner,
    ConversationResult,
)
from patient_agent_bench.runner.conversation import Conversation
from patient_agent_bench.runner.output_manager import OutputManager
from patient_agent_bench.runner.parallel import ParallelTaskOrchestrator
from patient_agent_bench.utils.retry import RetryConfig

logger = get_logger(__name__)


class ExperimentRunner:
    """
    Orchestrates multi-experiment benchmark execution.

    Generates assistant × user combinations, runs experiments with
    multi-evaluator evaluation and aggregation, and manages result
    storage via OutputManager.
    """

    def __init__(
        self,
        config: BenchConfig,
        output_manager: OutputManager,
        max_parallel: int = 1,
        role_pool: Optional[RolePoolManager] = None,
        retry_config: Optional[RetryConfig] = None,
        generate_only: bool = False,
    ):
        self.config = config
        self.output_manager = output_manager
        self.max_parallel = max_parallel
        self.role_pool = role_pool or RolePoolManager.from_env()
        self.retry_config = retry_config or RetryConfig()
        self.conversation_cache: Dict[str, List[Any]] = {}
        self.generate_only = generate_only

    def generate_experiments(self) -> List[ExperimentConfig]:
        """
        Generate all experiment combinations via Cartesian product of
        assistant × user agents. Each experiment runs all evaluator models.

        Returns:
            List of ExperimentConfig objects.
        """
        experiments = []

        for a_idx, u_idx in itertools.product(
            range(len(self.config.assistant_agents)),
            range(len(self.config.user_agents)),
        ):
            experiments.append(
                ExperimentConfig(
                    experiment_id=f"{a_idx}_{u_idx}",
                    assistant_agent=self.config.assistant_agents[a_idx],
                    user_agent=self.config.user_agents[u_idx],
                    evaluator_models=list(self.config.evaluator_models),
                    sandbox_model=self.config.sandbox_model,
                    analyzer_model=self.config.analyzer_model,
                    assistant_idx=a_idx,
                    user_idx=u_idx,
                    max_turns=self.config.max_turns,
                    strip_thinking_content=self.config.strip_thinking_content,
                    aggregation_method=self.config.aggregation_method,
                )
            )

        return experiments

    async def _generate_conversations_async(
        self,
        experiment: ExperimentConfig,
        entries: List[BenchmarkEntry],
        existing: Optional[List[Optional[ConversationResult]]] = None,
    ) -> List[ConversationResult]:
        """
        Generate conversations in parallel with incremental save and resume.

        Pre-initializes conversations.json with null slots, skips entries
        that already have results, and writes each result to its slot as
        soon as it completes.

        Args:
            experiment: Experiment configuration
            entries: All benchmark entries (defines positional order)
            existing: Previously loaded partial results (None = no prior data)

        Returns:
            Complete list of ConversationResult in entry order.
        """
        num_entries = len(entries)
        conversations: List[Optional[ConversationResult]] = list(
            existing if existing else [None] * num_entries
        )

        # Pre-initialize the file (no-op if already exists)
        self.output_manager.init_conversations_file(
            experiment.experiment_id, num_entries
        )

        # Determine which entries still need generation
        entry_index_map = {entry.id: idx for idx, entry in enumerate(entries)}
        pending: List[tuple] = []
        runner = ConversationRunner(experiment=experiment)

        for idx, entry in enumerate(entries):
            if conversations[idx] is not None:
                logger.info(
                    "Skipping already-completed conversation %s (slot %d)",
                    entry.id,
                    idx,
                )
                continue
            pending.append(
                (entry.id, runner.run_conversation_async, (entry,), {})
            )

        if not pending:
            logger.info(
                "All %d conversations already complete for experiment %s",
                num_entries,
                experiment.experiment_id,
            )
            return conversations  # type: ignore[return-value]

        logger.info(
            "Generating %d/%d conversations for experiment %s",
            len(pending),
            num_entries,
            experiment.experiment_id,
        )

        # Lock for serializing file writes from concurrent completions
        write_lock = asyncio.Lock()

        async def _on_conversation_complete(task_result: Any) -> None:
            """Save each conversation to disk immediately on completion."""
            if task_result.success and task_result.result is not None:
                conv_result: ConversationResult = task_result.result
            else:
                conv_result = ConversationResult(
                    case_id=task_result.task_id,
                    conversation=Conversation(),
                    user_profile="",
                    scenario="",
                    num_turns=0,
                    error=task_result.error,
                )

            idx = entry_index_map[task_result.task_id]
            conversations[idx] = conv_result

            async with write_lock:
                self.output_manager.save_conversation_at_index(
                    experiment.experiment_id, idx, conv_result.to_dict()
                )

        orchestrator = ParallelTaskOrchestrator(
            max_parallel=self.max_parallel,
            role_pool=self.role_pool,
            retry_config=self.retry_config,
        )
        await orchestrator.run_all(pending, on_complete=_on_conversation_complete)

        return conversations  # type: ignore[return-value]

    async def _evaluate_conversations_async(
        self,
        experiment: ExperimentConfig,
        conversations: List[ConversationResult],
    ) -> List[Dict[str, Any]]:
        """
        Evaluate conversations with all evaluator models and merge results.

        For each evaluator model:
        1. Load partial results from evaluations.json (skip completed slots)
        2. Run only pending conversations in parallel
        3. Save each result to its slot as soon as it completes

        After all evaluators complete:
        4. Merge results using configured aggregation method
        5. Build final dict with evaluation_0, ..., evaluation_N-1, and evaluation

        Returns:
            List of evaluation result dicts with merged conversation data.
        """
        num_evaluators = len(experiment.evaluator_models)
        num_conversations = len(conversations)

        # Pre-initialize evaluations.json (no-op if already exists)
        self.output_manager.init_evaluations_file(
            experiment.experiment_id, num_conversations
        )

        # Build case_id → index map for mapping parallel results back
        case_id_to_idx = {
            conv.case_id: idx for idx, conv in enumerate(conversations)
        }

        # Run evaluation for each evaluator model with per-conversation resume
        per_evaluator_results: List[List[Dict[str, Any]]] = []
        for e_idx, evaluator_model in enumerate(experiment.evaluator_models):
            # Load partial results for this evaluator
            existing = self.output_manager.load_partial_evaluator_results(
                experiment.experiment_id, e_idx, num_conversations
            )

            # Determine which conversations still need evaluation
            pending: List[tuple] = []
            for conv_idx, conv in enumerate(conversations):
                if existing[conv_idx] is not None:
                    continue
                if conv.error is not None:
                    # Immediately save error results without running evaluator
                    existing[conv_idx] = {"error": conv.error}
                    self.output_manager.save_evaluation_at_index(
                        experiment.experiment_id, e_idx, conv_idx,
                        {"error": conv.error},
                    )
                    continue
                pending.append((
                    conv.case_id,
                    evaluate_async,
                    (
                        evaluator_model,
                        conv.conversation,
                        conv.user_profile,
                        conv.scenario,
                    ),
                    {},
                ))

            if not pending:
                logger.info(
                    "Skipping evaluator %d/%d (%s) for experiment %s"
                    " — all %d conversations already evaluated",
                    e_idx + 1,
                    num_evaluators,
                    evaluator_model.model or evaluator_model.model_id,
                    experiment.experiment_id,
                    num_conversations,
                )
            else:
                logger.info(
                    "Running evaluator %d/%d (%s) for experiment %s"
                    " — %d/%d pending",
                    e_idx + 1,
                    num_evaluators,
                    evaluator_model.model or evaluator_model.model_id,
                    experiment.experiment_id,
                    len(pending),
                    num_conversations,
                )

                write_lock = asyncio.Lock()

                async def _on_eval_complete(
                    task_result: Any,
                    _e_idx: int = e_idx,
                    _existing: List = existing,
                ) -> None:
                    """Save each evaluation to disk immediately on completion."""
                    conv_idx = case_id_to_idx[task_result.task_id]
                    if task_result.success and task_result.result is not None:
                        labeled = task_result.result
                    else:
                        labeled = {"error": task_result.error or "Unknown error"}

                    _existing[conv_idx] = labeled

                    async with write_lock:
                        self.output_manager.save_evaluation_at_index(
                            experiment.experiment_id, _e_idx, conv_idx, labeled
                        )

                orchestrator = ParallelTaskOrchestrator(
                    max_parallel=self.max_parallel,
                    role_pool=self.role_pool,
                    retry_config=self.retry_config,
                )
                await orchestrator.run_all(pending, on_complete=_on_eval_complete)

            # Re-run post-processing on all cached results so that any
            # weight or threshold changes take effect. Fresh results from
            # _on_eval_complete are also covered (functions are idempotent).
            for idx, ev in enumerate(existing):
                if ev is not None and "error" not in ev:
                    ev = add_pass_labels(ev)
                    ev = calculate_aggregate_score(ev)
                    ev = generate_evaluation_summary(ev)
                    existing[idx] = ev

            per_evaluator_results.append(existing)  # type: ignore[arg-type]

        # Build final evaluation dicts
        evaluations = []
        for conv_idx, conv in enumerate(conversations):
            conv_dict = conv.to_dict()

            # Collect individual evaluator results for this conversation
            individual_evals = [
                per_evaluator_results[e_idx][conv_idx]
                for e_idx in range(num_evaluators)
            ]

            # Add individual evaluator results as evaluation_0, evaluation_1, ...
            for e_idx, eval_result in enumerate(individual_evals):
                conv_dict[f"evaluation_{e_idx}"] = eval_result

            # Merge evaluations
            merged = merge_evaluations(
                individual_evals, method=experiment.aggregation_method
            )
            conv_dict["evaluation"] = merged

            evaluations.append(conv_dict)

        return evaluations

    async def _run_experiment_async(
        self,
        experiment: ExperimentConfig,
        entries: List[BenchmarkEntry],
    ) -> Dict[str, Any]:
        """
        Run a single experiment asynchronously with parallel execution.

        Supports:
        - Incremental conversation save (each slot written on completion)
        - Resume from partial conversations.json (skip completed slots)
        - Conversation reuse across experiments via signature cache
        """
        signature = experiment.conversation_signature()

        # Try to load partial conversations from disk for resume
        existing_convs: Optional[List[Optional[ConversationResult]]] = None
        partial_dicts = self.output_manager.load_partial_conversations(
            experiment.experiment_id
        )

        if partial_dicts is not None and signature not in self.conversation_cache:
            completed = sum(1 for d in partial_dicts if d is not None)
            logger.info(
                "Found %d/%d existing conversations for experiment %s",
                completed,
                len(partial_dicts),
                experiment.experiment_id,
            )
            existing_convs = [
                ConversationResult(
                    case_id=d.get("case_id", ""),
                    conversation=Conversation.from_dicts(
                        d.get("conversation", [])
                    ),
                    user_profile=d.get("user_profile", ""),
                    scenario=d.get("scenario", ""),
                    num_turns=d.get("num_turns", 0),
                    personality=d.get("personality", ""),
                    error=d.get("error"),
                )
                if d is not None
                else None
                for d in partial_dicts
            ]
            # If all are complete, cache them
            if all(c is not None for c in existing_convs):
                self.conversation_cache[signature] = existing_convs  # type: ignore[assignment]

        if signature in self.conversation_cache:
            conversations = self.conversation_cache[signature]
            logger.info(
                "Reusing conversations from cache (signature: %s...)",
                signature[:8],
            )
        else:
            conversations = await self._generate_conversations_async(
                experiment, entries, existing=existing_convs
            )
            self.conversation_cache[signature] = conversations

        conversations_dicts = [conv.to_dict() for conv in conversations]

        if self.generate_only:
            # Save conversations only, skip evaluation
            exp_dir = self.output_manager.get_experiment_dir(experiment.experiment_id)
            self.output_manager._write_json(
                exp_dir / "experiment_config.json", experiment.to_dict()
            )
            self.output_manager._write_json(
                exp_dir / "conversations.json", conversations_dicts
            )
            return {
                "conversations": conversations_dicts,
                "experiment_config": experiment.to_dict(),
            }

        logger.info(
            "Evaluating conversations for experiment %s with %d evaluator(s)",
            experiment.experiment_id,
            len(experiment.evaluator_models),
        )

        conversations_dicts = [conv.to_dict() for conv in conversations]

        evaluations = await self._evaluate_conversations_async(
            experiment, conversations
        )

        # Build case metadata lookup for summary breakdowns
        case_metadata = {
            entry.id: entry.metadata for entry in entries
        }

        self.output_manager.save_experiment_results(
            experiment, conversations_dicts, evaluations, case_metadata
        )

        return {
            "conversations": conversations_dicts,
            "evaluations": evaluations,
            "experiment_config": experiment.to_dict(),
        }

    async def _run_all_async(
        self,
        entries: List[BenchmarkEntry],
    ) -> Dict[str, Any]:
        """
        Async implementation of run_all with parallel execution.

        Runs experiments sequentially but uses parallel execution for
        conversations and evaluations within each experiment.
        """
        experiments = self.generate_experiments()
        results: Dict[str, Any] = {}

        logger.info(
            "Running %d experiment(s) with max_parallel=%d",
            len(experiments),
            self.max_parallel,
        )

        for i, experiment in enumerate(experiments):
            logger.info(
                "Running experiment %d/%d: %s",
                i + 1,
                len(experiments),
                experiment.experiment_id,
            )

            try:
                result = await self._run_experiment_async(experiment, entries)
                results[experiment.experiment_id] = result

                evaluations = result.get("evaluations", [])
                if evaluations:
                    scores = [
                        e.get("evaluation", {}).get("aggregate_score", 0)
                        for e in evaluations
                    ]
                    avg_score = sum(scores) / len(scores) if scores else 0
                    logger.info(
                        "Experiment %s completed: avg_score=%.1f",
                        experiment.experiment_id,
                        avg_score,
                    )

            except Exception as e:  # noqa: BLE001
                logger.error(
                    "Experiment %s failed: %s", experiment.experiment_id, e
                )
                results[experiment.experiment_id] = {
                    "error": str(e),
                    "experiment_config": experiment.to_dict(),
                }

        return {
            "experiments": results,
            "total_experiments": len(experiments),
            "successful_experiments": len(
                [r for r in results.values() if "error" not in r]
            ),
        }

    def run_all(
        self,
        entries: List[BenchmarkEntry],
    ) -> Dict[str, Any]:
        """
        Run all experiments with parallel execution.

        Args:
            entries: Benchmark entries to run

        Returns:
            Dictionary with all experiment results and summary
        """
        return asyncio.run(self._run_all_async(entries))
