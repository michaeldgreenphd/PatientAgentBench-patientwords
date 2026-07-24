# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
LLM-based analysis generator for evaluation results.

Generates insights by clustering evaluation explanations and identifying
patterns across experiments.
"""

import asyncio
import json
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from langchain_core.language_models import BaseChatModel

from patient_agent_bench.config import (
    ModelConfig,
    RolePoolManager,
    create_bedrock_client_with_role,
    create_chat_model,
)
from patient_agent_bench.logging_config import get_logger
from patient_agent_bench.analyzer.prompts import (
    EXPERIMENT_ANALYSIS_PROMPT,
    ANALYSIS_SUMMARY_PROMPT,
)
from patient_agent_bench.eval.constants import (
    PASS_THRESHOLD, MAX_SCORE, MIN_SCORE, SCORE_RANGE, SCORE_LABELS,
    SCORING_DESCRIPTION,
)
from patient_agent_bench.runner.parallel import ParallelTaskOrchestrator
from patient_agent_bench.utils.llm_content import flatten_response_content
from patient_agent_bench.utils.retry import retry_sync_with_backoff, LLM_RETRY_CONFIG, RetryConfig

logger = get_logger(__name__)


def _discover_rubric_names(evaluations: List[Dict[str, Any]]) -> List[str]:
    """
    Dynamically discover rubric names from evaluation data.

    Scans all evaluations to find unique rubric names from rubric_results.
    Returns them in a consistent sorted order.

    Args:
        evaluations: List of evaluation dictionaries.

    Returns:
        Sorted list of unique rubric names found in the data.
    """
    rubric_names: Set[str] = set()

    for eval_data in evaluations:
        evaluation = eval_data.get("evaluation", {})
        if not evaluation or "error" in evaluation:
            continue

        rubric_results = evaluation.get("rubric_results", {})
        rubric_names.update(rubric_results.keys())

    # Return sorted for consistent ordering
    return sorted(rubric_names)


def create_analyzer_llm(
    model_config: ModelConfig,
    bedrock_client,
) -> BaseChatModel:
    """
    Create an LLM client for analysis generation.

    Args:
        model_config: Model configuration for analyzer LLM.
        bedrock_client: boto3 bedrock-runtime client (required for Bedrock models,
                       ignored for OpenAI API models).

    Returns:
        BaseChatModel instance for analysis generation.

    Raises:
        ValueError: If model_config fields are missing or bedrock_client is None for Bedrock models.
    """
    if not model_config.model_id:
        raise ValueError("analyzer_model.model_id is required in config")
    # temperature is optional: some models (e.g. Opus 4.8, Sonnet 5) reject the
    # `temperature` param, so None is legitimate and create_chat_model() omits it.
    if model_config.max_tokens is None:
        raise ValueError("analyzer_model.max_tokens is required in config")
    if bedrock_client is None and model_config.requires_bedrock:
        raise ValueError("bedrock_client is required for Bedrock models")

    return create_chat_model(model_config, bedrock_client)


class AnalysisGenerator:
    """
    Generates analysis reports from evaluation results using LLM.

    Processes evaluations.json files from experiment directories and
    generates markdown analysis reports with clustered insights.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        max_samples_per_score: int = 25,
        role_arn=None,
    ):
        """
        Initialize the analysis generator.

        Args:
            model_config: Model configuration for the analyzer LLM.
            max_samples_per_score: Max samples per score level for subsampling.
            role_arn: Optional AWS IAM role ARN for Bedrock credential management.
        """
        self.model_config = model_config
        self._role_arn = role_arn
        self.max_samples_per_score = max_samples_per_score
        self._llm: Optional[BaseChatModel] = None
        self._bedrock_client = None

    def _get_llm(self) -> BaseChatModel:
        """Get or create the LLM client."""
        if self._llm is None:
            if self.model_config.requires_bedrock and self._bedrock_client is None:
                # Analyzer prompts are very large and outputs can be 32K+ tokens,
                # so use a longer read timeout than the default 900s.
                self._bedrock_client = create_bedrock_client_with_role(
                    self._role_arn, read_timeout=1800
                )
            self._llm = create_analyzer_llm(self.model_config, self._bedrock_client)
        return self._llm

    def _refresh_llm(self) -> None:
        """Recreate the bedrock client and LLM after credential refresh."""
        if self.model_config.requires_bedrock:
            self._bedrock_client = create_bedrock_client_with_role(
                self._role_arn, read_timeout=1800
            )
            self._llm = create_analyzer_llm(self.model_config, self._bedrock_client)
            logger.debug("Refreshed LLM client for AnalysisGenerator")

    def _build_conversation_summaries(
        self,
        evaluations: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Build a brief conversation summary for each case from scenario + first human message.

        Args:
            evaluations: List of evaluation dictionaries.

        Returns:
            Dict mapping case_id to a 1-2 sentence conversation summary.
        """
        summaries: Dict[str, str] = {}
        for eval_data in evaluations:
            case_id = eval_data.get("case_id", "unknown")
            scenario = eval_data.get("scenario", "")
            conversation = eval_data.get("conversation", [])

            # Extract first human message
            first_human_msg = ""
            for msg in conversation:
                if isinstance(msg, dict) and msg.get("type") == "human":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        first_human_msg = content[:300]
                    break

            # Build a compact summary from scenario (truncated) + first message
            parts = []
            if scenario:
                # Take first 200 chars of scenario for context
                parts.append(scenario[:200].rstrip())
            if first_human_msg:
                parts.append(f"Patient says: \"{first_human_msg}\"")

            if parts:
                summaries[case_id] = " | ".join(parts)

        return summaries

    def _group_by_rubric_and_score(
        self,
        evaluations: List[Dict[str, Any]],
        rubric_names: List[str],
        conversation_summaries: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
        """
        Group evaluations by rubric and score level.

        Args:
            evaluations: List of evaluation dictionaries.
            rubric_names: List of rubric names to process.
            conversation_summaries: Optional dict mapping case_id to conversation summary.

        Returns:
            Nested dict: rubric_name -> score -> list of {case_id, explanation, conversation_summary}
        """
        if conversation_summaries is None:
            conversation_summaries = {}

        grouped: Dict[str, Dict[int, List[Dict[str, Any]]]] = {
            rubric: {s: [] for s in SCORE_RANGE} for rubric in rubric_names
        }

        for eval_data in evaluations:
            case_id = eval_data.get("case_id", "unknown")
            evaluation = eval_data.get("evaluation", {})

            if not evaluation or "error" in evaluation:
                continue

            rubric_results = evaluation.get("rubric_results", {})

            for rubric_name in rubric_names:
                result = rubric_results.get(rubric_name, {})
                score = result.get("score", 0)
                explanation = result.get("explanation", "")

                # Bucket into {MIN_SCORE}-{MAX_SCORE} — merged scores may be floats
                bucket = max(MIN_SCORE, min(MAX_SCORE, round(score)))

                if explanation:
                    entry: Dict[str, Any] = {
                        "case_id": case_id,
                        "explanation": explanation,
                    }
                    conv_summary = conversation_summaries.get(case_id, "")
                    if conv_summary:
                        entry["conversation_summary"] = conv_summary
                    grouped[rubric_name][bucket].append(entry)

        return grouped

    def _subsample(
        self,
        items: List[Dict[str, Any]],
        max_items: int,
    ) -> List[Dict[str, Any]]:
        """Randomly subsample items if exceeds max."""
        if len(items) <= max_items:
            return items
        return random.sample(items, max_items)

    def _format_evaluations_for_prompt(
        self,
        grouped: Dict[str, Dict[int, List[Dict[str, Any]]]],
        rubric_names: List[str],
    ) -> str:
        """Format grouped evaluations as text for the LLM prompt."""
        lines = []

        score_labels = {s: SCORE_LABELS[s] for s in SCORE_RANGE}
        # Override lowest score label for display
        score_labels[MIN_SCORE] = "Failures"

        for rubric_name in rubric_names:
            display_name = rubric_name.replace("_", " ").title()
            lines.append(f"### {display_name}")
            lines.append("")

            for score in SCORE_RANGE:
                items = grouped[rubric_name][score]
                # Subsample if needed
                items = self._subsample(items, self.max_samples_per_score)

                label = score_labels[score]
                lines.append(f"#### {label} (Score={score}, {len(items)} cases)")

                if not items:
                    lines.append("No cases in this category.")
                    lines.append("")
                    continue

                for item in items:
                    case_id = item["case_id"]
                    explanation = item["explanation"]
                    # Truncate very long explanations but keep enough for meaningful analysis
                    if len(explanation) > 1000:
                        explanation = explanation[:1000] + "..."
                    conv_summary = item.get("conversation_summary", "")
                    if conv_summary:
                        lines.append(f"- **{case_id}**: _Conversation:_ {conv_summary}")
                        lines.append(f"  _Evaluation:_ {explanation}")
                    else:
                        lines.append(f"- **{case_id}**: {explanation}")

                lines.append("")

        return "\n".join(lines)

    def _build_score_distribution(
        self,
        grouped: Dict[str, Dict[int, List[Dict[str, Any]]]],
        rubric_names: List[str],
    ) -> str:
        """Build score distribution table text."""
        lines = [
            "| Rubric | "
            + " | ".join(
                f"{SCORE_LABELS[s]} ({s})" for s in SCORE_RANGE
            )
            + " | Pass Rate |"
        ]
        lines.append(
            "|--------"
            + "|----------" * len(SCORE_RANGE)
            + "|-----------|"
        )

        for rubric_name in rubric_names:
            display_name = rubric_name.replace("_", " ").title()
            counts = {s: len(grouped[rubric_name][s]) for s in SCORE_RANGE}
            total = sum(counts.values())
            passes = sum(counts[s] for s in range(PASS_THRESHOLD, MAX_SCORE + 1))
            pass_rate = (passes / total * 100) if total > 0 else 0
            count_cells = " | ".join(str(counts[s]) for s in SCORE_RANGE)
            lines.append(
                f"| {display_name} | {count_cells} | {pass_rate:.0f}% |"
            )

        return "\n".join(lines)

    def generate_experiment_analysis(
        self,
        experiment_id: str,
        assistant_model: str,
        evaluations: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> str:
        """
        Generate analysis markdown for a single experiment.

        Args:
            experiment_id: The experiment identifier.
            assistant_model: Name of the assistant model.
            evaluations: List of evaluation dictionaries.
            summary: Summary dictionary with aggregate metrics.

        Returns:
            Markdown string with analysis report.
        """

        # Discover rubric names dynamically from evaluation data
        rubric_names = _discover_rubric_names(evaluations)

        if not rubric_names:
            logger.warning("No rubrics found in evaluations for experiment %s", experiment_id)
            return f"# No evaluation data found for experiment {experiment_id}"

        # Build conversation summaries from scenario + first human message
        conversation_summaries = self._build_conversation_summaries(evaluations)

        # Group evaluations (with conversation summaries attached)
        grouped = self._group_by_rubric_and_score(
            evaluations, rubric_names, conversation_summaries
        )

        # Build prompt components
        score_distribution = self._build_score_distribution(grouped, rubric_names)
        evaluations_text = self._format_evaluations_for_prompt(grouped, rubric_names)

        # Extract breakdowns as raw JSON (if available)
        breakdowns = summary.get("breakdowns", {})
        breakdowns_json = json.dumps(breakdowns, indent=2) if breakdowns else "No breakdowns available."

        # Build summary metrics JSON for added context
        summary_metrics = {
            "aggregate_metrics": summary.get("aggregate_metrics", {}),
            "aggregate_score_stats": summary.get("aggregate_score_stats", {}),
            "rubric_averages": summary.get("rubric_averages", {}),
            "rubric_score_stats": summary.get("rubric_score_stats", {}),
            "rubric_pass_rates": summary.get("rubric_pass_rates", {}),
            "inter_rater_agreement": summary.get("inter_rater_agreement", {}),
        }

        prompt = EXPERIMENT_ANALYSIS_PROMPT.format(
            experiment_id=experiment_id,
            assistant_model=assistant_model,
            summary_metrics=json.dumps(summary_metrics, indent=2),
            score_distribution=score_distribution,
            evaluations_by_rubric=evaluations_text,
            breakdowns=breakdowns_json,
            scoring_description=SCORING_DESCRIPTION,
        )

        logger.info(
            "Generating analysis for experiment %s (%d cases, %d rubrics)",
            experiment_id,
            len(evaluations),
            len(rubric_names),
        )

        # Invoke LLM with retry
        logger.debug("Invoking LLM with prompt length: %d chars", len(prompt))
        logger.debug("Model: %s", self.model_config.model_id)
        response = retry_sync_with_backoff(
            lambda p: self._get_llm().invoke(p),
            prompt,
            config=LLM_RETRY_CONFIG,
            on_credential_refresh=self._refresh_llm,
        )
        logger.debug("LLM response received")
        # Normalize str-or-list content (thinking/reasoning models return a list
        # of content blocks) to plain text — str(list) would emit an ugly repr.
        return flatten_response_content(getattr(response, "content", response))

    def generate_analysis_summary(
        self,
        run_dir: str,
        experiments_summary: Dict[str, Any],
        experiment_analyses: Dict[str, str],
        experiment_summaries: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate cross-experiment comparison summary.

        Args:
            run_dir: Path to the run directory.
            experiments_summary: The experiments_summary.json content.
            experiment_analyses: Dict mapping experiment_id to analysis markdown.
            experiment_summaries: Optional dict mapping experiment_id to summary.json content
                                 (for breakdowns and stats).

        Returns:
            Markdown string with comparative analysis.
        """
        total_experiments = experiments_summary.get("total_experiments", 0)

        # Get cases per experiment from first comparison entry
        comparison = experiments_summary.get("comparison_table", [])
        cases_per_experiment = (
            comparison[0].get("num_cases", 0) if comparison else 0
        )

        # Format individual analyses — pass full content, no truncation
        analyses_text_parts = []
        for exp_id, analysis in experiment_analyses.items():
            analyses_text_parts.append(
                f"## Experiment {exp_id}\n\n{analysis}"
            )

        experiment_analyses_text = "\n\n---\n\n".join(analyses_text_parts)

        # Build per-experiment breakdowns as raw JSON for cross-experiment comparison
        breakdowns_comparison_text = ""
        if experiment_summaries:
            breakdowns_map = {}
            for exp_id, exp_summary in sorted(experiment_summaries.items()):
                breakdowns = exp_summary.get("breakdowns", {})
                if breakdowns:
                    breakdowns_map[exp_id] = breakdowns
            if breakdowns_map:
                breakdowns_comparison_text = json.dumps(breakdowns_map, indent=2)

        # Discover available chart files
        charts_dir = Path(run_dir) / "charts"
        chart_files_text = ""
        if charts_dir.exists():
            chart_files = sorted(f.name for f in charts_dir.iterdir() if f.suffix == ".png")
            if chart_files:
                chart_files_text = (
                    "Available charts in charts/ directory:\n"
                    + "\n".join(f"- {f}" for f in chart_files)
                )

        prompt = ANALYSIS_SUMMARY_PROMPT.format(
            run_dir=run_dir,
            total_experiments=total_experiments,
            cases_per_experiment=cases_per_experiment,
            experiments_summary=json.dumps(
                experiments_summary, indent=2
            ),
            experiment_analyses=experiment_analyses_text,
            breakdowns_comparison=breakdowns_comparison_text,
            chart_files=chart_files_text,
            scoring_description=SCORING_DESCRIPTION,
        )

        logger.info(
            "Generating analysis summary for %d experiments",
            total_experiments,
        )

        # Invoke LLM with retry
        response = retry_sync_with_backoff(
            lambda p: self._get_llm().invoke(p),
            prompt,
            config=LLM_RETRY_CONFIG,
            on_credential_refresh=self._refresh_llm,
        )
        # Normalize str-or-list content (thinking/reasoning models return a list
        # of content blocks) to plain text — str(list) would emit an ugly repr.
        return flatten_response_content(getattr(response, "content", response))


async def analyze_experiment_async(
    experiment_id: str,
    assistant_model: str,
    evaluations: List[Dict[str, Any]],
    summary: Dict[str, Any],
    model_config: ModelConfig,
    max_samples_per_score: int = 25,
    output_path: Optional[str] = None,
    assigned_role: Optional[str] = None,
) -> str:
    """
    Async wrapper for experiment analysis with role-based credentials.

    Creates an isolated Bedrock client with the assigned role credentials,
    then generates analysis using the AnalysisGenerator. Saves the result
    to output_path immediately if provided.

    Args:
        experiment_id: The experiment identifier.
        assistant_model: Name of the assistant model.
        evaluations: List of evaluation dictionaries.
        summary: Summary dictionary with aggregate metrics.
        model_config: Model configuration for analyzer LLM.
        max_samples_per_score: Max samples per score level for subsampling.
        output_path: Optional path to save analysis.md immediately.
        assigned_role: AWS ARN role to use (from role pool).

    Returns:
        Markdown string with analysis report.
    """
    generator = AnalysisGenerator(
        model_config=model_config,
        max_samples_per_score=max_samples_per_score,
        role_arn=assigned_role,
    )

    # Run analysis in thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        partial(
            generator.generate_experiment_analysis,
            experiment_id,
            assistant_model,
            evaluations,
            summary,
        ),
    )

    # Save incrementally if output path provided
    if output_path:
        Path(output_path).write_text(result, encoding="utf-8")
        logger.info("Saved analysis to %s", output_path)

    return result


async def analyze_experiments_parallel(
    experiment_data: List[Dict[str, Any]],
    model_config: ModelConfig,
    role_pool: RolePoolManager,
    max_parallel: int = 1,
    max_samples: int = 25,
) -> Dict[str, str]:
    """
    Analyze experiments in parallel using role pool distribution.

    Uses ParallelTaskOrchestrator for concurrent execution with
    round-robin role assignment.

    Args:
        experiment_data: List of experiment data dicts with exp_id, evaluations, etc.
            Each dict should have: exp_id, assistant_model, evaluations, summary
        model_config: Model configuration for analyzer LLM.
        role_pool: Pool of AWS ARN roles for load distribution.
        max_parallel: Maximum concurrent analysis tasks.
        max_samples: Max samples per score level for subsampling.

    Returns:
        Dict mapping experiment_id to analysis markdown.
    """
    orchestrator = ParallelTaskOrchestrator(
        max_parallel=max_parallel,
        role_pool=role_pool,
        retry_config=RetryConfig(max_retries=3),
    )

    # Build task list
    tasks: List[tuple] = [
        (
            exp["exp_id"],
            analyze_experiment_async,
            (
                exp["exp_id"],
                exp["assistant_model"],
                exp["evaluations"],
                exp["summary"],
                model_config,
                max_samples,
                str(Path(exp["exp_dir"]) / "analysis.md"),
            ),
            {},
        )
        for exp in experiment_data
    ]

    # Run all analyses in parallel
    results = await orchestrator.run_all(tasks)

    # Collect successful results
    experiment_analyses: Dict[str, str] = {}
    failed_count = 0

    for task_result in results:
        if task_result.success and task_result.result is not None:
            experiment_analyses[task_result.task_id] = task_result.result
        else:
            failed_count += 1
            logger.error(
                "Failed to analyze %s: %s",
                task_result.task_id,
                task_result.error,
            )

    if failed_count > 0:
        logger.warning(
            "Failed to analyze %d/%d experiments",
            failed_count,
            len(experiment_data),
        )

    return experiment_analyses
