# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Output Manager for PatientAgentBench.

Handles creation and management of output directories and files
for benchmark runs, organizing results by stage.

Supports multi-experiment runs with:
- Per-experiment subdirectories
- Cross-experiment summary aggregation
- Run config bookkeeping with CLI params
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from patient_agent_bench.config import BenchConfig
from patient_agent_bench.logging_config import get_logger
from patient_agent_bench.runner.experiment_config import ExperimentConfig
from patient_agent_bench.eval.charts import (
    generate_experiment_breakdown_charts,
    generate_metrics_csv,
    generate_ridgeline_chart,
    generate_model_spread_charts,
    generate_token_usage_charts,
)
from patient_agent_bench.eval.stats import (
    compute_aggregate_ci,
    compute_breakdowns,
    compute_pass_rate_cis,
    compute_rubric_score_stats,
    compute_score_stats,
    wilson_ci,
)
from patient_agent_bench.eval.constants import MAX_SCORE, PASS_THRESHOLD
from patient_agent_bench.eval.tokens import compute_token_stats, compute_cost_block
from patient_agent_bench.model_registry import get_model_pricing

logger = get_logger(__name__)


class OutputManager:
    """
    Manages output directory structure and file operations for benchmark runs.

    Creates a timestamped output directory containing:
    - conversations.json: Generated conversation data
    - evaluations.json: Evaluation results
    - summary.json: Aggregate metrics and summary table
    """

    def __init__(
        self,
        input_file: str,
        base_output_dir: str = "output",
        timestamp: Optional[str] = None,
    ):
        """
        Initialize the output manager.

        Args:
            input_file: Path to the input benchmark file
            base_output_dir: Base directory for all outputs
            timestamp: Optional timestamp string (auto-generated if not provided)
        """
        self.input_file = Path(input_file)
        self.base_output_dir = Path(base_output_dir)
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directory name from input file stem + timestamp
        input_stem = self.input_file.stem
        self.run_dir = self.base_output_dir / f"{input_stem}_{self.timestamp}"

        # Define output file paths
        self.conversations_file = self.run_dir / "conversations.json"
        self.evaluations_file = self.run_dir / "evaluations.json"
        self.summary_file = self.run_dir / "summary.json"

    def setup(self) -> Path:
        """
        Create the output directory structure.

        Returns:
            Path to the created run directory
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created output directory: %s", self.run_dir)
        return self.run_dir

    def save_run_config(
        self, config: BenchConfig, cli_params: Dict[str, Any]
    ) -> Path:
        """
        Save the full run config with CLI parameters to run_config.json.

        This copies the input configuration to the output directory along with
        CLI run parameters for reproducibility and tracking.

        Args:
            config: The BenchConfig used for this run
            cli_params: Dictionary of CLI parameters (benchmark_file, num_cases, etc.)

        Returns:
            Path to the saved run_config.json file
        """
        run_config = {
            **config.to_dict(),
            "cli_params": cli_params,
        }
        path = self.run_dir / "run_config.json"
        self._write_json(path, run_config)
        logger.info("Saved run config to %s", path)
        return path

    def save_benchmark_cases(self, cases_file: str) -> Path:
        """
        Copy the benchmark cases file into the run directory.

        Saves as benchmark_cases.json so evaluate can load case metadata
        without depending on the original file path.

        Args:
            cases_file: Path to the original benchmark cases JSON file.

        Returns:
            Path to the saved benchmark_cases.json file.
        """
        import shutil

        dest = self.run_dir / "benchmark_cases.json"
        if not dest.exists():
            shutil.copy2(cases_file, dest)
            logger.info("Copied benchmark cases to %s", dest)
        return dest

    def get_experiment_dir(self, experiment_id: str) -> Path:
        """
        Get or create experiment subdirectory.

        Creates a subdirectory named by experiment_id under the run directory
        for storing experiment-specific results.

        Args:
            experiment_id: The experiment identifier (e.g., "0_0_0", "1_0_2")

        Returns:
            Path to the experiment subdirectory
        """
        exp_dir = self.run_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def save_experiment_results(
        self,
        experiment: ExperimentConfig,
        conversations: List[Dict[str, Any]],
        evaluations: List[Dict[str, Any]],
        case_metadata: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Path:
        """
        Save all results for a single experiment.

        Creates the experiment subdirectory and saves:
        - experiment_config.json: The specific model configs for this experiment
        - conversations.json: Generated conversation data
        - evaluations.json: Evaluation results
        - summary.json: Aggregate metrics for this experiment

        Args:
            experiment: The ExperimentConfig for this experiment
            conversations: List of conversation data dictionaries
            evaluations: List of evaluation result dictionaries
            case_metadata: Optional dict mapping case_id to seed attributes

        Returns:
            Path to the experiment subdirectory
        """
        exp_dir = self.get_experiment_dir(experiment.experiment_id)

        # Save experiment-specific config
        self._write_json(exp_dir / "experiment_config.json", experiment.to_dict())

        # Save conversations and evaluations
        self._write_json(exp_dir / "conversations.json", conversations)
        self._write_json(exp_dir / "evaluations.json", evaluations)

        # Generate the experiment summary (pass conversations so token usage
        # stats are computed from message-level usage_metadata).
        summary = self.generate_summary(evaluations, case_metadata, conversations)

        # Graft on an approximate per-conversation $ cost block when the
        # assistant model is priced in the registry. The model identity comes
        # from the experiment config we just wrote to experiment_config.json
        # (assistant_agent.model), and price is inferred from the registry
        # (get_model_pricing accepts a registry key or full model_id).
        # Computed HERE (not at publish) so it lives in summary.json and
        # travels with the run; unpriced models simply omit the cost block.
        if "token_stats" in summary:
            model = experiment.assistant_agent.model
            # Prefer the registry key, but skip the "custom" sentinel that
            # parse_model_config assigns to custom specs (it's truthy but not a
            # real name) — fall through to the full model_id, which
            # get_model_pricing also resolves. Otherwise custom-spec models
            # (e.g. thinking-effort overrides) silently lose their cost block.
            model_name = (
                model.model
                if (model.model and model.model != "custom")
                else model.model_id
            )
            pricing = get_model_pricing(model_name)
            if pricing:
                cost = compute_cost_block(
                    summary["token_stats"],
                    pricing["input_price_per_1m"],
                    pricing["output_price_per_1m"],
                )
                if cost:
                    summary["token_stats"]["cost"] = cost

        self._write_json(exp_dir / "summary.json", summary)

        # Generate breakdown charts if breakdowns were computed
        if "breakdowns" in summary:
            model_label = (
                experiment.assistant_agent.label
                or experiment.assistant_agent.model.model
                or experiment.assistant_agent.model.model_id
                or "unknown"
            )
            generate_experiment_breakdown_charts(
                summary, str(exp_dir), experiment_label=model_label
            )
            logger.info(
                "Generated breakdown charts for experiment %s",
                experiment.experiment_id,
            )

        logger.info(
            "Saved experiment %s results to %s",
            experiment.experiment_id,
            exp_dir,
        )
        return exp_dir

    def init_evaluations_file(
        self,
        experiment_id: str,
        num_entries: int,
    ) -> Path:
        """
        Pre-initialize evaluations.json with empty-dict placeholders.

        Creates a JSON array of N empty dicts so individual evaluator
        results can be written per-conversation as they complete.

        If the file already exists, it is left untouched.

        Args:
            experiment_id: The experiment identifier
            num_entries: Number of conversation slots to pre-allocate

        Returns:
            Path to the evaluations.json file
        """
        exp_dir = self.get_experiment_dir(experiment_id)
        path = exp_dir / "evaluations.json"
        if not path.exists():
            self._write_json(path, [{} for _ in range(num_entries)])
            logger.info(
                "Initialized evaluations.json with %d slots for experiment %s",
                num_entries,
                experiment_id,
            )
        return path

    def save_evaluation_at_index(
        self,
        experiment_id: str,
        evaluator_idx: int,
        conv_idx: int,
        result: Dict[str, Any],
    ) -> None:
        """
        Write a single evaluator result into its slot in evaluations.json.

        Reads the current file, sets evaluation_{evaluator_idx} on the
        entry at conv_idx, and writes back. Caller must ensure no
        concurrent writes to the same file (use a lock).

        Args:
            experiment_id: The experiment identifier
            evaluator_idx: Index of the evaluator model
            conv_idx: Positional index in the evaluations array
            result: Evaluation result dict for this conversation
        """
        exp_dir = self.run_dir / experiment_id
        path = exp_dir / "evaluations.json"
        data = self._read_json(path)
        data[conv_idx][f"evaluation_{evaluator_idx}"] = result
        self._write_json(path, data)

    def load_partial_evaluator_results(
        self,
        experiment_id: str,
        evaluator_idx: int,
        num_conversations: int,
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Load per-conversation evaluator results from evaluations.json.

        Returns a list of length num_conversations where completed slots
        have the result dict and pending slots are None.

        Args:
            experiment_id: The experiment identifier
            evaluator_idx: Index of the evaluator model
            num_conversations: Expected number of conversations

        Returns:
            List with result dicts for completed slots, None for pending.
        """
        exp_dir = self.run_dir / experiment_id
        path = exp_dir / "evaluations.json"
        key = f"evaluation_{evaluator_idx}"

        if not path.exists():
            return [None] * num_conversations

        data = self._read_json(path)
        results: List[Optional[Dict[str, Any]]] = []
        for i in range(num_conversations):
            if i < len(data) and key in data[i]:
                results.append(data[i][key])
            else:
                results.append(None)
        return results

    def init_conversations_file(
        self,
        experiment_id: str,
        num_entries: int,
    ) -> Path:
        """
        Pre-initialize conversations.json with null placeholders.

        Creates a JSON array of N nulls so individual slots can be
        updated atomically as each conversation completes.

        If the file already exists, it is left untouched.

        Args:
            experiment_id: The experiment identifier
            num_entries: Number of conversation slots to pre-allocate

        Returns:
            Path to the conversations.json file
        """
        exp_dir = self.get_experiment_dir(experiment_id)
        path = exp_dir / "conversations.json"
        if not path.exists():
            self._write_json(path, [None] * num_entries)
            logger.info(
                "Initialized conversations.json with %d slots for experiment %s",
                num_entries,
                experiment_id,
            )
        return path

    def save_conversation_at_index(
        self,
        experiment_id: str,
        index: int,
        conv_dict: Dict[str, Any],
    ) -> None:
        """
        Write a single conversation result into its slot in conversations.json.

        Reads the current file, updates the slot at `index`, and writes back.
        Caller must ensure no concurrent writes to the same file (use a lock).

        Args:
            experiment_id: The experiment identifier
            index: Positional index in the conversations array
            conv_dict: Serialized ConversationResult dict
        """
        exp_dir = self.run_dir / experiment_id
        path = exp_dir / "conversations.json"
        data = self._read_json(path)
        data[index] = conv_dict
        self._write_json(path, data)

    def load_partial_conversations(
        self,
        experiment_id: str,
    ) -> Optional[List[Optional[Dict[str, Any]]]]:
        """
        Load conversations.json which may contain null slots.

        Returns:
            List with dicts for completed conversations and None for
            pending slots, or None if the file doesn't exist.
        """
        exp_dir = self.run_dir / experiment_id
        path = exp_dir / "conversations.json"
        if not path.exists():
            return None
        result: List[Optional[Dict[str, Any]]] = self._read_json(path)
        return result

    def generate_summary(
        self,
        evaluations: List[Dict[str, Any]],
        case_metadata: Optional[Dict[str, Dict[str, str]]] = None,
        conversations: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate summary metrics from evaluation results.

        Args:
            evaluations: List of evaluation result dictionaries
            case_metadata: Optional dict mapping case_id to seed attributes
                (severity_level, task_type, scenario_complexity, etc.)
                When provided, adds confidence intervals and per-attribute
                breakdowns to the summary.
            conversations: Optional list of conversation records. When provided,
                adds a ``token_stats`` block (output/peak-input/reasoning token
                usage + coverage) computed from each conversation's messages.
                The approximate $ cost block is added later by the caller
                (save_experiment_results), which knows the run's model identity.

        Returns:
            Summary dictionary with aggregate metrics, inter-rater agreement,
            confidence intervals, per-attribute breakdowns, per-case table, and
            (when conversations are provided) token usage statistics.
        """
        if not evaluations:
            return {"error": "No evaluations to summarize"}

        # Extract scores and agreement metrics
        scores = []
        aggregate_score_stds = []
        rubric_totals: Dict[str, List[float]] = {}
        rubric_score_stds: Dict[str, List[float]] = {}
        rubric_pass_agreements: Dict[str, List[float]] = {}

        for eval_data in evaluations:
            evaluation = eval_data.get("evaluation", {})
            if not evaluation or "error" in evaluation:
                continue

            aggregate = evaluation.get("aggregate_score", 0)
            scores.append(aggregate)

            # Collect aggregate_score_std if present (multi-evaluator)
            agg_std = evaluation.get("aggregate_score_std")
            if agg_std is not None:
                aggregate_score_stds.append(agg_std)

            # Collect rubric scores and agreement metrics
            rubric_scores = evaluation.get("rubric_scores", {})
            rubric_results = evaluation.get("rubric_results", {})

            for rubric_name, score in rubric_scores.items():
                if rubric_name not in rubric_totals:
                    rubric_totals[rubric_name] = []
                    rubric_score_stds[rubric_name] = []
                    rubric_pass_agreements[rubric_name] = []
                rubric_totals[rubric_name].append(score)

                # Collect agreement metrics from rubric_results
                result = rubric_results.get(rubric_name, {})
                if "score_std" in result:
                    rubric_score_stds[rubric_name].append(result["score_std"])
                if "pass_agreement" in result:
                    rubric_pass_agreements[rubric_name].append(result["pass_agreement"])

        # Calculate averages
        avg_score = sum(scores) / len(scores) if scores else 0
        rubric_averages = {
            name: sum(vals) / len(vals) if vals else 0
            for name, vals in rubric_totals.items()
        }

        # Calculate pass rates per rubric (score >= PASS_THRESHOLD)
        rubric_pass_rates = {
            name: (sum(1 for v in vals if v >= PASS_THRESHOLD) / len(vals) * 100) if vals else 0
            for name, vals in rubric_totals.items()
        }

        # Calculate average agreement metrics per rubric
        rubric_score_std_avgs = {
            name: sum(vals) / len(vals) if vals else 0
            for name, vals in rubric_score_stds.items()
        }
        rubric_pass_agreement_avgs = {
            name: sum(vals) / len(vals) if vals else 0
            for name, vals in rubric_pass_agreements.items()
        }

        # Calculate average aggregate_score_std across conversations
        aggregate_score_std_avg = (
            sum(aggregate_score_stds) / len(aggregate_score_stds)
            if aggregate_score_stds
            else 0
        )

        # Build per-case table
        case_table = []
        num_turns_list: List[float] = []
        for eval_data in evaluations:
            case_id = eval_data.get("case_id", "unknown")
            evaluation = eval_data.get("evaluation", {})
            num_turns = eval_data.get("num_turns")

            row: Dict[str, Any] = {
                "case_id": case_id,
                "aggregate_score": evaluation.get("aggregate_score"),
                "num_turns": num_turns,
            }

            # Add individual rubric scores
            rubric_scores = evaluation.get("rubric_scores", {})
            for rubric_name, score in rubric_scores.items():
                row[rubric_name] = score

            case_table.append(row)
            if num_turns is not None:
                num_turns_list.append(num_turns)

        summary: Dict[str, Any] = {
            "run_info": {
                "input_file": str(self.input_file),
                "timestamp": self.timestamp,
                "output_dir": str(self.run_dir),
            },
            "aggregate_metrics": {
                "total_cases": len(evaluations),
                "evaluated_cases": len(scores),
                "average_score": round(avg_score, 2),
            },
            "rubric_averages": {k: round(v, 2) for k, v in rubric_averages.items()},
            "rubric_pass_rates": {k: round(v, 2) for k, v in rubric_pass_rates.items()},
            "case_results": case_table,
        }

        # Add score statistics (Family 1: variability across cases)
        if scores:
            agg_stats = compute_score_stats(scores)
            summary["aggregate_score_stats"] = agg_stats
            summary["rubric_score_stats"] = compute_rubric_score_stats(rubric_totals)

        # Add num_turns statistics
        if num_turns_list:
            summary["num_turns_stats"] = compute_score_stats(num_turns_list)

        # Add inter-rater agreement metrics if multi-evaluator data present
        # (Family 3: evaluator consistency, prefixed with ira_)
        has_agreement_data = (
            aggregate_score_stds
            or any(rubric_score_stds.values())
            or any(rubric_pass_agreements.values())
        )
        if has_agreement_data:
            agreement: Dict[str, Any] = {}
            if aggregate_score_stds:
                agreement["ira_aggregate_score_std_avg"] = round(
                    aggregate_score_std_avg, 3
                )
            if any(rubric_score_stds.values()):
                agreement["ira_rubric_score_std_avg"] = {
                    k: round(v, 3) for k, v in rubric_score_std_avgs.items()
                }
            if any(rubric_pass_agreements.values()):
                agreement["ira_rubric_pass_agreement_avg"] = {
                    k: round(v, 3)
                    for k, v in rubric_pass_agreement_avgs.items()
                }
            summary["inter_rater_agreement"] = agreement

        # Add confidence intervals (pass rate CIs use Wilson method)
        if scores:
            summary["confidence_intervals"] = {
                "aggregate_score_ci": list(
                    compute_aggregate_ci(scores)
                ),
                "rubric_pass_rate_cis": {
                    k: list(v)
                    for k, v in compute_pass_rate_cis(
                        rubric_totals
                    ).items()
                },
                "rubric_score_cis": {
                    name: list(stats["ci"])
                    for name, stats in summary.get("rubric_score_stats", {}).items()
                },
            }

        # Add per-attribute breakdowns if metadata provided
        if case_metadata and case_table:
            rubric_names = list(rubric_totals.keys())
            summary["breakdowns"] = compute_breakdowns(
                case_table, case_metadata, rubric_names
            )

        # Add token-usage statistics if conversations provided. The $ cost
        # block (which needs the model's registry price) is grafted on by the
        # caller in save_experiment_results, since that's where model identity
        # lives — generate_summary stays independent of the model registry.
        if conversations is not None:
            summary["token_stats"] = compute_token_stats(conversations)

        return summary

    def generate_experiments_summary(
        self,
        results: Dict[str, Dict[str, Any]],
        experiments: List[ExperimentConfig],
    ) -> Dict[str, Any]:
        """
        Generate cross-experiment comparison summary.

        Creates an aggregate summary file comparing all experiments, including:
        - Comparison table with scores from all experiments
        - Model specs table mapping experiment IDs to specific models used

        Args:
            results: Dictionary mapping experiment_id to result dict containing
                     "evaluations" list and optionally "error" string
            experiments: List of ExperimentConfig objects for all experiments

        Returns:
            Summary dictionary with comparison table and model specs
        """
        comparison_table = []
        model_specs_table = []

        # Build experiment lookup for model specs
        exp_lookup = {exp.experiment_id: exp for exp in experiments}

        for exp_id, result in results.items():
            experiment = exp_lookup.get(exp_id)

            # Build model specs entry
            if experiment:

                def _display_name(mc) -> str:
                    """Get a human-readable model name, preferring model over model_id."""
                    if mc.model and mc.model != "custom":
                        return mc.model
                    return mc.model_id or "unknown"

                model_specs_table.append({
                    "experiment_id": exp_id,
                    "assistant": experiment.assistant_agent.label or _display_name(experiment.assistant_agent.model),
                    "user": experiment.user_agent.label or _display_name(experiment.user_agent.model),
                    "evaluator": [
                        _display_name(m)
                        for m in experiment.evaluator_models
                    ],
                    "assistant_agent_class": experiment.assistant_agent.agent_class,
                    "user_agent_class": experiment.user_agent.agent_class,
                    "assistant_prompt": experiment.assistant_agent.prompt,
                    "user_prompt": experiment.user_agent.prompt,
                })

            # Skip failed experiments for comparison table
            if "error" in result:
                comparison_table.append({
                    "experiment_id": exp_id,
                    "error": result["error"],
                })
                continue

            # Extract scores for comparison
            evaluations = result.get("evaluations", [])
            if not evaluations:
                comparison_table.append({
                    "experiment_id": exp_id,
                    "average_score": 0,
                    "num_cases": 0,
                })
                continue

            # Calculate metrics
            scores = []
            num_turns_list: List[float] = []
            rubric_totals: Dict[str, List[float]] = {}

            for eval_data in evaluations:
                evaluation = eval_data.get("evaluation", {})
                if not evaluation or "error" in evaluation:
                    continue

                aggregate = evaluation.get("aggregate_score", 0)
                scores.append(aggregate)
                num_turns_list.append(eval_data.get("num_turns", 0))

                # Collect rubric scores
                rubric_scores = evaluation.get("rubric_scores", {})
                for rubric_name, score in rubric_scores.items():
                    if rubric_name not in rubric_totals:
                        rubric_totals[rubric_name] = []
                    rubric_totals[rubric_name].append(score)

            avg_score = sum(scores) / len(scores) if scores else 0

            # Calculate pass rates per rubric (score >= PASS_THRESHOLD)
            rubric_pass_rates: Dict[str, float] = {}
            for rubric_name, vals in rubric_totals.items():
                if vals:
                    rubric_pass_rates[rubric_name] = (
                        sum(1 for v in vals if v >= PASS_THRESHOLD) / len(vals)
                    ) * 100
                else:
                    rubric_pass_rates[rubric_name] = 0

            row: Dict[str, Any] = {
                "experiment_id": exp_id,
                "average_score": round(avg_score, 2),
                "num_cases": len(evaluations),
                "evaluated_cases": len(scores),
            }

            # Add aggregate score CI
            if scores:
                agg_stats = compute_score_stats(scores)
                row["average_score_ci"] = list(agg_stats["ci"])
                row["average_score_std"] = agg_stats["std"]

            # Add per-rubric averages, std, CI, and pass rates
            rubric_stats = compute_rubric_score_stats(rubric_totals)
            for rubric_name, vals in rubric_totals.items():
                stats = rubric_stats[rubric_name]
                row[f"{rubric_name}_avg"] = stats["mean"]
                row[f"{rubric_name}_std"] = stats["std"]
                row[f"{rubric_name}_ci"] = list(stats["ci"])
                pass_rate = rubric_pass_rates.get(rubric_name, 0)
                row[f"{rubric_name}_pass_rate"] = round(pass_rate, 1)
                # Add Wilson CI for pass rate
                if vals:
                    passes = sum(1 for v in vals if v >= PASS_THRESHOLD)
                    ci = wilson_ci(passes, len(vals))
                    row[f"{rubric_name}_pass_rate_ci"] = list(ci)

            # Add num_turns stats
            if num_turns_list:
                nt_stats = compute_score_stats(num_turns_list)
                row["num_turns_avg"] = nt_stats["mean"]
                row["num_turns_std"] = nt_stats["std"]

            # Surface a compact token-usage summary, read from the per-experiment
            # summary.json written earlier by save_experiment_results (avoids
            # re-reading large conversation files here).
            exp_summary_path = self.run_dir / exp_id / "summary.json"
            if exp_summary_path.exists():
                try:
                    ts = json.loads(exp_summary_path.read_text()).get("token_stats")
                except (ValueError, OSError):
                    ts = None
                if ts:
                    row["output_tokens_mean"] = ts["total_output_tokens"]["mean"]
                    row["peak_input_tokens_mean"] = ts["peak_input_tokens"]["mean"]
                    row["reasoning_tokens_mean"] = ts["reasoning_tokens"]["mean"]
                    row["token_usage_coverage"] = ts["usage_coverage"]

            comparison_table.append(row)

        # Compute pooled cross-experiment stats (all cases from all experiments)
        all_scores: List[float] = []
        all_num_turns: List[float] = []
        all_rubric_totals: Dict[str, List[float]] = {}
        all_ira_agg_stds: List[float] = []
        all_ira_rubric_stds: Dict[str, List[float]] = {}
        all_ira_pass_agreements: Dict[str, List[float]] = {}

        for result in results.values():
            if "error" in result:
                continue
            for eval_data in result.get("evaluations", []):
                evaluation = eval_data.get("evaluation", {})
                if not evaluation or "error" in evaluation:
                    continue
                agg = evaluation.get("aggregate_score", 0)
                all_scores.append(agg)
                all_num_turns.append(eval_data.get("num_turns", 0))

                # Collect IRA aggregate std
                agg_std = evaluation.get("aggregate_score_std")
                if agg_std is not None:
                    all_ira_agg_stds.append(agg_std)

                rubric_scores = evaluation.get("rubric_scores", {})
                rubric_results = evaluation.get("rubric_results", {})
                for rname, rscore in rubric_scores.items():
                    all_rubric_totals.setdefault(rname, []).append(rscore)
                    # Collect IRA rubric metrics
                    rresult = rubric_results.get(rname, {})
                    if "score_std" in rresult:
                        all_ira_rubric_stds.setdefault(rname, []).append(
                            rresult["score_std"]
                        )
                    if "pass_agreement" in rresult:
                        all_ira_pass_agreements.setdefault(rname, []).append(
                            rresult["pass_agreement"]
                        )

        pooled_stats: Dict[str, Any] = {}
        if all_scores:
            pooled_stats["aggregate_score_stats"] = compute_score_stats(all_scores)
            pooled_stats["rubric_score_stats"] = compute_rubric_score_stats(
                all_rubric_totals
            )
            pooled_stats["rubric_pass_rates"] = {
                name: round(
                    sum(1 for v in vals if v >= PASS_THRESHOLD) / len(vals) * 100, 1
                )
                for name, vals in all_rubric_totals.items()
                if vals
            }
            pooled_stats["rubric_pass_rate_cis"] = {
                k: list(v)
                for k, v in compute_pass_rate_cis(all_rubric_totals).items()
            }
        if all_num_turns:
            pooled_stats["num_turns_stats"] = compute_score_stats(all_num_turns)

        # Add pooled IRA if multi-evaluator data present
        pooled_ira: Dict[str, Any] = {}
        if all_ira_agg_stds:
            pooled_ira["ira_aggregate_score_std_avg"] = round(
                sum(all_ira_agg_stds) / len(all_ira_agg_stds), 3
            )
        if all_ira_rubric_stds:
            pooled_ira["ira_rubric_score_std_avg"] = {
                k: round(sum(v) / len(v), 3)
                for k, v in all_ira_rubric_stds.items()
                if v
            }
        if all_ira_pass_agreements:
            pooled_ira["ira_rubric_pass_agreement_avg"] = {
                k: round(sum(v) / len(v), 3)
                for k, v in all_ira_pass_agreements.items()
                if v
            }
        if pooled_ira:
            pooled_stats["inter_rater_agreement"] = pooled_ira

        summary: Dict[str, Any] = {
            "total_experiments": len(results),
            "successful_experiments": len(
                [r for r in results.values() if "error" not in r]
            ),
            "comparison_table": comparison_table,
            "model_specs": model_specs_table,
        }

        # Add pooled cross-experiment statistics
        if pooled_stats:
            summary["pooled_stats"] = pooled_stats

        # Save to run directory root
        path = self.run_dir / "experiments_summary.json"
        self._write_json(path, summary)
        logger.info("Saved experiments summary to %s", path)

        # Generate cross-experiment charts if per-experiment breakdowns exist
        per_exp_summaries = self._load_per_experiment_summaries()
        if per_exp_summaries:
            charts_dir = self.run_dir / "charts"
            # Generate ridgeline score distribution chart
            ridgeline_path = str(charts_dir / "ridgeline_scores.png")
            generate_ridgeline_chart(
                summary, per_exp_summaries, ridgeline_path
            )
            logger.info("Generated ridgeline chart: %s", ridgeline_path)

            # Generate model spread charts for all meaningful combos
            spread_combos = [
                ("personality", "aggregate_score_avg"),
                ("personality", "triage_quality"),
                ("personality", "clinical_safety"),
                ("personality", "clinical_helpfulness"),
                ("personality", "conversational_quality"),
                ("personality", "workflow_accuracy"),
                ("personality", "task_completion"),
                ("personality", "num_turns"),
                ("scenario_complexity", "aggregate_score_avg"),
                ("scenario_complexity", "clinical_safety"),
                ("scenario_complexity", "task_completion"),
                ("scenario_complexity", "num_turns"),
                ("severity_level", "aggregate_score_avg"),
                ("severity_level", "num_turns"),
                ("task_category", "aggregate_score_avg"),
                ("task_category", "num_turns"),
                ("age_group", "aggregate_score_avg"),
                ("age_group", "clinical_safety"),
                ("age_group", "num_turns"),
                ("gender_identity", "aggregate_score_avg"),
                ("gender_identity", "clinical_safety"),
                ("gender_identity", "clinical_helpfulness"),
                ("gender_identity", "num_turns"),
            ]
            for attr, metric in spread_combos:
                generate_model_spread_charts(
                    summary, per_exp_summaries, str(charts_dir),
                    attributes=[attr], metric=metric,
                )
            logger.info(
                "Generated %d model spread charts", len(spread_combos)
            )

            # Generate metrics CSV
            case_metadata = self._load_case_metadata()
            if case_metadata:
                csv_path = str(self.run_dir / "metrics_by_variant.csv")
                generate_metrics_csv(
                    summary, per_exp_summaries, case_metadata, csv_path,
                )
                logger.info("Generated metrics CSV: %s", csv_path)

            # Generate token-usage comparison charts (one per metric; skips
            # cleanly if no token data was captured for any experiment)
            token_charts = generate_token_usage_charts(
                summary, per_exp_summaries, str(charts_dir)
            )
            if token_charts:
                logger.info(
                    "Generated %d token usage chart(s)",
                    sum(1 for p in token_charts if p.endswith(".png")),
                )

        return summary

    def _load_case_metadata(self) -> Optional[Dict[str, Dict[str, str]]]:
        """Load case metadata from benchmark_cases.json in the run directory."""
        bc_path = self.run_dir / "benchmark_cases.json"
        if not bc_path.exists():
            return None
        try:
            from patient_agent_bench.benchmark_seed.benchmark_entry import load_benchmark_entries
            entries = load_benchmark_entries(str(bc_path))
            return {e.id: e.metadata for e in entries}
        except Exception:
            return None

    def _write_json(self, path: Path, data: Any) -> None:
        """Write data to JSON file."""
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _read_json(self, path: Path) -> Any:
        """Read data from JSON file."""
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_per_experiment_summaries(self) -> Dict[str, Dict[str, Any]]:
        """
        Load summary.json from each experiment subdirectory.

        Returns:
            Dict mapping experiment_id to its summary dict.
            Empty dict if no experiment summaries found.
        """
        summaries: Dict[str, Dict[str, Any]] = {}
        if not self.run_dir.exists():
            return summaries
        for child in sorted(self.run_dir.iterdir()):
            if not child.is_dir():
                continue
            summary_file = child / "summary.json"
            if summary_file.exists():
                summaries[child.name] = self._read_json(summary_file)
        return summaries

    def print_summary_table(self, summary: Dict[str, Any]) -> None:
        """Print a formatted summary table to console with colors."""
        from patient_agent_bench.logging_config import Colors
        
        # Header
        print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}{'=' * 70}{Colors.RESET}")
        print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}BENCHMARK SUMMARY{Colors.RESET}")
        print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}{'=' * 70}{Colors.RESET}")

        run_info = summary.get("run_info", {})
        print(f"{Colors.BRIGHT_CYAN}Input:{Colors.RESET} {run_info.get('input_file', 'N/A')}")
        print(f"{Colors.BRIGHT_CYAN}Output:{Colors.RESET} {run_info.get('output_dir', 'N/A')}")
        print()

        metrics = summary.get("aggregate_metrics", {})
        print(f"{Colors.BRIGHT_CYAN}Total cases:{Colors.RESET} {metrics.get('total_cases', 0)}")
        print(f"{Colors.BRIGHT_CYAN}Average score:{Colors.RESET} {metrics.get('average_score', 0):.2f}/{MAX_SCORE}")
        print()

        # Rubric averages
        rubric_avgs = summary.get("rubric_averages", {})
        rubric_pass_rates = summary.get("rubric_pass_rates", {})
        if rubric_avgs:
            print(f"{Colors.BRIGHT_CYAN}Rubric Scores:{Colors.RESET}")
            for rubric, avg in rubric_avgs.items():
                display_name = rubric.replace("_", " ").title()
                pass_rate = rubric_pass_rates.get(rubric, 0)
                print(
                    f"  {display_name}: {avg:.2f}/{MAX_SCORE} "
                    f"(pass: {pass_rate:.0f}%)"
                )
        print()

        # Per-case table
        case_results = summary.get("case_results", [])
        if case_results:
            print(f"{Colors.BRIGHT_CYAN}Per-Case Results:{Colors.RESET}")
            print(f"{Colors.BRIGHT_BLACK}{'-' * 70}{Colors.RESET}")
            header = f"{'Case ID':<20} {'Score':>8}"
            print(f"{Colors.BRIGHT_CYAN}{header}{Colors.RESET}")
            print(f"{Colors.BRIGHT_BLACK}{'-' * 70}{Colors.RESET}")
            for case in case_results:
                score = case.get('aggregate_score', 0)
                score_color = Colors.BRIGHT_GREEN if score >= (MAX_SCORE - 0.8) else Colors.BRIGHT_YELLOW if score >= PASS_THRESHOLD else Colors.BRIGHT_RED
                print(f"{case.get('case_id', 'N/A'):<20} {score_color}{score:>7.2f}{Colors.RESET}")

        print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}{'=' * 70}{Colors.RESET}")

    def print_experiments_summary_table(self, summary: Dict[str, Any]) -> None:
        """
        Print a formatted cross-experiment comparison table to console.

        Displays:
        - Overview of total/successful experiments
        - Comparison table with scores from all experiments
        - Model specs table mapping experiment IDs to models

        Args:
            summary: The experiments summary dictionary from generate_experiments_summary()
        """
        from patient_agent_bench.logging_config import Colors

        # Header
        print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}{'=' * 100}{Colors.RESET}")
        print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}MULTI-EXPERIMENT SUMMARY{Colors.RESET}")
        print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}{'=' * 100}{Colors.RESET}")

        # Overview
        total = summary.get("total_experiments", 0)
        successful = summary.get("successful_experiments", 0)
        print(f"{Colors.BRIGHT_CYAN}Total experiments:{Colors.RESET} {total}")
        print(f"{Colors.BRIGHT_CYAN}Successful:{Colors.RESET} {successful}")
        print()

        # Model specs table
        model_specs = summary.get("model_specs", [])
        if model_specs:
            print(f"{Colors.BRIGHT_CYAN}Model Configurations:{Colors.RESET}")
            print(f"{Colors.BRIGHT_BLACK}{'-' * 100}{Colors.RESET}")
            header = (
                f"{'Exp ID':<10} {'Assistant':<18} {'User':<18} "
                f"{'Evaluator(s)':<25} {'User Prompt':<15} {'Asst Prompt':<15}"
            )
            print(f"{Colors.BRIGHT_CYAN}{header}{Colors.RESET}")
            print(f"{Colors.BRIGHT_BLACK}{'-' * 105}{Colors.RESET}")
            for spec in model_specs:
                exp_id = spec.get("experiment_id", "N/A")
                assistant = self._truncate(spec.get("assistant", "N/A"), 16)
                user = self._truncate(spec.get("user", "N/A"), 16)
                evaluator_val = spec.get("evaluator", "N/A")
                if isinstance(evaluator_val, list):
                    evaluator = self._truncate(", ".join(evaluator_val), 23)
                else:
                    evaluator = self._truncate(str(evaluator_val), 23)
                user_prompt = self._truncate(spec.get("user_prompt", "N/A"), 13)
                asst_prompt = self._truncate(
                    spec.get("assistant_prompt", "N/A"), 13
                )
                print(
                    f"{exp_id:<10} {assistant:<18} {user:<18} "
                    f"{evaluator:<25} {user_prompt:<15} {asst_prompt:<15}"
                )
            print()

        # Comparison table
        comparison = summary.get("comparison_table", [])
        if comparison:
            print(f"{Colors.BRIGHT_CYAN}Experiment Results (pass%){Colors.RESET}")
            print(f"{Colors.BRIGHT_BLACK}{'-' * 120}{Colors.RESET}")
            header = (
                f"{'Exp ID':<10} {'Score':>8} {'Cases':>6} "
                f"{'Safety':>12} {'Task':>12} {'Triage':>12} "
                f"{'Workflow':>12} {'Helpful':>12}"
            )
            print(f"{Colors.BRIGHT_CYAN}{header}{Colors.RESET}")
            print(f"{Colors.BRIGHT_BLACK}{'-' * 120}{Colors.RESET}")
            for row in comparison:
                exp_id = row.get("experiment_id", "N/A")
                if "error" in row:
                    print(
                        f"{exp_id:<10} "
                        f"{Colors.BRIGHT_RED}ERROR: {row['error'][:50]}{Colors.RESET}"
                    )
                    continue

                score = row.get("average_score", 0)
                cases = row.get("num_cases", 0)

                score_color = (
                    Colors.BRIGHT_GREEN if score >= (MAX_SCORE - 0.8)
                    else Colors.BRIGHT_YELLOW if score >= PASS_THRESHOLD
                    else Colors.BRIGHT_RED
                )

                # Format each rubric as pass rate
                rubrics = [
                    "clinical_safety", "task_completion", "triage_quality",
                    "workflow_accuracy", "clinical_helpfulness",
                ]
                rubric_strs = []
                for r in rubrics:
                    pr = row.get(f"{r}_pass_rate", 0)
                    rubric_strs.append(f"{pr:>5.0f}%")

                print(
                    f"{exp_id:<10} "
                    f"{score_color}{score:>8.1f}{Colors.RESET} "
                    f"{cases:>6} "
                    + " ".join(f"{s:>12}" for s in rubric_strs)
                )

                # Compact token-usage line (only when token stats were computed)
                if "output_tokens_mean" in row:
                    cov = row.get("token_usage_coverage", 1.0)
                    cov_note = "" if cov >= 1.0 else f"  {Colors.BRIGHT_YELLOW}(token coverage {cov:.0%}){Colors.RESET}"
                    reasoning = row.get("reasoning_tokens_mean", 0)
                    reasoning_str = f", reasoning {reasoning:,.0f}" if reasoning else ""
                    print(
                        f"{Colors.BRIGHT_BLACK}{'':<10} tokens/conv: "
                        f"out {row['output_tokens_mean']:,.0f}, "
                        f"peak-in {row['peak_input_tokens_mean']:,.0f}"
                        f"{reasoning_str}{Colors.RESET}{cov_note}"
                    )

        print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}{'=' * 100}{Colors.RESET}")

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text to max_len, adding ellipsis if needed."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."
