# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Stratified conversation sampler for human review."""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from patient_agent_bench.review.models import (
    RUBRIC_DIMENSIONS,
    ConversationMessage,
    ExperimentInfo,
    RubricScore,
    SampledConversation,
)

logger = logging.getLogger(__name__)


def _conv_key(conv: SampledConversation) -> str:
    """Unique key for dedup: experiment_id::case_id."""
    return f"{conv.experiment.experiment_id}::{conv.case_id}"


def _is_experiment_dir(name: str) -> bool:
    """Check if directory name matches experiment ID format (X_Y or legacy X_Y_Z)."""
    parts = name.split("_")
    if len(parts) not in (2, 3):
        return False
    return all(part.isdigit() for part in parts)


class ConversationSampler:
    """Loads evaluations from a run directory and performs stratified sampling."""

    def __init__(self, run_dir: Path, seed: Optional[int] = None):
        self.run_dir = Path(run_dir)
        self.seed = seed
        self._conversations: List[SampledConversation] = []
        self._experiment_ids: List[str] = []

        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {self.run_dir}")

        self._load_experiments()

        if not self._conversations:
            raise ValueError(f"No experiments with evaluations found in {self.run_dir}")

    def _load_experiments(self) -> None:
        """Discover experiment directories and load evaluations."""
        for entry in sorted(self.run_dir.iterdir()):
            if not entry.is_dir() or not _is_experiment_dir(entry.name):
                continue

            evals_path = entry / "evaluations.json"
            config_path = entry / "experiment_config.json"

            if not evals_path.exists():
                logger.warning("Skipping %s: missing evaluations.json", entry.name)
                continue
            if not config_path.exists():
                logger.warning("Skipping %s: missing experiment_config.json", entry.name)
                continue

            try:
                with open(config_path, encoding="utf-8") as f:
                    config = json.load(f)
                with open(evals_path, encoding="utf-8") as f:
                    evaluations = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Skipping %s: %s", entry.name, e)
                continue

            experiment_id = config.get("experiment_id", entry.name)
            assistant_spec = config.get("assistant_agent", {})
            user_spec = config.get("user_agent", {})

            exp_info = ExperimentInfo(
                experiment_id=experiment_id,
                assistant_label=assistant_spec.get("label") or assistant_spec.get("agent_class", "unknown"),
                user_label=user_spec.get("label") or user_spec.get("agent_class", "unknown"),
                agent_class=assistant_spec.get("agent_class", "unknown"),
            )

            self._experiment_ids.append(experiment_id)

            for ev in evaluations:
                evaluation = ev.get("evaluation", {})
                rubric_results = evaluation.get("rubric_results", {})

                llm_scores: Dict[str, RubricScore] = {}
                for dim in RUBRIC_DIMENSIONS:
                    result = rubric_results.get(dim, {})
                    llm_scores[dim] = RubricScore(
                        score=result.get("score", 0),
                        explanation=result.get("explanation", ""),
                    )

                # Extract simple message list (type + content only)
                raw_conv = ev.get("conversation", [])
                messages = []
                for msg in raw_conv:
                    msg_type = msg.get("type", "human")
                    content = msg.get("content", "")
                    # For AI messages, build clean content
                    if msg_type == "ai":
                        raw_content = msg.get("content", "")
                        # Extract tool call names and inputs from list content
                        tool_calls = []
                        if isinstance(raw_content, list):
                            for p in raw_content:
                                if isinstance(p, dict) and p.get("type") == "tool_use":
                                    name = p.get("name", "unknown_tool")
                                    inputs = p.get("input", {})
                                    if inputs and isinstance(inputs, dict):
                                        params = []
                                        for k, v in inputs.items():
                                            v_str = str(v)
                                            if len(v_str.split()) > 3:
                                                v_str = " ".join(v_str.split()[:3]) + "…"
                                            params.append(f"{k}={v_str}")
                                        tool_calls.append(f"{name}({', '.join(params)})")
                                    else:
                                        tool_calls.append(f"{name}()")
                        tool_summary = (
                            "[" + " | ".join(tool_calls) + "]"
                            if tool_calls else ""
                        )
                        # Use response_shown_to_user_agent for text, or empty
                        shown = msg.get("response_metadata", {}).get(
                            "response_shown_to_user_agent", ""
                        )
                        parts = [p for p in [shown, tool_summary] if p]
                        content = "\n".join(parts)
                    if isinstance(content, list):
                        content = ""
                    messages.append(ConversationMessage(type=msg_type, content=content))

                agg_score = evaluation.get("aggregate_score", 0.0)

                self._conversations.append(
                    SampledConversation(
                        case_id=ev.get("case_id", "unknown"),
                        experiment=exp_info,
                        conversation=messages,
                        patient_profile=ev.get("user_profile", ""),
                        scenario=ev.get("scenario", ""),
                        num_turns=ev.get("num_turns", len(messages)),
                        llm_scores=llm_scores,
                        aggregate_score=agg_score,
                    )
                )

    @property
    def total_conversations(self) -> int:
        return len(self._conversations)

    @property
    def experiment_ids(self) -> List[str]:
        return list(self._experiment_ids)

    def sample(self, num_samples: int) -> List[SampledConversation]:
        """
        Multi-dimensional stratified sampling with guaranteed rubric bucket coverage.

        Strategy:
        1. Global guarantee: for each (rubric × score bucket × experiment) triple
           that exists, pick at least 1 conversation (deduped). First covers each
           unique (rubric × bucket) pair, then additional experiment triples.
        2. Fill: remaining slots via rubric-diversity round-robin.
        3. Use random.Random(seed) for reproducibility.
        """
        total = len(self._conversations)
        if num_samples >= total:
            if num_samples > total:
                logger.warning(
                    "Requested %d samples but only %d available; returning all.",
                    num_samples,
                    total,
                )
            return list(self._conversations)

        rng = random.Random(self.seed)
        from collections import Counter

        selected_ids: set[str] = set()
        result: List[SampledConversation] = []
        exp_pick_counts: Counter[str] = Counter()

        # --- Build index: (dim, bucket, experiment_id) -> [conversations] ---
        triple_buckets: Dict[
            tuple[str, int, str], List[SampledConversation]
        ] = defaultdict(list)
        for conv in self._conversations:
            exp_id = conv.experiment.experiment_id
            for dim in RUBRIC_DIMENSIONS:
                score = conv.llm_scores.get(dim)
                if score and score.score > 0:
                    bucket = max(1, min(5, round(score.score)))
                    triple_buckets[(dim, bucket, exp_id)].append(conv)

        for bucket_list in triple_buckets.values():
            rng.shuffle(bucket_list)

        # Group triples by (dim, bucket) -> list of experiment_ids
        from itertools import groupby
        dim_bucket_groups: Dict[tuple[str, int], List[str]] = defaultdict(list)
        for dim, bucket, exp_id in triple_buckets:
            if exp_id not in dim_bucket_groups[(dim, bucket)]:
                dim_bucket_groups[(dim, bucket)].append(exp_id)

        # Sort (dim, bucket) pairs: extremes first (1, 5, 2, 4, 3), then by dim
        sorted_dim_buckets = sorted(
            dim_bucket_groups.keys(),
            key=lambda db: (-abs(db[1] - 3), db[0]),
        )

        # --- Pass 1 round 1: 1 pick per unique (dim, bucket), balanced by experiment ---
        covered: set[tuple[str, int]] = set()
        for dim, bucket in sorted_dim_buckets:
            if len(result) >= num_samples:
                break
            if (dim, bucket) in covered:
                continue
            # Sort experiments by fewest picks so far
            exp_ids = sorted(
                dim_bucket_groups[(dim, bucket)],
                key=lambda e: exp_pick_counts[e],
            )
            picked = False
            for exp_id in exp_ids:
                for conv in triple_buckets[(dim, bucket, exp_id)]:
                    k = _conv_key(conv)
                    if k not in selected_ids:
                        selected_ids.add(k)
                        result.append(conv)
                        exp_pick_counts[exp_id] += 1
                        covered.add((dim, bucket))
                        picked = True
                        break
                if picked:
                    break

        # --- Pass 1 round 2: additional experiment coverage for each (dim, bucket) ---
        for dim, bucket in sorted_dim_buckets:
            if len(result) >= num_samples:
                break
            exp_ids = sorted(
                dim_bucket_groups[(dim, bucket)],
                key=lambda e: exp_pick_counts[e],
            )
            for exp_id in exp_ids:
                if len(result) >= num_samples:
                    break
                for conv in triple_buckets[(dim, bucket, exp_id)]:
                    k = _conv_key(conv)
                    if k not in selected_ids:
                        selected_ids.add(k)
                        result.append(conv)
                        exp_pick_counts[exp_id] += 1
                        break

        if len(result) >= num_samples:
            result = result[:num_samples]
            self._log_sampling_summary(result)
            return result

        # --- Pass 2: fill remaining via rubric-diversity round-robin ---
        remaining_pool = [
            c for c in self._conversations if _conv_key(c) not in selected_ids
        ]
        fill_needed = num_samples - len(result)
        filled = self._select_by_rubric_diversity(
            remaining_pool, fill_needed, rng, selected_ids
        )
        result.extend(filled)

        self._log_sampling_summary(result)
        return result


    def _select_by_rubric_diversity(
        self,
        conversations: List[SampledConversation],
        num_to_select: int,
        rng: random.Random,
        already_selected: Optional[set[str]] = None,
    ) -> List[SampledConversation]:
        """
        Select conversations maximizing score diversity across all rubric dimensions.

        For each rubric dimension, bucket conversations by rounded score (1-5).
        Round-robin across dimensions, picking from the score bucket with the fewest
        already-selected representatives. Deduplicates throughout.
        """
        if num_to_select >= len(conversations):
            return list(conversations)

        excluded = already_selected or set()

        # Build per-dimension buckets: dim -> score -> [conversations]
        dim_buckets: Dict[str, Dict[int, List[SampledConversation]]] = {}
        for dim in RUBRIC_DIMENSIONS:
            buckets: Dict[int, List[SampledConversation]] = defaultdict(list)
            for conv in conversations:
                if _conv_key(conv) in excluded:
                    continue
                score = conv.llm_scores.get(dim)
                if score:
                    bucket = max(1, min(5, round(score.score)))
                    buckets[bucket].append(conv)
            for b in buckets.values():
                rng.shuffle(b)
            dim_buckets[dim] = dict(buckets)

        local_ids: set[str] = set()
        selected: List[SampledConversation] = []

        # Round-robin across dimensions
        dim_list = list(RUBRIC_DIMENSIONS)
        rng.shuffle(dim_list)
        dim_cycle_idx = 0

        while len(selected) < num_to_select:
            made_progress = False

            for _ in range(len(dim_list)):
                if len(selected) >= num_to_select:
                    break

                dim = dim_list[dim_cycle_idx % len(dim_list)]
                dim_cycle_idx += 1
                buckets = dim_buckets[dim]

                bucket_selected_counts: Dict[int, int] = {}
                for bucket_score in buckets:
                    count = 0
                    for conv in selected:
                        s = conv.llm_scores.get(dim)
                        if s and max(1, min(5, round(s.score))) == bucket_score:
                            count += 1
                    bucket_selected_counts[bucket_score] = count

                sorted_buckets = sorted(
                    buckets.keys(),
                    key=lambda b: (bucket_selected_counts.get(b, 0), -abs(b - 3)),
                )

                picked = False
                for bucket_score in sorted_buckets:
                    for conv in buckets[bucket_score]:
                        k = _conv_key(conv)
                        if k not in local_ids and k not in excluded:
                            local_ids.add(k)
                            selected.append(conv)
                            picked = True
                            made_progress = True
                            break
                    if picked:
                        break

            if not made_progress:
                break

        # Fill remaining from unselected pool if needed
        if len(selected) < num_to_select:
            remaining_pool = [
                c for c in conversations
                if _conv_key(c) not in local_ids and _conv_key(c) not in excluded
            ]
            rng.shuffle(remaining_pool)
            selected.extend(remaining_pool[: num_to_select - len(selected)])

        return selected[:num_to_select]

    @staticmethod
    def _log_sampling_summary(samples: List[SampledConversation]) -> None:
        """Log a table showing per-experiment and per-rubric score bucket distribution."""
        if not samples:
            return

        from collections import Counter

        # Per-experiment counts
        exp_counts: Counter[str] = Counter()
        for conv in samples:
            label = conv.experiment.assistant_label or conv.experiment.experiment_id
            exp_counts[label] += 1

        # Per-rubric-dimension bucket counts
        dim_bucket_counts: Dict[str, Counter[int]] = {}
        for dim in RUBRIC_DIMENSIONS:
            dim_bucket_counts[dim] = Counter()
            for conv in samples:
                score = conv.llm_scores.get(dim)
                if score:
                    bucket = max(1, min(5, round(score.score)))
                    dim_bucket_counts[dim][bucket] += 1

        lines = [f"Sampling summary ({len(samples)} conversations):"]

        # Experiment distribution
        lines.append("  Experiments:")
        for label in sorted(exp_counts):
            lines.append(f"    {label}: {exp_counts[label]}")

        # Rubric dimension distribution (compact table)
        # Header
        lines.append("  Rubric score distribution:")
        header = f"    {'Dimension':<28s} | {'1':>3s} | {'2':>3s} | {'3':>3s} | {'4':>3s} | {'5':>3s}"
        lines.append(header)
        lines.append("    " + "-" * (len(header) - 4))
        for dim in RUBRIC_DIMENSIONS:
            counts = dim_bucket_counts[dim]
            row = (
                f"    {dim:<28s} |"
                f" {counts.get(1, 0):>3d} |"
                f" {counts.get(2, 0):>3d} |"
                f" {counts.get(3, 0):>3d} |"
                f" {counts.get(4, 0):>3d} |"
                f" {counts.get(5, 0):>3d}"
            )
            lines.append(row)

        logger.info("\n".join(lines))
