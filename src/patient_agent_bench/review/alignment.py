# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Inter-rater agreement computation between LLM and human scores."""

import json
import logging
import math
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from patient_agent_bench.review.models import (
    REALISM_DIMENSIONS,
    RUBRIC_DIMENSIONS,
    AnnotationFile,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data models for alignment report
# =============================================================================


class DimensionMetrics(BaseModel):
    """Agreement metrics for a single rubric dimension."""

    cohens_kappa_weighted: Optional[float] = None
    pearson_r: Optional[float] = None
    exact_agreement_pct: float = 0.0
    adjacent_agreement_pct: float = 0.0


class PairwiseMetrics(BaseModel):
    """Pairwise agreement between two annotators for one dimension."""

    cohens_kappa_weighted: Optional[float] = None
    pearson_r: Optional[float] = None
    exact_agreement_pct: float = 0.0
    adjacent_agreement_pct: float = 0.0


class RealismSummary(BaseModel):
    """Summary statistics for a realism dimension (human-only, no LLM comparison)."""

    mean: float = 0.0
    std: float = 0.0
    n: int = 0
    distribution: Dict[int, int] = {}  # score -> count
    inter_annotator_kappa: Optional[float] = None
    inter_annotator_adj_pct: float = 0.0


class AlignmentReport(BaseModel):
    """Full alignment report."""

    timestamp: str
    num_conversations: int
    num_annotators: int
    per_rubric: Dict[str, DimensionMetrics]
    per_rubric_avg: Dict[str, DimensionMetrics] = {}
    per_rubric_filtered_avg: Dict[str, DimensionMetrics] = {}
    per_rubric_filtered_avg_2: Dict[str, DimensionMetrics] = {}
    pairwise_annotator_agreement: Dict[str, Dict[str, PairwiseMetrics]] = {}
    fleiss_kappa: Dict[str, Optional[float]] = {}
    realism_summary: Dict[str, RealismSummary] = {}


# =============================================================================
# Pure-Python statistical functions
# =============================================================================


def cohens_weighted_kappa(
    scores_a: List[int], scores_b: List[int], num_categories: int = 5
) -> Optional[float]:
    """Cohen's weighted Kappa with quadratic weights. Returns None if undefined."""
    n = len(scores_a)
    if n < 2:
        return None

    # Build observed agreement matrix
    observed = [[0] * num_categories for _ in range(num_categories)]
    for a, b in zip(scores_a, scores_b):
        observed[a - 1][b - 1] += 1

    # Marginals
    row_sums = [sum(observed[i]) for i in range(num_categories)]
    col_sums = [
        sum(observed[i][j] for i in range(num_categories))
        for j in range(num_categories)
    ]

    # Quadratic weight matrix: w_ij = 1 - (i-j)^2 / (k-1)^2
    max_diff_sq = (num_categories - 1) ** 2
    if max_diff_sq == 0:
        return None

    weights = [
        [1.0 - ((i - j) ** 2) / max_diff_sq for j in range(num_categories)]
        for i in range(num_categories)
    ]

    # Weighted observed and expected agreement
    po = sum(
        weights[i][j] * observed[i][j]
        for i in range(num_categories)
        for j in range(num_categories)
    ) / n
    pe = sum(
        weights[i][j] * row_sums[i] * col_sums[j]
        for i in range(num_categories)
        for j in range(num_categories)
    ) / (n * n)

    if abs(1.0 - pe) < 1e-10:
        return None

    return (po - pe) / (1.0 - pe)


def pearson_correlation(x: List[int], y: List[int]) -> Optional[float]:
    """Pearson correlation coefficient. Returns None for constant arrays."""
    n = len(x)
    if n < 2:
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denom = math.sqrt(var_x * var_y)
    if denom < 1e-10:
        return None

    return cov / denom


def exact_agreement_rate(scores_a: List[int], scores_b: List[int]) -> float:
    """Percentage of exact matches."""
    if not scores_a:
        return 0.0
    matches = sum(1 for a, b in zip(scores_a, scores_b) if a == b)
    return 100.0 * matches / len(scores_a)


def adjacent_agreement_rate(scores_a: List[int], scores_b: List[int]) -> float:
    """Percentage of scores within ±1."""
    if not scores_a:
        return 0.0
    matches = sum(1 for a, b in zip(scores_a, scores_b) if abs(a - b) <= 1)
    return 100.0 * matches / len(scores_a)


def fleiss_kappa(ratings_matrix: List[List[int]], num_categories: int = 5) -> Optional[float]:
    """
    Fleiss' Kappa for multiple raters.

    ratings_matrix: list of lists, each inner list has counts per category for one subject.
    """
    n_subjects = len(ratings_matrix)
    if n_subjects < 1:
        return None

    n_raters = sum(ratings_matrix[0]) if ratings_matrix else 0
    if n_raters < 2:
        return None

    # P_i for each subject
    p_values = []
    for row in ratings_matrix:
        total = sum(row)
        if total < 2:
            continue
        p_i = (sum(r * r for r in row) - total) / (total * (total - 1))
        p_values.append(p_i)

    if not p_values:
        return None

    p_bar = sum(p_values) / len(p_values)

    # p_j: proportion of all assignments to category j
    total_assignments = sum(sum(row) for row in ratings_matrix)
    if total_assignments == 0:
        return None

    p_j = [sum(row[j] for row in ratings_matrix) / total_assignments for j in range(num_categories)]
    pe_bar = sum(pj * pj for pj in p_j)

    if abs(1.0 - pe_bar) < 1e-10:
        return None

    return (p_bar - pe_bar) / (1.0 - pe_bar)


# =============================================================================
# Annotation directory merge
# =============================================================================


def merge_annotation_dir(dir_path: Path) -> Path:
    """Merge all annotations_*.json files in a directory into a single file.

    Finds the base file (with AnnotationFile metadata structure) and merges
    annotations from all other files (bare list format from HTML download)
    by case_id, deduplicating by (annotator_id, timestamp).

    Returns the path to the merged output file.
    """
    json_files = sorted(dir_path.glob("annotations*.json"))
    # Skip previously generated merged files
    json_files = [f for f in json_files if "merged" not in f.name]
    if not json_files:
        raise ValueError(f"No annotations*.json files found in {dir_path}")

    # Find base file (has metadata dict structure) and downloaded files (bare lists)
    base_data = None
    base_path: Optional[Path] = None
    downloaded: list = []

    for fpath in json_files:
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", fpath.name, e)
            continue

        if isinstance(data, dict) and "metadata" in data and "conversations" in data:
            if base_data is None:
                base_data = data
                base_path = fpath
                logger.info("Base file: %s", fpath.name)
            else:
                logger.warning(
                    "Multiple base files found, using first: %s",
                    base_path.name if base_path else "unknown",
                )
        elif isinstance(data, list):
            downloaded.append((fpath, data))
            logger.info("Downloaded file: %s (%d conversations)", fpath.name, len(data))
        else:
            logger.warning("Skipping unrecognized format: %s", fpath.name)

    if base_data is None:
        raise ValueError(f"No base annotation file (with metadata) found in {dir_path}")

    # Index base conversations by experiment_id::case_id (unique per conversation)
    conv_by_key: Dict[str, dict] = {}
    for conv in base_data["conversations"]:
        case_id = conv["case_id"]
        exp_id = conv.get("experiment", {}).get("experiment_id", "")
        key = f"{exp_id}::{case_id}"
        conv_by_key[key] = conv
        if "annotations" not in conv:
            conv["annotations"] = []

    # Merge annotations from downloaded files
    merged_count = 0
    for fpath, convs in downloaded:
        for conv in convs:
            case_id = conv.get("case_id")
            exp_id = conv.get("experiment", {}).get("experiment_id", "")
            conv_key = f"{exp_id}::{case_id}"
            if not case_id or conv_key not in conv_by_key:
                continue
            target = conv_by_key[conv_key]
            for ann in conv.get("annotations", []):
                target["annotations"].append(ann)
                merged_count += 1

    # Deduplicate: keep only the latest annotation per annotator per conversation
    for conv in base_data["conversations"]:
        anns = conv.get("annotations", [])
        if not anns:
            continue
        # Group by annotator_id, keep the one with the latest timestamp
        by_annotator: Dict[str, dict] = {}
        for ann in anns:
            aid = ann.get("annotator_id", "")
            ts = ann.get("timestamp", "")
            if aid not in by_annotator or ts > by_annotator[aid].get("timestamp", ""):
                by_annotator[aid] = ann
        conv["annotations"] = list(by_annotator.values())

    # Count stats
    total_annotations = sum(
        len(c.get("annotations", [])) for c in base_data["conversations"]
    )
    annotators = {
        a.get("annotator_id")
        for c in base_data["conversations"]
        for a in c.get("annotations", [])
    }
    annotated_cases = sum(
        1 for c in base_data["conversations"] if c.get("annotations")
    )

    logger.info(
        "Merged %d new annotations from %d files. "
        "Total: %d annotations, %d annotators, %d/%d cases annotated.",
        merged_count,
        len(downloaded),
        total_annotations,
        len(annotators),
        annotated_cases,
        len(base_data["conversations"]),
    )

    # Save merged file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_path = dir_path / f"annotations_merged_{ts}.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(base_data, f, indent=2, ensure_ascii=False)

    logger.info("Merged file saved: %s", merged_path)
    return merged_path


# =============================================================================
# AlignmentAnalyzer
# =============================================================================


class AlignmentAnalyzer:
    """Computes inter-rater agreement between LLM and human scores."""

    def __init__(self, annotation_files: List[Path]):
        self._conversations: List[dict] = []  # raw conversation dicts with annotations
        self._annotator_ids: set = set()

        for fpath in annotation_files:
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                af = AnnotationFile.model_validate(data)
            except (json.JSONDecodeError, ValueError, OSError) as e:
                logger.warning("Failed to load %s: %s", fpath, e)
                continue

            has_annotations = False
            for conv in af.conversations:
                if conv.annotations:
                    has_annotations = True
                    self._conversations.append(conv.model_dump())
                    for ann in conv.annotations:
                        self._annotator_ids.add(ann.annotator_id)

            if not has_annotations:
                logger.warning("No human annotations in %s, skipping.", fpath)

        if not self._conversations:
            raise ValueError("No annotated conversations found in any provided files.")

    def compute_agreement(self) -> AlignmentReport:
        """Compute all agreement metrics."""
        per_rubric: Dict[str, DimensionMetrics] = {}
        per_rubric_avg: Dict[str, DimensionMetrics] = {}
        per_rubric_filtered_avg: Dict[str, DimensionMetrics] = {}
        per_rubric_filtered_avg_2: Dict[str, DimensionMetrics] = {}

        for dim in RUBRIC_DIMENSIONS:
            # Mode 1: Raw (one pair per annotation)
            llm_scores, human_scores = self._extract_llm_human_pairs(dim)
            if len(llm_scores) < 2:
                per_rubric[dim] = DimensionMetrics()
            else:
                per_rubric[dim] = DimensionMetrics(
                    cohens_kappa_weighted=cohens_weighted_kappa(llm_scores, human_scores),
                    pearson_r=pearson_correlation(llm_scores, human_scores),
                    exact_agreement_pct=exact_agreement_rate(llm_scores, human_scores),
                    adjacent_agreement_pct=adjacent_agreement_rate(
                        llm_scores, human_scores
                    ),
                )

            # Mode 2: Averaged human scores per conversation
            llm_avg, human_avg = self._extract_llm_human_avg(dim)
            if len(llm_avg) < 2:
                per_rubric_avg[dim] = DimensionMetrics()
            else:
                llm_r = [max(1, min(5, round(s))) for s in llm_avg]
                hum_r = [max(1, min(5, round(s))) for s in human_avg]
                per_rubric_avg[dim] = DimensionMetrics(
                    cohens_kappa_weighted=cohens_weighted_kappa(llm_r, hum_r),
                    pearson_r=pearson_correlation(llm_r, hum_r),
                    exact_agreement_pct=exact_agreement_rate(llm_r, hum_r),
                    adjacent_agreement_pct=adjacent_agreement_rate(llm_r, hum_r),
                )

            # Mode 3: Filtered-averaged (±1 agreement filter, then average)
            llm_f, human_f = self._extract_llm_human_filtered_avg(dim, tolerance=1)
            if len(llm_f) < 2:
                per_rubric_filtered_avg[dim] = DimensionMetrics()
            else:
                llm_fr = [max(1, min(5, round(s))) for s in llm_f]
                hum_fr = [max(1, min(5, round(s))) for s in human_f]
                per_rubric_filtered_avg[dim] = DimensionMetrics(
                    cohens_kappa_weighted=cohens_weighted_kappa(llm_fr, hum_fr),
                    pearson_r=pearson_correlation(llm_fr, hum_fr),
                    exact_agreement_pct=exact_agreement_rate(llm_fr, hum_fr),
                    adjacent_agreement_pct=adjacent_agreement_rate(llm_fr, hum_fr),
                )

            # Mode 4: Filtered-averaged (±2 agreement filter, then average)
            llm_f2, human_f2 = self._extract_llm_human_filtered_avg(dim, tolerance=2)
            if len(llm_f2) < 2:
                per_rubric_filtered_avg_2[dim] = DimensionMetrics()
            else:
                llm_fr2 = [max(1, min(5, round(s))) for s in llm_f2]
                hum_fr2 = [max(1, min(5, round(s))) for s in human_f2]
                per_rubric_filtered_avg_2[dim] = DimensionMetrics(
                    cohens_kappa_weighted=cohens_weighted_kappa(llm_fr2, hum_fr2),
                    pearson_r=pearson_correlation(llm_fr2, hum_fr2),
                    exact_agreement_pct=exact_agreement_rate(llm_fr2, hum_fr2),
                    adjacent_agreement_pct=adjacent_agreement_rate(llm_fr2, hum_fr2),
                )

        # Pairwise inter-annotator agreement
        pairwise: Dict[str, Dict[str, PairwiseMetrics]] = {}
        annotator_list = sorted(self._annotator_ids)

        if len(annotator_list) >= 2:
            for a1, a2 in combinations(annotator_list, 2):
                pair_key = f"{a1}_vs_{a2}"
                pairwise[pair_key] = {}
                for dim in RUBRIC_DIMENSIONS:
                    s1, s2 = self._extract_annotator_pairs(dim, a1, a2)
                    if len(s1) < 2:
                        pairwise[pair_key][dim] = PairwiseMetrics()
                    else:
                        pairwise[pair_key][dim] = PairwiseMetrics(
                            cohens_kappa_weighted=cohens_weighted_kappa(s1, s2),
                            pearson_r=pearson_correlation(s1, s2),
                            exact_agreement_pct=exact_agreement_rate(s1, s2),
                            adjacent_agreement_pct=adjacent_agreement_rate(s1, s2),
                        )

        # Fleiss' Kappa (when >2 annotators)
        fleiss: Dict[str, Optional[float]] = {}
        if len(annotator_list) > 2:
            for dim in RUBRIC_DIMENSIONS:
                matrix = self._build_fleiss_matrix(dim, annotator_list)
                fleiss[dim] = fleiss_kappa(matrix) if matrix else None

        # Realism dimension summaries (human-only, no LLM comparison)
        realism: Dict[str, RealismSummary] = {}
        for dim in REALISM_DIMENSIONS:
            all_scores: List[int] = []
            for conv in self._conversations:
                for ann in conv.get("annotations", []):
                    s = ann.get("scores", {}).get(dim)
                    if s is not None:
                        all_scores.append(s)
            if not all_scores:
                realism[dim] = RealismSummary()
                continue
            mean_val = sum(all_scores) / len(all_scores)
            std_val = (
                (sum((x - mean_val) ** 2 for x in all_scores) / len(all_scores)) ** 0.5
            )
            dist = {}
            for s in all_scores:
                dist[s] = dist.get(s, 0) + 1
            # Inter-annotator agreement on realism (pairwise if 2+ annotators)
            ia_kappa = None
            ia_adj = 0.0
            if len(annotator_list) >= 2:
                a1, a2 = annotator_list[0], annotator_list[1]
                s1, s2 = self._extract_annotator_pairs(dim, a1, a2)
                if len(s1) >= 2:
                    ia_kappa = cohens_weighted_kappa(s1, s2)
                    ia_adj = adjacent_agreement_rate(s1, s2)
            realism[dim] = RealismSummary(
                mean=round(mean_val, 2),
                std=round(std_val, 2),
                n=len(all_scores),
                distribution=dict(sorted(dist.items())),
                inter_annotator_kappa=ia_kappa,
                inter_annotator_adj_pct=ia_adj,
            )

        return AlignmentReport(
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            num_conversations=len(self._conversations),
            num_annotators=len(annotator_list),
            per_rubric=per_rubric,
            per_rubric_avg=per_rubric_avg,
            per_rubric_filtered_avg=per_rubric_filtered_avg,
            per_rubric_filtered_avg_2=per_rubric_filtered_avg_2,
            pairwise_annotator_agreement=pairwise,
            fleiss_kappa=fleiss,
            realism_summary=realism,
        )

    def _extract_llm_human_pairs(self, dim: str) -> Tuple[List[int], List[int]]:
        """Extract paired LLM and human scores for a dimension (raw, one pair per annotation)."""
        llm_scores = []
        human_scores = []
        for conv in self._conversations:
            llm_result = conv.get("llm_scores", {}).get(dim)
            if not llm_result:
                continue
            llm_score = llm_result.get("score", 0)
            # Round float scores to nearest int for agreement computation
            llm_int = max(1, min(5, round(llm_score)))
            for ann in conv.get("annotations", []):
                h_score = ann.get("scores", {}).get(dim)
                if h_score is not None:
                    llm_scores.append(llm_int)
                    human_scores.append(h_score)
        return llm_scores, human_scores

    def _extract_llm_human_avg(self, dim: str) -> Tuple[List[float], List[float]]:
        """Extract LLM vs averaged-human scores (one pair per conversation)."""
        llm_scores: List[float] = []
        human_scores: List[float] = []
        for conv in self._conversations:
            llm_result = conv.get("llm_scores", {}).get(dim)
            if not llm_result:
                continue
            llm_score = llm_result.get("score", 0)
            h_vals = []
            for ann in conv.get("annotations", []):
                h_score = ann.get("scores", {}).get(dim)
                if h_score is not None:
                    h_vals.append(h_score)
            if h_vals:
                llm_scores.append(round(llm_score, 2))
                human_scores.append(sum(h_vals) / len(h_vals))
        return llm_scores, human_scores

    def _extract_llm_human_filtered_avg(
        self, dim: str, tolerance: int = 1,
    ) -> Tuple[List[float], List[float]]:
        """Extract LLM vs filtered-averaged-human scores.

        For each conversation, only human scores within ±tolerance of each other
        are kept before averaging. If fewer than 2 scores remain (or only 1
        annotator), all available scores are averaged as fallback.
        """
        llm_scores: List[float] = []
        human_scores: List[float] = []
        for conv in self._conversations:
            llm_result = conv.get("llm_scores", {}).get(dim)
            if not llm_result:
                continue
            llm_score = llm_result.get("score", 0)
            h_vals = []
            for ann in conv.get("annotations", []):
                h_score = ann.get("scores", {}).get(dim)
                if h_score is not None:
                    h_vals.append(h_score)
            if not h_vals:
                continue
            if len(h_vals) < 2:
                # Single annotator — use as-is
                filtered = h_vals
            else:
                # Keep scores where all pairwise differences are within tolerance
                filtered = [
                    s for s in h_vals
                    if all(abs(s - o) <= tolerance for o in h_vals if o != s)
                ]
                if not filtered:
                    filtered = h_vals  # fallback if all disagree
            llm_scores.append(round(llm_score, 2))
            human_scores.append(sum(filtered) / len(filtered))
        return llm_scores, human_scores

    def _extract_annotator_pairs(
        self, dim: str, a1: str, a2: str
    ) -> Tuple[List[int], List[int]]:
        """Extract paired scores between two annotators for a dimension."""
        s1, s2 = [], []
        for conv in self._conversations:
            scores_by_annotator: Dict[str, int] = {}
            for ann in conv.get("annotations", []):
                aid = ann.get("annotator_id")
                score = ann.get("scores", {}).get(dim)
                if aid in (a1, a2) and score is not None:
                    scores_by_annotator[aid] = score
            if a1 in scores_by_annotator and a2 in scores_by_annotator:
                s1.append(scores_by_annotator[a1])
                s2.append(scores_by_annotator[a2])
        return s1, s2

    def _build_fleiss_matrix(
        self, dim: str, annotator_list: List[str]
    ) -> List[List[int]]:
        """Build ratings matrix for Fleiss' Kappa."""
        matrix = []
        for conv in self._conversations:
            row = [0] * 5  # categories 1-5
            for ann in conv.get("annotations", []):
                if ann.get("annotator_id") in annotator_list:
                    score = ann.get("scores", {}).get(dim)
                    if score is not None and 1 <= score <= 5:
                        row[score - 1] += 1
            if sum(row) >= 2:
                matrix.append(row)
        return matrix

    @staticmethod
    def format_report(report: AlignmentReport) -> str:
        """Format alignment report for console printing."""
        lines = []
        lines.append("=" * 60)
        lines.append("Alignment Report")
        lines.append("=" * 60)
        lines.append(
            f"Conversations: {report.num_conversations}  "
            f"Annotators: {report.num_annotators}"
        )
        lines.append("")

        # Per-rubric LLM vs Human — Raw
        lines.append("LLM vs Human Agreement — Raw (one pair per annotation):")
        lines.append("-" * 60)
        header = (
            f"{'Dimension':<28} {'Kappa':>7} {'Pearson':>8} "
            f"{'Exact%':>7} {'Adj%':>7}"
        )
        lines.append(header)
        lines.append("-" * 60)
        for dim, m in report.per_rubric.items():
            kappa = f"{m.cohens_kappa_weighted:.3f}" if m.cohens_kappa_weighted is not None else "  N/A"
            pearson = f"{m.pearson_r:.3f}" if m.pearson_r is not None else "  N/A"
            lines.append(
                f"{dim:<28} {kappa:>7} {pearson:>8} "
                f"{m.exact_agreement_pct:>6.1f}% "
                f"{m.adjacent_agreement_pct:>5.1f}%"
            )
        lines.append("")

        # Per-rubric LLM vs Human — Averaged
        if report.per_rubric_avg:
            lines.append("LLM vs Human Agreement — Avg Human (one pair per conversation):")
            lines.append("-" * 60)
            lines.append(header)
            lines.append("-" * 60)
            for dim, m in report.per_rubric_avg.items():
                kappa = f"{m.cohens_kappa_weighted:.3f}" if m.cohens_kappa_weighted is not None else "  N/A"
                pearson = f"{m.pearson_r:.3f}" if m.pearson_r is not None else "  N/A"
                lines.append(
                    f"{dim:<28} {kappa:>7} {pearson:>8} "
                    f"{m.exact_agreement_pct:>6.1f}% "
                    f"{m.adjacent_agreement_pct:>5.1f}%"
                )
            lines.append("")

        # Per-rubric LLM vs Human — Filtered Averaged
        if report.per_rubric_filtered_avg:
            lines.append(
                "LLM vs Human Agreement — Filtered Avg "
                "(±1 inter-annotator filter, then avg):"
            )
            lines.append("-" * 60)
            lines.append(header)
            lines.append("-" * 60)
            for dim, m in report.per_rubric_filtered_avg.items():
                kappa = f"{m.cohens_kappa_weighted:.3f}" if m.cohens_kappa_weighted is not None else "  N/A"
                pearson = f"{m.pearson_r:.3f}" if m.pearson_r is not None else "  N/A"
                lines.append(
                    f"{dim:<28} {kappa:>7} {pearson:>8} "
                    f"{m.exact_agreement_pct:>6.1f}% "
                    f"{m.adjacent_agreement_pct:>5.1f}%"
                )
            lines.append("")

        # Per-rubric LLM vs Human — Filtered Averaged ±2
        if report.per_rubric_filtered_avg_2:
            lines.append(
                "LLM vs Human Agreement — Filtered Avg "
                "(±2 inter-annotator filter, then avg):"
            )
            lines.append("-" * 60)
            lines.append(header)
            lines.append("-" * 60)
            for dim, m in report.per_rubric_filtered_avg_2.items():
                kappa = f"{m.cohens_kappa_weighted:.3f}" if m.cohens_kappa_weighted is not None else "  N/A"
                pearson = f"{m.pearson_r:.3f}" if m.pearson_r is not None else "  N/A"
                lines.append(
                    f"{dim:<28} {kappa:>7} {pearson:>8} "
                    f"{m.exact_agreement_pct:>6.1f}% "
                    f"{m.adjacent_agreement_pct:>5.1f}%"
                )
            lines.append("")

        # Pairwise annotator agreement
        if report.pairwise_annotator_agreement:
            lines.append("Pairwise Inter-Annotator Agreement:")
            lines.append("-" * 60)
            for pair, dims in report.pairwise_annotator_agreement.items():
                lines.append(f"  {pair}:")
                lines.append(
                    f"    {'Dimension':<26} {'Kappa':>7} {'Pearson':>8} "
                    f"{'Exact%':>7} {'Adj%':>6}"
                )
                for dim, pm in dims.items():
                    k = (
                        f"{pm.cohens_kappa_weighted:.3f}"
                        if pm.cohens_kappa_weighted is not None
                        else "N/A"
                    )
                    p = (
                        f"{pm.pearson_r:.3f}"
                        if pm.pearson_r is not None
                        else "N/A"
                    )
                    lines.append(
                        f"    {dim:<26} {k:>7}  "
                        f"{p:>7} "
                        f"{pm.exact_agreement_pct:>6.1f}% "
                        f"{pm.adjacent_agreement_pct:>5.1f}%"
                    )
            lines.append("")

        # Fleiss' Kappa
        if report.fleiss_kappa:
            lines.append("Fleiss' Kappa (multi-rater):")
            lines.append("-" * 60)
            for dim, fk in report.fleiss_kappa.items():
                val = f"{fk:.3f}" if fk is not None else "N/A"
                lines.append(f"  {dim:<28} {val}")
            lines.append("")

        # Realism dimensions summary
        if report.realism_summary:
            lines.append("Realism Assessment (human annotators only):")
            lines.append("-" * 60)
            lines.append(
                f"  {'Dimension':<28} {'Mean':>5} {'Std':>5} "
                f"{'N':>4}  {'Distribution (1-5)':>20}  "
                f"{'IA Kappa':>8} {'IA Adj%':>7}"
            )
            lines.append("-" * 60)
            for dim, rs in report.realism_summary.items():
                dist_str = " ".join(
                    f"{s}:{rs.distribution.get(s, 0)}" for s in range(1, 6)
                )
                kappa = (
                    f"{rs.inter_annotator_kappa:.3f}"
                    if rs.inter_annotator_kappa is not None
                    else "  N/A"
                )
                lines.append(
                    f"  {dim.replace('_', ' '):<28} {rs.mean:>5.2f} "
                    f"{rs.std:>5.2f} {rs.n:>4}  {dist_str:>20}  "
                    f"{kappa:>8} {rs.inter_annotator_adj_pct:>6.1f}%"
                )
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def save_report(report: AlignmentReport, output_path: Path) -> None:
        """Save alignment report as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)

    def generate_alignment_chart(self, output_path: Path) -> None:
        """Generate separate LLM-vs-Clinician agreement scatter charts per mode.

        Produces 4 PNG files (one per aggregation mode), each with 6 subplots
        (one per rubric dimension). Each subplot shows:
        - Perfect agreement diagonal (dashed)
        - ±1 adjacent agreement band (shaded green)
        - Jittered scatter: X = Clinician score, Y = LLM-J score
        - Exact% and ±1% in the legend
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        dims = RUBRIC_DIMENSIONS
        # (key, title, extractor, x-axis label). The x-label reflects whether the
        # human score is a single annotation (raw) or averaged per conversation.
        modes = [
            ("raw", "Raw (one pair per annotation)",
             self._extract_llm_human_pairs, "Human Annotator Score"),
            ("avg", "Avg Human (one pair per conversation)",
             self._extract_llm_human_avg, "Avg Human Annotator Score"),
            (
                "filtered_avg_1",
                "Filtered Avg (±1 inter-annotator filter)",
                lambda d: self._extract_llm_human_filtered_avg(d, 1),
                "Avg Human Annotator Score",
            ),
            (
                "filtered_avg_2",
                "Filtered Avg (±2 inter-annotator filter)",
                lambda d: self._extract_llm_human_filtered_avg(d, 2),
                "Avg Human Annotator Score",
            ),
        ]
        # Per-mode accent hues (validated categorical palette; none green, so
        # they never blend into the green +/-1 agreement band).
        colors = ["#2a78d6", "#4a3aa7", "#eb6834", "#e87ba4"]
        # Ink tokens (text never wears the series color) and chart surface.
        INK_PRIMARY = "#0b0b0b"
        INK_SECONDARY = "#52514e"
        SURFACE = "#fcfcfb"
        GRID = "#52514e"
        BAND_GREEN = "#1baf7a"   # validated aqua/teal "agreement" zone

        # Serif font for LaTeX-paper consistency (matches eval/charts.py).
        old_font = plt.rcParams.get("font.family")
        old_mathtext = plt.rcParams.get("mathtext.fontset")
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        stem = output_path.stem
        suffix = output_path.suffix or ".png"

        for mode_idx, (mode_key, mode_title, extractor, x_label) in enumerate(modes):
            n_cols = 3
            n_rows = math.ceil(len(dims) / n_cols)
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False,
            )
            fig.patch.set_facecolor(SURFACE)

            for idx, dim in enumerate(dims):
                row, col = divmod(idx, n_cols)
                ax = axes[row][col]
                llm_scores, human_scores = extractor(dim)

                if len(llm_scores) < 2:
                    ax.set_title(
                        dim.replace("_", " ").title(),
                        fontsize=12, fontweight="600", color=INK_PRIMARY, pad=8,
                    )
                    ax.text(
                        0.5, 0.5, "Insufficient\ndata", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color=INK_SECONDARY,
                    )
                    ax.set_xlim(1 - 0.18, 5 + 0.18)
                    ax.set_ylim(1 - 0.18, 5 + 0.18)
                    ax.set_facecolor(SURFACE)
                    for side in ("top", "right", "left", "bottom"):
                        ax.spines[side].set_color(INK_SECONDARY)
                        ax.spines[side].set_linewidth(0.8)
                    continue

                h = np.array(human_scores, dtype=float)
                l = np.array(llm_scores, dtype=float)
                n = len(h)
                exact = np.sum(np.abs(np.round(h) - np.round(l)) < 0.5) / n * 100
                adj = np.sum(np.abs(np.round(h) - np.round(l)) <= 1) / n * 100

                # ±1 adjacent-agreement band (semantic "good" green) and the
                # perfect-agreement diagonal on top of it.
                ax.fill_between(
                    [0.5, 5.5], [0.5 - 1, 5.5 - 1], [0.5 + 1, 5.5 + 1],
                    alpha=0.09, color=BAND_GREEN, lw=0, zorder=0,
                )
                ax.plot([0.5, 5.5], [0.5, 5.5], color=INK_SECONDARY, ls=(0, (5, 4)),
                        lw=1.1, alpha=0.55, zorder=1, label="_nolegend_")

                # Small jitter to separate stacked (discrete) scores without
                # straying far across the +/-1 band edge; density still reads
                # through the low point alpha.
                jitter = 0.06
                rng = np.random.default_rng(42 + mode_idx)
                jh = h + rng.uniform(-jitter, jitter, n)
                jl = l + rng.uniform(-jitter, jitter, n)

                # Two-color split by the only thing that matters here: whether a
                # point is inside the +/-1 agreement band. In-band points take the
                # mode accent hue; out-of-band points are a muted neutral gray.
                # Use the SAME continuous geometry as the shaded band
                # (|LLM - human| <= 1) so a dot's color always matches which side
                # of the band it sits on -- NOT the rounded adjacent-agreement test
                # (that stays the legend percentage only). Scores are discrete, so
                # points overplot; a lower alpha lets stacked dots darken with
                # density instead of jittering them off their true position.
                within = np.abs(jl - jh) <= 1
                ax.scatter(
                    jh[within], jl[within], s=38, alpha=0.5, color=colors[mode_idx],
                    edgecolors=SURFACE, linewidths=0.4, zorder=3,
                    label=f"Accuracy (±1) = {adj:.0f}%",
                )
                ax.scatter(
                    jh[~within], jl[~within], s=38, alpha=0.5, color="#9a9992",
                    edgecolors=SURFACE, linewidths=0.4, zorder=2,
                    label="_nolegend_",
                )

                ax.set_facecolor(SURFACE)
                ax.set_title(
                    dim.replace("_", " ").title(),
                    fontsize=12, fontweight="600", color=INK_PRIMARY, pad=8,
                )
                ax.set_xlabel(x_label, fontsize=11.5, color=INK_SECONDARY)
                ax.set_ylabel("LLM-J Score", fontsize=11.5, color=INK_SECONDARY)
                # Clamp to the 1-5 rubric range plus a small margin (3x the
                # jitter) so jittered points at the score extremes aren't clipped.
                margin = 3 * jitter
                ax.set_xlim(1 - margin, 5 + margin)
                ax.set_ylim(1 - margin, 5 + margin)
                ax.set_xticks([1, 2, 3, 4, 5])
                ax.set_yticks([1, 2, 3, 4, 5])
                ax.set_aspect("equal")
                ax.tick_params(colors=INK_SECONDARY, labelsize=9.5, length=3)
                leg = ax.legend(
                    fontsize=8.5, loc="upper left", framealpha=0.95,
                    handletextpad=0.4, borderpad=0.5,
                )
                frame = leg.get_frame()
                frame.set_facecolor(SURFACE)
                frame.set_edgecolor(INK_SECONDARY)
                frame.set_linewidth(0.7)
                for txt in leg.get_texts():
                    txt.set_color(INK_PRIMARY)
                # Recessive grid; keep the full box (all four borders).
                ax.grid(True, alpha=0.15, color=GRID, linewidth=0.6)
                ax.set_axisbelow(True)
                for side in ("top", "right", "left", "bottom"):
                    ax.spines[side].set_color(INK_SECONDARY)
                    ax.spines[side].set_linewidth(0.8)

            # Hide unused subplots
            for idx in range(len(dims), n_rows * n_cols):
                row, col = divmod(idx, n_cols)
                axes[row][col].set_visible(False)

            plt.tight_layout()
            chart_file = output_path.parent / f"{stem}_{mode_key}{suffix}"
            fig.savefig(str(chart_file), dpi=200, bbox_inches="tight")
            plt.close(fig)
            logger.info("Alignment chart (%s) saved to %s", mode_key, chart_file)

        # Restore default font settings.
        plt.rcParams.update({
            "font.family": old_font,
            "mathtext.fontset": old_mathtext,
        })
