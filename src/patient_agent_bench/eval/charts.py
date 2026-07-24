# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Chart generation for benchmark evaluation results.

Produces:
- Heatmaps: model × attribute (severity, task_type, complexity) with safety pass rate
- Grouped bar charts: per-attribute breakdown of rubric pass rates for a single experiment
- Ridgeline charts: per-model score distributions colored by pass rate
- Radar charts: per-model rubric pass rate profiles

All functions take pre-computed data (from stats.py / summary.json) and write PNGs.
No LLM calls, no file loading — pure visualization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from patient_agent_bench.eval.constants import PASS_THRESHOLD, MAX_SCORE, MIN_SCORE
from patient_agent_bench.eval.demographics import derive_age_group, derive_gender_identity

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI use

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns


# Consistent style
PALETTE = "YlOrRd_r"  # Yellow (bad) → Red (worse), reversed so green=good
FIGSIZE_HEATMAP = (12, 6)
FIGSIZE_BAR = (10, 5)
DPI = 300


def generate_heatmap(
    rows: List[str],
    cols: List[str],
    values: List[List[float]],
    title: str,
    output_path: str,
    value_label: str = "Safety Pass Rate (%)",
    vmin: float = 0,
    vmax: float = 100,
    annot_fmt: str = ".0f",
) -> str:
    """
    Generate an annotated heatmap and save as PNG.

    Args:
        rows: Row labels (e.g., model names).
        cols: Column labels (e.g., severity levels).
        values: 2D list of values, shape [len(rows)][len(cols)].
        title: Chart title.
        output_path: Path to save the PNG.
        value_label: Label for the color bar.
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
        annot_fmt: Format string for cell annotations.

    Returns:
        Path to the saved PNG file.
    """
    # Use Times/serif font for LaTeX paper consistency
    old_font = plt.rcParams.get("font.family")
    old_mathtext = plt.rcParams.get("mathtext.fontset")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)

    data = np.array(values, dtype=float)

    sns.heatmap(
        data,
        annot=True,
        fmt=annot_fmt,
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
        xticklabels=cols,
        yticklabels=rows,
        cbar_kws={"label": value_label},
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # Restore default font settings
    plt.rcParams.update({
        "font.family": old_font,
        "mathtext.fontset": old_mathtext,
    })

    return output_path


def generate_grouped_bar_chart(
    groups: List[str],
    series: Dict[str, List[float]],
    title: str,
    output_path: str,
    ylabel: str = "Pass Rate (%)",
    ylim: Tuple[float, float] = (0, 105),
) -> str:
    """
    Generate a grouped bar chart and save as PNG.

    Args:
        groups: X-axis group labels (e.g., severity levels).
        series: Dict mapping series name (rubric) to list of values per group.
        title: Chart title.
        output_path: Path to save the PNG.
        ylabel: Y-axis label.
        ylim: Y-axis limits.

    Returns:
        Path to the saved PNG file.
    """
    # Use Times/serif font for LaTeX paper consistency
    old_font = plt.rcParams.get("font.family")
    old_mathtext = plt.rcParams.get("mathtext.fontset")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)

    n_groups = len(groups)
    n_series = len(series)
    bar_width = 0.8 / max(n_series, 1)
    x = np.arange(n_groups)

    colors = sns.color_palette("Set2", n_series)

    for i, (name, vals) in enumerate(series.items()):
        offset = (i - n_series / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            vals,
            bar_width,
            label=name.replace("_", " ").title(),
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.05,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # Restore default font settings
    plt.rcParams.update({
        "font.family": old_font,
        "mathtext.fontset": old_mathtext,
    })

    return output_path


# ---- High-level functions that work with summary.json data ----


def generate_cross_experiment_heatmaps(
    experiments_summary: Dict[str, Any],
    per_experiment_summaries: Dict[str, Dict[str, Any]],
    output_dir: str,
    metric: str = "clinical_safety",
) -> List[str]:
    """
    Generate model × attribute heatmaps from cross-experiment data.

    Produces three heatmaps:
    - model × severity_level
    - model × task_type
    - model × scenario_complexity

    Each cell shows the average rubric score (1-5) for that model+attribute combination.

    Args:
        experiments_summary: The experiments_summary.json data.
        per_experiment_summaries: Dict mapping experiment_id to its summary.json.
        output_dir: Directory to save PNG files.
        metric: Rubric name to use as the heatmap metric (default: clinical_safety).

    Returns:
        List of paths to generated PNG files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build model name lookup from model_specs
    model_names = {}
    for spec in experiments_summary.get("model_specs", []):
        model_names[spec["experiment_id"]] = spec.get("assistant", spec["experiment_id"])

    # Collect breakdown data per experiment
    breakdown_keys = ["by_severity_level", "by_task_type", "by_scenario_complexity"]
    attribute_labels = {
        "by_severity_level": "Severity Level",
        "by_task_type": "Task Type",
        "by_scenario_complexity": "Scenario Complexity",
    }

    generated = []

    for bk in breakdown_keys:
        # Collect all attribute values across experiments
        all_attr_values = set()
        for exp_id, summary in per_experiment_summaries.items():
            bd = summary.get("breakdowns", {}).get(bk, {})
            all_attr_values.update(bd.keys())

        if not all_attr_values:
            continue

        cols = sorted(all_attr_values)
        rows = []
        values = []

        for exp_id in sorted(per_experiment_summaries.keys()):
            model = model_names.get(exp_id, exp_id)
            summary = per_experiment_summaries[exp_id]
            bd = summary.get("breakdowns", {}).get(bk, {})

            row_vals = []
            for attr_val in cols:
                group = bd.get(attr_val, {})
                averages = group.get("rubric_averages", {})
                row_vals.append(averages.get(metric, 0))

            rows.append(model)
            values.append(row_vals)

        # Clean up column labels
        display_cols = [c.replace("_", " ").title() for c in cols]
        attr_label = attribute_labels.get(bk, bk)
        metric_label = metric.replace("_", " ").title()

        path = str(out / f"heatmap_{bk}_{metric}.png")
        generate_heatmap(
            rows=rows,
            cols=display_cols,
            values=values,
            title=f"{metric_label} Avg Score: Model × {attr_label}",
            output_path=path,
            value_label="Average Score ({MIN_SCORE}-{MAX_SCORE})".format(
                MIN_SCORE=MIN_SCORE, MAX_SCORE=MAX_SCORE
            ),
            vmin=MIN_SCORE + 0.5,
            vmax=MAX_SCORE,
            annot_fmt=".2f",
        )
        generated.append(path)

    return generated


def generate_experiment_breakdown_charts(
    summary: Dict[str, Any],
    output_dir: str,
    experiment_label: str = "",
) -> List[str]:
    """
    Generate grouped bar charts from a single experiment's breakdown data.

    Produces one chart per breakdown attribute (severity, task_type, complexity),
    showing average rubric scores grouped by attribute value.

    Args:
        summary: A single experiment's summary.json data (must have breakdowns).
        output_dir: Directory to save PNG files.
        experiment_label: Label for chart titles (e.g., model name).

    Returns:
        List of paths to generated PNG files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    breakdowns = summary.get("breakdowns", {})
    if not breakdowns:
        return []

    generated = []

    for bk, groups in breakdowns.items():
        if not groups:
            continue

        attr_values = sorted(groups.keys())

        # Collect rubric names from first group
        first_group = groups[attr_values[0]]
        rubric_names = list(first_group.get("rubric_averages", {}).keys())

        if not rubric_names:
            continue

        # Build series: rubric_name -> [avg_for_attr1, avg_for_attr2, ...]
        series = {}
        for rubric in rubric_names:
            series[rubric] = [
                groups[av].get("rubric_averages", {}).get(rubric, 0)
                for av in attr_values
            ]

        display_groups = [v.replace("_", " ").title() for v in attr_values]
        attr_label = bk.replace("by_", "").replace("_", " ").title()
        label = f" ({experiment_label})" if experiment_label else ""

        path = str(out / f"bar_{bk}.png")
        generate_grouped_bar_chart(
            groups=display_groups,
            series=series,
            title=f"Avg Rubric Scores by {attr_label}{label}",
            output_path=path,
            ylabel=f"Average Score ({MIN_SCORE}-{MAX_SCORE})",
            ylim=(0, MAX_SCORE + 0.5),
        )
        generated.append(path)

    return generated


def generate_num_turns_heatmap(
    experiments_summary: Dict[str, Any],
    per_experiment_summaries: Dict[str, Dict[str, Any]],
    output_dir: str,
) -> Optional[str]:
    """
    Generate a model × task_type heatmap showing average num_turns.

    Each cell shows the average number of conversation turns for that
    model + task_type combination, extracted from per-experiment breakdown data.

    Args:
        experiments_summary: The experiments_summary.json data.
        per_experiment_summaries: Dict mapping experiment_id to its summary.json.
        output_dir: Directory to save PNG file.

    Returns:
        Path to the generated PNG, or None if no data available.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build model name lookup
    model_names = {}
    for spec in experiments_summary.get("model_specs", []):
        model_names[spec["experiment_id"]] = spec.get(
            "assistant", spec["experiment_id"]
        )

    # Collect task_type values across experiments
    all_task_types: set = set()
    for summary in per_experiment_summaries.values():
        bd = summary.get("breakdowns", {}).get("by_task_type", {})
        all_task_types.update(bd.keys())

    if not all_task_types:
        return None

    cols = sorted(all_task_types)
    rows = []
    values = []

    for exp_id in sorted(per_experiment_summaries.keys()):
        model = model_names.get(exp_id, exp_id)
        summary = per_experiment_summaries[exp_id]
        bd = summary.get("breakdowns", {}).get("by_task_type", {})

        row_vals = []
        for task_type in cols:
            group = bd.get(task_type, {})
            row_vals.append(group.get("num_turns_avg", 0))

        rows.append(model)
        values.append(row_vals)

    display_cols = [c.replace("_", " ").title() for c in cols]

    # Determine vmax from data
    flat = [v for row in values for v in row]
    vmax = max(flat) + 2 if flat else 20

    path = str(out / "heatmap_num_turns_by_task_type.png")
    generate_heatmap(
        rows=rows,
        cols=display_cols,
        values=values,
        title="Avg Conversation Turns: Model \u00d7 Task Type",
        output_path=path,
        value_label="Average Turns",
        vmin=0,
        vmax=vmax,
        annot_fmt=".1f",
    )
    return path


def _catmull_rom_spline(
    x_pts: np.ndarray, y_pts: np.ndarray, num_points: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a Catmull-Rom spline through discrete points for smooth ridgelines.

    Adds virtual start/end control points so the curve passes through all
    data points. Pure numpy, no scipy needed.

    Args:
        x_pts: X coordinates of control points.
        y_pts: Y coordinates of control points.
        num_points: Number of interpolated output points.

    Returns:
        Tuple of (x_smooth, y_smooth) arrays.
    """
    n = len(x_pts)
    if n < 2:
        return x_pts, y_pts

    # Pad with virtual endpoints for boundary tangents
    xp = np.concatenate([[2 * x_pts[0] - x_pts[1]], x_pts, [2 * x_pts[-1] - x_pts[-2]]])
    yp = np.concatenate([[y_pts[0]], y_pts, [y_pts[-1]]])

    x_out = []
    y_out = []
    segments = n - 1
    pts_per_seg = max(num_points // segments, 4)

    for i in range(1, len(xp) - 2):
        p0, p1, p2, p3 = yp[i - 1], yp[i], yp[i + 1], yp[i + 2]
        x0, x1, x2, x3 = xp[i - 1], xp[i], xp[i + 1], xp[i + 2]

        for t_idx in range(pts_per_seg):
            t = t_idx / pts_per_seg
            t2 = t * t
            t3 = t2 * t

            # Catmull-Rom basis (tension=0.5)
            y_val = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            x_val = 0.5 * (
                (2 * x1)
                + (-x0 + x2) * t
                + (2 * x0 - 5 * x1 + 4 * x2 - x3) * t2
                + (-x0 + 3 * x1 - 3 * x2 + x3) * t3
            )
            y_out.append(max(y_val, 0))  # clamp negative
            x_out.append(x_val)

    # Add final point
    x_out.append(xp[-2])
    y_out.append(max(yp[-2], 0))

    return np.array(x_out), np.array(y_out)


# Token metrics that get their own per-run chart. Each entry:
#   (token_stats key, filename stem, axis/title label)
_TOKEN_METRICS = [
    ("total_output_tokens", "output_tokens", "Output tokens / conversation"),
    ("peak_input_tokens", "peak_input_tokens", "Peak input tokens / conversation"),
    ("reasoning_tokens", "reasoning_tokens", "Reasoning tokens / conversation"),
]


def generate_token_usage_charts(
    experiments_summary: Dict[str, Any],
    per_experiment_summaries: Dict[str, Dict[str, Any]],
    output_dir: str,
) -> List[str]:
    """Per-model token-usage bar charts, one PNG per metric, matching the
    model-spread chart theme (serif font, Set2 palette, sorted models, std
    "shadow" band + whiskers showing per-conversation spread).

    Produces separate charts for output tokens and peak input tokens (very
    different magnitudes, so they must not share an axis), plus reasoning
    tokens only if any model reported them. Bars are sorted descending by mean
    and annotated with the mean; a shaded ±1 std band and error whiskers convey
    spread across conversations. Models with token coverage < 1.0 are flagged.

    Reads ``token_stats`` from each experiment's summary.json. Returns the list
    of generated PNG + caption paths (empty if no token data is available).
    """
    model_names: Dict[str, str] = {}
    exp_order: List[str] = []
    for spec in experiments_summary.get("model_specs", []):
        eid = spec["experiment_id"]
        model_names[eid] = spec.get("assistant", eid)
        exp_order.append(eid)

    # Collect token_stats for experiments that have them.
    stats_by_exp: Dict[str, Dict[str, Any]] = {}
    for eid in exp_order:
        summ = per_experiment_summaries.get(eid)
        ts = summ.get("token_stats") if summ else None
        if ts and ts.get("conversations_measured", 0) > 0:
            stats_by_exp[eid] = ts
    measured = [e for e in exp_order if e in stats_by_exp]
    if not measured:
        return []

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    old_font = plt.rcParams.get("font.family")
    old_mathtext = plt.rcParams.get("mathtext.fontset")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    generated: List[str] = []
    palette = sns.color_palette("Set2", 8)

    for key, stem, label in _TOKEN_METRICS:
        # Skip reasoning chart entirely unless some model reported reasoning.
        if key == "reasoning_tokens" and not any(
            stats_by_exp[e]["reasoning_tokens"]["mean"] > 0 for e in measured
        ):
            continue

        # Sort models descending by median (the charted central value).
        order = sorted(measured, key=lambda e: stats_by_exp[e][key]["median"], reverse=True)
        names = [model_names.get(e, e) for e in order]
        metrics = [stats_by_exp[e][key] for e in order]
        medians = [mt["median"] for mt in metrics]
        covs = [stats_by_exp[e].get("usage_coverage", 1.0) for e in order]
        n_models = len(order)

        fig, ax = plt.subplots(figsize=(max(10, n_models * 0.9 + 3), 5.5))
        x = np.arange(n_models)

        # Token usage is right-skewed, so summarize the distribution with median
        # + IQR (p25-p75) band + p10-p90 whiskers (a box-plot-style view), which
        # is robust to the runaway-conversation outliers that distort mean/std.
        y_top = max(mt["p90"] for mt in metrics) * 1.18
        label_pad = y_top * 0.02

        for xi, (mt, cov) in enumerate(zip(metrics, covs)):
            color = palette[xi % len(palette)]
            med = mt["median"]
            p10, p25, p75, p90 = mt["p10"], mt["p25"], mt["p75"], mt["p90"]
            # IQR band (p25-p75), behind everything.
            ax.bar(
                xi, p75 - p25, width=0.5, bottom=p25,
                color=color, alpha=0.25, edgecolor="none", zorder=1,
            )
            # p10-p90 whisker with caps, same color.
            ax.vlines(xi, p10, p90, colors=color, linewidth=1.6, zorder=2)
            for yv in (p10, p90):
                ax.hlines(yv, xi - 0.12, xi + 0.12, colors=color, linewidth=1.6, zorder=2)
            # Median marker (dot).
            ax.scatter(
                xi, med, s=75, color=color, zorder=3,
                edgecolors="white", linewidths=0.9,
            )
            # Median value label, padded above the p90 cap.
            ax.annotate(
                f"{med:,.0f}", (xi, p90 + label_pad), ha="center", va="bottom",
                fontsize=8, fontweight="bold", zorder=4,
            )
            if cov < 1.0:
                ax.annotate(
                    f"cov {cov:.0%}", (xi, 0), ha="center", va="bottom",
                    fontsize=6.5, color="#b00020", fontweight="bold", zorder=4,
                )

        ax.set_ylim(0, y_top)

        # Legend explaining the encoding (neutral gray proxies; marker/band
        # colors only distinguish models, they carry no other meaning).
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="#7f7f7f", markeredgecolor="white",
                   markersize=8, label="Median (dot + number label above)"),
            Patch(facecolor="#7f7f7f", alpha=0.25,
                  label="IQR: p25–p75 (middle 50% of conversations)"),
            Line2D([0], [0], color="#7f7f7f", linewidth=1.6,
                   label="Whisker: p10–p90"),
        ]
        ax.legend(handles=legend_handles, loc="upper right",
                  fontsize=8, framealpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(
            f"{label} by model",
            fontsize=13, fontweight="bold", pad=10,
        )

        plt.tight_layout()
        png_path = str(out / f"token_{stem}.png")
        fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        generated.append(png_path)

        # ---- Caption: full "how to read" + complete per-model data table ----
        low_cov = [(names[i], covs[i]) for i in range(n_models) if covs[i] < 1.0]
        cap = [f"## {label} by model\n"]
        cap.append(
            f"Per-conversation **{label.lower()}**, one marker per model, sorted "
            "descending by median. Token usage is right-skewed, so the chart "
            "uses a robust box-plot-style summary (median + IQR + p10–p90) "
            "rather than mean±std. Marker colors only distinguish models (no "
            "other meaning). Values are tokens per conversation.\n"
        )
        cap.append("**How to read this chart:**\n")
        cap.append(
            "- **Dot** = median (p50) across all measured conversations — the "
            "typical conversation, robust to outliers.\n"
            "- **Number above the dot** = that median value.\n"
            "- **Shaded band** = interquartile range (IQR), p25–p75: the middle "
            "50% of conversations.\n"
            "- **Whisker** = p10–p90: the middle 80% of conversations.\n"
            "- Together these show the **distribution / spread** of "
            "per-conversation values across models. (Mean, std, and the 95% CI "
            "of the mean are in the table below for reference.)\n"
            "- **Marker/band color** only distinguishes models; it has no other "
            "meaning.\n"
            "- **Red `cov NN%` at the base** (only if present) = token-usage "
            "coverage below 100%: some assistant messages lacked "
            "`usage_metadata`, so that model's numbers are under-counts.\n"
        )
        cap.append(
            "\n_median/p10/p25/p75/p90 = percentiles across conversations "
            "(distribution). mean = arithmetic average (skewed upward by "
            "outliers). std = standard deviation. CI = 95% confidence interval "
            "of the mean (mean ± 1.96·std/√n) — the precision of the mean, far "
            "tighter than the distribution at n≈1200. n = conversations "
            "measured._\n"
        )
        cap.append(f"\n### {label} — per-model statistics\n")
        cap.append(
            "| Model | Median | p10 | p25 | p75 | p90 | Max | Mean | Std | "
            "CI (mean) | n | Coverage |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        )
        for i, e in enumerate(order):
            mt = metrics[i]
            ci = mt.get("ci", [0, 0])
            nconv = stats_by_exp[e].get("conversations_measured", 0)
            cap.append(
                f"| {names[i]} | {mt['median']:,.0f} | {mt['p10']:,.0f} "
                f"| {mt['p25']:,.0f} | {mt['p75']:,.0f} | {mt['p90']:,.0f} "
                f"| {mt['max']:,.0f} | {mt['mean']:,.0f} | {mt.get('std', 0):,.0f} "
                f"| [{ci[0]:,.0f}, {ci[1]:,.0f}] | {nconv:,} | {covs[i]:.0%} |\n"
            )
        cap.append(f"\n- **Highest median**: {names[0]} ({medians[0]:,.0f})\n")
        cap.append(f"- **Lowest median**: {names[-1]} ({medians[-1]:,.0f})\n")
        if low_cov:
            flagged = ", ".join(f"{n} ({c:.0%})" for n, c in low_cov)
            cap.append(
                f"- ⚠️ **Incomplete token coverage**: {flagged}. "
                "These values are under-counts.\n"
            )
        else:
            cap.append("- **Token coverage: 100%** for all models "
                       "(every assistant message reported usage).\n")
        generated.append(_write_caption(png_path, "".join(cap)))

    plt.rcParams.update({
        "font.family": old_font,
        "mathtext.fontset": old_mathtext,
    })
    return generated


def generate_ridgeline_chart(
    experiments_summary: Dict[str, Any],
    per_experiment_summaries: Dict[str, Dict[str, Any]],
    output_path: str,
    rubrics: Optional[List[str]] = None,
    pass_threshold: int = PASS_THRESHOLD,
    max_score: int = MAX_SCORE,
) -> str:
    """
    Generate a multi-column ridgeline chart: one column per rubric, rows = models.

    First column is "Aggregate Score" (weighted average on 1-5 scale), separated
    from individual rubric columns by a dashed vertical line. Each ridge shows
    the score distribution (1-5) as a smooth spline profile, filled with a color
    encoding the pass rate (green=high, red=low). A marker with vertical line
    and score+pass-rate label indicates the average.

    Uses only matplotlib + numpy (no scipy or extra libs).

    Args:
        experiments_summary: The experiments_summary.json data.
        per_experiment_summaries: Dict mapping experiment_id to its summary.json.
        output_path: Path to save the PNG.
        rubrics: Optional list of rubric names to include. If None, auto-detected.
        pass_threshold: Score >= this counts as pass (default 3).
        max_score: Maximum score value (default 5).

    Returns:
        Path to the saved PNG file.
    """
    # Use Times/serif font for LaTeX paper consistency
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    # Build model name lookup
    model_names = {}
    exp_order = []
    for spec in experiments_summary.get("model_specs", []):
        eid = spec["experiment_id"]
        model_names[eid] = spec.get("assistant", eid)
        exp_order.append(eid)

    # Filter to experiments that have summaries
    exp_order = [e for e in exp_order if e in per_experiment_summaries]
    if not exp_order:
        return output_path

    # Auto-detect rubrics from first experiment's case_results
    first_summary = per_experiment_summaries[exp_order[0]]
    case_results = first_summary.get("case_results", [])
    if not case_results:
        return output_path
    first_case = case_results[0]
    if rubrics is None:
        rubrics = [
            k for k in first_case.keys()
            if k not in ("case_id", "aggregate_score", "num_turns")
        ]

    # Preferred column order for rubrics
    preferred_order = [
        "clinical_safety",
        "triage_quality",
        "workflow_accuracy",
        "task_completion",
        "clinical_helpfulness",
        "conversational_quality",
    ]
    rubrics = [r for r in preferred_order if r in rubrics] + [
        r for r in rubrics if r not in preferred_order
    ]

    # All columns: aggregate_score first, then individual rubrics
    all_columns = ["aggregate_score"] + rubrics
    n_columns = len(all_columns)
    n_models = len(exp_order)

    if n_columns < 2 or n_models == 0:
        return output_path

    # Collect per-case scores for all columns (1-5 scale)
    scores: Dict[str, Dict[str, List[float]]] = {}
    for eid in exp_order:
        summary = per_experiment_summaries[eid]
        scores[eid] = {}
        for col in all_columns:
            scores[eid][col] = [
                case[col]
                for case in summary.get("case_results", [])
                if case.get(col) is not None
            ]

    # Collect pass rates and averages
    pass_rates: Dict[str, Dict[str, float]] = {}
    col_avgs: Dict[str, Dict[str, float]] = {}
    for eid in exp_order:
        summary = per_experiment_summaries[eid]
        pr = summary.get("rubric_pass_rates", {})
        ra = summary.get("rubric_averages", {})
        agg = summary.get("aggregate_metrics", {})

        pass_rates[eid] = {}
        col_avgs[eid] = {}
        for col in all_columns:
            if col == "aggregate_score":
                agg_scores = scores[eid][col]
                if agg_scores:
                    pass_rates[eid][col] = (
                        sum(1 for s in agg_scores if s >= pass_threshold)
                        / len(agg_scores) * 100
                    )
                else:
                    pass_rates[eid][col] = 0
                col_avgs[eid][col] = agg.get("average_score", 0)
            else:
                pass_rates[eid][col] = pr.get(col, 0)
                col_avgs[eid][col] = ra.get(col, 0)

    # Rank models by aggregate score (highest at top of chart)
    # Higher row_idx = higher on y-axis, so sort ascending
    exp_order.sort(
        key=lambda eid: col_avgs[eid].get("aggregate_score", 0),
    )

    # Layout with gridspec: aggregate col | separator gap | rubric cols | colorbar
    ridge_height = 0.38
    row_spacing = 0.58
    col_width = 2.4
    sep_width = 0.15  # gap for separator
    cbar_width = 0.3  # narrow colorbar
    n_rubric_cols = len(rubrics)

    fig_width = col_width + sep_width + col_width * n_rubric_cols + cbar_width + 0.8
    fig_height = max(row_spacing * n_models + 1.4, 4)

    # gridspec: [agg_col, sep, rubric_0, rubric_1, ..., cbar]
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        1, n_columns + 2,  # +1 for separator, +1 for colorbar
        figure=plt.figure(figsize=(fig_width, fig_height)),
        width_ratios=(
            [col_width]          # aggregate
            + [sep_width]        # separator
            + [col_width] * n_rubric_cols  # rubrics
            + [cbar_width]       # colorbar
        ),
        wspace=0.08,
    )
    fig = gs.figure

    # Create axes: map column index in all_columns to gridspec position
    axes = []
    for i in range(n_columns):
        gs_idx = i if i == 0 else i + 1  # skip separator slot
        ax = fig.add_subplot(gs[0, gs_idx])
        axes.append(ax)

    # Colorbar axis
    cbar_ax = fig.add_subplot(gs[0, -1])

    # Colormap: red (low pass rate) -> yellow -> green (high pass rate)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "modern_pass", ["#e63946", "#f4a261", "#a29bfe", "#74b9ff", "#2a9d8f", "#00b894"]
    )
    norm = mcolors.Normalize(vmin=50, vmax=100)

    score_bins = np.arange(1, max_score + 1, dtype=float)

    for col_idx, col_name in enumerate(all_columns):
        ax = axes[col_idx]

        for row_idx, eid in enumerate(exp_order):
            data = scores[eid][col_name]
            pr = pass_rates[eid].get(col_name, 0)
            avg = col_avgs[eid].get(col_name, 0)
            color = cmap(norm(pr))
            y_base = row_idx * row_spacing

            if not data:
                continue

            # Build histogram: bin continuous aggregate scores into 1-5 integer bins
            total = len(data)
            counts = np.zeros(max_score)  # indices 0..4 map to scores 1..5
            for val in data:
                idx = int(round(val)) - 1  # shift to 0-based index
                idx = max(0, min(max_score - 1, idx))
                counts[idx] += 1
            fractions = counts / total
            heights = fractions * ridge_height

            # Smooth spline
            x_smooth, y_smooth = _catmull_rom_spline(score_bins, heights)

            ax.fill_between(
                x_smooth, y_base, y_base + y_smooth,
                color=color, alpha=0.85, linewidth=0,
            )
            ax.plot(
                x_smooth, y_base + y_smooth,
                color="black", linewidth=0.5, alpha=0.4,
            )

            # Average score: vertical line + marker + label with pass rate
            ax.vlines(
                avg, y_base, y_base + ridge_height + 0.02,
                colors="#333333", linewidths=0.6, alpha=0.7, zorder=5,
            )
            ax.plot(
                avg, y_base + ridge_height + 0.04,
                marker="v", markersize=3.5, color="#333333", zorder=6,
            )
            ax.text(
                avg - 0.12, y_base + ridge_height + 0.02,
                f"{avg:.2f} [{pr:.0f}%]",
                fontsize=8.5, fontweight="bold",
                va="center", ha="right", color="#333333", zorder=6,
            )

        # Axis formatting
        ax.set_xlim(0.5, max_score + 0.5)
        ax.set_xticks(score_bins)
        ax.set_xticklabels([str(int(s)) for s in score_bins], fontsize=9)
        display_name = (
            "Aggregate Score" if col_name == "aggregate_score"
            else col_name.replace("_", " ").title()
        )
        ax.set_title(display_name, fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("")
        ax.grid(axis="x", alpha=0.12, linewidth=0.4)
        ax.set_axisbelow(True)

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col_idx > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False)

    # Set consistent y-limits on all axes
    y_min = -0.15
    y_max = (n_models - 1) * row_spacing + ridge_height + 0.25
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    # Y-axis labels only on leftmost axis
    y_positions = [i * row_spacing for i in range(n_models)]
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(
        [model_names[eid] for eid in exp_order], fontsize=10,
    )
    # Hide y-tick labels on all other axes
    for ax in axes[1:]:
        ax.set_yticks(y_positions)
        ax.set_yticklabels([])

    # Dashed separator line between aggregate column and rubric columns
    # Get the right edge of aggregate axis and left edge of first rubric axis
    fig.canvas.draw()  # force layout computation
    agg_bbox = axes[0].get_position()
    rub_bbox = axes[1].get_position()
    sep_x = (agg_bbox.x1 + rub_bbox.x0) / 2
    sep_line = plt.Line2D(
        [sep_x, sep_x], [agg_bbox.y0, agg_bbox.y1],
        transform=fig.transFigure, color="#888888",
        linewidth=1.0, linestyle="--", alpha=0.6,
    )
    fig.add_artist(sep_line)

    # Subtle hue background on aggregate score column
    axes[0].set_facecolor("#f0f4ff")
    # Border via axes spines
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#c0c8e0")
        spine.set_linewidth(0.8)
        spine.set_visible(True)

    # Vertical colorbar for pass rate
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Pass Rate (%)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # Restore default font settings
    plt.rcParams.update({
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
    })

    return output_path


def _write_caption(path: str, content: str) -> str:
    """Write a caption markdown file next to a chart PNG."""
    md_path = path.rsplit(".", 1)[0] + ".md"
    Path(md_path).write_text(content, encoding="utf-8")
    return md_path


def generate_sensitivity_charts(
    experiments_summary: Dict[str, Any],
    per_experiment_summaries: Dict[str, Dict[str, Any]],
    output_dir: str,
    attributes: Optional[List[str]] = None,
    metric: str = "aggregate_score_avg",
) -> List[str]:
    """
    Generate strip-plot charts showing per-attribute score spread for each model.

    X-axis: models sorted by overall average (left=best).
    Y-axis: score.
    Circle size proportional to sample count per attribute value.
    Each model column shows scatter points connected by a vertical
    range line, with a diamond baseline marker.

    One PNG + caption .md per attribute.

    Args:
        experiments_summary: The experiments_summary.json data.
        per_experiment_summaries: Dict mapping experiment_id to its summary.json.
        output_dir: Directory to save PNG files.
        attributes: Breakdown attributes to chart.
        metric: "aggregate_score_avg" or a rubric name.

    Returns:
        List of paths to generated files (PNGs + MDs).
    """
    if attributes is None:
        attributes = ["severity_level", "scenario_complexity", "personality"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_names: Dict[str, str] = {}
    exp_order: List[str] = []
    for spec in experiments_summary.get("model_specs", []):
        eid = spec["experiment_id"]
        model_names[eid] = spec.get("assistant", eid)
        exp_order.append(eid)
    exp_order = [e for e in exp_order if e in per_experiment_summaries]
    if not exp_order:
        return []

    baselines: Dict[str, float] = {}
    for eid in exp_order:
        summary = per_experiment_summaries[eid]
        if metric == "aggregate_score_avg":
            baselines[eid] = summary.get("aggregate_metrics", {}).get(
                "average_score", 0
            )
        else:
            baselines[eid] = summary.get("rubric_averages", {}).get(metric, 0)

    exp_order.sort(key=lambda e: baselines[e], reverse=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    generated: List[str] = []

    for attr in attributes:
        bk = f"by_{attr}"

        all_vals: set = set()
        for eid in exp_order:
            bd = per_experiment_summaries[eid].get("breakdowns", {}).get(bk, {})
            all_vals.update(bd.keys())
        if not all_vals:
            continue

        attr_vals = sorted(all_vals)
        n_vals = len(attr_vals)
        n_models = len(exp_order)

        attr_palette = sns.color_palette("Set2", n_vals)
        attr_colors = {v: attr_palette[i] for i, v in enumerate(attr_vals)}

        # Collect all sample counts to scale circle sizes
        all_ns: List[int] = []
        for eid in exp_order:
            bd = per_experiment_summaries[eid].get("breakdowns", {}).get(bk, {})
            for av in attr_vals:
                all_ns.append(bd.get(av, {}).get("n", 1))
        max_n = max(all_ns) if all_ns else 1

        fig, ax = plt.subplots(
            figsize=(max(10, n_models * 0.9 + 3), 5.5)
        )

        x_positions = np.arange(n_models)
        all_scores: List[float] = []
        # For caption
        per_model_data: Dict[str, Dict[str, Tuple[float, int]]] = {}

        for xi, eid in enumerate(exp_order):
            bd = per_experiment_summaries[eid].get("breakdowns", {}).get(bk, {})
            baseline = baselines[eid]
            name = model_names.get(eid, eid)
            per_model_data[name] = {}

            scores_for_model: List[float] = []
            for av in attr_vals:
                group = bd.get(av, {})
                if metric == "aggregate_score_avg":
                    s = group.get("aggregate_score_avg", baseline)
                else:
                    s = group.get("rubric_averages", {}).get(metric, baseline)
                n = group.get("n", 1)
                scores_for_model.append(s)
                per_model_data[name][av] = (s, n)

            all_scores.extend(scores_for_model)
            all_scores.append(baseline)

            y_arr = np.array(scores_for_model)

            if len(y_arr) > 1:
                ax.vlines(
                    xi, y_arr.min(), y_arr.max(),
                    colors="#cccccc", linewidths=2.5, zorder=1,
                )

            for j, av in enumerate(attr_vals):
                n = bd.get(av, {}).get("n", 1)
                size = 25 + 75 * (n / max_n)  # scale 25-100
                ax.scatter(
                    xi, scores_for_model[j],
                    color=attr_colors[av], s=size, zorder=3,
                    edgecolors="white", linewidths=0.5,
                    label=av.replace("_", " ").title() if xi == 0 else "",
                )

            ax.scatter(
                xi, baseline, marker="D", s=70, color="#333333",
                zorder=4, edgecolors="white", linewidths=0.8,
                label="Overall Avg" if xi == 0 else "",
            )

        if all_scores:
            y_lo = min(all_scores) - 0.15
            y_hi = max(all_scores) + 0.15
            if y_hi - y_lo < 0.5:
                mid = (y_hi + y_lo) / 2
                y_lo, y_hi = mid - 0.25, mid + 0.25
            ax.set_ylim(y_lo, y_hi)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [model_names.get(e, e) for e in exp_order],
            rotation=35, ha="right", fontsize=8.5,
        )
        ax.set_ylabel("Score", fontsize=10)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        metric_label = (
            "Aggregate Score"
            if metric == "aggregate_score_avg"
            else metric.replace("_", " ").title()
        )
        attr_label = attr.replace("_", " ").title()
        ax.set_title(
            f"{metric_label} Spread by {attr_label}",
            fontsize=13, fontweight="bold", pad=10,
        )

        ax.legend(
            fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5),
            framealpha=0.9, borderaxespad=0,
        )

        plt.tight_layout()
        png_path = str(out / f"sensitivity_{attr}_{metric}.png")
        fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        generated.append(png_path)

        # Caption
        top_model = model_names.get(exp_order[0], exp_order[0])
        top_data = per_model_data.get(top_model, {})
        best_val = max(top_data, key=lambda k: top_data[k][0]) if top_data else ""
        worst_val = min(top_data, key=lambda k: top_data[k][0]) if top_data else ""

        # Widest and narrowest spread
        spreads = []
        for eid in exp_order:
            name = model_names.get(eid, eid)
            vals = [v for v, _ in per_model_data.get(name, {}).values()]
            if vals:
                spreads.append((name, max(vals) - min(vals)))
        spreads.sort(key=lambda t: t[1], reverse=True)

        caption = f"## {metric_label} Spread by {attr_label}\n\n"
        caption += (
            f"Models sorted left-to-right by overall {metric_label.lower()}. "
            f"Each dot is a {attr_label.lower()} value; "
            f"circle size reflects sample count. "
            f"Diamond = model's overall average.\n\n"
        )
        if spreads:
            caption += f"- **Widest spread**: {spreads[0][0]} ({spreads[0][1]:.2f})\n"
            caption += f"- **Narrowest spread**: {spreads[-1][0]} ({spreads[-1][1]:.2f})\n"
        if best_val and worst_val and top_data:
            caption += (
                f"- **{top_model}** peaks on "
                f"*{best_val.replace('_', ' ')}* ({top_data[best_val][0]:.2f}), "
                f"dips on *{worst_val.replace('_', ' ')}* ({top_data[worst_val][0]:.2f})\n"
            )

        md_path = _write_caption(png_path, caption)
        generated.append(md_path)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
    })

    return generated


def generate_sensitivity_scatter(
    experiments_summary: Dict[str, Any],
    per_experiment_summaries: Dict[str, Dict[str, Any]],
    output_dir: str,
    attributes: Optional[List[str]] = None,
    metric: str = "aggregate_score_avg",
) -> List[str]:
    """
    Generate avg-vs-spread scatter plots with directional axis arrows.

    X-axis: model's overall average score (arrow: higher = better).
    Y-axis: score range across attribute values.
    Color encodes overall pass rate (green=high, red=low).
    One PNG + caption .md per attribute.

    Args:
        experiments_summary: The experiments_summary.json data.
        per_experiment_summaries: Dict mapping experiment_id to its summary.json.
        output_dir: Directory to save PNG files.
        attributes: Breakdown attributes.
        metric: "aggregate_score_avg" or a rubric name.

    Returns:
        List of paths to generated files (PNGs + MDs).
    """
    if attributes is None:
        attributes = ["severity_level", "scenario_complexity", "personality"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_names: Dict[str, str] = {}
    exp_order: List[str] = []
    for spec in experiments_summary.get("model_specs", []):
        eid = spec["experiment_id"]
        model_names[eid] = spec.get("assistant", eid)
        exp_order.append(eid)
    exp_order = [e for e in exp_order if e in per_experiment_summaries]
    if not exp_order:
        return []

    baselines: Dict[str, float] = {}
    pass_rates_overall: Dict[str, float] = {}
    for eid in exp_order:
        summary = per_experiment_summaries[eid]
        if metric == "aggregate_score_avg":
            baselines[eid] = summary.get("aggregate_metrics", {}).get(
                "average_score", 0
            )
        else:
            baselines[eid] = summary.get("rubric_averages", {}).get(metric, 0)
        # Average pass rate across all rubrics for color encoding
        pr = summary.get("rubric_pass_rates", {})
        pass_rates_overall[eid] = (
            sum(pr.values()) / len(pr) if pr else 0
        )

    y_arrow_up_good = {"scenario_complexity"}

    # Custom color spectrum: pink → purple → blue → teal → green
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "pass_rate", ["#e84393", "#a29bfe", "#74b9ff", "#00cec9", "#00b894"]
    )
    pr_values = list(pass_rates_overall.values())
    norm = mcolors.Normalize(
        vmin=min(pr_values) - 5 if pr_values else 50,
        vmax=max(pr_values) + 5 if pr_values else 100,
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    generated: List[str] = []

    for attr in attributes:
        bk = f"by_{attr}"

        xs: List[float] = []
        ys: List[float] = []
        colors: List[Any] = []
        labels: List[str] = []
        per_model_scores: Dict[str, Dict[str, float]] = {}

        for eid in exp_order:
            bd = per_experiment_summaries[eid].get("breakdowns", {}).get(bk, {})
            if not bd:
                continue
            scores: Dict[str, float] = {}
            for val, group in bd.items():
                if metric == "aggregate_score_avg":
                    scores[val] = group.get("aggregate_score_avg", 0)
                else:
                    scores[val] = group.get("rubric_averages", {}).get(metric, 0)
            if not scores:
                continue
            vals = list(scores.values())
            name = model_names.get(eid, eid)
            xs.append(baselines[eid])
            ys.append(max(vals) - min(vals))
            colors.append(cmap(norm(pass_rates_overall[eid])))
            labels.append(name)
            per_model_scores[name] = scores

        if not xs:
            continue

        up_is_good = attr in y_arrow_up_good

        fig, ax = plt.subplots(figsize=(9, 7))

        # Compute data bounds with padding
        x_range = max(xs) - min(xs) or 0.1
        y_range = max(ys) - min(ys) or 0.01
        x_lo = min(xs) - x_range * 0.12
        x_hi = max(xs) + x_range * 0.20
        y_lo = min(ys) - y_range * 0.18
        y_hi = max(ys) + y_range * 0.20

        ax.scatter(
            xs, ys, s=220, c=colors, zorder=4,
            edgecolors="white", linewidths=1.5,
        )

        for i, label in enumerate(labels):
            ax.annotate(
                label, (xs[i], ys[i]),
                textcoords="offset points", xytext=(10, 10),
                fontsize=12, fontweight="bold", zorder=5,
            )

        # Remove all spines and ticks
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(
            left=False, bottom=False,
            labelleft=False, labelbottom=False,
        )

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

        metric_label = (
            "Aggregate Score"
            if metric == "aggregate_score_avg"
            else metric.replace("_", " ").title()
        )
        attr_label = attr.replace("_", " ").title()

        if up_is_good:
            y_arrow_label = "More Discriminating"
        else:
            y_arrow_label = "More Sensitive"

        # Quadrant labels — descriptive, no "best"
        x_med = float(np.median(xs))
        y_med = float(np.median(ys))
        if up_is_good:
            q_labels = {
                "tl": "Sensitive &\nLower Scoring",
                "tr": "Discriminating &\nHigher Scoring",
                "bl": "Flat &\nLower Scoring",
                "br": "Flat &\nHigher Scoring",
            }
        else:
            q_labels = {
                "tl": "Sensitive &\nLower Scoring",
                "tr": "Sensitive &\nHigher Scoring",
                "bl": "Consistent &\nLower Scoring",
                "br": "Consistent &\nHigher Scoring",
            }
        q_props = dict(
            fontsize=10, alpha=0.30, ha="center", va="center",
            fontstyle="italic", zorder=1, color="#333333",
        )
        ax.text((x_lo + x_med) / 2, (y_med + y_hi) / 2,
                q_labels["tl"], **q_props)
        ax.text((x_med + x_hi) / 2, (y_med + y_hi) / 2,
                q_labels["tr"], **q_props)
        ax.text((x_lo + x_med) / 2, (y_lo + y_med) / 2,
                q_labels["bl"], **q_props)
        ax.text((x_med + x_hi) / 2, (y_lo + y_med) / 2,
                q_labels["br"], **q_props)

        # Quadrant divider lines
        ax.axvline(x_med, color="#cccccc", linewidth=0.8,
                    linestyle="--", zorder=1)
        ax.axhline(y_med, color="#cccccc", linewidth=0.8,
                    linestyle="--", zorder=1)

        # Arrow axes in data coordinates
        arrow_kw = dict(
            arrowstyle="-|>", color="#444444", lw=2.0,
            mutation_scale=18,
        )
        ax.annotate(
            "", xy=(x_hi, y_lo), xytext=(x_lo, y_lo),
            arrowprops=arrow_kw, annotation_clip=False,
        )
        ax.annotate(
            "", xy=(x_lo, y_hi), xytext=(x_lo, y_lo),
            arrowprops=arrow_kw, annotation_clip=False,
        )

        # X-axis arrow label at tip
        ax.text(
            x_hi, y_lo - y_range * 0.06,
            f"Higher {metric_label}  \u2192",
            ha="right", va="top", fontsize=12, fontweight="bold",
            color="#444444",
        )

        # Y-axis arrow label — vertical, at tip
        ax.text(
            x_lo - x_range * 0.06, y_hi,
            y_arrow_label,
            ha="right", va="top", fontsize=12, fontweight="bold",
            color="#444444", rotation=90,
        )

        # Light reference grid
        ax.grid(alpha=0.10, linewidth=0.3)
        ax.set_axisbelow(True)

        # Colorbar for pass rate
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Avg Pass Rate (%)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        plt.tight_layout()
        png_path = str(out / f"avg_vs_spread_{attr}_{metric}.png")
        fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        generated.append(png_path)

        # Per-chart caption
        ranked = sorted(zip(xs, ys, labels), key=lambda t: t[0], reverse=True)
        most_sensitive = max(zip(ys, labels), key=lambda t: t[0])
        least_sensitive = min(zip(ys, labels), key=lambda t: t[0])
        best_model = ranked[0][2]
        best_avg = ranked[0][0]

        caption = f"## {metric_label}: Average vs {attr_label} Sensitivity\n\n"
        caption += (
            f"**X-axis**: overall {metric_label.lower()} (higher = better). "
            f"**Y-axis**: score range across {attr_label.lower()} values "
        )
        if up_is_good:
            caption += "(higher = better discrimination of difficulty levels). "
        else:
            caption += "(lower = more consistent across variations). "
        caption += "**Color**: average pass rate.\n\n"

        caption += f"- **Highest scoring**: {best_model} ({best_avg:.2f})\n"
        caption += (
            f"- **Most sensitive**: {most_sensitive[1]} "
            f"(range {most_sensitive[0]:.2f})\n"
        )
        caption += (
            f"- **Least sensitive**: {least_sensitive[1]} "
            f"(range {least_sensitive[0]:.2f})\n"
        )

        # Per-attribute-value scores for top model
        if best_model in per_model_scores:
            ms = per_model_scores[best_model]
            caption += f"\n**{best_model}** scores by {attr_label.lower()}:\n\n"
            for av in sorted(ms, key=ms.get, reverse=True):  # type: ignore[arg-type]
                caption += (
                    f"| {av.replace('_', ' ').title()} | {ms[av]:.2f} |\n"
                )

        # Per-attribute-value scores for most sensitive model
        ms_name = most_sensitive[1]
        if ms_name != best_model and ms_name in per_model_scores:
            ms2 = per_model_scores[ms_name]
            caption += f"\n**{ms_name}** scores by {attr_label.lower()}:\n\n"
            for av in sorted(ms2, key=ms2.get, reverse=True):  # type: ignore[arg-type]
                caption += (
                    f"| {av.replace('_', ' ').title()} | {ms2[av]:.2f} |\n"
                )

        md_path = _write_caption(png_path, caption)
        generated.append(md_path)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
    })

    return generated


def generate_attribute_difficulty_charts(
    experiments_summary: Dict[str, Any],
    per_experiment_summaries: Dict[str, Dict[str, Any]],
    output_dir: str,
    attributes: Optional[List[str]] = None,
    metric: str = "aggregate_score_avg",
) -> List[str]:
    """
    Generate horizontal bar charts showing average score per attribute value,
    pooled across all models. Answers "which attribute values are hardest?"

    One PNG + caption .md per attribute.

    Args:
        experiments_summary: The experiments_summary.json data.
        per_experiment_summaries: Dict mapping experiment_id to its summary.json.
        output_dir: Directory to save PNG files.
        attributes: Breakdown attributes.
        metric: "aggregate_score_avg" or a rubric name.

    Returns:
        List of paths to generated files.
    """
    if attributes is None:
        attributes = ["severity_level", "scenario_complexity", "personality"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    exp_ids = [
        s["experiment_id"] for s in experiments_summary.get("model_specs", [])
        if s["experiment_id"] in per_experiment_summaries
    ]
    if not exp_ids:
        return []

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    generated: List[str] = []

    for attr in attributes:
        bk = f"by_{attr}"

        # Collect scores per attribute value across all models
        val_scores: Dict[str, List[float]] = {}
        for eid in exp_ids:
            bd = per_experiment_summaries[eid].get("breakdowns", {}).get(bk, {})
            for val, group in bd.items():
                if metric == "aggregate_score_avg":
                    s = group.get("aggregate_score_avg", 0)
                else:
                    s = group.get("rubric_averages", {}).get(metric, 0)
                val_scores.setdefault(val, []).append(s)

        if not val_scores:
            continue

        # Sort by average score ascending (hardest at top)
        sorted_vals = sorted(val_scores, key=lambda v: sum(val_scores[v]) / len(val_scores[v]))
        avgs = [sum(val_scores[v]) / len(val_scores[v]) for v in sorted_vals]
        display_names = [v.replace("_", " ").title() for v in sorted_vals]

        # Color gradient: lowest score = warm, highest = cool
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "bar_grad", ["#e84393", "#a29bfe", "#74b9ff", "#00cec9", "#00b894"]
        )
        score_min, score_max = min(avgs), max(avgs)
        score_span = score_max - score_min or 1
        bar_colors = [cmap((a - score_min) / score_span) for a in avgs]

        fig, ax = plt.subplots(figsize=(8, max(3, len(sorted_vals) * 0.5 + 1)))

        bars = ax.barh(display_names, avgs, color=bar_colors, edgecolor="white",
                        linewidth=0.8, height=0.6, zorder=3)

        # Value labels on bars
        for bar, avg in zip(bars, avgs):
            ax.text(
                avg + score_span * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{avg:.2f}", va="center", ha="left",
                fontsize=11, fontweight="bold",
            )

        metric_label = (
            "Aggregate Score"
            if metric == "aggregate_score_avg"
            else metric.replace("_", " ").title()
        )
        attr_label = attr.replace("_", " ").title()

        ax.set_xlim(score_min - score_span * 0.1, score_max + score_span * 0.15)
        ax.set_xlabel(f"Avg {metric_label} (across all models)", fontsize=10)
        ax.set_title(
            f"{metric_label} by {attr_label}",
            fontsize=13, fontweight="bold", pad=10,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.15, linewidth=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=11)

        plt.tight_layout()
        png_path = str(out / f"difficulty_{attr}_{metric}.png")
        fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        generated.append(png_path)

        # Caption
        caption = f"## {metric_label} by {attr_label}\n\n"
        caption += (
            f"Average {metric_label.lower()} per {attr_label.lower()} value, "
            f"pooled across all {len(exp_ids)} models. "
            f"Sorted hardest (top) to easiest (bottom).\n\n"
        )
        caption += f"| {attr_label} | Avg Score |\n|---|---|\n"
        for name, avg in zip(display_names, avgs):
            caption += f"| {name} | {avg:.2f} |\n"

        md_path = _write_caption(png_path, caption)
        generated.append(md_path)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
    })

    return generated


def generate_attribute_spread_charts(
    experiments_summary: Dict[str, Any],
    per_experiment_summaries: Dict[str, Dict[str, Any]],
    output_dir: str,
    attributes: Optional[List[str]] = None,
    metric: str = "aggregate_score_avg",
) -> List[str]:
    """
    Generate box-and-strip charts: one per attribute.

    X-axis: attribute values (e.g., personalities).
    Y-axis: score for that metric.
    Each attribute value gets a box showing the distribution across models,
    with individual model dots overlaid and labeled for outliers.

    Shows in one chart: which values are hardest, how much models spread,
    and which models are outliers.

    Args:
        experiments_summary: The experiments_summary.json data.
        per_experiment_summaries: Dict mapping experiment_id to its summary.json.
        output_dir: Directory to save PNG files.
        attributes: Breakdown attributes.
        metric: "aggregate_score_avg" or a rubric name.

    Returns:
        List of paths to generated files.
    """
    if attributes is None:
        attributes = ["severity_level", "scenario_complexity", "personality"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_names: Dict[str, str] = {}
    exp_order: List[str] = []
    for spec in experiments_summary.get("model_specs", []):
        eid = spec["experiment_id"]
        model_names[eid] = spec.get("assistant", eid)
        exp_order.append(eid)
    exp_order = [e for e in exp_order if e in per_experiment_summaries]
    if not exp_order:
        return []

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    # Consistent color per model across charts
    palette = sns.color_palette("tab10", len(exp_order))
    if len(exp_order) > 10:
        palette = sns.color_palette("tab20", len(exp_order))
    model_colors = {eid: palette[i] for i, eid in enumerate(exp_order)}

    generated: List[str] = []

    for attr in attributes:
        bk = f"by_{attr}"

        # Collect all attribute values
        all_vals: set = set()
        for eid in exp_order:
            bd = per_experiment_summaries[eid].get("breakdowns", {}).get(bk, {})
            all_vals.update(bd.keys())
        if not all_vals:
            continue

        # Order: known orderings or sort by pooled average (hardest first)
        known_orders = {
            "severity_level": ["mild", "moderate", "severe"],
            "scenario_complexity": ["regular", "chronic", "complicated", "infeasible"],
        }
        if attr in known_orders:
            attr_vals = [v for v in known_orders[attr] if v in all_vals]
            attr_vals += sorted(all_vals - set(attr_vals))
        else:
            # Sort by pooled average ascending (hardest first on left)
            def _avg(v: str) -> float:
                scores = []
                for eid in exp_order:
                    bd = per_experiment_summaries[eid].get("breakdowns", {}).get(bk, {})
                    g = bd.get(v, {})
                    if metric == "aggregate_score_avg":
                        scores.append(g.get("aggregate_score_avg", 0))
                    else:
                        scores.append(g.get("rubric_averages", {}).get(metric, 0))
                return sum(scores) / len(scores) if scores else 0
            attr_vals = sorted(all_vals, key=_avg)

        # Build data: list of scores per attribute value
        box_data: List[List[float]] = []
        model_points: List[List[Tuple[float, str, Any]]] = []  # (score, name, color)

        for val in attr_vals:
            scores = []
            points = []
            for eid in exp_order:
                bd = per_experiment_summaries[eid].get("breakdowns", {}).get(bk, {})
                g = bd.get(val, {})
                if metric == "aggregate_score_avg":
                    s = g.get("aggregate_score_avg", 0)
                else:
                    s = g.get("rubric_averages", {}).get(metric, 0)
                if g:
                    scores.append(s)
                    points.append((s, model_names.get(eid, eid), model_colors[eid]))
            box_data.append(scores)
            model_points.append(points)

        n_vals = len(attr_vals)
        display_names = [v.replace("_", " ").title() for v in attr_vals]

        fig, ax = plt.subplots(figsize=(max(8, n_vals * 1.3 + 2), 6))

        # Box plots
        bp = ax.boxplot(
            box_data, positions=range(n_vals), widths=0.5,
            patch_artist=True, showfliers=False, zorder=2,
            medianprops=dict(color="#333333", linewidth=1.5),
            boxprops=dict(facecolor="#e8e8e8", edgecolor="#aaaaaa", linewidth=0.8),
            whiskerprops=dict(color="#aaaaaa", linewidth=0.8),
            capprops=dict(color="#aaaaaa", linewidth=0.8),
        )

        # Overlay individual model dots with jitter
        rng = np.random.RandomState(42)
        for xi, points in enumerate(model_points):
            for score, name, color in points:
                jitter = rng.uniform(-0.15, 0.15)
                ax.scatter(
                    xi + jitter, score, s=60, color=color, zorder=4,
                    edgecolors="white", linewidths=0.6,
                )

            # Label outliers: top and bottom model for this attribute value
            if len(points) >= 3:
                sorted_pts = sorted(points, key=lambda p: p[0])
                # Bottom outlier
                bot = sorted_pts[0]
                ax.annotate(
                    bot[1], (xi, bot[0]),
                    textcoords="offset points", xytext=(-8, -12),
                    fontsize=7, color="#666666", ha="center", zorder=5,
                )
                # Top outlier
                top = sorted_pts[-1]
                ax.annotate(
                    top[1], (xi, top[0]),
                    textcoords="offset points", xytext=(-8, 10),
                    fontsize=7, color="#666666", ha="center", zorder=5,
                )

        ax.set_xticks(range(n_vals))
        ax.set_xticklabels(display_names, fontsize=11, rotation=25, ha="right")

        metric_label = (
            "Aggregate Score"
            if metric == "aggregate_score_avg"
            else metric.replace("_", " ").title()
        )
        attr_label = attr.replace("_", " ").title()

        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(
            f"{metric_label} Distribution by {attr_label}",
            fontsize=13, fontweight="bold", pad=10,
        )
        ax.grid(axis="y", alpha=0.2, linewidth=0.3)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Auto-zoom y
        all_scores = [s for group in box_data for s in group]
        if all_scores:
            y_lo = min(all_scores) - 0.15
            y_hi = max(all_scores) + 0.15
            if y_hi - y_lo < 0.5:
                mid = (y_hi + y_lo) / 2
                y_lo, y_hi = mid - 0.25, mid + 0.25
            ax.set_ylim(y_lo, y_hi)

        plt.tight_layout()
        png_path = str(out / f"attr_spread_{attr}_{metric}.png")
        fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        generated.append(png_path)

        # Caption
        caption = f"## {metric_label} Distribution by {attr_label}\n\n"
        caption += (
            f"Box plots show the distribution of {metric_label.lower()} "
            f"across {len(exp_order)} models for each {attr_label.lower()} value. "
            f"Individual model dots overlaid. Top/bottom outliers labeled.\n\n"
        )
        caption += f"| {attr_label} | Median | IQR | Min | Max |\n|---|---|---|---|---|\n"
        for i, (name, scores) in enumerate(zip(display_names, box_data)):
            if scores:
                s = sorted(scores)
                n = len(s)
                med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
                q1 = s[n // 4]
                q3 = s[3 * n // 4]
                caption += f"| {name} | {med:.2f} | {q3 - q1:.2f} | {s[0]:.2f} | {s[-1]:.2f} |\n"

        md_path = _write_caption(png_path, caption)
        generated.append(md_path)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
    })

    return generated


def generate_model_spread_charts(
    experiments_summary: Dict[str, Any],
    per_experiment_summaries: Dict[str, Dict[str, Any]],
    output_dir: str,
    attributes: Optional[List[str]] = None,
    metric: str = "aggregate_score_avg",
    benchmark_cases: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Grouped mini-box chart: models on X, one thin vertical bar per
    attribute value within each model group, showing the actual score
    distribution for that (model, attribute-value) pair.

    Args:
        experiments_summary: The experiments_summary.json data.
        per_experiment_summaries: Dict mapping experiment_id to its summary.json.
        output_dir: Directory to save PNG files.
        attributes: Breakdown attributes.
        metric: "aggregate_score_avg" or a rubric name.
        benchmark_cases: List of benchmark case dicts with scenario_id
            and attribute fields. If provided, uses actual per-case scores.

    Returns:
        List of paths to generated files.
    """
    if attributes is None:
        attributes = ["severity_level", "scenario_complexity", "personality"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load benchmark_cases for case_id → attribute mapping
    if benchmark_cases is None:
        bc_path = Path(output_dir).parent / "benchmark_cases.json"
        if not bc_path.exists():
            # Try inside output_dir itself
            bc_path = Path(output_dir) / "benchmark_cases.json"
        # Also check one level up from charts/
        if not bc_path.exists():
            for p in Path(output_dir).parents:
                candidate = p / "benchmark_cases.json"
                if candidate.exists():
                    bc_path = candidate
                    break
        if bc_path.exists():
            import json as _json
            benchmark_cases = _json.loads(bc_path.read_text(encoding="utf-8"))

    # Build case_id → attribute lookup
    _TC_PREFIXES = [
        "health_concern", "medication", "profile_management", "provider_access",
    ]
    def _task_cat(tt: str) -> str:
        for p in _TC_PREFIXES:
            if tt.startswith(p):
                return p
        return tt

    def _derive_age_group(c: Dict) -> str:
        pp = c.get("patient_profile", {})
        if isinstance(pp, str):
            return "unknown"
        age_raw = pp.get("account_info", {}).get("age_in_years")
        if age_raw is None:
            return "unknown"
        try:
            return derive_age_group(int(age_raw))
        except (ValueError, TypeError):
            return "unknown"

    def _derive_gender_identity(c: Dict) -> str:
        pp = c.get("patient_profile", {})
        if isinstance(pp, str):
            return "unknown"
        pi = pp.get("personal_info", {})
        return derive_gender_identity(pi.get("sex", ""), pi.get("gender", ""))

    case_attrs: Dict[str, Dict[str, str]] = {}
    if benchmark_cases:
        for c in benchmark_cases:
            cid = c.get("scenario_id", "")
            case_attrs[cid] = {
                "severity_level": c.get("severity_level", ""),
                "scenario_complexity": c.get("scenario_complexity", ""),
                "personality": c.get("personality", ""),
                "task_type": c.get("task_type", ""),
                "task_category": _task_cat(c.get("task_type", "")),
                "age_group": _derive_age_group(c),
                "gender_identity": _derive_gender_identity(c),
            }

    model_names: Dict[str, str] = {}
    exp_order: List[str] = []
    for spec in experiments_summary.get("model_specs", []):
        eid = spec["experiment_id"]
        model_names[eid] = spec.get("assistant", eid)
        exp_order.append(eid)
    exp_order = [e for e in exp_order if e in per_experiment_summaries]
    if not exp_order:
        return []

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    generated: List[str] = []

    for attr in attributes:
        # task_category is virtual — derived from by_task_type
        if attr == "task_category":
            bk_source = "by_task_type"
        else:
            bk_source = f"by_{attr}"

        all_vals: set = set()
        for eid in exp_order:
            bd = per_experiment_summaries[eid].get("breakdowns", {}).get(
                bk_source, {}
            )
            if attr == "task_category":
                for tt in bd:
                    all_vals.add(_task_cat(tt))
            else:
                all_vals.update(bd.keys())
        if not all_vals:
            continue

        attr_vals = sorted(all_vals)
        n_vals = len(attr_vals)

        # Fixed attribute order for known attributes
        known_orders: Dict[str, List[str]] = {
            "severity_level": ["mild", "moderate", "severe"],
            "scenario_complexity": ["regular", "chronic", "complicated", "infeasible"],
        }
        if attr in known_orders:
            attr_vals = [v for v in known_orders[attr] if v in all_vals]
            attr_vals += sorted(all_vals - set(attr_vals))
            n_vals = len(attr_vals)

        # High-contrast colors + hatching for more
        _base_colors = ["#e63946", "#457b9d", "#f4a261", "#2a9d8f",
                         "#6a4c93", "#1d3557", "#e9c46a", "#264653"]
        _hatch_list = ["", "//", "\\\\", "xx", "..", "||", "--", "++"]
        attr_style: Dict[str, Tuple] = {}
        for i, v in enumerate(attr_vals):
            c = _base_colors[i % len(_base_colors)]
            h = _hatch_list[i // len(_base_colors)] if i >= len(_base_colors) else ""
            attr_style[v] = (c, h)

        # Build CI-based data: [ci_lower, avg, ci_upper] per (model, attr_value)
        model_attr_scores: Dict[str, Dict[str, List[float]]] = {}
        score_key = "aggregate_score" if metric == "aggregate_score_avg" else metric
        is_turns = metric == "num_turns"
        if is_turns:
            score_key = "num_turns"
        for eid in exp_order:
            bd = per_experiment_summaries[eid].get("breakdowns", {}).get(
                bk_source, {}
            )
            cr = per_experiment_summaries[eid].get("case_results", [])

            if attr == "task_category":
                # Aggregate task_type groups into categories using case scores
                cat_scores: Dict[str, List[float]] = {v: [] for v in attr_vals}
                for case in cr:
                    cid = case.get("case_id", "")
                    cat = case_attrs.get(cid, {}).get("task_category", "")
                    if cat in cat_scores and case.get(score_key) is not None:
                        cat_scores[cat].append(case[score_key])
                scores_map: Dict[str, List[float]] = {}
                for cat, vals in cat_scores.items():
                    if not vals or len(vals) < 2:
                        avg = vals[0] if vals else 0
                        scores_map[cat] = [avg, avg, avg]
                        continue
                    import math
                    avg = sum(vals) / len(vals)
                    se = math.sqrt(
                        sum((x - avg) ** 2 for x in vals) / (len(vals) - 1)
                    ) / math.sqrt(len(vals))
                    scores_map[cat] = [
                        round(avg - 1.96 * se, 2), round(avg, 2),
                        round(avg + 1.96 * se, 2),
                    ]
                model_attr_scores[eid] = scores_map
            elif metric == "aggregate_score_avg":
                scores_map = {}
                for val in attr_vals:
                    g = bd.get(val, {})
                    avg = g.get("aggregate_score_avg", 0)
                    ci = g.get("aggregate_score_ci", [avg, avg])
                    scores_map[val] = [ci[0], avg, ci[1]]
                model_attr_scores[eid] = scores_map
            elif is_turns:
                # num_turns: compute CI from case scores
                val_scores: Dict[str, List[float]] = {v: [] for v in attr_vals}
                for case in cr:
                    cid = case.get("case_id", "")
                    attr_val = case_attrs.get(cid, {}).get(attr, "")
                    if attr_val in val_scores and case.get("num_turns") is not None:
                        val_scores[attr_val].append(case["num_turns"])
                scores_map = {}
                for val in attr_vals:
                    vals = val_scores[val]
                    if not vals or len(vals) < 2:
                        avg = vals[0] if vals else 0
                        scores_map[val] = [avg, avg, avg]
                        continue
                    import math
                    avg = sum(vals) / len(vals)
                    se = math.sqrt(
                        sum((x - avg) ** 2 for x in vals) / (len(vals) - 1)
                    ) / math.sqrt(len(vals))
                    scores_map[val] = [
                        round(avg - 1.96 * se, 2), round(avg, 2),
                        round(avg + 1.96 * se, 2),
                    ]
                model_attr_scores[eid] = scores_map
            else:
                # Rubric metric: compute CI from case scores
                val_scores = {v: [] for v in attr_vals}
                for case in cr:
                    cid = case.get("case_id", "")
                    attr_val = case_attrs.get(cid, {}).get(attr, "")
                    if attr_val in val_scores and case.get(score_key) is not None:
                        val_scores[attr_val].append(case[score_key])
                scores_map = {}
                for val in attr_vals:
                    vals = val_scores[val]
                    if not vals or len(vals) < 2:
                        avg = vals[0] if vals else 0
                        scores_map[val] = [avg, avg, avg]
                        continue
                    import math
                    avg = sum(vals) / len(vals)
                    se = math.sqrt(
                        sum((x - avg) ** 2 for x in vals) / (len(vals) - 1)
                    ) / math.sqrt(len(vals))
                    scores_map[val] = [
                        round(avg - 1.96 * se, 2), round(avg, 2),
                        round(avg + 1.96 * se, 2),
                    ]
                model_attr_scores[eid] = scores_map

        # Sort models ascending by mean score from comparison_table
        ct_lookup = {
            r["experiment_id"]: r
            for r in experiments_summary.get("comparison_table", [])
        }
        def _case_mean(eid: str) -> float:
            row = ct_lookup.get(eid, {})
            if metric == "aggregate_score_avg":
                return row.get("average_score", 0)
            if is_turns:
                return row.get("num_turns_avg", 0)
            return row.get(f"{metric}_avg", 0)

        exp_sorted = sorted(exp_order, key=_case_mean)  # ascending
        n_models = len(exp_sorted)

        group_width = 0.8
        bar_width = group_width / n_vals
        fig, ax = plt.subplots(
            figsize=(max(12, n_models * 1.2 + 3), 6)
        )

        all_scores_flat: List[float] = []

        # Per-model 95% CI of the mean from comparison_table
        _CI_LABEL = "95% CI"
        model_ci_lo_line: List[float] = []
        model_ci_hi_line: List[float] = []
        model_avg_vals: List[float] = []
        model_std_vals: List[float] = []
        for eid in exp_sorted:
            row = ct_lookup.get(eid, {})
            if metric == "aggregate_score_avg":
                ci = row.get("average_score_ci", [0, 0])
                avg = row.get("average_score", 0)
                std_val = row.get("average_score_std", 0)
            elif is_turns:
                avg = row.get("num_turns_avg", 0)
                std_val = row.get("num_turns_std", 0)
                n = row.get("num_cases", 1)
                import math as _math
                se = std_val / _math.sqrt(n) if n > 0 else 0
                ci = [round(avg - 1.96 * se, 2), round(avg + 1.96 * se, 2)]
            else:
                ci = row.get(f"{metric}_ci", [0, 0])
                avg = row.get(f"{metric}_avg", 0)
                std_val = row.get(f"{metric}_std", 0)
            model_ci_lo_line.append(ci[0])
            model_ci_hi_line.append(ci[1])
            model_avg_vals.append(avg)
            model_std_vals.append(std_val)

        # Draw shaded CI envelope behind everything
        x_range_arr = np.arange(n_models)
        ax.fill_between(
            x_range_arr, model_ci_lo_line, model_ci_hi_line,
            color="#cccccc", alpha=0.25, zorder=1,
            label=f"{_CI_LABEL} (all cases)",
        )
        ax.plot(
            x_range_arr, model_ci_hi_line,
            color="#aaaaaa", linewidth=1.0, alpha=0.5, zorder=1,
        )
        ax.plot(
            x_range_arr, model_ci_lo_line,
            color="#aaaaaa", linewidth=1.0, alpha=0.5, zorder=1,
        )

        # Labels on top: avg (std=X) \n [ci_lower, ci_upper]
        for mi, eid in enumerate(exp_sorted):
            ax.text(
                mi, model_ci_hi_line[mi] + 0.25,
                f"{model_avg_vals[mi]:.2f} (std={model_std_vals[mi]:.2f})\n[{model_ci_lo_line[mi]:.2f}, {model_ci_hi_line[mi]:.2f}]",
                ha="center", va="bottom", fontsize=8.5,
                color="#333333", alpha=0.85, zorder=5,
                fontstyle="italic",
            )

        for mi, eid in enumerate(exp_sorted):
            scores_map = model_attr_scores[eid]

            for vi, val in enumerate(attr_vals):  # fixed order
                ci_lo, avg, ci_hi = scores_map.get(val, [0, 0, 0])
                all_scores_flat.extend([ci_lo, avg, ci_hi])
                x_pos = mi + (vi - n_vals / 2 + 0.5) * bar_width
                color, hatch = attr_style[val]

                # CI bar — 2x wider
                ax.vlines(
                    x_pos, ci_lo, ci_hi,
                    colors=color, linewidths=bar_width * 50,
                    alpha=0.65, zorder=3,
                )
                # Average dot
                ax.scatter(
                    x_pos, avg, s=22, color="white",
                    zorder=4, edgecolors=color, linewidths=0.8,
                )

        # Vertical separators between model groups
        for mi in range(1, n_models):
            ax.axvline(
                mi - 0.5, color="#dddddd", linewidth=0.8,
                linestyle="-", alpha=0.5, zorder=2,
            )

        # Dynamic Y axis with 0.5 margin
        if all_scores_flat:
            ax.set_ylim(min(all_scores_flat) - 0.5, max(all_scores_flat) + 0.5)

        # Model names centered, no tick marks
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(
            [model_names.get(e, e) for e in exp_sorted],
            fontsize=11, fontweight="bold", ha="center",
        )
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=10)

        metric_label = (
            "Aggregate Score"
            if metric == "aggregate_score_avg"
            else metric.replace("_", " ").title()
        )
        attr_label = attr.replace("_", " ").title()

        # Y-axis label: add "Score" suffix for rubric metrics
        if metric == "aggregate_score_avg":
            ax.set_ylabel("Aggregate Score", fontsize=13)
        elif is_turns:
            ax.set_ylabel("Number of Turns", fontsize=13)
        else:
            ax.set_ylabel(f"{metric_label} Score", fontsize=13)

        ax.grid(axis="y", alpha=0.2, linewidth=0.3)
        ax.set_axisbelow(True)

        # Thick bounding box
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0)
            spine.set_edgecolor("#333333")

        # Legend top-left inside chart — use textures in legend
        from matplotlib.patches import Patch
        attr_handles = [
            Patch(facecolor=attr_style[v][0], hatch=attr_style[v][1],
                  edgecolor="white", alpha=0.80,
                  label=v.replace("_", " ").title())
            for v in attr_vals
        ]
        # Add CI envelope to legend
        attr_handles.append(
            Patch(facecolor="#cccccc", alpha=0.25, edgecolor="#aaaaaa",
                  label=f"{_CI_LABEL} (all cases)")
        )
        ax.legend(
            handles=attr_handles, fontsize=10,
            loc="upper left", framealpha=0.9,
            title=attr_label, title_fontsize=11,
        )

        plt.tight_layout()
        png_path = str(out / f"model_spread_{attr}_{metric}.png")
        fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        generated.append(png_path)

        # Caption
        caption = f"## {metric_label} by {attr_label} (per model)\n\n"
        caption += (
            f"Models sorted left-to-right by mean {metric_label.lower()} "
            f"across all cases (ascending).\n\n"
            f"**How to read this chart:**\n\n"
            f"- **Colored mini-bars**: 95% confidence interval (CI) of the "
            f"mean score for each {attr_label.lower()} subgroup within a model. "
            f"The bar spans [CI lower, CI upper]; the white dot marks the subgroup mean.\n"
            f"- **Gray shaded band**: 95% CI of the overall mean across ALL cases "
            f"for that model. If a mini-bar extends outside this band, that subgroup "
            f"is statistically significantly different from the model's overall average.\n"
            f"- **Labels above each model**: overall mean (std=standard deviation) "
            f"and [CI lower, CI upper].\n\n"
        )

        # Overall CI table
        caption += f"### Overall {metric_label} (95% CI)\n\n"
        caption += f"| Model | Avg | Std | CI Lower | CI Upper | Δ |\n|---|---|---|---|---|---|\n"
        for mi, eid in enumerate(exp_sorted):
            w = model_ci_hi_line[mi] - model_ci_lo_line[mi]
            caption += (
                f"| {model_names.get(eid, eid)} | {model_avg_vals[mi]:.2f} "
                f"| {model_std_vals[mi]:.2f} "
                f"| {model_ci_lo_line[mi]:.2f} | {model_ci_hi_line[mi]:.2f} "
                f"| {w:.2f} |\n"
            )

        # Per-attribute-value breakdown table
        caption += f"\n### Per-{attr_label} Breakdown\n\n"
        header = f"| Model |"
        sep = "|---|"
        for v in attr_vals:
            header += f" {v.replace('_', ' ').title()} |"
            sep += "---|"
        caption += header + "\n" + sep + "\n"
        for eid in exp_sorted:
            sm = model_attr_scores[eid]
            row = f"| {model_names.get(eid, eid)} |"
            for v in attr_vals:
                ci_lo, avg, ci_hi = sm.get(v, [0, 0, 0])
                row += f" {avg:.2f} ({ci_lo:.2f}–{ci_hi:.2f}) |"
            caption += row + "\n"

        # Summary stats
        caption += f"\n### Summary\n\n"
        caption += f"| Model | Median | Range | Best {attr_label} | Worst {attr_label} |\n|---|---|---|---|---|\n"
        for eid in exp_sorted:
            sm = model_attr_scores[eid]
            avgs = {v: sm[v][1] for v in attr_vals}
            vals = list(avgs.values())
            s = sorted(vals)
            n = len(s)
            med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
            best_v = max(avgs, key=avgs.get)  # type: ignore[arg-type]
            worst_v = min(avgs, key=avgs.get)  # type: ignore[arg-type]
            caption += (
                f"| {model_names.get(eid, eid)} | {med:.2f} | {s[-1] - s[0]:.2f} "
                f"| {best_v.replace('_', ' ').title()} ({avgs[best_v]:.2f}) "
                f"| {worst_v.replace('_', ' ').title()} ({avgs[worst_v]:.2f}) |\n"
            )

        md_path = _write_caption(png_path, caption)
        generated.append(md_path)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
    })

    return generated


def generate_scatter_by_attribute(
    experiments_summary: Dict[str, Any],
    per_experiment_summaries: Dict[str, Dict[str, Any]],
    output_dir: str,
    attributes: Optional[List[str]] = None,
    metric: str = "aggregate_score_avg",
) -> List[str]:
    """
    Scatter where X = score for a specific attribute value, Y = model's
    total range across all attribute values. Each model gets N dots
    (one per attribute value). Color = attribute value, marker = model.

    Shows which attribute values pull scores down (left) and which
    models are most sensitive (high Y), all in one chart.

    Args:
        experiments_summary: The experiments_summary.json data.
        per_experiment_summaries: Dict mapping experiment_id to its summary.json.
        output_dir: Directory to save PNG files.
        attributes: Breakdown attributes.
        metric: "aggregate_score_avg" or a rubric name.

    Returns:
        List of paths to generated files.
    """
    if attributes is None:
        attributes = ["severity_level", "scenario_complexity", "personality"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_names: Dict[str, str] = {}
    exp_order: List[str] = []
    for spec in experiments_summary.get("model_specs", []):
        eid = spec["experiment_id"]
        model_names[eid] = spec.get("assistant", eid)
        exp_order.append(eid)
    exp_order = [e for e in exp_order if e in per_experiment_summaries]
    if not exp_order:
        return []

    # Marker shapes — cycle through distinct shapes for models
    markers = ["o", "s", "D", "^", "v", "P", "*", "X", "p", "h", "d",
               "<", ">", "8", "H"]

    # Known task_type category prefixes for virtual task_category attribute
    _TASK_CAT_PREFIXES = [
        "health_concern", "medication", "profile_management", "provider_access",
    ]

    def _task_category(task_type: str) -> str:
        for prefix in _TASK_CAT_PREFIXES:
            if task_type.startswith(prefix):
                return prefix
        return task_type

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    generated: List[str] = []

    for attr in attributes:
        # Virtual attribute: task_category aggregates task_type groups
        if attr == "task_category":
            bk_source = "by_task_type"
        else:
            bk_source = f"by_{attr}"

        all_vals: set = set()
        for eid in exp_order:
            bd = per_experiment_summaries[eid].get("breakdowns", {}).get(
                bk_source, {}
            )
            if attr == "task_category":
                for tt in bd:
                    all_vals.add(_task_category(tt))
            else:
                all_vals.update(bd.keys())
        if not all_vals:
            continue

        attr_vals = sorted(all_vals)
        n_vals = len(attr_vals)
        # Shapes for attribute values
        attr_markers = ["o", "s", "D", "^", "v", "P", "*", "X", "p", "h"]
        attr_marker_map = {v: attr_markers[i % len(attr_markers)]
                           for i, v in enumerate(attr_vals)}
        attr_line_palette = sns.color_palette("Set2", n_vals)
        attr_marker_color = {v: attr_line_palette[i] for i, v in enumerate(attr_vals)}

        # Precompute each model's range and median score
        model_ranges: Dict[str, float] = {}
        model_scores: Dict[str, Dict[str, float]] = {}
        model_medians: Dict[str, float] = {}
        model_ci_widths: Dict[str, Dict[str, float]] = {}
        for eid in exp_order:
            bd = per_experiment_summaries[eid].get("breakdowns", {}).get(
                bk_source, {}
            )
            scores: Dict[str, float] = {}
            if attr == "task_category":
                # Aggregate task_type groups into categories
                cat_scores: Dict[str, List[float]] = {}
                for tt, g in bd.items():
                    cat = _task_category(tt)
                    if metric == "aggregate_score_avg":
                        s = g.get("aggregate_score_avg", 0)
                    else:
                        s = g.get("rubric_averages", {}).get(metric, 0)
                    cat_scores.setdefault(cat, []).append(s)
                for cat, s_list in cat_scores.items():
                    scores[cat] = sum(s_list) / len(s_list)
            else:
                for val in attr_vals:
                    g = bd.get(val, {})
                    if metric == "aggregate_score_avg":
                        scores[val] = g.get("aggregate_score_avg", 0)
                    else:
                        scores[val] = g.get("rubric_averages", {}).get(metric, 0)
            model_scores[eid] = scores
            # CI width per attribute value (case-to-case consistency)
            ci_widths: Dict[str, float] = {}
            if attr == "task_category":
                # Average CI widths across task types in each category
                cat_cis: Dict[str, List[float]] = {}
                for tt, g in bd.items():
                    cat = _task_category(tt)
                    ci = g.get("aggregate_score_ci", [0, 0])
                    cat_cis.setdefault(cat, []).append(ci[1] - ci[0])
                for cat, ws in cat_cis.items():
                    ci_widths[cat] = sum(ws) / len(ws)
            else:
                for val in attr_vals:
                    g = bd.get(val, {})
                    if metric == "aggregate_score_avg":
                        ci = g.get("aggregate_score_ci", [0, 0])
                    else:
                        # Use pass rate CI width as proxy for rubric-level variance
                        ci = g.get("rubric_pass_rate_cis", {}).get(
                            metric, [0, 0]
                        )
                    ci_widths[val] = ci[1] - ci[0] if len(ci) == 2 else 0
            model_ci_widths[eid] = ci_widths
            vals = sorted(scores.values())
            model_ranges[eid] = (max(vals) - min(vals)) if vals else 0
            n = len(vals)
            model_medians[eid] = (
                vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
            ) if vals else 0

        # Color spectrum for models based on median score
        model_cmap = mcolors.LinearSegmentedColormap.from_list(
            "model_spectrum",
            ["#e84393", "#a29bfe", "#74b9ff", "#00cec9", "#00b894"],
        )
        med_vals = list(model_medians.values())
        med_norm = mcolors.Normalize(
            vmin=min(med_vals) - 0.05 if med_vals else 0,
            vmax=max(med_vals) + 0.05 if med_vals else 5,
        )
        model_color_map = {
            eid: model_cmap(med_norm(model_medians[eid])) for eid in exp_order
        }

        fig, ax = plt.subplots(figsize=(10, 7))

        all_xs: List[float] = []
        all_ys: List[float] = []

        # Plot dots per model — each dot now has unique X and Y
        for mi, eid in enumerate(exp_order):
            m_color = model_color_map[eid]

            for val in attr_vals:
                x_val = model_scores[eid].get(val, 0)
                y_val = model_ci_widths[eid].get(val, 0)
                all_xs.append(x_val)
                all_ys.append(y_val)
                ax.scatter(
                    x_val, y_val, s=90, marker=attr_marker_map[val],
                    color=m_color, zorder=5,
                    edgecolors="white", linewidths=0.6,
                )

        # Layout bounds
        x_range = (max(all_xs) - min(all_xs)) or 0.1
        y_range = (max(all_ys) - min(all_ys)) or 0.01
        x_lo = min(all_xs) - x_range * 0.10
        x_hi = max(all_xs) + x_range * 0.25
        y_lo = min(all_ys) - y_range * 0.20
        y_hi = max(all_ys) + y_range * 0.22

        # Line styles for attribute values
        line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)),
                       (0, (5, 2)), (0, (1, 1)), (0, (3, 5, 1, 5)),
                       (0, (5, 1)), (0, (3, 1))]
        attr_linestyle = {v: line_styles[i % len(line_styles)]
                          for i, v in enumerate(attr_vals)}

        # Model markers (shapes)
        model_marker_map = {eid: markers[i % len(markers)]
                            for i, eid in enumerate(exp_order)}

        # Connect dots of the same attribute value across models
        for val in attr_vals:
            pts = []
            for eid in exp_order:
                x_val = model_scores[eid].get(val, 0)
                y_val = model_ci_widths[eid].get(val, 0)
                pts.append((x_val, y_val, eid))
            pts.sort(key=lambda p: p[0])
            px = [p[0] for p in pts]
            py = [p[1] for p in pts]
            ax.plot(
                px, py, color="#999999",
                linestyle=attr_linestyle[val],
                linewidth=1.3, alpha=0.5, zorder=3,
                label=val.replace("_", " ").title(),
            )

        # Re-draw dots on top with model-specific markers and colors
        for eid in exp_order:
            m_color = model_color_map[eid]
            m_marker = model_marker_map[eid]
            for val in attr_vals:
                x_val = model_scores[eid].get(val, 0)
                y_val = model_ci_widths[eid].get(val, 0)
                ax.scatter(
                    x_val, y_val, s=100, marker=m_marker,
                    color=m_color, zorder=5,
                    edgecolors="white", linewidths=0.7,
                )

        # Model name labels at rightmost dot
        for eid in exp_order:
            name = model_names.get(eid, eid)
            pts_x = [model_scores[eid].get(v, 0) for v in attr_vals]
            pts_y = [model_ci_widths[eid].get(v, 0) for v in attr_vals]
            max_i = pts_x.index(max(pts_x))
            ax.text(
                pts_x[max_i], pts_y[max_i], f"  {name}",
                fontsize=8.5, fontweight="bold", va="center", ha="left",
                color="#333333", alpha=0.85, zorder=6,
            )

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_hi, y_lo)  # Inverted: top = more consistent (narrow CI)

        metric_label = (
            "Aggregate Score"
            if metric == "aggregate_score_avg"
            else metric.replace("_", " ").title()
        )
        attr_label = attr.replace("_", " ").title()

        # Quadrant crosshairs + labels in far corners
        x_med = float(np.median(all_xs))
        y_med = float(np.median(all_ys))
        ax.axvline(x_med, color="#cccccc", linewidth=0.8,
                    linestyle="--", zorder=1)
        ax.axhline(y_med, color="#cccccc", linewidth=0.8,
                    linestyle="--", zorder=1)

        q_margin_x = x_range * 0.04
        q_margin_y = y_range * 0.06
        qp = dict(fontsize=10, alpha=0.45, fontstyle="italic",
                   color="#333333", zorder=1)
        # Inverted Y: visual top = y_lo, visual bottom = y_hi
        ax.text(x_lo + q_margin_x, y_lo + q_margin_y,
                "More Consistent &\nLower Scoring", ha="left", va="top", **qp)
        ax.text(x_hi - q_margin_x, y_lo + q_margin_y,
                "More Consistent &\nHigher Scoring", ha="right", va="top", **qp)
        ax.text(x_lo + q_margin_x, y_hi - q_margin_y,
                "Less Consistent &\nLower Scoring", ha="left", va="bottom", **qp)
        ax.text(x_hi - q_margin_x, y_hi - q_margin_y,
                "Less Consistent &\nHigher Scoring", ha="right", va="bottom", **qp)

        # Two legends: line styles for attributes, markers for models
        attr_handles = [
            plt.Line2D([0], [0], color="#999999",
                       linestyle=attr_linestyle[v], linewidth=1.5,
                       label=v.replace("_", " ").title())
            for v in attr_vals
        ]
        model_handles = [
            plt.Line2D([0], [0], marker=model_marker_map[eid], color="w",
                       markerfacecolor=model_color_map[eid],
                       markeredgecolor="white", markersize=8,
                       label=model_names.get(eid, eid))
            for eid in exp_order
        ]
        leg1 = ax.legend(
            handles=attr_handles, fontsize=8,
            loc="upper right",
            framealpha=0.9, title=attr.replace("_", " ").title(),
            title_fontsize=9,
        )
        ax.add_artist(leg1)
        ax.legend(
            handles=model_handles, fontsize=7,
            loc="lower right",
            framealpha=0.9, title="Models",
            title_fontsize=9,
        )

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(
            left=False, bottom=False,
            labelleft=False, labelbottom=False,
        )

        # Arrow axes (axes fraction)
        arrow_kw = dict(
            arrowstyle="-|>", color="#444444", lw=2.0, mutation_scale=18,
        )
        ax.annotate(
            "", xy=(1.0, 0.0), xycoords="axes fraction",
            xytext=(0.0, 0.0), textcoords="axes fraction",
            arrowprops=arrow_kw, annotation_clip=False,
        )
        ax.annotate(
            "", xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=(0.0, 0.0), textcoords="axes fraction",
            arrowprops=arrow_kw, annotation_clip=False,
        )

        # Axis titles close to arrows
        ax.text(
            0.5, -0.04, f"Higher {metric_label} Score",
            transform=ax.transAxes,
            ha="center", va="top", fontsize=12, fontweight="bold",
            color="#444444",
        )
        ax.text(
            -0.04, 0.5, f"Consistency Across {attr_label}",
            transform=ax.transAxes,
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="#444444", rotation=90,
        )

        ax.grid(alpha=0.10, linewidth=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        png_path = str(out / f"scatter_attr_{attr}_{metric}.png")
        fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        generated.append(png_path)

        # Caption
        caption = f"## {metric_label} by {attr_label}\n\n"
        caption += (
            f"**X** = {metric_label.lower()} for a specific "
            f"{attr_label.lower()} value. "
            f"**Y** = consistency across {attr_label.lower()} "
            f"(inverted score range: higher = smaller max-min gap = more consistent). "
            f"**Color** = {attr_label.lower()} value. "
            f"**Shape** = model.\n\n"
            f"Top-right corner = high scoring and robust. "
            f"Horizontal box per model shows the score spread.\n\n"
        )
        # Attribute ranking
        val_avgs: Dict[str, float] = {}
        for val in attr_vals:
            s_list = [model_scores[eid][val] for eid in exp_order]
            val_avgs[val] = sum(s_list) / len(s_list) if s_list else 0

        caption += f"**{attr_label} ranking** (avg across models):\n\n"
        for val in sorted(val_avgs, key=val_avgs.get, reverse=True):  # type: ignore[arg-type]
            caption += f"- {val.replace('_', ' ').title()}: {val_avgs[val]:.2f}\n"

        md_path = _write_caption(png_path, caption)
        generated.append(md_path)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
    })

    return generated


def generate_metrics_csv(
    experiments_summary: Dict[str, Any],
    per_experiment_summaries: Dict[str, Dict[str, Any]],
    case_metadata: Dict[str, Dict[str, str]],
    output_path: str,
) -> str:
    """Generate CSV table with mean, std, CI, delta for all models × variants × metrics.

    Args:
        experiments_summary: The experiments_summary.json data.
        per_experiment_summaries: Dict of experiment_id -> summary.json data.
        case_metadata: Dict of case_id -> metadata dict (from entry.metadata).
        output_path: Path to write the CSV file.

    Returns:
        Path to the written CSV file.
    """
    import csv
    import math
    import numpy as np

    model_names = {}
    for spec in experiments_summary.get("model_specs", []):
        model_names[spec["experiment_id"]] = spec.get("assistant", spec["experiment_id"])

    rubrics = None
    variants_fns = {
        "gender_identity": lambda m: m.get("gender_identity", "unknown"),
        "age_group": lambda m: m.get("age_group", "unknown"),
        "severity_level": lambda m: m.get("severity_level", "unknown"),
        "scenario_complexity": lambda m: m.get("scenario_complexity", "unknown"),
        "personality": lambda m: m.get("personality", "unknown"),
    }

    def _stats(scores):
        arr = np.array(scores, dtype=float)
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        ci_half = 1.96 * std / math.sqrt(n) if n > 0 else 0.0
        return n, round(mean, 4), round(std, 4), round(mean - ci_half, 4), round(mean + ci_half, 4), round(2 * ci_half, 4)

    rows = []
    for eid, name in sorted(model_names.items(), key=lambda x: x[1]):
        s = per_experiment_summaries.get(eid, {})
        case_results = s.get("case_results", [])
        if not case_results:
            continue

        if rubrics is None:
            rubrics = [k for k in case_results[0] if k not in ("case_id", "aggregate_score", "num_turns")]

        def emit(variant_label, cases):
            agg = [c["aggregate_score"] for c in cases if c.get("aggregate_score") is not None]
            if agg:
                n, mean, std, ci_l, ci_u, delta = _stats(agg)
                rows.append({"model": name, "variant": variant_label, "metric": "aggregate_score", "n": n, "mean": mean, "std": std, "ci_lower": ci_l, "ci_upper": ci_u, "delta": delta})
            turns = [c["num_turns"] for c in cases if c.get("num_turns") is not None]
            if turns:
                n, mean, std, ci_l, ci_u, delta = _stats(turns)
                rows.append({"model": name, "variant": variant_label, "metric": "num_turns", "n": n, "mean": mean, "std": std, "ci_lower": ci_l, "ci_upper": ci_u, "delta": delta})
            for rubric in rubrics:
                scores = [c[rubric] for c in cases if c.get(rubric) is not None]
                if scores:
                    n, mean, std, ci_l, ci_u, delta = _stats(scores)
                    rows.append({"model": name, "variant": variant_label, "metric": rubric, "n": n, "mean": mean, "std": std, "ci_lower": ci_l, "ci_upper": ci_u, "delta": delta})

        emit("overall", case_results)

        for attr_name, attr_fn in variants_fns.items():
            groups: Dict[str, list] = {}
            for c in case_results:
                meta = case_metadata.get(c.get("case_id", ""), {})
                groups.setdefault(attr_fn(meta), []).append(c)
            for val, cases in sorted(groups.items()):
                emit(f"{attr_name}:{val}", cases)

    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "variant", "metric", "n", "mean", "std", "ci_lower", "ci_upper", "delta"])
        w.writeheader()
        w.writerows(rows)

    return output_path
