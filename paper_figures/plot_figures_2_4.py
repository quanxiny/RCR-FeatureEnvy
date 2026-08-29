#!/usr/bin/env python3
"""Reproduce EI-paper Figures 2--4 from the frozen JSON summaries.

The script deliberately contains no hard-coded result values.  Every plotted
mean, standard deviation, and confidence interval is read from the two JSON
files supplied on the command line.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NAVY = "#153E75"
TEAL = "#007F7B"
ORANGE = "#E88716"
PURPLE = "#6B3FA0"
GRAY = "#74797E"
LIGHT_BLUE = "#9AAEC7"
RED = "#D84A3A"
GRID = "#D9DEE3"


def configure_style() -> None:
    """Use a compact, vector-safe publication style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.2,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate(static: dict[str, Any], continual: dict[str, Any]) -> None:
    """Fail early if a different or incomplete experiment snapshot is used."""
    static_block = static["confirmatory_10_units"]
    if static_block["cl6_vs_supcon"]["f1"]["units"] != 10:
        raise ValueError("Figure 2 expects ten development-excluded static units")
    comparisons = continual["confirmatory_14_unit_comparisons"]
    if comparisons["relation_replay_vs_cl6_replay"]["outer_f1"]["units"] != 14:
        raise ValueError("Figure 3 expects fourteen development-excluded continual units")
    budgets = continual["buffer_sensitivity_seed42_fivefold"]
    if set(budgets) != {"5", "10", "20"}:
        raise ValueError("Figure 4 expects the 5%, 10%, and 20% replay budgets")
    if any(budgets[key]["units"] != 5 for key in budgets):
        raise ValueError("Figure 4 expects five folds at each replay budget")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": stem.replace("_", " ").title(),
        "Creator": "Matplotlib; data read from frozen experiment summaries",
    }
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        metadata=metadata,
    )
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        output_dir / f"{stem}.svg",
        bbox_inches="tight",
        pad_inches=0.04,
        metadata=metadata,
    )
    plt.close(fig)


def static_std(static_block: dict[str, Any], method: str, metric: str) -> float:
    """Recover the per-method SD stored with the paired static comparisons."""
    if method == "focal":
        return float(static_block["supcon_vs_focal"][metric]["reference_std"])
    if method == "supcon":
        return float(static_block["supcon_vs_focal"][metric]["candidate_std"])
    if method == "cl6":
        return float(static_block["cl6_vs_supcon"][metric]["candidate_std"])
    raise KeyError(method)


def plot_figure_2(
    static: dict[str, Any], continual: dict[str, Any], output_dir: Path
) -> None:
    """Descriptive static and continual means with unit-level SD whiskers."""
    sb = static["confirmatory_10_units"]
    sv = sb["method_means"]
    cv = continual["continual_aggregates_15_units"]

    # The Springer SVProc page is a narrow single column. Stacking the two
    # panels keeps labels near their native 8--10 pt size after inclusion.
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 6.2))
    fig.subplots_adjust(left=0.14, right=0.985, top=0.92, bottom=0.13, hspace=0.62)

    # Panel a: static objective comparison.
    static_metrics = [("f1", "F1"), ("mcc", "MCC"), ("pr_auc", "PR-AUC")]
    static_methods = [
        ("focal", "Focal", GRAY),
        ("supcon", "SupCon", ORANGE),
        ("cl6", "LRC", PURPLE),
    ]
    x = np.arange(len(static_metrics), dtype=float)
    width = 0.23
    for idx, (method, label, color) in enumerate(static_methods):
        means = np.array([sv[method][key] for key, _ in static_metrics], dtype=float)
        sds = np.array([static_std(sb, method, key) for key, _ in static_metrics])
        bars = axes[0].bar(
            x + (idx - 1) * width,
            means,
            width,
            yerr=sds,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            capsize=2.2,
            error_kw={"elinewidth": 0.85, "capthick": 0.85, "ecolor": "#34383C"},
            zorder=3,
        )
        for bar, mean, sd in zip(bars, means, sds):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                min(mean + sd + 0.018, 0.975),
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.1,
                color=color,
            )
    axes[0].set_xticks(x, [label for _, label in static_metrics])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("a  Static detector · 10 confirmatory units", loc="left", weight="bold")
    axes[0].grid(axis="y", color=GRID, linewidth=0.65, alpha=0.7)
    axes[0].legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        borderaxespad=0,
        columnspacing=1.4,
        handlelength=1.5,
    )

    # Panel b: continual comparison. Values remain visible in the accompanying
    # table; the plot focuses on comparable geometry and uncertainty.
    continual_metrics = [
        ("final_average_probe_accuracy", "Probe accuracy"),
        ("outer_f1", "Outer F1"),
        ("outer_mcc", "Outer MCC"),
    ]
    continual_methods = [
        ("finetune_cl6", "Fine-tune", "#9B9B9B"),
        ("ewc_cl6", "EWC", LIGHT_BLUE),
        ("focal_replay", "Focal+R", NAVY),
        ("supcon_replay", "SupCon+R", ORANGE),
        ("cl6_replay", "LRC+R", PURPLE),
        ("relation_replay", "RCR", TEAL),
    ]
    x = np.arange(len(continual_metrics), dtype=float)
    width = 0.126
    for idx, (method, label, color) in enumerate(continual_methods):
        means = np.array([cv[method][key]["mean"] for key, _ in continual_metrics])
        sds = np.array([cv[method][key]["std"] for key, _ in continual_metrics])
        axes[1].bar(
            x + (idx - 2.5) * width,
            means,
            width,
            yerr=sds,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            capsize=1.7,
            error_kw={"elinewidth": 0.7, "capthick": 0.7, "ecolor": "#34383C"},
            zorder=3,
        )
    axes[1].set_xticks(x, [label for _, label in continual_metrics])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Score")
    axes[1].set_title("b  Continual context · 15 fold–seed units", loc="left", weight="bold")
    axes[1].grid(axis="y", color=GRID, linewidth=0.65, alpha=0.7)
    axes[1].legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        borderaxespad=0,
        columnspacing=1.2,
        handlelength=1.5,
    )

    fig.text(0.5, 0.972, "Descriptive mean ± SD across the stated evaluation units", ha="center", fontsize=8.5, color="#52575C")
    save_figure(fig, output_dir, "main_results")


def plot_figure_3(continual: dict[str, Any], output_dir: Path) -> None:
    """Forest plots for the four preregistered oriented continual outcomes."""
    comparisons = continual["confirmatory_14_unit_comparisons"]
    rows = [
        ("supcon_replay_vs_focal_replay", "SupCon+R − Focal+R", ORANGE),
        ("cl6_replay_vs_focal_replay", "LRC+R − Focal+R", PURPLE),
        ("cl6_replay_vs_supcon_replay", "LRC+R − SupCon+R", NAVY),
        ("relation_replay_vs_cl6_replay", "RCR − LRC+R", TEAL),
    ]
    metrics = [
        ("final_average_probe_accuracy", "a  Final probe accuracy"),
        ("average_forgetting", "b  Forgetting reduction"),
        ("outer_f1", "c  Outer F1"),
        ("outer_mcc", "d  Outer MCC"),
    ]

    # A four-row forest plot remains readable at the 122-mm SVProc text width;
    # a 2x2 layout would shrink method labels below the template guidance.
    fig, axes = plt.subplots(4, 1, figsize=(5.0, 7.7))
    fig.subplots_adjust(left=0.35, right=0.94, top=0.94, bottom=0.075, hspace=0.78)
    y_positions = np.arange(len(rows), dtype=float)
    for ax, (metric, title) in zip(axes.flat, metrics):
        lows: list[float] = []
        highs: list[float] = []
        plotted: list[tuple[float, float, float, str]] = []
        for key, _label, color in rows:
            item = comparisons[key][metric]
            mean = float(item["oriented_benefit"])
            lo, hi = map(float, item["crossed_seed_fold_bootstrap_95_ci"])
            lows.append(lo)
            highs.append(hi)
            plotted.append((mean, lo, hi, color))

        span = max(highs) - min(lows)
        pad = max(0.012, 0.19 * span)
        ax.set_xlim(min(lows) - 0.08 * span, max(highs) + pad)
        for yi, (mean, lo, hi, color) in enumerate(plotted):
            ax.errorbar(
                mean,
                yi,
                xerr=np.array([[mean - lo], [hi - mean]]),
                fmt="o",
                color=color,
                ecolor=color,
                markersize=5.2,
                markeredgecolor="white",
                markeredgewidth=0.6,
                elinewidth=1.45,
                capsize=3.0,
                capthick=1.15,
                zorder=3,
            )
            ax.annotate(
                f"{mean:+.3f}",
                (hi, yi),
                xytext=(7, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7.6,
                color=color,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.95},
            )
        ax.axvline(0.0, color="#30353A", linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        ax.set_yticks(y_positions, [label for _, label, _ in rows])
        ax.set_ylim(len(rows) - 0.85, -0.15)
        ax.set_title(title, loc="left", weight="bold", pad=8)
        ax.set_xlabel("Oriented paired benefit (95% clustered bootstrap CI)")
        ax.grid(axis="x", color=GRID, linewidth=0.65, alpha=0.65)
    fig.text(
        0.5,
        0.985,
        "Four planned contrasts · 14 development-excluded continual units · rightward favors the first method",
        ha="center",
        fontsize=8.6,
        color="#52575C",
    )
    save_figure(fig, output_dir, "paired_effects")


def plot_figure_4(continual: dict[str, Any], output_dir: Path) -> None:
    """Five-fold seed-42 replay-budget sensitivity with mean ± SD."""
    values = continual["buffer_sensitivity_seed42_fivefold"]
    ratios = np.array([5.0, 10.0, 20.0])
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 6.0))
    fig.subplots_adjust(left=0.15, right=0.975, top=0.91, bottom=0.23, hspace=0.44)

    line_specs = [
        ("final_average_probe_accuracy", "Probe accuracy", TEAL, "o"),
        ("outer_f1", "Outer F1", NAVY, "s"),
        ("outer_mcc", "Outer MCC", PURPLE, "^"),
    ]
    handles: list[Any] = []
    labels: list[str] = []
    for key, label, color, marker in line_specs:
        means = np.array([values[str(int(r))][key]["mean"] for r in ratios])
        sds = np.array([values[str(int(r))][key]["std"] for r in ratios])
        artist = axes[0].errorbar(
            ratios,
            means,
            yerr=sds,
            label=label,
            color=color,
            marker=marker,
            markersize=5.8,
            markeredgecolor="white",
            markeredgewidth=0.6,
            linewidth=1.7,
            elinewidth=0.9,
            capsize=3,
            zorder=3,
        )
        handles.append(artist)
        labels.append(label)
        for x_value, mean in zip(ratios, means):
            axes[0].annotate(
                f"{mean:.3f}",
                (x_value, mean),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.4,
                color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3, "alpha": 0.92},
            )
    axes[0].set_xlim(4.0, 21.0)
    axes[0].set_ylim(0.48, 0.94)
    axes[0].set_xticks(ratios, ["5%", "10%\n(primary)", "20%"])
    axes[0].set_xlabel("Replay case budget")
    axes[0].set_ylabel("Mean score")
    axes[0].set_title("a  Detection and retention", loc="left", weight="bold")
    axes[0].grid(color=GRID, linewidth=0.65, alpha=0.7)

    for key, label, color, marker, lower in [
        ("average_forgetting", "Forgetting ↓", RED, "o", True),
        ("backward_transfer", "BWT ↑", ORANGE, "s", False),
    ]:
        means = np.array([values[str(int(r))][key]["mean"] for r in ratios])
        sds = np.array([values[str(int(r))][key]["std"] for r in ratios])
        artist = axes[1].errorbar(
            ratios,
            means,
            yerr=sds,
            label=label,
            color=color,
            marker=marker,
            markersize=5.8,
            markeredgecolor="white",
            markeredgewidth=0.6,
            linewidth=1.7,
            elinewidth=0.9,
            capsize=3,
            zorder=3,
        )
        handles.append(artist)
        labels.append(label)
        for idx, (x_value, mean) in enumerate(zip(ratios, means)):
            offset = -12 if (x_value == 20 and lower) else 7
            axes[1].annotate(
                f"{mean:+.3f}",
                (x_value, mean),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if offset > 0 else "top",
                fontsize=7.4,
                color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3, "alpha": 0.92},
            )
    axes[1].axhline(0.0, color="#30353A", linewidth=0.9, linestyle=(0, (4, 3)))
    axes[1].set_xlim(4.0, 21.0)
    axes[1].set_ylim(-0.065, 0.065)
    axes[1].set_xticks(ratios, ["5%", "10%\n(primary)", "20%"])
    axes[1].set_xlabel("Replay case budget")
    axes[1].set_ylabel("Mean value")
    axes[1].set_title("b  Forgetting and backward transfer", loc="left", weight="bold")
    axes[1].grid(color=GRID, linewidth=0.65, alpha=0.7)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        frameon=False,
        columnspacing=1.6,
        handlelength=2.0,
    )
    fig.text(0.5, 0.97, "Seed 42 · five project-disjoint folds · markers show means and whiskers show SD", ha="center", fontsize=8.5, color="#52575C")
    save_figure(fig, output_dir, "buffer_sensitivity")


def generate_all(
    static: dict[str, Any], continual: dict[str, Any], output_dir: Path
) -> None:
    configure_style()
    validate(static, continual)
    plot_figure_2(static, continual, output_dir)
    plot_figure_3(continual, output_dir)
    plot_figure_4(continual, output_dir)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static",
        type=Path,
        default=root / "data" / "static_detection_summary.json",
    )
    parser.add_argument(
        "--continual",
        type=Path,
        default=root / "data" / "continual_learning_summary.json",
    )
    parser.add_argument("--output", type=Path, default=root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    static = load_json(args.static)
    continual = load_json(args.continual)
    generate_all(static, continual, args.output)
    for stem in ("main_results", "paired_effects", "buffer_sensitivity"):
        for extension in ("pdf", "svg", "png"):
            print(args.output / f"{stem}.{extension}")


if __name__ == "__main__":
    main()
