from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "empirical" / "meps2001"

PROPOSED_FILE = (
    RESULTS_DIR
    / "proposed_income_exclusion"
    / "rhat_evolution.csv"
)

IQBAL_FILE = (
    RESULTS_DIR
    / "iqbal_fixed_log"
    / "rhat_evolution.csv"
)

OUTPUT_DIR = REPO_ROOT / "figures" / "empirical"

OUTPUT_STEM = OUTPUT_DIR / "figure_meps2001_rhat_evolution"


def read_rhat_file(path, expected_method):
    if not path.exists():
        raise FileNotFoundError(
            f"Rhat evolution file not found:\n{path}"
        )

    data = pd.read_csv(path)

    required = {
        "method",
        "retained_draws_per_chain",
        "total_iteration",
        "max_rhat",
        "median_rhat",
        "parameters_above_1_01",
        "worst_parameter",
    }

    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing)}"
        )

    data = data.copy()
    data["method"] = expected_method

    numeric_columns = [
        "retained_draws_per_chain",
        "total_iteration",
        "max_rhat",
        "median_rhat",
        "parameters_above_1_01",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    if data["max_rhat"].isna().any():
        raise ValueError(
            f"{path.name} contains missing max_rhat values."
        )

    if not np.isfinite(data["max_rhat"]).all():
        raise ValueError(
            f"{path.name} contains non-finite max_rhat values."
        )

    data = data.sort_values(
        "retained_draws_per_chain"
    ).reset_index(drop=True)

    return data


def make_figure(proposed, iqbal):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
            "sans-serif",
        ],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    colors = {
        "Proposed GBC model": "#0072B2",
        "Iqbal fixed-log model": "#D55E00",
    }

    markers = {
        "Proposed GBC model": "o",
        "Iqbal fixed-log model": "s",
    }

    fig, ax = plt.subplots(
        figsize=(6.496, 3.228)
    )

    all_values = np.concatenate([
        proposed["max_rhat"].to_numpy(),
        iqbal["max_rhat"].to_numpy(),
    ])

    y_min_data = float(np.min(all_values))
    y_max_data = float(np.max(all_values))
    y_range = max(y_max_data - y_min_data, 0.005)

    y_lower = 0.999
    y_upper = max(1.015, y_max_data + 0.10 * y_range)

    # The shaded region indicates the conventional Rhat target.
    ax.axhspan(
        1.0,
        1.01,
        color="#D9D9D9",
        alpha=0.35,
        linewidth=0,
        zorder=0,
    )

    ax.axhline(
        1.01,
        color="#555555",
        linewidth=0.9,
        linestyle="--",
        zorder=1,
    )

    ax.text(
        0.01,
        1.01,
        r"$\widehat{R}=1.01$",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#444444",
    )

    datasets = [
        ("Proposed GBC model", proposed),
        ("Iqbal fixed-log model", iqbal),
    ]

    for method, data in datasets:
        x = (
            data["retained_draws_per_chain"].to_numpy()
            / 1000.0
        )
        y = data["max_rhat"].to_numpy()

        ax.plot(
            x,
            y,
            color=colors[method],
            linewidth=1.6,
            marker=markers[method],
            markersize=3.8,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=method,
            zorder=3,
        )

    proposed_last = proposed.iloc[-1]
    iqbal_last = iqbal.iloc[-1]

    ax.annotate(
        f"{proposed_last['max_rhat']:.4f}",
        xy=(
            proposed_last["retained_draws_per_chain"] / 1000.0,
            proposed_last["max_rhat"],
        ),
        xytext=(7, 9),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=colors["Proposed GBC model"],
    )

    ax.annotate(
        f"{iqbal_last['max_rhat']:.4f}",
        xy=(
            iqbal_last["retained_draws_per_chain"] / 1000.0,
            iqbal_last["max_rhat"],
        ),
        xytext=(-7, 9),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=colors["Iqbal fixed-log model"],
    )

    ax.set_xlim(
        0,
        max(
            proposed["retained_draws_per_chain"].max(),
            iqbal["retained_draws_per_chain"].max(),
        ) / 1000.0 + 2.0,
    )

    ax.set_ylim(y_lower, y_upper)

    ax.set_xlabel(
        "Retained posterior draws per chain (thousands)"
    )
    ax.set_ylabel(
        r"Maximum rank-normalized split-$\widehat{R}$"
    )

    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.3f")
    )

    ax.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.6,
        alpha=0.75,
    )

    ax.legend(
        loc="upper right",
        handlelength=2.6,
    )

    fig.tight_layout()

    return fig


def main():
    proposed = read_rhat_file(
        PROPOSED_FILE,
        "Proposed GBC model",
    )
    iqbal = read_rhat_file(
        IQBAL_FILE,
        "Iqbal fixed-log model",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined = pd.concat(
        [proposed, iqbal],
        ignore_index=True,
    )

    combined.to_csv(
        OUTPUT_DIR / "figure_meps2001_rhat_source_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig = make_figure(
        proposed=proposed,
        iqbal=iqbal,
    )

    fig.savefig(
        OUTPUT_STEM.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_STEM.with_suffix(".svg"),
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_STEM.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_STEM.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Figure files created:")
    print(OUTPUT_STEM.with_suffix(".pdf"))
    print(OUTPUT_STEM.with_suffix(".svg"))
    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".tiff"))
    print(
        OUTPUT_DIR
        / "figure_meps2001_rhat_source_data.csv"
    )


if __name__ == "__main__":
    main()
