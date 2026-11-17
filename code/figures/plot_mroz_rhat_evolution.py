import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "empirical" / "mroz"

INPUT_FILES = {
    "baseline": (
        RESULTS_DIR
        / "proposed"
        / "rhat_evolution.csv"
    ),
    "no_selection": (
        RESULTS_DIR
        / "gbc_without_variable_selection"
        / "rhat_evolution.csv"
    ),
    "tau0": (
        RESULTS_DIR
        / "prior_sensitivity_tau0"
        / "rhat_evolution.csv"
    ),
    "tau1": (
        RESULTS_DIR
        / "prior_sensitivity_tau1"
        / "rhat_evolution.csv"
    ),
}

OUTPUT_DIR = REPO_ROOT / "figures" / "empirical"
OUTPUT_STEM = OUTPUT_DIR / "figure_mroz_rhat_evolution"


def require_matplotlib_panel_alignment(
    fig,
    axes,
    json_out,
    tolerance_pt=1.5,
):
    """Verify equal panel widths and aligned top and bottom edges."""
    fig.canvas.draw()

    figure_width_pt = fig.get_figwidth() * 72.0
    figure_height_pt = fig.get_figheight() * 72.0

    rectangles = []
    for index, ax in enumerate(axes):
        position = ax.get_position()
        rectangles.append({
            "panel": chr(ord("a") + index),
            "left_pt": position.x0 * figure_width_pt,
            "bottom_pt": position.y0 * figure_height_pt,
            "right_pt": position.x1 * figure_width_pt,
            "top_pt": position.y1 * figure_height_pt,
            "width_pt": position.width * figure_width_pt,
            "height_pt": position.height * figure_height_pt,
        })

    reference = rectangles[0]
    checks = {
        "top_edge_deviation_pt": max(
            abs(item["top_pt"] - reference["top_pt"])
            for item in rectangles
        ),
        "bottom_edge_deviation_pt": max(
            abs(item["bottom_pt"] - reference["bottom_pt"])
            for item in rectangles
        ),
        "width_deviation_pt": max(
            abs(item["width_pt"] - reference["width_pt"])
            for item in rectangles
        ),
        "height_deviation_pt": max(
            abs(item["height_pt"] - reference["height_pt"])
            for item in rectangles
        ),
    }

    passed = all(
        value <= tolerance_pt
        for value in checks.values()
    )

    report = {
        "status": "PASS" if passed else "FIX BEFORE DELIVERY",
        "tolerance_pt": tolerance_pt,
        "rectangles": rectangles,
        "checks": checks,
    }

    Path(json_out).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    if not passed:
        raise RuntimeError(
            "The two rendered panels are not aligned within "
            f"{tolerance_pt} pt."
        )


def read_rhat_file(path, method_label):
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
    data["method"] = method_label

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
            f"{path} contains missing max_rhat values."
        )

    if not np.isfinite(data["max_rhat"]).all():
        raise ValueError(
            f"{path} contains non-finite max_rhat values."
        )

    return data.sort_values(
        "retained_draws_per_chain"
    ).reset_index(drop=True)


def plot_trajectory(
    ax,
    data,
    label,
    color,
    marker,
    linestyle="-",
    zorder=3,
):
    x = (
        data["retained_draws_per_chain"].to_numpy()
        / 1000.0
    )
    y = data["max_rhat"].to_numpy()

    ax.plot(
        x,
        y,
        label=label,
        color=color,
        linewidth=1.45,
        linestyle=linestyle,
        marker=marker,
        markersize=3.3,
        markerfacecolor="white",
        markeredgewidth=0.85,
        zorder=zorder,
    )


def add_zoom_inset(ax, trajectories, x_upper):
    inset = ax.inset_axes([0.43, 0.39, 0.54, 0.34])

    for data, color, marker in trajectories:
        x = (
            data["retained_draws_per_chain"].to_numpy()
            / 1000.0
        )
        y = data["max_rhat"].to_numpy()
        inset.plot(
            x,
            y,
            color=color,
            linewidth=1.0,
            marker=marker,
            markersize=2.4,
            markerfacecolor="white",
            markeredgewidth=0.65,
        )

    inset.axhspan(
        0.999,
        1.01,
        color="#D9D9D9",
        alpha=0.35,
        linewidth=0,
        zorder=0,
    )
    inset.axhline(
        1.01,
        color="#555555",
        linewidth=0.7,
        linestyle="--",
        zorder=1,
    )
    inset.set_xlim(0, x_upper)
    inset.set_ylim(0.999, 1.03)
    inset.set_yticks([1.00, 1.01, 1.02, 1.03])
    inset.tick_params(labelsize=6.5, length=2.2, pad=1.5)
    inset.grid(
        axis="y",
        color="#E1E1E1",
        linewidth=0.4,
        alpha=0.75,
    )
    inset.set_title(
        r"Zoom: $1.00\leq\widehat{R}\leq1.03$",
        fontsize=7.2,
        pad=2,
    )


def configure_axis(ax, panel_label, title, y_lower):
    ax.axhspan(
        y_lower,
        1.01,
        color="#D9D9D9",
        alpha=0.35,
        linewidth=0,
        zorder=0,
    )

    ax.axhline(
        1.01,
        color="#555555",
        linewidth=0.85,
        linestyle="--",
        zorder=1,
    )

    ax.text(
        0.015,
        1.01,
        r"$\widehat{R}=1.01$",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#444444",
    )

    ax.text(
        -0.11,
        1.04,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )

    ax.set_title(
        title,
        loc="left",
        pad=7,
    )

    ax.set_xlabel(
        "Retained posterior draws per chain (thousands)"
    )

    ax.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.55,
        alpha=0.75,
    )

    ax.legend(
        loc="upper right",
        handlelength=2.4,
    )


def make_figure(
    baseline,
    no_selection,
    tau0,
    tau1,
):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
            "sans-serif",
        ],
        "font.size": 7.5,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    all_values = np.concatenate([
        baseline["max_rhat"].to_numpy(),
        no_selection["max_rhat"].to_numpy(),
        tau0["max_rhat"].to_numpy(),
        tau1["max_rhat"].to_numpy(),
    ])

    y_lower = 0.999
    y_max_data = float(np.max(all_values))
    y_upper = y_max_data + 0.05 * (y_max_data - 1.0)

    all_retained = np.concatenate([
        baseline["retained_draws_per_chain"].to_numpy(),
        no_selection["retained_draws_per_chain"].to_numpy(),
        tau0["retained_draws_per_chain"].to_numpy(),
        tau1["retained_draws_per_chain"].to_numpy(),
    ])

    x_upper = float(np.max(all_retained)) / 1000.0 + 1.0

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.35),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    blue = "#0072B2"
    gray = "#666666"
    orange = "#D55E00"
    green = "#009E73"

    plot_trajectory(
        axes[0],
        baseline,
        "Proposed, baseline prior",
        blue,
        "o",
        zorder=4,
    )
    plot_trajectory(
        axes[0],
        no_selection,
        "No variable selection",
        gray,
        "s",
    )

    plot_trajectory(
        axes[1],
        baseline,
        "Proposed, baseline prior",
        blue,
        "o",
        zorder=4,
    )
    plot_trajectory(
        axes[1],
        tau0,
        r"Proposed, $\tau_0=0.05$",
        orange,
        "s",
    )
    plot_trajectory(
        axes[1],
        tau1,
        r"Proposed, $\tau_1=1.00$",
        green,
        "^",
    )

    configure_axis(
        axes[0],
        "a",
        "Model specification",
        y_lower,
    )
    configure_axis(
        axes[1],
        "b",
        "Prior sensitivity",
        y_lower,
    )

    add_zoom_inset(
        axes[0],
        [
            (baseline, blue, "o"),
            (no_selection, gray, "s"),
        ],
        x_upper,
    )
    add_zoom_inset(
        axes[1],
        [
            (baseline, blue, "o"),
            (tau0, orange, "s"),
            (tau1, green, "^"),
        ],
        x_upper,
    )

    axes[0].set_ylabel(
        r"Maximum rank-normalized split-$\widehat{R}$"
    )

    for ax in axes:
        ax.set_xlim(0, x_upper)
        ax.set_ylim(y_lower, y_upper)

    return fig


def main():
    baseline = read_rhat_file(
        INPUT_FILES["baseline"],
        "Proposed, baseline prior",
    )
    no_selection = read_rhat_file(
        INPUT_FILES["no_selection"],
        "No variable selection",
    )
    tau0 = read_rhat_file(
        INPUT_FILES["tau0"],
        "Proposed, tau0 = 0.05",
    )
    tau1 = read_rhat_file(
        INPUT_FILES["tau1"],
        "Proposed, tau1 = 1.00",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_data = pd.concat(
        [
            baseline,
            no_selection,
            tau0,
            tau1,
        ],
        ignore_index=True,
    )

    source_data.to_csv(
        OUTPUT_DIR / "figure_mroz_rhat_source_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig = make_figure(
        baseline=baseline,
        no_selection=no_selection,
        tau0=tau0,
        tau1=tau1,
    )

    require_matplotlib_panel_alignment(
        fig,
        fig.axes[:2],
        OUTPUT_STEM.with_suffix(".alignment.json"),
        tolerance_pt=1.5,
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
    print(OUTPUT_STEM.with_suffix(".alignment.json"))
    print(
        OUTPUT_DIR
        / "figure_mroz_rhat_source_data.csv"
    )


if __name__ == "__main__":
    main()
