"""Generate the simulation figures from the saved summary files."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from audit_panel_alignment import require_matplotlib_panel_alignment

REPO_ROOT = Path(__file__).resolve().parents[2]


PROPOSED_DIR = REPO_ROOT / "results" / "simulation" / "proposed"
NO_SELECTION_DIR = REPO_ROOT / "results" / "simulation" / "gbc_without_variable_selection"
IQBAL_DIR = REPO_ROOT / "results" / "simulation" / "iqbal_spike_slab"
OUTCOME_ONLY_DIR = REPO_ROOT / "results" / "simulation" / "outcome_only"
HECKMAN_DIR = REPO_ROOT / "results" / "simulation" / "heckman_two_step"

N500_DIR = REPO_ROOT / "results" / "sensitivity" / "sample_size_n500"
N1000_DIR = REPO_ROOT / "results" / "sensitivity" / "sample_size_n1000"
N2000_DIR = REPO_ROOT / "results" / "sensitivity" / "sample_size_n2000"

OUTPUT_DIR = REPO_ROOT / "figures"


METHOD_LABELS = {
    "proposed_full": "Proposed",
    "bayesian_gbc_tobit_no_variable_selection": "Bayesian GBC\nwithout variable\nselection",
    "bd_outcome_only_mle": "Outcome-only",
    "normal_heckman_two_step": "Heckman two-step",
    "iqbal_spike_slab_normal_class_I": "Iqbal spike-and-slab",
}

METHOD_ORDER = [
    "proposed_full",
    "bayesian_gbc_tobit_no_variable_selection",
    "bd_outcome_only_mle",
    "normal_heckman_two_step",
    "iqbal_spike_slab_normal_class_I",
]

COLORS = {
    "proposed_full": "#0072B2",
    "bayesian_gbc_tobit_no_variable_selection": "#7A7A7A",
    "bd_outcome_only_mle": "#56B4E9",
    "normal_heckman_two_step": "#CC79A7",
    "iqbal_spike_slab_normal_class_I": "#D55E00",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen result file: {path}")
    return pd.read_csv(path)


def mean_mcse(values):
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    sd = np.std(x, ddof=1) if len(x) > 1 else 0.0
    return np.mean(x), sd, sd / np.sqrt(len(x)), len(x)


def add_panel_label(ax, label):
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-22, 8),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        annotation_clip=False,
    )


def save_figure(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=OUTPUT_DIR / f"{stem}.alignment.json",
        overlay_svg=OUTPUT_DIR / f"{stem}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_main_comparison_data():
    proposed_vs = read_csv(PROPOSED_DIR / "variable_selection_by_replication.csv")
    iqbal_vs = read_csv(IQBAL_DIR / "variable_selection_by_replication.csv")
    vs = pd.concat([proposed_vs, iqbal_vs], ignore_index=True, sort=False)

    prediction_files = {
        "proposed_full": PROPOSED_DIR / "prediction_metrics_by_replication.csv",
        "bayesian_gbc_tobit_no_variable_selection": (
            NO_SELECTION_DIR / "prediction_metrics_by_replication.csv"
        ),
        "bd_outcome_only_mle": OUTCOME_ONLY_DIR / "prediction_metrics_by_replication.csv",
        "normal_heckman_two_step": HECKMAN_DIR / "prediction_metrics_by_replication.csv",
        "iqbal_spike_slab_normal_class_I": (
            IQBAL_DIR / "prediction_metrics_by_replication.csv"
        ),
    }

    prediction_parts = []
    for method, path in prediction_files.items():
        data = read_csv(path)
        data["method"] = method
        prediction_parts.append(data)
    prediction = pd.concat(prediction_parts, ignore_index=True, sort=False)

    rows = []

    for (method, equation), group in vs.groupby(["method", "equation"]):
        for metric in ["TPR", "FPR"]:
            mean, sd, se, n = mean_mcse(group[metric])
            rows.append(
                {
                    "panel": "variable_selection",
                    "method": method,
                    "equation": equation,
                    "metric": metric,
                    "mean": mean,
                    "sd": sd,
                    "mcse": se,
                    "n": n,
                }
            )

    prediction_metrics = [
        "y_latent_rmse",
        "y_latent_mae",
        "selection_brier",
        "selection_log_loss",
    ]
    for method in METHOD_ORDER:
        group = prediction[prediction["method"] == method]
        for metric in prediction_metrics:
            mean, sd, se, n = mean_mcse(group[metric])
            if n == 0:
                continue
            rows.append(
                {
                    "panel": "prediction",
                    "method": method,
                    "equation": "",
                    "metric": metric,
                    "mean": mean,
                    "sd": sd,
                    "mcse": se,
                    "n": n,
                }
            )

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "source_data_figure1.csv", index=False)
    return summary


def plot_main_comparison(summary):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    selection_methods = ["proposed_full", "iqbal_spike_slab_normal_class_I"]
    equations = ["selection", "outcome"]
    offsets = {"proposed_full": -0.10, "iqbal_spike_slab_normal_class_I": 0.10}

    for ax, metric, ylim in [
        (ax_a, "TPR", (0.90, 1.005)),
        (ax_b, "FPR", (-0.003, 0.065)),
    ]:
        for method in selection_methods:
            part = summary[
                (summary["panel"] == "variable_selection")
                & (summary["method"] == method)
                & (summary["metric"] == metric)
            ].set_index("equation")
            means = [part.loc[e, "mean"] for e in equations]
            ses = [part.loc[e, "mcse"] for e in equations]
            x = np.arange(len(equations)) + offsets[method]
            ax.errorbar(
                x,
                means,
                yerr=ses,
                fmt="o",
                ms=5,
                capsize=2.5,
                lw=1.1,
                color=COLORS[method],
                label=METHOD_LABELS[method].replace("\n", " "),
            )
        ax.set_xticks(range(len(equations)), ["Selection equation", "Outcome equation"])
        ax.set_ylabel(metric)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)

    ax_a.set_title("True-positive rate (higher is better)", loc="left")
    ax_a.legend(loc="lower right")
    ax_b.set_title("False-positive rate (lower is better)", loc="left")

    y_positions = np.arange(len(METHOD_ORDER))[::-1]
    metric_offsets = {"y_latent_rmse": 0.10, "y_latent_mae": -0.10}
    metric_markers = {"y_latent_rmse": "o", "y_latent_mae": "s"}
    metric_labels = {"y_latent_rmse": "RMSE", "y_latent_mae": "MAE"}
    for method, y in zip(METHOD_ORDER, y_positions):
        for metric in ["y_latent_rmse", "y_latent_mae"]:
            row = summary[
                (summary["method"] == method) & (summary["metric"] == metric)
            ].iloc[0]
            ax_c.errorbar(
                row["mean"],
                y + metric_offsets[metric],
                xerr=row["mcse"],
                fmt=metric_markers[metric],
                ms=4.5,
                capsize=2,
                lw=1.0,
                color=COLORS[method],
            )
    ax_c.set_yticks(
        y_positions,
        [METHOD_LABELS[m] for m in METHOD_ORDER],
    )
    for tick, method in zip(ax_c.get_yticklabels(), METHOD_ORDER):
        if method == "proposed_full":
            tick.set_color(COLORS[method])
            tick.set_fontweight("bold")
    ax_c.set_xlabel("Prediction error")
    ax_c.set_title("Latent-outcome prediction (lower is better)", loc="left")
    ax_c.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    handles = [
        plt.Line2D([], [], marker=metric_markers[m], linestyle="None", color="#333333",
                   label=metric_labels[m], markersize=4.5)
        for m in ["y_latent_rmse", "y_latent_mae"]
    ]
    ax_c.legend(handles=handles, loc="lower right")

    probability_methods = [
        "proposed_full",
        "bayesian_gbc_tobit_no_variable_selection",
        "normal_heckman_two_step",
        "iqbal_spike_slab_normal_class_I",
    ]
    y_positions_d = np.arange(len(probability_methods))[::-1]
    loss_offsets = {"selection_brier": 0.10, "selection_log_loss": -0.10}
    loss_markers = {"selection_brier": "o", "selection_log_loss": "s"}
    loss_labels = {"selection_brier": "Brier score", "selection_log_loss": "Log loss"}
    for method, y in zip(probability_methods, y_positions_d):
        for metric in ["selection_brier", "selection_log_loss"]:
            row = summary[
                (summary["method"] == method) & (summary["metric"] == metric)
            ].iloc[0]
            ax_d.errorbar(
                row["mean"],
                y + loss_offsets[metric],
                xerr=row["mcse"],
                fmt=loss_markers[metric],
                ms=4.5,
                capsize=2,
                lw=1.0,
                color=COLORS[method],
            )
    ax_d.set_yticks(
        y_positions_d,
        [METHOD_LABELS[m] for m in probability_methods],
    )
    for tick, method in zip(ax_d.get_yticklabels(), probability_methods):
        if method == "proposed_full":
            tick.set_color(COLORS[method])
            tick.set_fontweight("bold")
    ax_d.set_xlabel("Loss")
    ax_d.set_title("Selection-probability prediction (lower is better)", loc="left")
    ax_d.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    handles = [
        plt.Line2D([], [], marker=loss_markers[m], linestyle="None", color="#333333",
                   label=loss_labels[m], markersize=4.5)
        for m in ["selection_brier", "selection_log_loss"]
    ]
    ax_d.legend(handles=handles, loc="lower right")

    for ax, label in zip(axes.ravel(), ["a", "b", "c", "d"]):
        add_panel_label(ax, label)

    save_figure(fig, "figure1_main_method_comparison")


def build_sample_size_data():
    result_dirs = {500: N500_DIR, 1000: N1000_DIR, 2000: N2000_DIR}
    rows = []

    for n, folder in result_dirs.items():
        recovery = read_csv(folder / "parameter_recovery_summary.csv")
        selection = read_csv(folder / "variable_selection_summary.csv")

        for parameter in ["lambda", "sigma12", "sigma_v2"]:
            row = recovery[recovery["parameter"] == parameter].iloc[0]
            rows.append(
                {
                    "panel": "structural_rmse",
                    "n": n,
                    "series": parameter,
                    "metric": "RMSE",
                    "value": row["RMSE"],
                }
            )

        for prefix, label in [("alpha_", "Selection coefficients"), ("beta_", "Outcome coefficients")]:
            active = recovery[recovery["parameter"].isin([f"{prefix}{j}" for j in range(1, 4)])]
            rows.append(
                {
                    "panel": "coefficient_rmse",
                    "n": n,
                    "series": label,
                    "metric": "Mean active-coefficient RMSE",
                    "value": active["RMSE"].mean(),
                }
            )

        for _, row in selection.iterrows():
            rows.append(
                {
                    "panel": "selection_tpr",
                    "n": n,
                    "series": row["equation"].capitalize() + " equation",
                    "metric": "TPR",
                    "value": row["TPR"],
                }
            )

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "source_data_figure2.csv", index=False)
    return summary


def plot_sample_size_sensitivity(summary):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.84, wspace=0.34)

    labels = {"lambda": "lambda", "sigma12": "sigma12", "sigma_v2": "sigma_v2"}
    colors = {"lambda": "#0072B2", "sigma12": "#D55E00", "sigma_v2": "#009E73"}
    for series in ["lambda", "sigma12", "sigma_v2"]:
        part = summary[
            (summary["panel"] == "structural_rmse") & (summary["series"] == series)
        ].sort_values("n")
        axes[0].plot(part["n"], part["value"], marker="o", lw=1.3,
                     color=colors[series], label=labels[series])
    axes[0].set_xlabel("Training sample size")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Structural parameters", loc="left")
    axes[0].legend()

    coef_colors = {"Selection coefficients": "#0072B2", "Outcome coefficients": "#D55E00"}
    for series in ["Selection coefficients", "Outcome coefficients"]:
        part = summary[
            (summary["panel"] == "coefficient_rmse") & (summary["series"] == series)
        ].sort_values("n")
        axes[1].plot(part["n"], part["value"], marker="o", lw=1.3,
                     color=coef_colors[series], label=series)
    axes[1].set_xlabel("Training sample size")
    axes[1].set_ylabel("Mean RMSE")
    axes[1].set_title("Active regression coefficients", loc="left")
    axes[1].legend()

    eq_colors = {"Selection equation": "#0072B2", "Outcome equation": "#D55E00"}
    for series in ["Selection equation", "Outcome equation"]:
        part = summary[
            (summary["panel"] == "selection_tpr") & (summary["series"] == series)
        ].sort_values("n")
        axes[2].plot(part["n"], part["value"], marker="o", lw=1.3,
                     color=eq_colors[series], label=series)
    axes[2].set_xlabel("Training sample size")
    axes[2].set_ylabel("TPR")
    axes[2].set_ylim(0.94, 1.005)
    axes[2].set_title("Variable-selection recovery", loc="left")
    axes[2].legend(loc="lower right")

    for ax, label in zip(axes, ["a", "b", "c"]):
        ax.set_xticks([500, 1000, 2000])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        add_panel_label(ax, label)

    save_figure(fig, "figure2_sample_size_sensitivity")


def main():
    main_summary = build_main_comparison_data()
    plot_main_comparison(main_summary)

    sample_summary = build_sample_size_data()
    plot_sample_size_sensitivity(sample_summary)

    print(f"Completed. Figures and source data were saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

