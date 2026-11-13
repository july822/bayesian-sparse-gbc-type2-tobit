"""Generate the model-structure diagram."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_STEM = REPO_ROOT / "figures" / "figure_model_structure"


sys.path.insert(0, str(SCRIPT_DIR))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


BLUE = "#0072B2"
LIGHT_BLUE = "#E8F3FA"
ORANGE = "#D55E00"
LIGHT_ORANGE = "#FCEFE7"
GREEN = "#009E73"
LIGHT_GREEN = "#E8F6F1"
PURPLE = "#7A4EAB"
LIGHT_PURPLE = "#F1ECF7"
GREY = "#5A5A5A"
LIGHT_GREY = "#F4F4F4"
DARK = "#202020"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.0,
        "axes.linewidth": 0.8,
    }
)


def add_node(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 8.0,
    linewidth: float = 1.15,
    textcolor: str = DARK,
    zorder: int = 3,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        linespacing=1.22,
        zorder=zorder + 1,
    )
    return {"x": x, "y": y, "w": width, "h": height}


def anchor(node, side: str, fraction: float = 0.5):
    if side == "left":
        return node["x"], node["y"] + fraction * node["h"]
    if side == "right":
        return node["x"] + node["w"], node["y"] + fraction * node["h"]
    if side == "top":
        return node["x"] + fraction * node["w"], node["y"] + node["h"]
    if side == "bottom":
        return node["x"] + fraction * node["w"], node["y"]
    raise ValueError(f"Unknown side: {side}")


def add_arrow(
    ax,
    start,
    end,
    *,
    color: str = GREY,
    linestyle: str = "-",
    linewidth: float = 1.15,
    mutation_scale: float = 9.0,
    connectionstyle: str = "arc3,rad=0",
    zorder: int = 2,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=2.0,
        shrinkB=2.0,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def build_figure():
    
    fig, ax = plt.subplots(figsize=(7.20, 4.22), constrained_layout=False)
    fig.subplots_adjust(left=0.018, right=0.982, bottom=0.055, top=0.955)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    
    ax.text(0.245, 0.966, "Selection equation", ha="center", va="top",
            fontsize=10.0, fontweight="bold", color=BLUE)
    ax.text(0.755, 0.966, "Outcome equation", ha="center", va="top",
            fontsize=10.0, fontweight="bold", color=ORANGE)

    sel_prior = add_node(
        ax, 0.055, 0.785, 0.31, 0.105,
        r"Spike--and--slab prior" + "\n" +
        r"$(\gamma_{\alpha,j},p_\alpha)\ \longrightarrow\ \alpha_j$",
        facecolor=LIGHT_BLUE, edgecolor=BLUE, fontsize=8.2,
    )
    out_prior = add_node(
        ax, 0.635, 0.785, 0.31, 0.105,
        r"Spike--and--slab prior" + "\n" +
        r"$(\gamma_{\beta,j},p_\beta)\ \longrightarrow\ \beta_j$",
        facecolor=LIGHT_ORANGE, edgecolor=ORANGE, fontsize=8.2,
    )

    sel_cov = add_node(
        ax, 0.040, 0.625, 0.165, 0.09,
        r"Selection covariates" + "\n" + r"$\mathbf{w}_i$",
        facecolor="white", edgecolor=BLUE, fontsize=8.0,
    )
    sel_eq = add_node(
        ax, 0.240, 0.600, 0.225, 0.14,
        r"Latent selection" + "\n" +
        r"$z_i^*=\mathbf{w}_i^\top\boldsymbol{\alpha}+\epsilon_{1i}$",
        facecolor=LIGHT_BLUE, edgecolor=BLUE, fontsize=8.2,
    )

    out_cov = add_node(
        ax, 0.795, 0.625, 0.165, 0.09,
        r"Outcome covariates" + "\n" + r"$\mathbf{x}_i$",
        facecolor="white", edgecolor=ORANGE, fontsize=8.0,
    )
    out_eq = add_node(
        ax, 0.535, 0.585, 0.225, 0.165,
        r"Transformed latent outcome" + "\n" +
        r"$t_i^*=\mathbf{x}_i^\top\boldsymbol{\beta}$" + "\n" +
        r"$\quad+\sigma_{12}\epsilon_{1i}+v_i$" + "\n" +
        r"$v_i\sim\mathcal{N}(0,\sigma_v^2)$",
        facecolor=LIGHT_ORANGE, edgecolor=ORANGE, fontsize=7.55,
    )

    
    shared_error = add_node(
        ax, 0.405, 0.795, 0.19, 0.085,
        r"Shared selection shock" + "\n" +
        r"$\epsilon_{1i}\sim\mathcal{N}(0,1)$",
        facecolor=LIGHT_PURPLE, edgecolor=PURPLE, fontsize=7.9,
    )
    select_rule = add_node(
        ax, 0.115, 0.435, 0.25, 0.105,
        r"Observation indicator" + "\n" +
        r"$d_i=\mathbb{I}(z_i^*>0)$",
        facecolor=LIGHT_GREEN, edgecolor=GREEN, fontsize=8.3,
    )
    inverse_transform = add_node(
        ax, 0.635, 0.385, 0.325, 0.125,
        r"Inverse generalized Box--Cox transform" + "\n" +
        r"$y_i^*=t^{-1}(t_i^*;\lambda)$",
        facecolor=LIGHT_GREEN, edgecolor=GREEN, fontsize=8.1,
    )
    lambda_node = add_node(
        ax, 0.820, 0.535, 0.166, 0.056,
        r"Transformation parameter" + "\n" + r"$\lambda$",
        facecolor=LIGHT_PURPLE, edgecolor=PURPLE, fontsize=8.0,
    )

    observed = add_node(
        ax, 0.325, 0.215, 0.35, 0.115,
        r"Observed-data mechanism" + "\n" +
        r"$y_i=y_i^*$ if $d_i=1$; unobserved if $d_i=0$",
        facecolor=LIGHT_GREY, edgecolor=GREY, fontsize=8.2, linewidth=1.25,
    )

    outputs = add_node(
        ax, 0.08, 0.035, 0.84, 0.095,
        r"Posterior summaries: equation-specific PIPs"
        r" $\;\vert\;$ latent-outcome predictions"
        r" $\;\vert\;$ model-based marginal effects",
        facecolor="white", edgecolor=DARK, fontsize=8.2, linewidth=1.25,
    )

    
    add_arrow(ax, anchor(sel_prior, "bottom", 0.72), anchor(sel_eq, "top", 0.35), color=BLUE)
    add_arrow(ax, anchor(out_prior, "bottom", 0.28), anchor(out_eq, "top", 0.65), color=ORANGE)
    add_arrow(ax, anchor(sel_cov, "right"), anchor(sel_eq, "left"), color=BLUE)
    add_arrow(ax, anchor(out_cov, "left"), anchor(out_eq, "right"), color=ORANGE)

    
    
    add_arrow(
        ax, anchor(shared_error, "bottom", 0.28), anchor(sel_eq, "top", 0.72),
        color=PURPLE, connectionstyle="arc3,rad=0.12",
    )
    add_arrow(
        ax, anchor(shared_error, "bottom", 0.72), anchor(out_eq, "top", 0.32),
        color=PURPLE, connectionstyle="arc3,rad=-0.12",
    )

    
    add_arrow(ax, anchor(sel_eq, "bottom", 0.38), anchor(select_rule, "top", 0.62), color=GREEN)
    add_arrow(ax, anchor(out_eq, "bottom", 0.68), anchor(inverse_transform, "top", 0.30), color=GREEN)
    add_arrow(ax, anchor(lambda_node, "bottom"), anchor(inverse_transform, "top", 0.78), color=PURPLE)
    add_arrow(
        ax, anchor(select_rule, "bottom", 0.65), anchor(observed, "top", 0.22),
        color=GREEN, connectionstyle="arc3,rad=-0.08",
    )
    add_arrow(
        ax, anchor(inverse_transform, "bottom", 0.25), anchor(observed, "top", 0.78),
        color=GREEN, connectionstyle="arc3,rad=0.08",
    )
    add_arrow(ax, anchor(observed, "bottom"), anchor(outputs, "top"), color=DARK, linewidth=1.25)

    
    ax.text(0.013, 0.725, "Joint latent model", rotation=90,
            rotation_mode="anchor", ha="center", va="center",
            fontsize=7.5, color=GREY, fontweight="bold")

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(OUTPUT_STEM) + ".alignment.json",
        overlay_svg=str(OUTPUT_STEM) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
    )
    return fig


def main():
    fig = build_figure()
    fig.savefig(str(OUTPUT_STEM) + ".pdf", bbox_inches="tight")
    fig.savefig(str(OUTPUT_STEM) + ".svg", bbox_inches="tight")
    fig.savefig(str(OUTPUT_STEM) + ".tiff", dpi=600, bbox_inches="tight")
    fig.savefig(str(OUTPUT_STEM) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Created {OUTPUT_STEM}.pdf/.svg/.tiff/.png")


if __name__ == "__main__":
    main()
