from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import (BoundaryNorm, LinearSegmentedColormap,
                               ListedColormap)

MM = 1 / 25.4
SINGLE_COL = 89 * MM
DOUBLE_COL = 183 * MM

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#ffffff"
LAND = "#f2f1ec"

CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
       "#e87ba4", "#eb6834"]
YEAR_COLORS = {"2010": CAT[0], "2022": CAT[5]}

REGION_COLORS = {"Costa": "#2a78d6", "Sierra": "#c98500", "Amazonía": "#199e70"}

SEQ_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS)

ORDINAL_5 = ["#6ece58", "#26ad81", "#26828e", "#3c4f8a", "#46085c"]
ORDINAL_CMAP = ListedColormap(ORDINAL_5)

GJSD_BINS = [0.0, 0.025, 0.05, 0.1, 0.2, 0.4]
GJSD_CMAP = ListedColormap(ORDINAL_5[::-1])
GJSD_NORM = BoundaryNorm(GJSD_BINS, ncolors=len(ORDINAL_5))

DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red",
    ["#104281", "#5598e7", "#b7d3f6", "#f0efec", "#f2b3b2", "#d96666", "#a02c2b"])

STYLE = {
    "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.4,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42, 
    "ps.fonttype": 42,
}


def use_style() -> None:
    mpl.rcParams.update(STYLE)


def level_label(threshold_m: float) -> str:
    if threshold_m < 1000:
        return f"{threshold_m:g} m"
    return f"{threshold_m / 1000:.2f} km"


def scale_class(threshold_m: float, levels_m) -> int:
    return int(np.searchsorted(sorted(levels_m)[1:5], threshold_m, side="left"))


def scale_class_labels(levels_m) -> list[str]:
    lv = sorted(levels_m)
    return ([f"≤ {level_label(lv[1])}"] + [level_label(t) for t in lv[2:5]]
            + [f"≥ {level_label(lv[5])}"])


def panel_label(ax, letter: str, title: str = "", pad: float = 2.0) -> None:
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, pad), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.5, fontweight="bold",
                color=INK, annotation_clip=False)
    if title:
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction",
                    xytext=(11, pad), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8, color=INK,
                    annotation_clip=False)


def scale_bar(ax, km: int = 100, x0: float = -81.0, y0: float = -4.9) -> None:
    deg = km / (111.32 * np.cos(np.radians(y0)))
    ax.plot([x0, x0 + deg], [y0, y0], color=INK_2, lw=1.0,
            solid_capstyle="butt", zorder=5)
    for x in (x0, x0 + deg):
        ax.plot([x, x], [y0 - 0.045, y0 + 0.045], color=INK_2, lw=0.8, zorder=5)
    ax.annotate(f"{km} km", (x0 + deg / 2, y0 + 0.07), ha="center", va="bottom",
                fontsize=6, color=INK_2)


def _style_legend(leg) -> None:
    leg.get_title().set_color(INK_2)
    for t in leg.get_texts():
        t.set_color(INK_2)


def swatch_legend(ax, entries, size: float = 22, **kw):
    handles = [plt.scatter([], [], s=size, color=c, edgecolor=SURFACE,
                           linewidth=0.3, label=lab) for c, lab in entries]
    leg = ax.legend(handles=handles, frameon=False, fontsize=6.5,
                    handletextpad=0.4, borderpad=0.0, **kw)
    _style_legend(leg)
    return leg


def size_legend(ax, area, entries, **kw):
    handles = [plt.scatter([], [], s=area(v), color=BASELINE, edgecolor=SURFACE,
                           linewidth=0.3, label=lab) for v, lab in entries]
    leg = ax.legend(handles=handles, frameon=False, fontsize=6.5,
                    title_fontsize=6.5, labelspacing=0.9, handletextpad=0.4,
                    borderpad=0.0, **kw)
    _style_legend(leg)
    return leg


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.(pdf|png)", flush=True)


def tree_positions(G, root: str, sort_key=None):
    if sort_key is None:
        def sort_key(n):
            return -G.nodes[n].get("n_nodes", 0)

    leaves_order = []

    def walk(n):
        children = sorted(G.successors(n), key=sort_key)
        if not children:
            leaves_order.append(n)
            return
        for c in children:
            walk(c)

    walk(root)
    pos = {leaf: (i, G.nodes[leaf]["level"]) for i, leaf in enumerate(leaves_order)}

    def place(n):
        if n in pos:
            return pos[n][0]
        xs = [place(c) for c in G.successors(n)]
        x = float(np.mean(xs))
        pos[n] = (x, G.nodes[n]["level"])
        return x

    place(root)
    return pos
