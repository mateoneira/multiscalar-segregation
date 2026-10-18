"""Experiment: objective detection of the percolation phase transitions.

The hierarchy levels in config.yaml (levels_m) were fixed by visually reading
jumps off the largest-component curve, following Arcaute et al. (2016). This
experiment locates the transitions with the standard finite-size percolation
diagnostics instead:

- susceptibility  chi(d) = sum_{i>=2} s_i^2 / N  (largest cluster excluded;
  Stauffer & Aharony) -- local maxima mark the transitions;
- second-largest component S2(d), which peaks when two macroscopic clusters
  are about to merge (da Silva, Lyra & Viswanathan 2002);
- exact macroscopic merge events (unions of two clusters both >= 600 nodes),
  the grid-free discrete analogue of the chi/S2 peaks;
- jump sizes dS1 in the largest component (explosive-percolation gap method).

Candidates are chi peaks, cross-validated against S2 peaks and merge events,
ranked by the size of the smaller merging cluster (how macroscopic the merge
was) and by chi prominence. A transition is retained as significant when the
smaller merging cluster holds >= 1% of all street nodes. Each retained d_c is
then checked against the critical cluster-size statistics: at a genuine 2D
percolation transition n_s ~ s^-tau with tau = 187/91 ~ 2.05.

Writes data/processed/{transition_diagnostics,detected_transitions}.csv and
figures/transition_{diagnostics,tau_check}.(pdf|png). Does NOT modify the
pipeline outputs or config.

Usage: python pipeline/experiments/detect_transitions.py [--recompute]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from multiscalar_segregation import load_config, percolation, repo_root, viz

CFG = load_config()
ROOT = repo_root()
PROCESSED = ROOT / CFG["paths"]["data_processed"]
FIGS = ROOT / CFG["paths"]["figures"]
CACHE = ROOT / CFG["paths"]["data_interim"] / "experiments"

STEP = float(CFG["percolation"]["sweep_step_m"])
SWEEP_MAX = float(CFG["percolation"]["sweep_max_m"])
MIN_CLUSTER = int(CFG["percolation"]["min_cluster_nodes"])
CURRENT_LEVELS = [float(t) for t in CFG["percolation"]["levels_m"]]

SIGNIFICANT_FRACTION = 0.01
MATCH_WINDOW_M = 30.0
TAU_2D = 187 / 91  # Fisher exponent, 2D percolation universality class


def load_network():
    nodes = pd.read_csv(ROOT / CFG["network"]["nodes"], header=None,
                        names=["node_id", "lon", "lat"])
    edges, edge_universe = percolation.load_edgelist(str(ROOT / CFG["network"]["edgelist"]))
    universe = np.union1d(nodes["node_id"].values, edge_universe)
    print(f"nodes: {len(universe):,}  edges: {len(edges):,}", flush=True)
    return edges, universe


def sweep(edges, universe, recompute: bool):
    """Full diagnostics sweep (cached: the union-find pass takes minutes)."""
    curve_path, merges_path = CACHE / "diagnostics_curve.csv", CACHE / "merge_events.csv"
    if curve_path.exists() and merges_path.exists() and not recompute:
        print(f"using cached sweep in {CACHE}", flush=True)
        return pd.read_csv(curve_path), pd.read_csv(merges_path)
    res = percolation.percolate(edges, universe, snapshot_thresholds=[],
                                curve_step=STEP, curve_max=SWEEP_MAX,
                                merge_min_size=MIN_CLUSTER)
    CACHE.mkdir(parents=True, exist_ok=True)
    res.curve.to_csv(curve_path, index=False)
    res.merges.to_csv(merges_path, index=False)
    return res.curve, res.merges


def detect(curve: pd.DataFrame, merges: pd.DataFrame, n_nodes: int) -> pd.DataFrame:
    """Candidate transitions: chi peaks, corroborated by S2 peaks and merges."""
    d = curve["threshold"].values
    chi = curve["susceptibility"].values
    s2 = curve["second_relative"].values

    min_prom = MIN_CLUSTER**2 / n_nodes
    chi_peaks, chi_props = find_peaks(chi, prominence=min_prom, plateau_size=(1, None))
    chi_dc = d[chi_props["right_edges"]]
    s2_peaks, s2_props = find_peaks(s2, prominence=MIN_CLUSTER / n_nodes,
                                    plateau_size=(1, None))
    s2_dc = d[s2_props["right_edges"]]

    ds1 = np.diff(curve["largest_component"].values, prepend=1) / n_nodes

    rows = []
    for i, dc in enumerate(chi_dc):
        win = merges[(merges["length"] > dc) & (merges["length"] <= dc + MATCH_WINDOW_M)]
        nearest_s2 = s2_dc[np.abs(s2_dc - dc).argmin()] if len(s2_dc) else np.nan
        grid_i = int(np.searchsorted(d, dc))
        rows.append({
            "d_c": dc,
            "chi": chi[grid_i],
            "chi_prominence": chi_props["prominences"][i],
            "s2_share": s2[grid_i],
            "s2_peak_within_window": bool(len(s2_dc)) and abs(nearest_s2 - dc) <= MATCH_WINDOW_M,
            "merge_length": win["length"].iloc[0] if len(win) else np.nan,
            "merge_size_large": win["size_large"].iloc[0] if len(win) else np.nan,
            "merge_size_small": win["size_small"].max() if len(win) else np.nan,
            "jump_next_step": ds1[min(grid_i + 1, len(ds1) - 1)],
        })
    cand = pd.DataFrame(rows)
    cand["small_frac"] = cand["merge_size_small"] / n_nodes
    cand["significant"] = cand["small_frac"] >= SIGNIFICANT_FRACTION
    cand["nearest_current_level"] = [
        CURRENT_LEVELS[int(np.argmin(np.abs(np.array(CURRENT_LEVELS) - dc)))]
        for dc in cand["d_c"]]
    return cand.sort_values("chi_prominence", ascending=False).reset_index(drop=True)


def subcritical_merges(merges: pd.DataFrame, detected_dc, n_nodes: int) -> pd.DataFrame:
    """Macroscopic merges NOT associated with a chi peak.

    Below the first system-wide transition chi still rises monotonically (the
    subcritical growth regime), so peak detection is blind there; the exact
    union-find merge log is not. These events are objective anchors for the
    finer (intra-urban) hierarchy levels, ranked by how macroscopic they are.
    """
    m = merges[merges["size_small"] >= SIGNIFICANT_FRACTION * n_nodes].copy()
    m["small_frac"] = m["size_small"] / n_nodes
    detected_dc = np.asarray(list(detected_dc), dtype=float)
    if len(detected_dc):
        gap = np.abs(m["length"].values[:, None] - detected_dc[None, :])
        near_peak = (gap <= MATCH_WINDOW_M).any(axis=1)
        m = m[~near_peak]
    return m.reset_index(drop=True)


def fig_diagnostics(curve, merges, cand, subcrit, n_nodes):
    """Stacked diagnostics panels on a shared log-distance axis."""
    detected = cand[cand["significant"]].sort_values("d_c")
    fig, axes = plt.subplots(4, 1, figsize=(viz.DOUBLE_COL, 6.4), sharex=True,
                             gridspec_kw={"hspace": 0.32})
    d = curve["threshold"].values

    def mark_transitions(ax, label_them=False):
        for t in CURRENT_LEVELS:
            ax.axvline(t, color=viz.BASELINE, lw=0.6, zorder=0)
        for i, (_, r) in enumerate(detected.iterrows()):
            ax.axvline(r["d_c"], color=viz.CAT[5], lw=0.7, ls=(0, (4, 2)), zorder=1)
            if label_them:
                ax.annotate(f"{r['d_c']:,.0f}", xy=(r["d_c"], 1.0),
                            xycoords=("data", "axes fraction"),
                            xytext=(0, 3 + 8 * (i % 3)), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6, color=viz.CAT[5],
                            annotation_clip=False)

    ax = axes[0]
    ax.plot(d, curve["largest_relative"], color=viz.CAT[0], lw=1.4,
            solid_capstyle="round")
    mark_transitions(ax, label_them=True)
    ax.scatter(subcrit["length"], np.full(len(subcrit), 0.03), marker="v", s=10,
               color=viz.CAT[2], linewidths=0, zorder=3, clip_on=False,
               label="macroscopic merge without $\\chi$ peak")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), handletextpad=0.2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("$P_\\infty$ = $s_1/N$")
    viz.panel_label(ax, "a", "order parameter: largest component", pad=14)

    ax = axes[1]
    ax.plot(d, curve["second_relative"], color=viz.CAT[1], lw=1.2,
            solid_capstyle="round")
    mark_transitions(ax)
    ax.set_ylabel("$s_2/N$")
    viz.panel_label(ax, "b", "second-largest component (peaks at transitions)")

    ax = axes[2]
    ax.plot(d, curve["susceptibility"], color=viz.CAT[0], lw=1.2,
            solid_capstyle="round")
    mark_transitions(ax)
    ax.set_yscale("log")
    ax.set_ylabel("$\\chi$ = $\\Sigma_{i\\geq 2}\\,s_i^2/N$")
    viz.panel_label(ax, "c", "susceptibility, largest cluster excluded")

    ax = axes[3]
    ds1 = np.diff(curve["largest_component"].values, prepend=1) / n_nodes
    pos = ds1 > 0
    ax.vlines(d[pos], 1e-6, ds1[pos], color=viz.MUTED, lw=0.5)
    macro = merges[merges["size_small"] / n_nodes >= SIGNIFICANT_FRACTION]
    ax.scatter(merges["length"], merges["size_small"] / n_nodes, s=4,
               color=viz.CAT[2], linewidths=0, zorder=3, label="merge event")
    ax.scatter(macro["length"], macro["size_small"] / n_nodes, s=12,
               color=viz.CAT[5], linewidths=0, zorder=4,
               label=f"macroscopic ($\\geq$ {SIGNIFICANT_FRACTION:.0%} of nodes)")
    ax.axhline(SIGNIFICANT_FRACTION, color=viz.BASELINE, lw=0.6, ls=":")
    mark_transitions(ax)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set_ylabel("$\\Delta s_1/N$, merge size$/N$")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2,
              handletextpad=0.2, columnspacing=1.0, borderaxespad=0.0)
    viz.panel_label(ax, "d", "largest-component jumps and exact merge events")

    ax.set_xscale("log")
    ax.set_xlim(100, SWEEP_MAX)
    ax.set_xlabel("percolation threshold $\\delta$ (m)")
    viz.save(fig, FIGS, "transition_diagnostics")


def fig_tau_check(edges, universe, detected_dc):
    """Cluster-size distributions at each detected d_c vs the 2D critical law."""
    res = percolation.percolate(edges, universe,
                                snapshot_thresholds=list(detected_dc),
                                curve_step=STEP, curve_max=SWEEP_MAX)
    fig, ax = plt.subplots(figsize=(viz.SINGLE_COL, 2.6))
    shades = viz.SEQ(np.linspace(0.35, 1.0, len(detected_dc)))
    bins = np.logspace(0, 6, 25)
    centers = np.sqrt(bins[:-1] * bins[1:])
    for dc, color in zip(detected_dc, shades):
        sizes = res.cluster_sizes(dc).values
        sizes = sizes[sizes < sizes.max()]
        counts, _ = np.histogram(sizes, bins=bins)
        density = counts / np.diff(bins)
        m = density > 0
        ax.plot(centers[m], density[m], color=color, lw=1.0, marker="o",
                markersize=2.4, markeredgewidth=0,
                label=f"{dc:,.0f} m" if dc < 1000 else f"{dc / 1000:g} km")
    guide_x = np.array([10, 1e4])
    guide_y = 3e3 * guide_x**-TAU_2D
    ax.plot(guide_x, guide_y, color=viz.INK_2, lw=0.8, ls="--")
    gx_lab = 2.0e3
    ax.annotate(f"$s^{{-\\tau}}$, $\\tau = 187/91 \\approx {TAU_2D:.2f}$",
                xy=(gx_lab, 3e3 * gx_lab**-TAU_2D), xytext=(0, -8),
                textcoords="offset points", ha="right", va="top", fontsize=6.5,
                color=viz.INK_2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cluster size $s$ (nodes)")
    ax.set_ylabel("$n_s$ (clusters per unit size)")
    ax.legend(title="$\\delta_c$", ncols=2, handlelength=1.2,
              handletextpad=0.4, columnspacing=0.8)
    viz.save(fig, FIGS, "transition_tau_check")


def main() -> None:
    recompute = "--recompute" in sys.argv[1:]
    viz.use_style()
    edges, universe = load_network()
    n = len(universe)

    curve, merges = sweep(edges, universe, recompute)
    curve.to_csv(PROCESSED / "transition_diagnostics.csv", index=False)

    cand = detect(curve, merges, n)
    detected = cand[cand["significant"]].sort_values("d_c")
    subcrit = subcritical_merges(merges, detected["d_c"], n)

    combined = pd.concat([
        detected.assign(type="critical (chi peak)").rename(columns={"d_c": "d"}),
        subcrit.assign(type="macroscopic merge").rename(columns={"length": "d"}),
    ], ignore_index=True).sort_values("d")
    combined["nearest_current_level"] = [
        CURRENT_LEVELS[int(np.argmin(np.abs(np.array(CURRENT_LEVELS) - dv)))]
        for dv in combined["d"]]
    combined.to_csv(PROCESSED / "detected_transitions.csv", index=False)

    pd.set_option("display.width", 160)
    print(f"\n{len(cand)} candidate transitions (chi peaks); "
          f"{len(detected)} significant (smaller merging cluster >= "
          f"{SIGNIFICANT_FRACTION:.0%} of {n:,} nodes)\n")
    print("Top 15 by chi prominence:")
    print(cand.head(15).to_string(
        float_format=lambda v: f"{v:,.4g}",
        columns=["d_c", "chi", "chi_prominence", "s2_share", "s2_peak_within_window",
                 "merge_length", "merge_size_small", "small_frac", "significant",
                 "nearest_current_level"]))
    print("\nCritical transitions (chi peak + macroscopic merge) vs pinned levels "
          f"({', '.join(f'{t:,.0f}' for t in CURRENT_LEVELS)} m):")
    for _, r in detected.iterrows():
        print(f"  d_c = {r['d_c']:>7,.0f} m   merge at {r['merge_length']:>9,.1f} m   "
              f"smaller cluster {r['merge_size_small']:>9,.0f} nodes "
              f"({r['small_frac']:.1%})   nearest level {r['nearest_current_level']:,.0f} m")
    print("\nMacroscopic merges without a chi peak (subcritical/growth regime; "
          "exact union-find events, smaller cluster >= 1%):")
    for _, r in subcrit.iterrows():
        print(f"  d   = {r['length']:>7,.0f} m   {r['size_large']:>9,.0f} + "
              f"{r['size_small']:>7,.0f} nodes ({r['small_frac']:.1%})")

    fig_diagnostics(curve, merges, cand, subcrit, n)
    fig_tau_check(edges, universe, detected["d_c"].values)
    print("\nwrote data/processed/transition_diagnostics.csv, "
          "detected_transitions.csv and figures/transition_*.pdf|png")


if __name__ == "__main__":
    sys.exit(main())
