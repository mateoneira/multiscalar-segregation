"""Robustness analyses: permutation null model, group-count and threshold
sensitivity (SI section 'Robustness and sensitivity analyses')."""

from __future__ import annotations

import numpy as np
import pandas as pd


def permuted_unit_counts(unit_counts: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One exact permutation of dwelling group labels across the country.

    Keeps every unit's dwelling count fixed and reassigns group labels by
    drawing without replacement from the pooled national label multiset.
    """
    counts = unit_counts.to_numpy()
    n_units, n_groups = counts.shape
    sizes = counts.sum(axis=1).astype(np.int64)
    labels = np.repeat(np.arange(n_groups, dtype=np.int8),
                       counts.sum(axis=0).astype(np.int64))
    rng.shuffle(labels)
    unit_of_dwelling = np.repeat(np.arange(n_units), sizes)
    out = np.zeros((n_units, n_groups), dtype=np.int64)
    np.add.at(out, (unit_of_dwelling, labels), 1)
    return pd.DataFrame(out, index=unit_counts.index, columns=unit_counts.columns)


def null_distribution(
    G,
    unit_labels: dict[int, pd.Series],
    unit_counts: pd.DataFrame,
    n_permutations: int,
    seed: int,
    progress: bool = True,
) -> pd.DataFrame:
    """Per-node null distribution of norm-GJSD under label permutation.

    Returns a DataFrame indexed by node with columns: observed, null_mean,
    null_lo, null_hi (2.5/97.5 percentiles), p_value (one-sided, upper).
    """
    from . import analysis

    observed = analysis.gjsd_table(G, unit_labels, unit_counts)["norm_gjsd"]
    rng = np.random.default_rng(seed)
    draws = np.full((n_permutations, len(observed)), np.nan)
    node_order = observed.index
    for i in range(n_permutations):
        perm = permuted_unit_counts(unit_counts, rng)
        t = analysis.gjsd_table(G, unit_labels, perm)["norm_gjsd"]
        draws[i] = t.reindex(node_order).to_numpy()
        if progress and (i + 1) % 50 == 0:
            print(f"  permutation {i + 1}/{n_permutations}", flush=True)

    with np.errstate(invalid="ignore"):
        exceed = (draws >= observed.to_numpy()).sum(axis=0)
    n_valid = np.isfinite(draws).sum(axis=0)
    return pd.DataFrame({
        "observed": observed,
        "null_mean": np.nanmean(draws, axis=0),
        "null_lo": np.nanpercentile(draws, 2.5, axis=0),
        "null_hi": np.nanpercentile(draws, 97.5, axis=0),
        "p_value": (exceed + 1) / (n_valid + 1),
    }, index=node_order)


class GjsdEngine:
    """Vectorised norm-GJSD over the tree for repeated count matrices.

    Precomputes, once, which unit rows feed which (parent, group) cell —
    children of each parent plus its remainder pool.
    """

    def __init__(self, G, unit_labels: dict[int, pd.Series], unit_counts_index: pd.Index):
        from . import tree as tree_mod

        row_of_unit = pd.Series(np.arange(len(unit_counts_index)), index=unit_counts_index)
        assign_rows, assign_groups = [], []
        group_parent: list[str] = []
        parents: list[str] = []
        g_id = 0
        for parent in G.nodes:
            level = G.nodes[parent]["level"]
            if level not in unit_labels:
                continue
            lab = unit_labels[level]
            parent_units = lab.index[lab == parent]
            if len(parent_units) == 0:
                continue
            children = list(G.successors(parent))
            covered = pd.Index([])
            start_g = g_id
            if children and (level - 1) in unit_labels:
                child_lab = unit_labels[level - 1]
                for child in children:
                    child_units = child_lab.index[child_lab == child]
                    covered = covered.union(child_units)
                    rows = row_of_unit.reindex(child_units).dropna()
                    if len(rows):
                        assign_rows.append(rows.to_numpy(dtype=np.int64))
                        assign_groups.append(np.full(len(rows), g_id, dtype=np.int64))
                        group_parent.append(parent)
                        g_id += 1
            remainder = parent_units.difference(covered)
            rows = row_of_unit.reindex(remainder).dropna()
            if len(rows):
                assign_rows.append(rows.to_numpy(dtype=np.int64))
                assign_groups.append(np.full(len(rows), g_id, dtype=np.int64))
                group_parent.append(parent)
                g_id += 1
            if g_id > start_g:
                parents.append(parent)

        self.rows = np.concatenate(assign_rows)
        self.groups = np.concatenate(assign_groups)
        self.n_groups_total = g_id
        self.group_parent = np.asarray(group_parent)
        self.parents = pd.Index(parents, name="node")
        self._parent_code = pd.Series(np.arange(len(self.parents)), index=self.parents)
        self._group_parent_code = self._parent_code.loc[self.group_parent].to_numpy()

    def norm_gjsd(self, counts: np.ndarray) -> pd.Series:
        """Normalised GJSD per parent for a (units x groups) count matrix."""
        cell = np.zeros((self.n_groups_total, counts.shape[1]))
        np.add.at(cell, self.groups, counts[self.rows])

        totals = cell.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = cell / totals[:, None]
            h_cell = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)

        n_parents = len(self.parents)
        parent_tot = np.zeros(n_parents)
        np.add.at(parent_tot, self._group_parent_code, totals)
        parent_counts = np.zeros((n_parents, counts.shape[1]))
        np.add.at(parent_counts, self._group_parent_code, cell)
        with np.errstate(divide="ignore", invalid="ignore"):
            pm = parent_counts / parent_tot[:, None]
            h_parent = -np.nansum(np.where(pm > 0, pm * np.log(pm), 0.0), axis=1)

        w = np.where(parent_tot[self._group_parent_code] > 0,
                     totals / parent_tot[self._group_parent_code], 0.0)
        wh = np.zeros(n_parents)
        np.add.at(wh, self._group_parent_code, np.where(totals > 0, w * h_cell, 0.0))
        upper = np.zeros(n_parents)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.add.at(upper, self._group_parent_code,
                      np.where(w > 0, -w * np.log(w), 0.0))
        n_pop_groups = np.zeros(n_parents)
        np.add.at(n_pop_groups, self._group_parent_code, (totals > 0).astype(float))

        raw = h_parent - wh
        out = np.where((n_pop_groups > 1) & (upper > 0), raw / upper, np.nan)
        return pd.Series(out, index=self.parents, name="norm_gjsd")


def regroup_unit_counts(dwellings_icv: pd.Series, unit_of_dwelling: pd.Series,
                        n_groups: int) -> pd.DataFrame:
    """Unit group counts for an alternative number of quantile groups."""
    from . import metrics

    edges = metrics.quantile_edges(dwellings_icv.to_numpy(), n_groups)
    idx = np.clip(np.searchsorted(edges, dwellings_icv.to_numpy(), side="right") - 1,
                  0, n_groups - 1)
    df = pd.DataFrame({"unit": unit_of_dwelling.to_numpy(), "g": idx})
    counts = df.pivot_table(index="unit", columns="g", aggfunc="size", fill_value=0)
    counts = counts.reindex(columns=range(n_groups), fill_value=0)
    counts.columns = [f"g{c}" for c in counts.columns]
    return counts
