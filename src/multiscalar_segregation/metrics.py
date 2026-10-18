"""Information-theoretic segregation metrics.

The Generalised Jensen-Shannon Divergence (GJSD) of the child compositions of a
tree node equals the mutual information between child membership and group
membership for that node's population. All
entropies use natural logarithms (nats).
"""

from __future__ import annotations

import numpy as np


def entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats of a probability vector; 0*log(0) is taken as 0."""
    p = np.asarray(p, dtype=float)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def miller_madow_entropy(counts: np.ndarray) -> float:
    """Miller-Madow bias-corrected entropy estimate from raw counts (nats).

    H_MM = H_plugin + (K_observed - 1) / (2 N), where K_observed is the number
    of bins with non-zero counts and N the total count.
    """
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if n == 0:
        return 0.0
    k_obs = int((counts > 0).sum())
    return entropy(counts / n) + (k_obs - 1) / (2 * n)


def gjsd(distributions: np.ndarray, weights: np.ndarray) -> float:
    """Generalised Jensen-Shannon Divergence.

    Parameters
    ----------
    distributions : (C, B) array, one probability vector per child node.
    weights : (C,) array, population shares pi_c of the children (sum to 1).
    """
    P = np.asarray(distributions, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    if P.ndim != 2 or w.shape[0] != P.shape[0]:
        raise ValueError(f"shape mismatch: distributions {P.shape}, weights {w.shape}")
    mixture = w @ P
    return entropy(mixture) - float(w @ np.array([entropy(p) for p in P]))


def gjsd_max(weights: np.ndarray) -> float:
    """Upper bound of the GJSD for given weights: the entropy of the weights."""
    return entropy(np.asarray(weights, dtype=float).ravel())


def norm_gjsd(distributions: np.ndarray, weights: np.ndarray) -> float:
    """GJSD normalised by its upper bound; 0 when the bound is 0 (single child)."""
    upper = gjsd_max(weights)
    if upper == 0.0:
        return 0.0
    return gjsd(distributions, weights) / upper


def quantile_edges(values: np.ndarray, n_groups: int) -> np.ndarray:
    """Lower bin edges for equal-population groups: quantiles 0, 1/n, ..., (n-1)/n."""
    return np.quantile(np.asarray(values, dtype=float), np.arange(n_groups) / n_groups)


def group_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Count values per group given lower bin edges (last bin open-ended).
    """
    values = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 1)
    return np.bincount(idx, minlength=len(edges)).astype(float)


def population_vector(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Share of values per group."""
    counts = group_counts(values, edges)
    total = counts.sum()
    return counts / total if total > 0 else counts
