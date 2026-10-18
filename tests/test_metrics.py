import numpy as np
import pytest

from multiscalar_segregation import metrics


rng = np.random.default_rng(42)


def mutual_information(joint: np.ndarray) -> float:
    """Reference MI (nats) from a joint distribution over (child, group)."""
    joint = joint / joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ pb)[nz])).sum())


def test_gjsd_equals_mutual_information():
    """Eq. GJSD_MI in the manuscript: GJSD of child compositions == I(X; B)."""
    joint = rng.random((5, 8))
    joint /= joint.sum()
    weights = joint.sum(axis=1)
    dists = joint / joint.sum(axis=1, keepdims=True)
    assert metrics.gjsd(dists, weights) == pytest.approx(mutual_information(joint), rel=1e-12)


def test_gjsd_zero_for_identical_children():
    p = rng.random(10)
    p /= p.sum()
    dists = np.tile(p, (4, 1))
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    assert metrics.gjsd(dists, weights) == pytest.approx(0.0, abs=1e-12)


def test_gjsd_attains_upper_bound_on_disjoint_support():
    dists = np.eye(4)
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    assert metrics.gjsd(dists, weights) == pytest.approx(metrics.gjsd_max(weights), rel=1e-12)
    assert metrics.norm_gjsd(dists, weights) == pytest.approx(1.0, rel=1e-12)


def test_norm_gjsd_single_child_is_zero():
    assert metrics.norm_gjsd(np.array([[0.5, 0.5]]), np.array([1.0])) == 0.0


def test_population_vector_matches_reference_10bin_counts():
    """Reference implementation of the per-unit decile counts."""
    values = rng.normal(size=5000)
    edges = metrics.quantile_edges(values, 10)

    reference = np.zeros(10)
    for i in range(len(edges)):
        if i >= len(edges) - 1:
            mask = values >= edges[i]
        else:
            mask = (values >= edges[i]) & (values < edges[i + 1])
        reference[i] = mask.sum() / len(values)

    np.testing.assert_allclose(metrics.population_vector(values, edges), reference)


def test_population_vector_generalises_group_count():
    values = rng.normal(size=1000)
    for n in (4, 5, 10):
        vec = metrics.population_vector(values, metrics.quantile_edges(values, n))
        assert vec.shape == (n,)
        assert vec.sum() == pytest.approx(1.0)
        assert vec.max() - vec.min() < 0.02


def test_miller_madow_reduces_plugin_bias():
    p = np.full(10, 0.1)
    true_h = metrics.entropy(p)
    plugin, mm = [], []
    for seed in range(200):
        counts = np.bincount(np.random.default_rng(seed).integers(0, 10, 50), minlength=10)
        plugin.append(metrics.entropy(counts / counts.sum()))
        mm.append(metrics.miller_madow_entropy(counts))
    assert abs(np.mean(mm) - true_h) < abs(np.mean(plugin) - true_h)
