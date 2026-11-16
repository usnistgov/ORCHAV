"""Stable batched CFAR sums retain direct-reference decisions and zero semantics."""

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

import shared.sensing.detection.cfar as cfar


def direct_training_sums(arr, locations, *, guard_cells, reference_cells):
    """Independent clipped-neighborhood enumeration, with no prefix sums."""
    gd, gr = guard_cells
    rd, rr = reference_cells
    sums = []
    for i, j in locations:
        values = [
            arr[d, r]
            for d in range(max(0, i - gd - rd), min(arr.shape[0], i + gd + rd + 1))
            for r in range(max(0, j - gr - rr), min(arr.shape[1], j + gr + rr + 1))
            if abs(d - i) > gd or abs(r - j) > gr
        ]
        sums.append(math.fsum(values))
    return np.array(sums)


@pytest.mark.parametrize("count", [1, 2, 3, 31, 336, 1025])
@pytest.mark.parametrize("exponents", [(-10, 10), (-1000, 900), (-1074, -1020)])
def test_compensated_sums_match_fsum(count, exponents):
    rng = np.random.default_rng(584)
    values = np.ldexp(rng.random((73, count)), rng.integers(*exponents, size=(73, count)))
    expected = np.array([math.fsum(row) for row in values])
    np.testing.assert_array_equal(cfar._accurate_positive_row_sums(values), expected)
    with localcontext() as context:
        context.prec = 1200
        for row, result in zip(values[:3], expected[:3]):
            assert result == float(sum((Decimal.from_float(x) for x in row), Decimal(0)))


def test_rounding_midpoints_and_subnormal_power_use_exact_fallback(monkeypatch):
    tiny = np.nextafter(0.0, 1.0)
    values = np.array(
        [
            [1.0, 2.0**-53, 0],
            [1.0, 2.0**-53, tiny],
            [np.nextafter(1.0, 2.0), 2.0**-53, 0],
            [tiny, tiny, tiny],
            [0, 0, 0],
        ]
    )
    original = math.fsum
    calls = []

    def counted(row):
        calls.append(1)
        return original(row)

    monkeypatch.setattr(cfar.math, "fsum", counted)
    np.testing.assert_array_equal(
        cfar._accurate_positive_row_sums(values), [original(row) for row in values]
    )
    assert len(calls) == len(values)


def power_map(kind):
    rng = np.random.default_rng(487)
    a = rng.exponential(size=(27, 39))
    if kind == "low":
        a *= 1e-240
    elif kind == "dense_dynamic":
        a = 10.0 ** rng.uniform(-280, 100, a.shape)
    elif kind == "weak_neighbor":
        a *= 1e-70
        a[12, 14] = 1
        a[12, 22] = 1e-60
    elif kind == "sparse":
        a.fill(0)
        a[8, 10] = 1
        a[13, 23] = 1e-200
    elif kind == "zeros":
        a.fill(0)
    elif kind == "zero_region":
        a[:14] = 0
        a[15, 15] = 1e80
    elif kind == "subnormal":
        a *= np.nextafter(0.0, 1.0)
    return a


@pytest.mark.parametrize(
    "kind",
    [
        "ordinary",
        "low",
        "dense_dynamic",
        "weak_neighbor",
        "sparse",
        "zeros",
        "zero_region",
        "subnormal",
    ],
)
@pytest.mark.parametrize("edge_policy", ["full_window", "truncate"])
def test_complete_cfar_matches_original_direct_fallback(kind, edge_policy, monkeypatch):
    a = power_map(kind)
    before = a.copy()
    kwargs = dict(
        guard_cells=(1, 2), reference_cells=(3, 5), threshold_scale=3.0, edge_policy=edge_policy
    )
    actual = cfar.cfar_2d(a, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(cfar, "_local_training_sums", direct_training_sums)
        expected = cfar.cfar_2d(a, **kwargs)
    for key in ("noise", "threshold", "detections", "eligible", "training_cell_count"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    np.testing.assert_array_equal(a, before)
    locations = np.argwhere(actual.eligible)
    direct = direct_training_sums(a, locations, guard_cells=(1, 2), reference_cells=(3, 5))
    # The unchanged integral branch has its existing rounding; the local
    # fallback itself is bit-identical above, including decision thresholds.
    np.testing.assert_allclose(
        actual.noise[actual.eligible],
        direct / actual.training_cell_count[actual.eligible],
        rtol=2e-10,
        atol=0,
    )


@pytest.mark.parametrize(
    "shape,guards,refs",
    [((8, 11), (0, 0), (20, 20)), ((40, 55), (2, 4), (5, 12)), ((6, 70), (0, 7), (1, 3))],
)
def test_local_sums_clip_large_windows_and_handle_all_edges(shape, guards, refs):
    rng = np.random.default_rng(81)
    a = 10.0 ** rng.uniform(-250, 10, shape)
    locations = np.argwhere(np.ones(shape, bool))
    for chosen in (locations[:3], locations):
        expected = direct_training_sums(a, chosen, guard_cells=guards, reference_cells=refs)
        actual = cfar._local_training_sums(a, chosen, guard_cells=guards, reference_cells=refs)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(1, 47), (47, 1), (37, 83)])
def test_local_gather_preserves_unordered_locations_and_strided_input(shape):
    """Flattened indexing must respect padded row strides and batch boundaries."""
    rng = np.random.default_rng(891)
    backing = np.ldexp(rng.random((shape[0] * 2, shape[1] * 3)), -200)
    arr = backing[::2, ::3]
    arr.flat[0] = 1.0
    arr.flat[-1] = np.nextafter(0.0, 1.0)
    before = backing.copy()
    locations = np.argwhere(np.ones(shape, dtype=bool))
    rng.shuffle(locations)
    # Include repeated locations, clipped edges and more than two full batches
    # for the rectangular map. Their output positions must remain unchanged.
    locations = np.concatenate((locations, locations[:19], locations[-19:]))
    expected = direct_training_sums(arr, locations, guard_cells=(2, 4), reference_cells=(4, 16))
    actual = cfar._local_training_sums(arr, locations, guard_cells=(2, 4), reference_cells=(4, 16))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(backing, before)


@pytest.mark.parametrize("shape", [(1, 1), (1, 19), (23, 1), (8, 19)])
@pytest.mark.parametrize("edge_policy", ["full_window", "truncate"])
def test_broadcast_geometry_matches_direct_annuli(shape, edge_policy):
    """Counts, eligibility and strict decisions retain both edge policies."""
    arr = np.random.default_rng(91).integers(1, 20, size=shape).astype(np.float64)
    guards, refs, scale = (1, 2), (2, 3), 1.75
    result = cfar.cfar_2d(
        arr,
        guard_cells=guards,
        reference_cells=refs,
        threshold_scale=scale,
        edge_policy=edge_policy,
    )
    for i, j in np.ndindex(shape):
        training = [
            arr[d, r]
            for d in range(max(0, i - 3), min(shape[0], i + 4))
            for r in range(max(0, j - 5), min(shape[1], j + 6))
            if abs(d - i) > guards[0] or abs(r - j) > guards[1]
        ]
        eligible = bool(training) and (
            edge_policy == "truncate" or (3 <= i < shape[0] - 3 and 5 <= j < shape[1] - 5)
        )
        count = len(training) if eligible else 0
        noise = math.fsum(training) / count if count else 0.0
        assert result.eligible[i, j] == eligible
        assert result.training_cell_count[i, j] == count
        assert result.noise[i, j] == noise
        assert result.threshold[i, j] == noise * scale
        assert result.detections[i, j] == (eligible and arr[i, j] > noise * scale)


@pytest.mark.parametrize("direction", [-1, 0, 1])
@pytest.mark.parametrize("edge_policy", ["full_window", "truncate"])
def test_adjacent_float_threshold_decisions_are_unchanged(direction, edge_policy):
    a = np.full((25, 35), 1e-180)
    a[0, 0] = 1e50  # Forces local sums around the distant weak CUT.
    target = (17, 24)
    settings = dict(
        guard_cells=(1, 2), reference_cells=(3, 4), threshold_scale=2.0, edge_policy=edge_policy
    )
    initial = cfar.cfar_2d(a, **settings)
    threshold = initial.threshold[target]
    a[target] = np.nextafter(threshold, 0 if direction < 0 else np.inf) if direction else threshold
    result = cfar.cfar_2d(a, **settings)
    assert result.threshold[target] == threshold
    assert result.detections[target] == (direction > 0)
