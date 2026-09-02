"""Tests for shared.statistics frame-level helpers."""

import numpy as np

from shared.statistics import compute_orders_hist


def test_compute_orders_hist_skips_invalid_dict_entries():
    hist = compute_orders_hist({0: 3, "bad-order": 4, 2: np.int64(5), 3: "invalid"})

    assert hist == {0: 3, 2: 5}


def test_compute_orders_hist_skips_invalid_iterable_entries():
    hist = compute_orders_hist([2, "invalid", np.int64(4), 0])

    assert hist == {0: 2, 2: 4}
