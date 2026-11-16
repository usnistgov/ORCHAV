"""Scientific support and persistence contracts, independent of widgets/RF."""

import json

import numpy as np
import pytest

from shared.coverage.comparison import (
    analyze_definition,
    file_identity,
    save_sample_bundle,
    site_subsets,
    weighted_subset_summary,
)


def test_empty_unknown_and_invalid_are_not_dropped():
    value = np.array([[-60, -np.inf], [-np.inf, np.nan], [-90, -90], [np.nan, -65.0]])
    code = np.array([[1, 2], [2, 0], [1, 1], [3, 1]])
    result = weighted_subset_summary(
        value, ["West", "East"], thresholds_dbm=[-75], availability=code
    )
    np.testing.assert_equal(result["count_lower"][:, 0], [1, 1, 2])
    np.testing.assert_equal(result["count_upper"][:, 0], [2, 2, 3])
    np.testing.assert_allclose(result["fraction_lower"][:, 0], [0.25, 0.25, 0.5])
    assert result["eligible_samples"] == 4
    assert result["availability_counts_by_site"]["3"].tolist() == [1, 0]


def test_mask_weights_and_empty_support():
    result = weighted_subset_summary(
        [[-60], [-90], [-20]],
        ["A"],
        thresholds_dbm=[-75],
        eligible=np.array([True, True, False]),
        weights=[1, 3, 100],
    )
    assert result["fraction_lower"][0, 0] == 0.25
    assert result["count_lower"][0, 0] == 1
    empty = weighted_subset_summary(np.empty((0, 1)), ["A"], thresholds_dbm=[-75])
    assert empty["eligible_samples"] == 0 and np.isnan(empty["fraction_lower"]).all()
    with pytest.raises(ValueError, match="zero weight"):
        weighted_subset_summary([[-60]], ["A"], thresholds_dbm=[-75], weights=[0])


def test_subset_count_bound_and_arbitrary_names():
    with pytest.raises(ValueError, match="explicit"):
        site_subsets([f"X{i}" for i in range(9)])
    assert site_subsets([f"X{i}" for i in range(9)], [["X8", "X2"]]) == [("X8", "X2")]
    with pytest.raises(ValueError, match="known"):
        site_subsets(["A"], [["B"]])


@pytest.mark.parametrize(
    "values,codes", [([[0]], [[2]]), ([[-np.inf]], [[1]]), ([[np.inf]], [[1]])]
)
def test_inconsistent_measurements_fail(values, codes):
    with pytest.raises(ValueError):
        weighted_subset_summary(values, ["A"], thresholds_dbm=[-75], availability=codes)


def make_bundle(tmp_path):
    path = tmp_path / "input.npz"
    save_sample_bundle(
        path,
        rss_dbm=np.array([[[-60], [-90]], [[-65], [-70]]]),
        site_names=["Arbitrary"],
        group_names=["low", "high"],
        eligible=np.array([[True, False], [True, True]]),
        weights=np.ones((2, 2)),
        availability=np.ones((2, 2, 1), dtype=np.uint8),
        common_mask=np.array([True, False]),
        metadata={
            "quantity": "Power in dBm",
            "receiver_model": "Test V",
            "support": "shared columns",
            "weighting": "equal",
            "provenance": {"fixture": 1},
            "common_columns_aligned": True,
        },
    )
    return path


def test_saved_support_and_pooling(tmp_path):
    path = make_bundle(tmp_path)
    definition = {
        "schema_version": 1,
        "kind": "coverage_comparison",
        "thresholds_dbm": [-75, -62],
        "sources": [
            {
                "path": path.name,
                "sha256": file_identity(path)["sha256"],
                "label": "Study",
                "support": "common_columns",
            }
        ],
    }
    value = analyze_definition(json.loads(json.dumps(definition)), base_dir=tmp_path)
    assert [r["eligible_samples"] for r in value["rows"]] == [1, 1]
    definition["sources"][0].update(pool_groups=True, group_weights=[1, 3])
    pooled = analyze_definition(definition, base_dir=tmp_path)["rows"][0]
    np.testing.assert_allclose(pooled["fraction_lower"], [[1, 0.25]])
    definition["sources"][0]["sha256"] = "incorrect"
    with pytest.raises(ValueError, match="changed"):
        analyze_definition(definition, base_dir=tmp_path)


def test_metadata_and_common_support_are_required(tmp_path):
    with pytest.raises(ValueError, match="receiver_model"):
        save_sample_bundle(
            tmp_path / "bad.npz",
            rss_dbm=np.zeros((1, 1, 1)),
            site_names=["A"],
            group_names=["g"],
            eligible=np.ones((1, 1), bool),
            weights=np.ones((1, 1)),
            availability=np.ones((1, 1, 1)),
            metadata={"quantity": "Power"},
        )
