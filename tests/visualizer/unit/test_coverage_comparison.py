"""Native controls preserve support decisions and export the shared numbers."""

import json

import numpy as np

from shared.coverage.comparison import save_sample_bundle
from tests.shared.test_coverage_volume import write_volume
from visualizer.src.coverage.comparison_window import CoverageComparisonWindow


def test_native_start_controls_save_and_export(qtbot, tmp_path):
    source = tmp_path / "samples.npz"
    save_sample_bundle(
        source,
        rss_dbm=np.array([[[-60], [-90]], [[-65], [-70]]]),
        site_names=["North"],
        group_names=["2 m", "9 m"],
        eligible=np.array([[True, False], [True, True]]),
        weights=np.ones((2, 2)),
        availability=np.ones((2, 2, 1), dtype=np.uint8),
        common_mask=np.array([True, False]),
        metadata={
            "quantity": "Power (dBm)",
            "receiver_model": "Test V",
            "support": "Aligned cells",
            "weighting": "Equal",
            "provenance": {"fixture": 1},
            "common_columns_aligned": True,
        },
    )
    window = CoverageComparisonWindow()
    qtbot.addWidget(window)
    window.add_source(source, "sample_bundle")
    assert [r["eligible_samples"] for r in window.result["rows"]] == [1, 2]
    window.support.setCurrentIndex(window.support.findData("common_columns"))
    assert [r["eligible_samples"] for r in window.result["rows"]] == [1, 1]
    window.threshold.setValue(-62.25)
    assert -62.25 in window.result["definition"]["thresholds_dbm"]
    window.curve.setCurrentIndex(1)
    saved = tmp_path / "saved.json"
    window.save_definition(saved)
    reopened = CoverageComparisonWindow(definition_path=saved)
    qtbot.addWidget(reopened)
    assert reopened.threshold.value() == -62.25 and reopened.curve.currentIndex() == 1
    files = reopened.export(tmp_path / "result")
    assert len(files) == 4
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["rows"][0]["eligible_samples"] == 1
    assert result["display"]["threshold_dbm"] == -62.25
    weighted = json.loads(saved.read_text())
    weighted["sources"][0].update(pool_groups=True, group_weights=[1, 3])
    weighted_path = tmp_path / "weighted.json"
    weighted_path.write_text(json.dumps(weighted))
    reopened.load_definition(weighted_path)
    reopened.support.setCurrentIndex(reopened.support.findData("per_group"))
    assert reopened.definition["sources"][0]["group_weights"] == [1, 3]
    assert "declared weights" in reopened.pool.text()
    np.testing.assert_allclose(reopened.result["rows"][0]["group_weights"], [0.25, 0.75])
    source.write_bytes(b"changed")
    reopened.refresh()
    assert reopened.result is None and "changed" in reopened.status.text()


def test_region_bounds_and_threshold_survive_comparison_export(qtbot, tmp_path):
    path = write_volume(tmp_path / "coverage.h5")
    bounds = [[11, 15], [-18.5, -15.5], [7, 11]]
    window = CoverageComparisonWindow(native_path=path, roi_bounds=bounds, threshold_dbm=-80)
    qtbot.addWidget(window)
    common = window.support.model().item(window.support.findData("common_columns"))
    assert not common.isEnabled()
    assert "no saved common-column eligibility mask" in common.toolTip()
    assert window.threshold.value() == -80
    assert all(row["eligible_samples"] == 6 for row in window.result["rows"])
    saved = tmp_path / "region.json"
    window.save_definition(saved)
    reopened = CoverageComparisonWindow(definition_path=saved)
    qtbot.addWidget(reopened)
    assert reopened.definition["sources"][0]["roi_bounds_m"] == bounds
    assert reopened.threshold.value() == -80
    reopened.export(tmp_path / "region-export")
    exported = json.loads((tmp_path / "region-export.json").read_text())
    assert all(row["eligible_samples"] == 6 for row in exported["rows"])
    assert exported["definition"]["sources"][0]["roi_bounds_m"] == bounds
