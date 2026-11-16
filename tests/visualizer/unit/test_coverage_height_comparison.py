"""Inline comparisons preserve unknown cells and reuse one lazy source read."""

from unittest.mock import Mock

import h5py
import numpy as np

from shared.coverage.volume import coverage_file_identity, load_coverage_volume
from tests.shared.test_coverage_volume import write_volume
from visualizer.src.coverage import height_comparison


def make_comparison(qtbot, path):
    with h5py.File(path, "r+") as h:
        if "availability/path_gain" not in h:
            gains = h["values/path_gain_linear"][...]
            codes = np.where(np.isnan(gains), 0, np.where(gains == 0, 2, 1)).astype(np.uint8)
            h.create_dataset("availability/path_gain", data=codes).attrs["schema_version"] = 1
    widget = height_comparison.CoverageHeightComparison(path, coverage_file_identity(path))
    qtbot.addWidget(widget)
    axes = load_coverage_volume(path, "A", "rss_dbm").axes
    widget.set_context(axes, None)
    return widget, axes


def test_lazy_height_comparison_preserves_denominator_and_reuses_samples(
    qtbot, tmp_path, monkeypatch
):
    path = write_volume(tmp_path / "coverage.h5")
    widget, axes = make_comparison(qtbot, path)
    read = Mock(wraps=height_comparison.native_coverage_bundle)
    monkeypatch.setattr(height_comparison, "native_coverage_bundle", read)
    assert widget.pending is None and not widget.poll.isActive()
    assert read.call_count == 0
    try:
        widget.enabled.setChecked(True)
        qtbot.waitUntil(lambda: widget.data is not None, timeout=10000)
        # B passes everywhere, so A's unknown sample does not change the union.
        np.testing.assert_allclose([r["fraction_lower"][0, 0] for r in widget.summaries], 1)
        widget.transmitter.setCurrentIndex(widget.transmitter.findText("A"))
        np.testing.assert_allclose(
            [r["fraction_lower"][0, 0] for r in widget.summaries], [0.05, 0, 0]
        )
        np.testing.assert_allclose(
            [r["fraction_upper"][0, 0] for r in widget.summaries], [0.10, 0, 0]
        )
        widget.threshold.setValue(-95)
        np.testing.assert_allclose(
            [r["fraction_lower"][0, 0] for r in widget.summaries], [0.90, 1, 1]
        )
        widget.threshold.setValue(-75)
        bounds = [[11, 15], [-18.5, -15.5], [7, 11]]
        widget.set_context(axes, bounds)
        np.testing.assert_allclose(
            [r["fraction_lower"][0, 0] for r in widget.summaries], [1 / 6, 0]
        )
        widget.enabled.setChecked(False)
        widget.enabled.setChecked(True)
        assert read.call_count == 1 and not widget.poll.isActive()
        monkeypatch.setattr(widget, "_draw", Mock(side_effect=AssertionError("Unchanged region")))
        widget.set_context(axes, bounds)
    finally:
        widget.shutdown()


def test_empty_region_and_changed_file_have_in_plot_messages(qtbot, tmp_path):
    path = write_volume(tmp_path / "coverage.h5")
    widget, axes = make_comparison(qtbot, path)
    try:
        widget.enabled.setChecked(True)
        qtbot.waitUntil(lambda: widget.data is not None, timeout=10000)
        widget.set_context(axes, [[11.1, 12.9], [-18.5, -15.5], [7, 11]])
        assert "no recorded cell centres" in widget.plot._empty_message.text
        assert not widget.summaries
    finally:
        widget.shutdown()
    changed, _ = make_comparison(qtbot, path)
    path.write_bytes(b"changed")
    try:
        changed.enabled.setChecked(True)
        qtbot.waitUntil(lambda: not changed.poll.isActive(), timeout=10000)
        assert "file changed" in changed.plot._empty_message.text
    finally:
        changed.shutdown()


def test_single_transmitter_has_no_redundant_selector(qtbot, tmp_path):
    path = write_volume(tmp_path / "coverage.h5")
    with h5py.File(path, "r+") as h:
        gains = h["values/path_gain_linear"][:, :, :1]
        for name in ("values/path_gain_linear", "tx/names", "tx/powers_dbm"):
            del h[name]
        h.create_dataset("values/path_gain_linear", data=gains)
        h.create_dataset("tx/names", data=[b"A"])
        h.create_dataset("tx/powers_dbm", data=[0])
    widget, _ = make_comparison(qtbot, path)
    try:
        widget.enabled.setChecked(True)
        qtbot.waitUntil(lambda: widget.data is not None, timeout=10000)
        assert widget.transmitter.count() == 1
        assert widget.transmitter.isHidden()
    finally:
        widget.shutdown()
