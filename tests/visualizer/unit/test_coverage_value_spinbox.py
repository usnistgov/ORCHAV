"""Coverage entry uses readable units while emitting the saved metric's units."""

import pytest
from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QValidator

from visualizer.src.coverage.value_spinbox import CoverageValueSpinBox


@pytest.mark.parametrize(
    "metric, unit, raw",
    [
        ("path_gain_linear/TX1", "dB", 1e-12),
        ("serving_path_gain_linear", "dB", 1e-12),
        ("sinr_linear", "dB", 1e-12),
        ("rss_w/TX1", "dBm", 1e-15),
    ],
)
def test_typed_decibels_and_arrow_steps_emit_native_values(qtbot, metric, unit, raw):
    spin = CoverageValueSpinBox()
    qtbot.addWidget(spin)
    spin.set_metric_display(metric)
    spin.setRange(1e-30, 1e12)
    spin.setValue(1e-9)
    changes = []
    spin.valueChanged.connect(changes.append)
    spin.show()
    spin.lineEdit().selectAll()
    qtbot.keyClicks(spin.lineEdit(), "-120")
    assert not changes  # Text is committed once, avoiding rebuilds on each key.
    qtbot.keyClick(spin.lineEdit(), Qt.Key_Return)
    assert spin.value() == pytest.approx(raw, rel=1e-9, abs=1e-30)
    assert spin.suffix() == f" {unit}"
    assert changes[-1] == pytest.approx(raw, rel=1e-9, abs=1e-30)
    spin.stepUp()
    assert spin.value() == pytest.approx(raw * 10**0.05, rel=1e-9, abs=1e-30)
    assert float(spin.cleanText()) == -119.5
    spin.stepDown()
    assert spin.value() == pytest.approx(raw, rel=1e-9, abs=1e-30)


def test_locale_bounds_and_invalid_decibel_values(qtbot):
    spin = CoverageValueSpinBox()
    qtbot.addWidget(spin)
    spin.set_metric_display("rss_w/TX")
    spin.setRange(1e-30, 1)
    spin.setLocale(QLocale(QLocale.German, QLocale.Germany))
    assert spin.valueFromText("-90,5 dBm") == pytest.approx(
        10 ** (-120.5 / 10), rel=1e-9, abs=1e-30
    )
    assert spin.valueFromText("100000 dBm") == 1
    assert spin.valueFromText("-100000 dBm") == pytest.approx(1e-30, rel=1e-9, abs=1e-40)
    spin.setValue(spin.minimum())
    spin.stepUp()
    assert spin.value() == pytest.approx(1e-30 * 10**0.05, rel=1e-9, abs=1e-40)
    for text in ("nan", "inf", "-inf"):
        assert spin.validate(text, len(text))[0] == QValidator.Invalid
