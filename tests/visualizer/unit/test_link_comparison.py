"""Native controls use the same shared definitions and export route."""

import json

from shared.statistics.link_comparison import COMPARISON_SCHEMA, export_recording
from tests.statistics_shared.test_link_comparison import recording
from visualizer.src.panels.link_comparison import LinkComparisonDialog


def test_load_select_analyze_export_controls(qtbot, tmp_path):
    export_recording(recording(), tmp_path / "input")
    definition = dict(
        schema=COMPARISON_SCHEMA,
        sources=[dict(label="Alternate fixture", path="input.npz")],
        receiver="Cyclist",
        transmitters=["Roof"],
        axis="time",
        paired=False,
    )
    path = tmp_path / "definition.json"
    path.write_text(json.dumps(definition))
    widget = LinkComparisonDialog()
    qtbot.addWidget(widget)
    widget.set_definition(path)
    assert widget.receiver.currentText() == "Cyclist"
    widget.analyze_button.click()
    assert widget.result["transmitters"] == ("Roof",)
    summary = widget.export_to(tmp_path / "export")
    assert summary["definition"]["receiver"] == "Cyclist"
    assert (tmp_path / "export/comparison.png").is_file()
    assert (tmp_path / "export/values.npz").is_file()
    widget.frame.setValue(999)
    widget._inspect()
    assert "not present" in widget.status.text()
    widget._remove()
    assert not widget.records
