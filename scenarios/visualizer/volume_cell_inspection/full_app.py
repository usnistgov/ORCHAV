"""Open the controlled sensing fixture in the real ORCHAV application shell.

The optional capture mode exercises the production Open trial recording button,
including manifest verification, the native frame loader, dashboard and breadcrumb.
"""

import argparse
import json
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PySide6.QtWidgets import QApplication, QPushButton

from visualizer.src.sensing.volume_window import open_sensing_volume
from visualizer.visualizer import OrchavVisualizer

if __package__:
    from .demo import prepare, wait
    from .retained_trial import prepare_retained_trial
else:
    from demo import prepare, wait
    from retained_trial import prepare_retained_trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    prepare(args.output)
    evidence = prepare_retained_trial(args.output)
    app = QApplication.instance() or QApplication([])
    visualizer = OrchavVisualizer(renderer_type="pygfx", viewport_mode="embedded")
    visualizer.resize(1500, 1000)
    visualizer.show()
    visualizer._deferred_init()
    app.processEvents()
    panel = open_sensing_volume(visualizer, args.output / "sensing_volume.json")
    wait(app, lambda: panel.reader is not None and panel.evaluation_controls.ready, seconds=50)
    if not args.capture:
        app.exec()
        return
    panel.workflow.select_slice("xz", 14, 7)
    panel.workflow.select_slice("xy", 14, -18)
    assert panel._cell_key() == (0, 0, 0, 2)
    assert panel.open_native.isEnabled(), panel.open_native.toolTip()
    evidence.update(
        selected_cell=panel._cell_key(), selected_text=panel.cell_note.text(), button_enabled=True
    )
    opened = []
    real_open = visualizer.open_scenario

    def observe_open(*args, **kwargs):
        outcome = real_open(*args, **kwargs)
        opened.append(outcome)
        return outcome

    # Observe, without substituting a result for the real application loader.
    visualizer.open_scenario = observe_open
    panel.open_native.click()
    wait(app, lambda: getattr(visualizer, "volume_return", None) is not None, seconds=60)
    current = Path(visualizer.current_scenario_path).resolve()
    current = current.parent if current.is_file() else current
    assert current == Path(evidence["native"]).resolve()
    assert opened and opened[-1].succeeded and opened[-1].frame_source_ready
    assert visualizer.volume_return.context["trial_id"] == evidence["trial_id"]
    evidence.update(
        opened_scenario=str(current),
        frame_source_ready=opened[-1].frame_source_ready,
        return_trial_id=visualizer.volume_return.context["trial_id"],
        production_open_trial_passed=True,
    )
    dashboard = getattr(visualizer.ui_manager.panels["sensing"], "_sensing_window", None)
    if dashboard is None:
        raise AssertionError("Production trial open did not create the sensing dashboard")
    dashboard._set_view_mode("analysis")
    dashboard.view_changed.emit()
    dashboard.table_button.setChecked(True)
    wait(
        app,
        lambda: dashboard._last_request is not None
        and dashboard._last_request.rd_result is not None
        and dashboard._rendered_request is dashboard._last_request
        and dashboard.detection_table_model.rowCount() > 0,
        seconds=60,
    )
    displayed = dashboard._last_request.display_frame["sensing"]
    reports = displayed.get("detections", [])
    assert any(report["detection_id"] == evidence["report_id"] for report in reports), reports
    truth = displayed.get("truth", {}).get("targets", [])
    assert any(row["target_id"] == evidence["target_id"] for row in truth), truth
    dashboard.select_detection_index(0)
    evidence.update(
        dashboard_opened=True,
        displayed_report_ids=[row["detection_id"] for row in reports],
        displayed_truth_ids=[row["target_id"] for row in truth],
        dashboard_selection=dashboard.selection_label.text(),
    )
    for _ in range(30):
        visualizer.renderer._canvas.force_draw()
        app.processEvents()
        time.sleep(0.02)
    # Wait for native overlays before using the dashboard's camera action.
    # The empty-environment fixture otherwise retains the origin camera even
    # though its target is translated. Data loading alone cannot prove visibility.
    assert dashboard.locate_selection_button.isEnabled()
    dashboard.locate_selection_button.click()
    wait(
        app,
        lambda: np.allclose(visualizer.renderer.get_camera_state().lookat, evidence["position_m"]),
        seconds=10,
    )
    assert dashboard._last_request.rd_result is not None
    detected_x, detected_y = dashboard.rd_det_scatter.getData()
    truth_x, truth_y = dashboard.rd_gt_scatter.getData()
    assert len(detected_x) == len(truth_x) == 1
    np.testing.assert_allclose(detected_x, truth_x)
    np.testing.assert_allclose(detected_y, truth_y)
    x_limits, y_limits = dashboard.rd_plot.getViewBox().viewRange()
    assert x_limits[0] <= detected_x[0] <= x_limits[1]
    assert y_limits[0] <= detected_y[0] <= y_limits[1]
    rp_limits = dashboard.rp_plot.getViewBox().viewRange()[0]
    assert rp_limits[0] <= detected_x[0] <= rp_limits[1]
    evidence.update(
        plotted_report_and_truth=True,
        plotted_range_m=float(detected_x[0]),
        plotted_velocity_m_s=float(detected_y[0]),
        rd_view_range=[x_limits, y_limits],
    )
    visualizer.grab().save(str(args.output / "full_app_trial_open.png"))
    visualizer.volume_return.grab().save(str(args.output / "trial_return_context.png"))
    pixels = visualizer.renderer.export_screenshot_to_array()
    assert np.count_nonzero(np.any(pixels != pixels[0, 0], axis=-1)) > 100
    evidence["native_view_nonempty"] = True
    evidence["center_selection_in_3d_passed"] = True
    evidence["native_camera"] = visualizer.renderer.get_camera_state().to_dict()
    iio.imwrite(args.output / "retained_trial_native.png", pixels)
    dashboard.grab().save(str(args.output / "retained_trial_dashboard.png"))
    visualizer.volume_return.findChild(QPushButton).click()
    wait(
        app,
        lambda: visualizer.sensing_volume_window.reader is not None
        and visualizer.sensing_volume_window.evaluation_controls.ready,
        seconds=50,
    )
    restored = visualizer.sensing_volume_window
    assert restored._cell_key() == (0, 0, 0, 2)
    assert restored.open_native.isEnabled()
    evidence["return_to_exact_source_cell_passed"] = True
    evidence["returned_cell"] = restored._cell_key()
    restored.grab().save(str(args.output / "returned_source_cell.png"))
    (args.output / "production_trial_result.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))
    visualizer.close()
    app.processEvents()


if __name__ == "__main__":
    main()
