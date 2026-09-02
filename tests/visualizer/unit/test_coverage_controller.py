from types import SimpleNamespace

import numpy as np

from visualizer.src.controllers.coverage_controller import CoverageController
from visualizer.src.renderers.protocol import RendererCapabilities


class StubCoverageService:
    def __init__(self):
        self.selected = None
        self.keys = []
        self.logged = False
        self.interpolation_calls = []
        self.mesh_copy_args = []
        self.interpolation_result = None

    def select_metric_layer(self, coverage_data, metric_name):
        self.selected = metric_name
        coverage_data["metric_name"] = metric_name
        coverage_data["value_min"] = 10.0
        coverage_data["value_max"] = 20.0

    def compute_cache_key(self, coverage_data, height_index, interpolation):
        key = (height_index, interpolation)
        self.keys.append(key)
        return key

    def get_mesh(self, cache_key, *, copy=True):
        self.mesh_copy_args.append(copy)
        return None

    def interpolate_values(self, values, interpolation):
        self.interpolation_calls.append((np.asarray(values).copy(), interpolation))
        if self.interpolation_result is not None:
            return np.asarray(self.interpolation_result)
        return values

    def log_stats(self):
        self.logged = True


class StubPipeline:
    def __init__(self):
        self.cleared = False
        self.precache_calls = 0
        self.overlay_calls = 0

    def clear_coverage_cache(self):
        self.cleared = True

    def precache_coverage_heights(self):
        self.precache_calls += 1
        return 2, 0

    def update_coverage_overlay(self):
        self.overlay_calls += 1
        return True


class StubCoveragePanel:
    def __init__(self):
        self.widgets = {}
        self.status_calls = []
        self.height_indices = []
        self.threshold_summaries = []
        self.threshold_values = []

    def update_coverage_status(self, has_data, metadata, supports_transparency):
        self.status_calls.append(
            {
                "has_data": has_data,
                "metric_name": metadata["metric_name"],
                "supports_transparency": supports_transparency,
                "value_min": metadata["value_min"],
                "value_max": metadata["value_max"],
            }
        )

    def set_height_index(self, index):
        self.height_indices.append(index)

    def set_threshold_summary(self, text, active):
        self.threshold_summaries.append((text, active))

    def set_threshold_value(self, value):
        self.threshold_values.append(float(value))


class StubToggle:
    def __init__(self, checked):
        self._checked = checked
        self.history = [checked]

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        self._checked = bool(checked)
        self.history.append(self._checked)


def test_metric_change_refreshes_coverage_panel_status():
    coverage_service = StubCoverageService()
    panel = StubCoveragePanel()
    graph_notifications = []
    viz = SimpleNamespace(
        coverage_data={
            "available_metrics": ["best_path_loss_db", "sinr_db"],
            "metric_name": "best_path_loss_db",
            "value_min": 1.0,
            "value_max": 2.0,
        },
        coverage_metric_name=None,
        coverage_height_index=1,
        pipeline=StubPipeline(),
        _coverage_interpolation_dirty=False,
        app_state=SimpleNamespace(show_coverage=False),
        renderer=SimpleNamespace(capabilities=RendererCapabilities(transparency=True)),
        ui_manager=SimpleNamespace(
            panels={"coverage": panel},
            notify_coverage_selection_changed=lambda **kwargs: graph_notifications.append(kwargs),
        ),
    )
    parent = SimpleNamespace(visualizer=viz, coverage_service=coverage_service)

    CoverageController(parent).handle_coverage_metric_changed("sinr_db")

    assert coverage_service.selected == "sinr_db"
    assert viz.coverage_metric_name == "sinr_db"
    assert viz.pipeline.cleared is True
    assert viz._coverage_interpolation_dirty is True
    assert panel.status_calls == [
        {
            "has_data": True,
            "metric_name": "sinr_db",
            "supports_transparency": True,
            "value_min": 10.0,
            "value_max": 20.0,
        }
    ]
    assert panel.height_indices == [1]
    assert graph_notifications == [{"render": True}]


def test_file_backed_height_change_loads_slice_before_publishing_state():
    coverage_service = StubCoverageService()
    loaded_heights = []
    graph_notifications = []

    def select_height_layer(coverage_data, height_index):
        loaded_heights.append(height_index)
        coverage_data["values_3d"] = np.asarray(
            [[[10.0 + height_index]]],
            dtype=np.float32,
        )
        coverage_data["_active_height_index"] = height_index

    coverage_service.select_height_layer = select_height_layer
    panel = StubCoveragePanel()
    app_state = SimpleNamespace(show_coverage=False, coverage_height_index=0)

    def set_state(**changes):
        for key, value in changes.items():
            setattr(app_state, key, value)

    viz = SimpleNamespace(
        coverage_data={
            "coverage_file": "coverage_maps.h5",
            "metric_name": "sinr_db",
            "value_min": 0.0,
            "value_max": 20.0,
            "grid_spacing": np.asarray([1.0, 1.0], dtype=np.float32),
            "values_3d": np.asarray([[[10.0]]], dtype=np.float32),
        },
        coverage_heights=[1.0, 5.0],
        coverage_height_index=0,
        pipeline=StubPipeline(),
        _coverage_interpolation_dirty=False,
        app_state=app_state,
        set_state=set_state,
        ui_manager=SimpleNamespace(
            panels={"coverage": panel},
            notify_coverage_selection_changed=lambda **kwargs: graph_notifications.append(kwargs),
        ),
    )
    controller = CoverageController(
        SimpleNamespace(visualizer=viz, coverage_service=coverage_service)
    )

    controller.handle_coverage_height_changed(1)

    assert loaded_heights == [1]
    assert viz.coverage_data["_active_height_index"] == 1
    assert viz.coverage_data["values_3d"].tolist() == [[[11.0]]]
    assert viz.coverage_height_index == 1
    assert viz.app_state.coverage_height_index == 1
    assert panel.height_indices == [1]
    assert graph_notifications == [{"render": True}]


def test_threshold_change_updates_panel_summary():
    panel = StubCoveragePanel()
    viz = SimpleNamespace(
        coverage_data={
            "available_metrics": ["best_path_loss_db"],
            "metric_name": "best_path_loss_db",
            "value_min": 80.0,
            "value_max": 130.0,
            "grid_spacing": np.array([2.0, 2.0], dtype=np.float32),
            "values_3d": np.array([[[80.0, 100.0], [130.0, np.nan]]], dtype=np.float32),
        },
        coverage_metric_name="best_path_loss_db",
        coverage_height_index=0,
        pipeline=StubPipeline(),
        _coverage_interpolation_dirty=False,
        app_state=SimpleNamespace(show_coverage=False),
        renderer=SimpleNamespace(capabilities=RendererCapabilities(transparency=True)),
        ui_manager=SimpleNamespace(panels={"coverage": panel}),
    )
    parent = SimpleNamespace(visualizer=viz, coverage_service=StubCoverageService())

    CoverageController(parent).handle_coverage_threshold_changed(True, 100.0)

    assert panel.threshold_summaries == [
        ("<= 100.0: 2/3 valid cells (66.7% valid; 50.0% total), 8.0 m^2", True)
    ]
    assert viz.coverage_threshold_enabled is True
    assert viz.coverage_threshold_value == 100.0


def test_threshold_visual_toggles_refresh_visible_coverage():
    panel = StubCoveragePanel()
    calls = []

    def record_process(step):
        calls.append(
            (
                "process",
                step,
                viz._coverage_interpolation_dirty,
                viz.force_update_next_frame,
            )
        )
        viz._coverage_interpolation_dirty = False
        viz.force_update_next_frame = False

    viz = SimpleNamespace(
        coverage_data={
            "available_metrics": ["best_path_loss_db"],
            "metric_name": "best_path_loss_db",
            "value_min": 80.0,
            "value_max": 130.0,
            "grid_spacing": np.array([1.0, 1.0], dtype=np.float32),
            "values_3d": np.array([[[80.0, 100.0], [130.0, 120.0]]], dtype=np.float32),
        },
        coverage_metric_name="best_path_loss_db",
        coverage_height_index=0,
        animation_step=4,
        pipeline=StubPipeline(),
        _coverage_interpolation_dirty=False,
        force_update_next_frame=False,
        app_state=SimpleNamespace(show_coverage=True),
        renderer=SimpleNamespace(capabilities=RendererCapabilities(transparency=True)),
        ui_manager=SimpleNamespace(panels={"coverage": panel}),
        _process_frame_step=record_process,
        update_visualizer=lambda: calls.append(("update", None)),
    )
    parent = SimpleNamespace(visualizer=viz, coverage_service=StubCoverageService())
    controller = CoverageController(parent)

    controller.handle_coverage_threshold_changed(True, 100.0)
    controller.handle_coverage_threshold_mask_changed(True)
    controller.handle_coverage_isolines_changed(True, 4)

    assert viz.coverage_threshold_mask_enabled is True
    assert viz.coverage_isolines_enabled is True
    assert viz.coverage_isoline_count == 4
    assert calls == [
        ("process", 4, True, True),
        ("update", None),
        ("process", 4, True, True),
        ("update", None),
        ("process", 4, True, True),
        ("update", None),
    ]


def test_cache_all_preserves_disabled_coverage_toggle_state():
    coverage_service = StubCoverageService()
    toggle = StubToggle(False)
    panel = StubCoveragePanel()
    panel.widgets["coverage_toggle"] = toggle
    viz = SimpleNamespace(
        coverage_data={"metric_name": "best_path_loss_db"},
        coverage_heights=[1.0, 2.0],
        coverage_height_index=0,
        animation_step=7,
        pipeline=StubPipeline(),
        _coverage_interpolation_dirty=False,
        app_state=SimpleNamespace(show_coverage=False, coverage_height_index=0),
        ui_manager=SimpleNamespace(panels={"coverage": panel}),
    )
    parent = SimpleNamespace(visualizer=viz, coverage_service=coverage_service)

    CoverageController(parent).handle_coverage_cache_all_clicked()

    assert viz.coverage_height_index == 0
    assert viz.app_state.coverage_height_index == 0
    assert viz.app_state.show_coverage is False
    assert toggle.isChecked() is False
    assert toggle.history == [False]
    assert viz.pipeline.precache_calls == 1
    assert coverage_service.logged is True


def test_categorical_metric_atomically_clears_scalar_analysis_state():
    coverage_service = StubCoverageService()
    panel = StubCoveragePanel()
    viz = SimpleNamespace(
        coverage_data={
            "available_metrics": ["best_path_loss_db", "serving_tx"],
            "metric_name": "best_path_loss_db",
            "value_min": 80.0,
            "value_max": 120.0,
        },
        coverage_metric_name="best_path_loss_db",
        coverage_height_index=0,
        coverage_interpolation_method="linear",
        pipeline=StubPipeline(),
        _coverage_interpolation_dirty=False,
        app_state=SimpleNamespace(show_coverage=False),
        renderer=SimpleNamespace(capabilities=RendererCapabilities(transparency=True)),
        ui_manager=SimpleNamespace(panels={"coverage": panel}),
    )
    controller = CoverageController(
        SimpleNamespace(visualizer=viz, coverage_service=coverage_service)
    )
    controller.coverage_interpolation_method = "linear"
    controller.coverage_threshold_enabled = True
    controller.coverage_threshold_value = 100.0
    controller.coverage_threshold_mask_enabled = True
    controller.coverage_isolines_enabled = True

    controller.handle_coverage_metric_changed("serving_tx")

    assert controller.coverage_interpolation_method == "none"
    assert controller.coverage_threshold_enabled is False
    assert controller.coverage_threshold_value is None
    assert controller.coverage_threshold_mask_enabled is False
    assert controller.coverage_isolines_enabled is False
    assert viz.coverage_interpolation_method == "none"
    assert viz.coverage_threshold_enabled is False
    assert viz.coverage_threshold_mask_enabled is False
    assert viz.coverage_isolines_enabled is False


def test_scene_only_coverage_refresh_uses_overlay_pipeline():
    calls = []
    pipeline = StubPipeline()
    viz = SimpleNamespace(
        coverage_data={"metric_name": "sinr_db"},
        coverage_height_index=0,
        animation_step=3,
        pipeline=pipeline,
        _scene_only_mode=True,
        ready=False,
        _coverage_interpolation_dirty=False,
        force_update_next_frame=False,
        app_state=SimpleNamespace(show_coverage=True),
        ui_manager=SimpleNamespace(panels={}),
        _process_frame_step=lambda step: calls.append(("process", step)),
        update_visualizer=lambda: calls.append(("update", None)),
    )
    controller = CoverageController(
        SimpleNamespace(visualizer=viz, coverage_service=StubCoverageService())
    )

    controller.handle_coverage_isolines_changed(True, 4)

    assert pipeline.overlay_calls == 1
    assert calls == [("update", None)]


def test_threshold_summary_uses_smoothed_displayed_slice():
    service = StubCoverageService()
    service.interpolation_result = np.zeros((1, 3), dtype=np.float32)
    panel = StubCoveragePanel()
    viz = SimpleNamespace(
        coverage_data={
            "available_metrics": ["sinr_db"],
            "metric_name": "sinr_db",
            "value_min": 0.0,
            "value_max": 100.0,
            "grid_spacing": np.array([1.0, 1.0], dtype=np.float32),
            "values_3d": np.array([[[0.0, 100.0, 0.0]]], dtype=np.float32),
        },
        coverage_height_index=0,
        pipeline=StubPipeline(),
        _coverage_interpolation_dirty=False,
        app_state=SimpleNamespace(show_coverage=False),
        ui_manager=SimpleNamespace(panels={"coverage": panel}),
    )
    controller = CoverageController(SimpleNamespace(visualizer=viz, coverage_service=service))
    controller.coverage_interpolation_method = "linear"

    # Enabling chooses a default from the displayed (smoothed) data.
    controller.handle_coverage_threshold_changed(True, 50.0)
    assert controller.coverage_threshold_value == 0.0

    # An explicit 50 dB threshold covers one raw spike but no displayed cells.
    controller.handle_coverage_threshold_changed(True, 50.0)
    assert panel.threshold_summaries[-1] == (
        ">= 50.0: 0/3 valid cells (0.0% valid; 0.0% total), 0.0 m^2",
        True,
    )
    assert service.interpolation_calls[-1][1] == "linear"


def test_uncached_probe_does_not_copy_cached_mesh_arrays():
    service = StubCoverageService()
    viz = SimpleNamespace(
        coverage_data={"metric_name": "sinr_db"},
        coverage_heights=[1.0, 2.0],
        coverage_height_index=0,
    )
    controller = CoverageController(SimpleNamespace(visualizer=viz, coverage_service=service))

    assert controller._any_heights_uncached() is True
    assert service.mesh_copy_args == [False]
