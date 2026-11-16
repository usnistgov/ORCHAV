from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel

from shared.extensions.sensing import UnsupportedSensingSchemaError, assert_sensing_contract
from shared.frames import FrameProjection, FrameReadRequest, StandardMPCFrame
from shared.frames.provider_base import DataProvider
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload
from visualizer.src.services.sensing_service import SensingService

_APP = QApplication.instance() or QApplication([])

try:
    from visualizer.src.panels.sensing_panel import SensingPanel, _select_variant

    HAS_SENSING_PANEL = True
except ImportError:
    HAS_SENSING_PANEL = False


def _typed_range_doppler_payload(
    fill_value: float,
    *,
    scene_frame_id: int,
    sample_time_s: float,
) -> dict:
    """Return one exact-axis schema-v4 range-Doppler power fixture."""
    return build_visualizer_sensing_payload(
        {
            "config": {"range_mode": "monostatic"},
            "range_axis_m": np.array(
                [0.0, 0.2, 0.55, 1.1, 1.9, 3.0, 4.4, 6.1],
                dtype=np.float64,
            ),
            "doppler_axis_hz": np.array([-13.0, -4.0, 2.5, 11.0], dtype=np.float64),
            "range_doppler_power_by_pair": np.full(
                (1, 4, 8),
                fill_value,
                dtype=np.float32,
            ),
        },
        scene_frame_id=scene_frame_id,
        sample_time_s=sample_time_s,
    )


def _typed_dashboard_payload(*, scene_frame_id: int, sample_time_s: float) -> dict:
    """Return one acquisition containing every dense dashboard product family."""
    range_axis_m = np.linspace(0.0, 3.5, 8, dtype=np.float64)
    doppler_axis_hz = np.array([-12.0, -4.0, 4.0, 12.0], dtype=np.float64)
    return build_visualizer_sensing_payload(
        {
            "config": {"range_mode": "monostatic"},
            "range_axis_m": range_axis_m,
            "doppler_axis_hz": doppler_axis_hz,
            "elevation_direction_cosine": np.array([0.0], dtype=np.float64),
            "azimuth_direction_cosine": np.array([-0.5, 0.0, 0.5], dtype=np.float64),
            "elevation_broadside_angle_deg": np.array([0.0], dtype=np.float64),
            "azimuth_broadside_angle_deg": np.array([-30.0, 0.0, 30.0], dtype=np.float64),
            "cir_by_pair_rx_element": np.ones((1, 1, 8), dtype=np.complex64),
            "slow_time_by_pair_rx_element": np.ones((1, 1, 1, 8), dtype=np.complex64),
            "range_profile_by_pair": np.ones((1, 8), dtype=np.float32),
            "range_doppler_power_by_pair": np.ones((1, 4, 8), dtype=np.float32),
            "range_doppler_angle_by_pair": np.ones((1, 1, 3, 4, 8), dtype=np.float32),
        },
        scene_frame_id=scene_frame_id,
        sample_time_s=sample_time_s,
    )


class _PanelFrameProvider(DataProvider):
    """Selective provider used to exercise the panel's production service path."""

    def __init__(self, sensing_by_step: dict[int, dict]) -> None:
        self.frames = {
            step: replace(
                build_standard_mpc_frame(frame_idx=step),
                sensing=sensing,
            )
            for step, sensing in sensing_by_step.items()
        }

    def list_frames(self) -> list[int]:
        return sorted(self.frames)

    def has_frame(self, step: int) -> bool:
        return step in self.frames

    def load_frame(self, step: int) -> StandardMPCFrame:
        return self.frames[step]

    def load_frame_projection(self, step: int, request: FrameReadRequest) -> FrameProjection:
        return super().load_frame_projection(step, request)


def test_cached_frame_fallback_projects_direct_standard_frame() -> None:
    """The dashboard accepts canonical frames from a non-provider source."""
    if not HAS_SENSING_PANEL:
        pytest.skip("SensingPanel dependencies are unavailable in this environment")

    range_profile = np.arange(16, dtype=np.float32)[np.newaxis, :]
    frame = replace(
        build_standard_mpc_frame(frame_idx=0),
        sensing=build_visualizer_sensing_payload(
            {
                "config": {"range_mode": "monostatic"},
                "range_axis_m": np.linspace(0.0, 3.0, 16, dtype=np.float64),
                "range_profile_by_pair": range_profile,
            },
        ),
    )
    parent = SimpleNamespace(
        cache_service=SimpleNamespace(get_frame=lambda _step: None),
        frame_loader=None,
        frame_source=SimpleNamespace(load_frame=lambda _step: frame),
        mpc_core=SimpleNamespace(canon_points_dtype=np.float32),
    )
    panel = SensingPanel.__new__(SensingPanel)
    panel.parent = parent

    loaded = panel._get_cached_frame(0)

    assert loaded is not None
    assert "canonical_data" in loaded
    np.testing.assert_array_equal(loaded["sensing"]["range_profile_by_pair"], range_profile)


def test_panel_exact_scene_load_failure_does_not_substitute_first_available_frame() -> None:
    if not HAS_SENSING_PANEL:
        pytest.skip("SensingPanel dependencies are unavailable in this environment")

    load_calls: list[int] = []

    def _load_frame(step: int):
        load_calls.append(step)
        if step == 20:
            raise KeyError(step)
        return build_standard_mpc_frame(frame_idx=step)

    status = QLabel()
    parent = SimpleNamespace(
        app_state=SimpleNamespace(step=20, extension_state={}),
        cache_service=SimpleNamespace(
            get_frame=lambda _step: None,
            store_frame=lambda *_args: None,
        ),
        frame_loader=None,
        frame_source=SimpleNamespace(
            load_frame=_load_frame,
            list_frames=lambda: [5, 20],
        ),
        extension_services={"sensing": object()},
    )
    panel = SensingPanel(parent)
    panel.widgets["status"] = status
    unavailable_calls = []
    panel._sensing_window = SimpleNamespace(
        isVisible=lambda: True,
        set_unavailable=lambda reason, **kwargs: unavailable_calls.append((reason, kwargs)),
    )

    panel._refresh_live_plots(force=True)

    assert unavailable_calls == [("No frame loaded at step 20", {"scene_frame_id": 20})]
    assert load_calls == [20]
    assert status.text() == "Status: No frame loaded at step 20"


def test_closed_dashboard_timer_does_not_load_or_process_frames() -> None:
    """Scene playback must not pay for a hidden dashboard's dense products."""
    parent = SimpleNamespace(extension_services={"sensing": object()})
    panel = SensingPanel(parent)
    panel.widgets["status"] = QLabel()

    def unexpected_load():
        pytest.fail("closed dashboard requested a frame")

    panel._resolve_frame = unexpected_load
    panel._refresh_live_plots()
    panel._refresh_live_plots(force=True)
    assert "Select Sensing" in panel.widgets["status"].text()


def test_recorded_paths_toggle_controls_mpc_and_mirrors_external_visibility(qtbot) -> None:
    from visualizer.src.sensing.runtime import state_value

    calls = []
    visibility = SimpleNamespace(enabled=False, paths=False)

    def change_visibility(reason, **changes):
        calls.append(changes)
        for key, value in changes.items():
            setattr(visibility, key, value)

    parent = SimpleNamespace(
        app_state=SimpleNamespace(step=1, extension_state={}, mpc_visibility=visibility),
        ui_controller=SimpleNamespace(_set_mpc_visibility=change_visibility),
        schedule_update=lambda: None,
    )
    parent.set_state = lambda **changes: vars(parent.app_state).update(changes)
    panel = SensingPanel(parent)
    panel._start_auto_refresh = lambda: None
    container = panel.create_panel()
    qtbot.addWidget(container)
    paths = panel.widgets["show_recorded_paths"]
    paths.setChecked(True)
    assert calls == [{"enabled": True, "paths": True}]
    assert state_value(parent.app_state, "show_rt_context")
    paths.setChecked(False)
    assert calls[-1] == {"paths": False}
    assert not state_value(parent.app_state, "show_rt_context")
    visibility.paths = True
    panel._sync_runtime_controls()
    assert paths.isChecked()
    assert len(calls) == 2  # Mirroring another panel never emits a new mutation.
    assert panel._sensing_window is None


def test_panel_routes_candidate_scan_through_service_metadata_index() -> None:
    """The production panel must not supply its broad visual-frame loader."""
    if not HAS_SENSING_PANEL:
        pytest.skip("SensingPanel dependencies are unavailable in this environment")

    captured: dict[str, object] = {}

    class _ServiceProbe:
        def resolve_display_sensing_frame(self, *_args, **kwargs):
            captured.update(kwargs)
            return None

    panel = SensingPanel.__new__(SensingPanel)
    panel.parent = SimpleNamespace()
    panel._prefer_fresh_sensing_display = lambda: False  # type: ignore[method-assign]

    panel._resolve_display_selection(_ServiceProbe(), {"sensing": {}}, 5)

    assert "load_candidate" not in captured
    assert captured["load_all_sensing_products"] is True
    assert captured["load_scene_frame"] == panel._get_cached_frame


def test_panel_loads_complete_selected_acquisition_for_dashboard_contract() -> None:
    """Metadata selection must not leave declared dashboard arrays unloaded."""
    if not HAS_SENSING_PANEL:
        pytest.skip("SensingPanel dependencies are unavailable in this environment")

    result_sensing = _typed_dashboard_payload(scene_frame_id=5, sample_time_s=5.0)
    provider = _PanelFrameProvider({5: {}, 10: result_sensing})
    parent = SimpleNamespace(
        app_state=SimpleNamespace(extension_state={}),
        frame_loader=SimpleNamespace(provider=provider),
        frame_source=SimpleNamespace(list_frames=provider.list_frames),
    )
    panel = SensingPanel(parent)
    frames = {5: {"sensing": {}}, 10: {"sensing": result_sensing}}
    panel._get_cached_frame = lambda step: frames.get(step)  # type: ignore[method-assign]

    selection = panel._resolve_display_selection(SensingService(parent), frames[5], 5)

    assert selection is not None
    assert_sensing_contract(
        selection.display_frame,
        require_schema_version=True,
        require_described_products=True,
    )
    sensing = selection.display_frame["sensing"]
    assert "cir_by_pair_rx_element" in sensing
    assert "slow_time_axis_s" in sensing
    assert "range_doppler_angle_by_pair" in sensing
    assert "azimuth_direction_cosine" in sensing


def test_panel_resolves_later_storage_row_only_when_binding_matches_scene() -> None:
    if not HAS_SENSING_PANEL:
        pytest.skip("SensingPanel dependencies are unavailable in this environment")
    result_sensing = _typed_range_doppler_payload(
        1.0,
        scene_frame_id=5,
        sample_time_s=5.0,
    )
    provider = _PanelFrameProvider({5: {}, 10: result_sensing})
    parent = SimpleNamespace(
        app_state=SimpleNamespace(extension_state={}),
        frame_loader=SimpleNamespace(provider=provider),
        frame_source=SimpleNamespace(list_frames=provider.list_frames),
    )
    panel = SensingPanel(parent)
    frames = {
        5: {
            "rx_positions": np.array([[5.0, 0.0, 0.0]], dtype=np.float32),
            "sensing": {},
        },
        10: {
            "rx_positions": np.array([[10.0, 0.0, 0.0]], dtype=np.float32),
            "sensing": result_sensing,
        },
    }
    panel._get_cached_frame = lambda step: frames.get(step)  # type: ignore[method-assign]
    service = SensingService(parent)

    selection = panel._resolve_display_selection(service, frames[5], 5)

    assert selection is not None
    assert selection.storage_frame_id == 10
    assert selection.bound_scene_frame_id == 5
    np.testing.assert_array_equal(selection.display_frame["rx_positions"], [[5.0, 0.0, 0.0]])
    assert np.size(selection.display_frame["sensing"]["range_doppler_power_by_pair"]) > 0


def test_panel_does_not_display_unbound_prior_or_future_product() -> None:
    if not HAS_SENSING_PANEL:
        pytest.skip("SensingPanel dependencies are unavailable in this environment")
    prior_sensing = _typed_range_doppler_payload(
        1.0,
        scene_frame_id=10,
        sample_time_s=10.0,
    )
    future_sensing = _typed_range_doppler_payload(
        3.0,
        scene_frame_id=30,
        sample_time_s=30.0,
    )
    provider = _PanelFrameProvider({10: prior_sensing, 20: {}, 30: future_sensing})
    parent = SimpleNamespace(
        app_state=SimpleNamespace(extension_state={}),
        frame_loader=SimpleNamespace(provider=provider),
        frame_source=SimpleNamespace(list_frames=provider.list_frames),
    )
    panel = SensingPanel(parent)
    frames = {
        10: {
            "sensing": prior_sensing,
        },
        20: {"sensing": {}},
        30: {
            "rx_positions": np.array([[30.0, 0.0, 0.0]], dtype=np.float32),
            "sensing": future_sensing,
        },
    }
    panel._get_cached_frame = lambda step: frames.get(step)  # type: ignore[method-assign]
    service = SensingService(parent)

    selection = panel._resolve_display_selection(service, frames[20], 20)

    assert selection is None


def test_panel_pin_freshest_is_explicit_and_surfaces_stale_binding() -> None:
    if not HAS_SENSING_PANEL:
        pytest.skip("SensingPanel dependencies are unavailable in this environment")
    prior_sensing = _typed_range_doppler_payload(
        1.0,
        scene_frame_id=10,
        sample_time_s=10.0,
    )
    latest_sensing = _typed_range_doppler_payload(
        3.0,
        scene_frame_id=30,
        sample_time_s=30.0,
    )
    provider = _PanelFrameProvider({10: prior_sensing, 20: {}, 30: latest_sensing})
    parent = SimpleNamespace(
        app_state=SimpleNamespace(extension_state={}),
        frame_loader=SimpleNamespace(provider=provider),
        frame_source=SimpleNamespace(list_frames=provider.list_frames),
    )
    panel = SensingPanel(parent)
    pin = QCheckBox()
    pin.setChecked(True)
    panel.widgets["pin_fresh_sensing"] = pin
    frames = {
        10: {
            "sensing": prior_sensing,
        },
        20: {"timestamp_s": 35.0, "sensing": {}},
        30: {
            "sensing": latest_sensing,
        },
    }
    panel._get_cached_frame = lambda step: frames.get(step)  # type: ignore[method-assign]
    service = SensingService(parent)

    selection = panel._resolve_display_selection(service, frames[20], 20)

    assert selection is not None
    assert selection.pinned is True
    assert selection.bound_scene_frame_id == 30
    assert selection.current_scene_frame_id == 20
    assert selection.stale_for_current_scene is True
    assert selection.scene_result_time_offset_s == pytest.approx(5.0)
    assert selection.stale_age_s == pytest.approx(5.0)
    status = panel._format_display_provenance(selection)
    assert "scene-result offset +5 s" in status
    assert "stale age 5 s" in status


def test_pin_freshest_toggle_schedules_3d_redraw(monkeypatch: pytest.MonkeyPatch) -> None:
    if not HAS_SENSING_PANEL:
        pytest.skip("SensingPanel dependencies are unavailable in this environment")

    state_updates: list[dict[str, bool]] = []
    redraws: list[bool] = []
    refreshes: list[tuple[bool, bool]] = []
    parent = SimpleNamespace(
        force_update_next_frame=False,
        schedule_update=lambda: redraws.append(True),
    )
    panel = SensingPanel.__new__(SensingPanel)
    panel.parent = parent
    panel._last_display_identity = (1, 2, 3)
    panel._refresh_live_plots = (  # type: ignore[method-assign]
        lambda force=False, refresh_rd=True: refreshes.append((force, refresh_rd))
    )
    monkeypatch.setattr(
        "visualizer.src.panels.sensing_panel.update_state",
        lambda _viz, **changes: state_updates.append(changes),
    )

    panel._on_pin_freshest_changed(1)

    assert state_updates == [{"pin_freshest": True}]
    assert panel._last_display_identity is None
    assert refreshes == [(True, True)]
    assert parent.force_update_next_frame is True
    assert redraws == [True]


def test_build_update_request_carries_display_and_state_steps() -> None:
    if not HAS_SENSING_PANEL:
        pytest.skip("SensingPanel dependencies are unavailable in this environment")

    replay_event = object()
    detection_stage_bundle = object()
    request = SensingPanel._build_update_request(  # type: ignore[attr-defined]
        display_frame={"sensing": build_visualizer_sensing_payload({"rd_source": "fft_cpi"})},
        state_frame={"sensing": {}},
        state_step=10,
        display_step=8,
        display_from_cache=True,
        coherent_cir_time_steps=4,
        rp_result=None,
        rd_result=None,
        variant="clean",
        range_mode="monostatic",
        carrier_freq_hz=28e9,
        prf_hz=1000.0,
        sampling_frequency=1e9,
        cir_history=None,
        cir_history_times_s=None,
        replay_event=replay_event,
        detection_stage_bundle=detection_stage_bundle,
    )

    assert request.state_step == 10
    assert request.display_step == 8
    assert request.effective_display_step == 8
    assert request.display_from_cache is True
    assert request.replay_event is replay_event
    assert request.detection_stage_bundle is detection_stage_bundle


def test_panel_sensing_boundary_requires_regeneration_for_pre_v3() -> None:
    if not HAS_SENSING_PANEL:
        pytest.skip("SensingPanel dependencies are unavailable in this environment")

    with pytest.raises(
        UnsupportedSensingSchemaError,
        match="Regenerate this sensing dataset",
    ):
        SensingPanel._has_displayable_sensing_products(  # type: ignore[attr-defined]
            {
                "schema_version": 2,
                "range_profile": np.ones(4, dtype=np.float32),
            }
        )


def test_panel_availability_uses_typed_descriptors_not_legacy_arrays() -> None:
    if not HAS_SENSING_PANEL:
        pytest.skip("SensingPanel dependencies are unavailable in this environment")

    legacy_only = build_visualizer_sensing_payload(
        {"range_doppler_map": np.ones((2, 3), dtype=np.float32)}
    )
    typed = _typed_range_doppler_payload(
        2.0,
        scene_frame_id=0,
        sample_time_s=0.0,
    )

    with pytest.raises(ValueError, match="forbids compatibility aliases"):
        SensingPanel._has_displayable_sensing_products(legacy_only)
    assert SensingPanel._has_displayable_sensing_products(typed) is True


def test_panel_runtime_config_uses_only_complete_config_resolved() -> None:
    if not HAS_SENSING_PANEL:
        pytest.skip("SensingPanel dependencies are unavailable in this environment")

    panel = SensingPanel.__new__(SensingPanel)
    sensing = build_visualizer_sensing_payload()

    runtime = panel._resolve_runtime_config(sensing)

    assert runtime == {
        "sampling_frequency": 1.0e9,
        "prf_hz": 1000.0,
        "fft_size_range": 1,
        "fft_size_doppler": 1,
        "carrier_freq_hz": 28.0e9,
        "cir_time_steps": 1,
    }

    sensing.pop("config_resolved")
    with pytest.raises(ValueError, match="requires a complete sensing.config_resolved mapping"):
        panel._resolve_runtime_config(sensing)


def test_panel_variant_selection_never_falls_back_across_primary_and_raw() -> None:
    if not HAS_SENSING_PANEL:
        pytest.skip("SensingPanel dependencies are unavailable in this environment")

    assert _select_variant("clean", frozenset({"primary"})) == "primary"
    assert _select_variant("raw", frozenset({"primary"})) is None
    assert _select_variant("primary", frozenset({"raw"})) is None
    assert _select_variant("auto", frozenset({"raw"})) == "raw"


def test_cached_sensing_getter_does_not_swallow_regeneration_error() -> None:
    if not HAS_SENSING_PANEL:
        pytest.skip("SensingPanel dependencies are unavailable in this environment")

    legacy = {"sensing": {"schema_version": 2, "detections": []}}
    panel = SensingPanel.__new__(SensingPanel)
    panel.parent = SimpleNamespace(
        cache_service=SimpleNamespace(get_frame=lambda _step: legacy),
        frame_loader=None,
        frame_source=None,
    )

    with pytest.raises(
        UnsupportedSensingSchemaError,
        match="Regenerate this sensing dataset",
    ):
        panel._get_cached_frame(0)


def test_generic_projection_can_ignore_legacy_sensing_extension() -> None:
    if not HAS_SENSING_PANEL:
        pytest.skip("SensingPanel dependencies are unavailable in this environment")

    legacy_frame = replace(
        build_standard_mpc_frame(frame_idx=0),
        sensing={"schema_version": 2, "range_profile": np.ones(4, dtype=np.float32)},
    )
    panel = SensingPanel.__new__(SensingPanel)
    panel.parent = SimpleNamespace(mpc_core=SimpleNamespace(canon_points_dtype=np.float32))

    projected = panel._project_source_frame(legacy_frame, include_sensing=False)

    assert projected is not None
    assert "sensing" not in projected


def test_sensing_projection_rejects_legacy_extension() -> None:
    if not HAS_SENSING_PANEL:
        pytest.skip("SensingPanel dependencies are unavailable in this environment")

    legacy_frame = replace(
        build_standard_mpc_frame(frame_idx=0),
        sensing={"schema_version": 2, "range_profile": np.ones(4, dtype=np.float32)},
    )
    panel = SensingPanel.__new__(SensingPanel)
    panel.parent = SimpleNamespace(mpc_core=SimpleNamespace(canon_points_dtype=np.float32))

    with pytest.raises(
        UnsupportedSensingSchemaError,
        match="Regenerate this sensing dataset",
    ):
        panel._project_source_frame(legacy_frame, include_sensing=True)


def test_preset_change_updates_variant_and_dashboard_controls() -> None:
    if not HAS_SENSING_PANEL:
        pytest.skip("SensingPanel dependencies are unavailable in this environment")

    class _FakeWindow:
        def __init__(self) -> None:
            self.applied: list[dict[str, object]] = []

        def apply_display_preset(self, preset: dict[str, object]) -> None:
            self.applied.append(dict(preset))

        def close(self) -> None:
            pass

    panel = SensingPanel(SimpleNamespace())
    panel._start_auto_refresh = lambda: None  # type: ignore[method-assign]
    container = panel.create_panel()
    assert panel.widgets["pin_fresh_sensing"].isChecked() is False
    panel._sensing_window = _FakeWindow()  # type: ignore[assignment]
    refresh_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    panel._refresh_live_plots = lambda *args, **kwargs: refresh_calls.append((args, kwargs))  # type: ignore[method-assign]

    preset_box = panel.widgets["sensing_preset"]
    preset_box.setCurrentIndex(preset_box.findData("diagnostics"))
    _APP.processEvents()

    assert panel.widgets["rd_variant"].currentData() == "clean"
    assert refresh_calls
    assert panel._sensing_window.applied[-1]["view_mode"] == "diagnostics"  # type: ignore[index]
    assert panel._sensing_window.applied[-1]["show_cfar"] is True  # type: ignore[index]
    assert panel._sensing_window.applied[-1]["show_gt"] is False  # type: ignore[index]
    assert panel._sensing_window.applied[-1]["micro_doppler_mode"] == "auto"  # type: ignore[index]
    assert panel._sensing_window.applied[-1]["rd_dynamic_range_db"] == 38  # type: ignore[index]
    container.deleteLater()


def test_display_presets_separate_evaluation_truth_from_oracle_processing() -> None:
    if not HAS_SENSING_PANEL:
        pytest.skip("SensingPanel dependencies are unavailable in this environment")

    assert SensingPanel.SENSING_PRESETS["analysis"]["dashboard"]["show_gt"] is True
    for preset_name in ("micro-doppler", "diagnostics"):
        preset = SensingPanel.SENSING_PRESETS[preset_name]
        dashboard = preset["dashboard"]
        assert dashboard["show_gt"] is False

    for preset in SensingPanel.SENSING_PRESETS.values():
        dashboard = preset["dashboard"]
        assert dashboard["micro_doppler_mode"] != "tracked"
        assert dashboard["show_detections"] is True

    panel = SensingPanel(SimpleNamespace())
    panel._start_auto_refresh = lambda: None  # type: ignore[method-assign]
    container = panel.create_panel()
    scene_mode = panel.widgets["sensing_scene_mode"]
    history_index = scene_mode.findData("evidence")

    assert history_index >= 0
    assert scene_mode.itemText(history_index) == "Detection history"
    assert "History cloud" in scene_mode.toolTip()
    assert scene_mode.itemData(history_index) == "evidence"
    container.deleteLater()


def test_scene_controls_follow_overlay_and_history_dependencies(qtbot):
    from PySide6.QtCore import QSignalBlocker

    panel = SensingPanel(SimpleNamespace())
    panel._start_auto_refresh = lambda: None
    container = panel.create_panel()
    qtbot.addWidget(container)
    mode = panel.widgets["sensing_scene_mode"]
    history = panel.widgets["sensing_scene_history_frames"]
    full = panel.widgets["sensing_scene_full_history"]
    labels = panel.widgets["sensing_scene_show_track_labels"]
    arrows = panel.widgets["sensing_scene_show_velocity_arrows"]
    for choice, expected_history, expected_tracks in (
        ("detections", False, False),
        ("evidence", True, False),
        ("tracks", True, True),
        ("full", True, True),
    ):
        with QSignalBlocker(mode), QSignalBlocker(full):
            mode.setCurrentIndex(mode.findData(choice))
            full.setChecked(False)
        panel._update_scene_overlay_controls_enabled(True)
        assert history.isEnabled() == full.isEnabled() == expected_history
        assert labels.isEnabled() == arrows.isEnabled() == expected_tracks
    with QSignalBlocker(full):
        full.setChecked(True)
    panel._update_scene_overlay_controls_enabled(True)
    assert not history.isEnabled()
    assert full.isEnabled()
    panel._update_scene_overlay_controls_enabled(False)
    assert all(not w.isEnabled() for w in (mode, history, full, labels, arrows))
    assert panel.widgets["scene_options"].isHidden()


def test_dashboard_reopens_after_deletion_without_retiring_replacement(monkeypatch):
    from PySide6.QtCore import QCoreApplication, QEvent, Qt, Signal
    from PySide6.QtWidgets import QCheckBox, QLabel, QWidget

    import visualizer.src.panels.sensing_panel as module

    class Window(QWidget):
        view_changed = Signal()
        result_step_requested = Signal(int)
        result_seek_requested = Signal(object)
        selection_changed = Signal(object)
        loading_control_requested = Signal()
        locate_selection_requested = Signal()
        range_shell_changed = Signal()
        range_shell_frame_requested = Signal()
        path_inspection_requested = Signal()

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setAttribute(Qt.WA_DeleteOnClose, True)
            self.range_shell_enabled = QCheckBox(self)
            self.range_shell_cutaway = QCheckBox(self)
            self.range_shell_status = QLabel(self)

        def apply_display_preset(self, *_args):
            pass

        def closeEvent(self, event):
            self._plots_released = True
            super().closeEvent(event)

    monkeypatch.setattr(module, "SensingDashboard", Window)
    monkeypatch.setattr(module, "HAS_SENSING_WINDOW", True)
    panel = SensingPanel(SimpleNamespace())
    panel._refresh_live_plots = lambda **_kwargs: None
    panel._on_open_dashboard()
    first = panel._sensing_window
    first.close()
    panel._on_open_dashboard()  # Replace before the old deferred delete arrives.
    second = panel._sensing_window
    assert second is not first
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert panel._sensing_window is second
    assert second.isVisible()
    second.close()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert panel._sensing_window is None
    panel._on_open_dashboard()
    assert panel._sensing_window.isVisible()
    panel.cleanup()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert panel._sensing_window is None


def test_sensing_panel_cleanup_stops_timer_and_ignores_queued_refresh(monkeypatch):
    panel = SensingPanel(SimpleNamespace())
    panel._start_auto_refresh()
    timer = panel._refresh_timer
    assert timer.isActive()
    panel.cleanup()
    assert not timer.isActive()
    assert panel._refresh_timer is None
    panel._get_service = lambda: pytest.fail("Shutdown callback read scenario state")
    panel._refresh_live_plots(force=True)
    panel._on_open_dashboard()
    panel.cleanup()
    assert panel._sensing_window is None


def test_scenario_state_reset_clears_stale_overlay_widgets_without_emitting_changes():
    parent = SimpleNamespace(
        app_state=SimpleNamespace(
            extension_state={
                "sensing": {"show_detections": True, "scene_mode": "full", "full_history": True}
            }
        )
    )
    panel = SensingPanel(parent)
    panel._start_auto_refresh = lambda: None
    container = panel.create_panel()
    assert panel.widgets["show_sensing_detections"].isChecked()
    changes = []
    panel.widgets["show_sensing_detections"].stateChanged.connect(changes.append)
    parent.app_state.extension_state = {}
    panel._sync_runtime_controls()
    assert not panel.widgets["show_sensing_detections"].isChecked()
    assert panel.widgets["sensing_scene_full_history"].isChecked()
    assert panel.widgets["sensing_scene_mode"].currentData() != "full"
    assert changes == []
    container.deleteLater()


def test_overlay_controls_survive_deferred_deletion_and_truth_remains_usable(monkeypatch):
    """Exercise the event-loop deletion that ordinary processEvents skips."""
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QWidget
    from shiboken6 import isValid

    import visualizer.src.panels.sensing_panel as module

    changes = []
    parent = SimpleNamespace(schedule_update=lambda: None, renderer=None)
    monkeypatch.setattr(module, "update_state", lambda _parent, **values: changes.append(values))
    panel = SensingPanel(parent)
    container = panel.create_panel()
    container.show()
    try:
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        panel._refresh_live_plots()
        # Every control must have a live owner even when hidden in 3D options.
        for name, widget in panel.widgets.items():
            assert isValid(widget), name
            if isinstance(widget, QWidget):
                assert container.isAncestorOf(widget), name
        panel.options_section.expand()
        panel.widgets["show_sensing_detections"].click()
        assert not panel.widgets["scene_options"].isHidden()
        truth = panel.widgets["sensing_scene_show_truth"]
        assert truth.isVisible()
        scene_tip = panel.widgets["show_sensing_detections"].toolTip()
        assert "sensor plane" in scene_tip
        assert "assumed" in scene_tip
        truth.setChecked(False)
        assert changes == [{"show_detections": True}, {"show_truth": False}]
        panel._sync_runtime_controls()
        assert truth.isChecked()  # Default state restored with signals blocked.
        assert changes == [{"show_detections": True}, {"show_truth": False}]
    finally:
        panel.cleanup()
        container.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_deleting_panel_retires_readers_and_both_timers():
    from PySide6.QtCore import QCoreApplication, QEvent, QTimer
    from shiboken6 import isValid

    panel = SensingPanel(SimpleNamespace())
    container = panel.create_panel()
    closed = []
    panel._dashboard_session = SimpleNamespace(close=lambda: closed.append(True))
    panel._dashboard_poll_timer = QTimer()
    panel._dashboard_poll_timer.start(25)
    refresh, poll = panel._refresh_timer, panel._dashboard_poll_timer
    container.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert panel._closed
    assert closed == [True]
    assert panel._dashboard_session is None
    assert panel._refresh_timer is panel._dashboard_poll_timer is None
    assert not panel.widgets
    assert not isValid(refresh) or not refresh.isActive()
    assert not isValid(poll) or not poll.isActive()
    panel._get_service = lambda: pytest.fail("Deleted panel accessed scenario state")
    panel._refresh_live_plots()
    panel._poll_dashboard()
