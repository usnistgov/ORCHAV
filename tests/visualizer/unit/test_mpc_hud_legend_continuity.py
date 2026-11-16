"""MPC color keys survive sparse CPI rows without retaining scene geometry."""

from dataclasses import replace
from types import SimpleNamespace
from weakref import ref

import numpy as np
import pytest

from visualizer.src.renderers.pygfx.overlays import PygfxOverlayMixin
from visualizer.src.state import MpcVisibility, create_initial_state


def _renderer(mode="mpc_type"):
    renderer = PygfxOverlayMixin()
    renderer.visualizer = SimpleNamespace(
        app_state=create_initial_state(color_mode=mode, show_mpc_type_markers=True),
        frame_source=object(),
        _mpc_presented_source_epoch=1,
        mpc_allowed_materials=None,
        mpc_core=SimpleNamespace(
            _order_palette=np.ones((7, 3)),
            _type_palette=np.ones((9, 3)),
            material_legend_items=lambda *_args, **_kwargs: [("Concrete", (0.5, 0.5, 0.5))],
        ),
    )
    renderer._semantic_legend_cache_key = None
    renderer._semantic_legend_cache_html = ""
    for method in (
        "_update_status_chip_overlay",
        "_update_filter_hud_overlay",
        "_update_coverage_hud_overlay",
        "_update_trajectory_hud_overlay",
    ):
        setattr(renderer, method, lambda *_args: None)
    overlays = {}
    renderer._set_hud_overlay = lambda key, **value: overlays.update({key: value["html"]})
    renderer._clear_hud_overlay = lambda key: overlays.pop(key, None)
    return renderer, overlays


def _packet(*, populated=True, markers=True, colorbar=None):
    return SimpleNamespace(
        mpc_points=np.ones((2 if populated else 0, 3)),
        mpc_lines=np.zeros((1 if populated else 0, 2), dtype=int),
        mpc_bounce_points=np.ones((1 if populated and markers else 0, 3)),
        mpc_bounce_itypes=np.ones(1 if populated and markers else 0, dtype=np.uint8),
        mpc_line_itype_codes=(0, 1) if populated else (),
        colorbar=colorbar,
    )


@pytest.mark.parametrize("mode", ["reflection_order", "mpc_type", "material", "delay", "path_loss"])
def test_legends_remain_identical_across_empty_cpi_frames(mode):
    renderer, overlays = _renderer(mode)
    packet = _packet(colorbar=("Scale", (10.0, 20.0)))
    points_ref = ref(packet.mpc_points)
    renderer._update_mpc_hud_overlays(packet)
    populated = dict(overlays)
    assert populated

    for _ in range(3):
        # Sparse packets may carry fallback ranges; they must not replace the
        # scale used for the last actual paths.
        renderer._update_mpc_hud_overlays(_packet(populated=False, colorbar=("Scale", (0.0, 1.0))))
        assert overlays == populated
    snapshot = renderer._mpc_legend_state
    assert snapshot.colorbar == ("Scale", (10.0, 20.0))
    del packet
    assert points_ref() is None


def test_never_show_mpc_legends_for_recording_without_paths():
    renderer, overlays = _renderer("delay")
    for _ in range(3):
        renderer._update_mpc_hud_overlays(
            _packet(populated=False, colorbar=("Delay (ns)", (0.0, 1.0)))
        )
    assert not overlays


@pytest.mark.parametrize("change", ["mode", "source", "epoch", "renderer", "filter", "selection"])
def test_sparse_legend_does_not_leak_across_changes(change):
    renderer, overlays = _renderer()
    renderer._update_mpc_hud_overlays(_packet())
    assert "mpc_semantic_legend" in overlays
    viz = renderer.visualizer
    if change == "mode":
        viz.app_state = replace(viz.app_state, color_mode="delay")
    elif change == "source":
        viz.frame_source = object()
    elif change == "epoch":
        viz._mpc_presented_source_epoch += 1
    elif change == "renderer":
        renderer._session_generation = 2
    elif change == "filter":
        viz.app_state = replace(viz.app_state, delay_filter_min_ns=10.0)
    else:
        viz.app_state = replace(viz.app_state, selected_tx=1)
    renderer._update_mpc_hud_overlays(_packet(populated=False))
    assert not overlays


def test_explicit_mpc_off_and_no_frame_reset_legend_history():
    renderer, overlays = _renderer()
    renderer._update_mpc_hud_overlays(_packet())
    state = renderer.visualizer.app_state
    renderer.visualizer.app_state = replace(state, mpc_visibility=MpcVisibility(enabled=False))
    renderer._update_mpc_hud_overlays(_packet(populated=False))
    assert not overlays
    renderer.visualizer.app_state = state
    renderer._update_mpc_hud_overlays(_packet(populated=False))
    assert not overlays
    renderer._update_mpc_hud_overlays(_packet())
    assert overlays
    renderer._update_mpc_hud_overlays(None)
    assert not overlays


@pytest.mark.parametrize("change", ["materials", "numeric_precision", "material_scope"])
def test_sparse_legend_filter_identity_uses_exact_values(change):
    renderer, overlays = _renderer()
    viz = renderer.visualizer
    viz.mpc_allowed_materials = {"concrete", "glass"}
    viz.mpc_material_filter_scope = "segment"
    viz.app_state = replace(viz.app_state, delay_filter_min_ns=10.000001)
    renderer._update_mpc_hud_overlays(_packet())
    assert overlays

    if change == "materials":
        viz.mpc_allowed_materials = {"brick", "wood"}
    elif change == "numeric_precision":
        viz.app_state = replace(viz.app_state, delay_filter_min_ns=10.000002)
    else:
        viz.mpc_material_filter_scope = "path"
    renderer._update_mpc_hud_overlays(_packet(populated=False))
    assert not overlays


@pytest.mark.parametrize("paths,bounces", [(True, False), (False, True), (False, False)])
def test_legends_follow_requested_path_and_bounce_visibility(paths, bounces):
    renderer, overlays = _renderer()
    renderer.visualizer.app_state = replace(
        renderer.visualizer.app_state,
        mpc_visibility=MpcVisibility(paths=paths, bounce_points=bounces),
    )
    packet = _packet()
    if not paths:
        packet.mpc_lines = np.empty((0, 2), dtype=int)
        packet.mpc_line_itype_codes = ()
    renderer._update_mpc_hud_overlays(packet)
    assert ("mpc_semantic_legend" in overlays) is (paths or bounces)
    assert ("mpc_marker_legend" in overlays) is bounces
    renderer._update_mpc_hud_overlays(_packet(populated=False))
    assert ("mpc_semantic_legend" in overlays) is (paths or bounces)
    assert ("mpc_marker_legend" in overlays) is bounces


def test_marker_key_requires_observed_bounce_data():
    renderer, overlays = _renderer()
    renderer._update_mpc_hud_overlays(_packet(markers=False))
    assert "mpc_semantic_legend" in overlays
    assert "mpc_marker_legend" not in overlays


def test_palette_change_updates_retained_key_and_next_paths_update_scale():
    renderer, overlays = _renderer()
    renderer._update_mpc_hud_overlays(_packet())
    before = overlays["mpc_semantic_legend"]
    renderer.visualizer.mpc_core._type_palette[:] = 0.0
    renderer._update_mpc_hud_overlays(_packet(populated=False))
    assert overlays["mpc_semantic_legend"] != before
    assert "#000000" in overlays["mpc_semantic_legend"]

    renderer.visualizer.app_state = replace(renderer.visualizer.app_state, color_mode="delay")
    renderer._update_mpc_hud_overlays(_packet(colorbar=("Delay (ns)", (10.0, 20.0))))
    first = overlays["mpc_colorbar"]
    renderer._update_mpc_hud_overlays(_packet(colorbar=("Delay (ns)", (30.0, 40.0))))
    assert overlays["mpc_colorbar"] != first


def test_historical_packet_visibility_does_not_override_user_legend_intent():
    renderer, overlays = _renderer()
    renderer._update_mpc_hud_overlays(_packet())
    previous = dict(overlays)
    packet = _packet(populated=False)
    packet.mpc_visibility = MpcVisibility(enabled=False)
    renderer._update_mpc_hud_overlays(packet)
    assert overlays == previous
