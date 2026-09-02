from types import SimpleNamespace

from visualizer.src.services.viewport_hud_service import (
    build_path_filter_summary,
    build_trajectory_hud_legend,
    normalize_viewport_hud_mode,
    viewport_hud_policy,
)
from visualizer.src.state import (
    DEFAULT_MPC_ALLOWED_ORDERS,
    DEFAULT_MPC_ALLOWED_TYPES,
    AppState,
    create_initial_state,
    update_state,
)


def _state(**changes):
    values = {
        "viewport_hud_enabled": True,
        "viewport_hud_mode": "compact",
        "viewport_hud_show_status": True,
        "viewport_hud_show_legends": True,
        "viewport_hud_show_filters": True,
        "viewport_hud_show_annotations": True,
        "mpc_allowed_orders": DEFAULT_MPC_ALLOWED_ORDERS,
        "mpc_allowed_types": DEFAULT_MPC_ALLOWED_TYPES,
        "delay_filter_min_ns": None,
        "delay_filter_max_ns": None,
        "power_filter_min_db": None,
        "power_filter_max_db": None,
        "aoa_az_filter_min_deg": None,
        "aoa_az_filter_max_deg": None,
        "aoa_el_filter_min_deg": None,
        "aoa_el_filter_max_deg": None,
        "aod_az_filter_min_deg": None,
        "aod_az_filter_max_deg": None,
        "aod_el_filter_min_deg": None,
        "aod_el_filter_max_deg": None,
        "topk_render_enabled": False,
        "topk_render_max_paths": 20_000,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_hud_policy_normalizes_mode_and_applies_master_visibility():
    assert normalize_viewport_hud_mode("DETAILED") == "detailed"
    assert normalize_viewport_hud_mode("invalid") == "compact"
    assert normalize_viewport_hud_mode("off") == "compact"

    policy = viewport_hud_policy(_state(viewport_hud_enabled=False, viewport_hud_mode="detailed"))

    assert not policy.enabled
    assert policy.mode == "detailed"
    assert not policy.show_status
    assert not policy.show_legends
    assert not policy.show_filters
    assert not policy.show_annotations


def test_hud_policy_accepts_legacy_combined_off_state():
    policy = viewport_hud_policy(
        SimpleNamespace(
            viewport_hud_mode="off",
            viewport_hud_show_status=True,
            viewport_hud_show_legends=True,
            viewport_hud_show_filters=True,
            viewport_hud_show_annotations=True,
        )
    )

    assert policy.enabled is False
    assert policy.mode == "compact"


def test_filter_summary_is_empty_for_default_state():
    summary = build_path_filter_summary(_state())

    assert not summary.active
    assert summary.active_count == 0


def test_filter_summary_compacts_groups_and_bounds_details():
    summary = build_path_filter_summary(
        _state(
            mpc_allowed_orders=frozenset({0, 1, 2, 4}),
            mpc_allowed_types=frozenset({0, 8, 99}),
            delay_filter_min_ns=10.0,
            power_filter_max_db=85.0,
            topk_render_enabled=True,
            topk_render_max_paths=1250,
        ),
        allowed_materials=("itu_concrete", "itu_glass"),
    )

    assert summary.active_count == 6
    assert summary.compact_text == "Paths filtered · 6 filters"
    assert summary.details == (
        "Orders: 0–2, 4",
        "Types: LoS, Diffract, Virtual",
        "Materials: 2 selected",
        "Delay ≥ 10 ns",
        "Path loss ≤ 85 dB",
        "Top paths: 1,250",
    )


def test_trajectory_legend_uses_semantic_units_and_normalizes_range():
    legend = build_trajectory_hud_legend("speed", (4.0, 1.0))

    assert legend is not None
    assert legend.title == "Trajectory Speed"
    assert legend.unit == "m/frame"
    assert legend.value_range == (1.0, 4.0)
    assert build_trajectory_hud_legend("node_color", (0.0, 1.0)) is None
    assert build_trajectory_hud_legend("altitude", None) is None


def test_hud_policy_round_trips_through_app_state_serialization():
    state = create_initial_state(
        viewport_hud_enabled=False,
        viewport_hud_mode="detailed",
        viewport_hud_show_status=False,
        viewport_hud_show_legends=True,
        viewport_hud_show_filters=False,
        viewport_hud_show_annotations=False,
    )

    restored = AppState.from_dict(state.to_dict())

    assert restored.viewport_hud_enabled is False
    assert restored.viewport_hud_mode == "detailed"
    assert restored.viewport_hud_show_status is False
    assert restored.viewport_hud_show_legends is True
    assert restored.viewport_hud_show_filters is False
    assert restored.viewport_hud_show_annotations is False


def test_overlay_manager_off_mode_does_not_create_or_update_labels():
    from visualizer.src.renderers.pygfx.overlays import PygfxOverlayMixin

    renderer = PygfxOverlayMixin()
    renderer.visualizer = SimpleNamespace(
        app_state=create_initial_state(
            viewport_hud_enabled=False,
            viewport_hud_mode="detailed",
        )
    )
    renderer._hud_overlay_labels = {}
    renderer._hud_overlay_specs = {}
    renderer._container = object()
    renderer._ensure_hud_overlay_label = lambda *_args: (_ for _ in ()).throw(
        AssertionError("off mode must not create labels")
    )

    renderer._set_hud_overlay(
        "legend",
        html="<b>unused</b>",
        visible=True,
        role="legend",
        corner="top_right",
        priority=10,
    )

    assert renderer._hud_overlay_labels == {}
    assert renderer._hud_overlay_specs == {}


def test_hud_off_skips_hover_metadata_work():
    from visualizer.src.renderers.pygfx.overlays import PygfxOverlayMixin
    from visualizer.src.renderers.pygfx.picking import PygfxPickingMixin

    class Renderer(PygfxPickingMixin, PygfxOverlayMixin):
        pass

    renderer = Renderer()
    renderer.visualizer = SimpleNamespace(
        app_state=create_initial_state(
            viewport_hud_enabled=False,
            viewport_hud_mode="detailed",
        )
    )
    renderer._tooltip_label = None
    renderer._infer_pick_metadata = lambda _name: (_ for _ in ()).throw(
        AssertionError("HUD-off must skip tooltip metadata work")
    )

    renderer._on_pointer_move(SimpleNamespace(target=object()))


def test_hud_toggle_changes_only_master_visibility():
    from visualizer.src.controllers.ui_controller import UIController
    from visualizer.src.renderers.protocol import RendererCapabilities

    class Renderer:
        capabilities = RendererCapabilities(viewport_hud=True)

        def __init__(self):
            self.refresh_calls = 0

        def refresh_viewport_hud(self):
            self.refresh_calls += 1

    class Visualizer:
        app_state = create_initial_state(viewport_hud_mode="detailed")
        renderer = Renderer()
        ui_manager = None

        def set_state(self, **changes):
            self.app_state = update_state(self.app_state, **changes)

    controller = UIController.__new__(UIController)
    controller.visualizer = Visualizer()

    controller.toggle_viewport_hud()
    assert controller.visualizer.app_state.viewport_hud_enabled is False
    assert controller.visualizer.app_state.viewport_hud_mode == "detailed"

    controller.toggle_viewport_hud()
    assert controller.visualizer.app_state.viewport_hud_enabled is True
    assert controller.visualizer.app_state.viewport_hud_mode == "detailed"
    assert controller.visualizer.renderer.refresh_calls == 2


def test_hud_toggle_is_noop_without_renderer_capability():
    from visualizer.src.controllers.ui_controller import UIController
    from visualizer.src.renderers.protocol import RendererCapabilities

    class Visualizer:
        app_state = create_initial_state()
        renderer = SimpleNamespace(capabilities=RendererCapabilities())
        ui_manager = None

        def set_state(self, **changes):
            raise AssertionError(f"unsupported HUD must not mutate state: {changes}")

    controller = UIController.__new__(UIController)
    controller.visualizer = Visualizer()

    controller.toggle_viewport_hud()

    assert controller.visualizer.app_state.viewport_hud_enabled is True


def test_filter_overlay_uses_compact_and_detailed_policy_without_widget_state():
    from visualizer.src.renderers.pygfx.overlays import PygfxOverlayMixin

    state = create_initial_state(
        mpc_allowed_orders=frozenset({0, 1}),
        viewport_hud_mode="compact",
    )
    renderer = PygfxOverlayMixin()
    renderer.visualizer = SimpleNamespace(app_state=state, mpc_allowed_materials=None)
    overlays = []
    renderer._set_hud_overlay = lambda overlay_id, **kwargs: overlays.append((overlay_id, kwargs))
    renderer._clear_hud_overlay = lambda _overlay_id: None

    renderer._update_filter_hud_overlay()

    assert overlays[-1][0] == "path_filters"
    assert overlays[-1][1]["role"] == "filter_chips"
    assert "1 filter" in overlays[-1][1]["html"]

    renderer.visualizer.app_state = AppState.from_dict(
        {
            **state.to_dict(),
            "viewport_hud_mode": "detailed",
        }
    )
    renderer._update_filter_hud_overlay()

    assert overlays[-1][1]["role"] == "filters"
    assert "Orders: 0–1" in overlays[-1][1]["html"]
