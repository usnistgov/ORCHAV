"""Renderer-side facade tests for semantic target-gizmo authoring."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from visualizer.src.renderers.pygfx.authoring import PygfxAuthoringRuntime


class _EventBackend:
    def __init__(self) -> None:
        self.logical_size = (640.0, 480.0)
        self.registrations = []
        self.removals = []

    def add_event_handler(self, callback, *event_types) -> None:
        self.registrations.append((callback, event_types))

    def remove_event_handler(self, callback, *event_types) -> None:
        self.removals.append((callback, event_types))


class _Owner:
    def __init__(self) -> None:
        self._renderer = _EventBackend()
        self._gfx = SimpleNamespace(Viewport=lambda renderer: SimpleNamespace(renderer=renderer))
        self._camera = None
        self._reverse_objects = {}
        self._controller = None
        self.gizmo_mode = None
        scale_children = [SimpleNamespace(visible=True) for _ in range(3)]
        rotate_children = [SimpleNamespace(visible=True) for _ in range(3)]
        arc_children = [SimpleNamespace(visible=True) for _ in range(3)]
        center_sphere = SimpleNamespace(visible=True)

        def _update_gizmo(_event) -> None:
            for child in scale_children + rotate_children + arc_children:
                child.visible = True
            center_sphere.visible = True

        self._transform_gizmo = SimpleNamespace(
            visible=True,
            children=(),
            _ref=None,
            _scale_children=scale_children,
            _rotate_children=rotate_children,
            _arc_children=arc_children,
            _center_sphere=center_sphere,
            toggle_mode=lambda mode: setattr(self, "gizmo_mode", mode),
            update_gizmo=_update_gizmo,
            set_object=lambda value: None,
        )
        self._pick_metadata = {
            "authoring:target": {"authoring_actor_pose": tuple(map(tuple, np.eye(4)))},
            "authoring:derived": {
                "authoring_actor_pose": tuple(map(tuple, np.eye(4))),
                "authoring_rotation_enabled": False,
            },
        }
        self._transform_session_callback = None
        self._transform_gizmo_target_name = None
        self._transform_gizmo_target_kind = None
        self._transform_gizmo_target_index = None
        self._transform_gizmo_control_object = None
        self.selected = None
        self.select_result = True
        self.proxy_removals = 0
        self.synced = None

    def _ensure_transform_gizmo(self, *, register_before_render=True):
        assert register_before_render is False
        return self._transform_gizmo

    def _select_transform_gizmo_target(
        self,
        object_id,
        kind,
        index,
        *,
        semantic_transform=None,
    ):
        self.selected = (object_id, kind, index, semantic_transform)
        if self.select_result:
            self._transform_gizmo_target_name = object_id
            self._transform_gizmo_target_kind = kind
            self._transform_gizmo_target_index = index
        return self.select_result

    def _parse_transform_target_name(self, object_id):
        if object_id == "node:tx_1::marker":
            return "tx", 1
        if object_id == "target:walker::mesh":
            return "target", 0
        return None

    def _remove_transform_gizmo_proxy(self):
        self.proxy_removals += 1

    def sync_active_transform_target_pose(self, object_id, transform):
        self.synced = (object_id, transform)
        return True


def test_authoring_runtime_uses_semantic_proxy_and_hides_scale_handles() -> None:
    owner = _Owner()
    runtime = PygfxAuthoringRuntime(owner)

    gizmo = runtime.ensure_gizmo()
    assert gizmo is owner._transform_gizmo
    assert owner.gizmo_mode == "world"
    assert all(not child.visible for child in gizmo._scale_children)
    assert gizmo._center_sphere.visible is False

    runtime.update_before_render(SimpleNamespace(type="before_render", target=None))
    assert all(not child.visible for child in gizmo._scale_children)
    assert gizmo._center_sphere.visible is False

    def callback(_event):
        return None

    assert runtime.attach_gizmo("authoring:target", callback) is True
    assert owner.selected[:3] == ("authoring:target", "authoring", 0)
    np.testing.assert_allclose(owner.selected[3], np.eye(4))
    assert owner._transform_session_callback is callback


def test_authoring_runtime_syncs_and_removes_semantic_proxy() -> None:
    owner = _Owner()
    runtime = PygfxAuthoringRuntime(owner)
    transform = np.eye(4)
    transform[:3, 3] = (1.0, 2.0, 3.0)

    assert runtime.sync_gizmo_pose("authoring:target", transform) is True
    assert owner.synced[0] == "authoring:target"
    np.testing.assert_allclose(owner.synced[1], transform)

    runtime.hide_gizmo()
    assert owner.proxy_removals == 1
    assert owner._transform_session_callback is None


def test_hiding_gizmo_resets_native_drag_highlight_and_pointer_capture() -> None:
    owner = _Owner()
    runtime = PygfxAuthoringRuntime(owner)
    released = []
    highlighted = []
    owner._transform_gizmo._ref = {"kind": "translate"}
    owner._transform_gizmo._orchav_pointer_id = 7
    owner._transform_gizmo.release_pointer_capture = released.append
    owner._transform_gizmo._highlight = lambda: highlighted.append(True)

    runtime.hide_gizmo()

    assert owner._transform_gizmo._ref is None
    assert owner._transform_gizmo._orchav_pointer_id is None
    assert released == [7]
    assert highlighted == [True]


def test_authoring_runtime_hides_rotation_for_derived_orientation() -> None:
    owner = _Owner()
    runtime = PygfxAuthoringRuntime(owner)

    assert runtime.attach_gizmo("authoring:derived", lambda _event: None) is True
    gizmo = owner._transform_gizmo
    assert all(not child.visible for child in gizmo._rotate_children)
    assert all(not child.visible for child in gizmo._arc_children)

    runtime.update_before_render(SimpleNamespace(type="before_render", target=None))
    assert all(not child.visible for child in gizmo._rotate_children)
    assert all(not child.visible for child in gizmo._arc_children)


def test_pose_sync_refreshes_rotation_handles_after_orientation_mode_changes() -> None:
    owner = _Owner()
    runtime = PygfxAuthoringRuntime(owner)
    object_id = "authoring:target"
    transform = np.eye(4)

    assert runtime.attach_gizmo(object_id, lambda _event: None) is True
    assert all(child.visible for child in owner._transform_gizmo._rotate_children)

    owner._pick_metadata[object_id]["authoring_rotation_enabled"] = False
    assert runtime.sync_gizmo_pose(object_id, transform) is True
    assert all(not child.visible for child in owner._transform_gizmo._rotate_children)
    assert all(not child.visible for child in owner._transform_gizmo._arc_children)

    owner._pick_metadata[object_id]["authoring_rotation_enabled"] = True
    assert runtime.sync_gizmo_pose(object_id, transform) is True
    assert all(child.visible for child in owner._transform_gizmo._rotate_children)
    assert all(child.visible for child in owner._transform_gizmo._arc_children)


def test_authoring_runtime_drops_callback_when_gizmo_attachment_fails() -> None:
    owner = _Owner()
    owner.select_result = False
    runtime = PygfxAuthoringRuntime(owner)

    assert runtime.attach_gizmo("authoring:target", lambda _event: None) is False
    assert owner._transform_session_callback is None


def test_runtime_live_tx_preview_exposes_translation_only() -> None:
    owner = _Owner()
    runtime = PygfxAuthoringRuntime(owner)

    def callback(_event):
        return None

    assert runtime.attach_live_preview_gizmo("node:tx_1::marker", callback) is True

    assert owner.selected[:3] == ("node:tx_1::marker", "tx", 1)
    assert owner._transform_session_callback is callback
    gizmo = owner._transform_gizmo
    assert all(not child.visible for child in gizmo._scale_children)
    assert gizmo._center_sphere.visible is False
    assert all(not child.visible for child in gizmo._rotate_children)
    assert all(not child.visible for child in gizmo._arc_children)

    runtime.update_before_render(
        SimpleNamespace(type="before_render", target=None),
        authoring=False,
    )
    assert all(not child.visible for child in gizmo._scale_children)
    assert all(not child.visible for child in gizmo._rotate_children)


def test_runtime_live_target_preview_keeps_supported_rotation_handles() -> None:
    owner = _Owner()
    runtime = PygfxAuthoringRuntime(owner)

    assert runtime.attach_live_preview_gizmo("target:walker::mesh", lambda _event: None)

    gizmo = owner._transform_gizmo
    assert all(not child.visible for child in gizmo._scale_children)
    assert all(child.visible for child in gizmo._rotate_children)
    assert all(child.visible for child in gizmo._arc_children)

    runtime.update_before_render(
        SimpleNamespace(type="before_render", target=None),
        authoring=False,
    )
    assert all(not child.visible for child in gizmo._scale_children)
    assert all(child.visible for child in gizmo._rotate_children)


def test_authoring_runtime_unregisters_handlers_when_closed() -> None:
    owner = _Owner()
    runtime = PygfxAuthoringRuntime(owner)

    def callback(_event):
        return None

    runtime.add_event_handler(callback, "pointer_down", "before_render")
    assert owner._renderer.registrations == [(callback, ("pointer_down", "before_render"))]

    runtime.close()
    runtime.close()

    assert owner._renderer.removals == [(callback, ("pointer_down", "before_render"))]
    assert owner.proxy_removals == 1
    with np.testing.assert_raises_regex(RuntimeError, "runtime is closed"):
        runtime.add_event_handler(callback, "pointer_down")
