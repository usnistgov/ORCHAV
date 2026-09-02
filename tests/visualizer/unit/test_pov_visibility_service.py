from __future__ import annotations

from types import SimpleNamespace

from visualizer.src.services.pov_visibility_service import PovVisibilityService, is_hidden_for_pov


class _Renderer:
    def __init__(self) -> None:
        self.updated = 0
        self.polled = 0
        self.direct_visibility_calls: list[tuple[str, bool]] = []

    def set_visible(self, object_id: str, visible: bool) -> bool:
        self.direct_visibility_calls.append((object_id, bool(visible)))
        return True

    def poll_events(self) -> None:
        self.polled += 1

    def update_renderer(self) -> None:
        self.updated += 1


class _SemanticEntityOwner:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, int], ...]] = []

    def sync_pov_entity_visibility(self, entity_refs) -> bool:
        self.calls.append(tuple(entity_refs))
        return True


def _make_visualizer(*, hidden=None):
    state = SimpleNamespace(camera_mode="pov", pov_hidden_node=hidden)
    renderer = _Renderer()
    owner = _SemanticEntityOwner()

    def set_state(**kwargs) -> None:
        for key, value in kwargs.items():
            setattr(state, key, value)

    return SimpleNamespace(
        vis=object(),
        renderer=renderer,
        node_service=owner,
        app_state=state,
        set_state=set_state,
    )


def test_is_hidden_for_pov_matches_active_entity_only() -> None:
    state = SimpleNamespace(camera_mode="pov", pov_hidden_node=("target", 1))

    assert is_hidden_for_pov(state, "target", 1) is True
    assert is_hidden_for_pov(state, "target", 0) is False
    assert is_hidden_for_pov(state, "tx", 1) is False

    state.camera_mode = "follow"
    assert is_hidden_for_pov(state, "target", 1) is False


def test_hide_changes_only_pov_state_and_requests_complete_semantic_sync() -> None:
    viz = _make_visualizer()
    service = PovVisibilityService(viz)

    service.hide({"type": "rx", "index": 2})

    assert viz.app_state.pov_hidden_node == ("rx", 2)
    assert viz.node_service.calls == [(("rx", 2),)]
    assert viz.renderer.direct_visibility_calls == []


def test_switching_pov_entity_republishes_previous_and_current_snapshots() -> None:
    viz = _make_visualizer(hidden=("target", 0))
    service = PovVisibilityService(viz)

    service.hide({"type": "tx", "index": 1})

    assert viz.app_state.pov_hidden_node == ("tx", 1)
    assert viz.node_service.calls == [
        (("target", 0), ("tx", 1)),
    ]
    assert viz.renderer.direct_visibility_calls == []


def test_restore_clears_pov_state_before_semantic_resynchronization() -> None:
    viz = _make_visualizer(hidden=("target", 3))
    observed_state = []

    def sync(entity_refs) -> bool:
        observed_state.append(viz.app_state.pov_hidden_node)
        viz.node_service.calls.append(tuple(entity_refs))
        return True

    viz.node_service.sync_pov_entity_visibility = sync
    service = PovVisibilityService(viz)

    service.restore(update_renderer=True)

    assert observed_state == [None]
    assert viz.app_state.pov_hidden_node is None
    assert viz.node_service.calls == [(("target", 3),)]
    assert viz.renderer.direct_visibility_calls == []
    assert viz.renderer.polled == 1
    assert viz.renderer.updated == 1


def test_restore_can_defer_renderer_presentation() -> None:
    viz = _make_visualizer(hidden=("tx", 0))
    service = PovVisibilityService(viz)

    service.restore(update_renderer=False)

    assert viz.app_state.pov_hidden_node is None
    assert viz.node_service.calls == [(("tx", 0),)]
    assert viz.renderer.polled == 0
    assert viz.renderer.updated == 0


def test_failed_snapshot_sync_rolls_back_state_and_converges_on_identical_retry() -> None:
    viz = _make_visualizer(hidden=("tx", 0))
    service = PovVisibilityService(viz)
    outcomes = iter((False, True, True))
    observed_states: list[tuple[str, int] | None] = []

    def sync(entity_refs) -> bool:
        observed_states.append(viz.app_state.pov_hidden_node)
        viz.node_service.calls.append(tuple(entity_refs))
        return next(outcomes)

    viz.node_service.sync_pov_entity_visibility = sync

    assert service.set_hidden_entity(("target", 1)) is False
    assert viz.app_state.pov_hidden_node == ("tx", 0)
    assert observed_states == [("target", 1), ("tx", 0)]

    assert service.set_hidden_entity(("target", 1)) is True
    assert viz.app_state.pov_hidden_node == ("target", 1)
    assert observed_states == [("target", 1), ("tx", 0), ("target", 1)]
    assert viz.node_service.calls == [
        (("tx", 0), ("target", 1)),
        (("tx", 0), ("target", 1)),
        (("tx", 0), ("target", 1)),
    ]
