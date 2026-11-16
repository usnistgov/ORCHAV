"""Parity and fail-safe regressions for target-manager identity indexing."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import generator.io.frames.builder as frame_builder


class _Vector3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class _IdentityTarget:
    __slots__ = ("position", "orientation")

    def __init__(self, x: float) -> None:
        self.position = _Vector3(x, x + 1.0, x + 2.0)
        self.orientation = _Vector3(x * 0.01, x * 0.02, x * 0.03)


class _EqualTarget(_IdentityTarget):
    __slots__ = ("token", "equality_calls")

    def __init__(self, x: float, token: str) -> None:
        super().__init__(x)
        self.token = token
        self.equality_calls = 0

    def __eq__(self, other: object) -> bool:
        self.equality_calls += 1
        return isinstance(other, _EqualTarget) and self.token == other.token

    __hash__ = None


class _ExplodingStringTarget(_IdentityTarget):
    def __str__(self) -> str:
        raise AssertionError("disabled debug logging must not format target objects")


class _ExplodingNameConfig:
    @property
    def name(self) -> str:
        raise AssertionError("disabled debug logging must not build manager-name lists")


class _RaisingTargetObjectManager:
    @property
    def target_object(self) -> Any:
        raise RuntimeError("manager target unavailable")


class _Manager:
    __slots__ = (
        "_target_object",
        "target_object_reads",
        "current_mesh_idx",
        "meshes",
        "relative_mesh_directory",
        "config",
    )

    def __init__(self, target_object: Any, name: str) -> None:
        self._target_object = target_object
        self.target_object_reads = 0
        self.current_mesh_idx = 0
        self.meshes = [f"/scenario/meshes/{name}.ply"]
        self.relative_mesh_directory = "meshes"
        self.config = SimpleNamespace(
            name=name,
            initial_position=[-1.0, -1.0, -1.0],
            scale=1.25,
            material_type="metal",
            use_ply_position=False,
            mobility=None,
        )

    @property
    def target_object(self) -> Any:
        self.target_object_reads += 1
        return self._target_object


def _legacy_reference(
    targets: list[Any], managers: list[_Manager]
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    positions = np.asarray(
        [
            [float(target.position.x), float(target.position.y), float(target.position.z)]
            for target in targets
        ],
        dtype=np.float64,
    ).reshape((-1, 3))
    metadata: list[dict[str, Any]] = []
    comparisons = 0
    for target in targets:
        manager = None
        for candidate in managers:
            comparisons += 1
            if candidate.target_object == target:
                manager = candidate
                break
        if manager is None:
            continue
        metadata.append(
            {
                "name": manager.config.name,
                "mesh_file": f"{manager.config.name}.ply",
                "mesh_directory": "meshes",
                "mesh_index": 0,
                "scale": 1.25,
                "orientation": [
                    float(target.orientation.x),
                    float(target.orientation.y),
                    float(target.orientation.z),
                ],
                "material_type": "metal",
                "material_id": f"mat-itu_metal_{manager.config.name}",
                "use_ply_position": False,
                "current_position": [
                    float(target.position.x),
                    float(target.position.y),
                    float(target.position.z),
                ],
                "mobility_type": "Stationary",
            }
        )
    return positions, metadata, comparisons


@pytest.mark.parametrize(
    "target_count",
    [
        frame_builder._TARGET_MANAGER_INDEX_MIN_TARGETS,
        frame_builder._TARGET_MANAGER_INDEX_MIN_TARGETS * 2,
    ],
)
def test_identity_index_is_linear_and_exactly_matches_legacy(
    target_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = [_IdentityTarget(float(index)) for index in range(target_count)]
    managers = [_Manager(target, f"target-{index:02d}") for index, target in enumerate(targets)]
    legacy_positions, legacy_metadata, comparisons = _legacy_reference(targets, managers)
    for manager in managers:
        manager.target_object_reads = 0

    def unexpected_legacy_lookup(*_args: object) -> None:
        raise AssertionError("identity-safe targets must not use the legacy scan")

    monkeypatch.setattr(
        frame_builder,
        "_legacy_target_manager_lookup",
        unexpected_legacy_lookup,
    )
    positions, metadata = frame_builder._process_target_metadata(targets, managers, 7)

    assert np.array_equal(positions, legacy_positions)
    assert metadata == legacy_metadata
    assert comparisons == target_count * (target_count + 1) // 2
    assert sum(manager.target_object_reads for manager in managers) == target_count
    assert [row["name"] for row in metadata] == [
        f"target-{index:02d}" for index in range(target_count)
    ]


@pytest.mark.parametrize(
    "target_count",
    [1, frame_builder._TARGET_MANAGER_INDEX_MIN_TARGETS - 1],
)
def test_small_identity_safe_collections_keep_exact_legacy_lookup(
    target_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = [_IdentityTarget(float(index)) for index in range(target_count)]
    managers = [_Manager(target, f"target-{index:02d}") for index, target in enumerate(targets)]
    calls = 0
    original = frame_builder._legacy_target_manager_lookup

    def unexpected_index(*_args: object) -> None:
        raise AssertionError("small target collections must not pay the identity-index cost")

    def counted_lookup(target_obj: Any, target_managers: list[Any]) -> Any | None:
        nonlocal calls
        calls += 1
        return original(target_obj, target_managers)

    monkeypatch.setattr(frame_builder, "_identity_target_manager_index", unexpected_index)
    monkeypatch.setattr(frame_builder, "_legacy_target_manager_lookup", counted_lookup)

    positions, metadata = frame_builder._process_target_metadata(targets, managers, 3)

    assert positions.shape == (target_count, 3)
    assert calls == target_count
    assert [row["name"] for row in metadata] == [
        f"target-{index:02d}" for index in range(target_count)
    ]


def test_duplicate_manager_identity_falls_back_to_historical_first_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_count = frame_builder._TARGET_MANAGER_INDEX_MIN_TARGETS
    targets = [_IdentityTarget(float(index)) for index in range(target_count)]
    managers = [
        _Manager(targets[0], "first"),
        _Manager(targets[0], "second"),
        *[
            _Manager(target, f"target-{index:02d}")
            for index, target in enumerate(targets[1:], start=1)
        ],
    ]
    calls = 0
    original = frame_builder._legacy_target_manager_lookup

    def counted_lookup(target_obj: Any, target_managers: list[Any]) -> Any | None:
        nonlocal calls
        calls += 1
        return original(target_obj, target_managers)

    monkeypatch.setattr(frame_builder, "_legacy_target_manager_lookup", counted_lookup)
    _positions, metadata = frame_builder._process_target_metadata(targets, managers, 0)

    assert calls == target_count
    assert metadata[0]["name"] == "first"


def test_duplicate_target_identity_falls_back_and_preserves_duplicate_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_count = frame_builder._TARGET_MANAGER_INDEX_MIN_TARGETS
    unique_targets = [_IdentityTarget(float(index)) for index in range(target_count - 1)]
    targets = [
        unique_targets[0],
        *unique_targets,
    ]
    managers = [
        _Manager(target, "repeated" if index == 0 else f"target-{index:02d}")
        for index, target in enumerate(unique_targets)
    ]
    calls = 0
    original = frame_builder._legacy_target_manager_lookup

    def counted_lookup(target_obj: Any, target_managers: list[Any]) -> Any | None:
        nonlocal calls
        calls += 1
        return original(target_obj, target_managers)

    monkeypatch.setattr(frame_builder, "_legacy_target_manager_lookup", counted_lookup)
    positions, metadata = frame_builder._process_target_metadata(targets, managers, 0)

    assert calls == target_count
    assert positions.shape == (target_count, 3)
    assert [row["name"] for row in metadata[:2]] == ["repeated", "repeated"]


def test_missing_identity_uses_non_identity_equality_match() -> None:
    target = _EqualTarget(3.0, "same-semantic-target")
    equal_but_distinct = _EqualTarget(99.0, "same-semantic-target")
    manager = _Manager(equal_but_distinct, "semantic-match")

    _positions, metadata = frame_builder._process_target_metadata([target], [manager], 0)

    assert metadata[0]["name"] == "semantic-match"
    assert equal_but_distinct.equality_calls == 1


def test_custom_equality_preserves_earlier_equal_match_over_identity_match() -> None:
    target_count = frame_builder._TARGET_MANAGER_INDEX_MIN_TARGETS
    target = _EqualTarget(4.0, "shared")
    earlier_equal = _EqualTarget(40.0, "shared")
    identity_targets = [_IdentityTarget(float(index)) for index in range(target_count - 1)]
    targets = [*identity_targets, target]
    managers = [
        *[
            _Manager(identity_target, f"identity-{index:02d}")
            for index, identity_target in enumerate(identity_targets)
        ],
        _Manager(earlier_equal, "earlier-equal"),
        _Manager(target, "later-identity"),
    ]

    _positions, metadata = frame_builder._process_target_metadata(targets, managers, 0)

    assert metadata[-1]["name"] == "earlier-equal"
    assert earlier_equal.equality_calls == 1


def test_missing_manager_preserves_position_and_omits_metadata() -> None:
    target = _IdentityTarget(5.0)
    unrelated = _IdentityTarget(6.0)

    positions, metadata = frame_builder._process_target_metadata(
        [target], [_Manager(unrelated, "unrelated")], 0
    )

    assert positions.tolist() == [[5.0, 6.0, 7.0]]
    assert metadata == []


def test_manager_target_access_exception_is_not_swallowed_above_threshold() -> None:
    target_count = frame_builder._TARGET_MANAGER_INDEX_MIN_TARGETS
    targets = [_IdentityTarget(float(index)) for index in range(target_count)]
    managers: list[Any] = [
        _RaisingTargetObjectManager(),
        *[_Manager(target, f"target-{index:02d}") for index, target in enumerate(targets)],
    ]

    with pytest.raises(RuntimeError, match="manager target unavailable"):
        frame_builder._process_target_metadata(targets, managers, 0)


def test_disabled_debug_logging_does_not_format_target_objects(
    caplog: pytest.LogCaptureFixture,
) -> None:
    target = _ExplodingStringTarget(1.0)
    manager = _Manager(target, "quiet")

    with caplog.at_level(logging.WARNING, logger=frame_builder.logger.name):
        positions, metadata = frame_builder._process_target_metadata([target], [manager], 0)

    assert positions.tolist() == [[1.0, 2.0, 3.0]]
    assert metadata[0]["name"] == "quiet"


def test_disabled_debug_logging_does_not_build_manager_name_list(
    caplog: pytest.LogCaptureFixture,
) -> None:
    target = _IdentityTarget(1.0)
    manager = _Manager(_IdentityTarget(2.0), "unrelated")
    manager.config = _ExplodingNameConfig()

    with caplog.at_level(logging.ERROR, logger=frame_builder.logger.name):
        positions, metadata = frame_builder._process_target_metadata([target], [manager], 0)

    assert positions.tolist() == [[1.0, 2.0, 3.0]]
    assert metadata == []


def test_lazy_logging_preserves_representative_emitted_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    target = _IdentityTarget(1.0)
    manager = _Manager(target, "logged")

    with caplog.at_level(logging.DEBUG, logger=frame_builder.logger.name):
        positions, metadata = frame_builder._process_target_metadata([target], [manager], 7)

    messages = [
        record.getMessage() for record in caplog.records if record.name == frame_builder.logger.name
    ]
    assert f"[TARGET EXPORT] Frame 7: target_pos: {positions}" in messages
    assert f"[TARGET METADATA] Frame 7: Processing target object 0: {target}" in messages
    assert (
        "[TARGET METADATA] Frame 7 (display=8): Extracting mesh_file from "
        "target_manager: mesh_idx=0, mesh_file=logged.ply" in messages
    )
    assert f"[TARGET METADATA] Frame 7: Full metadata: {metadata[0]}" in messages
