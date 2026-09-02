"""Failure/retry coverage for Open3D-owned coverage synchronization."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("open3d")

from visualizer.src.renderers.open3d.surface_overlays import Open3DSurfaceOverlayMixin


def _coverage_packet(
    *,
    isolines: bool = False,
    opacity: float = 0.65,
    signature: str = "coverage-revision-11",
    isoline_signature: str | None = None,
):
    return SimpleNamespace(
        show_coverage=True,
        coverage_vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        coverage_triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        coverage_colors=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        coverage_isoline_points=(
            np.asarray([[0.0, 0.5, 0.01], [1.0, 0.5, 0.01]], dtype=np.float32) if isolines else None
        ),
        coverage_isoline_lines=(np.asarray([[0, 1]], dtype=np.int32) if isolines else None),
        coverage_isoline_colors=None,
        coverage_signature=signature,
        coverage_metadata={"isoline_signature": isoline_signature or f"{signature}-isolines"},
        coverage_opacity=opacity,
    )


class _Open3DCoverageHarness(Open3DSurfaceOverlayMixin):
    COVERAGE_MESH_NAME = "coverage_mesh"
    COVERAGE_ISOLINES_NAME = "coverage_isolines"

    def __init__(self, failure: str | None) -> None:
        self.failure = failure
        self.failed_once = False
        self.native_names: set[str] = set()
        self.upload_calls: list[str] = []
        self.opacity_calls: list[float] = []
        self.visualizer = SimpleNamespace(coverage_mesh=None)
        self._last_coverage_signature = None
        self._applied_coverage_state = None
        self._line_width = 2.0

    def _add_or_update_geometry(self, name, _geometry, _material) -> bool:
        self.upload_calls.append(name)
        should_fail = self.failure == name and not self.failed_once
        missing_native = self.failure == "native_missing" and not self.failed_once
        if should_fail or missing_native:
            self.failed_once = True
            return not should_fail
        self.native_names.add(name)
        return True

    def set_coverage_transparency(self, alpha: float) -> bool:
        self.opacity_calls.append(float(alpha))
        if self.failure == "opacity" and not self.failed_once:
            self.failed_once = True
            return False
        return True

    def has_named_geometry(self, name: str) -> bool:
        return name in self.native_names

    def _remove_geometry(self, name: str) -> bool:
        self.native_names.discard(name)
        return True


@pytest.mark.parametrize(
    ("failure", "isolines"),
    [
        ("coverage_mesh", False),
        ("coverage_isolines", True),
        ("native_missing", False),
    ],
)
def test_open3d_coverage_failure_is_not_applied_and_same_packet_retries(
    failure: str,
    isolines: bool,
) -> None:
    renderer = _Open3DCoverageHarness(failure)
    packet = _coverage_packet(isolines=isolines)

    assert renderer._apply_coverage_data(packet) is False
    assert renderer._applied_coverage_state is None
    assert renderer._last_coverage_signature is None

    assert renderer._apply_coverage_data_diff(packet, packet) is True
    assert renderer._applied_coverage_state.signature == packet.coverage_signature
    assert renderer._last_coverage_signature == packet.coverage_signature

    calls_after_success = len(renderer.upload_calls)
    assert renderer._apply_coverage_data_diff(packet, packet) is True
    assert len(renderer.upload_calls) == calls_after_success


def test_open3d_coverage_opacity_failure_retries_without_advancing_applied_state() -> None:
    renderer = _Open3DCoverageHarness(None)
    first = _coverage_packet(opacity=0.8)
    changed = _coverage_packet(opacity=0.3)
    assert renderer._apply_coverage_data(first) is True

    renderer.failure = "opacity"
    assert renderer._apply_coverage_data_diff(first, changed) is False
    assert renderer._applied_coverage_state.opacity == pytest.approx(0.8)

    assert renderer._apply_coverage_data_diff(changed, changed) is True
    assert renderer._applied_coverage_state.opacity == pytest.approx(0.3)
    assert renderer.opacity_calls == [pytest.approx(0.3), pytest.approx(0.3)]


def test_open3d_failed_new_revision_preserves_prior_applied_signature() -> None:
    renderer = _Open3DCoverageHarness(None)
    first = _coverage_packet(signature="coverage-revision-1")
    changed = _coverage_packet(signature="coverage-revision-2")
    assert renderer._apply_coverage_data(first) is True

    renderer.failure = renderer.COVERAGE_MESH_NAME
    assert renderer._apply_coverage_data_diff(first, changed) is False
    assert renderer._applied_coverage_state.signature == "coverage-revision-1"

    assert renderer._apply_coverage_data_diff(changed, changed) is True
    assert renderer._applied_coverage_state.signature == "coverage-revision-2"


def test_open3d_isoline_revision_does_not_reupload_coverage_mesh() -> None:
    renderer = _Open3DCoverageHarness(None)
    first = _coverage_packet(isolines=True, isoline_signature="isolines-1")
    changed = _coverage_packet(isolines=True, isoline_signature="isolines-2")

    assert renderer._apply_coverage_data(first) is True
    mesh_uploads = renderer.upload_calls.count(renderer.COVERAGE_MESH_NAME)

    assert renderer._apply_coverage_data_diff(first, changed) is True
    assert renderer.upload_calls.count(renderer.COVERAGE_MESH_NAME) == mesh_uploads
    assert renderer._applied_coverage_state.isoline_signature == "isolines-2"
