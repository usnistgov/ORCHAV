"""Architecture checks for canonical frame production and consumption."""

from __future__ import annotations

import ast
from pathlib import Path

from examples.external_raytracer_import.fake_raytracer import trace
from examples.external_raytracer_import.import_to_orchav import to_standard_frame
from shared.frames import StandardMPCFrame
from shared.frames.provider_base import DataProvider
from visualizer.src.io.packed_frame_payload import (
    try_load_packed_visual_frame,
    visual_frame_read_request,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

_LOW_LEVEL_WRITER_ALLOWLIST = {
    Path("shared/frames/packed_hdf5_writer.py"),
    Path("shared/frames/frame_set_writer.py"),
    Path("benchmarks/bench_hdf5_v2.py"),
    Path("benchmarks/bench_hdf5_extensions.py"),
}

_RETIRED_VISUAL_FRAME_FIELDS = {
    "all_padded_vertices",
    "all_padded_interactions",
    "all_pair_delays_ns",
    "all_pair_path_loss_db",
    "all_material_names",
    "all_path_lengths",
}


def _production_python_files() -> tuple[Path, ...]:
    """Return source files that can participate in frame publication."""

    roots = ("shared", "generator", "visualizer", "scripts", "benchmarks", "examples")
    return tuple(
        path
        for root in roots
        for path in (REPOSITORY_ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_low_level_mpc_writer_has_an_explicit_usage_allowlist() -> None:
    """Prevent complete producers from bypassing FrameSetWriter publication."""

    users = {
        path.relative_to(REPOSITORY_ROOT)
        for path in _production_python_files()
        if "PackedMPCChunkWriter" in path.read_text(encoding="utf-8")
    }
    expected = {path for path in _LOW_LEVEL_WRITER_ALLOWLIST if (REPOSITORY_ROOT / path).is_file()}
    assert users == expected


def test_complete_frame_set_writer_has_one_definition() -> None:
    """Keep one authoritative complete-frame publication lifecycle."""

    definitions: set[Path] = set()
    for path in _production_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.ClassDef) and node.name == "FrameSetWriter"
            for node in ast.walk(tree)
        ):
            definitions.add(path.relative_to(REPOSITORY_ROOT))
    assert definitions == {Path("shared/frames/frame_set_writer.py")}


def test_visualizer_source_has_no_retired_padded_frame_fields() -> None:
    """Require visual construction to enter through compact projections."""

    occurrences: dict[str, list[str]] = {}
    for path in (REPOSITORY_ROOT / "visualizer" / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        matched = sorted(field for field in _RETIRED_VISUAL_FRAME_FIELDS if field in text)
        if matched:
            occurrences[str(path.relative_to(REPOSITORY_ROOT))] = matched
    assert occurrences == {}


def test_external_adapter_uses_the_complete_writer_boundary() -> None:
    """Keep the interoperability proof independent of generator storage code."""

    adapter = REPOSITORY_ROOT / "examples/external_raytracer_import/import_to_orchav.py"
    source = adapter.read_text(encoding="utf-8")
    assert "FrameSetWriter.create_new" in source
    assert "PackedMPCChunkWriter" not in source
    assert "generator.io" not in source


class _CompleteFrameOnlyProvider(DataProvider):
    """Nonselective provider exercising DataProvider's projection fallback."""

    def __init__(self, frame: StandardMPCFrame) -> None:
        self.frame = frame
        self.complete_loads = 0

    def list_frames(self) -> list[int]:
        return [self.frame.frame_index]

    def has_frame(self, step: int) -> bool:
        return step == self.frame.frame_index

    def load_frame(self, step: int) -> StandardMPCFrame:
        if not self.has_frame(step):
            raise KeyError(step)
        self.complete_loads += 1
        return self.frame


def test_nonselective_provider_reaches_the_visual_payload_seam() -> None:
    """Prove future providers need no selective-I/O implementation for playback."""

    frame = to_standard_frame(trace()[0])
    provider = _CompleteFrameOnlyProvider(frame)
    payload = try_load_packed_visual_frame(
        provider,
        frame.frame_index,
        request=visual_frame_read_request(),
    )

    assert payload is not None
    assert provider.complete_loads == 1
    assert payload["canonical_data"].path_orders.tolist() == [0, 1, 1]
    assert payload["num_tx"] == 2
    assert payload["num_rx"] == 2
