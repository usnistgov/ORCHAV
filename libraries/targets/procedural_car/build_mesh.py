"""Build ORCHAV's original car-shaped prism from an explicit side profile."""

from __future__ import annotations

import argparse
from pathlib import Path

# Metres, local X/Z. This authored polygon uses no imported vehicle mesh.
PROFILE_XZ = (
    (-2.0, 0.0),
    (2.0, 0.0),
    (2.0, 0.55),
    (0.9, 0.55),
    (0.55, 1.4),
    (-0.7, 1.4),
    (-1.15, 0.55),
    (-2.0, 0.55),
)
HALF_WIDTH_M = 0.85


def mesh_text() -> str:
    """Return a deterministic, closed triangular prism in ASCII PLY format."""
    count = len(PROFILE_XZ)
    vertices = [(x, y, z) for y in (-HALF_WIDTH_M, HALF_WIDTH_M) for x, z in PROFILE_XZ]
    faces = []
    # The polygon is star-shaped at vertex zero; this fan remains inside it.
    for index in range(1, count - 1):
        faces.extend(((0, index, index + 1), (count, count + index + 1, count + index)))
    for index in range(count):
        following = (index + 1) % count
        faces.extend(
            ((index, count + following, following), (index, count + index, count + following))
        )
    lines = [
        "ply",
        "format ascii 1.0",
        "comment Original ORCHAV procedural car-shaped prism; see build_mesh.py",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(" ".join(f"{value:.8g}" for value in vertex) for vertex in vertices)
    lines.extend("3 " + " ".join(str(index) for index in face) for face in faces)
    return "\n".join(lines) + "\n"


def main() -> int:
    """Regenerate or verify the adjacent authored mesh, with no network input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify without writing")
    args = parser.parse_args()
    destination = Path(__file__).with_name("car.ply")
    expected = mesh_text().encode("ascii")
    if args.check:
        if not destination.is_file() or destination.read_bytes() != expected:
            parser.error("car.ply is missing or differs from the original procedural recipe")
    else:
        destination.write_bytes(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
