#!/usr/bin/env python3
"""Render preview assets for animated target mesh libraries."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_VIEW_AZIM_ELEV = {
    "front": (-90.0, 12.0),
    "side": (0.0, 12.0),
}


@dataclass(frozen=True)
class PreviewOutputs:
    """Paths written for one mesh library preview."""

    output_dir: Path
    contact_sheet: Path
    front_gif: Path


def check_dependencies() -> None:
    """Verify that rendering dependencies are installed."""
    missing = []
    for pkg in ("trimesh", "matplotlib", "PIL"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error(
            "Missing dependencies: %s\nInstall with the environment used for mesh tools.",
            " ".join(missing),
        )
        sys.exit(1)


def list_mesh_frames(mesh_dir: Path) -> list[Path]:
    """Return sorted PLY frames in a mesh library directory."""
    return sorted(path for path in mesh_dir.glob("*.ply") if path.is_file())


def find_mesh_libraries(root_dir: Path, min_ply_count: int = 2) -> list[Path]:
    """Return immediate child directories that contain multi-frame PLY libraries."""
    libraries = []
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        if len(list_mesh_frames(child)) >= min_ply_count:
            libraries.append(child)
    return libraries


def select_sample_indices(total_frames: int, sample_count: int) -> list[int]:
    """Select evenly spaced frame indices across a sequence."""
    if total_frames <= 0:
        return []
    if sample_count >= total_frames:
        return list(range(total_frames))
    indices = np.linspace(0, total_frames - 1, num=sample_count, dtype=int)
    return sorted({int(index) for index in indices})


def resolve_preview_outputs(mesh_dir: Path, output_subdir: str) -> PreviewOutputs:
    """Return output paths for one library preview bundle."""
    output_dir = mesh_dir / output_subdir
    return PreviewOutputs(
        output_dir=output_dir,
        contact_sheet=output_dir / "contact_sheet.png",
        front_gif=output_dir / "front.gif",
    )


def _load_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one mesh as vertex and face arrays."""
    import trimesh

    mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Unsupported mesh structure in {mesh_path}")
    return vertices, faces


def _compute_bounds(
    frame_paths: list[Path],
    sample_indices: list[int],
    padding_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute shared cubic bounds across sampled frames."""
    mins = []
    maxs = []
    for index in sample_indices:
        vertices, _faces = _load_mesh(frame_paths[index])
        mins.append(vertices.min(axis=0))
        maxs.append(vertices.max(axis=0))
    min_corner = np.min(np.stack(mins), axis=0)
    max_corner = np.max(np.stack(maxs), axis=0)
    center = 0.5 * (min_corner + max_corner)
    max_extent = float(np.max(max_corner - min_corner))
    half_extent = 0.5 * max_extent * (1.0 + padding_fraction)
    min_bound = center - half_extent
    max_bound = center + half_extent
    return min_bound, max_bound


def _render_mesh_image(
    vertices: np.ndarray,
    faces: np.ndarray,
    min_bound: np.ndarray,
    max_bound: np.ndarray,
    *,
    view_name: str,
    face_stride: int,
    image_size_px: int,
):
    """Render one mesh image using a fixed camera view."""
    import matplotlib

    matplotlib.use("Agg")
    import io

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image

    dpi = 100
    fig = plt.figure(figsize=(image_size_px / dpi, image_size_px / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    azim, elev = _VIEW_AZIM_ELEV[view_name]
    face_stride = max(1, face_stride)
    mesh_faces = faces[::face_stride]
    collection = Poly3DCollection(
        vertices[mesh_faces],
        facecolor=(0.72, 0.62, 0.52, 1.0),
        edgecolor="none",
    )
    ax.add_collection3d(collection)
    ax.set_xlim(min_bound[0], max_bound[0])
    ax.set_ylim(min_bound[1], max_bound[1])
    ax.set_zlim(min_bound[2], max_bound[2])
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=0)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _build_contact_sheet(
    mesh_dir: Path,
    frame_paths: list[Path],
    min_bound: np.ndarray,
    max_bound: np.ndarray,
    *,
    contact_samples: int,
    contact_face_stride: int,
    image_size_px: int,
):
    """Render a front/side contact sheet."""
    from PIL import Image, ImageDraw

    sample_indices = select_sample_indices(len(frame_paths), contact_samples)
    front_images = []
    side_images = []
    labels = []

    for index in sample_indices:
        vertices, faces = _load_mesh(frame_paths[index])
        front_images.append(
            _render_mesh_image(
                vertices,
                faces,
                min_bound,
                max_bound,
                view_name="front",
                face_stride=contact_face_stride,
                image_size_px=image_size_px,
            )
        )
        side_images.append(
            _render_mesh_image(
                vertices,
                faces,
                min_bound,
                max_bound,
                view_name="side",
                face_stride=contact_face_stride,
                image_size_px=image_size_px,
            )
        )
        labels.append(frame_paths[index].stem)

    cell_width, cell_height = front_images[0].size
    banner_height = 28
    label_height = 16
    width = cell_width * len(front_images)
    height = banner_height + (cell_height + label_height) * 2
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), f"{mesh_dir.name} | {len(frame_paths)} frames", fill="black")

    for column, image in enumerate(front_images):
        x = column * cell_width
        y = banner_height
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + cell_height), labels[column], fill="black")

    for column, image in enumerate(side_images):
        x = column * cell_width
        y = banner_height + cell_height + label_height
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + cell_height), labels[column], fill="black")

    return sheet


def _build_front_gif(
    frame_paths: list[Path],
    min_bound: np.ndarray,
    max_bound: np.ndarray,
    *,
    gif_frames: int,
    gif_face_stride: int,
    image_size_px: int,
):
    """Render a front-view GIF across evenly sampled frames."""
    from PIL import Image

    sample_indices = select_sample_indices(len(frame_paths), gif_frames)
    frames = []
    for index in sample_indices:
        vertices, faces = _load_mesh(frame_paths[index])
        rendered = _render_mesh_image(
            vertices,
            faces,
            min_bound,
            max_bound,
            view_name="front",
            face_stride=gif_face_stride,
            image_size_px=image_size_px,
        )
        frames.append(rendered.convert("P", palette=Image.ADAPTIVE))
    return frames, sample_indices


def render_library_preview(
    mesh_dir: Path,
    *,
    output_subdir: str = "preview",
    contact_samples: int = 6,
    gif_frames: int = 60,
    gif_duration_ms: int = 33,
    image_size_px: int = 240,
    contact_face_stride: int = 2,
    gif_face_stride: int = 6,
    padding_fraction: float = 0.08,
    skip_existing: bool = False,
) -> PreviewOutputs:
    """Render preview assets for one mesh library."""
    frame_paths = list_mesh_frames(mesh_dir)
    if len(frame_paths) < 2:
        raise ValueError(f"{mesh_dir} must contain at least 2 PLY files")

    outputs = resolve_preview_outputs(mesh_dir, output_subdir)
    if skip_existing and outputs.contact_sheet.exists() and outputs.front_gif.exists():
        logger.info("Skipping %s (preview already exists)", mesh_dir)
        return outputs

    outputs.output_dir.mkdir(parents=True, exist_ok=True)
    bounds_indices = select_sample_indices(len(frame_paths), min(12, len(frame_paths)))
    min_bound, max_bound = _compute_bounds(frame_paths, bounds_indices, padding_fraction)

    contact_sheet = _build_contact_sheet(
        mesh_dir,
        frame_paths,
        min_bound,
        max_bound,
        contact_samples=contact_samples,
        contact_face_stride=contact_face_stride,
        image_size_px=image_size_px,
    )
    contact_sheet.save(outputs.contact_sheet)

    gif_images, gif_indices = _build_front_gif(
        frame_paths,
        min_bound,
        max_bound,
        gif_frames=gif_frames,
        gif_face_stride=gif_face_stride,
        image_size_px=image_size_px,
    )
    gif_images[0].save(
        outputs.front_gif,
        save_all=True,
        append_images=gif_images[1:],
        duration=gif_duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )

    logger.info(
        "Rendered preview for %s -> %s (contact samples=%d, gif frames=%d)",
        mesh_dir,
        outputs.output_dir,
        contact_samples,
        len(gif_indices),
    )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Render contact-sheet PNGs and front-view GIFs for target mesh libraries.",
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--mesh-dir",
        type=Path,
        help="Render previews for a single mesh library directory",
    )
    target_group.add_argument(
        "--root-dir",
        type=Path,
        help="Render previews for each immediate child mesh library under this root",
    )
    parser.add_argument(
        "--min-ply-count",
        type=int,
        default=2,
        help="Minimum number of PLY files for batch mode (default: 2)",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="preview",
        help="Subdirectory under each mesh library for preview outputs (default: preview)",
    )
    parser.add_argument(
        "--contact-samples",
        type=int,
        default=6,
        help="Number of frames sampled into the contact sheet (default: 6)",
    )
    parser.add_argument(
        "--gif-frames",
        type=int,
        default=60,
        help="Number of frames sampled into the GIF (default: 60)",
    )
    parser.add_argument(
        "--gif-duration-ms",
        type=int,
        default=33,
        help="Per-frame GIF duration in milliseconds (default: 33)",
    )
    parser.add_argument(
        "--image-size-px",
        type=int,
        default=240,
        help="Rendered image size per frame in pixels (default: 240)",
    )
    parser.add_argument(
        "--contact-face-stride",
        type=int,
        default=2,
        help="Triangle decimation stride for contact sheets (default: 2)",
    )
    parser.add_argument(
        "--gif-face-stride",
        type=int,
        default=6,
        help="Triangle decimation stride for GIFs (default: 6)",
    )
    parser.add_argument(
        "--padding-fraction",
        type=float,
        default=0.08,
        help="Extra axis padding fraction around mesh bounds (default: 0.08)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip libraries whose preview outputs already exist",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    check_dependencies()

    if args.mesh_dir is not None:
        mesh_dirs = [args.mesh_dir]
    else:
        if not args.root_dir.exists():
            logger.error("Root directory not found: %s", args.root_dir)
            sys.exit(1)
        mesh_dirs = find_mesh_libraries(args.root_dir, min_ply_count=args.min_ply_count)

    if not mesh_dirs:
        logger.warning("No mesh libraries found")
        return

    for mesh_dir in mesh_dirs:
        if not mesh_dir.exists():
            logger.error("Mesh directory not found: %s", mesh_dir)
            continue
        try:
            render_library_preview(
                mesh_dir,
                output_subdir=args.output_subdir,
                contact_samples=args.contact_samples,
                gif_frames=args.gif_frames,
                gif_duration_ms=args.gif_duration_ms,
                image_size_px=args.image_size_px,
                contact_face_stride=args.contact_face_stride,
                gif_face_stride=args.gif_face_stride,
                padding_fraction=args.padding_fraction,
                skip_existing=args.skip_existing,
            )
        except Exception as exc:  # noqa: BLE001 - CLI top-level boundary
            logger.error("Failed to render preview for %s: %s", mesh_dir, exc)


if __name__ == "__main__":
    main()
