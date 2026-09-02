#!/usr/bin/env python3
"""Generate synthetic packed HDF5 frames for visualizer benchmarking.

The generator creates one complete ``StandardMPCFrame`` at a time and hands it
to the transactional run-level writer. Memory therefore scales with one frame,
not with the configured HDF5 chunk size or total scenario length.

Examples:
    python generate.py
    python generate.py --frames 100 --mpcs 100000
    python generate.py --frames 60 --mpcs 100000 --max-interactions 5
    python generate.py --frames 1000 --chunk-size 50 --compression lzf
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_project_root_entry = str(PROJECT_ROOT)
if _project_root_entry in sys.path:
    sys.path.remove(_project_root_entry)
sys.path.insert(0, _project_root_entry)

from generator.core.configuration import build_simulation_config  # noqa: E402
from generator.io.storage.hdf5_frame_output import (  # noqa: E402
    HDF5FrameOutputStrategy,
)
from shared.frames.contracts import PATH_METRIC_VALIDITY_BITS  # noqa: E402
from shared.frames.types import StandardMPCFrame  # noqa: E402
from shared.scenarios import load_scenario_configuration  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_MAX_INTERACTIONS = 2
DEFAULT_COMPRESSION = "lzf"
DEFAULT_OUTPUT_SCENARIO = Path("tmp/orchav_bench_10k")
TX_POS = np.array([[0.0, 0.0, 30.0]], dtype=np.float64)
RX_POS = np.array([[50.0, 50.0, 1.5]], dtype=np.float64)
MATERIALS = (
    "mat-itu_concrete",
    "mat-itu_glass",
    "mat-itu_metal",
    "mat-itu_plasterboard",
)
MATERIAL_ITU_TYPES = (
    "concrete",
    "glass",
    "metal",
    "plasterboard",
)
INTERACTION_TYPES = np.array([1, 2, 4, 8], dtype=np.int32)
COMPRESSION_CHOICES = ("none", "lzf", "gzip")
ALL_METRICS_VALID = np.uint8(sum(PATH_METRIC_VALIDITY_BITS.values()))


def generate_standard_frame(
    num_mpcs: int,
    frame_idx: int,
    rng: np.random.Generator,
    max_interactions: int,
    *,
    source_provider: str = "synthetic_mpc_benchmark",
) -> StandardMPCFrame:
    """Generate one structurally valid compact MPC frame."""
    lengths = rng.integers(
        1,
        max_interactions + 1,
        size=num_mpcs,
        dtype=np.int32,
    )
    bounce_offsets = np.zeros((num_mpcs + 1,), dtype=np.int64)
    np.cumsum(lengths, dtype=np.int64, out=bounce_offsets[1:])
    num_bounces = int(bounce_offsets[-1])

    bounce_xyz_m = rng.uniform(
        low=[-100.0, -100.0, 0.0],
        high=[100.0, 100.0, 40.0],
        size=(num_bounces, 3),
    ).astype(np.float32)
    interactions = rng.choice(
        INTERACTION_TYPES,
        size=num_bounces,
    ).astype(np.uint8, copy=False)
    material_ids = rng.integers(
        1,
        len(MATERIALS) + 1,
        size=num_bounces,
        dtype=np.uint16,
    )

    return StandardMPCFrame(
        frame_index=frame_idx,
        provenance={
            "provider": source_provider,
            "frame_idx": frame_idx,
        },
        tx_rx_pairs=np.array([[0, 0]], dtype=np.int32),
        pair_path_offsets=np.array([0, num_mpcs], dtype=np.int64),
        bounce_offsets=bounce_offsets,
        tx_positions=TX_POS.copy(),
        rx_positions=RX_POS.copy(),
        tx_orientations=np.zeros((1, 3), dtype=np.float64),
        rx_orientations=np.zeros((1, 3), dtype=np.float64),
        tx_names=("tx_0",),
        rx_names=("rx_0",),
        bounce_xyz_m=bounce_xyz_m,
        interactions=interactions,
        material_ids=material_ids,
        material_names=("", *MATERIALS),
        material_itu_types=("", *MATERIAL_ITU_TYPES),
        delays_ns=rng.uniform(0.0, 500.0, size=num_mpcs).astype(np.float32),
        path_loss_db=rng.uniform(-150.0, -40.0, size=num_mpcs).astype(np.float32),
        aoa_az_deg=rng.uniform(-180.0, 180.0, size=num_mpcs).astype(np.float32),
        aoa_el_deg=rng.uniform(-90.0, 90.0, size=num_mpcs).astype(np.float32),
        aod_az_deg=rng.uniform(-180.0, 180.0, size=num_mpcs).astype(np.float32),
        aod_el_deg=rng.uniform(-90.0, 90.0, size=num_mpcs).astype(np.float32),
        metric_valid_bits=np.full((num_mpcs,), ALL_METRICS_VALID, dtype=np.uint8),
        target_positions_m=np.empty((0, 3), dtype=np.float64),
        targets_metadata=(),
    )


def write_scenario_yaml(
    scenario_dir: Path,
    num_frames: int,
    *,
    compression: str = DEFAULT_COMPRESSION,
    chunk_size: int = 50,
) -> None:
    """Write a file-backed scenario that uses the packed frame-set writer."""
    cfg = {
        "schema_version": 2,
        "debug_level": "WARNING",
        "scene": {"source": "sionna", "id": "etoile"},
        "data": {
            "mode": "files",
            "files": {
                "format": "hdf5",
                "directory": "frames",
                "compression": compression,
                "chunk_size": chunk_size,
            },
        },
        "raytracing": {"enabled": False},
        "timeline": {"steps": num_frames, "duration_s": 0.0},
        "view_defaults": {
            "camera_view": "isometric",
            "mpc_visibility": {
                "enabled": True,
                "paths": True,
                "bounce_points": False,
            },
        },
        "visualizer": {"panels": {"statistics": {"enabled": False}}},
    }
    scenario_dir.mkdir(parents=True, exist_ok=True)
    with open(scenario_dir / "scenario.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, default_flow_style=False, sort_keys=False)


def generate_scenario(
    scenario_dir: Path,
    *,
    num_frames: int,
    mpcs_min: int,
    mpcs_max: int,
    max_interactions: int,
    chunk_size: int,
    compression: str,
    seed: int,
) -> Path:
    """Generate and atomically publish one complete packed frame set."""
    if num_frames <= 0:
        raise ValueError("num_frames must be a positive integer")
    if max_interactions < 1:
        raise ValueError("max_interactions must be at least 1")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if mpcs_min <= 0 or mpcs_max <= 0:
        raise ValueError("MPC counts must be positive")
    if mpcs_min > mpcs_max:
        raise ValueError("mpcs_min must be less than or equal to mpcs_max")

    write_scenario_yaml(
        scenario_dir,
        num_frames,
        compression=compression,
        chunk_size=chunk_size,
    )
    scenario_configuration = load_scenario_configuration(
        scenario_dir,
        project_root=PROJECT_ROOT,
    )
    output = HDF5FrameOutputStrategy(
        build_simulation_config(scenario_configuration),
        scenario_configuration,
    )

    rng = np.random.default_rng(seed=seed)
    t0 = time.perf_counter()
    try:
        for frame_idx in range(num_frames):
            if mpcs_min == mpcs_max:
                num_mpcs = mpcs_min
            else:
                num_mpcs = int(rng.integers(mpcs_min, mpcs_max + 1))
            frame = generate_standard_frame(
                num_mpcs,
                frame_idx,
                rng,
                max_interactions,
            )
            output.save_standard_frame(frame)
            if (frame_idx + 1) % 100 == 0 or frame_idx + 1 == num_frames:
                elapsed = time.perf_counter() - t0
                logger.info(
                    "  Generated and appended [%4d/%d]  %.1fs",
                    frame_idx + 1,
                    num_frames,
                    elapsed,
                )
        output.finalize()
    except BaseException:
        output.abort()
        raise

    return scenario_configuration.frames_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate packed HDF5 frames for visualizer benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python generate.py
  python generate.py --mpcs 100000
  python generate.py --frames 1000 --mpcs 10000
  python generate.py --frames 60 --mpcs 100000 --max-interactions 5
""",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="Number of frames to generate (default: 10)",
    )
    parser.add_argument(
        "--mpcs",
        type=int,
        default=10_000,
        help="MPCs per frame (default: 10000)",
    )
    parser.add_argument(
        "--mpcs-min",
        type=int,
        default=None,
        help="Minimum MPCs per frame for variable-count generation.",
    )
    parser.add_argument(
        "--mpcs-max",
        type=int,
        default=None,
        help="Maximum MPCs per frame for variable-count generation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_SCENARIO,
        help=f"Output scenario directory (default: {DEFAULT_OUTPUT_SCENARIO.as_posix()})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--compression",
        choices=COMPRESSION_CHOICES,
        default=DEFAULT_COMPRESSION,
        help=f"HDF5 compression (default: {DEFAULT_COMPRESSION})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Maximum frames requested per chunk (default: 50)",
    )
    parser.add_argument(
        "--max-interactions",
        "--max-order",
        dest="max_interactions",
        type=int,
        default=DEFAULT_MAX_INTERACTIONS,
        help=(
            "Maximum synthetic interaction points per MPC " f"(default: {DEFAULT_MAX_INTERACTIONS})"
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    scenario_dir = args.output.resolve()
    mpcs_min = args.mpcs if args.mpcs_min is None else int(args.mpcs_min)
    mpcs_max = args.mpcs if args.mpcs_max is None else int(args.mpcs_max)
    mpc_label = f"{mpcs_min:,}" if mpcs_min == mpcs_max else f"{mpcs_min:,}..{mpcs_max:,}"
    logger.info(
        "Generating %d packed frames x %s MPCs "
        "(max_interactions=%d, compression=%s, chunk_size=%d) -> %s",
        args.frames,
        mpc_label,
        args.max_interactions,
        args.compression,
        args.chunk_size,
        scenario_dir,
    )
    t0 = time.perf_counter()
    frames_dir = generate_scenario(
        scenario_dir,
        num_frames=args.frames,
        mpcs_min=mpcs_min,
        mpcs_max=mpcs_max,
        max_interactions=args.max_interactions,
        chunk_size=args.chunk_size,
        compression=args.compression,
        seed=args.seed,
    )
    total_size = sum(path.stat().st_size for path in frames_dir.glob("*.h5"))
    logger.info(
        "Done in %.1fs  |  %.1f MB",
        time.perf_counter() - t0,
        total_size / 1e6,
    )
    logger.info("To open in the visualizer:")
    logger.info("  python -m visualizer --scenario %s", scenario_dir / "scenario.yaml")


if __name__ == "__main__":
    main()
