"""Adapt external ray-path records and publish an ORCHAV frame set.

The conversion functions are source-specific and intentionally small. They
end at ``StandardMPCFrame``; the shared writer owns the native storage layout,
and the existing provider owns reading it back.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from shared.frames import FrameSetWriter, StandardMPCFrame, standard_mpc_frame_from_pair_data
from shared.frames.providers import Hdf5Provider

from .fake_raytracer import DeviceRecord, FrameRecord, PathRecord, trace

_INTERACTION_CODES = {
    # LoS paths contain no bounce row. Code 1 identifies a physical specular
    # reflection when the external source reports a surface reflection.
    "surface_reflection": 1,
}


def _indices_by_name(devices: Sequence[DeviceRecord], axis: str) -> dict[str, int]:
    """Return stable device indices and reject ambiguous external names."""

    indices = {device.name: index for index, device in enumerate(devices)}
    if len(indices) != len(devices):
        raise ValueError(f"External {axis} device names must be unique")
    return indices


def _pair_groups(
    record: FrameRecord,
) -> tuple[tuple[tuple[int, int], tuple[PathRecord, ...]], ...]:
    """Group paths by sparse TX/RX pair while preserving source order."""

    tx_indices = _indices_by_name(record.transmitters, "transmitter")
    rx_indices = _indices_by_name(record.receivers, "receiver")
    grouped: dict[tuple[int, int], list[PathRecord]] = {}
    for path_index, path in enumerate(record.paths):
        try:
            pair = (tx_indices[path.transmitter], rx_indices[path.receiver])
        except KeyError as exc:
            raise ValueError(
                f"External path {path_index} references unknown device {exc.args[0]!r}"
            ) from exc
        grouped.setdefault(pair, []).append(path)
    return tuple((pair, tuple(paths)) for pair, paths in grouped.items())


def _bounce_interactions(path: PathRecord) -> np.ndarray:
    """Map source interaction names to canonical positive bounce codes."""

    try:
        values = [_INTERACTION_CODES[bounce.mechanism] for bounce in path.bounces]
    except KeyError as exc:
        raise ValueError(
            f"External path {path.transmitter!r} -> {path.receiver!r} uses "
            f"unsupported interaction {exc.args[0]!r}"
        ) from exc
    return np.asarray(values, dtype=np.uint8)


def _ragged_material_axis(
    paths: Sequence[PathRecord],
    attribute: str,
) -> np.ndarray:
    """Keep one variable-length material vector per external path."""

    values = np.empty((len(paths),), dtype=object)
    for path_index, path in enumerate(paths):
        values[path_index] = np.asarray(
            [getattr(bounce, attribute) for bounce in path.bounces],
            dtype=object,
        )
    return values


def to_standard_frame(record: FrameRecord) -> StandardMPCFrame:
    """Convert one source-owned frame record to the canonical ORCHAV frame."""

    pair_groups = _pair_groups(record)
    pairs = [pair for pair, _paths in pair_groups]
    paths_by_pair = [paths for _pair, paths in pair_groups]

    vertices_by_pair = [
        tuple(
            np.asarray([bounce.position_m for bounce in path.bounces], dtype=np.float64).reshape(
                (-1, 3)
            )
            for path in paths
        )
        for paths in paths_by_pair
    ]
    interactions_by_pair = [
        tuple(_bounce_interactions(path) for path in paths) for paths in paths_by_pair
    ]
    material_names_by_pair = [
        _ragged_material_axis(paths, "material_name") for paths in paths_by_pair
    ]
    material_families_by_pair = [
        _ragged_material_axis(paths, "material_family") for paths in paths_by_pair
    ]

    metrics_by_pair = {
        "delay_ns": [
            np.asarray([path.delay_s * 1e9 for path in paths], dtype=np.float64)
            for paths in paths_by_pair
        ],
        "path_loss_db": [
            np.asarray([path.path_loss_db for path in paths], dtype=np.float64)
            for paths in paths_by_pair
        ],
        "aoa_az_deg": [
            np.degrees([path.arrival_azimuth_rad for path in paths]) for paths in paths_by_pair
        ],
        "aoa_el_deg": [
            np.degrees([path.arrival_elevation_rad for path in paths]) for paths in paths_by_pair
        ],
        "aod_az_deg": [
            np.degrees([path.departure_azimuth_rad for path in paths]) for paths in paths_by_pair
        ],
        "aod_el_deg": [
            np.degrees([path.departure_elevation_rad for path in paths]) for paths in paths_by_pair
        ],
    }

    target_positions = [target.position_m for target in record.targets]
    target_metadata = [
        {
            "name": target.name,
            "current_position": list(target.position_m),
            "velocity_m_s": list(target.velocity_m_s),
            "category": target.category,
        }
        for target in record.targets
    ]

    return standard_mpc_frame_from_pair_data(
        frame_index=record.frame_index,
        timestamp_s=record.timestamp_s,
        tx_rx_pairs=pairs,
        tx_positions=[device.position_m for device in record.transmitters],
        rx_positions=[device.position_m for device in record.receivers],
        tx_orientations=np.radians([device.orientation_deg for device in record.transmitters]),
        rx_orientations=np.radians([device.orientation_deg for device in record.receivers]),
        tx_names=[device.name for device in record.transmitters],
        rx_names=[device.name for device in record.receivers],
        vertices_by_pair=vertices_by_pair,
        interactions_by_pair=interactions_by_pair,
        material_names_by_pair=material_names_by_pair,
        material_itu_types_by_pair=material_families_by_pair,
        metrics_by_pair=metrics_by_pair,
        target_positions_m=target_positions,
        targets_metadata=target_metadata,
        provenance={
            "source": "fake_external_raytracer",
            "source_frame_index": record.frame_index,
        },
    )


def publish_frames(records: Iterable[FrameRecord], destination: str | Path) -> None:
    """Publish ordered external records to a new native frame-set directory."""

    with FrameSetWriter.create_new(destination) as writer:
        for record in records:
            writer.append(to_standard_frame(record))
        manifest = writer.finalize(provenance={"importer": "external_raytracer_example"})
    if manifest is None:
        raise ValueError("The external ray tracer produced no frames")


def reload_frame_ids(destination: str | Path) -> tuple[list[int], str | None]:
    """Reload the new frame set through the normal consumer-side provider."""

    frames_dir = Path(destination).expanduser().absolute()
    with Hdf5Provider(frames_dir.parent, frames_subdir=frames_dir.name) as provider:
        frame_ids = provider.list_frames()
        for frame_id in frame_ids:
            provider.load_frame(frame_id)
        return frame_ids, provider.info.frame_set_id


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import deterministic external ray paths into a new ORCHAV frame set."
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="Absent directory to create; an existing path is never replaced.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the external import and verify the published frames."""

    args = _build_parser().parse_args(argv)
    publish_frames(trace(), args.destination)
    frame_ids, frame_set_id = reload_frame_ids(args.destination)
    print(  # noqa: T201 - command-line result
        f"Published frames {frame_ids} to {args.destination.resolve()}"
    )
    print(f"Frame-set ID: {frame_set_id}")  # noqa: T201 - command-line result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "publish_frames", "reload_frame_ids", "to_standard_frame"]
