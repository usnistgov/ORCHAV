"""Deterministic semantic MPC fixtures and renderer-neutral parity helpers.

The fixtures in this module describe paths as logical polylines first, then
encode them as ``StandardMPCFrame`` arrays.  Expected canonical and renderer
payloads are derived independently from that semantic description.  This makes
the helpers suitable for comparing storage implementations without retaining
binary golden files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

import numpy as np

from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.types import StandardMPCFrame

Point3 = tuple[float, float, float]
FrameVariant = Literal["baseline", "changed", "empty"]


@dataclass(frozen=True, slots=True)
class SemanticPath:
    """One producer-ordered MPC path and its per-path metadata."""

    bounces: tuple[Point3, ...]
    interactions: tuple[int, ...]
    materials: tuple[str, ...]
    material_itu_types: tuple[str, ...]
    delay_ns: float
    path_loss_db: float
    aoa_az_deg: float
    aoa_el_deg: float
    aod_az_deg: float
    aod_el_deg: float

    def __post_init__(self) -> None:
        """Reject fixture definitions whose bounce metadata is misaligned."""
        bounce_count = len(self.bounces)
        if len(self.interactions) != bounce_count:
            raise ValueError("interactions must contain one value per bounce")
        if len(self.materials) != bounce_count:
            raise ValueError("materials must contain one value per bounce")
        if len(self.material_itu_types) != bounce_count:
            raise ValueError("material_itu_types must contain one value per bounce")


@dataclass(frozen=True, slots=True)
class SemanticMPCFrame:
    """Logical frame definition with explicit pair and path ordering."""

    name: str
    tx_positions: tuple[Point3, ...]
    rx_positions: tuple[Point3, ...]
    pair_order: tuple[tuple[int, int], ...]
    paths_by_pair: tuple[tuple[SemanticPath, ...], ...]

    def __post_init__(self) -> None:
        """Ensure every pair owns exactly one producer-order path sequence."""
        if len(self.paths_by_pair) != len(self.pair_order):
            raise ValueError("paths_by_pair must align with pair_order")

    @property
    def paths(self) -> tuple[SemanticPath, ...]:
        """Return paths flattened in durable pair-major, path-major order."""
        return tuple(path for pair_paths in self.paths_by_pair for path in pair_paths)


@dataclass(frozen=True, slots=True)
class SemanticMPCOracle:
    """Expected canonical arrays derived from a logical semantic frame."""

    points: np.ndarray
    lines: np.ndarray
    point_orders: np.ndarray
    point_interactions: np.ndarray
    point_delays_ns: np.ndarray
    point_losses_db: np.ndarray
    point_tx: np.ndarray
    point_rx: np.ndarray
    point_path_ids: np.ndarray
    point_material_names: tuple[str, ...]
    path_starts: np.ndarray
    path_orders: np.ndarray
    path_delays_ns: np.ndarray
    path_losses_db: np.ndarray
    path_delay_is_estimated: np.ndarray
    path_loss_is_estimated: np.ndarray
    path_tx: np.ndarray
    path_rx: np.ndarray
    aoa_az_deg: np.ndarray
    aoa_el_deg: np.ndarray
    aod_az_deg: np.ndarray
    aod_el_deg: np.ndarray
    segment_interactions: np.ndarray
    segment_path_ids: np.ndarray
    segment_material_names: tuple[str, ...]
    bounce_points: np.ndarray
    bounce_interactions: np.ndarray


_TX_POSITIONS: tuple[Point3, ...] = (
    (0.0, 0.0, 0.0),
    (10.0, 0.0, 1.0),
)
_RX_POSITIONS: tuple[Point3, ...] = (
    (0.0, 10.0, 2.0),
    (10.0, 10.0, 3.0),
)

# Deliberately not Cartesian-product order.  The empty (1, 1) pair in the
# middle also detects implementations that infer pair identity from list index.
_PAIR_ORDER: tuple[tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (1, 1),
    (0, 0),
)

BASELINE_SEMANTIC_FRAME = SemanticMPCFrame(
    name="baseline",
    tx_positions=_TX_POSITIONS,
    rx_positions=_RX_POSITIONS,
    pair_order=_PAIR_ORDER,
    paths_by_pair=(
        (
            SemanticPath(
                bounces=(),
                interactions=(),
                materials=(),
                material_itu_types=(),
                delay_ns=0.0,
                path_loss_db=0.0,
                aoa_az_deg=0.0,
                aoa_el_deg=0.0,
                aod_az_deg=-180.0,
                aod_el_deg=90.0,
            ),
            SemanticPath(
                bounces=((7.0, 4.0, 2.0),),
                interactions=(1,),
                materials=("mat-itu_concrete",),
                material_itu_types=("concrete",),
                delay_ns=11.0,
                path_loss_db=41.0,
                aoa_az_deg=180.0,
                aoa_el_deg=-45.0,
                aod_az_deg=359.5,
                aod_el_deg=-90.0,
            ),
        ),
        (
            SemanticPath(
                bounces=((2.0, 3.0, 1.0), (4.0, 5.0, 1.0), (6.0, 7.0, 2.0)),
                interactions=(2, 4, 8),
                materials=("mat-itu_glass", "mat-itu_wood", "mat-itu_metal"),
                material_itu_types=("glass", "wood", "metal"),
                delay_ns=float("nan"),
                path_loss_db=52.0,
                aoa_az_deg=float("nan"),
                aoa_el_deg=10.0,
                aod_az_deg=20.0,
                aod_el_deg=float("nan"),
            ),
        ),
        (),
        (
            SemanticPath(
                bounces=((-1.0, 4.0, 1.0), (-2.0, 7.0, 1.5)),
                interactions=(99, 37),
                materials=("virtual-proxy", "custom-absorber"),
                material_itu_types=("virtual", "custom"),
                delay_ns=23.0,
                path_loss_db=63.0,
                aoa_az_deg=45.0,
                aoa_el_deg=float("nan"),
                aod_az_deg=-45.0,
                aod_el_deg=0.0,
            ),
        ),
    ),
)

CHANGED_SEMANTIC_FRAME = SemanticMPCFrame(
    name="changed",
    tx_positions=_TX_POSITIONS,
    rx_positions=_RX_POSITIONS,
    pair_order=_PAIR_ORDER,
    paths_by_pair=(
        (
            SemanticPath(
                bounces=((8.0, 6.0, 3.0),),
                interactions=(4,),
                materials=("mat-itu_glass",),
                material_itu_types=("glass",),
                delay_ns=17.0,
                path_loss_db=47.0,
                aoa_az_deg=15.0,
                aoa_el_deg=5.0,
                aod_az_deg=25.0,
                aod_el_deg=-5.0,
            ),
        ),
        (),
        (
            SemanticPath(
                bounces=(),
                interactions=(),
                materials=(),
                material_itu_types=(),
                delay_ns=3.0,
                path_loss_db=33.0,
                aoa_az_deg=90.0,
                aoa_el_deg=0.0,
                aod_az_deg=-90.0,
                aod_el_deg=0.0,
            ),
        ),
        (
            SemanticPath(
                bounces=((-3.0, 5.0, 1.0),),
                interactions=(37,),
                materials=("custom-absorber",),
                material_itu_types=("custom",),
                delay_ns=29.0,
                path_loss_db=69.0,
                aoa_az_deg=float("nan"),
                aoa_el_deg=float("nan"),
                aod_az_deg=float("nan"),
                aod_el_deg=float("nan"),
            ),
        ),
    ),
)

EMPTY_SEMANTIC_FRAME = SemanticMPCFrame(
    name="empty",
    tx_positions=_TX_POSITIONS,
    rx_positions=_RX_POSITIONS,
    pair_order=_PAIR_ORDER,
    paths_by_pair=((), (), (), ()),
)


def semantic_frame(variant: FrameVariant = "baseline") -> SemanticMPCFrame:
    """Return one immutable semantic frame definition."""
    if variant == "baseline":
        return BASELINE_SEMANTIC_FRAME
    if variant == "changed":
        return CHANGED_SEMANTIC_FRAME
    if variant == "empty":
        return EMPTY_SEMANTIC_FRAME
    raise ValueError(f"Unsupported semantic frame variant: {variant}")


def semantic_frame_sequence() -> tuple[SemanticMPCFrame, ...]:
    """Return the populated/changed/empty/populated transition sequence."""
    return (
        BASELINE_SEMANTIC_FRAME,
        CHANGED_SEMANTIC_FRAME,
        EMPTY_SEMANTIC_FRAME,
        BASELINE_SEMANTIC_FRAME,
    )


def _metric_array(paths: Iterable[SemanticPath], attribute: str) -> np.ndarray:
    """Build one float32 per-path metric array."""
    return np.asarray([getattr(path, attribute) for path in paths], dtype=np.float32)


def build_standard_mpc_frame(
    variant: FrameVariant | SemanticMPCFrame = "baseline",
    *,
    frame_idx: int = 0,
) -> StandardMPCFrame:
    """Encode a semantic fixture as the current ``StandardMPCFrame`` contract."""
    spec = semantic_frame(variant) if isinstance(variant, str) else variant

    all_vertices: list[np.ndarray] = []
    all_interactions: list[np.ndarray] = []
    all_lengths: list[np.ndarray] = []
    all_material_names: list[np.ndarray] = []
    all_material_itu_types: list[np.ndarray] = []
    all_delays: list[np.ndarray] = []
    all_losses: list[np.ndarray] = []
    all_aoa_az: list[np.ndarray] = []
    all_aoa_el: list[np.ndarray] = []
    all_aod_az: list[np.ndarray] = []
    all_aod_el: list[np.ndarray] = []

    for paths in spec.paths_by_pair:
        path_count = len(paths)
        depth = max((len(path.bounces) for path in paths), default=0)
        vertices = np.full((path_count, depth, 3), np.nan, dtype=np.float32)
        interactions = np.full((path_count, depth), -1, dtype=np.int32)
        material_names = np.full((path_count, depth), "", dtype=object)
        material_itu_types = np.full((path_count, depth), "", dtype=object)
        lengths = np.empty((path_count,), dtype=np.int32)

        for path_idx, path in enumerate(paths):
            bounce_count = len(path.bounces)
            lengths[path_idx] = bounce_count
            if bounce_count:
                vertices[path_idx, :bounce_count] = np.asarray(path.bounces, dtype=np.float32)
                interactions[path_idx, :bounce_count] = np.asarray(
                    path.interactions,
                    dtype=np.int32,
                )
                material_names[path_idx, :bounce_count] = path.materials
                material_itu_types[path_idx, :bounce_count] = path.material_itu_types

        all_vertices.append(vertices)
        all_interactions.append(interactions)
        all_lengths.append(lengths)
        all_material_names.append(material_names)
        all_material_itu_types.append(material_itu_types)
        all_delays.append(_metric_array(paths, "delay_ns"))
        all_losses.append(_metric_array(paths, "path_loss_db"))
        all_aoa_az.append(_metric_array(paths, "aoa_az_deg"))
        all_aoa_el.append(_metric_array(paths, "aoa_el_deg"))
        all_aod_az.append(_metric_array(paths, "aod_az_deg"))
        all_aod_el.append(_metric_array(paths, "aod_el_deg"))

    return standard_mpc_frame_from_pair_data(
        frame_index=frame_idx,
        tx_rx_pairs=np.asarray(spec.pair_order, dtype=np.int32),
        tx_positions=np.asarray(spec.tx_positions, dtype=np.float64),
        rx_positions=np.asarray(spec.rx_positions, dtype=np.float64),
        vertices_by_pair=all_vertices,
        interactions_by_pair=all_interactions,
        path_lengths_by_pair=all_lengths,
        tx_orientations=np.zeros((len(spec.tx_positions), 3), dtype=np.float64),
        rx_orientations=np.zeros((len(spec.rx_positions), 3), dtype=np.float64),
        tx_names=tuple(f"tx_{idx}" for idx in range(len(spec.tx_positions))),
        rx_names=tuple(f"rx_{idx}" for idx in range(len(spec.rx_positions))),
        material_names_by_pair=all_material_names,
        material_itu_types_by_pair=all_material_itu_types,
        metrics_by_pair={
            "delays_ns": all_delays,
            "path_loss_db": all_losses,
            "aoa_az_deg": all_aoa_az,
            "aoa_el_deg": all_aoa_el,
            "aod_az_deg": all_aod_az,
            "aod_el_deg": all_aod_el,
        },
        target_positions_m=np.empty((0, 3), dtype=np.float64),
        targets_metadata=(),
        beamforming=None,
        sensing=None,
        recomputed_from_stored_positions=False,
        provenance={
            "provider": "semantic-test-fixture",
            "frame_idx": int(frame_idx),
            "variant": spec.name,
        },
    )


def _geometric_metrics(
    coordinates: np.ndarray,
    path: SemanticPath,
) -> tuple[float, float, bool, bool]:
    """Apply the documented geometric fallback used for missing RF metrics."""
    segment_lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    length_m = float(np.sum(segment_lengths))

    delay_missing = not np.isfinite(path.delay_ns)
    loss_missing = not np.isfinite(path.path_loss_db)
    delay_ns = length_m / 0.3 if delay_missing else float(path.delay_ns)
    path_loss_db = (
        20.0 * np.log10(max(length_m, 1e-6)) + 30.0 if loss_missing else float(path.path_loss_db)
    )
    return delay_ns, path_loss_db, delay_missing, loss_missing


def build_semantic_oracle(
    variant: FrameVariant | SemanticMPCFrame = "baseline",
) -> SemanticMPCOracle:
    """Build expected canonical arrays without calling visualizer production code."""
    spec = semantic_frame(variant) if isinstance(variant, str) else variant

    point_blocks: list[np.ndarray] = []
    line_blocks: list[np.ndarray] = []
    point_orders: list[int] = []
    point_interactions: list[int] = []
    point_delays: list[float] = []
    point_losses: list[float] = []
    point_tx: list[int] = []
    point_rx: list[int] = []
    point_path_ids: list[int] = []
    point_material_names: list[str] = []
    path_starts: list[int] = []
    path_orders: list[int] = []
    path_delays: list[float] = []
    path_losses: list[float] = []
    path_delay_is_estimated: list[bool] = []
    path_loss_is_estimated: list[bool] = []
    path_tx: list[int] = []
    path_rx: list[int] = []
    aoa_az: list[float] = []
    aoa_el: list[float] = []
    aod_az: list[float] = []
    aod_el: list[float] = []
    segment_interactions: list[int] = []
    segment_path_ids: list[int] = []
    segment_material_names: list[str] = []
    bounce_points: list[Point3] = []
    bounce_interactions: list[int] = []

    point_offset = 0
    path_id = 0
    for (tx_idx, rx_idx), paths in zip(spec.pair_order, spec.paths_by_pair):
        tx_position = spec.tx_positions[tx_idx]
        rx_position = spec.rx_positions[rx_idx]
        for path in paths:
            coordinates = np.asarray(
                (tx_position, *path.bounces, rx_position),
                dtype=np.float64,
            )
            node_count = len(coordinates)
            segment_count = node_count - 1
            order = sum(interaction != 0 for interaction in path.interactions)
            delay_ns, loss_db, delay_estimated, loss_estimated = _geometric_metrics(
                coordinates,
                path,
            )

            point_blocks.append(coordinates)
            starts = np.arange(point_offset, point_offset + segment_count, dtype=np.int32)
            line_blocks.append(np.column_stack((starts, starts + 1)))
            path_starts.append(point_offset)
            point_orders.extend([order] * node_count)
            point_interactions.extend((0, *path.interactions, 0))
            point_delays.extend([delay_ns] * node_count)
            point_losses.extend([loss_db] * node_count)
            point_tx.extend([tx_idx] * node_count)
            point_rx.extend([rx_idx] * node_count)
            point_path_ids.extend([path_id] * node_count)
            point_material_names.extend(("", *path.materials, ""))
            path_orders.append(order)
            path_delays.append(delay_ns)
            path_losses.append(loss_db)
            path_delay_is_estimated.append(delay_estimated)
            path_loss_is_estimated.append(loss_estimated)
            path_tx.append(tx_idx)
            path_rx.append(rx_idx)
            aoa_az.extend([path.aoa_az_deg] * node_count)
            aoa_el.extend([path.aoa_el_deg] * node_count)
            aod_az.extend([path.aod_az_deg] * node_count)
            aod_el.extend([path.aod_el_deg] * node_count)
            if path.interactions:
                segment_interactions.extend(
                    (path.interactions[0], *path.interactions),
                )
            else:
                segment_interactions.append(0)
            segment_path_ids.extend([path_id] * segment_count)
            segment_material_names.extend(("", *path.materials))
            bounce_points.extend(path.bounces)
            bounce_interactions.extend(path.interactions)

            point_offset += node_count
            path_id += 1

    points = (
        np.vstack(point_blocks).astype(np.float64, copy=False)
        if point_blocks
        else np.empty((0, 3), dtype=np.float64)
    )
    lines = (
        np.vstack(line_blocks).astype(np.int32, copy=False)
        if line_blocks
        else np.empty((0, 2), dtype=np.int32)
    )

    return SemanticMPCOracle(
        points=points,
        lines=lines,
        point_orders=np.asarray(point_orders, dtype=np.uint8),
        point_interactions=np.asarray(point_interactions, dtype=np.uint8),
        point_delays_ns=np.asarray(point_delays, dtype=np.float32),
        point_losses_db=np.asarray(point_losses, dtype=np.float32),
        point_tx=np.asarray(point_tx, dtype=np.int16),
        point_rx=np.asarray(point_rx, dtype=np.int16),
        point_path_ids=np.asarray(point_path_ids, dtype=np.int32),
        point_material_names=tuple(point_material_names),
        path_starts=np.asarray(path_starts, dtype=np.int32),
        path_orders=np.asarray(path_orders, dtype=np.uint8),
        path_delays_ns=np.asarray(path_delays, dtype=np.float32),
        path_losses_db=np.asarray(path_losses, dtype=np.float32),
        path_delay_is_estimated=np.asarray(path_delay_is_estimated, dtype=bool),
        path_loss_is_estimated=np.asarray(path_loss_is_estimated, dtype=bool),
        path_tx=np.asarray(path_tx, dtype=np.int16),
        path_rx=np.asarray(path_rx, dtype=np.int16),
        aoa_az_deg=np.asarray(aoa_az, dtype=np.float32),
        aoa_el_deg=np.asarray(aoa_el, dtype=np.float32),
        aod_az_deg=np.asarray(aod_az, dtype=np.float32),
        aod_el_deg=np.asarray(aod_el, dtype=np.float32),
        segment_interactions=np.asarray(segment_interactions, dtype=np.uint8),
        segment_path_ids=np.asarray(segment_path_ids, dtype=np.int32),
        segment_material_names=tuple(segment_material_names),
        bounce_points=np.asarray(bounce_points, dtype=np.float64).reshape(-1, 3),
        bounce_interactions=np.asarray(bounce_interactions, dtype=np.uint8),
    )


def _assert_float_array_equal(actual: Any, expected: np.ndarray) -> None:
    """Compare scientific arrays with the parity tolerance and matching NaNs."""
    np.testing.assert_allclose(
        np.asarray(actual),
        expected,
        rtol=1e-5,
        atol=1e-5,
        equal_nan=True,
    )


def _require_array(value: Optional[np.ndarray], field_name: str) -> np.ndarray:
    """Return an expected optional canonical array or raise a useful assertion."""
    if value is None:
        raise AssertionError(f"canonical field {field_name!r} is unexpectedly None")
    return np.asarray(value)


def _material_names_from_ids(
    material_ids: np.ndarray,
    id_to_name: Optional[dict[int, str]],
) -> tuple[str, ...]:
    """Resolve canonical material IDs into stable semantic names."""
    if id_to_name is None:
        raise AssertionError("canonical material ID lookup is unexpectedly None")
    return tuple(id_to_name[int(material_id)] for material_id in material_ids)


def assert_canonical_matches_semantics(
    canonical: Any,
    variant: FrameVariant | SemanticMPCFrame = "baseline",
) -> None:
    """Assert that canonicalization preserves the complete semantic oracle."""
    oracle = build_semantic_oracle(variant)

    _assert_float_array_equal(canonical.points, oracle.points)
    np.testing.assert_array_equal(canonical.lines, oracle.lines)
    np.testing.assert_array_equal(canonical.order, oracle.point_orders)
    np.testing.assert_array_equal(canonical.itype, oracle.point_interactions)
    _assert_float_array_equal(canonical.delay, oracle.point_delays_ns)
    _assert_float_array_equal(canonical.loss, oracle.point_losses_db)
    np.testing.assert_array_equal(
        _require_array(canonical.tx_id, "tx_id"),
        oracle.point_tx,
    )
    np.testing.assert_array_equal(
        _require_array(canonical.rx_id, "rx_id"),
        oracle.point_rx,
    )
    np.testing.assert_array_equal(
        _require_array(canonical.path_id, "path_id"),
        oracle.point_path_ids,
    )
    np.testing.assert_array_equal(
        _require_array(canonical.path_start_indices, "path_start_indices"),
        oracle.path_starts,
    )
    np.testing.assert_array_equal(
        _require_array(canonical.path_orders, "path_orders"),
        oracle.path_orders,
    )
    _assert_float_array_equal(
        _require_array(canonical.path_delays, "path_delays"),
        oracle.path_delays_ns,
    )
    _assert_float_array_equal(
        _require_array(canonical.path_losses, "path_losses"),
        oracle.path_losses_db,
    )
    np.testing.assert_array_equal(
        _require_array(canonical.path_delay_is_estimated, "path_delay_is_estimated"),
        oracle.path_delay_is_estimated,
    )
    np.testing.assert_array_equal(
        _require_array(canonical.path_loss_is_estimated, "path_loss_is_estimated"),
        oracle.path_loss_is_estimated,
    )
    np.testing.assert_array_equal(
        _require_array(canonical.path_tx, "path_tx"),
        oracle.path_tx,
    )
    np.testing.assert_array_equal(
        _require_array(canonical.path_rx, "path_rx"),
        oracle.path_rx,
    )
    np.testing.assert_array_equal(
        _require_array(canonical.segment_start_indices, "segment_start_indices"),
        oracle.lines[:, 0],
    )
    np.testing.assert_array_equal(
        _require_array(canonical.segment_end_indices, "segment_end_indices"),
        oracle.lines[:, 1],
    )
    np.testing.assert_array_equal(
        _require_array(canonical.segment_itype, "segment_itype"),
        oracle.segment_interactions,
    )
    np.testing.assert_array_equal(
        _require_array(canonical.segment_path_id, "segment_path_id"),
        oracle.segment_path_ids,
    )

    if oracle.points.size:
        _assert_float_array_equal(
            _require_array(canonical.aoa_az, "aoa_az"),
            oracle.aoa_az_deg,
        )
        _assert_float_array_equal(
            _require_array(canonical.aoa_el, "aoa_el"),
            oracle.aoa_el_deg,
        )
        _assert_float_array_equal(
            _require_array(canonical.aod_az, "aod_az"),
            oracle.aod_az_deg,
        )
        _assert_float_array_equal(
            _require_array(canonical.aod_el, "aod_el"),
            oracle.aod_el_deg,
        )

        point_material_ids = _require_array(canonical.material_ids, "material_ids")
        assert (
            _material_names_from_ids(point_material_ids, canonical.material_id_to_name)
            == oracle.point_material_names
        )
        segment_material_ids = _require_array(
            canonical.segment_material_ids,
            "segment_material_ids",
        )
        assert (
            _material_names_from_ids(
                segment_material_ids,
                canonical.material_id_to_name,
            )
            == oracle.segment_material_names
        )
    else:
        assert canonical.points.shape == (0, 3)
        assert canonical.lines.shape == (0, 2)


def assert_render_packet_matches_semantics(
    packet: Any,
    variant: FrameVariant | SemanticMPCFrame = "baseline",
) -> None:
    """Assert a full, unfiltered renderer packet against semantic geometry."""
    oracle = build_semantic_oracle(variant)

    _assert_float_array_equal(packet.mpc_points, oracle.points)
    np.testing.assert_array_equal(packet.mpc_lines, oracle.lines)
    assert packet.mpc_colors.shape == (len(oracle.lines), 3)
    assert np.isfinite(packet.mpc_colors).all()

    if len(oracle.lines):
        np.testing.assert_array_equal(
            _require_array(packet.mpc_line_itypes, "mpc_line_itypes"),
            oracle.segment_interactions,
        )
        np.testing.assert_array_equal(
            _require_array(packet.segment_mask, "segment_mask"),
            np.ones(len(oracle.lines), dtype=bool),
        )
    else:
        assert packet.mpc_line_itypes is None
        assert packet.segment_mask is not None
        assert not np.asarray(packet.segment_mask).any()

    _assert_float_array_equal(packet.mpc_bounce_points, oracle.bounce_points)
    if len(oracle.bounce_interactions):
        np.testing.assert_array_equal(
            _require_array(packet.mpc_bounce_itypes, "mpc_bounce_itypes"),
            oracle.bounce_interactions,
        )
    else:
        assert packet.mpc_bounce_itypes is None

    np.testing.assert_array_equal(
        _require_array(packet.path_mask, "path_mask"),
        np.ones(len(oracle.path_orders), dtype=bool),
    )
    assert packet.mpc_points.flags.writeable is False
    assert packet.mpc_lines.flags.writeable is False
    assert packet.mpc_colors.flags.writeable is False
    assert_canonical_matches_semantics(packet.canonical_data, variant)


def assert_renderer_neutral_packets_equal(
    expected: Any,
    actual: Any,
) -> None:
    """Compare renderer-owned MPC payloads while ignoring object identities."""
    float_fields = (
        "mpc_points",
        "mpc_colors",
        "mpc_bounce_points",
        "mpc_bounce_colors",
    )
    exact_fields = (
        "mpc_lines",
        "mpc_bounce_itypes",
        "mpc_line_itypes",
        "path_mask",
        "segment_mask",
    )

    for field_name in float_fields:
        expected_value = getattr(expected, field_name)
        actual_value = getattr(actual, field_name)
        if expected_value is None or actual_value is None:
            assert actual_value is expected_value, field_name
        else:
            _assert_float_array_equal(actual_value, np.asarray(expected_value))

    for field_name in exact_fields:
        expected_value = getattr(expected, field_name)
        actual_value = getattr(actual, field_name)
        if expected_value is None or actual_value is None:
            assert actual_value is expected_value, field_name
        else:
            np.testing.assert_array_equal(actual_value, expected_value)

    assert actual.mpc_visibility == expected.mpc_visibility
    assert actual.colorbar == expected.colorbar
    assert actual.stats_text == expected.stats_text
