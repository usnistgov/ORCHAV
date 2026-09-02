"""Synthetic canonical MPC data shared by focused Explorer unit tests."""

from __future__ import annotations

import numpy as np

from visualizer.src.metrics.mpc_canon import CanonicalStepData
from visualizer.src.metrics.mpc_path_catalog import MpcPathCatalog


def make_test_catalog(*, validate: bool = True) -> MpcPathCatalog:
    """Return a small catalog covering LoS, pure, mixed, and unknown paths."""
    point_groups = (
        ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (5.0, 1.0, 0.0), (10.0, 0.0, 0.0)),
        (
            (0.0, 0.0, 0.0),
            (3.0, 1.0, 0.0),
            (7.0, -1.0, 0.0),
            (10.0, 0.0, 0.0),
        ),
        ((0.0, 0.0, 0.0), (5.0, 2.0, 0.0), (10.0, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), (5.0, 0.0, 0.0), (9.0, 0.0, 0.0)),
        (
            (1.0, 0.0, 0.0),
            (3.0, 2.0, 0.0),
            (6.0, 2.0, 0.0),
            (9.0, 0.0, 0.0),
        ),
        ((1.0, 0.0, 0.0), (5.0, -2.0, 0.0), (9.0, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), (5.0, 2.0, 0.0), (9.0, 0.0, 0.0)),
    )
    interaction_groups = (
        (),
        (1,),
        (1, 2),
        (2,),
        (4,),
        (8, 1),
        (99,),
        (3,),
    )
    material_groups = (
        (),
        (1,),
        (1, 2),
        (2,),
        (3,),
        (4, 1),
        (0,),
        (5,),
    )
    tx = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int16)
    rx = np.array([0, 0, 0, 1, 0, 0, 1, 1], dtype=np.int16)
    delays = np.array([10.0, 20.0, np.nan, 5.0, 15.0, 12.0, 30.0, 30.0], dtype=np.float32)
    losses = np.array([60.0, 50.0, 50.0, 70.0, 40.0, np.nan, 80.0, 80.0], dtype=np.float32)

    starts: list[int] = []
    point_path_ids: list[int] = []
    point_interactions: list[int] = []
    point_materials: list[int] = []
    lines: list[tuple[int, int]] = []
    segment_path_ids: list[int] = []
    segment_interactions: list[int] = []
    offset = 0
    for path_id, (points, interactions, materials) in enumerate(
        zip(point_groups, interaction_groups, material_groups)
    ):
        starts.append(offset)
        point_path_ids.extend([path_id] * len(points))
        point_interactions.extend((0, *interactions, 0))
        point_materials.extend((0, *materials, 0))
        for local_index in range(len(points) - 1):
            lines.append((offset + local_index, offset + local_index + 1))
            segment_path_ids.append(path_id)
            if interactions:
                segment_interactions.append(interactions[max(local_index - 1, 0)])
            else:
                segment_interactions.append(0)
        offset += len(points)

    points = np.concatenate(
        [np.asarray(group, dtype=np.float32) for group in point_groups],
        axis=0,
    )
    path_ids = np.asarray(point_path_ids, dtype=np.int32)
    itype = np.asarray(point_interactions, dtype=np.uint8)
    orders = np.asarray([len(group) for group in interaction_groups], dtype=np.uint8)
    point_counts = np.asarray([len(group) for group in point_groups], dtype=np.int32)
    canonical = CanonicalStepData(
        points=points,
        lines=np.asarray(lines, dtype=np.int32),
        order=np.repeat(orders, point_counts),
        itype=itype,
        delay=np.repeat(delays, point_counts),
        loss=np.repeat(losses, point_counts),
        tx_id=np.repeat(tx, point_counts),
        rx_id=np.repeat(rx, point_counts),
        path_id=path_ids,
        path_start_indices=np.asarray(starts, dtype=np.int32),
        path_orders=orders,
        path_delays=delays,
        path_losses=losses,
        path_tx=tx,
        path_rx=rx,
        path_delay_is_estimated=np.array(
            [False, False, False, True, False, True, False, False],
            dtype=bool,
        ),
        path_loss_is_estimated=np.array(
            [False, False, True, False, False, True, False, False],
            dtype=bool,
        ),
        path_aoa_az=np.array(
            [0.0, np.nan, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            dtype=np.float32,
        ),
        path_aoa_el=np.arange(8, dtype=np.float32),
        path_aod_az=np.arange(10, 18, dtype=np.float32),
        path_aod_el=np.arange(-4, 4, dtype=np.float32),
        segment_path_id=np.asarray(segment_path_ids, dtype=np.int32),
        segment_itype=np.asarray(segment_interactions, dtype=np.uint8),
        material_ids=np.asarray(point_materials, dtype=np.int16),
        material_id_to_name={
            0: "",
            1: "Concrete",
            2: "Glass",
            3: "Metal",
            4: "Wood",
            5: "Brick",
        },
    )
    filtered = np.array([True, True, False, True, True, True, False, True], dtype=bool)
    rendered_segments = np.isin(canonical.segment_path_id, (0, 2, 3))
    return MpcPathCatalog(
        canonical,
        filtered_path_mask=filtered,
        rendered_segment_mask=rendered_segments,
        validate=validate,
    )


def make_large_los_catalog(
    path_count: int,
    *,
    unique_pairs: bool = False,
) -> MpcPathCatalog:
    """Return a compact many-path catalog for fetch/scaling tests."""
    count = int(path_count)
    starts = np.arange(count, dtype=np.int32) * 2
    points = np.zeros((count * 2, 3), dtype=np.float32)
    points[1::2, 0] = 1.0
    lines = np.column_stack((starts, starts + 1)).astype(np.int32, copy=False)
    path_ids = np.arange(count, dtype=np.int32)
    per_path_float = np.arange(count, dtype=np.float32)
    if unique_pairs and count > np.iinfo(np.int16).max:
        raise ValueError("unique-pair fixture exceeds canonical int16 TX ID range")
    per_path_ids = (
        np.arange(count, dtype=np.int32) if unique_pairs else np.arange(count, dtype=np.int32) % 256
    )
    canonical = CanonicalStepData(
        points=points,
        lines=lines,
        order=np.zeros((count * 2,), dtype=np.uint8),
        itype=np.zeros((count * 2,), dtype=np.uint8),
        delay=np.repeat(per_path_float, 2),
        loss=np.repeat(per_path_float, 2),
        tx_id=np.repeat(per_path_ids.astype(np.int16), 2),
        rx_id=np.zeros((count * 2,), dtype=np.int16),
        path_id=np.repeat(path_ids, 2),
        path_start_indices=starts,
        path_orders=np.zeros((count,), dtype=np.uint8),
        path_delays=per_path_float,
        path_losses=per_path_float,
        path_tx=per_path_ids.astype(np.int16),
        path_rx=np.zeros((count,), dtype=np.int16),
        segment_path_id=path_ids,
        segment_itype=np.zeros((count,), dtype=np.uint8),
    )
    return MpcPathCatalog(canonical)
