"""Pair selection and bounded, phase-free delay/time summaries of saved paths."""

from __future__ import annotations

import numpy as np

from shared.frames import PATH_METRIC_ARRAY_FIELDS, ProjectedMPCFrame


def select_statistics_pair(frame: ProjectedMPCFrame, pair: tuple[int, int]) -> ProjectedMPCFrame:
    """View one pair's paths, preserving original device IDs and validity bits.

    An absent pair remains absent, rather than becoming a represented zero-path
    link. Only the components consumed by the statistics accumulator are copied.
    Metric and interaction vectors are slices of the provider's projection.
    """
    matches = np.flatnonzero(np.all(frame.tx_rx_pairs == pair, axis=1))
    if matches.size:
        row = int(matches[0])
        first, last = map(int, frame.pair_path_offsets[row : row + 2])
        pairs = frame.tx_rx_pairs[row : row + 1]
        offsets = np.asarray([0, last - first], dtype=np.int64)
    else:
        first = last = 0
        pairs = np.empty((0, 2), dtype=np.int32)
        offsets = np.zeros(1, dtype=np.int64)
    bounce_first, bounce_last = map(int, frame.bounce_offsets[[first, last]])
    metrics = {
        field: getattr(frame, field)[first:last] for field in PATH_METRIC_ARRAY_FIELDS.values()
    }
    return ProjectedMPCFrame(
        frame_index=frame.frame_index,
        timestamp_s=frame.timestamp_s,
        tx_rx_pairs=pairs,
        pair_path_offsets=offsets,
        bounce_offsets=frame.bounce_offsets[first : last + 1] - bounce_first,
        interactions=frame.interactions[bounce_first:bounce_last],
        metric_valid_bits=frame.metric_valid_bits[first:last],
        **metrics,
    )


def statistics_coordinates(frame_indices, timestamps):
    """Use actual seconds only when every timestamp is finite and increasing."""
    frames = np.asarray(frame_indices, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    if times.size == frames.size and times.size and np.all(np.isfinite(times)):
        if np.all(np.diff(times) > 0):
            return times, "Time (s)"
    return frames, "Frame"


def build_delay_time_heatmap(frames, timestamps, slots, delays, losses):
    """Bin co-valid path powers; bound the display to 512 × 512 cells.

    Delay bins sum *incoherent* path gains. If multiple recorded observations
    share a time cell, average their power sums before converting to dB. Empty
    observations do not contribute: the saved topology alone cannot establish
    whether a solver ran. Empty cells remain NaN, with no invented power floor.
    No raw observation is changed or discarded from the other statistics.
    """
    coordinates, axis = statistics_coordinates(frames, timestamps)
    empty = {
        "delay_time_power_db": np.empty((0, 0), dtype=np.float64),
        "delay_time_x_edges": np.empty(0, dtype=np.float64),
        "delay_time_delay_edges_ns": np.empty(0, dtype=np.float64),
        "delay_time_observation_counts": np.empty(0, dtype=np.int64),
        "delay_time_axis": axis,
    }
    if not delays:
        return empty
    unique = np.unique(coordinates)
    step = float(np.min(np.diff(unique))) if len(unique) > 1 else 1.0
    low, high = float(unique[0] - step / 2), float(unique[-1] + step / 2)
    time_bins = min(512, max(1, int(round((high - low) / step))))
    x_edges = np.linspace(low, high, time_bins + 1)
    delay_low = float(np.floor(min(np.min(values) for values in delays)))
    delay_high = float(np.floor(max(np.max(values) for values in delays)) + 1)
    delay_bins = min(512, max(1, int(np.ceil(delay_high - delay_low))))
    delay_edges = np.linspace(delay_low, delay_high, delay_bins + 1)
    # A fixed reference prevents underflow for uniformly weak channels without
    # peak-normalizing the published values. Restore the absolute dB ratio below.
    reference_loss = float(min(np.min(values) for values in losses))
    power = np.zeros((time_bins, delay_bins), dtype=np.float64)
    counts = np.zeros(time_bins, dtype=np.int64)
    for slot, path_delays, path_losses in zip(slots, delays, losses, strict=True):
        row = int(
            np.clip(np.searchsorted(x_edges, coordinates[slot], side="right") - 1, 0, time_bins - 1)
        )
        weights = 10.0 ** (-(path_losses.astype(np.float64) - reference_loss) / 10.0)
        bins = np.clip(
            np.searchsorted(delay_edges, path_delays, side="right") - 1, 0, delay_bins - 1
        )
        power[row] += np.bincount(bins, weights=weights, minlength=delay_bins)
        counts[row] += 1
    power /= np.maximum(counts[:, None], 1)
    db = np.full(power.shape, np.nan)
    nonzero = power > 0
    db[nonzero] = 10 * np.log10(power[nonzero]) - reference_loss
    return {
        "delay_time_power_db": db,
        "delay_time_x_edges": x_edges,
        "delay_time_delay_edges_ns": delay_edges,
        "delay_time_observation_counts": counts,
        "delay_time_axis": axis,
    }
