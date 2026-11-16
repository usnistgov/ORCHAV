"""Style samples from the existing geometric clock, never field physics."""

from dataclasses import dataclass

import numpy as np

from .path_analysis import SPEED_OF_LIGHT_M_PER_NS, GeometricProgress, geometric_progress


def cinematic_subset(catalog, path_ids, limit):
    """Rank exported finite losses first, then canonical IDs for missing evidence."""
    if limit not in (None, 10, 50, 100):
        raise ValueError("Cinematic path limit must be All, 10, 50 or 100")
    ids = np.unique(np.asarray(path_ids, dtype=np.int32))
    if np.any(ids < 0) or np.any(ids >= catalog.path_count):
        raise ValueError("Invalid cinematic path ID")
    losses = np.asarray(catalog.path_losses_db[ids])
    exported = (catalog.path_loss_provenance_codes[ids] == 0) & np.isfinite(losses)
    order = np.lexsort((ids, np.where(exported, losses, np.inf), ~exported))
    selected = ids[order[:limit]]
    note = (
        "Exported path-loss order (strongest first); missing/estimated loss uses canonical ID"
        if exported.any()
        else "Path loss unavailable: deterministic canonical-ID order"
    )
    return (
        selected,
        f"Animating {len(selected)}/{len(ids)} selected paths · {note}; full analytical population unchanged",
    )


@dataclass(frozen=True)
class PulseStyleSample:
    leading: np.ndarray
    leading_colors: np.ndarray
    trail: np.ndarray
    trail_colors: np.ndarray
    interactions: np.ndarray
    interaction_colors: np.ndarray


class CinematicPulseGeometry:
    """One frozen frame's selected path geometry; the service retains clock ownership."""

    TRAIL_SAMPLES = 10
    BATCH_PATHS = 100_000
    BATCH_POINTS = 2_000_000
    TRAIL_FRACTIONS = {"short": 0.025, "medium": 0.055, "long": 0.10}

    def __init__(self, catalog, ids):
        self.catalog = catalog
        self.ids = np.asarray(ids, dtype=np.int32)
        # Bound each temporary calculation, without truncating an All selection.
        counts = catalog.path_end_indices[self.ids] - catalog.path_start_indices[self.ids]
        if np.any(counts < 2) or np.any(counts > self.BATCH_POINTS):
            raise ValueError("Selected path exceeds the per-path geometry budget")
        self._batches = []
        start = 0
        while start < len(self.ids):
            stop = min(start + self.BATCH_PATHS, len(self.ids))
            size = np.searchsorted(
                np.cumsum(counts[start:stop], dtype=np.int64), self.BATCH_POINTS, side="right"
            )
            self._batches.append((self.ids[start : start + size], {}))
            start += int(size)
        if not self._batches:
            self._batches.append((self.ids, {}))
        first = self._progress(0.0)
        finite = first.geometric_arrival_ns[first.valid]
        self.extent = float(np.max(finite)) if len(finite) else 1.0
        self.trail_ns = self.extent * 0.055
        vertices, times = [], []
        path_ids, point_indices, interaction_types, orders, ordinals = [], [], [], [], []
        for path_id in self.ids:
            points = catalog.path_points(int(path_id))
            types = catalog.interaction_sequence(int(path_id))
            if not np.isfinite(points).all():
                continue
            cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            start = int(catalog.path_start_indices[int(path_id)])
            # Preserve recorded nonzero types, including explicitly virtual and
            # unknown points. Type zero is a non-interaction sentinel.
            for j, kind in enumerate(types, 1):
                if int(kind) != 0:
                    vertices.append(points[j])
                    times.append(cumulative[j] / SPEED_OF_LIGHT_M_PER_NS)
                    path_ids.append(int(path_id))
                    point_indices.append(start + j)
                    interaction_types.append(int(kind))
                    orders.append(int(catalog.canonical_data.order[start + j]))
                    ordinals.append(j)
        self.vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        self.interaction_ns = np.asarray(times)
        self.interaction_path_ids = np.asarray(path_ids, dtype=np.int32)
        self.interaction_point_indices = np.asarray(point_indices, dtype=np.int64)
        self.interaction_types = np.asarray(interaction_types, dtype=np.int32)
        self.interaction_orders = np.asarray(orders, dtype=np.int32)
        self.interaction_ordinals = np.asarray(ordinals, dtype=np.int32)
        for metadata in (
            self.interaction_path_ids,
            self.interaction_point_indices,
            self.interaction_types,
            self.interaction_orders,
            self.interaction_ordinals,
        ):
            metadata.setflags(write=False)
        self._interaction_steps = np.unique(self.interaction_ns)
        self.omitted_count = first.omitted_count

    def _progress(self, time_ns):
        samples = [
            geometric_progress(
                self.catalog,
                ids,
                time_ns,
                max_paths=self.BATCH_PATHS,
                max_points=self.BATCH_POINTS,
                _geometry_cache=cache,
            )
            for ids, cache in self._batches
        ]
        if len(samples) == 1:
            return samples[0]
        arrays = {
            field: np.concatenate([getattr(sample, field) for sample in samples])
            for field in (
                "path_ids",
                "positions",
                "geometric_arrival_ns",
                "valid",
                "arrived",
                "active",
            )
        }
        return GeometricProgress(
            time_ns=float(time_ns),
            **arrays,
            omitted_count=sum(sample.omitted_count for sample in samples),
            disclosures=tuple(note for sample in samples for note in sample.disclosures),
        )

    def interaction_time(self, time_ns, direction):
        """Return the next/previous known geometric interaction time, or None.

        Duplicate timestamps are one step. No recorded vertex or timestamp is
        fabricated at either boundary; the caller owns pause/seek behavior.
        """
        if not np.isfinite(time_ns):
            raise ValueError("Interaction stepping requires a finite geometric time")
        if isinstance(direction, (bool, np.bool_)) or direction not in (-1, 1):
            raise ValueError("Interaction stepping direction must be -1 or 1")
        if direction == 1:
            index = np.searchsorted(self._interaction_steps, time_ns, side="right")
        else:
            index = np.searchsorted(self._interaction_steps, time_ns, side="left") - 1
        if index < 0 or index >= len(self._interaction_steps):
            return None
        return float(self._interaction_steps[index])

    def sample(self, time_ns, *, interaction_mode="flash", trail_mode="medium", dwell_seconds=1.0):
        """Resample the SAME arc-length function for heads and short trailing times."""
        if interaction_mode not in ("flash", "reached", "all", "dwell"):
            raise ValueError("Unknown recorded interaction marker mode")
        if trail_mode not in self.TRAIL_FRACTIONS:
            raise ValueError("Unknown cinematic trail length")
        if (
            isinstance(dwell_seconds, (bool, np.bool_))
            or not np.isfinite(dwell_seconds)
            or dwell_seconds < 0
        ):
            raise ValueError("Dwell duration must be finite nonnegative presentation seconds at 1x")
        trail_ns = self.extent * self.TRAIL_FRACTIONS[trail_mode]
        heads = self._progress(time_ns)
        positions = np.nan_to_num(heads.positions).astype(np.float32)
        colors = np.tile([0.82, 0.96, 1.0, 1.0], (len(self.ids), 1)).astype(np.float32)
        colors[:, 3] = heads.valid * (
            (time_ns >= 0) & (time_ns <= heads.geometric_arrival_ns + trail_ns)
        )
        trails, alphas = [], []
        for fraction in np.linspace(0.1, 1.0, self.TRAIL_SAMPLES):
            moment = time_ns - trail_ns * fraction
            progress = self._progress(moment)
            trails.append(np.nan_to_num(progress.positions))
            alphas.append(
                progress.valid * (moment >= 0) * (~progress.arrived) * (1 - fraction) * 0.65
            )
        trail = np.concatenate(trails).astype(np.float32)
        trail_colors = np.tile([0.21, 0.72, 0.92, 1.0], (len(trail), 1)).astype(np.float32)
        trail_colors[:, 3] = np.concatenate(alphas)
        age = time_ns - self.interaction_ns
        alpha = np.where(age >= 0, np.maximum(0.0, 1.0 - age / max(self.extent * 0.035, 1e-9)), 0.0)
        flashes = np.tile([1.0, 0.79, 0.38, 1.0], (len(self.vertices), 1)).astype(np.float32)
        if interaction_mode == "dwell":
            # Presentation seconds at 1x map to fixed geometric-clock duration.
            # This makes backward scrubbing, pause and loop fully deterministic;
            # changing playback speed never alters which recorded times qualify.
            hold_ns = self.extent * float(dwell_seconds) / 5.0
            fade_age = np.maximum(0.0, age - hold_ns)
            alpha = (
                np.where(
                    age >= 0,
                    np.maximum(0.0, 1.0 - fade_age / max(self.extent * 0.035, 1e-9)),
                    0.0,
                )
                * 0.85
            )
        elif interaction_mode == "reached":
            # Derive from current geometric time, not playback history: reverse
            # scrubbing and restarting cannot leave future vertices highlighted.
            alpha = np.maximum(alpha * 0.85, (age >= 0) * 0.48)
        elif interaction_mode == "all":
            alpha = np.maximum(alpha * 0.85, 0.48)
        else:
            alpha *= 0.85
        flashes[:, 3] = alpha
        return PulseStyleSample(positions, colors, trail, trail_colors, self.vertices, flashes)
