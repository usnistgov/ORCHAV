"""Per-frame statistics from canonical MPC data.

The metrics service calls this on ViewModel canonical data. Basic statistics
are cheap enough for every frame; advanced products such as the power delay
profile, delay spread, and angular spread are computed only when the live
metrics window needs them. Optional path masks restrict results to currently
visible MPCs.
"""

from typing import Dict, Optional

import numpy as np

from shared.logging import get_logger
from shared.statistics import (
    FrameStats,
    compute_angular_spread,
    compute_binned_power_delay_profile,
    compute_delay_spread,
)


class MPCStatsComputer:
    """Computes per-frame statistics from canonical MPC data."""

    def __init__(self):
        """Initialize the metrics logger used for per-frame stat failures."""
        self.logger = get_logger("orchav.mpc_stats")

    @staticmethod
    def _path_count(canon_data) -> int:
        """Return the number of path-level records available in canonical data."""
        for name in ("path_orders", "path_delays", "path_losses", "path_tx", "path_rx"):
            arr = getattr(canon_data, name, None)
            if arr is not None and getattr(arr, "size", 0) > 0:
                return int(arr.shape[0])
        path_start_indices = getattr(canon_data, "path_start_indices", None)
        if path_start_indices is not None and getattr(path_start_indices, "size", 0) > 0:
            return int(path_start_indices.shape[0])
        path_id = getattr(canon_data, "path_id", None)
        if path_id is not None and getattr(path_id, "size", 0) > 0:
            return int(np.max(path_id)) + 1
        return 0

    @staticmethod
    def _valid_path_mask(path_mask: Optional[np.ndarray], path_count: int) -> Optional[np.ndarray]:
        """Return a bool path mask only when it matches canonical path count."""
        if path_mask is None:
            return None
        mask = np.asarray(path_mask, dtype=bool)
        if mask.shape != (path_count,):
            raise ValueError(f"path_mask must have shape ({path_count},), got {mask.shape}")
        return mask

    @staticmethod
    def _path_start_values(canon_data, point_field: str, path_count: int) -> Optional[np.ndarray]:
        """Read path-level values from point arrays at each canonical path start."""
        arr = getattr(canon_data, point_field, None)
        if arr is None or not hasattr(arr, "size") or arr.size == 0 or path_count <= 0:
            return None
        point_values = np.asarray(arr)
        starts = getattr(canon_data, "path_start_indices", None)
        if starts is not None and getattr(starts, "size", 0) >= path_count:
            start_idx = np.asarray(starts[:path_count], dtype=np.int64)
            valid = (start_idx >= 0) & (start_idx < point_values.shape[0])
            values = np.full(path_count, np.nan, dtype=float)
            values[valid] = point_values[start_idx[valid]]
            return values
        path_id = getattr(canon_data, "path_id", None)
        if path_id is not None and getattr(path_id, "size", 0) == point_values.shape[0]:
            values = np.full(path_count, np.nan, dtype=float)
            ids = np.asarray(path_id, dtype=np.int64)
            for path_idx in range(path_count):
                matches = np.flatnonzero(ids == path_idx)
                if matches.size:
                    values[path_idx] = point_values[matches[0]]
            return values
        return None

    def _path_array(
        self,
        canon_data,
        *,
        path_field: str,
        point_field: str,
        path_count: int,
    ) -> Optional[np.ndarray]:
        """Return one aligned value per path, preferring canonical path arrays."""
        if path_count <= 0:
            return None
        arr = getattr(canon_data, path_field, None)
        if arr is not None and getattr(arr, "size", 0) >= path_count:
            values = np.asarray(arr[:path_count], dtype=float)
        else:
            values = self._path_start_values(canon_data, point_field, path_count)
            if values is None:
                return None
            values = np.asarray(values, dtype=float)
        return values

    @staticmethod
    def _selection_mask(path_mask: Optional[np.ndarray], path_count: int) -> np.ndarray:
        """Return the caller's validated selection or an all-path selection."""
        if path_mask is None:
            return np.ones((path_count,), dtype=bool)
        return path_mask.copy()

    @staticmethod
    def _measured_mask(canon_data, flag_field: str, path_count: int) -> np.ndarray:
        """Exclude values explicitly marked as geometry-derived estimates."""
        flags = getattr(canon_data, flag_field, None)
        if flags is None or getattr(flags, "size", 0) < path_count:
            return np.ones((path_count,), dtype=bool)
        return ~np.asarray(flags[:path_count], dtype=bool)

    def _path_values(
        self,
        canon_data,
        *,
        path_field: str,
        point_field: str,
        estimated_flag_field: str,
        path_mask: Optional[np.ndarray],
        path_count: int,
    ) -> Optional[np.ndarray]:
        """Return selected, finite, non-estimated values for one RF metric."""
        values = self._path_array(
            canon_data,
            path_field=path_field,
            point_field=point_field,
            path_count=path_count,
        )
        if values is None:
            return None
        valid = self._selection_mask(path_mask, path_count)
        valid &= np.isfinite(values)
        valid &= self._measured_mask(canon_data, estimated_flag_field, path_count)
        selected = values[valid]
        return selected if selected.size else None

    def compute_frame_stats(
        self,
        canon_data,
        include_advanced: bool = False,
        path_mask: Optional[np.ndarray] = None,
    ) -> FrameStats:
        """Compute frame statistics from canonical MPC data.

        Args:
            canon_data: CanonicalStepData object containing MPC data.
            include_advanced: Whether to compute the 1 ns power-delay profile
                plus delay/angular spread.
            path_mask: Optional bool[P] mask selecting which paths to include.
                When *None*, all paths are used (unfiltered).

        Returns:
            FrameStats with computed statistics.
        """
        try:
            path_count = self._path_count(canon_data)
            valid_path_mask = self._valid_path_mask(path_mask, path_count)

            basic_stats = self._compute_basic_stats(canon_data, valid_path_mask, path_count)
            advanced_stats = {}
            if include_advanced:
                advanced_stats = self._compute_advanced_stats(
                    canon_data, valid_path_mask, path_count
                )
            return FrameStats(**{**basic_stats, **advanced_stats})
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            self.logger.error("Error computing frame stats: %s", exc)
            return FrameStats(
                total_paths=0, orders_hist={}, delay_range_ns=None, path_loss_range=None
            )

    def _compute_basic_stats(
        self,
        canon_data,
        path_mask: Optional[np.ndarray] = None,
        path_count: Optional[int] = None,
    ) -> Dict:
        """Compute path counts, order histogram, delay range, and loss range."""
        stats: Dict = {}
        if path_count is None:
            path_count = self._path_count(canon_data)

        po = getattr(canon_data, "path_orders", None)
        if po is not None and po.size > 0:
            filtered_po = po[:path_count][path_mask] if path_mask is not None else po[:path_count]
            stats["total_paths"] = int(filtered_po.shape[0])
        elif path_mask is not None:
            stats["total_paths"] = int(np.count_nonzero(path_mask))
        else:
            psi = getattr(canon_data, "path_start_indices", None)
            if psi is not None and psi.size > 0:
                stats["total_paths"] = int(path_count or psi.shape[0])
            elif hasattr(canon_data, "lines") and canon_data.lines.size > 0:
                n_pts = canon_data.points.shape[0]
                is_start = np.ones(n_pts, dtype=bool)
                is_start[canon_data.lines[:, 1]] = False
                stats["total_paths"] = int(np.sum(is_start))
            else:
                stats["total_paths"] = 0

        if po is not None and po.size > 0:
            filtered_po = po[:path_count][path_mask] if path_mask is not None else po[:path_count]
            if filtered_po.size > 0:
                unique, counts = np.unique(filtered_po, return_counts=True)
                stats["orders_hist"] = dict(zip(unique.astype(int), counts.astype(int)))
            else:
                stats["orders_hist"] = {}
        else:
            stats["orders_hist"] = {}

        delay = self._path_values(
            canon_data,
            path_field="path_delays",
            point_field="delay",
            estimated_flag_field="path_delay_is_estimated",
            path_mask=path_mask,
            path_count=path_count,
        )
        stats["delay_range_ns"] = (
            (float(np.min(delay)), float(np.max(delay))) if delay is not None else None
        )

        loss = self._path_values(
            canon_data,
            path_field="path_losses",
            point_field="loss",
            estimated_flag_field="path_loss_is_estimated",
            path_mask=path_mask,
            path_count=path_count,
        )
        stats["path_loss_range"] = (
            (float(np.min(loss)), float(np.max(loss))) if loss is not None else None
        )

        return stats

    def _compute_advanced_stats(
        self,
        canon_data,
        path_mask: Optional[np.ndarray] = None,
        path_count: Optional[int] = None,
    ) -> Dict:
        """Compute channel products that are needed only for the metrics window."""
        stats: Dict = {}
        if path_count is None:
            path_count = self._path_count(canon_data)
        try:
            delays_ns = self._path_array(
                canon_data,
                path_field="path_delays",
                point_field="delay",
                path_count=path_count,
            )
            path_loss_db = self._path_array(
                canon_data,
                path_field="path_losses",
                point_field="loss",
                path_count=path_count,
            )
            if path_loss_db is None:
                return stats

            loss_valid = self._selection_mask(path_mask, path_count)
            loss_valid &= np.isfinite(path_loss_db)
            loss_valid &= self._measured_mask(canon_data, "path_loss_is_estimated", path_count)
            if not np.any(loss_valid):
                return stats

            if delays_ns is not None:
                delay_valid = loss_valid & np.isfinite(delays_ns)
                delay_valid &= self._measured_mask(
                    canon_data, "path_delay_is_estimated", path_count
                )
                selected_delays = delays_ns[delay_valid]
                selected_losses = path_loss_db[delay_valid]
                with np.errstate(over="ignore", under="ignore"):
                    delay_gains = 10.0 ** (-selected_losses / 10.0)
                usable_delay_power = np.isfinite(delay_gains) & (delay_gains > 0.0)
                if np.any(usable_delay_power):
                    selected_delays = selected_delays[usable_delay_power]
                    delay_gains = delay_gains[usable_delay_power]
                    stats["delay_spread_ns"] = compute_delay_spread(selected_delays, delay_gains)
                    stats["binned_power_delay_profile"] = compute_binned_power_delay_profile(
                        selected_delays, delay_gains
                    )

            aod_az = self._path_array(
                canon_data,
                path_field="path_aod_az",
                point_field="aod_az",
                path_count=path_count,
            )
            if aod_az is not None:
                angle_valid = loss_valid & np.isfinite(aod_az)
                selected_aod = aod_az[angle_valid]
                selected_losses = path_loss_db[angle_valid]
                with np.errstate(over="ignore", under="ignore"):
                    angle_gains = 10.0 ** (-selected_losses / 10.0)
                usable_angle_power = np.isfinite(angle_gains) & (angle_gains > 0.0)
                if np.any(usable_angle_power):
                    stats["angular_spread_deg"] = compute_angular_spread(
                        selected_aod[usable_angle_power], angle_gains[usable_angle_power]
                    )

        except (ValueError, TypeError, ZeroDivisionError) as exc:
            self.logger.warning("Error computing advanced stats: %s", exc)

        return stats
