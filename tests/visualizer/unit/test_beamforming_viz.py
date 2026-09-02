"""Tests for beamforming visualization mesh generation."""

from __future__ import annotations

import numpy as np
import pytest

from visualizer.src.beamforming.visualization import (
    MAX_BEAM_PATTERN_WORK_ITEMS,
    bounded_pattern_sample_counts,
    generate_beamforming_mesh,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_weights(n: int) -> np.ndarray:
    """Return uniform complex weights for *n* elements."""
    return np.ones(n, dtype=np.complex128) / np.sqrt(n)


def _simple_positions(n: int, spacing: float = 0.005) -> np.ndarray:
    """Return a 1-D linear array along Y axis."""
    y = (np.arange(n, dtype=np.float64) - 0.5 * (n - 1)) * spacing
    return np.column_stack([np.zeros(n), y, np.zeros(n)])


# ---------------------------------------------------------------------------
# Triangulation correctness
# ---------------------------------------------------------------------------


class TestTriangulation:
    """Verify vectorized triangulation produces correct mesh topology."""

    @pytest.mark.parametrize(
        "n_az, n_el",
        [(12, 9), (72, 37), (360, 181), (720, 361)],
        ids=["min", "default", "medium", "max"],
    )
    def test_triangle_count(self, n_az: int, n_el: int):
        w = _uniform_weights(4)
        pos = _simple_positions(4)
        v, t, c = generate_beamforming_mesh(
            w,
            pos,
            28e9,
            azimuth_samples=n_az,
            elevation_samples=n_el,
        )
        expected_verts = n_az * n_el
        expected_tris = 2 * (n_el - 1) * n_az
        assert v.shape == (expected_verts, 3)
        assert t.shape == (expected_tris, 3)
        assert c.shape == (expected_verts, 3)

    def test_large_frame_array_sampling_is_bounded_before_mesh_allocation(self):
        weights = np.ones(64 * 64, dtype=np.complex128)

        azimuth, elevation = bounded_pattern_sample_counts(weights, 180, 91)

        assert len(weights) * azimuth * elevation <= MAX_BEAM_PATTERN_WORK_ITEMS
        assert azimuth == 180
        assert elevation < 91

    def test_sampling_bound_does_not_increase_requested_resolution(self):
        weights = np.ones(4, dtype=np.complex128)

        assert bounded_pattern_sample_counts(weights, 8, 5) == (8, 5)

    def test_triangle_indices_in_range(self):
        w = _uniform_weights(4)
        pos = _simple_positions(4)
        v, t, _ = generate_beamforming_mesh(w, pos, 28e9)
        assert (t >= 0).all()
        assert (t < v.shape[0]).all()

    def test_first_quad_indices(self):
        """First quad (i=0, j=0) should connect vertices 0,1,n_az,n_az+1."""
        n_az, n_el = 72, 37
        w = _uniform_weights(4)
        pos = _simple_positions(4)
        _, t, _ = generate_beamforming_mesh(
            w,
            pos,
            28e9,
            azimuth_samples=n_az,
            elevation_samples=n_el,
        )
        assert list(t[0]) == [0, n_az, 1]
        assert list(t[1]) == [1, n_az, n_az + 1]

    def test_azimuth_wraparound(self):
        """Last azimuth column should wrap to column 0."""
        n_az, n_el = 72, 37
        w = _uniform_weights(4)
        pos = _simple_positions(4)
        _, t, _ = generate_beamforming_mesh(
            w,
            pos,
            28e9,
            azimuth_samples=n_az,
            elevation_samples=n_el,
        )
        # Last quad in first elevation band: i=0, j=71 -> nj=0
        last_pair_idx = 2 * (n_az - 1)  # index of first triangle of last azimuth
        idx0 = 0 * n_az + (n_az - 1)  # 71
        idx1 = 0 * n_az + 0  # 0 (wraparound)
        idx2 = 1 * n_az + (n_az - 1)  # 143
        idx3 = 1 * n_az + 0  # 72
        assert list(t[last_pair_idx]) == [idx0, idx2, idx1]
        assert list(t[last_pair_idx + 1]) == [idx1, idx2, idx3]


# ---------------------------------------------------------------------------
# Geometry correctness
# ---------------------------------------------------------------------------


class TestGeometry:
    """Verify mesh geometry properties."""

    def test_origin_translation(self):
        w = _uniform_weights(4)
        pos = _simple_positions(4)
        origin = np.array([100.0, 200.0, 300.0])
        v, _, _ = generate_beamforming_mesh(w, pos, 28e9, origin=origin)
        # All vertices should be near the origin
        centroid = v.mean(axis=0)
        assert np.linalg.norm(centroid - origin) < 5.0

    def test_orientation_rotates_vertices(self):
        w = _uniform_weights(4)
        pos = _simple_positions(4)
        v0, _, _ = generate_beamforming_mesh(w, pos, 28e9)
        v1, _, _ = generate_beamforming_mesh(
            w,
            pos,
            28e9,
            orientation_radians=(0.5, 0.3, 0.1),
        )
        assert not np.allclose(v0, v1)

    def test_orientation_preserves_topology(self):
        w = _uniform_weights(4)
        pos = _simple_positions(4)
        _, t0, _ = generate_beamforming_mesh(w, pos, 28e9)
        _, t1, _ = generate_beamforming_mesh(
            w,
            pos,
            28e9,
            orientation_radians=(0.5, 0.3, 0.1),
        )
        np.testing.assert_array_equal(t0, t1)

    def test_scale_factor(self):
        w = _uniform_weights(4)
        pos = _simple_positions(4)
        v1, _, _ = generate_beamforming_mesh(w, pos, 28e9, scale=1.0)
        v2, _, _ = generate_beamforming_mesh(w, pos, 28e9, scale=3.0)
        r1 = np.linalg.norm(v1, axis=1).max()
        r2 = np.linalg.norm(v2, axis=1).max()
        assert abs(r2 / r1 - 3.0) < 0.01


# ---------------------------------------------------------------------------
# Color mapping
# ---------------------------------------------------------------------------


class TestColors:
    """Verify amplitude-to-color mapping."""

    def test_colors_in_range(self):
        w = _uniform_weights(4)
        pos = _simple_positions(4)
        _, _, c = generate_beamforming_mesh(w, pos, 28e9)
        assert (c >= 0.0).all()
        assert (c <= 1.0).all()

    def test_colors_dtype(self):
        w = _uniform_weights(4)
        pos = _simple_positions(4)
        _, _, c = generate_beamforming_mesh(w, pos, 28e9)
        assert c.dtype == np.float64


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_weights_raises(self):
        with pytest.raises(ValueError, match="empty"):
            generate_beamforming_mesh(
                np.array([], dtype=complex),
                np.zeros((0, 3)),
                28e9,
            )

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="matching length"):
            generate_beamforming_mesh(
                np.ones(4, dtype=complex),
                np.zeros((3, 3)),  # 3 positions but 4 weights
                28e9,
            )

    def test_negative_frequency_raises(self):
        with pytest.raises(ValueError, match="positive"):
            generate_beamforming_mesh(
                np.ones(4, dtype=complex),
                np.zeros((4, 3)),
                -1.0,
            )

    def test_real_imag_weight_format(self):
        """Weights stored as (N, 2) real/imag should be converted correctly."""
        w_complex = np.array([1 + 2j, 3 + 4j], dtype=complex)
        w_stacked = np.column_stack([w_complex.real, w_complex.imag])
        pos = _simple_positions(2)

        v1, _, _ = generate_beamforming_mesh(w_complex, pos, 28e9)
        v2, _, _ = generate_beamforming_mesh(w_stacked, pos, 28e9)
        np.testing.assert_allclose(v1, v2)


# ---------------------------------------------------------------------------
# dB scale
# ---------------------------------------------------------------------------


class TestDbScale:
    def test_db_changes_vertex_positions(self):
        w = _uniform_weights(4)
        pos = _simple_positions(4)
        v_lin, _, _ = generate_beamforming_mesh(w, pos, 28e9, db_scale=False)
        v_db, _, _ = generate_beamforming_mesh(w, pos, 28e9, db_scale=True)
        assert not np.allclose(v_lin, v_db)

    def test_db_dynamic_range(self):
        w = _uniform_weights(4)
        pos = _simple_positions(4)
        v20, _, _ = generate_beamforming_mesh(w, pos, 28e9, db_scale=True, dynamic_range_db=20)
        v60, _, _ = generate_beamforming_mesh(w, pos, 28e9, db_scale=True, dynamic_range_db=60)
        # Wider range → dB floor further from peak → nulls get more compression
        # 20 dB clips nulls harder → smaller min radius
        r20_min = np.linalg.norm(v20, axis=1).min()
        r60_min = np.linalg.norm(v60, axis=1).min()
        assert r60_min > r20_min


# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------


class TestColormaps:
    @pytest.mark.parametrize("cmap", ["jet", "viridis", "hot", "coolwarm"])
    def test_colormap_produces_valid_colors(self, cmap: str):
        w = _uniform_weights(4)
        pos = _simple_positions(4)
        _, _, c = generate_beamforming_mesh(w, pos, 28e9, colormap=cmap)
        assert (c >= 0.0).all() and (c <= 1.0).all()

    def test_different_colormaps_differ(self):
        w = _uniform_weights(4)
        pos = _simple_positions(4)
        _, _, c_jet = generate_beamforming_mesh(w, pos, 28e9, colormap="jet")
        _, _, c_vir = generate_beamforming_mesh(w, pos, 28e9, colormap="viridis")
        assert not np.allclose(c_jet, c_vir)


# ---------------------------------------------------------------------------
# Element patterns
# ---------------------------------------------------------------------------


class TestElementPattern:
    @pytest.mark.parametrize("pat", ["isotropic", "dipole", "tr38901"])
    def test_pattern_produces_mesh(self, pat: str):
        w = _uniform_weights(4)
        pos = _simple_positions(4)
        v, t, c = generate_beamforming_mesh(w, pos, 28e9, element_pattern=pat)
        assert v.shape[0] > 0

    def test_dipole_differs_from_isotropic(self):
        w = _uniform_weights(4)
        pos = _simple_positions(4)
        v_iso, _, _ = generate_beamforming_mesh(w, pos, 28e9, element_pattern="isotropic")
        v_dip, _, _ = generate_beamforming_mesh(w, pos, 28e9, element_pattern="dipole")
        assert not np.allclose(v_iso, v_dip)

    def test_tr38901_differs_from_isotropic(self):
        w = _uniform_weights(4)
        pos = _simple_positions(4)
        v_iso, _, _ = generate_beamforming_mesh(w, pos, 28e9, element_pattern="isotropic")
        v_tr, _, _ = generate_beamforming_mesh(w, pos, 28e9, element_pattern="tr38901")
        assert not np.allclose(v_iso, v_tr)


# ---------------------------------------------------------------------------
# Pattern metrics
# ---------------------------------------------------------------------------


class TestPatternMetrics:
    def test_basic_metrics(self):
        from visualizer.src.beamforming.visualization import compute_pattern_metrics

        w = _uniform_weights(16)
        pos = _simple_positions(16, spacing=0.00536)
        m = compute_pattern_metrics(w, pos, 28e9)
        assert m["peak_gain_dbi"] > 0
        assert m["hpbw_az_deg"] > 0
        assert m["hpbw_el_deg"] > 0
        assert m["sll_db"] < 0

    def test_empty_weights(self):
        from visualizer.src.beamforming.visualization import compute_pattern_metrics

        m = compute_pattern_metrics(np.array([], dtype=complex), np.zeros((0, 3)), 28e9)
        assert m["peak_gain_dbi"] == 0.0
