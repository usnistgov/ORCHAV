from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from generator.io.frames.builder import _extract_optional_path_metrics


class _ArrayHolder:
    def __init__(self, array: np.ndarray):
        self._array = np.asarray(array)

    def numpy(self) -> np.ndarray:
        return self._array


def test_extract_optional_path_metrics_supports_frozen_paths_without_cir() -> None:
    valid = np.array([[[True, False]]], dtype=bool)
    tau = _ArrayHolder(np.array([[[1.5e-9, 2.5e-9]]], dtype=np.float32))
    a = _ArrayHolder(
        np.array(
            [[[[[[1.0 + 0.0j], [0.25 + 0.0j]]]]]],
            dtype=np.complex64,
        )
    )
    theta = _ArrayHolder(np.array([[[np.pi / 2, np.pi / 3]]], dtype=np.float32))
    phi = _ArrayHolder(np.array([[[0.0, np.pi / 4]]], dtype=np.float32))
    paths = SimpleNamespace(
        _is_frozen_paths=True,
        tau=tau,
        a=a,
        theta_r=theta,
        phi_r=phi,
        theta_t=theta,
        phi_t=phi,
    )
    simulation_config = SimpleNamespace(export_path_metrics=True)

    metrics = _extract_optional_path_metrics(
        paths, valid, num_tx=1, num_rx=1, simulation_config=simulation_config
    )

    assert metrics is not None
    np.testing.assert_allclose(metrics["all_pair_delays_ns"][0], np.array([1.5], dtype=np.float32))
    np.testing.assert_allclose(
        metrics["all_pair_path_loss_db"][0], np.array([0.0], dtype=np.float32)
    )
    np.testing.assert_allclose(metrics["all_pair_aoa_az_deg"][0], np.array([0.0], dtype=np.float32))
    np.testing.assert_allclose(metrics["all_pair_aoa_el_deg"][0], np.array([0.0], dtype=np.float32))


def test_extract_optional_path_metrics_supports_frozen_path_coefficients_without_time_axis() -> (
    None
):
    num_paths = 218
    valid = np.zeros((1, 1, num_paths), dtype=bool)
    valid[0, 0, :3] = True

    tau = _ArrayHolder(
        np.linspace(1.0e-9, 3.0e-9, num_paths, dtype=np.float32).reshape(1, 1, num_paths)
    )
    a = _ArrayHolder(np.ones((1, 1, 1, 1, num_paths), dtype=np.complex64))
    theta = _ArrayHolder(np.full((1, 1, 1, 1, num_paths), np.pi / 2, dtype=np.float32))
    phi = _ArrayHolder(np.zeros((1, 1, 1, 1, num_paths), dtype=np.float32))
    paths = SimpleNamespace(
        _is_frozen_paths=True,
        tau=tau,
        a=a,
        theta_r=theta,
        phi_r=phi,
        theta_t=theta,
        phi_t=phi,
    )
    simulation_config = SimpleNamespace(export_path_metrics=True)

    metrics = _extract_optional_path_metrics(
        paths,
        valid,
        num_tx=1,
        num_rx=1,
        simulation_config=simulation_config,
    )

    assert metrics is not None
    np.testing.assert_allclose(
        metrics["all_pair_path_loss_db"][0],
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )


def test_extract_optional_path_metrics_keeps_float32_azimuths_below_360() -> None:
    valid = np.array([[[True, True]]], dtype=bool)
    tau = _ArrayHolder(np.ones((1, 1, 2), dtype=np.float32) * 1.0e-9)
    a = _ArrayHolder(np.ones((1, 1, 1, 1, 2), dtype=np.complex64))
    theta = _ArrayHolder(np.full((1, 1, 2), np.pi / 2, dtype=np.float32))
    phi = _ArrayHolder(np.array([[[0.0, -1.0e-8]]], dtype=np.float32))
    paths = SimpleNamespace(
        _is_frozen_paths=True,
        tau=tau,
        a=a,
        theta_r=theta,
        phi_r=phi,
        theta_t=theta,
        phi_t=phi,
    )
    simulation_config = SimpleNamespace(export_path_metrics=True)

    metrics = _extract_optional_path_metrics(
        paths,
        valid,
        num_tx=1,
        num_rx=1,
        simulation_config=simulation_config,
    )

    assert metrics is not None
    expected = np.array([0.0, 0.0], dtype=np.float32)
    for field in ("all_pair_aoa_az_deg", "all_pair_aod_az_deg"):
        azimuth = metrics[field][0]
        np.testing.assert_array_equal(azimuth, expected)
        assert np.all((azimuth >= 0.0) & (azimuth < 360.0))
