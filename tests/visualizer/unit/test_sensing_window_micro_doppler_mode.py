from __future__ import annotations

import os
from copy import deepcopy
from types import SimpleNamespace
from uuid import UUID, uuid5

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QApplication

from shared.extensions.sensing import (
    SENSING_SCHEMA_VERSION,
    UnsupportedSensingSchemaError,
    build_sensing_temporal_contract,
    canonical_sensing_hash,
)
from shared.sensing.signal_processing import MicroDopplerResult
from shared.sensing.targets import build_target_catalog, build_target_state, build_target_truth
from tests.visualizer.fixtures.sensing_v3 import (
    build_visualizer_sensing_payload,
    materialize_visualizer_product_specs,
)
from visualizer.src.sensing.update_request import SensingUpdateRequest

_APP = QApplication.instance() or QApplication([])

try:
    from visualizer.src.sensing.sensing_window import SensingWindow

    HAS_SENSING_WINDOW = True
except ImportError:
    HAS_SENSING_WINDOW = False


def _timestamped_sensing_payload(
    products: dict,
    *,
    sample_times_s: list[float],
    strategy: str,
    acquisition_key: str,
    scene_frame_id: int = 0,
) -> dict:
    """Build a valid multi-sample fixture without changing the shared fixture."""
    typed_products = dict(products)
    legacy_cir = typed_products.pop("cir", None)
    legacy_slow_time = typed_products.pop("slow_time_matrix", None)
    range_size: int | None = None
    if legacy_cir is not None:
        cir = np.asarray(legacy_cir, dtype=np.complex128)
        range_size = int(cir.shape[-1])
        typed_products["cir_by_pair_rx_element"] = cir[np.newaxis, np.newaxis, :]
    if legacy_slow_time is not None:
        slow_time = np.asarray(legacy_slow_time, dtype=np.complex128)
        range_size = int(slow_time.shape[-1])
        typed_products["slow_time_by_pair_rx_element"] = slow_time[np.newaxis, np.newaxis, :, :]
    if range_size is not None:
        typed_products["range_axis_m"] = np.arange(range_size, dtype=np.float64) ** 2 * 0.125

    base = build_visualizer_sensing_payload(scene_frame_id=scene_frame_id)
    supplied_resolved = typed_products.pop("config_resolved", None)
    resolved = deepcopy(base["config_resolved"])
    if isinstance(supplied_resolved, dict):
        for key, value in supplied_resolved.items():
            if isinstance(value, dict) and isinstance(resolved.get(key), dict):
                resolved[key].update(deepcopy(value))
            else:
                resolved[key] = deepcopy(value)
    resolved.setdefault("processing", {})["strategy"] = strategy
    base_temporal = base["temporal"]
    clock = base_temporal["clock"]
    acquisition = base_temporal["acquisition"]
    result = base_temporal["result"]
    samples = [
        {
            "sample_index": index,
            "sample_time_s": sample_time_s,
            "valid": True,
            "phase_coherent": strategy == "coherent_cpi_fft",
            "source": {
                "source_generation_id": clock["clock_id"],
                "scene_frame_id": f"{scene_frame_id}:{acquisition_key}",
                "sample_offset": index,
            },
        }
        for index, sample_time_s in enumerate(sample_times_s)
    ]
    temporal = build_sensing_temporal_contract(
        clock_id=clock["clock_id"],
        acquisition_id=str(
            uuid5(
                UUID(acquisition["acquisition_id"]),
                f"{acquisition_key}:{strategy}:{sample_times_s!r}",
            )
        ),
        acquisition_config_id=canonical_sensing_hash(
            {"strategy": strategy, "sample_times_s": sample_times_s}
        ),
        pair_id=acquisition["pair_id"],
        tx_sensor_id=acquisition["tx_sensor_id"],
        rx_sensor_id=acquisition["rx_sensor_id"],
        samples=samples,
        processing_run_id=result["processing_run_id"],
        processing_recipe_id=result["processing_recipe_id"],
        reference_time_s=(sample_times_s[0] + sample_times_s[-1]) / 2.0,
        reference_time_policy="test_acquisition_center",
        display_scene_frame_id=scene_frame_id,
        display_policy="exact",
        display_reason="Timestamped visualizer test fixture.",
    )
    payload = {
        "schema_version": SENSING_SCHEMA_VERSION,
        "temporal": temporal,
        "processing_strategy": strategy,
        "config_resolved": resolved,
        "config": deepcopy(resolved),
        **typed_products,
    }
    materialize_visualizer_product_specs(payload)
    return payload


def test_micro_doppler_mode_auto_prefers_validated_stored_blind_product() -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    assert (
        SensingWindow._resolve_micro_doppler_mode(  # type: ignore[attr-defined]
            "auto",
            has_stored=True,
            can_track=True,
            can_collapse=True,
            can_fixed=True,
        )
        == "stored"
    )


def test_micro_doppler_mode_auto_never_selects_tracked_truth() -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    assert (
        SensingWindow._resolve_micro_doppler_mode(  # type: ignore[attr-defined]
            "auto",
            has_stored=False,
            can_track=True,
            can_collapse=True,
            can_fixed=True,
        )
        == "collapsed"
    )

    assert (
        SensingWindow._resolve_micro_doppler_mode(  # type: ignore[attr-defined]
            "auto",
            has_stored=False,
            can_track=True,
            can_collapse=False,
            can_fixed=False,
        )
        == "unavailable"
    )


def test_micro_doppler_mode_explicit_tracked_fails_closed() -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    assert (
        SensingWindow._resolve_micro_doppler_mode(  # type: ignore[attr-defined]
            "tracked",
            has_stored=False,
            can_track=False,
            can_collapse=True,
            can_fixed=True,
        )
        == "unavailable"
    )


def test_micro_doppler_mode_explicit_stored_fails_closed() -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    assert (
        SensingWindow._resolve_micro_doppler_mode(  # type: ignore[attr-defined]
            "stored",
            has_stored=False,
            can_track=False,
            can_collapse=False,
            can_fixed=True,
        )
        == "unavailable"
    )


def test_coherent_history_expands_exact_timestamped_rows_once(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    first_rows = np.arange(12, dtype=np.float64).reshape(3, 4).astype(np.complex128)
    first = _timestamped_sensing_payload(
        {"slow_time_matrix": first_rows},
        sample_times_s=[0.0, 0.001, 0.002],
        strategy="coherent_cpi_fft",
        acquisition_key="first",
    )
    first_request = SensingUpdateRequest(
        display_frame={"sensing": first},
        state_frame={},
        state_step=0,
    )

    history, times_s, error = window._resolve_timestamped_fast_time_history(
        first,
        first_request,
    )
    replay_history, replay_times_s, replay_error = window._resolve_timestamped_fast_time_history(
        first, first_request
    )

    assert error is None
    assert times_s == pytest.approx([0.0, 0.001, 0.002])
    assert replay_error is None
    assert replay_times_s == pytest.approx(times_s)
    assert len(history) == len(replay_history) == 3
    np.testing.assert_array_equal(history[-1], first_rows[-1])
    np.testing.assert_array_equal(window._fast_time_range_axis_m, first["range_axis_m"])

    second_rows = first_rows + 20.0
    second = _timestamped_sensing_payload(
        {
            "slow_time_matrix": second_rows,
            "config_resolved": {
                "acquisition": {
                    "sample_source": "coherent_one_solve",
                    "cpi_length_samples": 3,
                    "cpi_hop_samples": 3,
                },
                "processing": {
                    "mode": "offline_reprocess",
                    "strategy": "shared_sensing_postprocessor",
                },
                "source_processing": {
                    "mode": "coherent_cpi",
                    "strategy": "coherent_cpi_fft",
                },
            },
        },
        sample_times_s=[0.003, 0.004, 0.005],
        strategy="shared_sensing_postprocessor",
        acquisition_key="second",
    )
    for sample in second["temporal"]["acquisition"]["samples"]:
        sample["phase_coherent"] = True
    accumulated, accumulated_times_s, accumulated_error = (
        window._resolve_timestamped_fast_time_history(
            second,
            SensingUpdateRequest(
                display_frame={"sensing": second},
                state_frame={},
                state_step=1,
            ),
        )
    )

    assert accumulated_error is None
    assert accumulated_times_s == pytest.approx([0.0, 0.001, 0.002, 0.003, 0.004, 0.005])
    assert len(accumulated) == 6
    np.testing.assert_array_equal(accumulated[-1], second_rows[-1])
    window.close()


def test_coherent_history_restarts_from_valid_cpi_after_inter_cpi_gap(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    first_rows = np.arange(12, dtype=np.float64).reshape(3, 4).astype(np.complex128)
    first = _timestamped_sensing_payload(
        {"slow_time_matrix": first_rows},
        sample_times_s=[0.0, 0.001, 0.002],
        strategy="coherent_cpi_fft",
        acquisition_key="first-gap-cpi",
    )
    window._resolve_timestamped_fast_time_history(
        first,
        SensingUpdateRequest(
            display_frame={"sensing": first},
            state_frame={},
            state_step=0,
        ),
    )

    second_rows = first_rows + 20.0
    second = _timestamped_sensing_payload(
        {"slow_time_matrix": second_rows},
        sample_times_s=[0.003125, 0.004125, 0.005125],
        strategy="coherent_cpi_fft",
        acquisition_key="second-gap-cpi",
        scene_frame_id=1,
    )
    history, times_s, error = window._resolve_timestamped_fast_time_history(
        second,
        SensingUpdateRequest(
            display_frame={"sensing": second},
            state_frame={},
            state_step=1,
        ),
    )

    assert error is None
    assert times_s == pytest.approx([0.003125, 0.004125, 0.005125])
    assert len(history) == 3
    np.testing.assert_array_equal(history[-1], second_rows[-1])
    window.close()


def test_coherent_history_backward_seek_keeps_current_cpi_renderable(qtbot) -> None:
    window = SensingWindow()
    qtbot.addWidget(window)
    rows = np.arange(12, dtype=np.float64).reshape(3, 4).astype(np.complex128)
    for key, times in (
        ("later", [0.003, 0.004, 0.005]),
        ("earlier", [0.0, 0.001, 0.002]),
        ("earlier", [0.0, 0.001, 0.002]),
        ("later", [0.003, 0.004, 0.005]),
    ):
        sensing = _timestamped_sensing_payload(
            {"slow_time_matrix": rows},
            sample_times_s=times,
            strategy="coherent_cpi_fft",
            acquisition_key=key,
        )
        history, retained_times, error = window._resolve_timestamped_fast_time_history(
            sensing,
            SensingUpdateRequest(display_frame={"sensing": sensing}, state_frame={}, state_step=0),
        )
        assert error is None
        assert retained_times[-1] == times[-1]
        if key == "earlier":
            assert retained_times == times
            assert len(history) == 3
    assert retained_times == pytest.approx([0.0, 0.001, 0.002, 0.003, 0.004, 0.005])
    window.close()


def test_request_history_cannot_replace_absent_typed_delay_product(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    sensing = _timestamped_sensing_payload(
        {},
        sample_times_s=[0.0],
        strategy="sequential_stft",
        acquisition_key="no-delay-product",
    )
    history, times_s, error = window._resolve_timestamped_fast_time_history(
        sensing,
        SensingUpdateRequest(
            display_frame={"sensing": sensing},
            state_frame={},
            state_step=0,
            cir_history=[
                np.ones(4, dtype=np.complex128),
                np.ones(4, dtype=np.complex128),
            ],
            cir_history_times_s=[0.0, 0.001],
        ),
    )

    assert history == []
    assert times_s == []
    assert error == "no typed delay-domain sensing product is available"
    assert window._fast_time_range_axis_m is None
    window.close()


def test_request_history_cannot_replace_unaligned_typed_product_membership(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    rows = np.arange(12, dtype=np.float64).reshape(3, 4).astype(np.complex128)
    sensing = _timestamped_sensing_payload(
        {"slow_time_matrix": rows},
        sample_times_s=[0.0, 0.001, 0.002],
        strategy="coherent_cpi_fft",
        acquisition_key="unaligned-membership",
    )
    sensing["temporal"]["acquisition"]["samples"][1]["sample_time_s"] = 0.0015

    with pytest.raises(
        ValueError,
        match="slow-time coordinates differ from temporal acquisition samples",
    ):
        window._resolve_timestamped_fast_time_history(
            sensing,
            SensingUpdateRequest(
                display_frame={"sensing": sensing},
                state_frame={},
                state_step=0,
                cir_history=[row.copy() for row in rows],
                cir_history_times_s=[0.0, 0.001, 0.002],
            ),
        )

    assert window._fast_time_range_axis_m is None
    window.close()


def test_pregrouped_measurement_history_expands_and_deduplicates_overlap(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    resolved = {
        "acquisition": {
            "sample_source": "measured_pulse_stream",
            "cpi_length_samples": 3,
            "cpi_hop_samples": 2,
        },
        "processing": {
            "mode": "measurement_import",
            "strategy": "nist_mpc_csv_import",
        },
    }
    first_rows = np.arange(12, dtype=np.float64).reshape(3, 4).astype(np.complex128)
    first = _timestamped_sensing_payload(
        {
            "slow_time_matrix": first_rows,
            "config_resolved": resolved,
        },
        sample_times_s=[0.0, 0.001, 0.002],
        strategy="nist_mpc_csv_import",
        acquisition_key="first-measurement",
    )

    first_history, first_times_s, first_error = window._resolve_timestamped_fast_time_history(
        first,
        SensingUpdateRequest(
            display_frame={"sensing": first},
            state_frame={},
            state_step=0,
        ),
    )

    assert first_error is None
    assert first_times_s == pytest.approx([0.0, 0.001, 0.002])
    assert len(first_history) == 3

    second_rows = np.vstack(
        (
            first_rows[-1],
            first_rows[-1] + 10.0,
            first_rows[-1] + 20.0,
        )
    )
    second = _timestamped_sensing_payload(
        {
            "slow_time_matrix": second_rows,
            "config_resolved": resolved,
        },
        sample_times_s=[0.002, 0.003, 0.004],
        strategy="nist_mpc_csv_import",
        acquisition_key="second-measurement",
        scene_frame_id=1,
    )

    history, times_s, error = window._resolve_timestamped_fast_time_history(
        second,
        SensingUpdateRequest(
            display_frame={"sensing": second},
            state_frame={},
            state_step=1,
        ),
    )

    assert error is None
    assert times_s == pytest.approx([0.0, 0.001, 0.002, 0.003, 0.004])
    assert len(history) == 5
    np.testing.assert_array_equal(history[-1], second_rows[-1])
    window.close()


def test_sequential_history_uses_current_cir_not_overlapping_stm_rows(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    current_cir = np.array([91.0, 92.0, 93.0, 94.0], dtype=np.complex128)
    slow_time = np.arange(12, dtype=np.float64).reshape(3, 4).astype(np.complex128)
    slow_time[-1] = current_cir
    sensing = _timestamped_sensing_payload(
        {
            "cir": current_cir,
            "slow_time_matrix": slow_time,
            # A configured PRF cannot manufacture cadence for this one sample.
            "config": {"prf_hz": 50_000.0},
        },
        sample_times_s=[0.0, 0.001, 0.002],
        strategy="sequential_stft",
        acquisition_key="overlapping-window",
    )
    request = SensingUpdateRequest(
        display_frame={"sensing": sensing},
        state_frame={},
        state_step=0,
    )

    history, times_s, error = window._resolve_timestamped_fast_time_history(
        sensing,
        request,
    )

    assert len(history) == 1
    np.testing.assert_array_equal(history[0], current_cir)
    assert times_s == pytest.approx([0.002])
    assert error == "at least two persisted sample timestamps are required"
    window.close()


def test_micro_doppler_rejects_nonuniform_persisted_cadence(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    _APP.processEvents()
    window._view_mode_box.setCurrentIndex(window._view_mode_box.findData("micro-doppler"))
    window._micro_mode_box.setCurrentIndex(window._micro_mode_box.findData("fixed-bin"))
    _APP.processEvents()
    calls: list[bool] = []
    monkeypatch.setattr(
        "visualizer.src.sensing.sensing_window.compute_micro_doppler",
        lambda *_args, **_kwargs: calls.append(True),
    )
    history = [np.ones(4, dtype=np.complex128) for _ in range(64)]
    times_s = (np.arange(64, dtype=float) * 0.001).tolist()
    times_s[32] += 0.0002

    window._update_micro_doppler(
        {},
        50_000.0,
        1e9,
        "monostatic",
        step=63,
        cir_history=history,
        cir_history_times_s=times_s,
    )

    assert calls == []
    assert window._micro_mode_used == "unavailable"
    assert window._micro_status == "persisted sample timestamps are nonuniform"
    window.close()


def test_micro_doppler_cache_reuses_fixed_bin_result(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    _APP.processEvents()
    # Switch to micro-doppler view so the micro_plot is visible
    window._view_mode_box.setCurrentIndex(window._view_mode_box.findData("micro-doppler"))
    _APP.processEvents()
    window._micro_mode_box.setCurrentIndex(
        window._micro_mode_box.findData("fixed-bin")
    )  # Fixed bin
    calls = {"count": 0}
    physical_prfs_hz: list[float] = []

    def _fake_compute(*_args, **_kwargs):
        calls["count"] += 1
        physical_prfs_hz.append(float(_kwargs["prf_hz"]))
        return MicroDopplerResult(
            spectrogram=np.ones((5, 32), dtype=np.float64),
            time_axis_s=np.linspace(0.0, 0.032, 5),
            doppler_axis_hz=np.linspace(-500.0, 500.0, 32),
            range_bin=2,
            range_m=3.0,
        )

    monkeypatch.setattr(
        "visualizer.src.sensing.sensing_window.compute_micro_doppler",
        _fake_compute,
    )

    t = np.arange(64)
    cir_history = [
        np.eye(1, 4, 2, dtype=np.complex128).ravel() * np.exp(1j * 2 * np.pi * 0.125 * i) for i in t
    ]
    sample_times_s = (t * 0.001).tolist()
    range_axis_m = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float64)
    sensing_block = {"detections": [{"range_m": 3.0, "power_linear": 10.0}]}

    window._update_micro_doppler(
        sensing_block,
        37.0,
        1e9,
        "monostatic",
        step=63,
        cir_history=cir_history,
        cir_history_times_s=sample_times_s,
        range_axis_m=range_axis_m,
    )
    window._update_micro_doppler(
        sensing_block,
        37.0,
        1e9,
        "monostatic",
        step=63,
        cir_history=cir_history,
        cir_history_times_s=sample_times_s,
        range_axis_m=range_axis_m,
    )

    assert calls["count"] == 1
    assert physical_prfs_hz == pytest.approx([1000.0])
    assert window._micro_status == "fixed-bin cached from step 63"
    window.close()


def test_micro_doppler_cache_invalidates_when_gate_changes(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    _APP.processEvents()
    # Switch to micro-doppler view so the micro_plot is visible
    window._view_mode_box.setCurrentIndex(window._view_mode_box.findData("micro-doppler"))
    _APP.processEvents()
    window._micro_mode_box.setCurrentIndex(
        window._micro_mode_box.findData("gated")
    )  # Collapsed gate
    calls = {"count": 0}

    def _fake_collapsed(*_args, **_kwargs):
        calls["count"] += 1
        return MicroDopplerResult(
            spectrogram=np.ones((5, 32), dtype=np.float64),
            time_axis_s=np.linspace(0.0, 0.032, 5),
            doppler_axis_hz=np.linspace(-500.0, 500.0, 32),
            range_bin=-1,
            range_m=3.0,
            collapsed=True,
            range_bin_start=1,
            range_bin_end=3,
        )

    monkeypatch.setattr(
        "visualizer.src.sensing.sensing_window.compute_range_collapsed_micro_doppler",
        _fake_collapsed,
    )

    cir_history = [np.array([0.0, 1.0, 2.0, 0.0], dtype=np.complex128) for _ in range(64)]
    sample_times_s = (np.arange(64) * 0.001).tolist()
    range_axis_m = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float64)
    sensing_block = {"detections": [{"range_m": 2.0, "power_linear": 10.0}]}

    window._micro_gate_box.setValue(1)
    window._update_micro_doppler(
        sensing_block,
        1000.0,
        1e9,
        "monostatic",
        step=63,
        cir_history=cir_history,
        cir_history_times_s=sample_times_s,
        range_axis_m=range_axis_m,
    )

    window._micro_gate_box.setValue(2)
    window._update_micro_doppler(
        sensing_block,
        1000.0,
        1e9,
        "monostatic",
        step=63,
        cir_history=cir_history,
        cir_history_times_s=sample_times_s,
        range_axis_m=range_axis_m,
    )

    assert calls["count"] == 2
    window.close()


def test_truth_toggle_does_not_change_auto_blind_micro_doppler(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._view_mode_box.setCurrentIndex(window._view_mode_box.findData("micro-doppler"))
    _APP.processEvents()

    sample_times_s = (np.arange(64, dtype=np.float64) * 0.001).tolist()
    range_axis_m = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float64)
    cir_history = [np.ones(4, dtype=np.complex128) for _ in sample_times_s]
    sensing_block = {"detections": [{"range_m": 2.0, "power_linear": 10.0}]}
    window._target_history = {
        "walker.only": {
            "times_s": list(sample_times_s),
            "range": [2.0] * len(sample_times_s),
            "speed": [0.5] * len(sample_times_s),
        }
    }
    blind_calls: list[bool] = []
    oracle_calls: list[bool] = []

    def _fake_collapsed(*_args, **_kwargs):
        blind_calls.append(True)
        return MicroDopplerResult(
            spectrogram=np.arange(160, dtype=np.float64).reshape(5, 32) + 1.0,
            time_axis_s=np.linspace(0.0, 0.032, 5),
            doppler_axis_hz=np.linspace(-500.0, 500.0, 32),
            range_bin=-1,
            range_m=2.0,
            collapsed=True,
            range_bin_start=0,
            range_bin_end=3,
        )

    monkeypatch.setattr(
        "visualizer.src.sensing.sensing_window.compute_range_collapsed_micro_doppler",
        _fake_collapsed,
    )
    monkeypatch.setattr(
        "visualizer.src.sensing.sensing_window.compute_range_tracked_micro_doppler",
        lambda *_args, **_kwargs: oracle_calls.append(True),
    )

    window._update_micro_doppler(
        sensing_block,
        1000.0,
        1e9,
        "monostatic",
        step=63,
        cir_history=cir_history,
        cir_history_times_s=sample_times_s,
        range_axis_m=range_axis_m,
    )
    first_image = np.asarray(window.micro_image.image).copy()
    first_cache_keys = tuple(window._micro_result_cache)

    window._show_gt_cb.setChecked(True)
    _APP.processEvents()
    window._update_micro_doppler(
        sensing_block,
        1000.0,
        1e9,
        "monostatic",
        step=63,
        cir_history=cir_history,
        cir_history_times_s=sample_times_s,
        range_axis_m=range_axis_m,
    )

    assert window._micro_mode_used == "collapsed"
    assert blind_calls == [True]
    assert oracle_calls == []
    assert tuple(window._micro_result_cache) == first_cache_keys
    np.testing.assert_array_equal(window.micro_image.image, first_image)
    window.close()


def test_micro_doppler_defers_compute_while_window_is_hidden(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    # Switch to micro-doppler view so micro_plot is in the grid
    window._view_mode_box.setCurrentIndex(window._view_mode_box.findData("micro-doppler"))
    window._micro_mode_box.setCurrentIndex(
        window._micro_mode_box.findData("fixed-bin")
    )  # Fixed bin
    calls = {"count": 0}

    def _fake_compute(*_args, **_kwargs):
        calls["count"] += 1
        return MicroDopplerResult(
            spectrogram=np.ones((5, 32), dtype=np.float64),
            time_axis_s=np.linspace(0.0, 0.032, 5),
            doppler_axis_hz=np.linspace(-500.0, 500.0, 32),
            range_bin=2,
            range_m=3.0,
        )

    monkeypatch.setattr(
        "visualizer.src.sensing.sensing_window.compute_micro_doppler",
        _fake_compute,
    )

    cir_history = [np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128) for _ in range(64)]
    sample_times_s = (np.arange(64) * 0.001).tolist()
    range_axis_m = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float64)
    sensing_block = {"detections": [{"range_m": 2.0, "power_linear": 10.0}]}

    # Window not shown yet → deferred (window hidden)
    window._update_micro_doppler(
        sensing_block,
        1000.0,
        1e9,
        "monostatic",
        step=63,
        cir_history=cir_history,
        cir_history_times_s=sample_times_s,
        range_axis_m=range_axis_m,
    )

    assert calls["count"] == 0
    assert window._micro_mode_used == "deferred"
    assert window._micro_status == "deferred (window hidden)"

    window.show()
    _APP.processEvents()
    window._update_micro_doppler(
        sensing_block,
        1000.0,
        1e9,
        "monostatic",
        step=63,
        cir_history=cir_history,
        cir_history_times_s=sample_times_s,
        range_axis_m=range_axis_m,
    )

    assert calls["count"] == 1
    window.close()


def test_tracked_micro_doppler_passes_exact_identity_time_and_range_axis(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._view_mode_box.setCurrentIndex(window._view_mode_box.findData("micro-doppler"))
    window._micro_mode_box.setCurrentIndex(window._micro_mode_box.findData("tracked"))
    _APP.processEvents()

    sample_times_s = (np.arange(64, dtype=np.float64) * 0.001).tolist()
    range_axis_m = np.array([0.0, 0.75, 2.0, 4.5], dtype=np.float64)
    cir_history = [np.ones(4, dtype=np.complex128) for _ in sample_times_s]
    window._target_history = {
        "walker.only": {
            "times_s": list(sample_times_s),
            "range": [2.0] * len(sample_times_s),
            "speed": [1.25] * len(sample_times_s),
        }
    }
    captured: dict[str, object] = {}

    def _fake_tracked(gt_entries, **kwargs):
        captured["gt_entries"] = gt_entries
        captured.update(kwargs)
        return MicroDopplerResult(
            spectrogram=np.ones((5, 32), dtype=np.float64),
            time_axis_s=np.linspace(0.0, 0.032, 5),
            doppler_axis_hz=np.linspace(-500.0, 500.0, 32),
            range_bin=2,
            range_m=2.0,
            range_tracking=True,
            truth_assisted=True,
            selection_source="ground_truth_target_range",
            target_id="walker.only",
        )

    monkeypatch.setattr(
        "visualizer.src.sensing.sensing_window.compute_range_tracked_micro_doppler",
        _fake_tracked,
    )

    window._update_micro_doppler(
        {},
        999_999.0,
        1.0,
        "monostatic",
        step=63,
        cir_history=cir_history,
        cir_history_times_s=sample_times_s,
        range_axis_m=range_axis_m,
    )

    assert captured["target_id"] == "walker.only"
    assert captured["sample_times_s"] == pytest.approx(sample_times_s)
    np.testing.assert_array_equal(captured["range_axis_m"], range_axis_m)
    entries = captured["gt_entries"]
    assert isinstance(entries, list)
    assert entries[0] == {
        "reference_time_s": 0.0,
        "target_ids": ["walker.only"],
        "target_ranges": [2.0],
        "target_speeds": [1.25],
    }
    assert entries[-1]["reference_time_s"] == pytest.approx(sample_times_s[-1])
    assert window._micro_mode_used == "tracked"
    assert "Oracle-assisted (GT tracked)" in window._micro_status
    assert "Oracle-assisted (GT tracked)" in window.micro_plot.getPlotItem().titleLabel.text
    window.close()


def test_tracked_micro_doppler_disables_for_multiple_canonical_target_ids(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._view_mode_box.setCurrentIndex(window._view_mode_box.findData("micro-doppler"))
    window._micro_mode_box.setCurrentIndex(window._micro_mode_box.findData("tracked"))
    _APP.processEvents()

    sample_times_s = (np.arange(64, dtype=np.float64) * 0.001).tolist()
    range_axis_m = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    cir_history = [np.ones(4, dtype=np.complex128) for _ in sample_times_s]
    window._target_history = {
        target_id: {
            "times_s": list(sample_times_s),
            "range": [target_range_m] * len(sample_times_s),
            "speed": [0.5] * len(sample_times_s),
        }
        for target_id, target_range_m in (("walker.alpha", 1.0), ("walker.beta", 2.0))
    }
    calls: list[bool] = []
    monkeypatch.setattr(
        "visualizer.src.sensing.sensing_window.compute_range_tracked_micro_doppler",
        lambda *_args, **_kwargs: calls.append(True),
    )

    window._update_micro_doppler(
        {"detections": [{"range_m": 2.0, "power_linear": 100.0}]},
        1000.0,
        1e9,
        "monostatic",
        cir_history=cir_history,
        cir_history_times_s=sample_times_s,
        range_axis_m=range_axis_m,
    )

    assert calls == []
    assert window._micro_mode_used == "unavailable"
    assert window._micro_status == (
        "Oracle-assisted (GT tracked) requires exactly one canonical target_id (found 2)"
    )
    window.close()


def test_tracked_micro_doppler_disables_for_unaligned_truth_times(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._view_mode_box.setCurrentIndex(window._view_mode_box.findData("micro-doppler"))
    window._micro_mode_box.setCurrentIndex(window._micro_mode_box.findData("tracked"))
    _APP.processEvents()

    sample_times_s = (np.arange(64, dtype=np.float64) * 0.001).tolist()
    range_axis_m = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    cir_history = [np.ones(4, dtype=np.complex128) for _ in sample_times_s]
    window._target_history = {
        "walker.only": {
            "times_s": [value + 0.5 for value in sample_times_s],
            "range": [2.0] * len(sample_times_s),
            "speed": [0.5] * len(sample_times_s),
        }
    }
    calls: list[bool] = []
    monkeypatch.setattr(
        "visualizer.src.sensing.sensing_window.compute_range_tracked_micro_doppler",
        lambda *_args, **_kwargs: calls.append(True),
    )

    window._update_micro_doppler(
        {},
        1000.0,
        1e9,
        "monostatic",
        cir_history=cir_history,
        cir_history_times_s=sample_times_s,
        range_axis_m=range_axis_m,
    )

    assert calls == []
    assert window._micro_mode_used == "unavailable"
    assert window._micro_status == (
        "Oracle-assisted (GT tracked) requires canonical truth exactly aligned to every "
        "cir_history_times_s sample"
    )
    window.close()


@pytest.mark.parametrize("mode", ["gated", "fixed-bin"])
def test_blind_micro_doppler_modes_disable_without_explicit_detection(
    qtbot,
    mode: str,
) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._view_mode_box.setCurrentIndex(window._view_mode_box.findData("micro-doppler"))
    window._micro_mode_box.setCurrentIndex(window._micro_mode_box.findData(mode))
    _APP.processEvents()

    sample_times_s = (np.arange(64, dtype=np.float64) * 0.001).tolist()
    range_axis_m = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    cir_history = [np.ones(4, dtype=np.complex128) for _ in sample_times_s]

    window._update_micro_doppler(
        {},
        1000.0,
        1e9,
        "monostatic",
        cir_history=cir_history,
        cir_history_times_s=sample_times_s,
        range_axis_m=range_axis_m,
    )

    assert window._micro_mode_used == "unavailable"
    assert "explicit finite range_m" in window._micro_status
    window.close()


@pytest.mark.parametrize("cpi_count", [1, 2])
def test_collapsed_views_show_weak_returns_across_all_ranges_without_detections(
    qtbot,
    cpi_count: int,
) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._view_mode_box.setCurrentIndex(window._view_mode_box.findData("micro-doppler"))
    _APP.processEvents()
    samples_per_cpi = 128
    sample_times_s = (3.072 + np.arange(samples_per_cpi * cpi_count) * 0.001).tolist()
    range_axis_m = np.arange(16, dtype=np.float64)
    pulse = np.arange(samples_per_cpi)
    cpi = np.zeros((samples_per_cpi, 16), dtype=np.complex128)
    cpi[:, 1] = 1e-10 * np.exp(2j * np.pi * 8 * pulse / samples_per_cpi)
    cpi[:, 14] = 1e-10 * np.exp(-2j * np.pi * 16 * pulse / samples_per_cpi)
    # An independent phase jump must not create windows straddling CPIs.
    cir_history = np.concatenate([cpi * np.exp(1.3j * i) for i in range(cpi_count)])
    original = cir_history.copy()
    sensing = _timestamped_sensing_payload(
        {"slow_time_matrix": cir_history[-samples_per_cpi:]},
        sample_times_s=sample_times_s[-samples_per_cpi:],
        strategy="coherent_cpi_fft",
        acquisition_key=f"weak-all-range-{cpi_count}",
    )

    window._update_collapsed_doppler(
        sensing,
        1000.0,
        1e9,
        "monostatic",
        cir_history=list(cir_history),
        cir_history_times_s=sample_times_s,
        range_axis_m=range_axis_m,
    )
    window._update_micro_doppler(
        sensing,
        1000.0,
        1e9,
        "monostatic",
        cir_history=list(cir_history),
        cir_history_times_s=sample_times_s,
        range_axis_m=range_axis_m,
    )

    title = window.collapsed_doppler_plot.getPlotItem().titleLabel.text
    assert "all 16 range bins, 128 pulses/window" in title
    assert window._micro_mode_used == "collapsed"
    for image in (window.collapsed_doppler_image, window.micro_image):
        assert image.image.shape == (cpi_count, 128)
        assert np.ptp(image.image) > 40.0
        np.testing.assert_array_equal(np.sort(np.argsort(image.image[0])[-2:]), [48, 72])
        rect = image.mapRectToParent(image.boundingRect())
        assert rect.left() == pytest.approx(3.072)
        assert rect.width() == pytest.approx(cpi_count * 0.128)
        if cpi_count == 2:
            np.testing.assert_allclose(image.image[0], image.image[1], atol=1e-7)
    np.testing.assert_array_equal(cir_history, original)
    window.close()


def test_detection_overlays_remain_above_masks_and_inside_automatic_zoom(qtbot) -> None:
    window = SensingWindow()
    qtbot.addWidget(window)
    window.rd_det_scatter.setData([90.0], [4.0])
    window.rd_gt_scatter.setData([95.0], [3.0])
    window.rd_det_scatter.setVisible(True)
    window.rd_gt_scatter.setVisible(True)
    window._zoom_data_cb.setChecked(True)
    power_db = np.full((11, 101), -200.0)
    power_db[3, 10] = 0.0

    window._apply_zoom_to_data(window.rd_plot, power_db, np.arange(101), np.arange(-5, 6))

    x_limits, y_limits = window.rd_plot.viewRange()
    assert x_limits[0] <= 10.0 and x_limits[1] >= 95.0
    assert y_limits[0] <= -2.0 and y_limits[1] >= 4.0
    assert window.rd_det_scatter.zValue() > window.rd_gt_scatter.zValue()
    assert window.rd_det_scatter.zValue() > window.rd_cfar_overlay.zValue()
    window.close()


def test_oracle_micro_doppler_preserves_absolute_times_without_double_offset(qtbot) -> None:
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._view_mode_box.setCurrentIndex(window._view_mode_box.findData("micro-doppler"))
    window._micro_mode_box.setCurrentIndex(window._micro_mode_box.findData("tracked"))
    _APP.processEvents()
    times = 3.072 + np.arange(64) * 0.001
    window._target_history = {
        "walker": {"times_s": times.tolist(), "range": [1.0] * 64, "speed": [1.0] * 64}
    }
    rows = np.zeros((64, 4), dtype=complex)
    rows[:, 1] = np.exp(2j * np.pi * np.arange(64) / 8)
    window._update_micro_doppler(
        {},
        1000.0,
        1e9,
        "monostatic",
        cir_history=list(rows),
        cir_history_times_s=times.tolist(),
        range_axis_m=np.arange(4),
    )
    assert window._micro_mode_used == "tracked"
    cached = next(iter(window._micro_result_cache.values()))
    assert cached.time_axis_s[0] == pytest.approx(3.088)
    window.close()


def test_format_stage_profile_summary_surfaces_total_and_hot_stages() -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    summary = SensingWindow._format_stage_profile_summary(  # type: ignore[attr-defined]
        {
            "stage_profile_ms": {
                "cir_extract": 0.4,
                "rd_fft": 2.25,
                "detection": 1.5,
                "total_strategy": 5.0,
            }
        }
    )

    assert summary == "Timing: 5.0 ms (rd_fft 2.2 ms, detection 1.5 ms)"


def test_apply_display_preset_updates_dashboard_controls(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    assert not window._zoom_data_cb.isChecked()
    from visualizer.src.panels.sensing_panel import SensingPanel

    for preset in ({}, *(item["dashboard"] for item in SensingPanel.SENSING_PRESETS.values())):
        window.apply_display_preset({"zoom_to_data": True})
        assert window._zoom_data_cb.isChecked()
        window.apply_display_preset(preset)
        assert not window._zoom_data_cb.isChecked()
    window.apply_display_preset(
        {
            "view_mode": "diagnostics",
            "micro_doppler_mode": "tracked",
            "micro_gate_half_width": 4,
            "rd_dynamic_range_db": 36,
            "micro_dynamic_range_db": 74,
            "show_detections": False,
            "show_gt": True,
            "show_cfar": False,
            "zoom_to_data": False,
        }
    )

    assert window._view_mode_box.currentData() == "diagnostics"
    assert window._micro_mode_box.currentData() == "auto"
    assert window._micro_gate_box.value() == 4
    assert window._rd_dynamic_range_box.value() == 36
    assert window._micro_dynamic_range_box.value() == 74
    assert window._show_det_cb.isChecked() is False
    assert window._show_gt_cb.isChecked() is True
    # This preset has no recorded detector stages, so duplicate CFAR views stay hidden.
    assert window.cfar_rd_plot.isHidden() is True


def test_sensing_window_defaults_to_blind_layers_and_labels_oracle_explicitly(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)

    assert window._show_gt_cb.isChecked() is True
    assert window._show_gt_cb.text() == "Ground truth (evaluation)"
    assert window._micro_mode_box.currentData() == "auto"
    tracked_index = window._micro_mode_box.findData("tracked")
    assert tracked_index >= 0
    assert window._micro_mode_box.itemText(tracked_index) == "Around true range (oracle)"

    range_curve = window.gt_range_plot.plot([0.0], [1.0])
    speed_curve = window.gt_speed_plot.plot([0.0], [0.5])
    window._gt_range_curves["walker.only"] = range_curve
    window._gt_speed_curves["walker.only"] = speed_curve
    window._apply_overlay_visibility()
    assert range_curve.isVisible() is True
    assert speed_curve.isVisible() is True
    window._show_gt_cb.setChecked(False)
    assert range_curve.isVisible() is False
    assert speed_curve.isVisible() is False

    window._show_gt_cb.setChecked(True)
    assert range_curve.isVisible() is True
    assert speed_curve.isVisible() is True
    window.close()


def test_analysis_view_hides_diagnostics_widgets(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    _APP.processEvents()
    window.apply_display_preset({"view_mode": "analysis"})
    _APP.processEvents()

    # In analysis mode, CFAR plots, micro-Doppler spectrogram, and timeline
    # are not in the grid (dynamic repopulation removes them).
    assert window.cfar_rd_plot.isVisible() is False
    assert window.cfar_rp_plot.isVisible() is False
    assert window.micro_plot.isVisible() is False
    # Analysis mode should show these
    assert window.rd_plot.isVisible() is True
    assert window.rp_plot.isVisible() is True
    window.close()


def test_switching_to_micro_view_makes_plot_visible_before_render(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    _APP.processEvents()

    window._set_view_mode("micro-doppler")

    assert window.collapsed_doppler_plot.isVisible() is True
    assert window.micro_plot.isVisible() is False
    assert window._micro_doppler_defer_reason() is None
    window.close()


def test_native_v4_dashboard_does_not_reconstruct_untyped_cfar_products(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    profile = SimpleNamespace(
        power_linear=np.array([1.0, 4.0, 2.0], dtype=np.float64),
        range_axis_m=np.array([0.0, 0.5, 1.5], dtype=np.float64),
        range_axis_label="Monostatic range (m)",
    )
    untyped_stage_arrays = {
        "range_profile_cfar_threshold": np.ones(3, dtype=np.float64),
        "range_doppler_cfar_mask": np.ones((2, 3), dtype=np.uint8),
        "config": {
            "cfar_guard_cells": 0,
            "cfar_ref_cells": 1,
            "cfar_threshold_scale": 99.0,
        },
    }
    window.cfar_rp_thresh.setData([0.0], [1.0])
    window._update_cfar_rp(untyped_stage_arrays, profile, "")
    threshold_x, threshold_y = window.cfar_rp_thresh.getData()

    window.rd_cfar_overlay.setImage(np.ones((2, 3), dtype=np.float64))
    window._update_rd_cfar_overlay(
        untyped_stage_arrays,
        SimpleNamespace(power_linear=np.ones((2, 3), dtype=np.float64)),
        QRectF(0.0, 0.0, 1.0, 1.0),
    )

    assert threshold_x is None or len(threshold_x) == 0
    assert threshold_y is None or len(threshold_y) == 0
    assert (
        "Detector threshold slice not recorded" in window.cfar_rp_plot.getPlotItem().titleLabel.text
    )
    assert window.rd_cfar_overlay.image is None
    window.close()


def test_range_velocity_renders_gt_overlay_points(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    window._target_history = {"walker.primary": {"times_s": [7.0], "range": [12.0], "speed": [1.5]}}
    captured: dict[str, list[float]] = {}

    def _capture(xs, ys):
        captured["x"] = list(xs)
        captured["y"] = list(ys)

    monkeypatch.setattr(window.rd_gt_scatter, "setData", _capture)

    rd_result = SimpleNamespace(
        power_linear=np.ones((4, 8), dtype=np.float64),
        range_axis_m=np.arange(8, dtype=np.float64),
        doppler_axis_hz=np.linspace(-500.0, 500.0, 4),
        range_axis_label="Range (m)",
        metadata={},
    )

    window._update_range_velocity(rd_result, {"sensing": {}}, "", 28e9)

    assert captured["x"] == [12.0]
    assert captured["y"] == [1.5]
    window.close()


@pytest.mark.parametrize(
    ("range_axis_m", "doppler_axis_hz", "axis_name"),
    [
        (
            np.array([0.0, 0.5, 2.0], dtype=np.float64),
            np.array([-2.0, 0.0, 2.0], dtype=np.float64),
            "range",
        ),
        (
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            np.array([-4.0, -1.0, 4.0], dtype=np.float64),
            "Doppler/velocity",
        ),
    ],
)
def test_range_velocity_rejects_nonuniform_producer_axes(
    qtbot,
    range_axis_m: np.ndarray,
    doppler_axis_hz: np.ndarray,
    axis_name: str,
) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    rd_result = SimpleNamespace(
        power_linear=np.ones((3, 3), dtype=np.float64),
        range_axis_m=range_axis_m,
        doppler_axis_hz=doppler_axis_hz,
        range_axis_label="Path Length (m)",
        metadata={},
    )

    with pytest.raises(
        ValueError,
        match=rf"nonuniform producer-authored {axis_name} axis",
    ):
        window._update_range_velocity(rd_result, {"sensing": {}}, "", 28e9)
    window.close()


def test_axis_rect_preserves_exact_uniform_bin_centers() -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    rect = SensingWindow._rd_rect(
        np.array([2.0, 5.0, 8.0], dtype=np.float64),
        np.array([-4.0, 0.0, 4.0], dtype=np.float64),
    )

    assert rect.x() == pytest.approx(0.5)
    assert rect.width() == pytest.approx(9.0)
    assert rect.y() == pytest.approx(-6.0)
    assert rect.height() == pytest.approx(12.0)


def _range_time_result(
    reference_time_s: float,
    *,
    range_axis_m: np.ndarray | None = None,
) -> SimpleNamespace:
    axis = (
        np.asarray(range_axis_m, dtype=np.float64)
        if range_axis_m is not None
        else np.array([0.0, 1.0, 2.0], dtype=np.float64)
    )
    return SimpleNamespace(
        power_linear=np.array(
            [reference_time_s + 1.0, reference_time_s + 2.0, reference_time_s + 3.0],
            dtype=np.float64,
        ),
        range_axis_m=axis,
        range_axis_label="Path Length (m)",
    )


def _range_time_sensing(reference_time_s: float) -> dict:
    return build_visualizer_sensing_payload(
        scene_frame_id=int(reference_time_s * 10.0),
        sample_time_s=reference_time_s,
    )


def test_range_time_waterfall_is_reverse_order_invariant_and_dedupes_rerenders(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    reverse_window = SensingWindow()
    qtbot.addWidget(reverse_window)
    for reference_time_s in (3.0, 1.0, 2.0):
        reverse_window._update_range_time_waterfall(
            _range_time_result(reference_time_s),
            sensing_block=_range_time_sensing(reference_time_s),
            display_step=int(reference_time_s * 1000.0),
            variant_suffix=" (clean)",
        )

    assert reverse_window._range_time_reference_times_s == [1.0, 2.0, 3.0]
    assert len(reverse_window._range_time_records) == 3
    assert reverse_window.range_time_raster.image is not None
    reverse_data = np.asarray(reverse_window.range_time_raster.image).copy()
    reverse_keys = tuple(reverse_window._range_time_event_keys)

    reverse_window._update_range_time_waterfall(
        _range_time_result(2.0),
        sensing_block=_range_time_sensing(2.0),
        display_step=-999_999,
        variant_suffix=" (clean)",
    )

    assert tuple(reverse_window._range_time_event_keys) == reverse_keys
    assert len(reverse_window._range_time_records) == 3
    np.testing.assert_array_equal(reverse_window.range_time_raster.image, reverse_data)

    forward_window = SensingWindow()
    qtbot.addWidget(forward_window)
    for reference_time_s in (1.0, 2.0, 3.0):
        forward_window._update_range_time_waterfall(
            _range_time_result(reference_time_s),
            sensing_block=_range_time_sensing(reference_time_s),
            display_step=int(reference_time_s),
            variant_suffix=" (clean)",
        )

    assert forward_window._range_time_event_keys == reverse_window._range_time_event_keys
    np.testing.assert_array_equal(forward_window.range_time_raster.image, reverse_data)
    assert "physical reference-time order" in (
        reverse_window.range_time_plot.getPlotItem().titleLabel.text
    )
    reverse_window.close()
    forward_window.close()


def test_range_time_waterfall_clears_visibly_on_axis_disagreement(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    window._update_range_time_waterfall(
        _range_time_result(1.0),
        sensing_block=_range_time_sensing(1.0),
        variant_suffix="",
    )
    window._update_range_time_waterfall(
        _range_time_result(
            2.0,
            range_axis_m=np.array([0.0, 1.25, 2.0], dtype=np.float64),
        ),
        sensing_block=_range_time_sensing(2.0),
        variant_suffix="",
    )

    assert window._range_time_records == {}
    assert window._range_time_reference_times_s == []
    assert window.range_time_image.isVisible() is False
    assert window.range_time_raster.isVisible() is False
    assert "range-axis disagreement" in window.range_time_plot.getPlotItem().titleLabel.text
    window.close()


def test_range_time_waterfall_cap_retains_latest_physical_results(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    monkeypatch.setattr(
        "visualizer.src.sensing.sensing_window._MAX_RANGE_TIME_HISTORY",
        3,
    )
    window = SensingWindow()
    qtbot.addWidget(window)
    for reference_time_s in (4.0, 1.0, 3.0, 2.0):
        window._update_range_time_waterfall(
            _range_time_result(reference_time_s),
            sensing_block=_range_time_sensing(reference_time_s),
            variant_suffix="",
        )

    assert window._range_time_reference_times_s == [2.0, 3.0, 4.0]
    assert len(window._range_time_records) == 3
    window.close()


def test_range_time_waterfall_never_uses_numeric_frame_id_as_time(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    window._update_range_time_waterfall(
        _range_time_result(1.0),
        sensing_block={},
        display_step=1_700_000_000,
        variant_suffix="",
    )

    assert window._range_time_records == {}
    assert window._range_time_reference_times_s == []
    assert "temporal contract missing" in window.range_time_plot.getPlotItem().titleLabel.text
    window.close()


def _timeline_sensing_with_invalid_sample() -> dict:
    sensing = _timestamped_sensing_payload(
        {},
        sample_times_s=[10.0, 11.0, 12.0],
        strategy="sequential_stft",
        acquisition_key="timeline-invalid-sample",
        scene_frame_id=77,
    )
    acquisition = sensing["temporal"]["acquisition"]
    invalid_sample = acquisition["samples"][1]
    invalid_sample["valid"] = False
    invalid_sample["source"] = None
    invalid_sample["invalid_reason"] = "source pulse unavailable"
    acquisition["observed_sample_count"] = 2
    acquisition["valid_sample_count"] = 2
    acquisition["completeness"] = "incomplete_processed_by_explicit_policy"
    return sensing


def test_acquisition_timeline_renders_all_samples_and_stale_provenance(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    sensing = _timeline_sensing_with_invalid_sample()
    display_frame = {
        "_sensing_display_provenance": {
            "current_scene_frame_id": 9001,
            "current_scene_time_s": 14.5,
            "stale_for_current_scene": True,
            "stale_age_s": 3.5,
            "pinned": True,
        }
    }

    window._update_acquisition_timeline(sensing, display_frame)

    valid_x, _valid_y = window.acquisition_valid_samples.getData()
    invalid_x, _invalid_y = window.acquisition_invalid_samples.getData()
    span_x, _span_y = window.acquisition_span_curve.getData()
    np.testing.assert_array_equal(valid_x, [10.0, 12.0])
    np.testing.assert_array_equal(invalid_x, [11.0])
    np.testing.assert_array_equal(span_x, [10.0, 12.0])
    assert window.acquisition_result_marker.value() == pytest.approx(11.0)
    assert window.acquisition_scene_time_marker.value() == pytest.approx(14.5)
    assert window.acquisition_scene_time_marker.isVisible() is True

    temporal = sensing["temporal"]
    expected_labels = {
        "acq": temporal["acquisition"]["acquisition_id"][:8],
        "pair": temporal["acquisition"]["pair_id"][:8],
        "run": temporal["result"]["processing_run_id"][:8],
        "recipe": temporal["result"]["processing_recipe_id"][:8],
    }
    for label, value in expected_labels.items():
        assert f"{label}={value}" in window._acquisition_timeline_status
    assert "held-stale stale=3.5s" in window._acquisition_timeline_status
    assert window._acquisition_timeline_status in (
        window.acquisition_timeline_plot.getPlotItem().titleLabel.text
    )
    window.close()


def test_acquisition_timeline_does_not_infer_scene_time_from_numeric_ids(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    sensing = _timeline_sensing_with_invalid_sample()
    window._update_acquisition_timeline(
        sensing,
        {
            "_sensing_display_provenance": {
                "storage_frame_id": 1_700_000_000,
                "current_scene_frame_id": 1_800_000_000,
                "stale_for_current_scene": False,
            }
        },
    )

    assert window.acquisition_result_marker.value() == pytest.approx(11.0)
    assert window.acquisition_scene_time_marker.isVisible() is False
    assert window._acquisition_timeline_status.endswith("| exact")

    window._update_acquisition_timeline(
        sensing,
        {
            "timestamp_s": 13.25,
            "_sensing_display_provenance": {
                "current_scene_frame_id": 1_800_000_000,
                "stale_for_current_scene": False,
            },
        },
    )
    assert window.acquisition_scene_time_marker.value() == pytest.approx(13.25)
    assert window.acquisition_scene_time_marker.isVisible() is True
    window.close()


def _truth_result_frame(
    *,
    scene_frame_id: int,
    reference_time_s: float,
    target_range_m: float,
    target_speed_m_s: float,
    target_id: str = "walker.primary",
) -> dict:
    return _truth_result_frame_for_targets(
        scene_frame_id=scene_frame_id,
        reference_time_s=reference_time_s,
        target_rows=[(target_id, target_range_m, target_speed_m_s)],
    )


def _truth_result_frame_for_targets(
    *,
    scene_frame_id: int,
    reference_time_s: float,
    target_rows: list[tuple[str, float, float]],
) -> dict:
    sensing = build_visualizer_sensing_payload(
        {
            "config": {"range_mode": "monostatic"},
            "detections": (
                [
                    {
                        "range_m": target_rows[0][1] + 0.25,
                        "velocity_m_s": target_rows[0][2] + 0.5,
                    }
                ]
                if target_rows
                else []
            ),
        },
        scene_frame_id=scene_frame_id,
        sample_time_s=reference_time_s,
    )
    # This truth-only frame carries no dense product, so it must not retain an
    # unreferenced numeric coordinate synthesized by the general fixture helper.
    sensing.pop("slow_time_axis_s", None)
    pair_id = sensing["temporal"]["acquisition"]["pair_id"]
    catalog = build_target_catalog(
        [
            {
                "target_id": target_id,
                "display_name": target_id.replace(".", " ").title(),
            }
            for target_id, _target_range_m, _target_speed_m_s in target_rows
        ]
    )
    sensing["target_catalog"] = catalog
    dynamic_targets = [
        {
            "target_id": target_id,
            "lifecycle": "active",
            "valid": True,
            "position_m": [target_range_m, 0.0, 0.0],
            "orientation_rad": None,
            "velocity_m_s": [target_speed_m_s, 0.0, 0.0],
            "invalid_reason": None,
        }
        for target_id, target_range_m, target_speed_m_s in target_rows
    ]
    sensing["target_state"] = build_target_state(
        catalog,
        scene_frame_id=scene_frame_id,
        reference_time_s=reference_time_s,
        targets=dynamic_targets,
    )
    sensing["truth"] = build_target_truth(
        catalog,
        targets=[
            {
                **dynamic_target,
                "pair_observables": [
                    {
                        "pair_id": pair_id,
                        "bistatic_path_length_m": 2.0 * target_range_m,
                        "monostatic_equivalent_range_m": target_range_m,
                        "path_closing_rate_m_s": 2.0 * target_speed_m_s,
                        "monostatic_equivalent_velocity_m_s": target_speed_m_s,
                        "bistatic_bisector_velocity_m_s": target_speed_m_s,
                        "velocity_valid": True,
                        "velocity_derivation_source": "analytic_velocity_projection",
                        "velocity_reference_dt_s": None,
                        "undefined_reason": None,
                    }
                ],
            }
            for dynamic_target, (target_id, target_range_m, target_speed_m_s) in zip(
                dynamic_targets,
                target_rows,
                strict=True,
            )
        ],
    )
    sensing["temporal"]["truth"] = {
        "reference_time_s": reference_time_s,
        "policy": "exact_result_reference",
        "scene_frame_id": scene_frame_id,
        "source": "persisted_test_truth",
    }
    return {
        # Deliberately unrelated outer-frame poses: the dashboard must not
        # recompute GT range or speed from these values.
        "tx_positions": np.array([[1000.0, 0.0, 0.0]]),
        "rx_positions": np.array([[-1000.0, 0.0, 0.0]]),
        "target_pos": np.array(
            [[5000.0 + index, 0.0, 0.0] for index, _target in enumerate(target_rows)]
        ).reshape((-1, 3)),
        "sensing": sensing,
    }


def test_sensing_window_direct_boundary_rejects_pre_v4_request(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")
    window = SensingWindow()
    qtbot.addWidget(window)
    request = SensingUpdateRequest(
        display_frame={"sensing": {"schema_version": 3, "range_profile": [1.0]}},
        state_frame={},
        state_step=0,
    )

    with pytest.raises(
        UnsupportedSensingSchemaError,
        match="Regenerate this sensing dataset",
    ):
        window.update_from_request(request)
    window.close()


def test_sensing_window_boundary_rejects_undeclared_stored_micro_doppler_alias(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")
    window = SensingWindow()
    qtbot.addWidget(window)
    frame = _truth_result_frame(
        scene_frame_id=2,
        reference_time_s=0.2,
        target_range_m=2.0,
        target_speed_m_s=0.5,
    )
    frame["sensing"]["micro_doppler_spectrogram"] = np.ones((4, 8), dtype=np.float64)

    with pytest.raises(
        ValueError,
        match=r"forbids undeclared numeric payloads: micro_doppler_spectrogram",
    ):
        window.update_from_request(
            SensingUpdateRequest(
                display_frame=frame,
                state_frame=frame,
                state_step=2,
            )
        )
    window.close()


def test_sensing_window_update_request_consumes_canonical_truth_by_target_id(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")
    window = SensingWindow()
    qtbot.addWidget(window)
    window._view_mode_box.setCurrentIndex(window._view_mode_box.findData("analysis"))
    frame = _truth_result_frame(
        scene_frame_id=12,
        reference_time_s=1.2,
        target_range_m=14.0,
        target_speed_m_s=-0.75,
        target_id="walker.bound",
    )

    window.update_from_request(
        SensingUpdateRequest(
            display_frame=frame,
            state_frame=frame,
            state_step=12,
            display_step=12,
        )
    )

    assert window._target_history["walker.bound"] == {
        "times_s": [1.2],
        "range": [14.0],
        "speed": [-0.75],
    }
    assert window._get_target_label("walker.bound") == "Walker Bound"
    assert "GT: evaluation shown" in window.summary_label.toolTip()
    window.close()


def test_unavailable_scene_hides_previous_result_until_next_valid_request(qtbot) -> None:
    window = SensingWindow()
    qtbot.addWidget(window)
    frame = _truth_result_frame(
        scene_frame_id=12, reference_time_s=1.2, target_range_m=14.0, target_speed_m_s=-0.75
    )
    request = SensingUpdateRequest(
        display_frame=frame, state_frame=frame, state_step=12, display_step=12
    )
    window.update_from_request(request)
    history = deepcopy(window._target_history)
    window.set_unavailable("No result bound to scene 13", scene_frame_id=13)

    assert window._last_request is None
    assert window._plots_widget.isHidden()
    assert not window._unavailable_label.isHidden()
    assert "13" in window.provenance_label.text()
    assert not window.detection_stage_export_button.isEnabled()
    window._set_view_mode("angular")
    window._show_gt_cb.setChecked(True)
    assert window._last_request is None
    assert window._plots_widget.isHidden()
    assert window._target_history == history

    window.update_from_request(request)
    assert window._last_request is request
    assert not window._plots_widget.isHidden()
    assert window._unavailable_label.isHidden()
    window.close()


def test_gt_history_preserves_unavailable_velocity_as_plot_gap(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")
    window = SensingWindow()
    qtbot.addWidget(window)
    frame = _truth_result_frame(
        scene_frame_id=13,
        reference_time_s=1.3,
        target_range_m=14.0,
        target_speed_m_s=-0.75,
        target_id="walker.unavailable",
    )
    observable = frame["sensing"]["truth"]["targets"][0]["pair_observables"][0]
    observable.update(
        path_closing_rate_m_s=None,
        monostatic_equivalent_velocity_m_s=None,
        bistatic_bisector_velocity_m_s=None,
        velocity_valid=False,
        velocity_derivation_source="unavailable",
        velocity_reference_dt_s=None,
        undefined_reason="truth_velocity_unavailable",
    )

    window._update_ground_truth(frame)

    assert window._target_history["walker.unavailable"]["speed"] == [None]
    assert window._current_gt_range_speed() == ([], [])
    window.close()


def test_overlay_legends_retain_detection_and_ground_truth_entries(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")

    window = SensingWindow()
    qtbot.addWidget(window)
    frame = _truth_result_frame(
        scene_frame_id=14,
        reference_time_s=1.4,
        target_range_m=15.0,
        target_speed_m_s=-1.0,
        target_id="walker.legend",
    )
    window._update_target_names(frame)
    window._update_ground_truth(frame)

    def labels(plot) -> list[str]:
        legend = plot.getPlotItem().legend
        assert legend is not None
        return [label.text for _sample, label in legend.items]

    assert labels(window.rd_plot) == ["Detection", "Ground truth", "Selected saved bin"]
    assert labels(window.gt_range_plot) == ["Detections", "GT Walker Legend"]
    assert labels(window.gt_speed_plot) == ["Detections", "GT Walker Legend"]
    assert labels(window.range_azimuth_plot) == ["Detection"]
    assert labels(window.doppler_azimuth_plot) == ["Detection"]
    assert window._gt_range_curves["walker.legend"].opts["symbol"] == "+"
    assert window._gt_speed_curves["walker.legend"].opts["symbol"] == "+"
    window.close()


def test_gt_detection_history_uses_result_time_without_frame_id_backfill(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")
    window = SensingWindow()
    qtbot.addWidget(window)

    window._update_ground_truth(
        _truth_result_frame(
            scene_frame_id=100,
            reference_time_s=1.25,
            target_range_m=12.0,
            target_speed_m_s=3.5,
        )
    )
    window._update_ground_truth(
        _truth_result_frame(
            scene_frame_id=900,
            reference_time_s=4.75,
            target_range_m=18.0,
            target_speed_m_s=-2.0,
        )
    )

    assert window._target_history["walker.primary"] == {
        "times_s": [1.25, 4.75],
        "range": [12.0, 18.0],
        "speed": [3.5, -2.0],
    }
    assert window._detection_history["times_s"] == [1.25, 4.75]
    assert window._detection_history["range"] == [12.25, 18.25]
    assert window._detection_history["speed"] == [4.0, -1.5]
    window.close()


def test_gt_history_omits_unbound_truth_and_resets_on_reverse(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")
    window = SensingWindow()
    qtbot.addWidget(window)
    newer = _truth_result_frame(
        scene_frame_id=70,
        reference_time_s=7.0,
        target_range_m=17.0,
        target_speed_m_s=2.0,
    )
    older = _truth_result_frame(
        scene_frame_id=20,
        reference_time_s=2.0,
        target_range_m=11.0,
        target_speed_m_s=-1.0,
    )

    window._update_ground_truth(newer)
    window._update_ground_truth(older)

    assert window._target_history["walker.primary"] == {
        "times_s": [2.0],
        "range": [11.0],
        "speed": [-1.0],
    }
    assert window._detection_history["times_s"] == [2.0]

    unbound = _truth_result_frame(
        scene_frame_id=80,
        reference_time_s=8.0,
        target_range_m=99.0,
        target_speed_m_s=99.0,
    )
    del unbound["sensing"]["temporal"]["truth"]
    window._update_ground_truth(unbound)

    assert window._target_history == {}
    assert "no persisted truth" in window.gt_range_plot.getPlotItem().titleLabel.text

    for offset, alias in enumerate(("gt_sensing", "gt_state"), start=9):
        legacy_channel = _truth_result_frame(
            scene_frame_id=offset * 10,
            reference_time_s=float(offset),
            target_range_m=101.0,
            target_speed_m_s=101.0,
        )
        legacy_channel["sensing"][alias] = legacy_channel["sensing"].pop("truth")
        window._update_ground_truth(legacy_channel)

        assert window._target_history == {}
    window.close()


def test_gt_history_joins_reordered_truth_rows_by_semantic_target_id(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")
    window = SensingWindow()
    qtbot.addWidget(window)

    first = _truth_result_frame_for_targets(
        scene_frame_id=1,
        reference_time_s=1.0,
        target_rows=[
            ("walker.alpha", 10.0, 1.0),
            ("walker.beta", 20.0, -2.0),
        ],
    )
    second = _truth_result_frame_for_targets(
        scene_frame_id=2,
        reference_time_s=2.0,
        target_rows=[
            ("walker.beta", 21.0, -2.5),
            ("walker.alpha", 11.0, 1.5),
        ],
    )

    window._update_target_names(first)
    window._update_ground_truth(first)
    window._update_ground_truth(second)

    assert window._target_history["walker.alpha"] == {
        "times_s": [1.0, 2.0],
        "range": [10.0, 11.0],
        "speed": [1.0, 1.5],
    }
    assert window._target_history["walker.beta"] == {
        "times_s": [1.0, 2.0],
        "range": [20.0, 21.0],
        "speed": [-2.0, -2.5],
    }
    assert window._get_target_label("walker.alpha") == "Walker Alpha"
    window.close()


def test_gt_history_requires_truth_reference_to_match_result_reference(qtbot) -> None:
    if not HAS_SENSING_WINDOW:
        pytest.skip("SensingWindow dependencies are unavailable in this environment")
    window = SensingWindow()
    qtbot.addWidget(window)
    mismatched = _truth_result_frame(
        scene_frame_id=3,
        reference_time_s=3.0,
        target_range_m=13.0,
        target_speed_m_s=1.25,
    )
    mismatched["sensing"]["temporal"]["truth"]["reference_time_s"] = 3.5

    window._update_ground_truth(mismatched)

    assert window._target_history == {}
    assert "no persisted truth" in window.gt_range_plot.getPlotItem().titleLabel.text
    window.close()
