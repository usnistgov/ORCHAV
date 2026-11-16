"""Tests for the installed, renderer-free sensing replay command."""

from __future__ import annotations

import builtins
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from shared.extensions.sensing import (
    SENSING_SCHEMA_VERSION,
    build_sensing_temporal_contract,
    canonical_sensing_hash,
)
from shared.sensing.products import SENSING_PRODUCT_CONTRACT_VERSION
from visualizer.src.sensing import replay_cli

_CLOCK_ID = "00000000-0000-0000-0000-000000000010"
_PROCESSING_RUN_1 = "00000000-0000-0000-0000-000000000001"
_PROCESSING_RUN_2 = "00000000-0000-0000-0000-000000000002"
_ACQUISITION_1 = "00000000-0000-0000-0000-000000000011"
_ACQUISITION_2 = "00000000-0000-0000-0000-000000000012"


class _Provider:
    def __init__(self, result_frame, *, bound_timestamp_s: float | None = None) -> None:
        self.result_frame = result_frame
        self.bound_timestamp_s = bound_timestamp_s

    def has_frame(self, frame_id: int) -> bool:
        return frame_id == 7 and self.bound_timestamp_s is not None

    def load_frame_projection(self, frame_id: int, _request):
        if frame_id == 7:
            return SimpleNamespace(
                frame=SimpleNamespace(frame_index=7, timestamp_s=self.bound_timestamp_s)
            )
        assert frame_id == 19
        return SimpleNamespace(frame=self.result_frame)


class _ProbeProvider:
    def __init__(self, sensing_by_frame: dict[int, dict]) -> None:
        self.sensing_by_frame = sensing_by_frame

    def load_frame_projection(self, frame_id: int, _request):
        return SimpleNamespace(frame=SimpleNamespace(sensing=self.sensing_by_frame[frame_id]))


def _result_frame() -> SimpleNamespace:
    return SimpleNamespace(
        frame_index=19,
        timestamp_s=0.019,
        sensing={
            "schema_version": 4,
            "temporal": {
                "acquisition": {
                    "samples": [
                        {
                            "sample_time_s": 0.007,
                            "source": {"scene_frame_id": 7},
                        }
                    ]
                },
                "display_binding": {"scene_frame_id": 7},
            },
        },
    )


def _probe_sensing(
    *,
    reference_time_s: float,
    processing_run_id: str,
    acquisition_id: str,
) -> dict:
    processing_recipe = {
        "window_range": "hann",
        "window_doppler": "hann",
    }
    samples = [
        {
            "sample_index": index,
            "sample_time_s": index * 0.001,
            "valid": True,
            "phase_coherent": True,
            "source": {
                "source_generation_id": _CLOCK_ID,
                "scene_frame_id": 100 + index,
                "sample_offset": 0,
            },
        }
        for index in range(4)
    ]
    temporal = build_sensing_temporal_contract(
        clock_id=_CLOCK_ID,
        acquisition_id=acquisition_id,
        acquisition_config_id=canonical_sensing_hash({"samples": 4, "prf_hz": 1000.0}),
        pair_id=canonical_sensing_hash({"tx_sensor_id": "probe-tx", "rx_sensor_id": "probe-rx"}),
        tx_sensor_id="probe-tx",
        rx_sensor_id="probe-rx",
        samples=samples,
        processing_run_id=processing_run_id,
        processing_recipe_id=canonical_sensing_hash(processing_recipe),
        reference_time_s=reference_time_s,
        reference_time_policy="explicit_test_reference",
        display_scene_frame_id=101,
        display_policy="explicit_test_binding",
        display_reason="Frame 101 is the explicit test scene binding.",
    )
    return {
        "schema_version": SENSING_SCHEMA_VERSION,
        "temporal": temporal,
        "product_contract_version": SENSING_PRODUCT_CONTRACT_VERSION,
        "products": {},
        "processing_strategy": "coherent_cpi_fft",
        "config_resolved": {
            "contract_version": 1,
            "fft_size_range": 64,
            "fft_size_doppler": 4,
            "sampling_frequency": 20.0e6,
            "prf_hz": 1000.0,
            "carrier_freq_hz": 3.5e9,
            "range_mode": "monostatic",
            "acquisition": {
                "sample_source": "coherent_one_solve",
                "rt_solve_stride_scene_steps": 4,
                "sample_period_s": 0.001,
                "sample_rate_hz": 1000.0,
                "cpi_length_samples": 4,
                "cpi_hop_samples": 4,
                "overlap_samples": 0,
                "partial_window_policy": "retain_unprocessed",
                "phase_contract": "explicit_test_phase_contract",
            },
            "processing": {
                "mode": "coherent_cpi",
                "strategy": "coherent_cpi_fft",
                "fft_size_doppler": 4,
                "result_stride_samples": 4,
                "product_semantics": "explicit_test_product",
                "evidence_status": "explicit_test_evidence",
            },
            "topology": {
                "pair_count": 1,
                "pair_index": 0,
                "tx_device_count": 1,
                "rx_device_count": 1,
                "tx_elements": 1,
                "rx_elements": 1,
                "rx_array_processing": "siso",
            },
            "processing_mode": "coherent_cpi",
            "processing_recipe": processing_recipe,
        },
    }


def _fail_replay_export_import(monkeypatch, missing_name: str) -> None:
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "visualizer.src.sensing.replay_export":
            raise ModuleNotFoundError(
                f"No module named {missing_name!r}",
                name=missing_name,
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_event_loader_uses_exact_persisted_bound_scene_time(monkeypatch) -> None:
    frame = _result_frame()
    captured = {}

    def fake_build_replay_event(candidate, context, *, require_product_data):
        captured.update(
            candidate=candidate,
            context=context,
            require_product_data=require_product_data,
        )
        return "event"

    monkeypatch.setattr(replay_cli, "build_replay_event", fake_build_replay_event)

    event = replay_cli._event_from_frame(_Provider(frame, bound_timestamp_s=0.0075), 19)

    assert event == "event"
    assert captured["candidate"] is frame
    assert captured["require_product_data"] is True
    context = captured["context"]
    assert context.storage_frame_id == 19
    assert context.bound_scene_frame_id == context.current_scene_frame_id == 7
    assert context.bound_scene_time_s == context.current_scene_time_s == pytest.approx(0.0075)
    assert context.pinned is False


def test_event_loader_recovers_bound_time_from_acquisition_lineage(monkeypatch) -> None:
    frame = _result_frame()
    captured = {}

    def fake_build_replay_event(_candidate, context, *, require_product_data):
        captured["context"] = context
        assert require_product_data is True
        return "event"

    monkeypatch.setattr(replay_cli, "build_replay_event", fake_build_replay_event)

    assert replay_cli._event_from_frame(_Provider(frame), 19) == "event"
    assert captured["context"].bound_scene_time_s == pytest.approx(0.007)


def test_default_selection_uses_physical_result_order_and_closes_provider(monkeypatch) -> None:
    provider = SimpleNamespace(closed=False)
    provider.close = lambda: setattr(provider, "closed", True)
    selected = []

    monkeypatch.setattr(replay_cli, "_resolve_frames_dir", lambda _source: Path("frames"))
    monkeypatch.setattr(
        replay_cli,
        "load_frame_manifest",
        lambda _path: SimpleNamespace(frame_ids=(2, 9)),
    )
    monkeypatch.setattr(replay_cli, "Hdf5Provider", lambda *_args, **_kwargs: provider)
    monkeypatch.setattr(
        replay_cli,
        "_probe_selection_key",
        lambda _provider, frame_id: {
            2: (10.0, "run-new", "acquisition-new", "int:2"),
            9: (5.0, "run-old", "acquisition-old", "int:9"),
        }[frame_id],
    )

    def fake_event(_provider, frame_id):
        selected.append(frame_id)
        return "event"

    monkeypatch.setattr(replay_cli, "_event_from_frame", fake_event)

    assert replay_cli.load_replay_event("scenario") == "event"
    assert selected == [2]
    assert provider.closed is True


def test_probe_selection_key_validates_and_orders_native_results() -> None:
    sensing_by_frame = {
        2: _probe_sensing(
            reference_time_s=0.001,
            processing_run_id=_PROCESSING_RUN_1,
            acquisition_id=_ACQUISITION_1,
        ),
        3: _probe_sensing(
            reference_time_s=0.002,
            processing_run_id=_PROCESSING_RUN_1,
            acquisition_id=_ACQUISITION_1,
        ),
        4: _probe_sensing(
            reference_time_s=0.002,
            processing_run_id=_PROCESSING_RUN_2,
            acquisition_id=_ACQUISITION_1,
        ),
        5: _probe_sensing(
            reference_time_s=0.002,
            processing_run_id=_PROCESSING_RUN_2,
            acquisition_id=_ACQUISITION_2,
        ),
        6: _probe_sensing(
            reference_time_s=0.002,
            processing_run_id=_PROCESSING_RUN_2,
            acquisition_id=_ACQUISITION_2,
        ),
    }
    provider = _ProbeProvider(sensing_by_frame)

    keyed_frames = [
        (replay_cli._probe_selection_key(provider, frame_id), frame_id)
        for frame_id in sensing_by_frame
    ]

    assert keyed_frames[0][0] == (
        0.001,
        _PROCESSING_RUN_1,
        _ACQUISITION_1,
        "int:2",
    )
    assert keyed_frames == sorted(keyed_frames)
    assert max(keyed_frames)[1] == 6


def test_probe_selection_key_fails_closed_on_invalid_complete_contract() -> None:
    sensing = _probe_sensing(
        reference_time_s=0.0015,
        processing_run_id=_PROCESSING_RUN_1,
        acquisition_id=_ACQUISITION_1,
    )
    sensing["temporal"]["result"]["processing_run_id"] = "not-a-uuid"

    with pytest.raises(
        replay_cli.ReplaySourceError,
        match=r"complete native sensing frame 8.*processing_run_id",
    ):
        replay_cli._probe_selection_key(_ProbeProvider({8: sensing}), 8)


def test_default_selection_fails_closed_on_malformed_complete_result(monkeypatch) -> None:
    provider = SimpleNamespace(closed=False)
    provider.close = lambda: setattr(provider, "closed", True)

    monkeypatch.setattr(replay_cli, "_resolve_frames_dir", lambda _source: Path("frames"))
    monkeypatch.setattr(
        replay_cli,
        "load_frame_manifest",
        lambda _path: SimpleNamespace(frame_ids=(1, 2)),
    )
    monkeypatch.setattr(replay_cli, "Hdf5Provider", lambda *_args, **_kwargs: provider)

    def fake_probe(_provider, frame_id):
        if frame_id == 2:
            raise replay_cli.ReplaySourceError("complete native sensing frame 2 is malformed")
        return (1.0, "run", "acquisition", "int:1")

    monkeypatch.setattr(replay_cli, "_probe_selection_key", fake_probe)

    with pytest.raises(replay_cli.ReplaySourceError, match="frame 2 is malformed"):
        replay_cli.load_replay_event("scenario")
    assert provider.closed is True


def test_help_imports_without_gui_renderer_or_plot_dependencies(tmp_path: Path) -> None:
    script = r"""
import importlib.abc
import sys

sys.path.insert(0, sys.argv[1])

blocked = {
    "OpenGL", "PIL", "PyQt5", "PyQt6", "PySide2", "PySide6", "drjit",
    "matplotlib", "mitsuba", "open3d", "pygfx", "pyqtgraph", "qtpy",
    "rendercanvas", "scipy", "sionna", "sklearn", "tensorflow", "vtk", "wgpu",
}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError(f"forbidden help import: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
from visualizer.src.sensing import replay_cli

try:
    replay_cli.main(["--help"])
except SystemExit as exc:
    if exc.code != 0:
        raise
else:
    raise AssertionError("--help did not exit through argparse")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(Path(replay_cli.__file__).resolve().parents[3])],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("missing_name", ("matplotlib", "PIL"))
def test_main_converts_absent_sensing_export_dependency_to_install_guidance(
    monkeypatch,
    tmp_path: Path,
    capsys,
    missing_name: str,
) -> None:
    monkeypatch.setattr(replay_cli, "load_replay_event", lambda *_args, **_kwargs: object())
    _fail_replay_export_import(monkeypatch, missing_name)

    with pytest.raises(SystemExit, match="2"):
        replay_cli.main(["frames", "-o", str(tmp_path / "replay")])

    assert "python -m pip install orchav" in capsys.readouterr().err


@pytest.mark.parametrize(
    "missing_name",
    (
        "visualizer.src.sensing.stage_export",
        "matplotlib.backends.backend_agg",
        "numpy",
    ),
)
def test_main_preserves_internal_or_broken_install_import_errors(
    monkeypatch,
    tmp_path: Path,
    missing_name: str,
) -> None:
    monkeypatch.setattr(replay_cli, "load_replay_event", lambda *_args, **_kwargs: object())
    _fail_replay_export_import(monkeypatch, missing_name)

    with pytest.raises(ModuleNotFoundError) as raised:
        replay_cli.main(["frames", "-o", str(tmp_path / "replay")])

    assert raised.value.name == missing_name


def test_main_exports_and_prints_machine_readable_summary(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    event = SimpleNamespace(
        key=SimpleNamespace(reference_time_s=0.125),
        manifest_id="a" * 64,
        scene_binding=SimpleNamespace(storage_frame_id=17),
        view_manifest_id="b" * 64,
        preflight=SimpleNamespace(disposition="ready"),
    )
    captured = {}

    monkeypatch.setattr(replay_cli, "load_replay_event", lambda *_args, **_kwargs: event)

    from visualizer.src.sensing import replay_export

    def fake_export(candidate, destination, *, allow_blocked):
        captured.update(
            candidate=candidate,
            destination=destination,
            allow_blocked=allow_blocked,
        )
        return SimpleNamespace(
            artifacts=(object(), object()),
            blocked_override_used=False,
            event_manifest_id=event.manifest_id,
            event_view_manifest_id=event.view_manifest_id,
            output_directory=destination,
        )

    monkeypatch.setattr(replay_export, "export_replay_session", fake_export)
    destination = tmp_path / "replay"

    assert replay_cli.main(["frames", "-o", str(destination)]) == 0

    assert captured == {
        "allow_blocked": False,
        "candidate": event,
        "destination": destination,
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "artifact_count": 2,
        "blocked_override_used": False,
        "event_manifest_id": "a" * 64,
        "event_view_manifest_id": "b" * 64,
        "output_directory": str(destination),
        "preflight_disposition": "ready",
        "reference_time_s": 0.125,
        "storage_frame_id": 17,
    }
