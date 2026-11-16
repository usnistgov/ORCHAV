"""Check actual ordinary-generator recordings against authored scene geometry.

Run as ``python -m examples.visualizer_experiments.validate_recordings --input DIR``.
This reads canonical HDF5 through the normal provider. It never writes frames.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import numpy as np

from shared.frames.providers import Hdf5Provider

from .build_scenarios import SCENARIO_NAMES

C_M_S = 299_792_458.0


def _expected_lengths(name: str) -> list[list[float]]:
    if name == "arrival_courtyard":
        return [[8.0, 10.0, math.sqrt(208)], [math.sqrt(68), math.sqrt(80), math.sqrt(260)]]
    if name == "angular_corridor":
        return [[8.0, 10.0, 10.0, 20.0]]
    return [[16.0 if name == "monostatic_range_locus" else 2 * math.sqrt(73)]]


def validate_recording(root: Path, name: str, *, check_viewer: bool = False) -> dict:
    """Verify one generated bundle and report measured values, not fixture data."""
    provider = Hdf5Provider(root / name)
    try:
        sensing_example = name.endswith("range_locus")
        expected_steps = list(range(8)) if sensing_example else [0]
        assert provider.list_frames() == expected_steps, "Unexpected frame sequence"
        frame = provider.load_frame(expected_steps[-1])
        expected_lengths = _expected_lengths(name)
        assert len(frame.tx_rx_pairs) == len(expected_lengths), "Unexpected pair count"
        pairs = []
        for pair_index, expected in enumerate(expected_lengths):
            start, end = frame.pair_path_offsets[pair_index : pair_index + 2]
            delays = np.asarray(frame.delays_ns[start:end], dtype=float)
            observed_lengths = delays * 1e-9 * C_M_S
            np.testing.assert_allclose(
                np.sort(observed_lengths), np.sort(expected), rtol=0, atol=1e-3
            )
            assert np.all(np.isfinite(frame.path_loss_db[start:end]))
            pairs.append(
                {
                    "pair_index": pair_index,
                    "delay_ns": delays.tolist(),
                    "delay_equivalent_length_m": observed_lengths.tolist(),
                    "aoa_az_deg": frame.aoa_az_deg[start:end].tolist(),
                }
            )
        report = {"name": name, "frames": expected_steps, "paths_by_pair": pairs}
        if name == "angular_corridor":
            np.testing.assert_allclose(
                np.sort(frame.aoa_az_deg),
                [
                    0,
                    180 - math.degrees(math.atan2(3, 4)),
                    180,
                    180 + math.degrees(math.atan2(3, 4)),
                ],
                rtol=0,
                atol=1e-3,
            )
        if sensing_example:
            for step in range(7):
                assert provider.load_frame(step).sensing is None, "Unexpected early sensing result"
            sensing = frame.sensing
            assert sensing is not None
            assert sensing["frame_role"] == "cpi_end"
            assert sensing["processing_strategy"] == "coherent_cpi_fft"
            temporal = sensing["temporal"]
            acquisition = temporal["acquisition"]
            assert acquisition["observed_sample_count"] == 8
            assert acquisition["valid_sample_count"] == 8
            solves = {sample["source"]["solve_id"] for sample in acquisition["samples"]}
            assert len(solves) == 1, "Expected a single coherent Sionna solve"
            assert temporal["display_binding"]["scene_frame_id"] == 0
            assert temporal["result"]["reference_time_s"] == 0.0
            rd = np.asarray(sensing["range_doppler_by_pair"])[0]
            assert rd.shape == (8, 128)
            assert np.all(np.isfinite(rd)) and np.any(np.abs(rd) > 0)
            doppler_bin, range_bin = (
                int(v) for v in np.unravel_index(np.abs(rd).argmax(), rd.shape)
            )
            range_axis = np.asarray(sensing["range_axis_m"])
            doppler_axis = np.asarray(sensing["doppler_axis_hz"])
            range_m = float(range_axis[range_bin])
            expected_range = expected_lengths[0][0]
            if name == "monostatic_range_locus":
                expected_range /= 2
            spacing = float(range_axis[1] - range_axis[0])
            assert abs(range_m - expected_range) <= spacing / 2
            assert doppler_axis[doppler_bin] == 0.0
            report["sensing"] = {
                "published_frame": frame.frame_index,
                "bound_scene_frame": 0,
                "reference_time_s": 0.0,
                "coherent_samples": 8,
                "rt_solves": len(solves),
                "range_mode": sensing["range_mode"],
                "peak_cell_doppler_range": [doppler_bin, range_bin],
                "peak_range_m": range_m,
                "peak_doppler_hz": float(doppler_axis[doppler_bin]),
                "range_sample_spacing_m": spacing,
                "geometric_range_m": expected_range,
            }
        if check_viewer and sensing_example:
            report["viewer_binding"] = _validate_range_viewer_binding(
                provider, range_bin, doppler_bin
            )
        return report
    finally:
        provider.close()


def _validate_range_viewer_binding(
    provider: Hdf5Provider, range_bin: int, doppler_bin: int
) -> dict:
    """Exercise the real dashboard resolver and range kernel, without a window."""
    from dataclasses import replace
    from types import SimpleNamespace

    from visualizer.src.experiments.range_geometry import build_range_locus
    from visualizer.src.sensing.dashboard_data import DashboardSession, load_dashboard_request
    from visualizer.src.sensing.selection import SensingSelection, inspection_cell, selection_scope
    from visualizer.src.services.sensing_service import SensingService

    service = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    request = load_dashboard_request(service, 0, "diagnostics", "primary", False, "")
    assert request is not None
    assert request.detection_stage_bundle is not None
    assert request.detection_stage_bundle.available, request.detection_stage_bundle.reason
    request = replace(request, source_identity=DashboardSession.provider_identity(provider))
    assert request.state_step == request.display_step == request.effective_display_step == 0
    sensing = request.display_frame["sensing"]
    provenance = request.display_frame["_sensing_display_provenance"]
    assert provenance["storage_frame_id"] == 7
    cell = inspection_cell(request, range_bin, doppler_bin)
    assert cell is not None
    selection = SensingSelection(
        selection_scope(sensing),
        sensing["temporal"]["acquisition"]["acquisition_id"],
        cell=cell,
        source=request.source_identity,
    )
    locus = build_range_locus(request, selection)
    assert len(locus.surfaces) == 2
    return {
        "main_scene_step": request.state_step,
        "bound_scene_step": request.effective_display_step,
        "storage_frame": provenance["storage_frame_id"],
        "surface_count": len(locus.surfaces),
        "detector_stage_inspection_available": request.detection_stage_bundle.available,
        "label": locus.label,
        "foci_m": locus.foci.tolist(),
    }


def main() -> None:
    """Report bounded checks on the actual recordings as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--check-viewer",
        action="store_true",
        help="Also exercise the dashboard/range service seam (requires viewer dependencies)",
    )
    args = parser.parse_args()
    reports = [
        validate_recording(args.input, name, check_viewer=args.check_viewer)
        for name in SCENARIO_NAMES
    ]
    sys.stdout.write(
        json.dumps(
            {
                "validator_sionna_rt_version": importlib.metadata.version("sionna-rt"),
                "scenarios": reports,
                "scope": "Authored geometric path lengths, pair isolation, and static coherent-CPI publication; not amplitude, detection, or localization validation.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
