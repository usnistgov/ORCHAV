"""Bounded Doppler-edge experiment; circular neighborhoods are not a runtime policy.

The candidate wraps Doppler alone. Range retains full-window eligibility.
It is intentionally isolated until CFAR, NMS, clustering and velocity-seam
association can be qualified together; no saved recipe or default is changed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path

import numpy as np

from shared.sensing.detection.cfar import CFAR2DResult, cfar_2d, cluster_detections_2d, nms_peaks_2d


def circular_doppler_cfar(power, *, guard_cells, reference_cells, threshold_scale):
    width = guard_cells[0] + reference_cells[0]
    if 2 * width + 1 > power.shape[0]:
        raise ValueError("Circular neighborhood must not repeat a physical Doppler cell")
    padded = np.pad(power, ((width, width), (0, 0)), mode="wrap")
    result = cfar_2d(
        padded,
        guard_cells=guard_cells,
        reference_cells=reference_cells,
        threshold_scale=threshold_scale,
        edge_policy="full_window",
    )
    return CFAR2DResult(
        **{f.name: getattr(result, f.name)[width:-width].copy() for f in fields(CFAR2DResult)}
    )


def circular_reference(power, guard=(1, 2), reference=(4, 6), scale=7.0):
    """Independent direct sums: modulo only the Doppler index."""
    gd, gr = guard
    rd, rr = reference
    rows, cols = power.shape
    noise = np.zeros_like(power)
    count = np.zeros(power.shape, dtype=np.int32)
    for i in range(rows):
        for j in range(gr + rr, cols - gr - rr):
            vals = [
                float(power[(i + di) % rows, j + dj])
                for di in range(-gd - rd, gd + rd + 1)
                for dj in range(-gr - rr, gr + rr + 1)
                if abs(di) > gd or abs(dj) > gr
            ]
            count[i, j] = len(vals)
            noise[i, j] = math.fsum(vals) / len(vals)
    eligible = count > 0
    threshold = scale * noise
    return CFAR2DResult(eligible & (power > threshold), threshold, noise, eligible, count)


def experiment(saved: Path | None):
    rng = np.random.default_rng(20261014)
    guard = (1, 2)
    ref = (4, 6)
    count = (2 * (guard[0] + ref[0]) + 1) * (2 * (guard[1] + ref[1]) + 1) - (2 * guard[0] + 1) * (
        2 * guard[1] + 1
    )
    alpha = count * (0.001 ** (-1 / count) - 1)
    kwargs = dict(guard_cells=guard, reference_cells=ref, threshold_scale=alpha)
    report = {
        "noise_model": "Independent unit-mean exponential RD power; not window-correlated acquisition noise",
        "threshold_scale": alpha,
        "nominal_pfa": 0.001,
        "training_count": count,
    }
    controls = []
    for position in [0, 1, 3, 5, 32, 60, 62, 63]:
        power = rng.exponential(size=(64, 40))
        power[position, 20] = 1000
        controls.append(
            dict(
                bin=position,
                **{
                    mode: bool(result.detections[position, 20])
                    for mode, result in [
                        ("full", cfar_2d(power, **kwargs)),
                        ("truncate", cfar_2d(power, **kwargs, edge_policy="truncate")),
                        ("circular", circular_doppler_cfar(power, **kwargs)),
                    ]
                },
            )
        )
    report["targets"] = controls
    rates = {mode: [] for mode in ("full", "truncate", "circular")}
    counts = {mode: [0, 0] for mode in ("full", "truncate", "circular")}
    for _ in range(200):
        power = rng.exponential(size=(64, 40))
        for mode in counts:
            result = (
                circular_doppler_cfar(power, **kwargs)
                if mode == "circular"
                else cfar_2d(
                    power, **kwargs, edge_policy="full_window" if mode == "full" else "truncate"
                )
            )
            counts[mode][0] += int(result.detections.sum())
            counts[mode][1] += int(result.eligible.sum())
            rates[mode].append(float(result.detections.sum() / result.eligible.sum()))
    report["noise_only"] = {
        mode: dict(
            passes=a,
            eligible=b,
            empirical_pfa=a / b,
            independent_map_standard_error=float(
                np.std(rates[mode], ddof=1) / np.sqrt(len(rates[mode]))
            ),
            nominal_independent_cell_binomial_standard_error=float(
                np.sqrt((a / b) * (1 - a / b) / b)
            ),
        )
        for mode, (a, b) in counts.items()
    }
    seam = np.zeros((64, 40))
    seam[0, 20] = 100
    seam[-1, 20] = 99
    mask = circular_doppler_cfar(seam, **kwargs).detections
    peaks = nms_peaks_2d(mask, seam)
    clusters = cluster_detections_2d(
        peaks,
        seam,
        np.arange(40) * 0.3,
        np.fft.fftshift(np.fft.fftfreq(64, 0.001)),
        wavelength=299792458 / 28e9,
        cluster_range_gate_m=0.6,
        cluster_velocity_gate_m_s=0.2,
    )
    report["seam"] = {
        "current_nonwrapping_cluster_count": len(clusters),
        "cluster_velocity_gate_m_s": 0.2,
        "current_nonwrapping_nms_peaks": np.argwhere(peaks).tolist(),
        "circular_distance_bins": 1,
        "ordinary_distance_bins": 63,
        "conclusion": "CFAR-only wrapping leaves two reports for one seam-straddling peak; NMS and clustering need an explicit periodic contract.",
    }
    power = rng.exponential(size=(64, 40))
    actual = circular_doppler_cfar(power, **kwargs)
    expected = circular_reference(power, guard, ref, alpha)
    report["independent_reference"] = {
        "max_abs_noise_error": float(np.max(abs(actual.noise - expected.noise))),
        "mask_exact": bool(np.array_equal(actual.detections, expected.detections)),
        "count_exact": bool(
            np.array_equal(actual.training_cell_count, expected.training_cell_count)
        ),
    }
    if saved:
        from shared.sensing.loader import _select_primary_range_doppler, load_frames_sensing

        cases = []
        for frame in load_frames_sensing(saved):
            s = frame["_typed_sensing"]
            _, values, doppler, ranges, _ = _select_primary_range_doppler(s)
            power = np.asarray(values[0])
            power = abs(power) ** 2 if np.iscomplexobj(power) else power
            recipe = s["config_resolved"]["processing_recipe"]
            valid = (ranges >= recipe["min_detection_range_m"]) & (
                ranges <= recipe["max_detection_range_m"]
            )
            peak = np.unravel_index(np.argmax(np.where(valid[None, :], power, -1)), power.shape)
            opts = dict(
                guard_cells=(
                    recipe["cfar_rd_guard_cells_doppler"],
                    recipe["cfar_rd_guard_cells_range"],
                ),
                reference_cells=(
                    recipe["cfar_rd_ref_cells_doppler"],
                    recipe["cfar_rd_ref_cells_range"],
                ),
                threshold_scale=recipe["cfar_threshold_scale"],
            )
            modes = {
                name: fun()
                for name, fun in [
                    ("full", lambda: cfar_2d(power, **opts)),
                    ("truncate", lambda: cfar_2d(power, **opts, edge_policy="truncate")),
                    ("circular", lambda: circular_doppler_cfar(power, **opts)),
                ]
            }
            cases.append(
                dict(
                    source_step=s["source_frame_step"],
                    peak_bins=list(map(int, peak)),
                    peak_doppler_hz=float(doppler[peak[0]]),
                    saved_reports=len(s["detections"]),
                    authored_edge_guard_bins=recipe["pre_cfar_doppler_edge_guard_bins"],
                    comparisons={
                        name: dict(
                            eligible=bool(r.eligible[peak]),
                            pass_at_peak=bool(r.detections[peak]),
                            nms_count=int(nms_peaks_2d(r.detections, power).sum()),
                        )
                        for name, r in modes.items()
                    },
                )
            )
        report["saved_micro_motion"] = cases
        report["saved_scope"] = (
            "CFAR comparison before the separately authored pre-CFAR Doppler edge exclusion; unchanged publication may still exclude these bins."
        )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--saved", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(experiment(a.saved), indent=2) + "\n")


if __name__ == "__main__":
    main()
