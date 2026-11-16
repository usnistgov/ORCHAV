"""Prepare an independent coverage run using canonical scenario copy and generation.

Scene assets and references are relocated by Scenario Builder persistence. The
source recording is never a generator destination. The normal generator retains
ownership of frame/coverage transactions and cancellation cleanup.
"""

import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from shared.scenarios.yaml import validate_scenario_data

from ..authoring.document import ScenarioDocument
from ..authoring.domain import ScenarioSourceSnapshot
from ..authoring.persistence import (
    create_scenario_copy,
    save_document,
)


def parse_heights(text: str) -> list[float]:
    """Accept explicit metres or an inclusive start:stop:step regular sweep."""
    if ":" in text:
        fields = [float(v.strip()) for v in text.split(":")]
        if len(fields) != 3:
            raise ValueError("Height sweep must be start:stop:step, in metres")
        low, high, step = fields
        if not np.isfinite(fields).all() or step <= 0 or high < low:
            raise ValueError("Heights need a finite increasing sweep with positive step")
        count = int(round((high - low) / step)) + 1
        if not 1 <= count <= 512 or not math.isclose(low + (count - 1) * step, high, abs_tol=1e-6):
            raise ValueError("Use at most 512 heights and a step that reaches the final height")
        heights = [low + i * step for i in range(count)]
    else:
        heights = [float(v.strip()) for v in text.split(",")]
    if not 1 <= len(heights) <= 512 or not np.isfinite(heights).all():
        raise ValueError("Specify 1 to 512 finite heights in metres")
    if any(b <= a for a, b in zip(heights, heights[1:])):
        raise ValueError("Heights must be strictly increasing")
    return heights


def coverage_mapping(
    source: dict, *, heights, resolution, bbox, samples, depth, mechanisms
) -> dict:
    """Build a validated coverage-only acquisition at the saved scenario start."""
    mapping = deepcopy(source)
    # Keep the authored timeline valid for moving-actor preparation. The existing
    # coverage-only pipeline evaluates its initial scene and publishes one frame.
    mapping["data"] = {"mode": "files", "files": {"directory": "frames", "format": "hdf5"}}
    mapping.setdefault("raytracing", {})["enabled"] = False
    if isinstance(mapping.get("sensing"), dict):
        mapping["sensing"]["enabled"] = False
    mapping["generator_summary"] = {"enabled": False}
    coverage = dict(mapping.get("coverage") or {})
    mapping["coverage"] = coverage
    coverage.update(enabled=True, stride=1)
    coverage["grid"] = {
        "bbox_xy": bbox,
        "resolution_m": list(resolution),
        "heights_m": list(heights),
    }
    solver = dict(coverage.get("solver") or {})
    coverage["solver"] = solver
    solver.update(samples_per_tx=samples, max_depth=depth, **mechanisms)
    coverage["save"] = {"data": {"enabled": True}, "figure": {"enabled": False}}
    validate_scenario_data(mapping)
    if not mapping.get("actors", {}).get("tx"):
        raise ValueError("Coverage requires a saved transmitter")
    return mapping


def estimate_work(mapping: dict) -> str:
    """Give structural work/memory estimates, never an invented time prediction."""
    cov = mapping["coverage"]
    heights = len(cov["grid"]["heights_m"])
    tx = len(mapping["actors"]["tx"])
    rays = cov["solver"]["samples_per_tx"] * tx * heights
    bbox = cov["grid"]["bbox_xy"]
    cells = None
    if bbox != "auto":
        cells = (
            math.prod(
                math.ceil((hi - lo) / d) for (lo, hi), d in zip(bbox, cov["grid"]["resolution_m"])
            )
            * heights
        )
        if cells > 20_000_000 or cells * tx > 100_000_000:
            raise ValueError(
                "Reduce the grid: maximum 20 million cells and 100 million TX-cell values"
            )
    result = f"{heights} height solve(s) · {tx} TX · {rays:,} launched samples total."
    if cells is not None:
        result += f"\nApproximately {cells:,} cells; {cells * tx * 4 / 2**20:.2f} MiB canonical gain, before solver/scene memory."
        if cells > 2_000_000:
            result += (
                "\nThis exceeds the 3D inspector limit; ordinary height planes remain available."
            )
    else:
        result += "\nGrid dimensions come from scene bounds; memory depends on scene size."
    return result + "\nRuntime depends on scene and interaction depth; this is not a time estimate."


def prepare_run(source: Path, destination: Path, settings: dict) -> Path:
    """Copy only authored inputs into a new directory and save the derived recipe."""
    destination = destination.expanduser().absolute()
    document = create_scenario_copy(source, destination)
    base = document.scenario.source_snapshot
    mapping = coverage_mapping(base.to_mapping(), **settings)
    estimate_work(mapping)
    snapshot = ScenarioSourceSnapshot.from_mapping(
        mapping,
        actor_bindings=base.actor_bindings,
        group_bindings=base.group_bindings,
    )
    derived = ScenarioDocument(
        replace(document.scenario, source_snapshot=snapshot),
        path=document.path,
        ownership=document.ownership,
    )
    # Reserve this exact destination; never reuse recordings or another job's run.
    destination.mkdir(parents=True, exist_ok=False)
    return save_document(derived)
