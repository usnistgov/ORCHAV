"""Small, source-bound coverage view files; never contain or modify solver data."""

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QIODevice, QSaveFile

from shared.coverage.schema import (
    COVERAGE_PER_TX_METRICS,
    COVERAGE_SCALAR_METRICS,
    coverage_metric_base,
    resolve_coverage_tx_index,
)
from shared.coverage.volume import ALL_TRANSMITTERS

from ..types.camera_state import CameraState


def validate_view(state, identity, transmitters, axes):
    """Validate the entire bookmark before replacing a working inspector."""
    if not isinstance(state, dict) or state.get("schema") not in (
        "orchav.coverage-view.v1",
        "orchav.coverage-view.v2",
        "orchav.coverage-view.v3",
    ):
        raise ValueError("Unsupported coverage view format")
    if state.get("source") != list(identity):
        raise ValueError("This view belongs to a different or changed coverage result")
    version3 = state["schema"] == "orchav.coverage-view.v3"
    if version3 and type(state.get("canonical")) is not bool:
        raise ValueError("Invalid coverage metric contract")
    version2 = state["schema"] == "orchav.coverage-view.v2" or (version3 and state["canonical"])
    state = dict(state)
    if not version3:
        # Old volume mode implied threshold membership and a single cyan fill.
        state.setdefault("filter_enabled", state.get("mode") == "iso")
        state.setdefault("colour_mode", "uniform" if state.get("mode") == "iso" else "metric")
        if state.get("mode") == "planar":
            state["mode"] = "slice"
    for field in ("filter_enabled", "contours", "region_enabled", "highlight_cell", "cut_enabled"):
        if type(state.get(field, False)) is not bool:
            raise ValueError(f"Invalid saved {field}")
    cut_axis = state.get("cut_axis", 2)
    if type(cut_axis) is not int or cut_axis not in (0, 1, 2):
        raise ValueError("Invalid saved cutaway axis")
    cut_coordinate = state.get("cut_coordinate", float(axes[cut_axis][-1]))
    if (
        type(cut_coordinate) not in (int, float)
        or not np.isfinite(cut_coordinate)
        or not np.any(np.isclose(axes[cut_axis], cut_coordinate, rtol=0, atol=1e-6))
    ):
        raise ValueError("Saved cutaway is not a recorded plane")
    indices = state.get("selected_indices")
    if indices is not None and (
        not isinstance(indices, list)
        or len(indices) != 3
        or any(
            type(i) is not int or not 0 <= i < len(a) for i, a in zip(indices, axes, strict=True)
        )
    ):
        raise ValueError("Invalid saved selected cell")
    if state.get("smoothing", "none") not in ("none", "linear", "cubic"):
        raise ValueError("Invalid saved smoothing")
    if state.get("colour_mode", "metric") not in ("metric", "uniform"):
        raise ValueError("Invalid saved colour mode")
    if type(state.get("solid_colour", 0)) is not int or not 0 <= state.get("solid_colour", 0) < 3:
        raise ValueError("Invalid saved uniform colour")
    if (
        type(state.get("contour_count", 6)) is not int
        or not 2 <= state.get("contour_count", 6) <= 12
    ):
        raise ValueError("Invalid saved contour count")
    if state.get("region_enabled", False):
        from shared.coverage.spatial import coverage_roi_mask

        if state.get("region_bounds") is None:
            raise ValueError("Missing saved region bounds")
        coverage_roi_mask(axes, state["region_bounds"])
    selections = list(transmitters) + (
        [ALL_TRANSMITTERS] if len(transmitters) > 1 or version2 else []
    )
    if state.get("transmitter") not in selections:
        raise ValueError("Saved transmitter is unavailable")
    if version2:
        if not isinstance(state.get("metric"), str) or state.get("quantity") != state["metric"]:
            raise ValueError("Invalid saved coverage metric")
        base, selector = coverage_metric_base(state["metric"])
        if base not in COVERAGE_PER_TX_METRICS | COVERAGE_SCALAR_METRICS:
            raise ValueError("Invalid saved coverage metric")
        if selector is not None:
            resolve_coverage_tx_index(selector, transmitters, len(transmitters))
        if base == "serving_tx" and state.get("mode") == "mip":
            raise ValueError("Serving identities cannot be maximum-projected")
        index = state.get("region_tx")
        if type(index) is not int or not (-1 if version3 else 0) <= index < len(transmitters):
            raise ValueError("Invalid saved serving region")
        if (
            base in {"path_gain_linear", "serving_path_gain_linear", "rss_w", "sinr_linear"}
            and state.get("low", 0) <= 0
        ):
            raise ValueError("Logarithmic colour limits must be positive")
    elif state.get("quantity") not in ("path_gain_db", "path_loss_db", "rss_dbm"):
        raise ValueError("Invalid saved coverage quantity")
    if state.get("colormap", "viridis") not in ("viridis", "RdYlGn", "RdYlGn_r"):
        raise ValueError("Invalid saved colormap")
    if state.get("mode") not in ("slice", "iso", "mip"):
        raise ValueError("Invalid saved display mode")
    axis = state.get("axis")
    if type(axis) is not int or axis not in (0, 1, 2):
        raise ValueError("Invalid saved slice orientation")
    bound = 1e12 if version2 else 250
    for key, low, high in (
        ("coordinate", -np.inf, np.inf),
        ("threshold", -bound, bound),
        ("opacity", 0.05, 1),
        ("low", -bound, bound),
        ("high", -bound, bound),
        ("rate", 0.2, 20),
    ):
        value = state.get(key)
        if type(value) not in (int, float) or not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid saved {key}")
    if state["low"] >= state["high"]:
        raise ValueError("Saved colour minimum must be below its maximum")
    if not np.any(np.isclose(axes[axis], state["coordinate"], rtol=0, atol=1e-6)):
        raise ValueError("Saved slice is not a recorded plane")
    if any(type(state.get(key)) is not bool for key in ("visible", "loop")):
        raise ValueError("Invalid saved visibility or loop setting")
    if type(state.get("show_transmitter", True)) is not bool:
        raise ValueError("Invalid saved transmitter visibility")
    if type(state.get("readable_transmitter", False)) is not bool:
        raise ValueError("Invalid saved transmitter locator style")
    for key in (
        "cell_edges",
        "unavailable_edges",
        "uniform_colour",
        "threshold_edges",
        "show_nonfinite",
    ):
        if type(state.get(key, False)) is not bool:
            raise ValueError(f"Invalid saved {key}")
    try:
        camera = CameraState.from_dict(state["camera"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid saved camera") from exc
    vectors = np.asarray([camera.eye, camera.lookat, camera.up])
    direction = vectors[1] - vectors[0]
    if (
        not np.isfinite(vectors).all()
        or not np.isfinite(camera.fov_deg)
        or not 0 < camera.fov_deg < 180
        or np.linalg.norm(direction) < 1e-9
        or np.linalg.norm(np.cross(direction, vectors[2])) < 1e-9
    ):
        raise ValueError("Invalid saved camera geometry")
    return state


def read_view(path):
    """Read a bounded JSON document; source/control validation is separate."""
    with Path(path).open("rb") as stream:
        encoded = stream.read(100_001)
    if len(encoded) > 100_000:
        raise ValueError("Coverage view exceeds 100 kB")
    return json.loads(encoded)


def write_view(path, state):
    """Atomically replace a JSON bookmark, protecting the recording itself."""
    path = Path(path)
    if path.suffix.lower() != ".json" or path.resolve() == Path(state["source"][0]):
        raise ValueError("Save the coverage view to a separate .json file")
    encoded = (json.dumps(state, indent=2, allow_nan=False) + "\n").encode("utf-8")
    target = QSaveFile(str(path))
    if not target.open(QIODevice.WriteOnly):
        raise OSError(target.errorString())
    if target.write(encoded) != len(encoded):
        target.cancelWriting()
        raise OSError(target.errorString())
    if not target.commit():
        raise OSError(target.errorString())
