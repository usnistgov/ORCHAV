"""CPU-only inspection quantities and source-bound bookmarks for saved volumes.

All maps live at tested target positions. Report positions remain separate;
visibility never changes matching or the measured support.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

DETAIL_LABELS = ("Match, no extras", "Match with extras", "Reports, no match", "No reports")
DETAIL_COLORS = ("#1accf2", "#bba2ff", "#f7cd55", "#f2731a")
PAIR_LABELS = ("Matched in both", "Matched only in A", "Matched only in B", "Matched in neither")
PAIR_COLORS = ("#1accf2", "#bba2ff", "#f7cd55", "#f2731a")
FOCUS_LABELS = {
    "all": "All measured positions",
    "hit": "Matched cells",
    "miss": "Missed cells",
    "unmatched": "Reports with no current match",
    "extras": "Matches with extra reports",
    "error": "Nearest position error ≥ threshold",
    "count": "Unmatched reports ≥ K",
    "blocked": "Centre path blocked, with reports",
    "all_blocked": "All recorded probes blocked, with reports",
    "clear": "Centre path clear, without match",
}


def detail_grid(snapshot):
    hits, count = snapshot["outcomes"], snapshot["report_counts"]
    return np.where(
        np.isfinite(hits),
        np.where(hits == 1, np.where(count == 1, 0, 1), np.where(count > 0, 2, 3)),
        np.nan,
    )


def extras_grid(snapshot):
    return np.where(
        np.isfinite(snapshot["outcomes"]), snapshot["report_counts"] - snapshot["outcomes"], np.nan
    )


def category_counts(values, labels=DETAIL_LABELS):
    return {label: int(np.sum(values == i)) for i, label in enumerate(labels)}


def focus_mask(reader, heading, snapshot, predicate="all", threshold=5.0, minimum=1):
    measured = np.isfinite(snapshot["outcomes"])
    hits = snapshot["outcomes"] == 1
    reports = snapshot["report_counts"] > 0
    if predicate == "all":
        return measured.copy()
    if predicate == "hit":
        return measured & hits
    if predicate == "miss":
        return measured & ~hits
    if predicate == "unmatched":
        return measured & reports & ~hits
    if predicate == "extras":
        return measured & hits & (extras_grid(snapshot) > 0)
    if predicate == "error":
        return measured & (snapshot["nearest_error_m"] >= threshold)
    if predicate == "count":
        return measured & (extras_grid(snapshot) >= minimum)
    geometry = reader.geometry_grid(heading)
    if predicate == "blocked":
        return measured & (geometry == 1) & reports
    if predicate == "clear":
        return measured & (geometry == 0) & ~hits
    if predicate == "all_blocked":
        mask = np.zeros(reader.shape, bool)
        for (ih, z, y, x), trial in reader.cells.items():
            if ih == heading and trial.get("target_probe_count", 0) > 0:
                mask[z, y, x] = trial.get("blocked_target_probes") == trial["target_probe_count"]
        return measured & mask & reports
    raise ValueError("Unknown evidence focus")


def slice_data(values, axes, orientation, index):
    """Return row-major vertical/horizontal values and exact half-cell extents."""
    horizontal, vertical, normal = {
        "xy": ("x", "y", "z"),
        "xz": ("x", "z", "y"),
        "yz": ("y", "z", "x"),
    }[orientation]
    if not 0 <= index < len(axes[normal]):
        raise ValueError("Slice outside the sampled grid")
    plane = np.take(values, index, axis={"x": 2, "y": 1, "z": 0}[normal])

    def edges(name):
        a = axes[name]
        step = float(a[1] - a[0])
        return float(a[0] - step / 2), float(a[-1] + step / 2)

    return plane, horizontal, vertical, normal, (*edges(horizontal), *edges(vertical))


def cell_support(reader, key):
    """Exact display-cell bounds; neither target extent nor estimator uncertainty."""
    centre = np.array([reader.axes[n][i] for n, i in zip("xyz", key[1:][::-1], strict=True)])
    spacing = np.array([reader.axes[n][1] - reader.axes[n][0] for n in "xyz"])
    return centre, spacing, centre - spacing / 2, centre + spacing / 2


def report_details(model, key, snapshot):
    """Expose all saved rows, their scoped IDs and the evaluator's chosen match."""
    trial = model.reader.cells.get(key)
    if trial is None or trial["status"] != "measured":
        return []
    selected = int(snapshot["selected_report"][key[1:]])
    selected_id = model.report_ids[selected] if selected >= 0 else None
    reference = np.asarray(trial["position_m"])
    _, _, lower, upper = cell_support(model.reader, key)
    return [
        dict(
            report,
            matched=report["detection_id"] == selected_id,
            error_m=(
                float(np.linalg.norm(np.asarray(report["position_m"]) - reference))
                if report["position_m"] is not None
                else None
            ),
            scoped_id=f"{model.reader.identity}:{trial['id']}:{report['detection_id']}",
            inside_sample_cell=(
                bool(
                    np.all(np.asarray(report["position_m"]) >= lower)
                    and np.all(np.asarray(report["position_m"]) <= upper)
                )
                if report["position_m"] is not None
                else None
            ),
        )
        for report in model.rows[trial["id"]]
    ]


def zero_radial_polygon(sensor, velocity, axes):
    """Intersect dot(v,p-s)=0 with the sample-cell bounding box; no RT involved."""
    s, v = np.asarray(sensor, float), np.asarray(velocity, float)
    if np.linalg.norm(v) == 0:
        return np.empty((0, 3))
    bounds = [
        (
            float(axes[n][0] - (axes[n][1] - axes[n][0]) / 2),
            float(axes[n][-1] + (axes[n][1] - axes[n][0]) / 2),
        )
        for n in ("x", "y", "z")
    ]
    corners = np.array([[x, y, z] for x in bounds[0] for y in bounds[1] for z in bounds[2]])
    points = []
    for i, a in enumerate(corners):
        for j, b in enumerate(corners):
            if j <= i or np.count_nonzero(a != b) != 1:
                continue
            fa, fb = np.dot(v, a - s), np.dot(v, b - s)
            if abs(fa) < 1e-9:
                points.append(a)
            if abs(fb) < 1e-9:
                points.append(b)
            if fa * fb < 0:
                points.append(a + (b - a) * fa / (fa - fb))
    if not points:
        return np.empty((0, 3))
    points = np.unique(np.round(points, 10), axis=0)
    if len(points) < 3:
        return np.empty((0, 3))
    centre = points.mean(axis=0)
    u = points[0] - centre
    u /= np.linalg.norm(u)
    w = np.cross(v / np.linalg.norm(v), u)
    return points[np.argsort(np.arctan2((points - centre) @ w, (points - centre) @ u))]


def fixed_sensor_orientation(config):
    """Resolve a fixed RX pose from the recorded scenario's schema, in radians.

    Version 2 explicitly defines an omitted orientation as fixed zero. Legacy
    configurations still require a stated fixed yaw; never infer it from motion.
    Consumers of native array data must also verify the saved device pose.
    """
    receivers = config.get("actors", {}).get("rx", [])
    if len(receivers) != 1:
        return None
    orientation = receivers[0].get("orientation", {})
    if config.get("schema_version") == 2:
        from shared.scenarios import FixedOrientationSpec, RxActorSpec

        spec = RxActorSpec.model_validate(receivers[0]).orientation
        if not isinstance(spec, FixedOrientationSpec):
            return None
        return np.deg2rad([spec.yaw_deg, spec.pitch_deg, spec.roll_deg])
    if orientation.get("type") != "fixed" or "yaw_deg" not in orientation:
        return None
    return np.deg2rad(
        [orientation["yaw_deg"], orientation.get("pitch_deg", 0), orientation.get("roll_deg", 0)]
    )


def sensor_forward(reader):
    """Use the recorded fixed RX convention, including versioned schema defaults."""
    import yaml

    orientation = fixed_sensor_orientation(yaml.safe_load(reader.scene_path.read_text()))
    if orientation is None:
        return None
    yaw, pitch, _ = orientation
    return np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])


def _heading_identity(heading):
    return (heading["name"], tuple(heading["velocity_m_s"]))


class PairedVolume:
    """Verified exact-grid pair; each arm is evaluated with the same criterion."""

    def __init__(self, a, b, *, strict_experiment=True):
        self.a, self.b = a, b
        if strict_experiment:
            for source, other in ((a, b), (b, a)):
                declared = source.metadata.get("comparison_volume")
                if not declared or (source.path.parent / declared).resolve() != other.path.parent:
                    raise ValueError("Volumes are not a mutually declared pair")
        if a.metadata["world_frame"] != b.metadata["world_frame"] or any(
            not np.array_equal(a.axes[n], b.axes[n]) for n in ("x", "y", "z")
        ):
            raise ValueError(
                "Pair requires an identical world frame and exact grid; no registration is inferred"
            )
        identities = [_heading_identity(h) for h in b.headings]
        if (
            len(set(map(_heading_identity, a.headings))) != len(a.headings)
            or len(set(identities)) != len(identities)
            or set(map(_heading_identity, a.headings)) != set(identities)
        ):
            raise ValueError("Pair heading/motion identities disagree")
        self.headings = [identities.index(_heading_identity(h)) for h in a.headings]
        self.changed_variable = "Saved conditions; acquisition settings may differ"
        if not strict_experiment:
            # Explicitly selected sources permit deployment/recipe comparisons.
            # Exact coordinates and authored motion above are still mandatory;
            # this view makes no single-variable causal claim.
            return
        # Schema-v1 wall/control authored this reciprocal relation and the
        # changed variable explicitly. Arbitrary adjacent experiments are refused.
        if (
            {a.metadata.get("condition"), b.metadata.get("condition")} != {"wall", "open"}
            or a.metadata.get("wall_bounds_m") != b.metadata.get("wall_bounds_m")
            or not a.metadata.get("wall_bounds_m")
        ):
            raise ValueError("No supported declared changed variable (wall geometry)")
        self.changed_variable = "Wall geometry present / absent"
        for field in ("sensor_position_m", "target_label", "cpi_samples"):
            if a.metadata.get(field) != b.metadata.get(field) or a.metadata.get(field) is None:
                raise ValueError(f"Pair {field} differs")
        import yaml

        configs = []
        for reader in (a, b):
            reader.verify_scene()
            cfg = yaml.safe_load(reader.scene_path.read_text())
            # Scene identity deliberately differs; other settings must agree.
            configs.append({k: cfg.get(k) for k in ("actors", "raytracing", "sensing", "timeline")})
        if configs[0] != configs[1]:
            raise ValueError("Pair sensor/acquisition/processing settings differ")
        # Bound target contracts are supplied by compact packages; legacy arms
        # retain a representative verified native configuration.
        targets = []
        for reader in (a, b):
            contract = reader.metadata.get("inspection_target")
            if contract is None:
                row = next((t for t in reader.cells.values() if t["status"] == "measured"), None)
                if row is None:
                    raise ValueError("Pair has no target reference")
                path = reader.path.parent / row["native"] / "scenario.yaml"
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != row["configuration_sha256"]:
                    raise ValueError("Pair native target reference changed")
                cfg = yaml.safe_load(raw)
                targets_cfg = cfg.get("actors", {}).get("targets", [])
                if len(targets_cfg) != 1:
                    raise ValueError("Pair requires one declared target")
                contract = {k: v for k, v in targets_cfg[0].items() if k != "mobility"}
            targets.append(contract)
        if targets[0] != targets[1]:
            raise ValueError("Pair target model/reference differs")

    def evaluate(self, a_model, b_model, heading, **criteria):
        a = a_model.evaluate(heading, **criteria)
        b = b_model.evaluate(self.headings[heading], **criteria)
        support = np.isfinite(a["outcomes"]) & np.isfinite(b["outcomes"])
        values = np.where(
            support,
            np.where(
                a["outcomes"] == 1,
                np.where(b["outcomes"] == 1, 0, 1),
                np.where(b["outcomes"] == 1, 2, 3),
            ),
            np.nan,
        )
        return dict(
            values=values,
            common=int(support.sum()),
            outside_common=int(support.size - support.sum()),
            a=a,
            b=b,
            counts=category_counts(values, PAIR_LABELS),
        )


def context_snapshot(reader, heading, key, view, camera=None, selected_report=None, pair=None):
    trial = reader.cells.get(key)
    return dict(
        schema_version=1,
        kind="sensing_volume_inspection",
        volume=str(reader.path),
        volume_sha256=reader.identity,
        world_frame=reader.metadata["world_frame"],
        heading=reader.headings[heading],
        trial_id=trial["id"] if trial else None,
        index_zyx=list(key[1:]),
        position_m=[
            float(reader.axes[n][i]) for n, i in zip(("x", "y", "z"), key[1:][::-1], strict=True)
        ],
        view=view,
        camera=camera,
        selected_report=selected_report,
        pair=pair,
        native_evidence=(
            trial.get("native_evidence", "external_reference")
            if trial and trial["status"] == "measured"
            else "not_measured"
        ),
        native_reference=trial.get("native") if trial else None,
    )


def validate_context(reader, state):
    if state.get("schema_version") != 1 or state.get("kind") != "sensing_volume_inspection":
        raise ValueError("Unsupported inspection context")
    if (
        reader.identity != state["volume_sha256"]
        or reader.metadata["world_frame"] != state["world_frame"]
    ):
        raise ValueError("Bookmark source identity changed; explicit rebinding required")
    identities = [_heading_identity(h) for h in reader.headings]
    if len(set(identities)) != len(identities):
        raise ValueError("Ambiguous recorded heading identity")
    try:
        heading = identities.index(_heading_identity(state["heading"]))
    except ValueError as exc:
        raise ValueError("Bookmark heading unavailable") from exc
    indices = state["index_zyx"]
    if len(indices) != 3 or any(
        isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n
        for i, n in zip(indices, reader.shape, strict=True)
    ):
        raise ValueError("Bookmark position is outside the exact grid")
    key = (heading, *indices)
    trial = reader.cells.get(key)
    position = [
        float(reader.axes[n][i]) for n, i in zip(("x", "y", "z"), indices[::-1], strict=True)
    ]
    if (trial["id"] if trial else None) != state["trial_id"] or position != state["position_m"]:
        raise ValueError("Bookmark trial reference changed")
    view = state.get("view", {})
    if not isinstance(view, dict):
        raise ValueError("Bookmark view must be an object")
    for name, allowed in {
        "quantity": {
            "outcomes",
            "reports",
            "details",
            "extras",
            "geometry",
            "pair",
            "contributions",
        },
        "mode": {"slice", "both", "hit", "miss", "mip"},
    }.items():
        if name in view and view[name] not in allowed:
            raise ValueError(f"Unsupported bookmark {name}")
    if view.get("quantity") == "contributions" and not reader.is_fusion:
        raise ValueError("Sensor-contribution bookmarks require a fusion volume")
    for name, value in zip(("x", "y", "altitude"), position, strict=True):
        if name in view and view[name] != value:
            raise ValueError("Bookmark view and trial coordinates disagree")
    for name in ("cell_edges", "unavailable_edges", "readable_sensor"):
        if not isinstance(view.get(name, False), bool):
            raise ValueError(f"Invalid bookmark {name}")
    evaluation = view.get("evaluation", {})
    if evaluation.get("basis", "position") not in {"position", "range_velocity"}:
        raise ValueError("Unsupported bookmark matching basis")
    for name in ("position_gate", "range_gate", "velocity_gate", "error_max"):
        value = evaluation.get(name, 1)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not 0.001 <= value <= 10000
        ):
            raise ValueError("Invalid bookmark evaluation limit")
    inspection = view.get("inspection", {})
    if (
        inspection.get("orientation", "xy") not in {"xy", "xz", "yz"}
        or inspection.get("focus", "all") not in FOCUS_LABELS
    ):
        raise ValueError("Unsupported bookmark slice or focus")
    categories = inspection.get("categories", [True] * 4)
    if len(categories) != 4 or any(not isinstance(v, bool) for v in categories):
        raise ValueError("Invalid bookmark category filters")
    opacity = view.get("opacity", 0.6)
    if (
        not isinstance(opacity, (int, float))
        or not np.isfinite(opacity)
        or not 0.05 <= opacity <= 1
    ):
        raise ValueError("Invalid bookmark opacity")
    camera = state.get("camera")
    if camera is not None:
        for name in ("eye", "lookat", "up"):
            vector = np.asarray(camera.get(name), dtype=float)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError("Invalid bookmark camera")
        if not np.isfinite(camera.get("fov", np.nan)) or not 0 < camera["fov"] < 180:
            raise ValueError("Invalid bookmark camera field of view")
    return key


def save_bookmark(path, state, *, title="Volume inspection", description=""):
    path = Path(path).resolve()
    payload = dict(state)
    payload.update(
        title=title, description=description, volume=os.path.relpath(state["volume"], path.parent)
    )
    if payload.get("pair") and payload["pair"].get("volume"):
        payload["pair"] = dict(
            payload["pair"], volume=os.path.relpath(payload["pair"]["volume"], path.parent)
        )
    encoded = json.dumps(payload, allow_nan=False, indent=2)
    if len(encoded) > 100_000:
        raise ValueError("Inspection bookmark exceeds 100 kB")
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(encoded + "\n")
    temporary.replace(path)


def load_bookmark(path):
    path = Path(path).resolve()
    if path.stat().st_size > 100_000:
        raise ValueError("Inspection bookmark exceeds 100 kB")
    state = json.loads(path.read_text())
    state["volume"] = str((path.parent / state["volume"]).resolve())
    if state.get("pair") and state["pair"].get("volume"):
        state["pair"]["volume"] = str((path.parent / state["pair"]["volume"]).resolve())
    return state
