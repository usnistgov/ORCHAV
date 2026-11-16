"""Validate authored demo shots in the existing camera-sequence format."""

from copy import deepcopy
from math import isfinite

from ..types.camera_state import CameraState
from .camera_motion import validate_pose


def scenario_tour(scenario):
    """Return a small validated tour, without reading frames or moving the camera."""
    defaults = getattr(scenario, "view_defaults", None) or {}
    tour = defaults.get("demo_tour") if isinstance(defaults, dict) else None
    if tour is None:
        return None
    if not isinstance(tour, dict):
        raise ValueError("The scenario demo tour must be a mapping")
    views, shots = tour.get("views"), tour.get("shots")
    if not isinstance(views, dict) or not 2 <= len(views) <= 32:
        raise ValueError("A demo tour needs between 2 and 32 named views")
    if not isinstance(shots, list) or not 2 <= len(shots) <= 64:
        raise ValueError("A demo tour needs between 2 and 64 shots")
    data = deepcopy(tour)
    for name, view in data["views"].items():
        if not isinstance(name, str) or not name.strip() or not isinstance(view, dict):
            raise ValueError("Invalid demo view name or pose")
        try:
            view["pose"] = validate_pose(CameraState.from_dict(view["pose"])).to_dict()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid demo camera pose for {name!r}") from exc
        actor = view.setdefault("actor", None)
        if actor is not None:
            if not isinstance(actor, dict) or actor.get("type") not in {"tx", "rx", "target"}:
                raise ValueError("Invalid demo camera actor")
            if actor["type"] in {"tx", "rx"} and (
                not isinstance(actor.get("index"), int) or actor["index"] < 0
            ):
                raise ValueError("Invalid demo radio actor index")
    for shot in data["shots"]:
        if not isinstance(shot, dict) or shot.get("view") not in views:
            raise ValueError("A demo shot references an unavailable view")
        for key in ("hold", "move"):
            try:
                value = float(shot[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Invalid demo shot timing") from exc
            if not isfinite(value) or not 0 <= value <= 120:
                raise ValueError("Demo shot timings must be between 0 and 120 seconds")
            shot[key] = value
    if sum(shot["hold"] + (shot["move"] if i else 0) for i, shot in enumerate(data["shots"])) <= 0:
        raise ValueError("A demo tour needs a nonzero duration")
    # Explicitly return to the first pose before the sequence clock wraps.
    # CameraPresentation otherwise loops by jumping from its last pose.
    if data["shots"][-1]["view"] != data["shots"][0]["view"]:
        data["shots"].append({"view": data["shots"][0]["view"], "hold": 0.0, "move": 4.0})
    label = str(data.get("label") or "Scenario tour").strip()
    data["label"] = label[:100] or "Scenario tour"
    return data
