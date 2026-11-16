"""Authored demo cameras pass through the real scenario loader without data edits."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from shared.scenarios import load_scenario, load_scenario_configuration, validate_scenario_data

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def tour_scenario(tmp_path):
    """Use a shipped scientific scenario and an explicit presentation-only tour."""
    source = ROOT / "scenarios/sensing/cube_radar/stationary/scenario.yaml"
    authored = yaml.safe_load(source.read_text())
    names = ("Overview", "Approach", "Side", "Receiver")
    tour = {
        "label": "Public fixture",
        "views": {
            name: {
                "pose": {"eye": [5 + index, -8, 4], "lookat": [0, 0, 1], "up": [0, 0, 1], "fov": 60}
            }
            for index, name in enumerate(names)
        },
        "shots": [{"view": name, "hold": 2, "move": 1} for name in names],
    }
    tour["views"]["Receiver"]["actor"] = {"type": "rx", "index": 0}
    authored.setdefault("view_defaults", {})["demo_tour"] = tour
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(authored, sort_keys=False))
    return path


def test_tour_loads_through_schema_and_visualizer_configuration(tour_scenario):
    authored = yaml.safe_load(tour_scenario.read_text())
    typed = load_scenario(tour_scenario)
    configuration = load_scenario_configuration(tour_scenario, project_root=ROOT)

    assert typed.view_defaults.demo_tour.label == "Public fixture"
    assert len(typed.view_defaults.demo_tour.views) == 4
    assert configuration.view_defaults["demo_tour"] == authored["view_defaults"]["demo_tour"]
    without_tour = deepcopy(authored)
    without_tour["view_defaults"].pop("demo_tour")
    baseline = validate_scenario_data(without_tour)
    for field in ("timeline", "actors", "raytracing", "sensing", "data"):
        assert getattr(typed, field) == getattr(baseline, field)


@pytest.mark.parametrize("case", ["missing_view", "nonfinite", "zero_duration", "pose", "typo"])
def test_malformed_authored_tours_are_rejected_at_scenario_load(case, tour_scenario):
    authored = yaml.safe_load(tour_scenario.read_text())
    tour = authored["view_defaults"]["demo_tour"]
    if case == "missing_view":
        tour["shots"][1]["view"] = "Unavailable"
    elif case == "nonfinite":
        tour["shots"][1]["move"] = float("nan")
    elif case == "zero_duration":
        for shot in tour["shots"]:
            shot.update(hold=0, move=0)
    elif case == "pose":
        tour["views"][tour["shots"][0]["view"]]["pose"]["up"] = [0, 0, 0]
    else:
        tour["shots"][1]["hoold"] = 5
    with pytest.raises(ValueError, match="demo_tour"):
        validate_scenario_data(authored)
