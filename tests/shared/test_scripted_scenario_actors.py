"""Explicit actor factories stay separate from scenario generation drivers."""

from dataclasses import replace

import pytest

from shared.scenarios import load_scenario_configuration
from shared.scenarios.scripted import load_scripted_actors


@pytest.fixture
def scenario(tmp_path):
    (tmp_path / "scenario.yaml").write_text(
        "schema_version: 2\nscene: {source: sionna, id: etoile}\n"
        "timeline: {steps: 2, duration_s: 1.0}\n"
    )
    return load_scenario_configuration(tmp_path)


@pytest.mark.parametrize(
    "source, error, message",
    [
        ("build_actors = None\n", ValueError, "must define build_actors"),
        ("def build_actors(scenario): return {}\n", TypeError, "must return ActorsSpec"),
    ],
)
def test_actor_factory_requires_the_shared_contract(scenario, source, error, message):
    (scenario.root / "actors.py").write_text(source)
    with pytest.raises(error, match=message):
        load_scripted_actors(scenario)


def test_factory_uses_each_scenarios_own_file_and_timeline(scenario, tmp_path):
    source = """from shared.scenarios import ActorsSpec, TxActorSpec, StationaryMobilitySpec
def build_actors(scenario):
    return ActorsSpec(tx=(TxActorSpec(name=NAME, mobility=StationaryMobilitySpec(
        position_m=(float(scenario.timeline.steps), 0.0, 1.0))),))
"""
    (scenario.root / "actors.py").write_text("NAME = 'first'\n" + source)
    other_root = tmp_path / "second"
    other_root.mkdir()
    (other_root / "actors.py").write_text("NAME = 'second'\n" + source)
    other = replace(scenario, root=other_root)
    first = load_scripted_actors(scenario)
    second = load_scripted_actors(other)
    assert first.tx[0].name == "first"
    assert second.tx[0].name == "second"
    assert first.tx[0].mobility.position_m[0] == scenario.timeline.steps
    assert not scenario.actors.tx
