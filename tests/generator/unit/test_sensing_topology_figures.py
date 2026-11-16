"""Sensing-only topology labels; kept outside the non-sensing authoring tests."""

import json

import numpy as np
import pytest
import yaml

from generator.figures.topology import _node_marker, load_topology, main, render_topology
from tests.generator.unit.test_topology_figures import authored  # noqa: F401


def test_colocated_sensing_pair_keeps_both_source_identities(request):
    path, cfg = request.getfixturevalue("authored")
    cfg["sensing"] = dict(enabled=True, coherent_clutter_mode="none")
    path.write_text(yaml.safe_dump(cfg))
    case = load_topology(path)
    radar, target = case["actors"]
    assert radar["label"] == "Monostatic radar"
    assert radar["members"] == ["RadarTX", "RadarRX"]
    assert _node_marker(radar) == "^"
    assert target["name"] == "Reflector"
    cfg["actors"]["rx"][0]["mobility"]["position_m"][0] = 0.1
    path.write_text(yaml.safe_dump(cfg))
    assert len(load_topology(path)["actors"]) == 3


def test_moving_target_uses_shared_position_keys_with_stationary_radar(
    request, tmp_path, monkeypatch
):
    import matplotlib.figure

    path, cfg = request.getfixturevalue("authored")
    cfg["sensing"] = dict(enabled=True, coherent_clutter_mode="none")
    path.write_text(yaml.safe_dump(cfg))
    labels = []
    savefig = matplotlib.figure.Figure.savefig

    def capture(fig, *args, **kwargs):
        labels.extend(t.get_text() for legend in fig.legends for t in legend.get_texts())
        return savefig(fig, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture)
    render_topology(load_topology(path), tmp_path / "radar.png", dpi=30)
    assert "1  Monostatic radar" in labels
    assert "Stationary node" not in labels
    assert labels.count("Start position") == labels.count("End position") == 1


def test_non_sensing_colocation_and_bistatic_nodes_keep_their_symbols(request):
    path, cfg = request.getfixturevalue("authored")
    assert [_node_marker(a) for a in load_topology(path)["actors"][:2]] == ["D", "D"]
    cfg["sensing"] = dict(enabled=True, coherent_clutter_mode="none")
    cfg["actors"]["rx"][0]["mobility"]["position_m"][0] = 3
    path.write_text(yaml.safe_dump(cfg))
    assert [_node_marker(a) for a in load_topology(path)["actors"][:2]] == ["D", "D"]


def test_native_proxy_uses_declared_geometry_without_native_solver():
    from pathlib import Path

    from generator.core.target.representation import NativeTargetConfig
    from generator.figures.topology_targets import native_proxy_geometry
    from shared.scenarios.actors import NativeBoxGeometrySpec, TR38901RCSSpec

    config = NativeTargetConfig(
        name="Car",
        representation="tr38901",
        geometry=NativeBoxGeometrySpec(),
        tr38901=TR38901RCSSpec(object_type="vehicle-multi-sp"),
    )
    vertices, faces, source = native_proxy_geometry(
        config, config.preview_metadata(), Path("scenario.yaml")
    )
    np.testing.assert_allclose(np.ptp(vertices, axis=0), [5, 2, 1.6])
    assert faces.shape == (12, 3)
    assert source == Path("scenario.yaml")


@pytest.mark.parametrize(
    "override, expected_option, has_detail",
    [(None, True, True), ("true", True, True), ("false", False, False), ("auto", "auto", False)],
)
def test_cli_motion_detail_override_preserves_source_and_records_choice(
    request, tmp_path, override, expected_option, has_detail
):
    path, cfg = request.getfixturevalue("authored")
    # Auto hides detail for this non-sensing scene, even when YAML enables it.
    cfg["generator_summary"] = {"visualization": {"topology_motion_detail": True}}
    path.write_text(yaml.safe_dump(cfg))
    source_bytes = path.read_bytes()
    output = tmp_path / "topology.png"
    argv = ["--scenario", str(path), "--output", str(output), "--dpi", "30"]
    if override is not None:
        argv.extend(["--motion-detail", override])

    assert main(argv) == 0

    receipt = json.loads(output.with_suffix(".json").read_text())
    assert path.read_bytes() == source_bytes
    assert receipt["topology_options"]["motion_detail"] == expected_option
    assert bool(receipt["motion_details"]) is has_detail
    if override is None:
        assert "--motion-detail" not in receipt["command"]
    else:
        choice = receipt["command"].index("--motion-detail")
        assert receipt["command"][choice + 1] == override
