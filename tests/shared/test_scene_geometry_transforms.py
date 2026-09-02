from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from shared.geometry.transforms import (
    UnsupportedLightweightTransformError,
    parse_lightweight_shape_transform,
)


def _shape(transform_body: str = "", *, shape_id: str = "wall") -> ET.Element:
    transform = f'<transform name="to_world">{transform_body}</transform>' if transform_body else ""
    return ET.fromstring(f"""<shape type="ply" id="{shape_id}">
  <string name="filename" value="wall.ply"/>
  {transform}
</shape>""")


def _parse(shape: ET.Element) -> dict[str, object]:
    return parse_lightweight_shape_transform(
        shape,
        source_xml="fixtures/scene.xml",
        shape_index=3,
    )


def test_missing_and_empty_transforms_are_identity() -> None:
    assert _parse(_shape()) == {
        "scale": 1.0,
        "rotation": [0.0, 0.0, 0.0],
        "translate": [0.0, 0.0, 0.0],
    }
    assert _parse(_shape("<!-- intentionally empty -->")) == {
        "scale": 1.0,
        "rotation": [0.0, 0.0, 0.0],
        "translate": [0.0, 0.0, 0.0],
    }


def test_supported_transform_is_normalized_without_approximation() -> None:
    state = _parse(_shape("""
    <scale value="2.5"/>
    <rotate x="1" y="0" z="0" angle="10"/>
    <rotate y="1" angle="20"/>
    <rotate z="1" angle="30"/>
    <translate value="1, 2 3"/>
"""))

    assert state == {
        "scale": 2.5,
        "rotation": [30.0, 20.0, 10.0],
        "translate": [1.0, 2.0, 3.0],
    }


@pytest.mark.parametrize(
    ("body", "operation"),
    [
        ('<matrix value="1 0 0 0 1 0 0 0 1"/>', "matrix"),
        ('<lookat origin="0 0 0" target="1 0 0" up="0 0 1"/>', "lookat"),
        ('<scale value="2 3 4"/>', "scale"),
        ('<scale x="2" y="3" z="4"/>', "scale"),
        ('<scale value="0"/>', "scale"),
        ('<scale value="nan"/>', "scale"),
        ('<rotate x="1" y="1" angle="20"/>', "rotate"),
        ('<rotate x="0.5" y="0.5" angle="20"/>', "rotate"),
        ('<rotate z="0" angle="20"/>', "rotate"),
        ('<rotate z="1" angle="inf"/>', "rotate"),
        ('<translate value="1 2"/>', "translate"),
        ('<translate x="bad"/>', "translate"),
        ('<translate/><scale value="2"/>', "translate"),
        ('<translate x="1"/><scale value="2"/>', "scale"),
        ('<rotate z="1" angle="1"/><rotate y="1" angle="2"/>', "rotate"),
        ('<scale value="2"/><scale value="3"/>', "scale"),
    ],
)
def test_unsupported_transform_fails_with_source_context(body: str, operation: str) -> None:
    with pytest.raises(UnsupportedLightweightTransformError) as exc_info:
        _parse(_shape(body, shape_id="custom-wall"))

    message = str(exc_info.value)
    assert "scene.xml" in message
    assert "shape[3]" in message
    assert "custom-wall" in message
    assert "wall.ply" in message
    assert f"<{operation}>" in message
    assert "Supported form:" in message


def test_sensor_transform_is_outside_mesh_shape_validation() -> None:
    root = ET.fromstring("""<scene>
  <sensor type="perspective">
    <transform name="to_world"><lookat origin="0 0 1" target="0 0 0"/></transform>
  </sensor>
  <shape type="obj"><string name="filename" value="wall.obj"/></shape>
</scene>""")

    assert (
        parse_lightweight_shape_transform(
            root.find("shape"),
            source_xml="scene.xml",
            shape_index=0,
        )["scale"]
        == 1.0
    )


def test_repeated_to_world_transform_is_rejected() -> None:
    shape = ET.fromstring("""<shape type="ply" id="wall">
  <string name="filename" value="wall.ply"/>
  <transform name="to_world"><scale value="2"/></transform>
  <transform name="to_world"><translate x="1"/></transform>
</shape>""")

    with pytest.raises(UnsupportedLightweightTransformError, match="repeats the to_world"):
        _parse(shape)


def test_retained_release_scene_mesh_transforms_use_supported_subset() -> None:
    root = Path(__file__).resolve().parents[2]
    scene_paths = (
        root / "libraries/scenes/box/open_room6materials.xml",
        root / "libraries/scenes/empty/empty.xml",
        root / "libraries/scenes/ground/ground_60x50.xml",
        root / "scenarios/generator/propagation_and_materials/"
        "refraction_and_diffraction/glass_panel_scene.xml",
    )

    for scene_path in scene_paths:
        xml_root = ET.parse(scene_path).getroot()
        for shape_index, shape in enumerate(xml_root.findall("shape")):
            if shape.get("type") in {"ply", "obj"}:
                parse_lightweight_shape_transform(
                    shape,
                    source_xml=scene_path,
                    shape_index=shape_index,
                )
