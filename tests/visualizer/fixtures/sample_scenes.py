"""
Sample scene XML generators for testing.

This module provides functions to create synthetic scene XML files
for testing scene loading functionality.
"""

from pathlib import Path
from typing import List, Tuple


def create_simple_scene_xml(filepath: Path, meshes: List[Tuple[str, str]] = None) -> None:
    """Create a simple Mitsuba scene XML file.

    Args:
        filepath: Path to save XML file
        meshes: List of (mesh_id, mesh_file) tuples to include
    """
    if meshes is None:
        meshes = [("ground", "ground.ply"), ("building", "building.ply")]

    xml_content = """<?xml version="1.0" encoding="utf-8"?>
<scene version="3.0.0">
    <!-- Integrator -->
    <integrator type="path">
        <integer name="max_depth" value="8"/>
    </integrator>

    <!-- Camera -->
    <sensor type="perspective">
        <transform name="to_world">
            <lookat origin="10, 10, 5"
                    target="0, 0, 0"
                    up="0, 0, 1"/>
        </transform>
        <float name="fov" value="45"/>
    </sensor>

    <!-- Meshes -->
"""

    for mesh_id, mesh_file in meshes:
        xml_content += f"""    <shape type="ply" id="{mesh_id}">
        <string name="filename" value="{mesh_file}"/>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="0.5, 0.5, 0.5"/>
        </bsdf>
    </shape>

"""

    xml_content += "</scene>\n"

    filepath.write_text(xml_content)


def create_scene_with_materials(
    filepath: Path, materials: List[Tuple[str, str, Tuple[float, float, float]]] = None
) -> None:
    """Create a scene XML with different materials.

    Args:
        filepath: Path to save XML file
        materials: List of (shape_id, material_type, rgb_color) tuples
    """
    if materials is None:
        materials = [
            ("concrete_wall", "itu_concrete", (0.7, 0.7, 0.7)),
            ("metal_surface", "itu_metal", (0.8, 0.8, 0.9)),
            ("glass_window", "itu_glass", (0.9, 0.9, 1.0)),
        ]

    xml_content = """<?xml version="1.0" encoding="utf-8"?>
<scene version="3.0.0">
"""

    for shape_id, material_type, rgb in materials:
        xml_content += f"""    <shape type="rectangle" id="{shape_id}">
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{rgb[0]}, {rgb[1]}, {rgb[2]}"/>
        </bsdf>
        <string name="radio_material" value="{material_type}"/>
    </shape>

"""

    xml_content += "</scene>\n"

    filepath.write_text(xml_content)
