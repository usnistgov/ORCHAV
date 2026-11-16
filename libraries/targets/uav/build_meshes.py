"""Author two original, metre-scaled quadrotors; no downloaded geometry.

The combined OBJ is the rigid sensing baseline. Named parts and rotor pivots
are retained for future articulated geometry; no micro-Doppler is simulated.
Run from any directory to reproduce the checked-in OBJ and metadata files.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


class Mesh:
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.parts = []

    def part(self, name, vertices, faces):
        offset = len(self.vertices)
        self.parts.append({"name": name, "first_face": len(self.faces), "face_count": len(faces)})
        self.vertices.extend(vertices)
        self.faces.extend(tuple(i + offset for i in face) for face in faces)

    def box(self, name, center, size, yaw=0):
        c, s = math.cos(yaw), math.sin(yaw)
        v = []
        for x, y, z in (
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
        ):
            x, y, z = x * size[0] / 2, y * size[1] / 2, z * size[2] / 2
            v.append((center[0] + c * x - s * y, center[1] + s * x + c * y, center[2] + z))
        faces = (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        )
        self.part(name, v, faces)

    def ellipsoid(self, name, center, radii, segments=16, rings=8):
        # Two poles and closed latitude rings, with outward winding.
        v = [(center[0], center[1], center[2] - radii[2])]
        for j in range(1, rings):
            lat = -math.pi / 2 + math.pi * j / rings
            for i in range(segments):
                a = 2 * math.pi * i / segments
                v.append(
                    (
                        center[0] + radii[0] * math.cos(lat) * math.cos(a),
                        center[1] + radii[1] * math.cos(lat) * math.sin(a),
                        center[2] + radii[2] * math.sin(lat),
                    )
                )
        top = len(v)
        v.append((center[0], center[1], center[2] + radii[2]))
        f = []
        for i in range(segments):
            k = (i + 1) % segments
            f.append((0, 1 + k, 1 + i))
            for j in range(rings - 2):
                a, b = 1 + j * segments + i, 1 + j * segments + k
                f.extend(((a, b, b + segments), (a, b + segments, a + segments)))
            f.append((top, 1 + (rings - 2) * segments + i, 1 + (rings - 2) * segments + k))
        self.part(name, v, f)

    def cylinder(self, name, center, radius, height, segments=16):
        vertices = [
            (
                center[0] + radius * math.cos(2 * math.pi * i / segments),
                center[1] + radius * math.sin(2 * math.pi * i / segments),
                center[2] + sign * height / 2,
            )
            for sign in (-1, 1)
            for i in range(segments)
        ]
        vertices.extend(
            (
                (center[0], center[1], center[2] - height / 2),
                (center[0], center[1], center[2] + height / 2),
            )
        )
        faces = []
        for i in range(segments):
            j = (i + 1) % segments
            faces.extend(
                (
                    (i, j, j + segments),
                    (i, j + segments, i + segments),
                    (2 * segments, j, i),
                    (2 * segments + 1, i + segments, j + segments),
                )
            )
        self.part(name, vertices, faces)


def build(name, dimensions, reference):
    x, y, z = dimensions
    radius = min(x, y) * 0.17
    mesh = Mesh()
    mesh.ellipsoid("fuselage", (0, 0, 0.05 * z), (0.21 * x, 0.23 * y, 0.24 * z))
    mesh.box("battery", (0, 0, -0.13 * z), (0.29 * x, 0.29 * y, 0.13 * z))
    mesh.ellipsoid("camera", (0.20 * x, 0, -0.18 * z), (0.045 * x, 0.07 * y, 0.08 * z), 12, 6)
    rotors = []
    for index, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        px, py = sx * (x / 2 - radius), sy * (y / 2 - radius)
        length = math.hypot(px, py)
        mesh.box(
            f"arm_{index}",
            (px / 2, py / 2, 0.12 * z),
            (length, 0.035 * min(x, y), 0.065 * z),
            math.atan2(py, px),
        )
        mesh.cylinder(f"motor_{index}", (px, py, 0.22 * z), 0.28 * radius, 0.16 * z)
        mesh.cylinder(f"shaft_{index}", (px, py, 0.40 * z), 0.055 * radius, 0.20 * z)
        yaw = 0 if index in (0, 3) else math.pi / 2
        # Finite-thickness two-blade propeller. Fixed phase is intentional.
        mesh.box(
            f"rotor_{index}", (px, py, 0.4925 * z), (2 * radius, 0.16 * radius, 0.015 * z), yaw
        )
        mesh.box(
            f"leg_{index}", (0.75 * px, 0.75 * py, -0.16 * z), (0.025 * x, 0.025 * y, 0.66 * z)
        )
        mesh.box(
            f"foot_{index}", (0.75 * px, 0.75 * py, -0.485 * z), (0.10 * x, 0.045 * y, 0.03 * z)
        )
        rotors.append(
            {
                "part": f"rotor_{index}",
                "pivot_m": [px, py, 0.4925 * z],
                "axis": [0, 0, 1],
                "radius_m": radius,
                "fixed_phase_rad": yaw,
                "nominal_rotation_sign": 1 if index in (0, 3) else -1,
            }
        )
    directory = Path(__file__).parent / name
    directory.mkdir(exist_ok=True)
    lines = ["# Original ORCHAV quadrotor; metres; rigid baseline; not calibrated RCS"]
    lines.extend("v " + " ".join(f"{a:.9f}" for a in v) for v in mesh.vertices)
    for part in mesh.parts:
        lines.append("g " + part["name"])
        lines.extend(
            "f " + " ".join(str(i + 1) for i in f)
            for f in mesh.faces[part["first_face"] : part["first_face"] + part["face_count"]]
        )
    obj = directory / "drone.obj"
    obj.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    metadata = {
        "schema_version": 1,
        "authorship": "Original procedural ORCHAV geometry",
        "units": "m",
        "axes": ["forward x", "left y", "up z"],
        "origin": "geometric envelope centre; not a measured centre of mass",
        "envelope_xyz_m": dimensions,
        "dimension_reference": reference,
        "bounds_m": [
            [min(v[i] for v in mesh.vertices) for i in range(3)],
            [max(v[i] for v in mesh.vertices) for i in range(3)],
        ],
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.faces),
        "sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
        "parts": mesh.parts,
        "rotors": rotors,
        "motion": "rigid; rotors fixed",
        "rf_assumption": "Scenario-assigned material; no calibrated RCS or blade micro-motion",
    }
    (directory / "geometry.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    build(
        "compact",
        [0.3, 0.4, 0.2],
        "TR 38.765 V20.0.0 Annex A UAV size example; author-assigned XYZ convention",
    )
    build("large", [1.6, 1.5, 0.7], "TS 22.137 V19.1.0 UAV example; author-assigned XYZ convention")
