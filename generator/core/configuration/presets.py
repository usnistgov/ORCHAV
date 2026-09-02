"""Static configuration presets.

``QUALITY_PRESETS`` are shared defaults for ray tracing and, after translation,
coverage solving.  The table is an ordered solver-budget ladder: debug and
ultra-low are fast diagnostic runs, low is the ordinary MPC default, medium
enables diffuse reflection with a larger budget, and high/ultra progressively
enable more expensive propagation effects and ray budgets. Services copy these
dictionaries before applying per-scenario
overrides so edits for one run do not mutate the global preset table.

``AVAILABLE_SCENES`` is an advisory list used for validation and user-facing
messages.  Scene loading still happens in ``SceneService`` because only that
service talks to Sionna.
"""

from typing import Any, Dict

QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "debug": {
        "max_depth": 1,
        "samples_per_src": 10000,
        "max_num_paths_per_src": 1000,
        "seed": 42,
        "los": True,
        "specular_reflection": False,
        "diffuse_reflection": False,
        "refraction": False,
        "diffraction": False,
        "synthetic_array": True,
    },
    "ultra-low": {
        "max_depth": 2,
        "samples_per_src": 100000,
        "max_num_paths_per_src": 100000,
        "seed": 42,
        "los": True,
        "specular_reflection": True,
        "diffuse_reflection": False,
        "refraction": False,
        "diffraction": False,
        "synthetic_array": True,
    },
    "low": {
        "max_depth": 3,
        "seed": 42,
        "los": True,
        "specular_reflection": True,
        "diffuse_reflection": False,
        "max_num_paths_per_src": 500000,
        "samples_per_src": 1000000,
        "refraction": False,
        "diffraction": False,
        "synthetic_array": True,
    },
    "medium": {
        "max_depth": 4,
        "samples_per_src": 10000000,
        "max_num_paths_per_src": 1000000,
        "seed": 42,
        "los": True,
        "specular_reflection": True,
        "diffuse_reflection": True,
        "refraction": False,
        "diffraction": False,
        "synthetic_array": True,
    },
    "high": {
        "max_depth": 5,
        "samples_per_src": 10000000,
        "max_num_paths_per_src": 5000000,
        "seed": 42,
        "los": True,
        "specular_reflection": True,
        "diffuse_reflection": True,
        "refraction": True,
        "diffraction": True,
        "synthetic_array": True,
    },
    "ultra": {
        "max_depth": 6,
        "samples_per_src": 100000000,
        "max_num_paths_per_src": 10000000,
        "seed": 42,
        "los": True,
        "specular_reflection": True,
        "diffuse_reflection": True,
        "refraction": True,
        "diffraction": True,
        "synthetic_array": True,
    },
}

AVAILABLE_SCENES: Dict[str, str] = {
    "etoile": "Place de l´Étoile roundabout in Paris (default)",
    "simple_street_canyon": "Simple street canyon with buildings",
    "simple_street_canyon_with_cars": "Street canyon with parked cars",
    "munich": "Munich city environment",
    "san_francisco": "San Francisco city environment",
    "florence": "Florence city environment",
    "box": "Simple box environment",
    "box_one_screen": "Box with one screen",
    "box_two_screens": "Box with two screens",
    "floor_wall": "Simple floor and wall setup",
    "simple_reflector": "Simple reflector setup",
    "double_reflector": "Double reflector setup",
    "triple_reflector": "Triple reflector setup",
    "simple_wedge": "Simple wedge setup",
    "sphere": "Sphere object (not a full scene)",
}
