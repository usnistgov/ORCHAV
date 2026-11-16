"""Authored camera stops along the open plaza in front of the Palais Garnier."""


def paris_demo_tour():
    """Return an ordinary camera sequence for the visualizer's Demo panel."""
    poses = (
        ("Opéra approach", (-6, -129, 16), (-38, -55, 9), 60),
        ("Paris overview", (-5, -142, 42), (-36, -59, 10), 56),
        ("Garnier façade", (-15, -112, 22), (-36, -45, 14), 58),
        ("Street view", (-14, -98, 6), (-36, -45, 11), 65),
    )
    return {
        "label": "Paris Opéra",
        "views": {
            name: {
                "pose": {"eye": list(eye), "lookat": list(lookat), "up": [0, 0, 1], "fov": fov},
                "actor": None,
            }
            for name, eye, lookat, fov in poses
        },
        "shots": [{"view": name, "hold": 5, "move": 4} for name, *_ in poses],
    }
