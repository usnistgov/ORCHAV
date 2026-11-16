"""Optical blue-hour lights for the Opéra showcase; never used by ray tracing."""

from copy import deepcopy


def paris_dusk():
    """Return a bounded, opt-in rig in world metres and BMW-local coordinates.

    The source car faces +X. Its front lamp cover is UV slot 9, rings slot 11,
    and rear lamps slot 3. Positions refer to the existing centred target mesh.
    The two facade washes are illustrative architectural illumination.
    """
    headlight = dict(
        position=[2.15, 0, -0.04],
        lookat=[16, 0, -0.70],
        color=[0.85, 0.91, 1.0],
        intensity=2.8,
        distance=24,
        angle_deg=28,
    )
    return dict(
        label="Paris blue hour",
        facade_lights=[
            dict(
                position=[-60, -70, 3],
                lookat=[-48, -39, 18],
                color=[1.0, 0.72, 0.43],
                intensity=5.5,
                distance=85,
                angle_deg=42,
            ),
            dict(
                position=[-14, -57, 3],
                lookat=[-30, -30, 18],
                color=[1.0, 0.72, 0.43],
                intensity=5.5,
                distance=85,
                angle_deg=42,
            ),
        ],
        targets=[
            dict(
                object_id=f"target:{name}::mesh",
                light=deepcopy(headlight),
                palette_slots=19,
                emissive_slots={
                    3: [1.0, 0.025, 0.006],
                    9: [0.65, 0.75, 1.0],
                    11: [0.85, 0.93, 1.0],
                },
            )
            for name in ("car_inbound", "car_outbound")
        ],
        window_accents=True,
    )
