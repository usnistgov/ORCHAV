from __future__ import annotations

from sionna.rt import InteractionType

from shared.frames.sionna_metadata import (
    SIONNA_INTERACTION_DIFFRACTION,
    SIONNA_INTERACTION_DIFFUSE,
    SIONNA_INTERACTION_LABELS,
    SIONNA_INTERACTION_LABELS_LOWER,
    SIONNA_INTERACTION_LOS,
    SIONNA_INTERACTION_REFRACTION,
    SIONNA_INTERACTION_SPECULAR,
    SIONNA_INVALID_OBJECT_ID,
    SIONNA_NON_LOS_INTERACTION_TYPES,
)


def test_sionna_interaction_constants_match_runtime() -> None:
    assert SIONNA_INTERACTION_LOS == InteractionType.NONE
    assert SIONNA_INTERACTION_SPECULAR == InteractionType.SPECULAR
    assert SIONNA_INTERACTION_DIFFUSE == InteractionType.DIFFUSE
    assert SIONNA_INTERACTION_REFRACTION == InteractionType.REFRACTION
    assert SIONNA_INTERACTION_DIFFRACTION == InteractionType.DIFFRACTION


def test_sionna_interaction_labels_cover_known_types() -> None:
    assert SIONNA_INTERACTION_LABELS == {
        SIONNA_INTERACTION_LOS: "LOS",
        SIONNA_INTERACTION_SPECULAR: "Specular",
        SIONNA_INTERACTION_DIFFUSE: "Diffuse",
        SIONNA_INTERACTION_REFRACTION: "Refraction",
        SIONNA_INTERACTION_DIFFRACTION: "Diffraction",
    }
    assert SIONNA_INTERACTION_LABELS_LOWER[SIONNA_INTERACTION_DIFFRACTION] == "diffraction"
    assert SIONNA_NON_LOS_INTERACTION_TYPES == (
        SIONNA_INTERACTION_SPECULAR,
        SIONNA_INTERACTION_DIFFUSE,
        SIONNA_INTERACTION_REFRACTION,
        SIONNA_INTERACTION_DIFFRACTION,
    )


def test_sionna_invalid_object_id_is_uint32_sentinel() -> None:
    assert int(SIONNA_INVALID_OBJECT_ID) == 0xFFFFFFFF
