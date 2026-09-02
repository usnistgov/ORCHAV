"""Interpret Sionna RT interaction and object-ID metadata.

Sionna path tensors use the maximum ``uint32`` value when no scene object is
associated with a path depth. Positive interaction codes identify physical
bounces; line-of-sight is represented separately by the absence of a bounce
in ``StandardMPCFrame``. Keeping the source codes here gives converters,
diagnostics, and statistics one interpretation boundary.
"""

from __future__ import annotations

import numpy as np

SIONNA_INVALID_OBJECT_ID = np.uint32(np.iinfo(np.uint32).max)

SIONNA_INTERACTION_LOS = 0
SIONNA_INTERACTION_SPECULAR = 1
SIONNA_INTERACTION_DIFFUSE = 2
SIONNA_INTERACTION_REFRACTION = 4
SIONNA_INTERACTION_DIFFRACTION = 8

SIONNA_INTERACTION_LABELS = {
    SIONNA_INTERACTION_LOS: "LOS",
    SIONNA_INTERACTION_SPECULAR: "Specular",
    SIONNA_INTERACTION_DIFFUSE: "Diffuse",
    SIONNA_INTERACTION_REFRACTION: "Refraction",
    SIONNA_INTERACTION_DIFFRACTION: "Diffraction",
}
SIONNA_INTERACTION_LABELS_LOWER = {
    code: label.lower() for code, label in SIONNA_INTERACTION_LABELS.items()
}
SIONNA_NON_LOS_INTERACTION_TYPES = (
    SIONNA_INTERACTION_SPECULAR,
    SIONNA_INTERACTION_DIFFUSE,
    SIONNA_INTERACTION_REFRACTION,
    SIONNA_INTERACTION_DIFFRACTION,
)
