"""Backend-local 3D textures for the renderer-neutral volume payload."""

import logging

import numpy as np

from .compatibility import volume_unavailable_reason

logger = logging.getLogger(__name__)


def ensure_volume(renderer, name, payload):
    """Reuse texture storage across slice/opacity changes; remove by the usual ID."""
    reason = volume_unavailable_reason(getattr(renderer._gfx, "__version__", ""))
    if reason:
        logger.warning("Cannot render volume %s: %s", name, reason)
        return False
    if not renderer._initialized or renderer._scene is None:
        return False
    gfx = renderer._gfx
    native = renderer._objects.get(name)
    if renderer._kinds.get(name) != "volume" and native is not None:
        if not renderer.remove_named_geometry(name):
            return False
        native = None
    previous = getattr(native, "_orchav_volume_payload", None)
    try:
        if native is None:
            native = gfx.Volume(
                gfx.Geometry(grid=gfx.Texture(payload.values.copy(), dim=3)),
                gfx.VolumeSliceMaterial(),
            )
            renderer._scene.add(native)
            handle = renderer._allocate_handle()
            renderer._name_to_handle[name] = handle
            renderer._handle_to_name[handle] = name
            renderer._objects[name] = native
            renderer._kinds[name] = "volume"
            renderer._reverse_objects[id(native)] = name
            renderer._geometry_upload_center[name] = np.zeros(3, dtype=np.float32)
        elif previous is None or previous.values is not payload.values:
            native.geometry = gfx.Geometry(grid=gfx.Texture(payload.values.copy(), dim=3))
        material_changed = (
            previous is None
            or previous.mode != payload.mode
            or previous.lighting != payload.lighting
            or previous.surface_grid != payload.surface_grid
        )
        if material_changed:
            from .cell_volume import CellVolumeMaterial

            material = (
                CellVolumeMaterial(lighting=payload.lighting, surface_grid=payload.surface_grid)
                if payload.mode == "iso"
                else {"mip": gfx.VolumeMipMaterial, "slice": gfx.VolumeSliceMaterial}[
                    payload.mode
                ]()
            )
            # Auto blending writes depth and sorts whole objects by their centre.
            # A translucent region intersecting buildings (or another region)
            # can then hide geometry differently as the camera rotates.
            material.alpha_mode = "weighted_blend"
            native.material = material
        material = native.material
        material.pick_write = True
        if material_changed or previous.palette is not payload.palette:
            material.map = gfx.TextureMap(
                gfx.Texture(payload.palette.copy(), dim=1), filter="nearest", wrap="clamp"
            )
        material.maprange = payload.limits
        material.interpolation = "nearest"
        material.opacity = payload.opacity
        # Alpha-tested opaque regions write hit depth but still discard palette
        # zero (unknown/hidden cells). A solid pass would incorrectly paint
        # those transparent samples black. Fractional palette alpha and the
        # slice/MIP views keep order-independent transparent blending.
        binary_alpha = np.all((payload.palette[:, 3] == 0) | (payload.palette[:, 3] == 1))
        material.alpha_mode = (
            "dither"
            if payload.mode == "iso" and payload.opacity == 1.0 and binary_alpha
            else "weighted_blend"
        )
        if payload.mode == "slice":
            material.plane = payload.plane
        elif payload.mode == "iso":
            material.threshold = payload.threshold
            material.emissive = (0.12, 0.12, 0.12)
            # Scientific colour should not acquire a white specular highlight.
            # Diffuse shading still conveys the sampled region's surface shape.
            material.shininess = 0
        native._orchav_volume_payload = payload
        # Material-only edits leave transform/visibility unchanged. Those paths
        # correctly skip redraws, so an idle canvas needs this explicit request.
        renderer.request_redraw()
        return True
    except (RuntimeError, ValueError, TypeError, AttributeError):
        logger.warning("Could not apply sensing volume %s", name, exc_info=True)
        return False
