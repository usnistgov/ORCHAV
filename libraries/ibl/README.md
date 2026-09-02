# IBL Lighting Assets

This directory contains image-based lighting (IBL) assets used by the ORCHAV
Visualizer renderers.

## Included Neutral Environments

ORCHAV includes one generated neutral fallback environment:

- `neutral_outdoor.*` for a sky-like outdoor response.

The neutral environment is generated rather than downloaded from an HDRI
library. It provides deterministic lighting for examples while keeping the
distribution self-contained and avoiding third-party HDRI licensing
requirements.

## What Is IBL?

Image-Based Lighting uses a high-dynamic-range panoramic image to light a 3D
scene. The `open3d` renderer uses Filament KTX cubemaps. The `pygfx` renderer
can load standard HDR equirectangular images.

| Backend | File type | Notes |
|---------|-----------|-------|
| `open3d` | KTX pair (`*_ibl.ktx` and `*_skybox.ktx`) | Both files must exist |
| `pygfx` | `.hdr` | Converted to a cubemap at runtime |

## Adding New Environments

Users may add their own properly licensed HDRI or KTX assets under this
directory. For `open3d`, provide matching `*_ibl.ktx` and `*_skybox.ktx`
files. For `pygfx`, provide a standard `.hdr` file.

When distributing additional environments, keep the source, license, and any
required attribution with the assets.

Renderer selection and lighting behavior are summarized in
[Renderers](../../docs/visualizer/renderers.md).
