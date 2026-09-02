# Texture Assets

This directory contains visual texture assets for the ORCHAV Visualizer.
Texture maps are experimental, visual-only, and disabled by default. They
affect rendered appearance only. They do not change
[Sionna RT](https://nvlabs.github.io/sionna/) propagation
material properties.

## Directory Layout

```text
libraries/textures/
├── *.png                 # Small flat sample textures for explicit examples
└── pbr/                  # Bundled PBR texture sets
```

The `pbr/` sets are used by texture-backed visual presets such as Brick,
Concrete, Asphalt, Grass, and NIST CTL Floor, and by material mappings for
`asphalt`, `brick`, `concrete`, `grass`, `marble`, `metal`, and
`nist_ctl_floor`. See [`pbr/credits.md`](pbr/credits.md) for bundled asset
credits.

## Material Names Versus Texture Names

Material names and texture filenames are separate concepts.

- `material_type` names such as `concrete`, `wood`, `brick`, and `metal`
  identify material classes.
- BSDF/material IDs such as `mat-itu_concrete` or `wood` identify materials in
  a Mitsuba/Sionna RT scene file.
- Texture paths identify image files sampled by the Visualizer during
  rendering.

Scenario YAML selects scenes and can assign target or visual material metadata.
Scene XML BSDF IDs and material-type strings are loaded by Sionna RT for radio
propagation, then the Visualizer maps the loaded material metadata to scalar PBR
properties or explicit texture-map paths. Texture maps become active only when
texture loading is enabled.

A generic material ID that matches its material type is not treated as an
implicit texture filename. For example, a scene material named `wood` is not
automatically assigned `libraries/textures/wood.png`. Use a texture-backed PBR
preset or explicit texture path when an image map should be active.

## Supported Maps

PBR-capable renderers can use these texture-map fields:

| Field | Purpose |
|-------|---------|
| `texture_path` | Albedo/base-color image. Locks the normal material color picker when active. |
| `normal_map_path` | Surface relief direction. |
| `roughness_map_path` | Spatially varying roughness. |
| `ao_map_path` | Ambient occlusion. |
| `metallic_map_path` | Spatially varying metallic response. |

Supported image formats include PNG, JPG/JPEG, and BMP. Meshes need UV
coordinates for image maps. The scene loader can generate box-projection UVs
for loaded scene meshes when texture maps are active and UVs are missing.

## Authoring Example

Scene XML can reference texture maps from a BSDF. Relative paths are resolved
from the XML file's directory:

```xml
<bsdf type="itu" id="mat-custom_concrete">
  <string name="type" value="concrete"/>
  <string name="texture_path" value="textures/concrete_albedo.png"/>
  <string name="normal_map_path" value="textures/concrete_normal.png"/>
  <string name="roughness_map_path" value="textures/concrete_roughness.png"/>
  <string name="ao_map_path" value="textures/concrete_ao.png"/>
  <string name="metallic_map_path" value="textures/concrete_metallic.png"/>
</bsdf>
```

Python integrations carry the same fields in the renderer-neutral
`MaterialPayload` attached to a `RenderObject`. These fields request visual
texture maps only. They do not change the radio-material model used by
Sionna RT.

## Texture Launch Mode

Texture maps are disabled by default at Visualizer startup. Use
`--enable-textures` or `ORCHAV_ENABLE_TEXTURES=1` to activate image-map loading
for PBR-capable renderers. Materials remain lit and PBR-capable when textures
are disabled. Scalar PBR properties such as color, roughness, metallic,
reflectance, and alpha still apply. Use `--disable-textures` or
`ORCHAV_DISABLE_TEXTURES=1` as an explicit scalar-PBR override when an
environment or wrapper enables textures.

See [Renderers](../../docs/visualizer/renderers.md) for renderer support notes.
