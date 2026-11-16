# Texture Assets

The Visualizer can use image maps to add surface color, relief, and other visual detail. Texture maps are disabled at startup by default. Radio propagation uses the scene's radio materials independently of these maps.

## Directory Layout

```text
libraries/textures/
├── *.png                 # Optional user-provided flat textures
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

To apply an image, choose a texture-backed PBR preset or provide an explicit image path. A generic material ID such as `wood` does not by itself select a same-named image file.

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

Python integrations pass the same fields in the renderer's `MaterialPayload` on a `RenderObject`.

## Texture Launch Mode

Use `--enable-textures` or `ORCHAV_ENABLE_TEXTURES=1` when launching the Visualizer to load image maps. Use `--disable-textures` or `ORCHAV_DISABLE_TEXTURES=1` to override a texture-enabled environment. Materials still use scalar color, roughness, reflectance, and alpha when image maps are off.

See [Renderers](../../docs/visualizer/renderers.md) for renderer support and the desktop controls.
