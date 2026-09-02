# PBR Texture Credits

Most textures in this directory are sourced from **Poly Haven**
(https://polyhaven.com/) and licensed under **CC0 1.0 Universal**
(https://creativecommons.org/publicdomain/zero/1.0/). The
[`nist_ctl_floor/`](nist_ctl_floor/README.md) pack is procedurally generated
and does not embed downloaded logo files.

Under CC0 you are free to use, modify, and redistribute these assets
for any purpose (including commercial) without attribution. We provide
attribution here anyway as a courtesy to the Poly Haven project, which
funds asset creation via community donations at https://polyhaven.com/.

Each directory contains a 1K-resolution PBR set (albedo, normal, roughness,
ambient occlusion) used by the Open3D and pygfx renderers to add
surface detail to ITU-material meshes that lack artist-authored textures.

| Directory | Poly Haven slug | Asset page |
|---|---|---|
| `concrete/` | `concrete_wall_007` | https://polyhaven.com/a/concrete_wall_007 |
| `brick/` | `brick_wall_001` | https://polyhaven.com/a/brick_wall_001 |
| `asphalt/` | `aerial_asphalt_01` | https://polyhaven.com/a/aerial_asphalt_01 |
| `marble/` | `marble_01` | https://polyhaven.com/a/marble_01 |
| `metal/` | `metal_plate` | https://polyhaven.com/a/metal_plate |
| `grass/` | `aerial_grass_rock` | https://polyhaven.com/a/aerial_grass_rock |
| `nist_ctl_floor/` | Procedural NIST/CTL presentation pack | [`README`](nist_ctl_floor/README.md) |

File mapping per directory:

- `albedo.png`: base color (Poly Haven `diff` / `diffuse`)
- `normal.png`: OpenGL tangent-space normal map (Poly Haven `nor_gl`)
- `roughness.png`: linear roughness (Poly Haven `rough`)
- `ao.png`: ambient occlusion (Poly Haven `ao`)
- `metallic.png`: metalness (Poly Haven `metal`, only present for `metal/`)

All files are 1024×1024 PNGs optimized with PIL. Maps that do not use alpha
are stored as RGB. The six CC0 texture sets occupy about 32 MiB. Including the
procedural NIST/CTL pack, the complete PBR bundle occupies about 37 MiB.
Higher-resolution variants (2K, 4K, 8K) are available on the Poly Haven asset
pages linked above if a specific scene needs them.

## Bundling Policy

ORCHAV bundles these assets directly in the repository for
offline-installability. Third-party bundled texture assets must be CC0.
Project-created procedural packs should document their provenance and avoid
embedding external brand or logo files.
