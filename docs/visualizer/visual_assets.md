# Visual asset authoring

Visual annotations and texture mappings add appearance to existing scene and
target meshes. They do not change the radio geometry, electromagnetic materials
or saved sensing results. Use [Scene appearance](appearance.md) to choose
included styles. Use the formats below when preparing your own assets.

## Choose an asset mapping

| Goal | Input |
|---|---|
| Enable procedural facades on an imported scene | Building-face and building-part annotations. |
| Texture selected faces of a static scene | `visual_surfaces.json` beside the scene XML. |
| Texture an animated target | An appearance pack referenced by a target's visual profile. |

Texture coordinates, commonly called UV coordinates, attach image positions to
triangle corners. Both texture formats below use `u = 0` at the image's left
edge and `v = 0` at its top edge. Their schemas are separate because static face
selections and animated topology have different requirements.

## Classify building faces

[OpenStreetMap (OSM) generation](../generator/geography/osm_scene_generation.md) writes `visual_building_faces` in `osm_metadata.json`. An imported
scene can supply the same mapping in `building_visuals.json` beside its XML.
See the included
[three-building annotation](../../scenarios/generator/building_appearance/scene/building_visuals.json).

Each relative mesh filename maps to integer `building_triangles` and
`total_triangles` counts. Building triangles must form the leading segment of
the mesh, and the counts must agree with the loaded mesh. For **Facades by
building**, add ordered `parts` entries of `[first_triangle, triangle_count]`
covering those classified faces. Update annotations whenever mesh topology
changes. Unannotated meshes retain their original appearance.

### Landmark exclusions

An optional `facade_exclusions` list suppresses procedural windows on known
landmark parts:

```json
"facade_exclusions": [
  {
    "first_triangle": 50442,
    "triangle_count": 68,
    "source_id": "osm:way/226413508",
    "name": "Arc de Triomphe"
  }
]
```

These indices illustrate one particular mesh and cannot be copied to another.
Each range must exactly match a validated `parts` entry. Include every part and
separate roof chunk that should be excluded. The source ID records an existing
association rather than requesting a geographic lookup. Invalid annotations
leave the affected mesh at its original appearance.

An exclusion retains a plain swatch of the active material. It does not create
a separate per-part material. Authored textures and explicit visual material
assignments retain precedence.

### Road annotations

Procedural road markings require `visual_road_faces` metadata describing ordered
quad segments, arc-length bounds, road class, lane tags and direction. New OSM
builds supply it. A street graph alone does not identify the source mesh faces.
Keep annotations tied to the matching mesh topology when preparing imported scenes.

## Texture selected static faces

Place `visual_surfaces.json` beside the scene XML to select faces and assign
their textures. This is useful when one radio-material mesh contains a landmark
and surrounding buildings. The display can distinguish them while retaining
their shared radio mesh.

The [Paris Opéra mapping](../../scenarios/showcases/paris_opera_multitarget/inputs/scene/visual_surfaces.json)
is an included example. Its [texture documentation](../../scenarios/showcases/paris_opera_multitarget/inputs/scene/textures/README.md)
records the source image, preparation and credits.

### Mapping example

This version-1 record maps one triangle of `mesh/wall.ply`:

```json
{
  "schema_version": 1,
  "meshes": {
    "mesh/wall.ply": {
      "source_sha256": "REPLACE_WITH_SHA256_OF_THE_EXACT_MESH_FILE",
      "surfaces": [
        {
          "id": "front",
          "name": "Landmark front",
          "triangle_indices": [0],
          "triangle_uvs": [[0, 0], [1, 0], [0, 1]],
          "material": {
            "texture_path": "textures/front.png",
            "color": [1, 1, 1],
            "roughness": 0.8,
            "metallic": 0,
            "reflectance": 0.3
          }
        }
      ]
    }
  }
}
```

| Field | Meaning |
|---|---|
| Mesh key | Filename used in the scene XML. |
| `source_sha256` | Fingerprint of the exact mesh bytes. Rebuild the mapping after changing that file. |
| `id` | Unique lowercase letters, digits, underscores or hyphens. `remainder` is reserved. |
| `triangle_indices` | Nonempty, zero-based source-face indices, distinct and disjoint across surfaces. |
| `triangle_uvs` | Three `[u, v]` pairs per selected triangle, in source corner order. Values outside `[0, 1]` repeat the texture. |
| `texture_path` | Required albedo image, resolved relative to this JSON file. |
| Other maps | Optional `normal_map_path`, `roughness_map_path`, `ao_map_path` and `metallic_map_path`. |
| Optical scalars | RGB `color` and optional `roughness`, `metallic`, `reflectance`, `alpha`, `clearcoat` and `transmission` use values from zero to one. |

The loader validates a complete mesh mapping before applying it. A changed
fingerprint, overlapping selection, missing texture or malformed UV array leaves
that source mesh unchanged and logs the reason. These display pieces cannot be
moved or assigned separate radio materials interactively. Edit the source scene
and regenerate if the radio geometry must change.

### Reloading and portable scenarios

Reopen the scenario after changing the mapping or its textures. Scenario Builder
also checks these dependencies when recompiling an edited scene. A portable
scenario copy must include the sidecar and its local maps. The copy workflow
rejects dependencies outside the scenario, although shared assets can be used
for normal viewing when the same repository layout is available.

Authored textures survive procedural facade changes. **Custom surface textures**
and the normal texture-map switch control their visibility. See
[renderer texture behavior](renderers.md#materials-textures-and-lighting) for
the retained Open3D limitations.

## Texture an animated target

A compatible appearance pack adds skin, hair or clothing detail to an animated
mesh. It follows the mesh through its poses without adding clothing volume or
changing motion. Supply a pack authored for the animation's exact topology.
No texture atlas or UV map for this workflow is bundled.

### Add a target profile

Add `surface` to the target's existing
[visual profile](scenario_defaults.md#visual-material-profiles):

```yaml
view_defaults:
  visual_profiles:
    - match:
        name: Pedestrian
        type: target
      preset: Matte
      surface: appearance/surface.json
```

The name matches the actor, not its mesh filename. The first matching valid
profile wins, so preserve existing profile order. Enable texture maps when
opening the scenario:

```bash
orchav-visualizer --scenario path/to/scenario.yaml --enable-textures --no-resume
```

`--no-resume` avoids restoring an older appearance choice. The documented
animated-texture workflow uses pygfx. Change the `surface` path and reopen the
scenario to select another pack. Remove it to restore the profile's scalar
material.

### Appearance pack format

Keep the sidecar, image and UV data together:

| File | Contents |
|---|---|
| `surface.json` | Schema identifier and relative image/UV paths. |
| `albedo.png` | RGB texture atlas. |
| `surface_uv.npz` | Expected topology and texture coordinates. |

The sidecar contains:

```json
{
  "schema": "orchav.target_surface.v1",
  "texture": "albedo.png",
  "uv_map": "surface_uv.npz"
}
```

The NPZ contains numeric arrays loaded without pickle:

| Key | Shape and meaning |
|---|---|
| `vertex_count` | Positive integer scalar, equal to the vertex count of every pose. |
| `triangles` | Integer `(F, 3)` ordered vertex indices for `F` faces. |
| `triangle_uvs` | Finite float `(3F, 2)` coordinates in the same corner order. |

The `surface` path is relative to the scenario YAML and supports
`${PROJECT_ROOT}`. Image and UV paths are relative to the sidecar.

### Topology and troubleshooting

Another animation can reuse the pack only if its vertex count and ordered
triangle indices are identical. Reordered vertices or remeshing require a new
mapping. See [animated targets](../generator/motion/animated_targets.md)
for geometry preparation.

| Symptom | Check |
|---|---|
| Uniform color | Enable texture maps and check the matched profile's `surface` path and startup warnings. |
| Topology mismatch | Use a UV map authored for that exact vertex and face ordering. |
| Visible seams | Inspect several poses and camera angles when preparing the atlas. |
| Silhouette differs from painted detail | Textures do not add hair, clothing folds or shoe geometry. |

Up: [Visualizer](README.md) | Related: [Scene appearance](appearance.md) |
[Scenario defaults](scenario_defaults.md) | [SMPL motion import](../generator/motion/smpl_mesh_generation.md)
