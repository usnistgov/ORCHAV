# Paris Showcase Inputs

This directory contains the static inputs used by `prepare.py`. No live map
service or private experiment directory is needed. The walking body and motion
remain explicit external inputs; they are not included here.

- `scene/` contains eight mesh layers, the XML scene, local projected-coordinate
  metadata, the street graph and visual bindings. Map geometry and cached tags
  are © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
  under the [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
  Preserve contributor attribution and the database license when redistributing
  these map-derived resources. The source geometry remains unchanged here;
  preparation removes explicitly tagged underground parts in its output copy
  and records every affected triangle in `scene_clean/surface_scene_audit.json`.
- `osm_building_parts.geojson` is the matching cached OSM building-part subset.
  Its SHA-256 is `42b967400db78bbaabb67e59234da20a46952fb0914cd3db301d5ad9ff011155`.
- `scene/textures/` includes the two photo-informed Opéra albedos and their
  [credit and mapping record](scene/textures/README.md). Their source photograph
  is by Peter Rivera, CC BY 2.0; the documented transformations remain credited.
- `vehicles/bmw/` holds the debadged Mike Pan/Morgan McGuire CC0 model, palette,
  UV map, conversion records and license evidence. `vehicles/scooter/` holds
  Innerscene's CC0 Urban Motor Scooter GLB and source records; the matching
  `CC0-1.0.txt` and `library_terms.txt` are retained beside it.
- `uav_surface/` is the original ORCHAV survey-drone visual atlas, under
  GPL-3.0-only. The matching original UAV geometry is bundled separately at
  `libraries/targets/uav/large/`.
- `paris_limestone_paving.png` is an original texture generated for this
  scenario with OpenAI image generation on 2026-10-23. Its SHA-256 is
  `818910a84fa1caaf5734c02706865dfab42b586f3515f80c1c1cffc98aad1897`.
  Its 3.2 m × 2.0 m visual tile changes neither radio geometry nor materials.

[`asset_manifest.json`](asset_manifest.json) records the original source revision
and hashes, and the hashes of the relocated files. Only local path bindings and
standalone provenance wording change during relocation. Preparation copies
these resources into the new run and keeps generated files separate from inputs.
See the [scenario instructions](../README.md#running) and
[complete credits](../ATTRIBUTION.md).
