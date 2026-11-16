# Third-Party Code Attribution

This module (`generator/osm/`) builds Sionna RT scenes from OpenStreetMap data.
It contains code adapted from two upstream projects, as well as original code
written for ORCHAV.

## Adapted Code

### NVlabs/sionna-large-radio-maps

- **Repository**: https://github.com/NVlabs/sionna-large-radio-maps
- **License**: Apache License 2.0
- **Reviewed upstream revision**: `1ba19ae1df1d26302fcfbaab14efc2347313da5d`
  (23 February 2026); the three adapted source files are unchanged from the
  initial release `4ed3436cdd2966f9979962b8143ac443aafbe981` (12 December 2025).
- **Copyright**: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES
- **Files adapted**:
  - `scripts/scene_builder.py` → `scene_builder.py` (building extrusion
    pipeline, water body mesh generation, Mitsuba XML assembly)
  - `sionna_lrm/scene/utils.py` → `coords.py` (UTM projection, polygon
    reordering, building height estimation, flat ground mesh)
  - `sionna_lrm/scene/heightmap.py` → `heightmap.py` (USGS S3 tile download,
    bilinear height interpolation)

### Geo2SigMap

- **Repository**: https://github.com/functions-lab/geo2sigmap
- **Historical scene-generation license**: Apache 2.0 in the reviewed
  [5 November 2025 package](https://github.com/functions-lab/geo2sigmap/tree/2ad20982ec9ff6f4076ffae4118bbed7b8f5b1f9/package); its `LICENSE` contains the full text.
  Current upstream terms may differ by revision/component.
- **Paper**: Li et al., "Geo2SigMap: High-Fidelity RF Signal Mapping Using
  Geographic Databases," IEEE DySPAN 2024 (arXiv:2312.14303)
- **Note**: NVIDIA distributes its adaptation under Apache 2.0 and credits
  Geo2SigMap. This project uses that identified NVIDIA adaptation. The exact
  Geo2SigMap revision used by NVIDIA is not recorded here; no independent
  re-licensing claim is made.

## Changes Made (per Apache 2.0 Section 4(b))

Compared to the NVIDIA upstream, the following changes were made:

1. **Removed** NVIDIA-specific infrastructure: `sionna_lrm` package imports,
   `BaseStationDB`, QuadTree adaptive meshing, SLURM parallel dispatch,
   ZIP compression, remote/local scene directory management.
2. **Replaced** logging with Python `logging.getLogger(__name__)` (was
   process-ID-based file+console logger).
3. **Added** the ORCHAV `flat_ground_mesh()` helper and a configurable
   flat-ground scene-building route. The upstream builder uses HeightMap and
   QuadTree, while its separate `bbox_to_mesh()` utility already supports flat
   ground; flat meshing itself is not claimed as an ORCHAV invention.
4. **Changed** scene origin from first bounding-box corner to bounding-box
   center (better for visualization).
5. **Added** `build_scene_from_center()` convenience API and CLI entry point
   (`__main__.py`).
6. **Added** graceful timeout/error handling for OSM water body queries.
7. **Kept** material IDs as `mat-itu_*` with `type="diffuse"` for the
   Sionna scene-loading interface. Consult the project installation guide for
   the tested runtime; this notice is not a cross-version qualification.

## Third-Party Libraries

### Triangle

Polygon triangulation uses the separately installed `triangle` package. Its
Python wrapper declares LGPL-3.0, while the compiled extension incorporates
Shewchuk Triangle under separate custom redistribution terms. See the
[versioned dependency notice](../../THIRD_PARTY_NOTICES.md#triangle-wrapper-and-embedded-library).
Neither component is bundled in the ORCHAV source export; do not treat the
wrapper metadata or ORCHAV's GPL license as permission to redistribute every
component of an installed environment.

### shapely-polyskel

- **Repository**: https://github.com/vec2pt/shapely-polyskel
- **License**: GNU Lesser General Public License v3 (LGPL-3.0)
- **Copyright**: Ármin Scipiades (vec2pt)
- **Purpose**: Straight skeleton computation for parametric roof generation
  (hipped, gabled, mansard, pyramidal roof shapes from the Simple 3D Buildings
  spec). Used at scene generation time only — not bundled into the application.
- **Dependency**: `euclid3` (BSD license, Alex Holkner)

### Microsoft Global ML Building Footprints

- **Repository**: https://github.com/microsoft/GlobalMLBuildingFootprints
- **License**: CDLA Permissive 2.0 (permissive, commercial use allowed)
- **Data**: 225M+ building footprints with ML-derived heights from satellite imagery
- **Purpose**: Primary building height enrichment — used as fallback when OSM
  buildings lack ``height`` or ``building:levels`` tags. Heights are derived from
  ML models trained on Bing Maps / Vexcel imagery (2014-2023), independent of
  OSM volunteer tagging.
- **Access**: Azure Blob static site, no authentication. Tiles indexed by Bing
  quadkey at zoom level 9. Downloaded and cached in ``~/.orchav/cache/globalml/``.
- **Attribution**: Microsoft Building Footprints. See
  https://github.com/microsoft/GlobalMLBuildingFootprints#license


### USGS / Mapzen Elevation Data

- **Source**: Mapzen/Tilezen terrain tiles on AWS S3 (``elevation-tiles-prod``
  bucket, public, no authentication). Global coverage at zoom level 14.
- **License**: Public domain (USGS), CC-BY (other contributors)
- **Purpose**: Terrain elevation for ground mesh and building/street placement

### ETH Global Canopy Height (2020 v1)

- **Publication**: Lang, N., Jetz, W., Schindler, K., & Wegner, J. D. (2023).
  "A high-resolution canopy height model of the Earth." *Nature Ecology &
  Evolution*, 7(11), 1778–1789.
- **DOI**: https://doi.org/10.3929/ethz-b-000609802
- **License**: CC-BY 4.0
- **Tile hosting**:
  https://share.phys.ethz.ch/~pf/nlangdata/ETH_GlobalCanopyHeight_10m_2020_version1/3deg_cogs/
- **Data**: 10 m global canopy height raster (uint8 meters) derived from
  Sentinel-2 and GEDI LiDAR. Cloud-Optimized GeoTIFF, 3°×3° tiles named by
  the SW corner (e.g. `ETH_GlobalCanopyHeight_10m_2020_N48E003_Map.tif`).
- **Purpose**: Per-tree and per-polygon canopy heights for vegetation
  realism. Replaces the previous 8 m fixed default. Used as the middle
  step of the fallback chain: OSM `height` tag > ETH canopy > 8 m default.
- **Access**: Anonymous HTTPS; no authentication. Tiles are downloaded on
  demand and cached under `~/.orchav/cache/eth_canopy/`.
- **Required citation**: Citations are emitted at runtime on every scene
  build that uses this dataset (see the `INFO` log line from
  `_cite_external_dataset`) and recorded in the generated
  `osm_metadata.json` under `external_data_sources`.

### ESA WorldCover v200 (2021 epoch)

- **Source**: [ESA WorldCover data access](https://esa-worldcover.org/en/data-access).
  ORCHAV requests individual 2021 map tiles under the S3 key prefix
  `v200/2021/map/` in the `esa-worldcover` bucket.
- **License**: CC-BY 4.0
- **Required attribution**:
  `"© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel
  data (2021) processed by ESA WorldCover consortium"`
- **Data**: 10 m global land cover raster (uint8 LCCS class codes) with
  11 classes (tree_cover, shrubland, grassland, cropland, built_up,
  bare/sparse, snow/ice, permanent_water_bodies, herbaceous_wetland,
  mangroves, moss_lichen). 3°×3° tiles named by SW corner
  (e.g. `ESA_WorldCover_10m_2021_v200_N48E003_Map.tif`).
- **Purpose**: Vegetation gap-fill (scatter trees in `tree_cover` /
  `shrubland` cells that OSM missed) and water-body validation
  (cross-check `permanent_water_bodies` against OSM water polygons).
- **Access**: Anonymous HTTPS or anonymous S3; cached under
  `~/.orchav/cache/worldcover/`.
- **Required citation**: Emitted at runtime and recorded in
  `osm_metadata.json` under `external_data_sources`.

## Original Code (ORCHAV)

The following are original to this project:

- `__main__.py` — CLI entry point and argument parsing
- `__init__.py` — Package exports and lazy imports
- `metadata.py` — Scene metadata, config normalization, reuse detection
- `coords.py: bbox_from_center()` — Center-point-to-bbox conversion
- `coords.py: flat_ground_mesh()` — Flat ground plane generator
- `scene_builder.py: build_scene_from_center()` — Convenience wrapper
- `scene_builder.py: _building_min_height()` — S3DB min_height parsing
- `scene_builder.py: _match_parts_to_buildings()` — Spatial part-to-parent matching
- `scene_builder.py: _generate_roof_mesh()` — Parametric roof generation
  (hipped, gabled, pyramidal) using straight skeleton
- `scene_builder.py: _parse_roof_tags()` — S3DB roof tag extraction
- The public `scenarios/generator/paris_opera_osm/scenario.yaml` recipe

- `acquisition.py`, `_http_worker.py` and `publication.py` — bounded HTTP
  acquisition, batching support, cancellation and staged publication.
- Processed GlobalML lookup caching and bounded/cached terrain HTTP acquisition
  are ORCHAV additions to the adapted scene builder and heightmap code.
