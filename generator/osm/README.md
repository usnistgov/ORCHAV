# OpenStreetMap scene builder

The Generator can build a scene from OpenStreetMap data when a scenario sets `scene.source` to `osm`. You can also run the builder independently to inspect the geometry before using it in a simulation.

## Build a scene

Install the OSM dependencies from the repository root.

```bash
python -m pip install -e ".[osm]"
```

Generate a 500 m square around the Arc de Triomphe in Paris.

```bash
python -m generator.osm --center 48.8738 2.2950 --size 500 --output scenarios/local_osm/etoile
```

The destination contains `scene.xml` and the meshes referenced by that scene. The builder records acquisition settings and data sources in `osm_metadata.json`. A generated street network can also supply routes for moving actors.

For a rectangle, pass `--bbox SOUTH WEST NORTH EAST` in WGS84 degrees. Run `python -m generator.osm --help` for terrain, cache, and timeout options. Python callers can use [build_scene and build_scene_from_center](scene_builder.py).

## How the geometry is built

Street meshes use a valid OSM width tag when one is present. Otherwise the
[street-width policy](street_width.py) selects a width from the road class.
Changing this geometry affects propagation paths, so regenerate the frames
when comparing different road widths.

Recipes that combine a tree inventory with mapped vegetation can use the
[vegetation merge helper](vegetation_merge.py). It keeps the source of each
tree and distinguishes recorded positions from inferred ones.

## Related guides

- [OpenStreetMap scene generation](../../docs/generator/geography/osm_scene_generation.md) covers scenario configuration, output files, and data attribution.
- [Acquisition and caching](../../docs/generator/geography/osm_scene_generation.md#reuse-and-recovery) explains download budgets, endpoints, and recovery after a failed request.
- The [Paris Opéra scenario](../../scenarios/generator/paris_opera_osm/README.md) shows how a geographic scene and street network fit into an ORCHAV experiment.

The builder includes code adapted from NVlabs' Sionna Large Radio Maps and Geo2SigMap. See the [module notice](NOTICE.md) for upstream credits and license information.
