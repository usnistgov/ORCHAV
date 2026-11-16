# Procedural Car-Shaped Target

This is an original ORCHAV mathematical prism, generated from the explicit
side profile in [build_mesh.py](build_mesh.py). It contains no downloaded,
scanned, copied, or transformed vehicle mesh. The recipe and generated asset
use the repository's GPL-3.0-only license.

The closed mesh is 4 m long, 1.7 m wide, and 1.4 m high. The lower surface is
z = 0 and the front is local −X. [Target metadata](target_metadata.json)
declares a 180 deg yaw offset for mobility-aligned orientation.

It is a deliberately simple geometric target, not a calibrated vehicle radar
cross section, detailed car model, or equivalent replacement for historical
vehicle results. Regenerate the acquisition and score its outputs after
selecting this asset. Old figures using a different mesh do not validate it.

Run commands from the repository root.

Rebuild the procedural car mesh:

```bash
python libraries/targets/procedural_car/build_mesh.py
```

Check that the saved mesh matches the generator:

```bash
python libraries/targets/procedural_car/build_mesh.py --check
```

Use `path: libraries/targets/procedural_car`, `pattern: car.ply`, and
`source: directory` in a target asset block. See the
[mesh-target tutorial](../../../scenarios/generator/targets/mesh_targets/README.md).
