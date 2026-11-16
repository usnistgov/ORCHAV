# Prepare Paris Opéra Inputs

## Preparation

Run from the repository root after installing ORCHAV with its OSM dependencies:

```bash
python -m pip install -e ".[osm]"
```

The city, credited textures and vehicle inputs are bundled under
[inputs/](inputs/README.md). Supply a canonical walking manifest from the
[SMPL importer](../../../docs/generator/motion/smpl_mesh_generation.md), the
[Kimodo importer](../../../docs/generator/motion/kimodo_motion_import.md), or
another compatible source. Importing raw SMPL/AMASS inputs also needs the
`[smpl]` extra and separately obtained model and motion files.
The `animated_target.json` file and its pose resources must already exist;
the Paris preparation command does not obtain or import raw walking inputs.

Use a clip covering at least 5.1195 s for the full recording without repetition.
Preparation keeps its articulation, scale and pose times, replaces the source
root motion with the Paris route, and applies the declared front/up basis.
For a pelvis-centred asset, supply `--walking-offset X Y Z` in metres to place
the feet at road level; preparation does not ground each pose automatically.

Prepare a new run directory using the command template below. Replace
`WALKING_ASSET_JSON` with the actual path to your existing `animated_target.json`
file, quoting the path if it contains spaces. It is a required user-supplied
input, not a file bundled with this scenario.

```bash
python scenarios/showcases/paris_opera_multitarget/prepare.py \
  --walking-asset WALKING_ASSET_JSON \
  --output scenarios/showcases/paris_opera_multitarget/runs/my_run \
  --epochs 80 --rays 800000
```

The output must not already exist. It contains the scene, assets and generated
`scenario.yaml`; copy the complete run if moving it elsewhere. For a short
check, use `--epochs 4` in a separate output directory.

Continue with generation, inspection and optional review analysis in the
[main walkthrough](README.md#running).

<details>
<summary>Short clips, historical walking inputs and surface options</summary>

A shorter clip is rejected unless `--pose-period SECONDS` explicitly declares
repetition. The period must exceed its last relative pose time, preserving the
last pose's sample slot. For example, 263 poses at 120 Hz have a 263/120 s
period **if you deliberately choose to repeat the entire clip**. That does not
make the endpoint postures seamless or recover the historical gait. Every
output-to-source pose, source time and cycle index is recorded in
`assets/walker/pose_mapping.json` and authenticated adapter provenance.

For the 263-pose CMU `07_12_poses.npz` clip and male SMPL model, import at
native 120 Hz with the documented AMASS importer first. The checked input pair
is identified by motion SHA-256
`5b1b8ec0e71f9b2df83b69367bf7ae39c60a2ad5b743407d10a80fe046a55d07`
and model SHA-256
`a8d91bb7305ceeb407e73098371750ac8f6076f9dc2aea8f587018e98be048bc`.
For this pair, `--pose-period 2.191666666666667 --walking-offset 0 0
0.9261288423769855` repeats the supplied poses and places the first pose's
lowest vertex at the road height. This is one fixed root offset for the entire
recording; later poses retain their original local heights. Other clips or
models require their own explicit placement choice.

The separately supplied historical library is an alternative input. Its directory
must contain all 34 `smpl_walk_*.ply` files and the declaring `mesh_library.json`.
Replace `WALK_LIBRARY_DIR` in this command template with that existing directory's
actual path, quoting it if it contains spaces:

```bash
python scenarios/showcases/paris_opera_multitarget/prepare.py \
  --walk-library WALK_LIBRARY_DIR \
  --output scenarios/showcases/paris_opera_multitarget/runs/historical_input \
  --epochs 80 --rays 800000
```

It preserves the historical 30 Hz sampling and explicit 34/30 s cycle. The
original raw-clip selection and body-normalization derivation are not established
by that library metadata; an AMASS CMU07_12 import must not be called identical.
Neither library, model, motion clip nor derived body geometry is bundled.

An optional matching `--walking-surface` accepts the actual path to an existing
`surface.json` and copies its local texture and UV resources into the run.
Without it the pedestrian uses scalar
appearance; no absent SMPL texture is referenced. A different vertex topology
requires its own matching UV map.

`--start 4.8` shifts both the route and pose clock by 4.8 s for a later
segment. This changes the sampled interval; it does not reproduce the full
80-CPI acquisition.

</details>


Return to the [walkthrough](README.md#running).
