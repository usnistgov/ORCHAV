# External Ray-Tracer Import

This example converts plain ray-path records from an external propagation tool
into an ORCHAV frame set. A small frame-producing adapter assigns ORCHAV units
and meanings before the HDF5 frame set is written.

```text
External ray-path source (fake_raytracer.py records)
    -> frame-producing adapter (import_to_orchav.py)
    -> ORCHAV frame (StandardMPCFrame)
    -> FrameSetWriter
    -> ORCHAV HDF5 frame set
    -> Hdf5Provider
```

The deterministic input contains two frames, two transmitters, two receivers,
two sparse TX/RX pairs, LoS and reflected paths, all six path metrics, bounce
materials, and a moving target. It is an interoperability example, not a
physical propagation simulation.

## Run The Import

Run from the repository root and choose a destination that does not exist:

```bash
python -m examples.external_raytracer_import.import_to_orchav external_raytracer_frames
```

The command creates HDF5 chunks and `frames_manifest.json`, reloads both frames
through `Hdf5Provider`, and prints the frame-set identity. Inspect the result
without opening the Visualizer:

```bash
orchav-inspect external_raytracer_frames --frame 0
orchav-inspect external_raytracer_frames --frame 1
```

`FrameSetWriter.create_new()` never replaces an existing path. Choose another
destination for a second run, or remove the disposable result yourself after
you finish inspecting it.

## Where The Boundary Lives

[`fake_raytracer.py`](fake_raytracer.py) represents code owned by external
software. It imports only the Python standard library and reports source-native
records:

- positions in meters.
- device orientations in degrees.
- path delay in seconds.
- arrival and departure angles in radians.
- named bounce mechanisms and material families.

[`import_to_orchav.py`](import_to_orchav.py) is the source-specific adapter. It
maps stable device names to indices, groups paths by TX/RX pair, converts units,
maps each physical interaction to a canonical code, and calls
`standard_mpc_frame_from_pair_data()`. LoS paths have no bounce point or
interaction entry. The adapter then gives ORCHAV frames (`StandardMPCFrame`
objects) to `FrameSetWriter`. It does not construct HDF5 datasets or manifests
itself.

`Hdf5Provider` appears only after the HDF5 frame set has been written.
It is the [frame provider](../../docs/reference/glossary.md#frame-provider) used
by [consumers](../../docs/reference/glossary.md#consumer) such as the Visualizer
and Python analysis. Importing external results therefore does not require a
new frame provider implementation.

## Adapting Real External Results

Keep the same three responsibilities separate:

1. Decode the external software's files or API into source-owned records.
2. Convert coordinates, units, interaction meanings, materials, device state,
   and target state into one `StandardMPCFrame` per time step.
3. Append those frames in increasing frame-ID order through
   `FrameSetWriter.create_new()`.

An adapter may use the direct compact `StandardMPCFrame` constructor when its
source already supplies flat arrays and offsets. Per-pair padded or ragged
sources can use `standard_mpc_frame_from_pair_data()` as this example does.
Neither path requires Generator configuration or knowledge of the physical
HDF5 layout.

Desktop visualization also needs a scenario whose scene assets use the same
coordinate system as the imported paths. Place a newly imported `frames/`
directory under that scenario, or select it through a read-only scenario
profile. See the [Frame Contract](../../docs/shared/frame_reference.md#frame-contract) for the
canonical data boundary and [Choose a Data Mode](../../docs/shared/README.md#choose-a-data-mode)
for the available loading workflows.

---

Up: [Examples](../README.md)
