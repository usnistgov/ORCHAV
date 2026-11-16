# Generator source

The Generator is ORCHAV's main [frame producer](../docs/reference/glossary.md#producer).
It reads a scenario,
prepares its scene and [actors](../docs/reference/glossary.md#actor), and orchestrates
[Sionna RT](https://nvlabs.github.io/sionna/rt/) to compute propagation paths.
It converts simulation state into the shared
[`StandardMPCFrame` contract](../docs/shared/frame_reference.md), so consumers
can read a recording without running the simulation again.

The [shared data layer](../shared/README.md) owns the frame contract, storage
codecs and providers. Generator output adapters write local
[HDF5 frame sets](../docs/shared/frame_reference.md) or serve propagation frames
over gRPC. Providers expose those frames to the
[Visualizer](../visualizer/README.md), sensing postprocessing and analysis tools.
`frames/` is the reusable recording. `summary/` contains standalone diagnostic
figures that can be inspected independently of the Visualizer.

In a [sensing scenario](../docs/sensing/README.md), the Generator also acquires
channel observations and can save range and Doppler maps, detections, or tracks.

To generate your first recording, follow [Hello World](../scenarios/getting_started/hello_world/README.md). For scenario configuration and output options, see the [Generator guide](../docs/generator/README.md).

| Folder | Purpose |
| --- | --- |
| `core/` | Scenario preparation, mobility, propagation, and the [offline simulation loop](core/pipeline/offline_pipeline.py). |
| `figures/` | Scene, motion, coverage, and [summary plots](figures/generator_summary_fig.py). |
| `io/` | Frame conversion, [HDF5 output](io/storage/hdf5_frame_output.py), and gRPC delivery. |
| `tools/` | Command-line utilities for working with simulation assets and outputs. |
| [osm/](osm/README.md) | Build scene geometry and street networks from OpenStreetMap data. |
