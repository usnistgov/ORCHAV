# Shared package

The Generator, Visualizer, and command-line tools exchange data through this package. Start with [the shared data guide](../docs/shared/README.md) to choose between local HDF5 playback, a live Generator, and remote HDF5 playback.

| Folder | Purpose |
| --- | --- |
| `frames/` | The [frame contract](../docs/shared/frame_reference.md), validation, storage, and [frame providers](frames/providers.py). |
| `scenarios/` | [Scenario models](scenarios/model.py), YAML loading, and path resolution. |
| `cli/` | Commands for [inspecting frames](cli/inspect.py) and [validating scenarios](cli/validate.py). |
| `coverage/` | [HDF5 schema](coverage/schema.py) for radio coverage data. |
| `extensions/` | Contracts for optional frame payloads. |
| `geometry/` | Scene geometry extraction and caching. |
| `logging/` | Logging configuration, filters, and formatters. |
| `protos/` | The [wire protocol](protos/visualizer.proto) and generated gRPC bindings. |
| `statistics/` | Propagation metrics, statistics, and plotting. |
| `sensing/` | Process saved channel observations, detect and track targets, and evaluate the results. See the [postprocessor](sensing/postprocessor.py) or run `orchav-sensing --help`. |
