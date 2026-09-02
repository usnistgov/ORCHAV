# Shared Package

The `shared` package contains the contracts used by the Generator and
Visualizer:

- `shared.frames`: `StandardMPCFrame`, validation, HDF5 frame providers, and protobuf.
- `shared.scenarios`: scenario models, YAML loading, and path policy.
- `shared.coverage`: coverage HDF5 schema constants.
- `shared.geometry`: scene geometry extraction and cache helpers.
- `shared.extensions`: optional structured frame payload contracts.

For the user-facing overview, see [docs/shared/README.md](../docs/shared/README.md).
