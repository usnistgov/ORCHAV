# Application Configuration (`app.toml`)

`config/app.toml` contains checkout-local defaults for Visualizer asset lookup
and the Live Generator endpoint. Keep portable scenario behavior and asset
references in `scenario.yaml`.

The file is resolved from the project root. Relative paths inside it also start
at the project root. If the file is missing or cannot be read, ORCHAV uses the
built-in defaults shown below.

## Supported Settings

```toml
[paths]
scenes = "libraries/scenes"
ibl = "libraries/ibl"

[live_grpc]
sionna = "grpc://localhost:50051"
```

| Key | Applies to | Behavior |
|-----|------------|----------|
| `paths.scenes` | Desktop Visualizer | Selects the library directory used when a scenario declares `scene.source: library`. |
| `paths.ibl` | Desktop Open3D/Filament renderer | Selects the directory containing paired `_ibl.ktx` and `_skybox.ktx` environments. |
| `live_grpc.sionna` | Generator and Visualizer | Supplies the default Live Generator advertised host and port and the Visualizer connection target when the scenario does not provide one. |

Generator products remain in the scenario's managed `frames/`, `summary/`,
and `coverage/` directories. Configure those workflows in `scenario.yaml`.

## Live Generator Endpoint Precedence

ORCHAV resolves the Live Generator endpoint in this order:

1. Process-local CLI overrides, when supplied.
2. Scenario endpoint settings, with `data.live_grpc.endpoint` taking
   precedence.
3. `[live_grpc].sionna` in `config/app.toml`.
4. The built-in localhost endpoint, with `ORCHAV_GRPC_SIONNA` available as a
   Visualizer-only environment fallback.

`--data-mode` selects a data mode for one process. `--grpc-port` replaces only
the configured port and keeps the configured host. Neither option rewrites
`scenario.yaml` or `app.toml`.

The advertised endpoint is separate from the Generator listener interface.
The Generator listens on `127.0.0.1` by default. Use `--bind-host HOST` only
when clients on a trusted network need to connect. See
[Choose A Data Mode](../shared/README.md#choose-a-data-mode) for workflow
selection and the [Data Mode Scenarios](../../scenarios/visualizer/data_modes/README.md)
for runnable commands.

## Optional Visualizer Environment Fallbacks

These variables apply when the scenario and checkout configuration do not
supply the corresponding Visualizer endpoint:

| Variable | Fallback target |
|----------|-----------------|
| `ORCHAV_GRPC_SIONNA` | Live Generator endpoint |
| `ORCHAV_REMOTE_HDF5_SERVER` | Remote HDF5 frame-file server |

---

Home: [Documentation](../README.md) | Related: [Shared Data Layer](../shared/README.md) | [Data Mode Scenarios](../../scenarios/visualizer/data_modes/README.md) | [Troubleshooting](../help/troubleshooting.md)
