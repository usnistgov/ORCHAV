# Planar-Array Configuration

This is the elevation part of the [azimuth and elevation example](../README.md).
Its topology, settings, commands and interpretation are on that shared page.

The [YAML](scenario.yaml) uses an 8 × 8 receive array to observe a rising cube.
Detection estimates both angles; tracking is disabled. Generation writes
`frames/` and `summary/` here.

Generate the planar-array recording:

```bash
orchav-generator scenarios/sensing/cube_radar/azimuth/elevation
```

Open its angular results:

```bash
orchav-visualizer --scenario scenarios/sensing/cube_radar/azimuth/elevation --no-resume
```

Continue to [Cartesian tracking](../../moving_tracking_3d/README.md).

## Benchmark

| Invocation | Elapsed |
|---|---:|
| Generation | 27.73 s |
