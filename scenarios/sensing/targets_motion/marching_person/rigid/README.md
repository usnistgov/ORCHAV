# Rigid Human-Mesh Configuration

This is the first step of the [human-mesh example](../README.md#1-rigid-mesh-shape-and-bulk-motion),
which contains the topology, parameter explanations, commands and interpretation.
It uses a bundled non-animated mesh; no external body model is needed.

The [YAML](scenario.yaml) translates one fixed pose, detects its returns and
tracks range and radial velocity. Generation writes `frames/` and `summary/`
beside this file.

Generate the rigid-human recording:

```bash
orchav-generator scenarios/sensing/targets_motion/marching_person/rigid
```

Open the mesh and its sensing results:

```bash
orchav-visualizer --scenario scenarios/sensing/targets_motion/marching_person/rigid --no-resume
```

Continue with the [articulated comparison](../README.md#2-prepare-the-user-supplied-motion).

## Benchmark

| Invocation | Elapsed |
|---|---:|
| Generation | 19.68 s |
