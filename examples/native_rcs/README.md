# Small native RCS control

This self-contained control acquires two 16-pulse coherent processing intervals
for a rigid 1 m² constant-RCS target approaching a monostatic radar. It exercises
native target construction, finite-bandwidth acquisition, retained identities
and ordinary CPU replay. For illustrated detection and tracking examples, use
the [native RCS scenarios](../../scenarios/sensing/foundations/RCS/README.md).

Install the [native RCS extra](../../docs/sensing/target_models.md) first. From the
checkout root, use the paths below outside the source tree. Choose different
directory names if either output directory already exists.

Copy the example to a separate working directory:

```bash
cp -R examples/native_rcs ../orchav-native-control
```

Validate its required assets:

```bash
orchav-validate ../orchav-native-control --require-assets
```

Generate the native RCS recording:

```bash
orchav-generator ../orchav-native-control
```

Inspect frame 31:

```bash
orchav-inspect ../orchav-native-control --frame 31
```

Reprocess the saved observations into a separate recording:

```bash
orchav-sensing ../orchav-native-control/frames -o ../orchav-native-replay
```

Open the original recording:

```bash
orchav-visualizer --scenario ../orchav-native-control/scenario.yaml
```

The expected completed results are stored at frames 15 and 31. Inspect their
recorded range/Doppler and native target/point identity. The proxy is absorbing
and occluding; the native model supplies the target return. Geometry is fixed
within each CPI and translation enters through native Doppler. This small
software control does not establish calibrated target strength or arbitrary
motion/multipath fidelity.

Up: [Examples](../README.md) · [Native model contract](../../docs/sensing/target_models.md)
