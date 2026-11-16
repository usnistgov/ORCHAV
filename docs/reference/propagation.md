# Propagation Concepts

A radio signal can reach a receiver along several propagation paths through the same
[scene](glossary.md#scene). ORCHAV uses
[Sionna RT](https://nvlabs.github.io/sionna/rt/) to compute these paths so you
can relate channel behavior to geometry, materials and motion.

## Ray Tracing

The [Generator](../generator/README.md) places the
[actors](glossary.md#actor) and asks Sionna RT to find propagation
paths. Scene geometry and materials determine which direct and indirect paths
can reach the receiver. The enabled interactions and ray budget determine which paths
the solver searches. See [Simulation Configuration](../generator/configuration.md#ray-tracing-yaml-fields)
for those controls.

## Multipath Components

Each propagation path contributes a
[multipath component (MPC)](glossary.md#propagation-path-mpc) to the channel.
A [frame](glossary.md#frame) records the MPC geometry and channel measurements
for one simulation step. The [MPC Explorer](../visualizer/path_inspection.md) lets
you select individual MPCs and inspect their geometry and measurements together.

## Interaction Types

A propagation path can include reflection, diffuse scattering, refraction or
diffraction. Several interactions can occur along one path. A direct
line-of-sight path reaches the receiver without
a surface interaction. The [frame reference](../shared/frame_reference.md#interaction-codes)
lists the stored codes used to distinguish them.

### Materials

[Radio materials](glossary.md#material) affect how much energy
follows each path. Visual appearance is separate from the radio response.
Use [material overrides](../generator/configuration.md#material-overrides) to
change that response in a simulation.

### Diffuse Scattering

A rough surface can scatter reflected energy into a spread of directions. This
can create additional paths beyond the mirror-like reflection from a smooth
surface. [Diffuse-scattering configuration](../generator/configuration.md#diffuse-scattering)
explains how solver settings and material coefficients work together.

## Antennas

Antennas weight the contributions arriving from or departing in different
directions. With the standard synthetic-array model, changing an array changes
its response to the paths while retaining their traced geometry. See
[antenna configuration](../generator/configuration.md#antenna-arrays) and
[beam-pattern inspection](../visualizer/beam_patterns.md).

## Path Metrics

Each MPC has channel measurements that describe its propagation characteristics.
[Delay](glossary.md#delay) measures travel time from transmitter to receiver.
[Path loss](glossary.md#path-loss) describes attenuation. Departure and arrival
angles specify the path direction at the transmitter and receiver, respectively.

[Visual Analysis](../visualizer/analysis.md) connects these measurements to
filters and plots. The [frame reference](../shared/frame_reference.md#units-and-validation)
defines their units and missing values, and [Statistics Helpers](../shared/statistics.md)
explains calculations across multiple paths.

## Coverage Maps

Paths describe particular transmitter–receiver links. A
[coverage map](glossary.md#coverage-map) instead samples potential
receiver locations across an area, independently of the placed receiver actors.
Coverage is stored as a separate grid dataset.

Use [coverage configuration](../generator/configuration.md#coverage-configuration)
to choose the sampled area and [coverage inspection](../visualizer/analysis.md#coverage-inspection)
to explore the saved quantities.

### Radio coverage volumes

Sampling several heights extends the map into a volume. [3D Volume Inspection](../visualizer/volume_inspection.md)
explains slices, thresholds and missing data. Radio coverage describes the
communication channel and does not by itself establish sensing detectability.

## Motion Across Frames

As actors move, propagation paths can appear, disappear or change length and direction.
Playback lets you relate those changes to the moving scene. Use
[Mobility and Orientation](../generator/mobility_and_orientation.md) to define
the motion.
[ORCHAV Sensing](../sensing/README.md) also processes channel observations over
time to estimate range, Doppler and direction, then detect and track targets.

---

Home: [Documentation](../README.md) | Related: [Glossary](glossary.md) | Begin: [Hello World](../../scenarios/getting_started/hello_world/README.md)
