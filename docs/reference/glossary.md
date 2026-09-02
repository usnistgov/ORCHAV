# Glossary

Use this page when an ORCHAV term is unfamiliar. The definitions explain what
the terms mean in ORCHAV. The links after each definition lead to instructions
and technical details.

## Terms

### Actor

*Actor* is ORCHAV's general name for a [transmitter](#transmitter-tx), a
[receiver](#receiver-rx), or a [target](#target) placed in a
[scenario](#scenario). Each actor has a position and direction at every step on
the scenario [timeline](#timeline), and it can stay still or move as time
advances.

See [Scenario Authoring](../generator/scenario_authoring.md#start-with-yaml) for
actor examples.

### Consumer

A consumer is a program that reads [frames](#frame) to display, inspect, or
analyze their contents. The [Visualizer](#visualizer), `orchav-inspect`, and
Python analysis code are consumers.

See [Producer, Shared Data Layer, And Consumer
Roles](../shared/README.md#producer-shared-data-layer-and-consumer-roles) for
the complete flow.

### Coverage Map

A coverage map shows how a radio quantity, such as [path loss](#path-loss),
varies across a grid of locations at one or more heights. It answers questions
about an area, while the [propagation paths](#propagation-path-mpc) in a
[frame](#frame) describe routes between particular
[transmitters](#transmitter-tx) and [receivers](#receiver-rx).

See [Coverage Maps](../concepts/propagation.md#coverage-maps) for the workflow.

### Data Mode

A data mode tells the [Visualizer](#visualizer) where to get [frames](#frame):
from a saved [HDF5](#hdf5) [frame set](#frame-set) on the same computer, from a
running [Generator](#generator) acting as a [producer](#producer), or over
[gRPC](#grpc) from a frame set served by another computer.

See [Choose A Data Mode](../shared/README.md#choose-a-data-mode) for when to use
Local HDF5 Playback, Live Generator, or Remote HDF5 Playback.

### Delay

Delay is the time a radio signal takes to travel along one
[propagation path](#propagation-path-mpc). ORCHAV records it in nanoseconds. A
larger value means that path arrives later.

See [Path Metrics](../concepts/propagation.md#path-metrics) for the other stored
measurements.

### Frame

A frame is one snapshot of a [scenario](#scenario) at one step on its
[timeline](#timeline). It records where the [actors](#actor) are and
the [propagation paths](#propagation-path-mpc), if any, found between the
[transmitters](#transmitter-tx) and [receivers](#receiver-rx). A scenario can
produce one frame or a sequence of frames.

See the [Frame Contract](../shared/frame_reference.md#frame-contract) for the
stored fields.

### Frame Contract

The frame contract is the agreed structure of an ORCHAV [frame](#frame): its
fields, array shapes, units, and validation rules. It lets a
[producer](#producer) and a [consumer](#consumer) exchange frame data without
sharing [Sionna RT](https://nvlabs.github.io/sionna/) objects.
`StandardMPCFrame` is the Python type that implements
this contract.

See the [Frame Contract reference](../shared/frame_reference.md#frame-contract)
for the exact fields.

### Frame Manifest

The frame manifest is the `frames_manifest.json` file in a saved
[frame set](#frame-set). It lists the [HDF5](#hdf5) chunks and frame numbers
that belong together. Copy the manifest and its listed chunks as one set. Do
not edit the inventory by hand.

See the [HDF5 Frame Layout](../shared/frame_reference.md#hdf5-frame-layout) for
the file relationship.

<a id="provider"></a>

### Frame Provider

A frame provider is the software interface that retrieves [frames](#frame) for
a [consumer](#consumer). It hides whether those frames come from local files, a
running [Generator](#generator), or a remote frame server. Each provider
returns the same frame structure, so the consumer does not need separate logic
for each [data mode](#data-mode).

See [Frame Providers](../shared/frame_reference.md#frame-providers) for the
available implementations.

### Frame Set

A frame set is the group of files that stores a sequence of [frames](#frame) on
disk. In the `frames/` directory created for a [scenario](#scenario), the
[frame manifest](#frame-manifest) lists the [HDF5](#hdf5) chunk files that
belong to the set. Treat the manifest and those files as one collection when
copying or sharing generated data.

See the [HDF5 Frame Layout](../shared/frame_reference.md#hdf5-frame-layout) for
the on-disk organization.

### Generator

The ORCHAV Generator reads a [scenario](#scenario), prepares every
[actor](#actor) on its [timeline](#timeline), asks Sionna RT to perform
[ray tracing](#ray-tracing), and creates [frames](#frame),
[coverage maps](#coverage-map), or [summary figures](#summary-figure). It is
ORCHAV's primary [frame producer](#producer).

See the [Generator guide](../generator/README.md) for its workflows.

### gRPC

gRPC is the network communication used by Live Generator and Remote HDF5
Playback. ORCHAV encodes a [frame](#frame) as a Protocol Buffers message, sends
it between running processes, and reconstructs the same frame for a
[consumer](#consumer). gRPC carries data. It is not a ray tracer or a fourth
[data mode](#data-mode).

See [Choose A Data Mode](../shared/README.md#choose-a-data-mode) for the local,
live, and remote choices.

### HDF5

HDF5 is a file format for large, structured numerical arrays. ORCHAV uses it
to save [frame sets](#frame-set) and [coverage maps](#coverage-map). A saved
frame set also needs its [frame manifest](#frame-manifest), so an HDF5 chunk by
itself is not the complete saved output.

See the [HDF5 Frame Layout](../shared/frame_reference.md#hdf5-frame-layout) for
the files ORCHAV writes.

### Interaction (Bounce)

An interaction is a point where a [propagation path](#propagation-path-mpc)
meets or bends around [scene](#scene) geometry. For example, it may reflect off
a wall or bend around an edge. ORCHAV also calls an interaction a *bounce*. A
direct line-of-sight path has no bounces.

See [Interaction Types](../concepts/propagation.md#interaction-types) for the
supported meanings.

### Material

A material describes how a [scene](#scene) or [target](#target) surface affects
radio energy, including reflection, scattering, and penetration behavior.
These radio properties help determine the
[propagation paths](#propagation-path-mpc) that [ray tracing](#ray-tracing)
finds. A surface's visual color or texture is a separate rendering choice.

See [Materials](../concepts/propagation.md#materials) for configuration and
interpretation.

### Path Loss

Path loss describes how much a radio signal is weakened along one
[propagation path](#propagation-path-mpc). It is measured in decibels (dB). A
larger path-loss value generally means a weaker received contribution.

See [Path Metrics](../concepts/propagation.md#path-metrics) for related values.

### Path Metric

A path metric is a measured or derived value for one
[propagation path](#propagation-path-mpc), such as [delay](#delay),
[path loss](#path-loss), or arrival and departure angles. Metrics help compare
paths. They do not change the path geometry.

See [Path Metrics](../concepts/propagation.md#path-metrics) for the values
available in generated frames.

### Producer

A producer is a program that creates [frames](#frame). The ORCHAV
[Generator](#generator) is the primary frame producer. The included synthetic
generator and adapters for results from other propagation tools can also
create ORCHAV frames.

See [Producer, Shared Data Layer, And Consumer
Roles](../shared/README.md#producer-shared-data-layer-and-consumer-roles) for
the complete flow.

### Propagation Path (MPC)

A propagation path is one route that radio energy can take from one
[transmitter](#transmitter-tx) to one [receiver](#receiver-rx) during a
[frame](#frame). It can travel directly or include one or more
[interactions](#interaction-bounce) with the surroundings. Several routes can
exist for the same transmitter-receiver pair, which is why a path is also
called a *multipath component* (MPC).

See [Multipath Components](../concepts/propagation.md#multipath-components) for
the stored path information.

### Ray Tracing

Ray tracing calculates routes that radio energy can follow through a
[scene](#scene). ORCHAV prepares the [scenario](#scenario), asks Sionna RT to
compute the [propagation paths](#propagation-path-mpc), and converts the results
into [frames](#frame). ORCHAV orchestrates Sionna RT. It does not replace the
ray tracer.

See [Ray Tracing](../concepts/propagation.md#ray-tracing) for its role in an
ORCHAV run.

### Receiver (RX)

A receiver, abbreviated RX, is an [actor](#actor) that receives the simulated
radio signal. It is paired with a [transmitter](#transmitter-tx) when ORCHAV
computes [propagation paths](#propagation-path-mpc). Its position, direction,
and antenna settings can change the paths that are found.

See [Actors And Groups](scenario_yaml.md#actors-and-groups) for its YAML fields.

### Renderer

A renderer is the graphics backend that draws the [scene](#scene),
[actors](#actor), [propagation paths](#propagation-path-mpc), and overlays in
the [Visualizer](#visualizer). ORCHAV includes pygfx and Open3D/Filament
renderer paths. Changing the renderer changes presentation, not the saved
[frames](#frame).

See [Renderers](../visualizer/renderers.md) for when to use each backend.

### Scenario

A scenario is a directory that describes one ORCHAV study or example. Its
required `scenario.yaml` selects a [scene](#scene), defines the
[timeline](#timeline), declares the [actors](#actor), and chooses run
settings. It may also include input assets or a `generate.py` script. Generated
[frames](#frame), [coverage maps](#coverage-map), and
[summary figures](#summary-figure) are outputs that can be recreated. They are
not required scenario inputs.

See [Scenario Directory](../generator/scenario_authoring.md#scenario-directory)
for the expected files.

### Scene

A scene is the three-dimensional environment in which the study takes place,
including fixed geometry such as buildings and surfaces and their radio
[materials](#material). A [scenario](#scenario) selects a scene, then adds its
[actors](#actor) and settings. [Ray tracing](#ray-tracing) uses the scene to
find [propagation paths](#propagation-path-mpc).

See [Scene Selection](../generator/configuration.md#scene-selection) for the
available sources.

### Shared Data Layer

The Shared Data Layer connects a [producer](#producer) to its
[consumers](#consumer). It defines the [frame contract](#frame-contract), saves
[frame sets](#frame-set), and provides [frame providers](#frame-provider) that
retrieve frames through each supported delivery route. When a live or remote
connection is used, it carries those frames over
[gRPC](#grpc). This lets the [Visualizer](#visualizer), terminal inspection,
and Python analysis use the same generated data.

See the [Shared Data Layer guide](../shared/README.md) for the frame flow and
data-mode choices.

### Summary Figure

A summary figure is an image produced after a [scenario](#scenario) runs. It
can show the [actor](#actor) layout, motion, direction, or a
[coverage map](#coverage-map) without opening the [Visualizer](#visualizer).
It is a diagnostic picture, not a reusable [frame](#frame) or
[frame set](#frame-set).

See [Summary and Coverage Figures](../generator/generated_figures.md) for the
available figures.

### Target

A target is an [actor](#actor) represented by a three-dimensional mesh, such as
a vehicle or person. It can stay still or move, and its geometry and
[material](#material) can affect
[propagation paths](#propagation-path-mpc). Unlike a
[transmitter](#transmitter-tx) or [receiver](#receiver-rx), a target is not a
radio endpoint.

See [Add A Target](../generator/scenario_authoring.md#add-a-target) for an
example.

### Timeline

A timeline is the schedule that turns a [scenario](#scenario) into a sequence
of sampled moments. The scenario sets the number of moments, called *steps*,
and the total duration. At each selected step, the [Generator](#generator)
determines every [actor's](#actor) position and direction and normally produces
one [frame](#frame).

See [Common Scenario Fields](scenario_yaml.md#common-scenario-fields) for the
step count and duration settings.

### Transmitter (TX)

A transmitter, abbreviated TX, is an [actor](#actor) that sends the simulated
radio signal. It is paired with a [receiver](#receiver-rx) when ORCHAV computes
[propagation paths](#propagation-path-mpc). Its position, direction, and antenna
settings can change the paths that are found. Optional transmit power affects
received-strength calculations.

See [Actors And Groups](scenario_yaml.md#actors-and-groups) for its YAML fields.

### Visualizer

The ORCHAV Visualizer is the primary interactive [frame consumer](#consumer)
that combines a [scene](#scene) with generated [frames](#frame). It lets users inspect
[actors](#actor), [propagation paths](#propagation-path-mpc),
[path metrics](#path-metric), [coverage maps](#coverage-map), and motion
without rerunning the [Generator](#generator) for ordinary saved playback. A
[renderer](#renderer) draws the 3D view.

See [Your First Visualizer Session](../visualizer/first_session.md) for the
short interface tour.

---

Home: [Documentation](../README.md)
