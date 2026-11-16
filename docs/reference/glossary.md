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

### Analytical Doppler

In analytical Doppler (`coherent_cpi`), one ray-tracing solve supplies the paths
for a coherent processing interval. ORCHAV holds the geometry fixed and evolves
path phase using bulk motion over the pulses. See [Mode Selection](../sensing/concepts.md#acquisition-modes-and-timing).

### Consumer

A consumer is a program that reads [frames](#frame) to display, inspect, or
analyze their contents. The [Visualizer](#visualizer), `orchav-inspect`,
and Python analysis code are consumers.

The [sensing postprocessor](../sensing/reprocessing.md#run-processing) also consumes saved frames.

See [Producer, Shared Data Layer, And Consumer
Roles](../shared/README.md#producer-shared-data-layer-and-consumer-roles) for
the complete flow.

### Constant-RCS Target

A constant-RCS target uses Sionna RT's native RCS module to model the target
return with a single scattering point assigned a fixed radar cross section
(in square meters), independent of incident and scattered directions. Its
geometry is an absorbing proxy for blockage, not the source of the scattered
return.

See [Target representations](../sensing/target_models.md) for configuration
and comparison with mesh-based targets.

### Coverage Map

A coverage map shows how a radio quantity, such as [path loss](#path-loss),
varies across a grid of locations at one or more heights. It answers questions
about an area, while the [propagation paths](#propagation-path-mpc) in a
[frame](#frame) describe paths between particular
[transmitters](#transmitter-tx) and [receivers](#receiver-rx).

See [Coverage Maps](propagation.md#coverage-maps) for how area
estimates relate to individual propagation paths.

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

See [Path Metrics](propagation.md#path-metrics) for the other stored
measurements.

### Delay-Bin Accumulation

For `point_bin_floor`, ORCHAV assigns each path to a delay bin by rounding its
sample delay down, then adds the complex gains of paths in the same bin.
See [range-response models](../sensing/concepts.md#range-response-models).

### Frequency-Domain Channel Synthesis

For `cfr_ifft`, ORCHAV builds the channel frequency response (CFR) from the
paths' complex gains and continuous delays. It weights the frequency samples
and applies an inverse fast Fourier transform (IFFT) to obtain delay samples.
See [range-response models](../sensing/concepts.md#range-response-models).

### Frame

A frame is one snapshot of a [scenario](#scenario) at one step on its
[timeline](#timeline). It records where the [actors](#actor) are and
the [propagation paths](#propagation-path-mpc), if any, found between the
[transmitters](#transmitter-tx) and [receivers](#receiver-rx). A scenario can
produce one frame or a sequence of frames.

With sensing enabled, a saved
frame can also contain channel observations and selected sensing products
such as range profiles, detections, or tracks. A product may summarize
observations from several timeline steps.

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

When sensing is enabled, it
samples the channel and writes selected observations and products into local frames.

See the [Generator guide](../generator/README.md) for its workflows.

### gRPC

gRPC is the network communication used by Live Generator and Remote HDF5
Playback. ORCHAV encodes a propagation [frame](#frame) as a Protocol Buffers message, sends
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

See [Interaction Types](propagation.md#interaction-types) for the
supported meanings.

### Material

A material describes how a [scene](#scene) or [target](#target) surface affects
radio energy, including reflection, scattering, and penetration behavior.
These radio properties help determine the
[propagation paths](#propagation-path-mpc) that [ray tracing](#ray-tracing)
finds. A surface's visual color or texture is a separate rendering choice.

See [Materials](propagation.md#materials) for their role in propagation
and [Material Overrides](../generator/configuration.md#material-overrides) for
configuration.

<a id="meshmaterial-sensing"></a>

### Mesh-Based Sensing

In mesh-based sensing, the surfaces of a target model act as scattering surfaces. Sionna RT
computes the target return from interactions with explicit mesh surfaces and
their radio materials. The result depends on geometry, pose, visibility,
materials, and enabled propagation mechanisms rather than a prescribed target
RCS. This workflow does not require Sionna RT's native RCS module.

Compare [constant-RCS](#constant-rcs-target) and [TR 38.901](#tr-38901-sensing-target)
models in [Native RCS Targets](../sensing/target_models.md).

### Path Loss

Path loss describes how much a radio signal is weakened along one
[propagation path](#propagation-path-mpc). It is measured in decibels (dB). A
larger path-loss value generally means a weaker received contribution.

See [Path Metrics](propagation.md#path-metrics) for related values.

### Path Metric

A path metric is a measured or derived value for one
[propagation path](#propagation-path-mpc), such as [delay](#delay),
[path loss](#path-loss), or arrival and departure angles. Metrics help compare
paths. They do not change the path geometry.

See [Path Metrics](propagation.md#path-metrics) for the values
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
[interactions](#interaction-bounce) with the surroundings. Several paths can
exist for the same transmitter-receiver pair, which is why a path is also
called a *multipath component* (MPC).

See [Multipath Components](propagation.md#multipath-components) for
the stored path information.

### Ray Tracing

Ray tracing calculates paths that radio energy can follow through a
[scene](#scene). ORCHAV prepares the [scenario](#scenario), asks Sionna RT to
compute the [propagation paths](#propagation-path-mpc), and converts the results
into [frames](#frame). ORCHAV orchestrates Sionna RT. It does not replace the
ray tracer.

See [Ray Tracing](propagation.md#ray-tracing) for its role in an
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
connection is used, it carries propagation frames over
[gRPC](#grpc). This lets the [Visualizer](#visualizer), terminal inspection,
and Python analysis use the same generated data.

See the [Shared Data Layer guide](../shared/README.md) for the frame flow and
data-mode choices.

### Simulated Doppler

In simulated Doppler (`sequential_stft`), each pulse uses a new ray-tracing solve
with the geometry at that instant. ORCHAV estimates Doppler from the resulting
sequence of complex channel samples. See [Mode Selection](../sensing/concepts.md#acquisition-modes-and-timing).

### Summary Figure

A summary figure is an image produced after a [scenario](#scenario) runs. It
can show the [actor](#actor) layout, motion, direction, or a
[coverage map](#coverage-map) without opening the [Visualizer](#visualizer).
It is a diagnostic picture, not a reusable [frame](#frame) or
[frame set](#frame-set).

See [Summary and Coverage Figures](../generator/generated_figures.md) for the
available figures.

### Target

A target is an [actor](#actor) such as a vehicle or person. It can stay still
or move. Mesh targets affect [propagation paths](#propagation-path-mpc) through
their geometry and [material](#material). Unlike a
[transmitter](#transmitter-tx) or [receiver](#receiver-rx), a target is not a
radio endpoint.

Native sensing targets use an [RCS model](../sensing/target_models.md) for their return.

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

### TR 38.901 Sensing Target

A 3GPP TR 38.901 sensing target uses Sionna RT's native RCS model specified
in clause 7.9.2 of the 3GPP report. Human, vehicle, and UAV presets use one
or more scattering points, with model-dependent angular responses and optional
statistical and polarization behavior. Any mesh serves as a blockage proxy.
The prescribed scattering model supplies the target return.

Unlike a [constant-RCS target](#constant-rcs-target), this representation
uses standardized target-type-specific response models rather than a single
fixed-RCS scattering point.

See [Native RCS Targets](../sensing/target_models.md) for the
supported presets and configuration.

### Visualizer

The ORCHAV Visualizer is the primary interactive [frame consumer](#consumer)
that combines a [scene](#scene) with generated [frames](#frame). It lets users inspect
[actors](#actor), [propagation paths](#propagation-path-mpc),
[path metrics](#path-metric), [coverage maps](#coverage-map), and motion. A
[renderer](#renderer) draws the 3D view.

For local sensing runs it also displays saved radar products, detections, and tracks.

See [Your First Visualizer Session](../visualizer/first_session.md) for the
short interface tour.

<a id="sensing-glossary"></a>

## Sensing terms

These terms describe acquisition, processing and the resulting observations.
The [sensing concepts](../sensing/concepts.md) connect them in processing order.

### Acquisition

Acquisition is the process of generating simulated complex channel observations
and their timestamps from the configured scene and motion. The acquisition mode
chooses whether one solved geometry supplies a whole coherent interval or each
sample receives an independent solve. See [acquisition modes](../sensing/concepts.md#acquisition-modes-and-timing).

### Analysis window

An analysis window is the set of consecutive samples processed together into
one spectrum. Its length controls the time/frequency tradeoff, while the number
of samples advanced between windows controls update cadence. See
[Doppler processing](../sensing/concepts.md#doppler-and-velocity).

### Bandwidth

Bandwidth is the frequency span used to construct the simulated channel
response. It sets the physical delay-resolution scale in CFR/IFFT processing.
See [finite-bandwidth range models](../sensing/concepts.md#range-response-models).

### CFR

The channel frequency response (CFR) gives the complex channel gain at sampled
frequencies. ORCHAV synthesizes it from the ray-traced MPC gains and delays.
Its IFFT supplies a finite-bandwidth delay response. See
[range-response models](../sensing/concepts.md#range-response-models).

### CIR

The channel impulse response (CIR) describes how received signal contributions
are distributed over propagation delay. With ORCHAV's `cfr_ifft` model, an IFFT of
the frequency-windowed CFR produces a **sampled, band-limited CIR** (also called
a delay response). Its range-bin shape depends on bandwidth and weighting.
This is not a full transmitted-waveform receiver. See
[range-response models](../sensing/concepts.md#range-response-models).

### CFAR

Constant false alarm rate (CFAR) detection compares a cell's power with a
threshold derived from neighboring background cells. See
[detection decisions](../sensing/detection_and_tracking.md#what-each-stage-decides).

### Clutter suppression

Clutter suppression reduces components that a configured processing model treats
as unwanted background, such as contributions constant over slow time. Mean
removal can also attenuate stationary targets and slow motion. See
[clutter operators](../sensing/concepts.md).
The [static-clutter experiment](../../scenarios/sensing/foundations/static_and_moving_cubes/README.md)
compares processing of one stationary and one moving reflector.

### CPI

A coherent processing interval (CPI) is a block of complex samples processed
together while retaining their relative phase. In ORCHAV's `coherent_cpi` mode,
one solved geometry supplies that block and motion produces analytical phase
progression. See [coherent acquisition](../sensing/concepts.md#acquisition-modes-and-timing).

### Detection

A detection is a reported observation selected by the configured thresholding,
peak selection, grouping and filtering steps. See
[detection stages](../sensing/detection_and_tracking.md#detection).

### Doppler

Doppler frequency describes phase change over slow time and is used to estimate
path closing rate. ORCHAV's monostatic velocity convention is positive for
approach. See [Doppler and velocity](../sensing/concepts.md#doppler-and-velocity).

### FFT

A fast Fourier transform (FFT) computes the discrete frequency spectrum of
uniformly sampled data. ORCHAV applies a Doppler FFT across complex slow-time
samples. See [spectral processing](../sensing/concepts.md#doppler-and-velocity).

### Ground truth

Ground truth is the known scene state or a reference observable derived from
that state, used to evaluate an estimate. Production detection and tracking use
simulated observations rather than target truth. See
[evaluation](../sensing/evaluation.md).

### Hann window

A Hann window smoothly weights samples before a transform to reduce sidelobes,
while broadening the main response. Frequency-domain Hann weighting shapes the
range response, and slow-time Hann weighting shapes the Doppler response. See
[windowing tradeoffs](../sensing/concepts.md#range-response-models).
Try the [range-windowing experiment](../../scenarios/sensing/foundations/windowing/README.md)
to compare the two frequency windows on the same moving-cube recording.

### IFFT

An inverse fast Fourier transform (IFFT) converts uniformly spaced frequency
samples into a sampled delay response. Increasing its length through zero-padding
samples that response more densely without adding bandwidth. See
[range-response settings](../sensing/concepts.md#range-response-models).

### Monostatic

Monostatic sensing uses a collocated transmitter and receiver. Its range
convention is half the total propagation path length. See
[range interpretation](../sensing/concepts.md#range-and-range-profiles).

### Micro-Doppler

Micro-Doppler describes Doppler modulation from motion within a target, such as
limb articulation, in addition to its bulk motion. A time–Doppler plot alone
does not identify the moving body part. Interpretation depends on the simulated
geometry, pose cadence and acquisition model. See the
[motion-resolution discussion](../sensing/concepts.md#motion-and-phase-consistency).

### Range bin

A range bin is one discrete sample position on a saved range axis. Bin spacing
is different from localization accuracy or the ability to resolve two nearby
returns. See [range sampling](../sensing/concepts.md#range-response-models).

### Range-Doppler map

A range-Doppler map organizes the result of slow-time spectral processing by
range and Doppler frequency, often displaying power with a velocity axis. One
return can span several cells. See [range-Doppler products](../sensing/concepts.md#power-clutter-and-detections).

### Range profile

The saved range profile is the power of the last valid acquisition sample's
delay response, using the declared receive-element combination and before
slow-time mean removal or windowing. It is not a sum over Doppler bins. See
[saved products](../sensing/concepts.md).

### SISO

Single-input single-output (SISO) sensing uses one transmit element and one
receive element. This configuration provides no receive-array angular estimate.
See [supported topologies](../sensing/concepts.md#acquisition-modes-and-timing).

### Sensor fusion

Sensor fusion combines observations from different sensor views in a common
coordinate frame and time reference. ORCHAV's synchronized monostatic workflow
uses independent range, angle and closing-velocity observations to estimate
world-coordinate tracks. See [multi-sensor fusion](../sensing/detection_and_tracking.md#multi-sensor-fusion)
and the [drone experiment](../../scenarios/sensing/applications/drone_tracking/README.md).

### Slow time

Slow time indexes successive complex channel samples, often called pulses in
the sensing configuration. Doppler processing follows phase evolution along
this axis, separately from the delay/range axis. See
[channel and Doppler samples](../sensing/concepts.md#doppler-and-velocity).

### Specular reflection

Specular reflection is a mirror-like surface interaction. The incident ray,
reflected ray and local surface normal lie in one plane, and the incidence and
reflection angles to that normal are equal. See
[propagation interactions](propagation.md#interaction-types).

### STFT

A short-time Fourier transform (STFT) applies a Fourier transform to successive
analysis windows to show how a spectrum changes over time. Overlapping windows
share samples, so a shorter hop gives more frequent outputs without improving
the unchanged window's Doppler resolution. See
[acquisition timing](../sensing/concepts.md#result-timing).

### Tracking

Tracking associates detections over time and estimates their evolving states
under a chosen motion model. Track confirmation requires repeated support, and
track IDs are assigned independently of scene target names. See
[association and confirmation](../sensing/detection_and_tracking.md#association-and-confirmation).

---

Home: [Documentation](../README.md)
