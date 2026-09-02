# Scenario Builder

The Scenario Builder is a pygfx-only workspace for creating Generator-ready
ORCHAV YAML scenarios without editing YAML by hand. It uses the same scenario
schema and canonical actor-pose preparation as the Generator, so the motion
shown in the viewport is the motion used during generation.

The Builder is opt-in. Enable it for the current terminal, then launch the
Visualizer in authoring mode.

On Linux or macOS:

```bash
export ORCHAV_ENABLE_SCENARIO_BUILDER=1
orchav-visualizer --author
```

On Windows PowerShell:

```powershell
$env:ORCHAV_ENABLE_SCENARIO_BUILDER = "1"
orchav-visualizer --author
```

On Windows Command Prompt:

```bat
set ORCHAV_ENABLE_SCENARIO_BUILDER=1
orchav-visualizer --author
```

To open an existing scenario directly, pass either its directory or its
`scenario.yaml` file:

```bash
orchav-visualizer --author --scenario path/to/scenario
```

Schema-valid `scenario.yaml` files open in place, whether or not they already
contain the Scenario Builder document marker. Opening does not write the file.
The first explicit **Save** adds the marker while preserving settings outside
the Builder controls. A differently named YAML file opens read-only. Place it
in its own scenario directory as `scenario.yaml` before editing it.

`--author` always uses the pygfx renderer. Combining it with
`--renderer open3d` is rejected before the Qt application starts. The same
workspace is available from the top-level **Scenario Builder** menu while the
Builder is enabled. Its direct actions are grouped by purpose:

- **Create**: **New Scenario** and **Copy Current Scenario and Edit...**
- **Edit or Continue**: **Open Scenario for Authoring...**,
  **Edit Current Scenario**, and **Resume Authoring Draft**
- **Save**: **Save Scenario** and **Save Scenario As...**
- **Workspace**: **Return to Visualization**

When a scenario is already open for visualization, **Edit Current Scenario**
opens that same scenario in the embedded Builder. **Copy Current Scenario and
Edit...** asks only for a destination directory, creates an editable branch of
the active scenario, and opens that copy in the Builder. Entering the Builder
changes the bottom status to `Scenario Builder ready`.

## Supported Builder Surface

| Area | Supported |
|---|---|
| Scene | Supported sources can be imported. Sionna, ORCHAV library, and local XML scenes can be previewed without rebuilding during Open. |
| Actors | Any number of transmitters, receivers, and catalog-backed targets. Names are globally unique, and a valid scenario has at least one TX and one RX. |
| Groups | Shared group motion, actor-local right/forward/up offsets, and optional seeded per-member deviation |
| Mobility | All editable shared models: stationary, linear, waypoint, circular, survey, grid scan, oscillating, pendulum, figure eight, spiral, random sampling, Gauss-Markov, random waypoint, Manhattan grid, network route, mesh sequence, and group member |
| Orientation | All editable shared models: fixed, keyframes, align with motion, look at an actor or point, spin, and random |
| Target appearance | Catalog, file, and directory assets. Imported file/directory locators stay locked while material, scale, and applicable mesh animation remain editable. |
| Timeline | Steps, duration, quality preset, and path-metrics toggle |
| Output | Imported frame pattern, HDF5 format, compression, and chunk size are preserved. Configured Generator summaries are written alongside frames. |

Exact sampled trajectories are recognized, previewed, and preserved, but their
externally prepared positions remain read-only in the Builder.

The workspace does not edit coverage, antenna or RF settings, material
overrides, custom ray-tracing settings, live or remote output, Python-scripted
scenarios, imported target locators, or Open3D authoring. Imported YAML keeps
schema-recognized fields outside the Builder controls, but the Builder does not
edit or execute reserved extensions. Network-route mobility can use an
existing GraphML/XML or node-link JSON resource. Network creation is outside
the Builder. Use
[general scenario authoring](scenario_authoring.md) or a
[`generate.py` driver](scenario_authoring.md#python-scripted-actors)
for those cases.

## Quick Hands-on Check

After launching the Builder, this short sequence exercises the main
interaction boundaries before the detailed control reference:

1. Confirm that **ORCHAV library** and `empty/empty.xml` are selected by default.
2. Add and place one TX. Change its mobility to **Circular** and apply it. The
   30-sample prepared circle should appear immediately even though the missing
   RX still prevents saving.
3. Add an RX, choose **Waypoint**, and use **Draw Waypoints** for an arbitrary
   sequence. Increase **Steps** to see a denser prepared path without changing
   the authored waypoint count.
4. Select **Edit Motion**. Drag an individual model handle, then use the XYZ
   gizmo to translate the complete trajectory. One drag should produce one undo
   operation.
5. Add another actor and use **Form Group...**. Move the group path, then drag
   one member to change only its local offset.
6. Switch **Axes** between Off, Selected, and All. Try fixed, keyframes, align
   with motion, look-at, spin, and random orientations.
7. Validate and inspect the exact YAML preview before choosing a scenario
   directory with **Save As**. Keep the directory inside the active ORCHAV
   project when the scenario uses an ORCHAV library scene or catalog target.

## Workspace

The actor and group tree is on the left, the embedded pygfx viewport is in the
center, and the selected subject's inspector is on the right. The lower drawer contains
the timeline, validation problems, exact YAML preview, and generation log.
The scene selector discovers built-ins from the installed Sionna package and
included XML files from `libraries/scenes`. Local XML remains available through
the browse action.

Use **Add TX**, **Add RX**, or **Add Target**, then click a surface or the visible
horizontal work plane to place the actor. Select its mobility type in the
inspector, adjust the numeric spin boxes by typing or with their arrow controls,
then choose **Apply Mobility** or **Apply Orientation**. Inspector changes
remain a protected draft until they are applied. The viewport shows the
candidate motion as a ghost without changing the document, undo history, or
generated YAML. The pending banner can apply all drafts together or reset the
inspector from the applied actor or group. Selecting another subject, saving,
generating, switching workspaces, or closing prompts before a pending draft can
be lost.

Changing mobility type seeds the new model from the actor's position at the
visible timeline step. For example, converting an actor at `(5, 5, 5)` to
circular motion makes `(5, 5, 5)` the first point on the circle rather than
resetting the actor to the origin.

Newly selected models use previewable shapes around that position. Waypoint
starts with a three-point right-angle path, Grid Scan starts with a `3 x 3 x 2`
volume spanning 2 m vertically, and Random Waypoint uses compact horizontal
bounds and enough speed to show multiple seeded turns on the default timeline.

Choose **Edit Motion** to expose controls for the selected model. Stationary
motion has a position handle, linear motion has start and end handles, waypoint
motion has one handle per point, and circular motion has center and start/radius
handles. Every self-contained spatial model can also be translated by dragging
its path. The persistent XYZ gizmo translates the complete motion of a TX, RX,
or target. For a group member, it changes the actor's local group offset through
the prepared group frame. It also exposes rotation rings when the actor has a
fixed orientation. Resource-backed paths are edited through their source data,
not displaced copies of that data. Time-varying and derived orientations are
edited in the inspector, so the gizmo does not imply an unsupported
whole-timeline rotation operation.

For waypoint motion, **Draw Waypoints** starts at the current actor position
and accepts one click per additional point. Press **Enter** or double-click to
finish, **Backspace** to remove the pending point, or **Escape** to cancel. This
is a model-specific drawing shortcut, not a separate mobility mode or a limit
on the number of waypoints. Drawing replaces only the waypoint coordinates.
The selected Linear or Catmull-Rom interpolation and traversal settings remain
in effect when the path is finished and applied.

Selection is shared by the tree and viewport. Movement handles, start/end
markers, direction markers, labels, and orientation overlays remain associated
with the same actor or group after it is renamed. The viewport's **Axes** and
**Look-at rays** choices independently show those overlays for no actors, the
selected actor, or all actors. A selected prepared path has a wider halo,
sparse arrows show its direction without covering the route, and group-member
tethers expose the local formation offsets from the prepared group frame.

The inspector shows only fields relevant to the selected mobility type:

- **Point and path models**: stationary position, linear endpoints, arbitrary
  waypoint lists, circular center and radius, and survey or grid-scan extents.
- **Parametric models**: oscillation axis/amplitude/frequency, pendulum pivot
  and plane, figure-eight plane and size, and spiral radius/altitudes/turns.
- **Seeded models**: random-sampling bounds and optional exact first sample,
  Gauss-Markov parameters, random waypoint bounds/speeds/pauses, and
  Manhattan-grid geometry and turn policy.
- **Resource and relationship models**: cached network graph and route
  parameters, target-only position sequences, or a group and local
  right/forward/up offset.

Less frequently changed parameters are under **Advanced**. Path models default
to fitting the full path to the document duration. Expanding traversal options
allows a positive physical speed plus hold, loop, or ping-pong behavior.
Defaults are omitted from the canonical YAML when the shared schema already
defines them.

Random Sampling creates independent observations rather than continuous
motion. Uniform sampling allows observations to cluster anywhere in the
configured bounds. Poisson-disk sampling rejects observations closer than its
minimum-spacing value. A very small spacing can therefore look the same as
Uniform. The Builder exposes that required control and starts new Poisson-disk
drafts at 1 m.

Gauss-Markov creates continuous, correlated XY motion. Memory `alpha` controls
how much speed and heading persist: `0` follows the preferred values with fresh
noise, while values near `1` retain the previous motion. Preferred heading is
measured from +X toward +Y, the two deviation fields are the random speed and
heading standard deviations, bounds clamp the position, and the seed makes the
trajectory reproducible.

The Orientation inspector follows the same adaptive pattern:

- **Fixed**: static yaw, pitch, and roll.
- **Keyframes**: at least two explicitly timed yaw/pitch/roll samples.
- **Align with motion**: prepared motion direction with optional smoothing,
  rate limits, pitch policy, and offsets.
- **Look at**: another actor or an explicit point, with optional smoothing,
  rate limits, angular limits, and offsets.
- **Spin**: yaw, pitch, or roll axis, angular rate, and starting angles.
- **Random**: seeded yaw/pitch/roll ranges and an optional update interval.

Numeric mobility and orientation fields accept typed values and also provide
adaptive up/down arrow controls. Their increment follows the displayed value's
magnitude instead of imposing a bounded slider range.

Timeline controls also use an explicit draft. Changing Steps, Duration,
Quality, or path metrics leaves the applied document and YAML Preview unchanged
until **Apply Timeline** is selected. The draft survives actor selection and
other document refreshes. Save, generation, undo, document replacement, and
workspace switching require the draft to be applied or reset.

The authored controls and the generated timeline samples are deliberately
shown as different geometry. A waypoint model may contain any number of control
points. **Steps** determines how many positions and orientations the Generator
prepares for preview and output. When waypoint segment boundaries do not land
exactly on the selected frame times, the Timeline page says so and suggests a
denser step count. The viewport shows the Generator's prepared values rather
than inserting extra positions. Random Sampling is drawn as independent
observations rather than a connected physical path, including while its
inspector draft or whole-model translation is being previewed.

Average speed is computed from the prepared path and timeline and remains
read-only. Constant-speed traversal is an explicit advanced path setting rather
than a second editable speed value.

Use **Form Group...** to select at least two actors and optionally reuse one
actor's current motion as the group path. The builder projects every initial
position into the Generator's prepared right/forward/up group frame, then
replaces the selected actor mobilities in the same undoable command. A group is
the shared motion and local coordinate frame, not another simulated actor. Its
tree/viewport entry is an editing guide. At least two actors are required
because a one-member group has no shared behavior and can use the same mobility
directly. Groups cannot contain other groups. Every member has a deterministic
right/forward/up formation offset from the shared path. The separate optional
group jitter setting adds seeded random displacement at each frame. Leave it
disabled for a rigid formation such as a drone swarm.

Scenario copying collects only root-relative inputs: local XML and referenced
meshes/textures/includes, network GraphML, mesh-position sequences, and
scenario-local file/directory targets and metadata. Sionna and ORCHAV library
scenes and catalog targets are reused in place. A scenario that uses an ORCHAV
library scene or catalog target must therefore be saved inside the active
ORCHAV project root. To save elsewhere, first select a Sionna or local scene and
use scenario-local file or directory targets. The copy omits generated frames,
summaries, coverage results, caches, and unrelated scenario files. YAML and
dependencies are copied together. Absolute external references remain absolute
with a portability warning. Missing, traversing, escaped, or colliding
dependencies block Save and Generate at their exact field path.

Every seed field accepts the same portable non-negative signed 32-bit range.
The seed is serialized exactly, so reopening the scenario reproduces the same
random mobility, orientation, or group deviation.

New documents start with the ORCHAV library's `empty/empty.xml` scene,
ultra-low quality, 30 steps, a 3-second duration, and path metrics enabled. A
static-only scenario may use a zero duration. A moving scenario requires at
least two steps and a positive duration.

## Open, Copy, and Save

A scenario saved by the Builder carries this document marker:

```yaml
visualizer:
  scenario_builder:
    document_version: 2
```

When `document_version` is higher than the supported value, the Builder opens
the scenario read-only.

A canonical `scenario.yaml` accepted by the shared schema opens for editing in
place. Opening a scenario never writes to it. Invalid YAML and unsupported or
newer Builder document versions open read-only. A field the Builder cannot edit
locks only that part of the workspace and reports the reason in Problems.

**Copy Current Scenario and Edit** is available while a canonical scenario is
open for visualization. The source is therefore unambiguous. The action asks
only for a destination directory. It validates and writes the copied
`scenario.yaml`, copies the local scene and motion inputs required by that
YAML, then opens the copy in the Builder. The source remains unchanged, and
returning to visualization targets the copy.

Generated `frames/`, `summary/`, `coverage/`, diagnostics, caches, scripts,
README files, and unrelated files are not copied. Generate the copied scenario
when its derived outputs are needed. Shared library assets and absolute
external inputs remain references rather than being duplicated. **Save As**
serves the related but different case of branching the current in-memory
Builder draft, and it never replaces an existing unrelated `scenario.yaml`.

**Copy Current Scenario and Edit**, **Save**, and **Save As** refuse an external
destination while the document uses an ORCHAV library scene or catalog target.
The message identifies the affected scene or target and the active project root.

Actor, group, scene, and timeline edits preserve recognized fields outside the
Builder's editable subset, such as transmitter `power_dbm`, `generator_summary`,
`data`, coverage, and advanced ray-tracing settings. Reserved extension
metadata remains in the document, but the Builder does not expose or execute
those extensions.

**Save** and **Save As** always target
`<scenario-directory>/scenario.yaml`. They do not overwrite another YAML file.

## Validate and Save

Apply pending inspector or timeline edits before choosing **Validate**. The
Builder checks the shared schema, actor and group references, mobility and
orientation settings, and required scene, target, network, and sequence files.
Problems identify the affected field and actor when applicable. **Save** runs
the same validation and refuses an invalid draft. **Generate** also applies its
generation-only checks. Saving replaces `scenario.yaml` atomically so an
interrupted write does not truncate the previous file.

## Generate and Preview

**Generate** launches the saved document revision, not an unsaved in-memory
edit. The builder keeps the draft and undo history available while generation
runs and records the revision associated with the result. If you continue
editing, the result is marked as out of date relative to the current draft.

Generation writes the normal `frames/` and configured `summary/` outputs under
the saved scenario directory. After a failure or cancellation, the previous
complete output remains in place rather than being replaced by partial output.

When summary generation is requested, an unchanged normalized YAML hash can
reuse the existing `summary/` directory. ORCHAV logs a warning because library
scene meshes, textures, target catalogs, and other external inputs are not
part of that hash. Set `generator_summary.force: true` after changing such an
external input.

Builder generation requires the `files` data mode, HDF5 frame format, and the canonical
scenario-local `frames/` destination. It also requires ray tracing or persisted
coverage data. A document using another data mode, format, a noncanonical
`data.files.directory`, or a summary-only workflow disables only **Generate**. The
scenario remains open, editable, and saveable, and Problems provides the
corresponding `orchav-generator` CLI guidance.

Use **Preview Generated Result** to switch to normal playback explicitly. The
authoring document, selection, and undo history are retained so returning to
the builder restores the intact draft.

---

Up: [Generator](README.md) | Related: [Scenario Authoring](scenario_authoring.md) | [Mobility and Orientation](mobility_and_orientation.md)
