# Extra views and Demo mode

Actor views, live figures and Demo mode arrange an existing recording for
inspection or presentation. Actor views and Demo mode use pygfx. Live figures
reuse plots from the channel and sensing dashboards.

The [ORCHAV demo videos](demo_video.md) show camera tours, appearance changes
and saved sensing results. The controls below let you compose a presentation
from your own recording.

## Arrange views and plots

Actor views, live figures, [Path Explorer](path_inspection.md) and
[Path Analysis](path_inspection.md) share docking controls. Drag a
view's title bar to float it or dock it at the top, right or bottom of the main
window. Its **Dock** menu also offers **Tab with** and **Place beside**.

### Actor views

Choose **Actor view +** in the camera controls to add a view. Select a target,
TX or RX and choose **Fit**. Drag to orbit and use the wheel to zoom. The view
follows that actor during playback while the main camera remains independent.
Add another view to follow another actor.

**Refresh → Main priority** favors the main viewport. **Smooth actors** updates
auxiliary views more often at additional rendering cost. Close views you no
longer need. Each uses the same scene and current frame, so it does not load a
second recording. Actor views close when the scene changes. Screenshot and
video export use the main viewport.

### Live figures

Choose **Live figure +** to open up to four individual plots. Available choices
include channel delay and angle distributions, power-delay profiles and saved
sensing range, velocity, angle and tracking plots. Sensing choices depend on
which products were recorded.

A dock borrows its plot from the original dashboard and preserves its zoom and
selection. Closing the dock returns the plot. Visible figures update even when
their dashboard tab is closed. Hidden figures stop updating, and all figure
docks close when the recording changes.

## Demo mode

Open a generated recording. To use its authored view with textures enabled,
replace the path below with the recording directory:

```bash
python -m visualizer --scenario path/to/recorded_run \
  --enable-textures --no-resume
```

`--no-resume` skips an autosaved workspace. Omit it to resume the saved view.
Frame the scene, open **Demo** through **Tabs** if hidden, choose a preset and
select **Start demo**. Stop the demo before selecting another preset.

| Preset | Behavior |
|---|---|
| Presentation | Cycles scene looks and coordinated MPC styles, with an authored camera tour or slow orbit when available. |
| Explore | Preserves appearance, loops recorded playback and slowly orbits. |
| MPC story | Holds a frame for appearance changes and slowed MPC pulses, then plays the recording once. Requires visible paths and at least two frames. |
| Sensing story | Replays saved results, periodically pausing to illustrate a detection's available range, angle and position information. Requires loaded local sensing metadata. |
| Sensing presentation | Loops saved sensing results in Detailed appearance, with a range–velocity map and range profile stacked beside the scene. Requires saved range–Doppler maps. |
| Follow targets | Aims at each recorded target in turn from the current camera position. |
| Custom | Uses your selected advanced options. |

Only supported choices appear. For example, a scenario camera tour requires
`view_defaults.demo_tour`, and sensing reveals cannot supply detections or
products missing from the recording.

**Hide interface during demo** hides controls and docks around the embedded
viewport. The footer keeps **Stop demo** available. **Show explanatory captions**
identifies the current phase and path coloring below the scene. Viewport-only
image and video exports do not include this footer.

Drag or scroll to take control of the camera while Demo continues. Select
**Resume camera** to restart automatic motion. In Sensing story, this also pauses
automatic detection reveals. Press **Esc** or choose **Stop demo** to finish.
The demo restores temporary appearance, sky, playback and layout settings.
Manual camera framing is preserved.

For a small screen, choose **Sensing presentation**. The range–velocity map shows
saved detections and ground truth when available. The range profile shows its
full power span and detection markers. A compact footer provides pause, seek and stop
controls. You can rotate the scene, zoom a graph or choose another live figure
while playback continues. Stopping restores your previous figures and layout.

### Advanced options

Use these controls to customize a preset:

| Option | Purpose |
|---|---|
| Preset tour + MPC pulses, then play once | Traverse up to 50 visible paths on a held frame for each selected look, then replay the recording once. |
| Replay with periodic detection reveals | Pause at saved sensing results and display the available geometric support. **Reveal every** sets the interval. |
| Slow camera orbit | Orbit the current look-at point in Overview mode. |
| Cycle targets from current camera position | Aim at successive targets using the camera's smooth-transition settings. **Follow each target for** sets the interval. |
| Scenario camera tour | Play the scenario's authored views and shot timings. Replaces the orbit. |
| Loop recorded playback | Use the current playback speed and selected frames, including sensing-only playback. |
| Cycle appearance presets | Visit the checked looks, temporarily pairing them with MPC styles, then return to the initial appearance. |
| Scenario lighting | Use an authored Blue hour look or a repeating Sunset cycle. Appears only when a lighting recipe is present. |
| Cycle sky / IBL environments | Visit checked installed skies with matching lighting. Downloads are never started by Demo. |
| Cycle current, soft and side lighting | Compare lighting while appearance and sky cycling are off. |
| Exhibition layout | Place views beside the embedded scene and temporarily hide other controls. **Exhibition views** selects an actor view with a live figure, or the sensing map with its range profile. |

**Change look every** sets the hold time from 1 to 120 seconds. Target holds and
sunset cycles have their own durations. Appearance preparation adds to the hold
time, so it is not a frame-rate setting. Texture-locked sessions cannot use
Detailed. Finish actor editing or camera export before starting Demo.

Use [camera navigation](camera.md) to author and export camera sequences, and
[scenario defaults](scenario_defaults.md#authored-demo-tours) to include a tour
with a scenario. Demo illustrates saved data and does not run new acquisitions.

## Authored scenario lighting

**Rendering → Lighting and environment → Scenario blue hour** applies a
scenario's optional dusk recipe. **Demo → Scenario lighting** can cycle between
its initial daylight and dusk looks. A scene without the recipe leaves these
choices unavailable.

The prepared [Paris Opéra showcase](../../scenarios/showcases/paris_opera_multitarget/README.md)
uses a cool environment, warm facade lighting and moving vehicle lamps. Follow
its preparation and acquisition instructions before selecting those effects.
The lighting is an illustrative appearance, not measured site illumination.
Choosing another background, sky or appearance preset turns blue hour off.

## Rendering costs

Every visible actor view and plot adds drawing work. Use **Main priority** when
main-scene playback matters most, and close unused views. Costs depend on the
scene, window size, graphics backend and overlays. See
[sensing loading choices](sensing_dashboard.md#loading-and-playback) for preparing
saved figures and [scene appearance](appearance.md#visibility-and-scene-guides)
for grids, coordinate guides and cutaway controls.

Up: [Visualizer](README.md) | Related: [Demo videos](demo_video.md) |
[Camera navigation](camera.md) | [Scene appearance](appearance.md)
