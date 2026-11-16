# Paris Opéra Presentation and Asset Workflows

## Optional Presentation

With the recording open, use **Demo → Presentation → Start demo** for the
camera tour. If Demo is hidden, enable it through **Tabs**. **Sensing story**
adds explanations of the saved detections. Drag or scroll to explore, choose
**Resume camera** to resume the tour, and use **Stop demo** or **Esc** to finish.

For the authored textures, add `--enable-textures` to the Visualizer command.
**Rendering → Lighting and environment → Scenario blue hour** enables the dusk
look; **Demo → Advanced options → Scenario lighting** also offers a sunset
cycle. These change appearance only. See
[presentation controls](../../../docs/visualizer/presentation_tools.md) for
exhibition layouts, surface styles, scene guides and camera controls.

<details>
<summary>Export a portable presentation or movie</summary>

From a Linux desktop with an active graphical session and an NVIDIA Vulkan
adapter, capture the completed recording from the repository root. The helper
uses the current display and Qt environment; it does not create a display.

```bash
python scenarios/showcases/paris_opera_multitarget/capture.py --scenario scenarios/showcases/paris_opera_multitarget/runs/my_run --output scenarios/showcases/paris_opera_multitarget/captures --width 1920 --height 1080 --gpu 0 --movie-frames --movie-step 32
```

Build the portable presentation from those captures, using your FFmpeg executable:

```bash
python scenarios/showcases/paris_opera_multitarget/export_demo.py \
  scenarios/showcases/paris_opera_multitarget/runs/my_run \
  --captures scenarios/showcases/paris_opera_multitarget/captures \
  --output scenarios/showcases/paris_opera_multitarget/demo \
  --encoder /path/to/ffmpeg
```

The capture helper exports native Visualizer images. `--gpu` selects a
zero-based index among detected NVIDIA Vulkan adapters (default: `0`).
Configure `DISPLAY` and `QT_QPA_PLATFORM` for the actual session when needed,
rather than reusing values from another machine. This optional workflow is not
required to run the Generator or use the Visualizer.

After export, open `demo/index.html` locally to play or scrub the recording,
inspect synchronized range–Doppler and angular-scan plots, and view actor
close-ups. `demo/scene.mp4` contains the native scene movie.

Copy the complete generated `demo/` directory to another computer to view the HTML
presentation. It contains the movie, plots, stills, metrics and attribution;
it does not need the HDF5 recording or the Python environment for playback.
The native scene and acquisition need the source assets and recorded frames.
Generated runs, captures and presentation files are ignored by Git.

Capture 320 presentation frames using Node.js 22 or later and Chrome
at `/usr/bin/google-chrome`:

```bash
node scenarios/showcases/paris_opera_multitarget/scripts/capture_film.mjs \
  scenarios/showcases/paris_opera_multitarget/demo \
  scenarios/showcases/paris_opera_multitarget/captures/presentation 320
```

Encode those frames as a 25-frame/s movie with FFmpeg:

```bash
/path/to/ffmpeg -y -framerate 25 \
  -i scenarios/showcases/paris_opera_multitarget/captures/presentation/%04d.png \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -movflags +faststart \
  scenarios/showcases/paris_opera_multitarget/demo/paris_opera.mp4
```

</details>

## Assets And Provenance

See [ATTRIBUTION.md](ATTRIBUTION.md) for the geographic, vehicle, drone,
walking-body and appearance sources. `scripts/clean_osm.py` reproduces the
surface-scene filtering from the cached source GeoJSON with the optional OSM
dependencies. `scripts/paving.py` binds the local limestone texture to existing
ground triangles. Check building clearance and radar-to-centre visibility along the prepared
routes with:

```bash
python scenarios/showcases/paris_opera_multitarget/scripts/audit_routes.py \
  --scenario scenarios/showcases/paris_opera_multitarget/runs/my_run
```

The audit writes `evidence/route_audit.json`, `evidence/routes.png`, and
`evidence/routes.pdf` within the prepared run. It checks static building
clearance and radar-to-centre visibility, not dynamic inter-target occlusion.
Other files under `scripts/` support scene preparation, appearance and camera
workflows; they are not additional steps for generating the recording.

Return to the [walkthrough](README.md).
