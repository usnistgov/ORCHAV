# Your First Visualizer Session

This tour starts with the Hello World frame set generated in the repository
Quickstart. It introduces the viewport, primary controls, current-frame
metrics, and screenshot export.

If `scenarios/getting_started/hello_world/frames/` does not exist yet, complete
the [Quickstart](../getting_started/quickstart.md) before continuing.

## 1. Open Hello World

If Hello World is still open from the Quickstart, continue to step 2. Otherwise,
open it from the repository root:

```bash
orchav-visualizer --scenario scenarios/getting_started/hello_world
```

The default `pygfx` renderer places the 3D viewport beside the controls in one
window. The first frame shows the scene, one transmitter, one receiver, and the
available propagation paths.

## 2. Orient Yourself

The persistent **Context** controls choose the active transmitter and receiver
and remain visible when you change tabs. The timeline changes the displayed
frame. The tabs group related scene, path, analysis, rendering, and export
controls.

For this one-frame scenario:

1. Confirm the TX and RX markers in the 3D view.
2. Orbit, pan, and zoom the camera.
3. Choose **Top**, **Side**, or **Front**, then return with **Iso**.
4. Leave the Context selection on the only TX/RX pair.

Hello World has one frame, so its timeline does not animate. The later
[`metrics_evolution`](../../scenarios/visualizer/metrics_evolution/README.md)
scenario introduces playback over time.

## 3. Inspect MPCs

Open **Paths** and try these controls one at a time:

1. Under **Color By**, choose **Order**, **Type**, **Delay**, **Loss**, or
   **Material**.
2. Hide one available interaction order or type, then restore it.
3. Change **Path line width**, or toggle **Bounce Points** and **Interaction
   Markers**.
4. Choose **Open MPC Explorer...**, then select one path to inspect its geometry,
   interactions, material sequence, delay, and path loss. See [MPC
   Explorer](mpc_explorer.md) for its complete workflow.

The filters and display controls change only the current view. They do not
rewrite the generated HDF5 frame set.

## 4. Compare Analysis Scopes

Press `M` to open **Metrics**. It describes the currently displayed frame and
follows the active TX/RX and path-filter selection.

The **Analysis** tab's **Statistics** and **Graphs** sections instead summarize
the complete available frame sequence. With Hello World that sequence contains
only one frame. The difference becomes visible in the
[`statistics`](../../scenarios/visualizer/statistics/README.md) scenario.

See [Visual Analysis](analysis.md) for Metrics, Statistics, coverage, camera
inspection, and RF X-Ray.

## 5. Export An Image

Choose **Screenshot** in the toolbar, or open **Capture & Export** > **Export**
and choose **Screenshot**. The saved image records the current camera,
coloring, and visible layers. It does not modify the scenario or frame data.

## Where To Go Next

Follow the [Visualizer scenario core path](../../scenarios/visualizer/README.md#core-learning-path):

1. [MPC Inspection](../../scenarios/visualizer/mpc_inspection/README.md)
2. [Metrics Evolution](../../scenarios/visualizer/metrics_evolution/README.md)
3. [Statistics](../../scenarios/visualizer/statistics/README.md)
4. [Multi-Device Trajectory](../../scenarios/visualizer/multi_device_trajectory/README.md)

Choose an optional guide only when the task needs it:

- [Interactive Recomputation](interactive_recomputation.md) for temporary
  actor or solver changes.
- [Shared Data Layer](../shared/README.md) for local, live, and
  remote frame delivery.
- [Renderers](renderers.md) for the Open3D/Filament backend.
- [Headless Rendering](headless_rendering.md) for command-line or Python image
  output, and [Notebook Visualization](notebooks.md) for Jupyter workflows
  outside the desktop application.

---

Up: [Visualizer](README.md) | Continue: [MPC Inspection Scenario](../../scenarios/visualizer/mpc_inspection/README.md)
