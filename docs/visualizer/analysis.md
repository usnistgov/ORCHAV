# Visual Analysis

The Visualizer is a [frame consumer](../reference/glossary.md#consumer). Use
**Metrics** to inspect the frame currently displayed, and use the **Analysis**
tab's **Statistics** and **Graphs** sections to summarize the complete sequence
available from the active [frame provider](../reference/glossary.md#frame-provider).
The same workspace also provides coverage and camera inspection. When the
selected data mode supports it,
[Interactive Recomputation](interactive_recomputation.md) lets you test
temporary actor or ray-tracing changes without rewriting the scenario or its
saved frames.

| Question | Tool | Data scope |
|---|---|---|
| What is happening in the displayed frame? | Metrics window | Current TX/RX and path-filter selection |
| How does the channel change across the scenario? | Statistics and Graphs | Every frame and represented TX/RX pair available from the frame provider |
| Which individual MPC produced an observation? | [MPC Explorer](mpc_explorer.md) | One selected path in the displayed frame |
| Where does a coverage metric meet a threshold? | Coverage tab and coverage graphs | Selected coverage metric and height |
| What would one temporary actor or solver change do? | [Interactive Recomputation](interactive_recomputation.md) | Current session. Scenario files are unchanged. |

Use these included scenarios for hands-on examples:

- [Metrics Evolution](../../scenarios/visualizer/metrics_evolution/README.md)
  contrasts current-frame Metrics with changes over a moving receiver route.
- [Statistics](../../scenarios/visualizer/statistics/README.md) adds
  whole-sequence graphs and a companion coverage map.
- [Multi-Device
  Trajectory](../../scenarios/visualizer/multi_device_trajectory/README.md)
  demonstrates actor selection, trajectories, Follow, and POV camera modes.
- [MPC Inspection](../../scenarios/visualizer/mpc_inspection/README.md) provides
  a compact urban frame for path selection, material inspection, and the
  advanced RF X-Ray overlays.

## Current-Frame Metrics

Open **View > Metrics Window**, or press `M`. The window follows the latest
displayed frame and redraws only its visible tab:

- **Overview** shows delay, angle, [path-loss](../concepts/propagation.md#path-metrics),
  and interaction-order distributions. It also shows the **Power Delay Profile
  (1 ns resolution)**.
- **Channel** shows a delay-power scatter plot, an AoD or AoA direction map,
  per-pair values, and compact link summaries.
- **Materials** counts material contacts by bounce depth and relates retained
  paths to their material categories.

The 1 ns power-delay profile groups selected paths into 1 ns bins, sums their
linear path gains, and places each displayed bin at its power-weighted delay.
It does not use phase and is therefore not labeled as a complex channel
impulse response. Select one concrete TX/RX pair in **Context** when you need a
per-link profile. Selecting multiple pairs pools their paths.

Material plots describe associations within retained paths. In particular,
the associated path-gain proxy is not a measurement of energy deposited at an
individual bounce.

### Scope And Drawing Limits

The TX/RX choice in **Context** and the filters in **Paths** determine which
paths Metrics analyzes. The **Limit rendered MPCs** Top-K control reduces 3D
drawing work only. It does not reduce Metrics calculations or exports.

**Max plotted markers** is a separate dashboard drawing limit for dense plots.
Choose **No limit** to draw every selected sample. Calculations and CSV export
still use every path selected by the Context and Paths filters.

**Adaptive axes** lets the Overview plots follow the displayed frame. Turn it
off to retain their current X and Y ranges while the scenario plays. Retained
ranges are the current plot ranges, not global bounds scanned from every
frame.

### Refresh Behavior

Choose a fixed update rate when you want a predictable upper refresh limit, or
choose **Maximum (adaptive)** for the fastest responsive update. Metrics keeps
only the latest pending frame and leaves a cooldown based on the preceding
calculation and drawing cost. Dense frames can therefore skip intermediate
dashboard updates instead of building an unbounded queue behind playback.

Use **Pause** to freeze automatic Metrics work and **Refresh** for one immediate
update.

## Scenario Statistics And Graphs

Open the **Analysis** tab and expand **Statistics**. The Visualizer automatically
checks for a matching cache and otherwise streams the required fields from the
active frame provider. **Compute Statistics** forces a fresh scan when one is
needed. The result can be exported as CSV, and chart figures can be exported as
PNG files.

The **Graphs** section includes:

- MPC-count, delay, path-loss, and angular distributions.
- Delay and path-loss cumulative distributions.
- Interaction-mechanism and interaction-order evolution.
- TX/RX pair path-state and channel-metric summaries.
- coverage distributions and threshold-success curves for the selected height
  when [coverage data](../concepts/propagation.md#coverage-maps) is loaded.

Every graph provides an explanation of what it counts. Hover over a plotted
item to inspect its value when the chart supports point inspection. Evolution
graphs appear only when the frame provider offers multiple frames. Graph
rendering and chart export require Matplotlib.

Statistics describes the complete sequence available from the active frame
provider. It does not apply the current **Context** or **Paths** filters. Use
Metrics for selection-scoped analysis.

### Statistics Cache

For local and remote HDF5 providers, the Visualizer can reuse derived numeric
results from a consumer-side cache. Live Generator sessions always scan the
available frames instead. See [Caching](caching.md#statistics-cache) for cache
location, identity, deletion, and refresh behavior.

## Coverage Inspection

When coverage output is present, the **Coverage** tab controls the 3D overlay:

- Choose an available metric and height.
- Adjust opacity and optional smoothing.
- Apply a threshold and optionally dim cells that fail it.
- Draw isolines for a scalar layer.
- cycle through the available heights.

For a multi-transmitter map, compatible metrics can use either the serving-TX
layer or one selected transmitter. A one-transmitter map hides this redundant
choice. The serving/specific-transmitter selection remains in place when you
change between compatible metrics.

Positive linear RF quantities use logarithmic colors where appropriate, while
tooltips, thresholds, and displayed values retain the metric's documented
physical units. Coverage graphs in **Analysis** follow the active metric and
height selection.

## Camera And Scene Inspection

The camera and rendering controls provide several non-destructive ways to
inspect a dense scene:

- **Overview** supports free orbit, pan, and zoom.
- **Follow** keeps a selected TX, RX, or target centered during playback.
- **POV** places the camera at the selected actor and looks along its forward
  direction or a selected cardinal axis.
- The **Views** controls store four temporary viewpoints for comparison.
- The pygfx **Minimap** provides a top-view orientation aid.
- AoD and AoA aperture previews show the selected angular filter around one
  concrete TX/RX pair.
- pygfx **Cutaway** planes reveal geometry inside a scene without modifying
  its XML or generated frames.

See [Renderers](renderers.md) for the capabilities that differ between pygfx
and Open3D/Filament.

## Interactive Recomputation

The **Edit** tab can ask a local worker or a running Generator to recompute a
temporary frame when the selected data mode supports it. Remote HDF5 Playback
remains read-only. Use [Interactive Recomputation](interactive_recomputation.md)
for the mode boundaries, controls, reset behavior, and persistence rules.

## RF X-Ray (Advanced, pygfx)

Expand **Analysis > RF X-Ray** while using the pygfx renderer. RF X-Ray is an
inspection overlay, not a material-appearance editor:

- **Material Map** recolors visible geometry by its assigned radio-material
  class.
- **MPC Material Usage** colors materials by their relative, path-loss-weighted
  contacts in the current visible MPC selection. **Top Paths** can additionally
  highlight a bounded number of the strongest material-bearing paths.
- **Material Properties** colors materials by a selected configured RF value,
  such as relative permittivity, conductivity, scattering coefficient, XPD
  coefficient, or thickness.

The overlay works at material or material-family granularity because current
frames do not identify every bounce by a stable scene-object ID. Treat the
results as comparative inspection aids, not as per-bounce power measurements.

---

Up: [Visualizer](README.md) | Related: [MPC Explorer](mpc_explorer.md) | [Interactive Recomputation](interactive_recomputation.md)
