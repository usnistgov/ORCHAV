# Renderers

The ORCHAV Visualizer can display the same frames with two renderer backends:

- `pygfx`, built on pygfx and wgpu-py
- `open3d`, built on Open3D and its Filament renderer

Use **pygfx** for new work. It is the default, appears inside the main
Visualizer window, and receives ORCHAV's renderer development.

Development of **ORCHAV's Open3D/Filament renderer has stopped**. It remains
available on Windows and Linux for workflows that need its native render
window or Open3D-specific lighting and material controls. ORCHAV Sensing's 3D
overlays have not been integrated into this renderer; use pygfx to inspect
sensing results alongside the scene.

The Open3D/Filament Visualizer renderer is unavailable on macOS; an
explicit request is rejected before Qt starts. On Apple Silicon, pygfx/wgpu
renders through Metal, Apple's native graphics API.

Changing the renderer changes how a frame is drawn. It does not regenerate the
scenario or modify its frame data. Both backends receive the same frame through
the Shared Data Layer and participate in the same Visualizer
[consumer](../reference/glossary.md#consumer) workflow.

## Choose A Renderer

| Use case | Recommended choice | What to expect |
|----------|--------------------|----------------|
| First launch and day-to-day scenario inspection | `pygfx` | The renderer is embedded beside the Visualizer controls. |
| ORCHAV Sensing visualization | `pygfx` | Sensing overlays, target-response cues, and tracking uncertainty are integrated with the scene. |
| Notebook visualization | `pygfx` | It provides the recommended in-cell and offscreen Jupyter interfaces. |
| Open3D/Filament material or renderer inspection on Windows or Linux | `open3d` | The scene opens in a separate native render window. |
| Standalone headless image generation | `pygfx` | The notebook and command-line image interfaces use pygfx and wgpu. |

Both backends support the main scene, actor, target, propagation-path,
trajectory, camera, material, screenshot, and video workflows. The most useful
differences for ordinary use are:

- pygfx provides the embedded desktop viewport, notebook integration, the
  in-view coverage legend, and the complete selected-path visual cues.
- Open3D provides its native Filament window and Open3D-specific lighting and
  material controls.

The main Visualizer window is used with either backend. With Open3D,
renderer-specific controls act on the separate native window.

pygfx keeps labels at screen size. Open3D label size depends on view distance;
adjust **Label font** under **Font and world offset** for the current view.

The normal install retains the Open3D package on macOS because ORCHAV uses its
geometry utilities outside the renderer.

On Windows, the package requirements exclude Open3D 0.20.0 because its native
window can crash when closing. Open3D 0.19.0 supports the separate-window
workflow. Use pygfx for the main desktop workflow.

<a id="materials-textures-and-lighting"></a>

## Appearance controls

Both renderers support visual materials, textures, lighting and shadows.
These settings change the display, not the radio materials or propagation
computed by [Sionna RT](https://nvlabs.github.io/sionna/rt/).

Use the [appearance guide](appearance.md) for presets, texture switches,
lighting, skies and shadow detail in pygfx. Open3D exposes its own material
and lighting controls in the separate native window. Its texture switch can
retain already displayed maps after **Off**. Relaunch with `--disable-textures`
when a scalar-only Open3D view is required.

## Graphics Environment

pygfx uses wgpu. Open3D desktop rendering uses its native graphics window.
Test the chosen desktop backend in the graphics environment where it will run.
On Linux VNC, Open3D may use Mesa llvmpipe. Use the graphics adapter reported by the active backend when diagnosing performance. Prefer pygfx for GPU-backed VNC or
performance work and record the selected graphics adapter.
The standalone [Headless Rendering](headless_rendering.md) interface uses
pygfx. See that page and
[Troubleshooting](../help/troubleshooting.md) for those workflows.

## Try Both Renderers On Windows Or Linux

After generating the [Hello
World](../../scenarios/getting_started/hello_world/README.md) frame set, open
the same scenario with each backend:

```bash
orchav-visualizer --renderer pygfx --scenario scenarios/getting_started/hello_world
```

Open the same recording with Open3D:

```bash
orchav-visualizer --renderer open3d --scenario scenarios/getting_started/hello_world
```

Omit `--renderer` to use the default pygfx desktop renderer.

On macOS, use only the first command. An explicit `--renderer open3d` request
exits with an explanation before Qt or the renderer starts.

---

Up: [Visualizer](README.md) | Related: [Your First Visualizer Session](first_session.md) | [Notebook Visualization](notebooks.md) | [CLI Reference](cli_reference.md) | [Headless Rendering](headless_rendering.md)
