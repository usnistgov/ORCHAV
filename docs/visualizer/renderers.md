# Renderers

The ORCHAV Visualizer can display the same frames with two renderer backends:

- `pygfx`, built on pygfx and wgpu-py
- `open3d`, built on Open3D and its Filament renderer

Use `pygfx` for day-to-day desktop work. It is the default, appears inside the
main Visualizer window, and provides ORCHAV's most complete interactive
workflow. On Windows and Linux, Open3D/Filament is also available, with its own
native render window and backend-specific material controls. The
Open3D/Filament Visualizer renderer is unavailable on macOS in v0.1; an
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

The normal install retains the Open3D package on macOS because ORCHAV uses its
geometry utilities outside the renderer.

## Materials, Textures, And Lighting

Both renderers support physically based material appearance, optional texture
maps, image-based lighting, and shadows. These controls affect only the visual
representation. They do not change the radio materials or propagation paths
computed by [Sionna RT](https://nvlabs.github.io/sionna/).

Texture maps are disabled by default. Enable them for one launch with:

```bash
orchav-visualizer --scenario path/to/scenario --enable-textures
```

Use `--disable-textures` to force scalar material colors when another setting
enables textures. The Render panel exposes the controls supported by the active
backend. ORCHAV includes the neutral `neutral_outdoor` environment under
[`libraries/ibl/`](../../libraries/ibl/README.md).

Large scenes can batch compatible static meshes automatically. Configure
`view_defaults.merge_scene_meshes` only when a scenario needs to force or
disable that behavior. See [Visualizer Scenario
Defaults](scenario_defaults.md#renderer-dependent-defaults).

## Graphics Environment

pygfx uses wgpu. Open3D desktop rendering uses its native graphics window.
Test the chosen desktop backend in the graphics environment where it will run.
On Linux VNC, Open3D may use Mesa llvmpipe. Run Open3D directly; Open3D through
VirtualGL was not included in v0.1 testing. Prefer pygfx for GPU-backed VNC or
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
orchav-visualizer --renderer open3d --scenario scenarios/getting_started/hello_world
```

Omit `--renderer` to use the default pygfx desktop renderer.

On macOS, use only the first command. An explicit `--renderer open3d` request
exits with an explanation before Qt or the renderer starts.

---

Up: [Visualizer](README.md) | Related: [Your First Visualizer Session](first_session.md) | [Notebook Visualization](notebooks.md) | [CLI Reference](cli_reference.md) | [Headless Rendering](headless_rendering.md)
