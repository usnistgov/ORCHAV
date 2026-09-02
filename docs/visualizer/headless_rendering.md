# Headless Rendering

Headless rendering turns generated frames into images without opening the
desktop Visualizer. Use it for scripts, reports, CI checks, and batch output.
It is a [consumer](../reference/glossary.md#consumer) workflow that reads an
existing local frame set through the HDF5
[frame provider](../reference/glossary.md#frame-provider). Run the
[Generator](../generator/README.md) first when the scenario does not yet have a
`frames/` directory.

This page covers the command-line and Python image interfaces. For inline
Jupyter views and animation, use [Notebook Visualization](notebooks.md).

## Render One Frame

From the repository root:

```bash
python -m visualizer.headless scenarios/getting_started/hello_world \
  --frame 0 -o tmp/orchav-frame.png
```

The command opens the scenario through the Shared Data Layer, renders frame 0
with pygfx, writes the PNG, and exits. Standalone headless rendering uses pygfx.
The desktop Visualizer offers both pygfx and Open3D/Filament on Windows and
Linux. macOS desktop, notebook, and headless visualization use pygfx.

## Render From Python

`render_scenario()` returns an `(H, W, 3)` `uint8` NumPy array. It can also
write the image to disk:

```python
from visualizer.headless import render_scenario

image = render_scenario(
    "scenarios/visualizer/notebook_mode",
    frame=1,
    azimuth=45,
    elevation=30,
)

render_scenario(
    "scenarios/visualizer/notebook_mode",
    frame=1,
    output="tmp/orchav-headless/frame1.png",
)
```

The Python API accepts the same frame, camera, coloring, visibility, lighting,
and image-size choices as the command-line interface.

## Select Frames, Camera, And Appearance

```bash
python -m visualizer.headless scenarios/visualizer/notebook_mode \
  --frame 1 \
  --azimuth 45 --elevation 30 --distance 100 \
  --color-mode path_loss \
  -o tmp/orchav-headless/frame1.png

python -m visualizer.headless scenarios/visualizer/notebook_mode \
  --frames 0,1,2 \
  -o tmp/orchav-headless/frames/
```

Common option groups are:

| Area | Options |
|------|---------|
| Frame selection | `--frame`, `--frames` |
| Output | `--output`, `--base64` |
| Camera | `--azimuth`, `--elevation`, `--distance`, `--center X Y Z` |
| Image | `--width`, `--height`, `--color-mode` |
| MPC display | `--disable-mpc-layer`, `--hide-mpc-paths`, `--hide-mpc-bounce-points`, `--line-width`, `--point-size` |
| Lighting | `--ibl-name`, `--ibl-intensity` |
| Textures | `--enable-textures`, `--disable-textures` |

`--base64` writes a base64-encoded PNG to standard output. Run
`python -m visualizer.headless --help` for the exact current options.

## Graphics Environment

The standalone interface uses pygfx, wgpu, and rendercanvas. It depends on the
target system's graphics driver, so test one representative render in the
environment where automation will run.

See [Renderers](renderers.md) for the desktop renderer capabilities and
[Troubleshooting](../help/troubleshooting.md) for display-server and remote
desktop workflows.

## Related Workflows

- Use `orchav-visualizer --render-frames DIR` when batch rendering belongs to a
  desktop Visualizer launch. See the [CLI Reference](cli_reference.md#batch-rendering).
- Use [Notebook Visualization](notebooks.md) for inline images, interactive
  views, animation, or notebook-driven regeneration.
- Use the [Notebook Mode scenario](../../scenarios/visualizer/notebook_mode/README.md)
  for a small three-frame example.

---

Up: [Visualizer](README.md) | Related: [Renderers](renderers.md) | [Notebook Visualization](notebooks.md)
