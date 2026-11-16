# Notebook Examples

Explore stored scenario frames with the pygfx notebook viewer. Run the commands
below from the repository root with your ORCHAV environment active.

Install notebook support:

```bash
python -m pip install -e ".[jupyter]"
```

Generate the notebook recording:

```bash
orchav-generator scenarios/visualizer/notebook_mode/
```

Start Jupyter and follow its displayed connection instructions:

```bash
jupyter lab --no-browser examples/notebook/notebook_mode.ipynb
```

The Generator command creates `frames/` for the dedicated notebook-mode
scenario. The runnable scene and frame data live under
`scenarios/visualizer/notebook_mode`. This examples folder contains the
ready-to-open notebook and the copy-pastable cells below. Notebook rendering can
open an existing frame set, or it can regenerate frames directly from a cell
when you want to iterate on simple TX/RX positions.

See [Notebook Visualization](../../docs/visualizer/notebooks.md) for the full
interface, its local HDF5 scope, and the difference between rendering and
regeneration.

The [example notebook](notebook_mode.ipynb) opens a looping interactive view
with TX/RX cards, then compares appearance presets, static renders, and
callouts before session cleanup. Its
saved outputs are cleared. Run its cells in order. The snippets below use paths
relative to the checkout root. If your notebook runs in another directory,
pass the absolute path to `scenarios/visualizer/notebook_mode` to `PygfxNotebookViz`. `--no-browser` is recommended on remote or VNC-backed
machines. Copy the URL printed by Jupyter into your browser.

## Static Notebook Image

Use `PygfxNotebookViz.render()` for remote Jupyter sessions and reports. It
does not require a Qt desktop session.

```python
from visualizer.src.notebook.pygfx import PygfxNotebookViz

viz = PygfxNotebookViz("scenarios/visualizer/notebook_mode")
img = viz.render(frame=0, azimuth=45, elevation=30, display=True)
```

## Interactive Widget

Use `show()` for an in-cell orbit/pan/zoom widget. This path requires
`jupyter_rfb`, which is included in the `jupyter` extra.

```python
from visualizer.src.notebook.pygfx import PygfxNotebookViz

viz = PygfxNotebookViz(
    "scenarios/visualizer/notebook_mode", appearance="Dark", label_style="card"
)
viz.show(frame=0, width=900, height=650)
```

## Playback And Cleanup

Reuse the same viewer to loop through the 100-frame recording:

```python
animation = viz.animate(frames=viz.frames, fps=24, loop=True)
animation
```

Orbit, pan, and zoom remain available during playback. Set `loop=False` to stop
on the last frame. Select indices from `viz.frames`; unavailable indices raise
`ValueError`.

Use `appearance="Detailed"`, `"Paper"`, `"Dark"`, or `"Glow"` on the constructor
or a rendering call. Each view retains its own appearance. Use
`label_style="plain"`, `"callout"`, or `"card"` for TX/RX labels; a settings
dictionary can adjust pixel sizes and offsets. See
[Appearance And Labels](../../docs/visualizer/notebooks.md#choose-appearance-and-labels)
for color and marker options.

When finished, call `viz.close()` to stop playback, close the viewer's canvases,
and release its frame files. `viz.reload()` also closes existing views before
reopening the recording. Create a new widget or animation after reloading.

## Generate and Visualize

For quick stationary position studies, regenerate from the notebook and reload
the generated frame set automatically:

```python
from visualizer.src.notebook.pygfx import PygfxNotebookViz

viz = PygfxNotebookViz("scenarios/visualizer/notebook_mode")
viz.regenerate(
    tx_positions=[[80.0, 100.0, 30.0]],
    rx_positions=[[25.0, 80.0, 1.5]],
    quality="ultra-low",
    steps=1,
)
viz.render(frame=0, azimuth=35, elevation=25)
```

Each supplied position row creates a separate stationary device with zero fixed
orientation. The rows are not trajectory samples. Use normal
[Scenario Authoring](../../docs/generator/scenario_authoring.md) for motion,
orientation, authored targets, external preprocessing, or more complex actor
logic. For an external producer, call `viz.close()`, run the Generator,
then call `viz.reload()`. Integrated `viz.regenerate()` performs that
lifecycle automatically.

## More Detail

- [Visualizer First Session](../../docs/visualizer/first_session.md)
- [Notebook Visualization](../../docs/visualizer/notebooks.md)
- [Frame Data Reference](../../docs/shared/frame_reference.md)
- [Statistics Helpers](../../docs/shared/statistics.md)

---

Up: [Examples](../README.md)
