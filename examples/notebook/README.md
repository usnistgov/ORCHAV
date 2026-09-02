# Notebook Examples

These cells show the shortest notebook workflows. They assume a source
checkout with the optional notebook renderer dependencies installed:

```bash
python -m pip install -e ".[jupyter]"
orchav-generator scenarios/visualizer/notebook_mode/
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

Open the full notebook at
[`notebook_mode.ipynb`](notebook_mode.ipynb), or use the cells below in another
notebook. `--no-browser` is recommended on remote or VNC-backed machines. Copy
the URL printed by Jupyter into your browser.

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

viz = PygfxNotebookViz("scenarios/visualizer/notebook_mode")
viz.show(frame=0, width=900, height=650)
```

## Generate and Visualize

For quick stationary position studies, regenerate from the notebook and reload
the generated frame set automatically:

```python
from visualizer.src.notebook.pygfx import PygfxNotebookViz

viz = PygfxNotebookViz("scenarios/visualizer/notebook_mode")
viz.regenerate(
    tx_positions=[[0.0, 0.0, 25.0]],
    rx_positions=[[70.0, -25.0, 1.5]],
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
