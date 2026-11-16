# Image-based lighting assets

The Visualizer uses an environment image to light scene surfaces and show reflections. This directory includes a generated neutral outdoor sky for offline use.

| Renderer | Bundled environment |
| --- | --- |
| pygfx | `neutral_outdoor.hdr`, a Radiance panorama converted to a cubemap at runtime. |
| Open3D | `neutral_outdoor_ibl.ktx` and `neutral_outdoor_skybox.ktx`, a matched Filament pair. |

For a different sky in pygfx, choose a Poly Haven preset and click **Download & apply**, or use **Load HDR…** for a local panorama. For Open3D, use **Load KTX…** with a matched pair. If you redistribute an environment, include its source, license, and required attribution.

See [Sky environments](../../docs/visualizer/appearance.md#lighting-and-skies) for the available presets, cache locations, and offline use. The [renderer guide](../../docs/visualizer/renderers.md) explains platform support.
