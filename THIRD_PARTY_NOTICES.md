# Third-Party Notices

Summary of third-party dependencies and third-party assets included with ORCHAV.
Python packages, user-provided external inputs, and generated outputs are not
bundled in this source distribution unless they are explicitly identified
below as bundled assets.

## Core Runtime Dependencies

| Component | Role | License |
|-----------|------|---------|
| Sionna / [Sionna RT](https://nvlabs.github.io/sionna/) | radio propagation and ray tracing | Apache-2.0 |
| Mitsuba / Dr.Jit | rendering and differentiable ray-tracing backend used by Sionna RT | BSD-3-Clause |
| NumPy | numerical arrays | BSD-3-Clause |
| ContourPy | contour geometry generation for plots | BSD-3-Clause |
| Open3D | geometry and visualization utilities | MIT |
| PySide6 / Qt for Python | desktop GUI bindings | LGPL-3.0 / GPL-3.0 / commercial |
| PyYAML | YAML parsing | MIT |
| Pydantic | configuration validation | MIT |
| toml | TOML parsing | MIT |
| h5py | HDF5 storage | BSD-3-Clause |
| imageio | image I/O | BSD-2-Clause |
| pyqtgraph | plotting | MIT |
| NetworkX | cached street-network loading and routing | BSD-3-Clause |
| SciPy | optional numerical helpers | BSD-3-Clause |
| grpcio / protobuf | optional streaming and protobuf transport | Apache-2.0 / BSD-3-Clause |
| grpcio-tools | contributor protobuf code generation | Apache-2.0 |
| pygfx / wgpu / rendercanvas | default Visualizer renderer stack | BSD-2-Clause |
| pylinalg | default renderer helper | MIT |
| JupyterLab | optional notebook environment | BSD-3-Clause |
| jupyter_rfb | optional notebook display | MIT |

Python dependency packages are installed from the user's environment and are not
bundled as binary artifacts in this source tree.

## Bundled Assets

| Asset group | Distribution status | License/provenance |
|-------------|---------------------|--------------------|
| `libraries/textures/pbr/` | Included | Poly Haven texture sets under CC0 1.0 Universal plus a project-generated NIST/CTL pack. See [`credits.md`](libraries/textures/pbr/credits.md). |
| `libraries/ibl/neutral_outdoor*` | Included | Generated neutral fallback environment |
| Curated scene and target assets | Included | Project assets or generic primitives used by included examples |
| `libraries/targets/nist_human_walking/` | Included | Curated NIST human target mesh sequence for target-scattering examples |

Other configurations may reference user-provided scene meshes, textures,
target catalogs, scripts, or source data. Unless an input is named under
[Bundled Assets](#bundled-assets), it is outside the ORCHAV source distribution;
users are responsible for its provenance and applicable terms.
