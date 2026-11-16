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
| PySide6 / Qt for Python | desktop GUI bindings | LGPL-3.0-only / GPL-2.0-only / GPL-3.0-only / commercial |
| packaging | dependency version checks | Apache-2.0 or BSD-2-Clause |
| PyYAML | YAML parsing | MIT |
| Pydantic | configuration validation | MIT |
| toml | TOML parsing | MIT |
| h5py | HDF5 storage | BSD-3-Clause |
| imageio | image I/O | BSD-2-Clause |
| pyqtgraph | plotting | MIT |
| NetworkX | cached street-network loading and routing | BSD-3-Clause |
| SciPy | sensing association and numerical helpers | BSD-3-Clause |
| Matplotlib | sensing diagnostic, diagram, and replay figures | Matplotlib License |
| Pillow | sensing image composition and figure/GIF export | MIT-CMU |
| scikit-learn | DBSCAN clustering for supported sensing detector paths | BSD-3-Clause |
| grpcio / protobuf | optional streaming and protobuf transport | Apache-2.0 / BSD-3-Clause |
| grpcio-tools | contributor protobuf code generation | Apache-2.0 |
| pygfx / wgpu / rendercanvas | default Visualizer renderer stack | BSD-2-Clause |
| pylinalg | default renderer helper | MIT |
| JupyterLab | optional notebook environment | BSD-3-Clause |
| jupyter_rfb | optional notebook display | MIT |
| smplx | optional SMPL conversion | [Non-commercial research license](https://github.com/vchoutas/smplx/blob/main/LICENSE) |
| trimesh / torch | optional mesh conversion and model evaluation | MIT / BSD-3-Clause |
| osmnx | optional OSM acquisition and street graphs | [MIT](https://github.com/gboeing/osmnx/blob/main/LICENSE.txt) |
| pyproj / PROJ | optional geographic coordinate transforms | [MIT](https://github.com/pyproj4/pyproj/blob/main/LICENSE); retain the installed PROJ and bundled-component notices |
| shapely / GEOS | optional geographic geometry processing | [Shapely BSD-3-Clause](https://github.com/shapely/shapely/blob/main/LICENSE.txt); bundled GEOS has separate LGPL-2.1 terms in the installed `LICENSE_GEOS` |
| boto3 | optional external terrain/data access | [Apache-2.0](https://github.com/boto/boto3/blob/develop/LICENSE) |
| triangle | optional polygon triangulation | Python wrapper LGPL-3.0; embedded Shewchuk Triangle has separate custom redistribution terms, detailed below |

Python dependencies are installed separately. Their license files also cover
components bundled inside their wheels. ORCHAV does not redistribute those
wheels or an installed environment.

### Triangle Wrapper And Embedded Library

The optional `triangle` 20250106 package combines an
[LGPL-3.0 Python wrapper](https://github.com/drufat/triangle/blob/v20250106/LICENSE)
with Shewchuk's Triangle 1.6 under
[separate terms](https://github.com/drufat/triangle-c/blob/8b9e1046e5cddab1298d3204f10c93665836cf99/README).
Private, research, and institutional use is free. Redistribution requires
retained notices and no charge; modified versions have additional conditions.
Including Triangle in a commercial system requires an arrangement with its author.

ORCHAV's source release does not bundle Triangle: users install it separately.
The Triangle notice explicitly permits directing users to obtain it for free
without that arrangement. If you distribute an environment, installer, or
container containing Triangle, follow its redistribution terms. ORCHAV's code
license does not replace them.

## Bundled Assets

| Asset group | Distribution status | License/provenance |
|-------------|---------------------|--------------------|
| `libraries/textures/pbr/` | Included | Six Poly Haven texture sets under CC0 1.0 Universal, plus ORCHAV's NIST/CTL and water-ripple textures under the project license. See [`credits.md`](libraries/textures/pbr/credits.md). |
| `libraries/ibl/neutral_outdoor*` | Included | Generated neutral fallback environment |
| Curated scene and target assets | Included | Project-authored primitives and the [procedural car-shaped target](libraries/targets/procedural_car/README.md), under the project license |
| `visualizer/resources/` | Included | ORCHAV branding and interface icons under the project license |
| `libraries/targets/digital_twin_update1/` | Included | 38 walking poses provided by NIST through [NextG submission 156](https://nextg.nist.gov/submissions/156). See the [asset README](libraries/targets/digital_twin_update1/README.md) for attribution and terms. |
| Curated sensing documentation figures | Included in the v0.2 source export | ORCHAV-generated figures for the public scenario catalog, with retained scenario commands and adjacent provenance receipts where provided. Geographic figures retain the applicable source-data attribution. |
| `libraries/targets/uav/` | Included in the v0.2 source export | Original procedural ORCHAV quadrotors, GPL-3.0-only; dimensional references and component metadata accompany each mesh |
| Urban-road, urban-multi-target and port geographic inputs | Not bundled; obtain or generate locally | © OpenStreetMap contributors; applicable ODbL and other source conditions remain with local inputs. See [external geographic inputs](docs/generator/geography/osm_scene_generation.md#local-scene-inputs) and port attribution (retained with the source-only Port scene) |
| Geographic scene illustrations | Illustrated outputs only; raw scene data is not bundled | Retain OpenStreetMap contributor attribution and the recorded source context; illustrations are not a reproducible geographic dataset |

The [Paris Opéra showcase inputs](scenarios/showcases/paris_opera_multitarget/inputs/README.md)
include an attributed OpenStreetMap scene subset: © OpenStreetMap contributors,
ODbL-1.0. Its mesh layers, street graph and source receipts accompany the
example. The bundled BMW model is credited to Mike Pan / Morgan McGuire's
Computer Graphics Archive (CC0/Public Domain), and the scooter to Innerscene
(CC0 1.0). Source notices and provenance accompany the assets. Original ORCHAV
surface artwork and procedural drone surfaces retain their accompanying
credits. Body models, walking motion inputs and acquisition recordings are not
bundled. ORCHAV's code license does not replace the geographic data or asset
licenses.

Other configurations may reference user-provided scene meshes, textures,
target catalogs, scripts, or source data. Unless an input is named under
[Bundled Assets](#bundled-assets), it is outside the ORCHAV source distribution;
users are responsible for its provenance and applicable terms.

## Optional External Data Workflows

Sionna RT's geographic example scenes, including Munich and Étoile, use
© [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) data
under ODbL 1.0. ORCHAV illustrations of those scenes retain that attribution.
See the [Sionna RT scene catalog](https://nvlabs.github.io/sionna/rt/api/scene.html#built-in-scenes).

The pygfx environment selector can download four 2K PureSky panoramas from
[Poly Haven](https://polyhaven.com/): Overcast Soil (Jarod Guest and Sergej
Majboroda), Kloofendal 43d Clear (Greg Zaal), Kloofendal 48d Partly Cloudy
(Greg Zaal and Jarod Guest), and Qwantani Sunset (Greg Zaal and Jarod Guest).
These optional assets are licensed under [CC0 1.0](https://polyhaven.com/license)
and are not bundled with ORCHAV. Their source URLs, author credits, checksums,
and visual preparation settings are retained in the
[environment catalog](visualizer/src/lighting/environments.yaml). Downloads
remain in the user cache; see [Sky environments](docs/visualizer/appearance.md#lighting-and-skies).

ORCHAV can generate scenes from
[OpenStreetMap](https://www.openstreetmap.org/) data and optional external
building-height sources. Apart from the explicitly attributed Paris Opéra
subset above, downloaded geographic data and generated city datasets or scene
meshes are not bundled.
See [OpenStreetMap Scene Generation](docs/generator/geography/osm_scene_generation.md)
for the ORCHAV workflow. Users who generate and distribute derived data should
review the applicable OSM
[Open Database License](https://www.openstreetmap.org/copyright) attribution
and share-alike requirements, plus any terms for optional external height
sources.

The [SMPL importer](docs/generator/motion/smpl_mesh_generation.md) can evaluate separately acquired [SMPL body models](https://smpl.is.tue.mpg.de/) against [AMASS motion clips](https://amass.is.tue.mpg.de/). Obtain both inputs from their respective projects. Their terms govern generation and sharing of derived meshes. The optional `smplx` package has its own use and redistribution restrictions.

The [Kimodo/SOMA importer](docs/generator/motion/kimodo_motion_import.md) reads
user-supplied motion and skin files. Models, checkpoints, skins, and generated
motion are not bundled. Their source terms also apply to derived assets.

### Marching-person illustration

The [marching-person topology](scenarios/sensing/targets_motion/marching_person/README.md#scene-layout)
is a research illustration rendered by ORCHAV from a separately acquired SMPL
body model and AMASS CMU `08/08_03` motion. Credit: Max Planck Institute for
Intelligent Systems; Loper et al., *SMPL: A Skinned Multi-Person Linear Model*
(2015), and Mahmood et al., *AMASS: Archive of Motion Capture as Surface Shapes*
(2019). ORCHAV replaces the captured root route with the documented 1.4 m/s
approach, interpolates poses and renders sampled surfaces.
See the [SMPL project and terms](https://smpl.is.tue.mpg.de/),
[SMPL-Body attribution](https://smpl.is.tue.mpg.de/license.html), and
[AMASS terms](https://amass.is.tue.mpg.de/license.html). The PNG and its input-hash
receipt illustrate the experiment; body-model files, motion clips and prepared
mesh sequences are not included or relicensed under ORCHAV's code license.

## Adapted OpenStreetMap Scene-Builder Code

The OSM producer contains adaptations of NVIDIA's
[Sionna Large Radio Maps](https://github.com/NVlabs/sionna-large-radio-maps),
with its upstream Geo2SigMap provenance. Original copyright and Apache-2.0
terms remain applicable to those portions; the ORCHAV GPL-3.0-only choice
does not relicense third-party code. The [producer module guide](generator/osm/README.md) describes the API.
Source revisions and modification notices are recorded in
[the module notice](generator/osm/NOTICE.md), with
[the complete Apache License 2.0](generator/osm/LICENSE).
