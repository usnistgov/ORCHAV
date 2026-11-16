# Assets and appearance

- **Geographic geometry:** © OpenStreetMap contributors, ODbL 1.0.
  [OSM copyright and attribution](https://www.openstreetmap.org/copyright).
  The local source is the cached ORCHAV Paris Opéra scene. The surface-scene
  audit records omitted underground parts. Source geometry, tags and local
  coordinate information are bundled in `inputs/scene/` and retained in each
  prepared run's `scene/` and `scene_clean/`.
- **Cars:** Mike Pan and Morgan McGuire, CC0/Public Domain BMW model from
  the [Computer Graphics Archive](https://casual-effects.com/data/).
  The hood, rear and four wheel roundels and rear model emblem are removed.
  The retained mesh has 355,786 triangles, is 4.5 m long, and is used in both
  radio simulation and native display. Source hashes, badge-removal records,
  conversion details and license evidence are bundled in `inputs/vehicles/bmw/`
  and copied to the run's `assets/vehicles/bmw/`.
  Source material colors are baked to a palette texture with blue paint and
  dark opaque windows; the normal visualizer applies its Satin material preset.
- **Scooter:** Innerscene, [Urban motor scooter](https://www.innerscene.com/tools/library/3d-parts/urban-motor-scooter-9dd945f9), CC0 1.0.
  The original GLB, embedded textures, source page, hashes and the CC0 legal
  code are retained under `assets/vehicles/scooter/`. Source repeating albedos
  are packed into a native UV skin without changing the mesh topology.
  Display appearance does not set the radio material.
- **Drone:** original ORCHAV dimensioned quadrotor geometry from
  `libraries/targets/uav/large`, under the repository’s GPL-3.0-only license.
  The 1.6 m × 1.5 m × 0.7 m envelope is a dimensional example, not a
  manufacturer model or a calibrated RCS reference.
- **Walking body:** explicitly supplied canonical motion, or the separately
  supplied historical 34-pose library. Neither body geometry nor motion is
  bundled. Generated manifests record input hashes, coordinate policy, pose
  repetition and root replacement. Models, motion and derived geometry retain
  their separate terms; the code license does not relicense them. The optional
  matching body texture is also supplied separately.
- **Opéra albedos:** photo-informed derivatives of **Peter Rivera**, *Paris
  Opera full frontal architecture, May 2009*, 1 May 2009,
  [Wikimedia Commons source](https://commons.wikimedia.org/wiki/File:Paris_Opera_full_frontal_architecture,_May_2009.jpg),
  [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). Changes include
  AI-assisted architectural interpretation, perspective correction, foreground
  removal and normalization for mapping. Retain this credit with derivatives;
  it does not imply endorsement. The reference photograph is not bundled.
- **Other visual appearance:** ORCHAV-authored Haussmann façades and
  `uav_survey` material atlas are illustrative surface appearances, under
  GPL-3.0-only. The limestone paving albedo
  is an original image generated for this scenario with OpenAI image
  generation on 2026-10-23; its visual binding is recorded in
  `scene_clean/paving_provenance.json`.

The presentation contains native scene captures and plots of saved simulated
signals. Landmark and material textures do not represent a photogrammetric
survey or calibrated electromagnetic measurements.
