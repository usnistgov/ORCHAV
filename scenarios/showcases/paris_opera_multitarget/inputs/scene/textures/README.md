# Palais Garnier visual material

The two included PNGs are photo-informed illustrative albedos for the cached
Palais Garnier geometry. They preserve mesh geometry and RF materials.
`opera_south_facade_albedo.png` (1536 × 1024) represents the six mapped front
triangles; `opera_verdigris_copper_albedo.png` (1254 × 1254) covers the mapped
copper roofs. These are the original generated images without raster
postprocessing, not surveyed reconstruction or measured material response.

Reference photograph: **Peter Rivera**, *Paris Opera full frontal architecture,
May 2009*, 1 May 2009,
[Wikimedia Commons source](https://commons.wikimedia.org/wiki/File:Paris_Opera_full_frontal_architecture,_May_2009.jpg),
[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).
The photograph itself is not bundled. Changes in the included derivatives:
AI-assisted architectural interpretation, perspective correction, foreground
removal and normalization for mapping. Generated inscriptions and ornamental
details may differ from the monument. Preserve this credit with distributed
derivatives; credit does not imply the photographer's endorsement.

[`mapping.json`](mapping.json) retains the source image and mesh hashes,
projection, geographic IDs and the historical mapping-helper hash. The source
landmark is OSM way 54667456; its front-bearing component is way 822291002.
Map geometry and tags are © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
ODbL 1.0. The runnable visual binding is `../visual_surfaces.json` and uses local
texture paths. Preparation adjusts triangle indices when removing explicitly
underground parts and records the changes in its output.
