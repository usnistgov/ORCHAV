# NIST Walking Human — DigitalTwin Update 1

Colored mesh sequence: **38 PLY poses at 7 fps**, confirmed by
the provider. The first-to-last timestamp span is **37/7 ≈ 5.286 s**. Each pose
contains 18,439 vertices and 36,874 triangles with the same topology. Vertex
colors are retained. At scale 1, the person is approximately 1.9 m tall.

The original PLY bytes are unchanged. `mesh_library.json` records their hashes,
the source archive hash, timing and provenance.

## Source and attribution

Source: [NIST NextG Channel Model Alliance, submission 156](https://nextg.nist.gov/submissions/156).
The `DT_Pjh_02` walking meshes are provided by NIST and are available through
the submission page. Credit NIST and the contributors identified there.

ORCHAV bundles the supplied PLY geometry and vertex colors with timing and
front-direction metadata for its examples. Actor placement and orientation are
scenario choices; the supplied mesh bytes are unmodified. The bundle does not
include the source RF measurements, camera images, or LiDAR data. Consult the
submission-specific terms and [NextG data-sharing agreement](https://nextg.nist.gov/data_sharing_agreement)
for other dataset uses.

## Use in a scenario

```yaml
timeline:
  steps: 38
  duration_s: 5.285714285714286
actors:
  targets:
    - name: Walker
      asset:
        source: catalog
        id: digital_twin_update1
        switch_meshes: true
        mesh_end_behavior: hold_last
      mobility:
        type: linear
        start_m: [0, 0, 0.98]
        end_m: [1.876, 0, 0.98]
      orientation:
        type: align_motion
        allow_pitch: false
```

The example height leaves approximately 1–4 cm of clearance on flat ground
near z = 0; adjust each pose's center height for precise foot contact.
Catalog timing metadata is descriptive: the scenario timeline must select
7 fps explicitly. One mesh is used per generated frame with the default mesh
update interval. `hold_last` avoids assuming the supplied clip is a seamless loop.

The source has +Z upward, an embedded global translation, and a front bearing
of approximately −140.12° from +X toward +Y. `target_metadata.json` corrects
this front to actor +X. Combined with `align_motion`, the person faces the
horizontal direction of travel. Actor positioning recenters each mesh; the
original global translation is not added again. The front was checked from
the mesh face/profile views, using the overall source displacement as its
approximate heading.

The default glass RF material is an illustrative placeholder for human
scattering. Mesh color does not determine the RF material properties.

See [Mesh Targets](../../../scenarios/generator/targets/mesh_targets/README.md)
and [Target Orientation](../../../scenarios/generator/targets/target_orientation/README.md).
