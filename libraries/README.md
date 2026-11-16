# Scene and rendering assets

Scenarios use scene geometry and target meshes to define the environment and the objects that move through it. The Visualizer uses lighting environments and texture maps to control how those objects appear.

| Folder | Contents |
| --- | --- |
| `scenes/` | Scene XML and geometry, including a [box room](scenes/box/open_room6materials.xml), an [empty scene](scenes/empty/empty.xml), and a [ground plane](scenes/ground/ground_60x50.xml). |
| `targets/` | Object meshes and, where supplied, metadata for scale, orientation, or animation. See the [mesh-target example](../scenarios/generator/targets/mesh_targets/README.md) for how to use them. |
| [ibl/](ibl/README.md) | Image-based lighting environments for the Visualizer. |
| [textures/](textures/README.md) | Visual texture maps and physically based rendering presets. |

For asset sources and licensing, see the [third-party notices](../THIRD_PARTY_NOTICES.md) and the credits in the relevant asset directory.
