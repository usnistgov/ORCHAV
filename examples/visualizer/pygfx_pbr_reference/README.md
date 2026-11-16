# Native ORCHAV PBR reference

This is a controlled **visual QA fixture**, not a radio-propagation simulation.
The XML includes display-only PBR metadata and must not be passed to Mitsuba or
Sionna as a validated physical scene. Ray tracing is disabled. The canonical
frame contains Tx/Rx positions and **zero MPC paths**, at synthetic time 0 s.

From the repository root, in the normal ORCHAV environment:

```bash
python examples/visualizer/pygfx_pbr_reference/launch.py
```

The launcher generates three small PLY assets, a named deterministic HDR input,
a local scene XML, and one canonical HDF5 frame using `FrameSetWriter`. It then
opens the **actual ORCHAV Qt Visualizer**, selects the reference camera and HDR,
and leaves the application interactive. No installed dependency is modified.

You can instead prepare the files and open the scenario normally.

Generate the reference scene:

```bash
python examples/visualizer/pygfx_pbr_reference/generate.py
```

Open it with the pygfx renderer:

```bash
orchav-visualizer --renderer pygfx --scenario examples/visualizer/pygfx_pbr_reference/scenario.yaml --no-resume
```

The normal scenario-open command uses normal application lighting. The launcher
selects `assets/reference.hdr` explicitly for repeatable comparisons; it is not
a replacement presented as the repository's `neutral_outdoor.hdr` asset.

## Matrix

- Back row: dielectric, roughness 0.08 / 0.25 / 0.55 / 0.9 from left to right
- Middle row: metallic 1.0, same roughness values
- Front grid row: clearcoat 0.4, coat roughness 0.15, same base roughness values
- Bottom-left three spheres: reflectance 0.0 / 0.5 / 1.0
- Bottom-right cyan panels: alpha 0.25 / 0.5 / 0.75

The separate `tests/visualizer/scenarios/pygfx_pbr_reference.py` fixture adds
texture-alpha and immutable scientific-color reference checks. Its renderer-only
captures are distinct from this native application fixture.

## Reproducible native captures

Capture the current checkout:

```bash
python examples/visualizer/pygfx_pbr_reference/launch.py --capture-dir /path/to/native-after
```

Capture the reference checkout into a different directory:

```bash
python examples/visualizer/pygfx_pbr_reference/launch.py --repo-root /path/to/pristine-main --capture-dir /path/to/native-before
```

Both use the same generated input files, camera, explicit HDR, window dimensions,
and runtime. Output includes the full Qt application image, 1× and 2× viewport
exports, and metadata naming the actual imported renderer source file. On a
headless test machine use Qt's offscreen platform and a supported Vulkan adapter;
these images do not establish interactive GPU performance on another machine.

Generated assets, frames, and screenshots are intentionally excluded from the
source commit. Regenerate them with the commands above. The launcher replaces
only this fixture's manifest-owned frame set using the normal writer's safety
checks.
