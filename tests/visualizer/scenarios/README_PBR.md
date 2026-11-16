# PBR correctness reference

These fixtures render ORCHAV materials and HDR environments with pygfx 0.17. They run without simulating a radio channel. To inspect the surfaces in the desktop Visualizer, open the [PBR reference example](../../../examples/visualizer/pygfx_pbr_reference/README.md).

## Reproduce matched renders

Run the same script in separate Python processes against baseline and candidate
checkouts. Both checkouts need their normal visualizer dependencies. Do not
change the script, camera, lights, exposure, or input images between runs.

```bash
QT_QPA_PLATFORM=offscreen python tests/visualizer/scenarios/pygfx_pbr_reference.py --repo-root /path/to/baseline --output /path/to/evidence/pbr_before.png
QT_QPA_PLATFORM=offscreen python tests/visualizer/scenarios/pygfx_pbr_reference.py --repo-root /path/to/candidate --output /path/to/evidence/pbr_after.png

QT_QPA_PLATFORM=offscreen python tests/visualizer/scenarios/pygfx_hdr_neutral_reference.py --repo-root /path/to/baseline --output /path/to/evidence/hdr_before.png
QT_QPA_PLATFORM=offscreen python tests/visualizer/scenarios/pygfx_hdr_neutral_reference.py --repo-root /path/to/candidate --output /path/to/evidence/hdr_after.png
```

Use a working hardware or software WGPU adapter. Each run records actual adapter,
render settings, input values, and sample pixels in its adjacent JSON. Inputs
are deterministic generated RGBE/PNG files, independent of repository LFS assets.

## Contracts and interpretation

- Payload alpha applies once through native opacity. Uniform alpha is one;
  texture and explicit vertex alpha still multiply opacity independently.
- Explicit unlit materials stay unlit even with nonzero clearcoat or reflectance.
- Display reflectance `r` maps to dielectric `F0 = 0.16*r*r`, with `r` clamped
  to `[0,1]`. Physical IOR is `(1+0.4*r)/(1-0.4*r)`, hence finite from `1` to
  `7/3`. The default `r=0.5` means 4% F0 and IOR 1.5. Standard retains its fast
  default path; nondefault reflectance selects Physical. Clearcoat promotion
  no longer halves default base-layer F0. Metallic reflectance still blends
  toward albedo. The live renderer's existing no-IBL metalness cap of 0.9 is
  unchanged; with IBL it accepts 1.0.
- HDR samples and generated mip levels remain linear float radiance. Local
  material/skybox subclasses skip only the erroneous environment sRGB decode
  when the source texture is physical. Ordinary sRGB environment sources still
  decode normally. No global pygfx shader or albedo/vertex-color path is patched.
- The three lower neutral patches have equal `radiance * gain = 0.2`; their
  pixels should match within RGBE and 8-bit quantization. This is a correctness
  criterion, not an aesthetic preference. Low radiance must not be decoded
  twice, and values above one must not be amplified by an sRGB transfer curve.
- Scientific vertex streams and their existing pygfx physical-space
  interpretation remain unchanged. Do not interpret the input palette bytes
  as an assertion that screen RGB bytes must equal them.

## Tradeoffs and limits

Existing translucent surfaces become more opaque at the same alpha because
alpha-squaring is removed. HDR scenes may become lighter in shadow and less
clipped in highlights. Nondefault reflectance now changes dielectric appearance
and uses the Physical shader. Scenes tuned around the old behavior may need
visual retuning; these changes do not imply that every image is prettier.

Derived cubemap caches are versioned and stored as float32 instead of float16
(to preserve HDR values above 65504), doubling that disk-cache payload. GPU
texture storage was already float32. Existing cache files are left untouched.

This is a compatibility specialization for the project's pinned pygfx 0.17
shader structure and fails visibly if that structure changes. It does not add
anisotropy, transmission, a new tone mapper, or importance-sampled environment
prefiltering. Existing cubemap mip filtering remains an approximation to PBR
convolution. It does not change radio propagation or simulation values.

## Regression checks

```bash
QT_QPA_PLATFORM=offscreen ORCHAV_RUN_OPTIONAL_TESTS=1 python -m pytest tests/visualizer/unit/test_pygfx_pbr_correctness.py tests/visualizer/unit/test_pygfx_pbr_gpu_reference.py tests/visualizer/unit/test_pygfx_native_state_failures.py tests/visualizer/unit/test_pygfx_renderer_helpers.py tests/visualizer/unit/test_pygfx_ibl_manager_cache.py -q --no-cov -rs
```

The GPU suite includes alpha uniform/vertex/texture cases, HDR linear scaling
above one, linear mip averaging, sRGB and linear skyboxes, scientific-color
identity, and scene-level/per-material environment parity. The non-GPU suite
covers material promotion/demotion, finite reflectance endpoints, no-IBL and IBL
metalness behavior, cache versions/precision, and visual-only clearcoat parsing.
