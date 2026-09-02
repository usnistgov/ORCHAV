# Simulation Configuration

Use this guide after a first scenario works to choose its scene, ray-tracing
quality, antenna arrays, material behavior, path filtering, and coverage-map
settings. For the complete scenario workflow and actor structure, start with
[Scenario Authoring](scenario_authoring.md).

For exact YAML fields, defaults, allowed values, and schema constraints, use
the canonical [Scenario YAML Reference](../reference/scenario_yaml.md). For
data access options, see the [Shared Data Layer](../shared/README.md). For actor
movement and orientation, see [Mobility and
Orientation](mobility_and_orientation.md). For diagnostic images, see [Summary
and Coverage Figures](generated_figures.md).

## Scene Selection

Choose where ORCHAV should find the scene identified by `scene.id`:

### Scene Sources

| Source | Meaning |
|--------|---------|
| `library` | Resolve `scene.id` under `libraries/scenes/` in the project tree. |
| `local` | Resolve `scene.id` relative to the scenario directory. |
| `sionna` | Resolve a scene bundled with the installed [Sionna RT](https://nvlabs.github.io/sionna/) package, such as `etoile`. |

The default source is `library`. Exact `scene` fields are listed in the
[Scenario YAML Reference](../reference/scenario_yaml.md#scene).

## Ray-Tracing YAML Fields

Set `raytracing.enabled: true` when the Generator should compute propagation
paths. The remaining keys change radio parameters, solver quality, antenna
arrays, material behavior, and the data retained in each generated frame.

For example, this block selects a 3.5 GHz, 100 MHz study and retains the
optional path metrics used by downstream analysis:

```yaml
raytracing:
  enabled: true
  carrier_frequency_hz: 3.5e9
  bandwidth_hz: 100e6
  export_path_metrics: true
```

Normal generation defaults to the `low` quality preset, a 28 GHz carrier, and
2 GHz bandwidth. Omit those settings when the defaults represent the intended
study.

Exact `raytracing`, `quality`, `antenna`, `path_filter`, and `materials`
fields are listed in the
[Scenario YAML Reference](../reference/scenario_yaml.md).

### Export Path Metrics

Set `raytracing.export_path_metrics: true` when downstream tools need per-path
delay, path loss, angle of departure, or angle of arrival. The Generator then
stores those values with each path in the HDF5 frame output.

Leave this option `false` when those metrics are not needed. The Generator then
avoids computing and writing the optional metric arrays, which can reduce
generation work, output size, and write time for frame sets with many paths.
Path vertices, interaction codes, and material information are still written.

When `path_filter` sets a threshold or per-pair limit, ORCHAV automatically
enables path metrics because the filter needs path-loss values to decide which
paths to retain.

### Quality Presets

Presets are convenience starting points. Choose one for the scene and runtime
budget, then override only the settings the scenario needs to change.

| Preset | max_depth | samples_per_src | max_num_paths_per_src | Reflections | Refraction | Diffraction |
|--------|-----------|-----------------|-----------------------|-------------|------------|-------------|
| `ultra-low` | 2 | 100K | 100K | Specular only | No | No |
| `low` | 3 | 1M | 500K | Specular only | No | No |
| `medium` | 4 | 10M | 1M | Specular + diffuse | No | No |
| `high` | 5 | 10M | 5M | Specular + diffuse | Yes | Yes |
| `ultra` | 6 | 100M | 10M | Specular + diffuse | Yes | Yes |

Ordinary MPC generation defaults to `low`. Coverage-map generation has its own
`medium` default.

### Synthetic Array Model

All presets set `synthetic_array: true`, which is the array model supported by
the current ORCHAV frame workflow. Sionna RT traces one geometric path set
between each transmitter and receiver device, then uses phase shifts to account
for the individual array elements in the channel response. ORCHAV can therefore
store the paths at the transmitter-receiver pair level used by the shared frame
contract.

Sionna RT can instead trace each array element when `synthetic_array` is
`false`. The current ORCHAV frame contract does not preserve that per-element
path topology, so leave this setting `true`. Supporting per-element paths would
require a future extension to the frame contract.

### Custom Quality

The `quality.custom` block overrides individual settings from the chosen
preset. Only the listed keys are replaced. All other values come from the
preset.

```yaml
raytracing:
  quality:
    preset: low
    custom:
      max_depth: 5
      refraction: true
      diffraction: true
```

### Antenna Arrays

Configure TX and RX arrays independently. Array spacing is expressed in
wavelengths. Rows and columns describe the rectangular element layout.

```yaml
raytracing:
  antenna:
    tx:
      pattern: dipole
      polarization: V
      num_rows: 2
      num_cols: 2
      vertical_spacing: 0.5
      horizontal_spacing: 0.5
    rx:
      pattern: iso
```

Omit the block for the default single-element isotropic arrays. Pattern names
are passed to Sionna RT, while ORCHAV validates array dimensions, polarization,
and positive element spacing.

### Diffuse Scattering

Sionna RT computes diffuse paths only when the effective quality settings have
`diffuse_reflection: true`. Material scattering coefficients then determine
how much reflected energy is scattered instead of staying purely specular.

By default, ORCHAV leaves loaded Sionna RT scene material coefficients
unchanged:

```yaml
raytracing:
  scene_materials:
    scattering_coefficient_preset: none
```

Use the `itu` preset when the study needs ORCHAV's material-family coefficients
for known scene materials such as concrete, glass, metal, wood, plasterboard,
and ceiling board:

```yaml
raytracing:
  scene_materials:
    scattering_coefficient_preset: itu
```

Explicit overrides under `raytracing.materials` are applied after the broad
preset and take precedence:

```yaml
raytracing:
  quality:
    preset: low
    custom:
      diffuse_reflection: true
  materials:
    itu_concrete:
      scattering_coefficient: 0.4
    itu_glass:
      scattering_coefficient: 0.05
```

### Material Overrides

Material names can be exact scene material names or common aliases. For
example, `concrete`, `itu_concrete`, `mat-concrete`, and `mat-itu_concrete`
can resolve to the same loaded Sionna RT material when it exists in the scene.

Target meshes create their own material entries during scene setup. If a
target material is named after the target, use that target-specific key, such
as `itu_metal_Drone`, when overriding its scattering behavior.

A coefficient-only override is valid:

```yaml
raytracing:
  materials:
    itu_concrete:
      scattering_coefficient: 0.4
```

When an override sets a scattering pattern or pattern parameter, it must also
set `scattering_coefficient`:

```yaml
raytracing:
  materials:
    itu_concrete:
      scattering_coefficient: 0.4
      scattering_pattern: directive
      alpha_r: 2
```

Exact override fields are listed in the
[Scenario YAML Reference](../reference/scenario_yaml.md#raytracing-material-override). See
the [scene diffuse-scattering comparison](../../scenarios/generator/propagation_and_materials/scene_diffuse_scattering/README.md)
and [target diffuse scattering](../../scenarios/generator/targets/target_diffuse_scattering/README.md)
scenarios for runnable examples.

### Scattering Patterns

Supported scattering patterns include `lambertian`, `directive`,
`backscattering`, `g-rer`, `iso`, `isotropic`, and `er-isotropic`. `iso` is an
alias for `isotropic`. `directive` and `backscattering` require `alpha_r`, and
`g-rer` requires `alpha_g`. Pattern-specific parameters are validated before
generation starts.

### Path Filtering

Path filtering reduces output file size by keeping only the most significant
paths before writing frames.

```yaml
raytracing:
  path_filter:
    relative_threshold_db: 40.0
    max_paths_per_pair: 100
```

Exact path-filter fields are listed in the
[Scenario YAML Reference](../reference/scenario_yaml.md#raytracing-path-filter).

## Coverage Configuration

The coverage solver produces a spatial grid dataset separately from the MPC
frame stream. Use it when a scenario needs path-gain, RSS, SINR,
serving-transmitter, or margin products across an area.

```yaml
coverage:
  enabled: true
  grid:
    resolution_m: [5.0, 5.0]
    heights_m: [1.5]
```

`bbox_xy: auto` derives the study area from scene geometry. Set explicit
`[[x_min, x_max], [y_min, y_max]]` bounds for scenes that do not expose
geometry bounds. Because `auto` is the default, the example does not repeat
it. The default coverage solver uses its independent `medium` preset, persists
canonical path gain with LZF compression, and exposes the standard derived
metrics. Coverage data is written to the fixed scenario-local path
`coverage/coverage_maps.h5`. Add `solver`, `metrics`, or `save` only to
override those defaults.

Exact coverage fields are listed in the
[Scenario YAML Reference](../reference/scenario_yaml.md#coverage). See
[Coverage Maps](../concepts/propagation.md#coverage-maps) for metric definitions and
solver tradeoffs.

## Advanced Local Scene Mesh Transforms

Read this section when a local Mitsuba XML scene applies transforms directly
to PLY or OBJ shapes. Sionna RT and Mitsuba evaluate the complete scene
transform grammar for ray tracing. ORCHAV also reads those meshes for
Visualizer and Scenario Builder previews, summary overlays, and fallback scene
bounds. This lightweight mesh reader accepts the following ordered transform
subset:

```xml
<transform name="to_world">
  <scale value="2"/>
  <rotate x="1" angle="10"/>
  <rotate y="1" angle="20"/>
  <rotate z="1" angle="30"/>
  <translate x="1" y="2" z="3"/>
</transform>
```

Each operation may be omitted. Scale must be one positive uniform scalar.
Rotations use the positive X, Y, then Z canonical axes. Translation may use
X/Y/Z attributes or a three-value vector. Keep the written order as scale,
rotations, then translation.

Matrices, `lookat`, nonuniform or mirrored scale, arbitrary rotation axes,
repeated or reordered operations, and symbolic values are valid in the wider
Mitsuba grammar but cannot be represented by ORCHAV's lightweight reader. A
direct mesh using one of those forms is rejected for ORCHAV preview instead of
being displayed at an approximate position. Camera and sensor transforms are
not subject to this lightweight mesh restriction.

## Related Tasks

| Task | Guide |
|------|-------|
| Create the scenario directory and actors | [Scenario Authoring](scenario_authoring.md) |
| Choose mobility, orientation, or groups | [Mobility and Orientation](mobility_and_orientation.md) |
| Choose file, Live Generator, or Remote HDF5 data access | [Shared Data Layer](../shared/README.md) |
| Create standalone diagnostic figures | [Summary and Coverage Figures](generated_figures.md) |
| Look up one exact YAML field | [Scenario YAML Reference](../reference/scenario_yaml.md) |

---

Up: [Generator](README.md) | Previous: [Scenario Authoring](scenario_authoring.md) | Next: [Mobility and Orientation](mobility_and_orientation.md)
