# ITU Preset Diffuse Scattering

[Scenarios](../../../../README.md) > [Generator](../../../README.md) > [Scene Diffuse Scattering](../README.md) > ITU Preset

This variant enables diffuse solving and applies ORCHAV's known scattering
coefficients to recognized scene-material families. It uses the same open room,
fixed transmitter, and fixed receiver as the parent comparison.

To run it, see [Running](#running).

## Scene Layout

The transmitter is stationary at `(-3.0 m, -3.0 m, 4.0 m)`, and the receiver
is stationary at `(3.0 m, 3.0 m, 1.0 m)`. The room contains concrete, wood,
brick, plasterboard, and chipboard surfaces.

## Configuration Walkthrough

```yaml
raytracing:
  scene_materials:
    scattering_coefficient_preset: itu
  quality:
    custom:
      max_depth: 1
      diffuse_reflection: true
```

The solver switch allows diffuse interactions. The preset then gives known
material families nonzero scattering coefficients. Brick stays at its loaded
value because the preset does not define its family.

Material values are resolved from the loaded scene, then the optional preset,
then any explicit override. This variant has no explicit override.

## Running

```bash
orchav-validate scenarios/generator/propagation_and_materials/scene_diffuse_scattering/itu_preset
orchav-generator scenarios/generator/propagation_and_materials/scene_diffuse_scattering/itu_preset
orchav-inspect scenarios/generator/propagation_and_materials/scene_diffuse_scattering/itu_preset --frame 0 --materials
orchav-visualizer --scenario scenarios/generator/propagation_and_materials/scene_diffuse_scattering/itu_preset
```

Generation writes this variant's frame set under `itu_preset/frames/`. Inspect
the effective coefficients and confirm that the interaction summary contains
`code 2 (diffuse)`. Exact path counts can vary with Sionna RT and the tracing
environment.

## Browse Scenarios

> **Scenario path:** [All scenarios](../../../../README.md) | [Generator](../../../README.md) |
> Parent: [Scene Diffuse Scattering](../README.md) | Current: **ITU Preset**
>
> Compare: [Solver Switch Only](../README.md#1-enable-diffuse-solving-only) | [Preset With Concrete Override](../itu_preset_concrete_override/README.md) | [Explicit Concrete](../explicit_concrete/README.md)
