# Beam-Pattern Visualization

The **Antennas** tab overlays 3D transmitter and receiver array-response
surfaces for one TX/RX pair in the current frame. These surfaces help inspect
array orientation, steering, and sidelobes. They do not change Generator
physics, generated frames, or `scenario.yaml`.

## Try the Included Scenarios

Generate and open the
[circular-receiver example](../../scenarios/visualizer/beamforming/circular_rx/README.md):

```bash
orchav-generator scenarios/visualizer/beamforming/circular_rx
orchav-visualizer \
  --scenario scenarios/visualizer/beamforming/circular_rx \
  --renderer pygfx
```

Then:

1. Select one transmitter and one receiver in the persistent **Context** row.
2. Open **Antennas** and enable **Show Beam Patterns**.
3. Keep **Standalone** selected and compare **SVD (Current MPCs)**,
   **LOS Steering**, and **Manual Steering**.
4. Scrub the timeline to see the receiver and its pattern follow the motion.

The
[multiple-device example](../../scenarios/visualizer/beamforming/multi_device/README.md)
contains two transmitters and three receivers. Use it to check that choosing a
concrete pair updates both the TX and RX surfaces.

## Pair Selection

Beam patterns always follow the TX and RX selected in **Context**. The TX and
RX fields in **Antennas** are read-only reminders of that selection.

Select one concrete transmitter and one concrete receiver. Selecting `All`
still shows the broader MPC scope, but intentionally disables pair-specific
beam-pattern drawing. The status line reports when the pair is incomplete,
being computed, ready, or unavailable.

## Pattern Sources

| Source | Behavior |
|--------|----------|
| **Standalone** | Computes display weights from the selected arrays and steering strategy. This is the normal inspection mode and the default. |
| **Frame Data** | Uses beamforming weights carried by the current frame. It is enabled only when the frame provides recognized beamforming metadata. |

If frame metadata becomes unavailable while **Frame Data** is selected, the
Visualizer returns to **Standalone** instead of displaying a stale pattern.

## Standalone Steering

| Strategy | Behavior |
|----------|----------|
| **SVD (Current MPCs)** | Derives TX and RX weights from a channel approximation built for the selected pair. If no usable MPCs are available, it uses a line-of-sight fallback. |
| **LOS Steering** | Points each array toward the other selected device. |
| **Manual Steering** | Uses the shared azimuth and elevation controls. |

The SVD preview reconstructs a channel approximation from retained path
geometry and path-loss values. It does not consume full complex channel
coefficients or polarization terms, so treat it as a visualization aid rather
than a link-budget-accurate precoder or combiner export.

## Controls and Limits

The standalone controls set shared TX/RX array rows and columns, carrier
frequency, horizontal and vertical spacing, and steering strategy. Spacing is
shown in wavelength multiples. `0.5 lambda` is a useful starting point, while
larger spacing can create visible grating lobes.

Display controls set angular sampling, TX/RX surface scale, linear or dB
display, dB dynamic range, colormap, and separate TX/RX element patterns.
Supported visual element patterns are isotropic, dipole, and 3GPP TR 38.901.
TX/RX scale changes only the displayed surface size.

Scenario carrier and antenna settings initialize these controls when present.
Panel edits remain runtime visualization state and do not rewrite the scenario.

Interactive arrays are limited to 32 by 32 elements, with at most 180 azimuth
and 91 elevation samples. The Visualizer can reduce requested sampling when a
large array would exceed its temporary-work budget. The Antennas status line
reports that reduction. When **Show Beam Patterns** is off, changing controls
does not build new beam meshes.

## Renderer Support

Beam-pattern overlays work with both the
[Open3D/Filament and pygfx/wgpu renderers](renderers.md). Each renderer replaces
the existing surfaces when the selected pair changes and removes them when
patterns are hidden or a pattern cannot be produced.

See [Visual Analysis](analysis.md) for the surrounding inspection controls and
the [Frame Data Reference](../shared/frame_reference.md) for optional
beamforming metadata.

---

Up: [Visualizer](README.md) | Related: [MPC Explorer](mpc_explorer.md) |
[Visual Analysis](analysis.md) | [Renderers](renderers.md)
