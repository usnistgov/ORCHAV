# Beamforming

[Scenarios](../../README.md) > [Visualizer](../README.md) > Beamforming

These examples focus on the Visualizer Antennas tab and beam-pattern overlays.
Their [`circular_rx/`](circular_rx/README.md) and
[`multi_device/`](multi_device/README.md) child scenarios keep the two generated
frame sets independent.

Choose the [circular-receiver](circular_rx/README.md) case for one moving pair,
or [multiple devices](multi_device/README.md) for explicit selection among two
transmitters and three receivers. Each child page owns its runnable commands.

## Scenario Configuration

Both child scenarios use the default 28 GHz carrier and half-wavelength element
spacing, and explicitly configure 4x8 TX/RX arrays. Those values initialize the
beamforming view. Changing Antennas panel controls does not rewrite
`scenario.yaml`.

The circular case uses isotropic TX elements. The multiple-device case uses
directional `tr38901` TX elements.

Each child writes its generated chunks to its own `frames/` directory and its
diagnostic figures to its own `summary/` directory. Regenerating one case does
not replace the other case's data.

## What To Notice

- **Show Beam Patterns** displays transmitter and receiver antenna surfaces for
  the selected pair.
- **SVD (Current MPCs)**, **LOS Steering**, and **Manual Steering** change
  overlay steering behavior.
- Manual azimuth and elevation controls appear only for manual steering.
- Pair selection comes from the persistent **Context** controls, not from a
  separate **Antennas** selector.

## Variants

- [Circular receiver](circular_rx/README.md) follows one moving receiver around
  a stationary transmitter.
- [Multiple devices](multi_device/README.md) exercises explicit selection among
  two transmitters and three receivers.

## Browse Scenarios

> **Scenario path:** [All scenarios](../../README.md) | [Visualizer](../README.md) |
> Current: **Beamforming**
>
> Choose: [Circular Receiver](circular_rx/README.md) |
> [Multiple Devices](multi_device/README.md)
