# Circular Receiver Beamforming

[Scenarios](../../../README.md) > [Visualizer](../../README.md) >
[Beamforming](../README.md) > Circular Receiver

This scenario moves one receiver around a stationary transmitter. It provides a
small generated data set for checking antenna overlays, steering choices, and
the way a beam pattern follows the selected TX/RX pair.

To run it, see [Running](#running).

## Scene Layout

- `CenterTX` is fixed at `(0.0 m, 0.0 m, 8.0 m)`.
- `OrbitRX` follows a 12.0 m-radius counter-clockwise circle at `1.5 m`
  height and keeps its orientation pointed at `CenterTX`.
- `CenterTX` uses a 4x8 array with isotropic elements. `OrbitRX` uses a 4x8
  array with dipole elements.
- The scenario uses ORCHAV's 28 GHz carrier and half-wavelength spacing
  defaults.

## Running

```bash
orchav-generator scenarios/visualizer/beamforming/circular_rx
orchav-visualizer --scenario scenarios/visualizer/beamforming/circular_rx
```

Generation writes only this scenario's `frames/` and `summary/` directories.
The circular motion has 32 steps over 8 seconds.

## What To Check

- In **Context**, select `CenterTX` and `OrbitRX`.
- Open **Antennas**, keep **Standalone** selected, and enable
  **Show Beam Patterns**.
- Compare **SVD (Current MPCs)**, **LOS Steering**, and **Manual Steering**.
- Scrub the timeline and follow the selected beam pattern as `OrbitRX` moves.

## Browse Scenarios

> **Scenario path:** [All scenarios](../../../README.md) |
> [Visualizer](../../README.md) | [Beamforming](../README.md) |
> Current: **Circular Receiver**
>
> Up: [Beamforming](../README.md) | Other variant:
> [Multiple Devices](../multi_device/README.md)
