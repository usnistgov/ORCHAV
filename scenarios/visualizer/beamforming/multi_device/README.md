# Multiple-Device Beamforming

[Scenarios](../../../README.md) > [Visualizer](../../README.md) >
[Beamforming](../README.md) > Multiple Devices

This scenario contains two transmitters and three receivers. It demonstrates
explicit TX/RX pair selection, beam-overlay updates, and steering behavior
when the selected devices have different positions and orientations.

To run it, see [Running](#running).

## Scene Layout

- `WestTX` and `EastTX` are fixed at `(-8.0 m, 0.0 m, 8.0 m)` and
  `(8.0 m, 0.0 m, 8.0 m)`.
- `OrbitRX` follows a 12.0 m-radius clockwise circle at `1.5 m` height.
  `StaticNorthRX` and `StaticSouthRX` remain at `y = 18.0 m` and
  `y = -18.0 m`.
- The scenario uses ORCHAV's 28 GHz carrier and half-wavelength spacing
  defaults, with explicit 4x8 `tr38901` transmitter and `dipole` receiver
  arrays.

## Running

```bash
orchav-generator scenarios/visualizer/beamforming/multi_device
orchav-visualizer --scenario scenarios/visualizer/beamforming/multi_device
```

Generation writes only this scenario's `frames/` and `summary/` directories.
The scenario has 24 steps over 6 seconds.

## What To Check

- In **Context**, select `WestTX` and `OrbitRX`.
- Open **Antennas**, keep **Standalone** selected, and enable
  **Show Beam Patterns**.
- Change the selected pair to `EastTX` and `StaticSouthRX`, then confirm that
  both antenna surfaces follow the new pair.
- Compare **SVD (Current MPCs)**, **LOS Steering**, and **Manual Steering** for
  both pairs.

## Browse Scenarios

> **Scenario path:** [All scenarios](../../../README.md) |
> [Visualizer](../../README.md) | [Beamforming](../README.md) |
> Current: **Multiple Devices**
>
> Up: [Beamforming](../README.md) | Other variant:
> [Circular Receiver](../circular_rx/README.md)
