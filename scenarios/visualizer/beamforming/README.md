# Beamforming

Display antenna beam patterns for a selected transmitter and receiver, then
compare steering choices as the receiver moves. These overlays help inspect
array response; changing them does not recompute propagation or modify the
saved frames.

To run it, see [Running](#running).

## Choose An Example

| Example | What to explore |
|---|---|
| [Circular Receiver](circular_rx/README.md) | Follow one receiver around a fixed transmitter. Start here. |
| [Multiple Devices](multi_device/README.md) | Switch between two transmitters and three receivers. |

Both examples use 4 × 8 antenna arrays at 28 GHz with half-wavelength element
spacing. The circular example uses isotropic TX elements; the multiple-device
example uses directional TX elements.

## Running

Run commands from the repository root.

Generate the circular-receiver recording:

```bash
orchav-generator scenarios/visualizer/beamforming/circular_rx
```

Open it to inspect the antenna pattern:

```bash
orchav-visualizer --scenario scenarios/visualizer/beamforming/circular_rx --no-resume
```

Generation saves 32 frames under `circular_rx/frames/`. Paths start hidden so
you can focus on the antenna surfaces.

## Show And Steer The Patterns

1. Under **Scene → Nodes → Labels**, set **TX/RX text** to **Device Name**.
2. In **Context → TX/RX Scope**, select TX **CenterTX** and RX **OrbitRX**.
   Select an individual device in each field: **All** disables beam patterns.
3. Open **Antennas → Pattern Source**, select **Standalone**, and enable
   **Show Beam Patterns**. A surface appears at each selected device.
4. Under **Shared TX/RX Parameters**, set **Strategy** to **LOS Steering**.
   Play or scrub the timeline: the patterns point toward the other device as
   `OrbitRX` moves around `CenterTX`.
5. Try **Manual Steering** and adjust azimuth and elevation. Then compare
   **SVD (Current MPCs)**, which estimates steering from the saved paths.

These recordings do not contain saved beamforming weights, so **Frame Data**
is unavailable. For the meaning and limits of each strategy, see
[Beam-Pattern Visualization](../../../docs/visualizer/beam_patterns.md).

## Switch Between Pairs

Generate the multiple-device recording:

```bash
orchav-generator scenarios/visualizer/beamforming/multi_device
```

Open it to compare device patterns:

```bash
orchav-visualizer --scenario scenarios/visualizer/beamforming/multi_device --no-resume
```

Repeat the steps above with **WestTX** and **OrbitRX**, then change the pair
in **Context** to **EastTX** and **StaticSouthRX**. Both surfaces follow the
new selection. The TX/RX fields in **Antennas** show the selected names;
change the pair in **Context**.

## Browse Scenarios

Up: [Visualizer](../README.md) ·
[Circular Receiver](circular_rx/README.md) ·
[Multiple Devices](multi_device/README.md)
