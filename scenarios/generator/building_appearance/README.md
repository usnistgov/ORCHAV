# Three-Building Appearance Inputs

This local scene supplies the Visualizer's
[Building Appearance exercise](../../visualizer/building_appearance/README.md).
Use that walkthrough to compare scene looks, procedural facades, and the
metadata needed to apply them.

To run it, see [Running](#running).

## Scene And Configuration

The scene contains three original concrete blocks, 12 m, 18 m, and 9 m high.
Two are rotated by 25° and −15°. TX and RX are stationary at
(−25 m, −20 m, 3 m) and (25 m, −20 m, 1.5 m).

[`scenario.yaml`](scenario.yaml) selects the local
[`scene.xml`](scene/scene.xml), an ultra-low ray budget, and one static frame.
[`building_visuals.json`](scene/building_visuals.json) identifies building
faces and parts for the Visualizer. This metadata changes no radio material
or mesh geometry. No OSM acquisition or texture download is needed.

## Running

Generate the frame:

```bash
orchav-generator scenarios/generator/building_appearance
```

Then follow the [Visualizer exercise](../../visualizer/building_appearance/README.md#running)
to open it and compare appearances. Generated HDF5 data is saved under
`frames/` and can be reused for every comparison.


## Browse Scenarios

[Building Appearance exercise](../../visualizer/building_appearance/README.md) ·
[Tagged road and path surfaces](surfaces/README.md) ·
[Visualizer scenarios](../../visualizer/README.md)
