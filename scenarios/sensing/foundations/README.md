# Processing Comparisons

Reuse one acquisition to isolate a processing choice. This avoids repeating
ray tracing and keeps the underlying channel observations identical.

| Goal | Example | Benchmark |
|---|---|---|
| Compare how frequency windows change a return | [Range windowing](windowing/README.md): compare sidelobes and peak width. | [Range windowing](windowing/README.md#benchmark) |
| Suppress stationary returns | [Static clutter and motion](static_and_moving_cubes/README.md): compare the same samples with and without weighted mean removal. | [Static clutter and motion](static_and_moving_cubes/README.md#benchmark) |

The examples explain the retained observations they require and report
generation and separate processing times in their Benchmark sections.
Optional [figure-generation scripts](../../../scripts/README.md#sensing-figure-and-foundation-utilities)
are documented separately.

## Browse Scenarios

Up: [Sensing](../README.md) | Begin: [Cube radar](../cube_radar/README.md)

Each walkthrough includes its own [measured benchmark](windowing/README.md#benchmark).
