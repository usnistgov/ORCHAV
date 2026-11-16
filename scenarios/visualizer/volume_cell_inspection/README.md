# Direct source-cell inspection

This controlled fixture exercises the production `PygfxRenderer`, `CoverageVolumeWindow`
and `SensingVolumeWindow`. It is a small standalone native Qt harness, not a replacement
application shell. Its numeric coverage estimates and sensing reports are synthetic;
no propagation solver, acquisition or detector is run. The generated files are explicitly
labelled and are written outside the checkout to the output directory you choose.

## Running

From the repository root in the normal ORCHAV environment:

Generate a coverage-volume example:

```bash
python -m scenarios.visualizer.volume_cell_inspection.demo --kind coverage --output /tmp/orchav-volume-coverage
```

Generate a sensing-volume example:

```bash
python -m scenarios.visualizer.volume_cell_inspection.demo --kind sensing --output /tmp/orchav-volume-sensing
```

To exercise a retained trial in the complete application, run:

```bash
python -m scenarios.visualizer.volume_cell_inspection.full_app --output /tmp/orchav-volume-retained
```

This adds one genuine typed synthetic frame/manifest for source XYZ `(2, 0, 0)`.
In the linked slices select world `(14, -18, 7)` m, click **Open trial recording**,
and inspect `controlled-report` with target truth `controlled-volume-target` in
its sensing dashboard. **Return to volume** restores the exact source cell.
The remaining synthetic cells have compact reports without retained recordings.
No acquisition or detector is run by either command.

The native scene uses the normal dark-grey background, a black transmitter pole,
and the usual recorded-site markers. Drag to orbit. Click a visible volume cell:
its raw value, world-space centre, source indices and available source context
appear immediately, with three XY/XZ/YZ slices sharing the selected cell.
Scroll the inspector to the selected-cell section if necessary. Click in any
slice to inspect interior or unavailable cells without switching the 3D mode.

The existing slice orientation and spatial sweep controls remain available.
The linked slices show the complete source quantity, including positions hidden
by 3D focus/category/threshold/cutaway controls. They use nearest cells, without
interpolating categories or new measurements.

## Expected checks

The grid is translated and anisotropic: XYZ centres begin at `(10, -18, 7)` m,
with `(2, 3, 4)` m spacing and `(6, 5, 4)` cells.

- Coverage: a MIP click reports a ray-maximum hit and its unclipped dB source value,
  not the palette index. At XYZ indices `(2, 2, 0)` and `(2, 2, 3)`, raw values are
  `-10` and `-20 dB`, although both saturate above the `-50 dB` colour maximum.
  Orbit to look down the Z axis and click their shared projected column at
  `(14, -12)` m. The selected raw maximum is `-10 dB` at Z=7 m, even though
  the nearer Z=19 m cell has the same saturated color. Check the source indices
  and raw value in the inspector rather than inferring the hit from its color.
- Coverage XY slice at Z=7: `(10, -18)` is a saved zero solver contribution (`-inf dB`);
  `(12, -18)` is unavailable with an unknown historical reason. Neither proves no
  physical signal, and neither is a sensing miss
- Sensing XY slice at Z=7: `(10, -18)` is pending, `(12, -18)` is excluded geometry,
  `(16, -18)` is a measured miss with zero reports, and `(14, -18)` is a matched
  controlled report. Trial IDs encode Z/Y/X; the displayed report/trial remains
  tied to the selected source cell
- In the standalone `demo.py` fixture, trial recordings are deliberately absent.
  The Open trial recording button is disabled and explains that compact reports
  remain available. `full_app.py` adds the single retained matched trial above,
  verifies its hashes and opens it with the real application loader
- Native ISO clicks are labelled visible surface hits. They do not reveal arbitrary
  hidden cells. MIP selects one ray-marched maximum, not every cell along the ray;
  equal maxima can resolve to one of the tied source cells
- Camera drags do not select; changing heading/TX/quantity invalidates stale
  presentation. Selecting a cell, changing opacity, and moving crosshairs do not
  modify the HDF5 or manifest
