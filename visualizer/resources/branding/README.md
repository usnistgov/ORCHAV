# ORCHAV Branding

The editable vector sources are:

- `orchav-wordmark-on-light.svg`
- `orchav-app-icon.svg`
- `orchav-app-icon-small.svg`
- `orchav-app-icon-symbolic.svg`

Run `python scripts/generate_brand_assets.py` from the repository root after
editing a source. The generator preserves these four files and refreshes the
dark-background wordmark, platform icon containers, raster exports, and review
sheets under `.artifacts/branding-review/`.

The optical-size icon supplies raster sizes through 48 px. Sizes of 64 px and
larger use the detailed application-icon master.
