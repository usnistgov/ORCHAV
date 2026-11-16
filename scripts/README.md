# Repository scripts

These scripts maintain the repository, build documentation assets, and run checks. To run a simulation or open a recording, start with the [Quickstart](../docs/getting_started/quickstart.md).

| Folder | Purpose |
| --- | --- |
| `benchmarks/` | Measure renderer and processing performance. |
| `ci/` | Run repository checks, including [Markdown link checks](ci/check_markdown_links.py) and the [release smoke test](ci/release_smoke.sh). |
| `docs/` | Generate reference material and [scenario previews](docs/generate_scenario_previews.py). |
| `protobuf/` | [Regenerate the Python bindings](protobuf/generate_protobuf.py) for the shared frame protocol. |
| `analysis/` | Check sensing scenarios and their signal-processing outputs. |
| `validation/` | Inspect saved sensing volumes and their rendered cells. |

Run commands from the repository root. For test setup and focused commands, see the [test guide](../tests/README.md).

<a id="sensing-figure-and-foundation-utilities"></a>

## Sensing figure utilities

These optional tools reproduce documentation figures; they
are not part of ordinary scenario generation.

Render a PNG and JSON provenance receipt from an existing saved cube recording:

```bash
python -m scripts.docs.generate_sensing_curriculum_figures --scenario scenarios/sensing/cube_radar/stationary --name stationary --output-dir analysis/sensing-doc-figures
```
