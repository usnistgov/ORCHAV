# ORCHAV test suite

Run tests from the repository root in an environment with the [development dependencies](../CONTRIBUTING.md) installed. Some integration and rendering tests also require Sionna RT, a GPU, or a display.

## Find a test

| Folder | What it tests |
| --- | --- |
| [generator/](generator/README.md) | Scenario preparation, ray-tracing orchestration, and frame generation. |
| `visualizer/` | Rendering, playback, UI controls, and scenario authoring. |
| `shared/` | The frame contract, data providers, scenario loading, and processing utilities. |
| `statistics_shared/` | Statistics and observables computed from frames. |
| `integration/` | Interactions between packages and [gRPC round trips](integration/test_grpc_roundtrip.py). |
| `regression/` | Behavior and [frame schema compatibility](regression/test_frame_schema_regression.py) across changes. |
| `ci/` | Repository checks, archive contents, and release smoke safeguards. |

Top-level `test_*.py` files include gRPC transport and frame compatibility checks. Put new package tests in the folder that owns the behavior and keep reusable test data with its fixtures.

## Run a focused check

Install the test tools with `python -m pip install -e ".[dev,grpc]"`. Each example below runs independently.

```bash
python -m pytest --no-cov tests/shared/test_frame_schema.py -q
python -m pytest --no-cov tests/generator/unit/test_generator_cli.py -q
QT_QPA_PLATFORM=offscreen python -m pytest --no-cov tests/visualizer/unit/test_frame_types.py -q
```

Run notebook tests in a separate process with the Qt plugin disabled.

```bash
python -m pytest --no-cov -p no:pytest-qt tests/visualizer/unit/test_notebook_generator.py tests/visualizer/unit/test_notebook_pygfx.py -q
```

To test the pygfx notebook renderer against an existing recording, supply a scenario directory with frame 0 and scene geometry.

```bash
ORCHAV_NOTEBOOK_TEST_SCENARIO=/path/to/generated/scenario python -m pytest --no-cov -p no:pytest-qt tests/visualizer/unit/test_notebook_pygfx.py -q
```

## Run the release smoke test

From a clean Git checkout, run the [release smoke script](../scripts/ci/release_smoke.sh). It checks formatting, types, documentation, the package build, test collection, selected tests, and the Hello World workflow. It defaults to Qt's offscreen platform.

```bash
bash scripts/ci/release_smoke.sh
```

To include the display-backed benchmark in a Linux VNC session, run the following in that session's terminal.

```bash
QT_QPA_PLATFORM=xcb RUN_VISUALIZER_BENCHMARK=1 bash scripts/ci/release_smoke.sh
```

On Apple Silicon, run broader Visualizer tests one file or package at a time. Large test collections in one process can encounter native Qt teardown failures.
