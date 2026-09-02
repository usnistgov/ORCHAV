# ORCHAV Test Suite

Tests are organized by package and workflow. The release smoke command runs
formatting, linting, type checks, generator tests, selected shared and
visualizer tests, package build and import checks, scenario validation,
Hello World generation and cleanup, and an optional display-backed visualizer
benchmark.

## Test Organization

### Shared Tests

- `tests/shared/`: frame schema, frame providers, validation, coverage data, and
  shared utilities.
- `tests/regression/`: schema and behavior regression checks.
- `tests/statistics_shared/`: statistics utilities.

### Generator Tests

- `tests/generator/unit/`: generator unit tests.
- `tests/generator/integration/`: generator integration smoke tests.
- `tests/generator/scenarios/`: small test fixtures.

See [tests/generator/README.md](generator/README.md).

### Visualizer Tests

- `tests/visualizer/unit/`: renderer, services, controllers, and UI logic.
- `tests/visualizer/integration/`: data loading and renderer playback checks.
- `tests/visualizer/fixtures/`: synthetic frame and scene fixtures.

### Integration and Transport Tests

- `tests/integration/`: generator/visualizer and gRPC round-trip checks.
- Top-level `tests/test_grpc_*.py`: focused gRPC transport checks.

## Running Tests

Run the complete release smoke from a clean Git checkout. This command checks
that the checkout is clean before it starts and verifies that its status is
unchanged when it finishes:

```bash
bash scripts/ci/release_smoke.sh
```

The normal release smoke uses Qt's offscreen platform. To exercise a Linux VNC
display, open a terminal inside the VNC session, confirm the display assigned
to that session, and select Qt's XCB platform explicitly:

```bash
echo "$DISPLAY"
QT_QPA_PLATFORM=xcb RUN_VISUALIZER_BENCHMARK=1 bash scripts/ci/release_smoke.sh
```

Run focused checks during development:

```bash
python -m pytest --no-cov tests/shared/test_frame_schema.py -q
python -m pytest --no-cov tests/generator/unit/test_generator_cli.py -q
QT_QPA_PLATFORM=offscreen python -m pytest --no-cov tests/visualizer/unit/test_frame_types.py -q
```

The notebook and standalone headless renderer tests do not use Qt. Disable the
`pytest-qt` plugin for that focused process so the test environment preserves
the same Qt-free boundary as the supported runtime:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest --no-cov -p no:pytest-qt \
  tests/visualizer/unit/test_notebook_generator.py \
  tests/visualizer/unit/test_notebook_pygfx.py -q
```

On Apple Silicon macOS, use the release smoke and focused test files as the
normal contributor checks. A very large Visualizer collection in one pytest
process can encounter native Qt teardown instability after otherwise passing
files. Run broader macOS investigations in test-file or package partitions.
The monolithic coverage command in `CONTRIBUTING.md` is not the macOS v0.1
release gate.

## Prerequisites

- Install the package with the relevant extras, for example
  `python -m pip install -e ".[dev]"`.
- Use `QT_QPA_PLATFORM=offscreen` for visualizer tests on headless machines.
- Some integration, renderer, [Sionna RT](https://nvlabs.github.io/sionna/),
  CUDA, and performance tests need a
  configured local environment. Run those explicitly when validating the
  corresponding workflow.

## Adding New Tests

1. Put the test in the matching package area:
   - Generator unit tests -> `tests/generator/unit/`
   - Generator integration tests -> `tests/generator/integration/`
   - Visualizer tests -> `tests/visualizer/unit/` or `tests/visualizer/integration/`
   - Shared data tests -> `tests/shared/`
2. Use the `test_*.py` naming convention.
3. Prefer deterministic fixtures and mock heavy external systems where possible.
4. Keep generated frames, logs, and cache directories out of version control.
