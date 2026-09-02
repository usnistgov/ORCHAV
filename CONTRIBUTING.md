# Contributing

Contributions through GitHub issues and pull requests are welcome. Changes are
subject to maintainer review and applicable NIST repository policies.

## Development Setup

Follow the [installation guide](docs/getting_started/installation.md) to create and activate a
Python 3.12 environment for your operating system. Then install the contributor
dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,grpc]"
```

Install the `jupyter` extra as well only when working on notebook examples.

## Code Style

| Tool | Purpose | Configuration |
|------|---------|---------------|
| Black | required Python formatting | `pyproject.toml` |
| Ruff | required linting and import ordering | `pyproject.toml` |
| MyPy | static checks for selected ORCHAV-owned modules | `pyproject.toml` and `scripts/ci/mypy_typed_surface.sh` |
| pytest | automated tests (coverage is opt-in) | `pyproject.toml` |

- Add type hints for new exported APIs.
- Use module loggers rather than `print()` in production code.
- Keep generated outputs out of version control.
- Include focused tests for behavior changes.

See [docs/development/coding_style.md](docs/development/coding_style.md) for
the detailed style conventions.

Documentation changes should follow the
[documentation style and ownership rules](docs/development/documentation_style.md).
In particular, public workflow diagrams are vertical by default and Mermaid
labels must not use HTML line breaks.

## Checks

Run focused tests for the area you change while developing:

```bash
python -m pytest --no-cov -q path/to/test_file.py
```

Before opening or updating a pull request, commit the intended changes, make
sure the Git worktree is clean, and run the release smoke test:

```bash
bash scripts/ci/release_smoke.sh
```

The release smoke requires Bash. On Windows, run it from Git Bash or another
Bash environment configured to use the same Python environment. It checks
Black, Ruff, Python bytecode compilation, the selected MyPy surface, wheel
construction, source-tree imports and CLI entry points, scenario validation,
generator and focused shared/visualizer tests, and hello-world generation. It
writes temporary outputs outside the checkout and verifies that the worktree
remains unchanged.

The MyPy check covers the ORCHAV-owned modules listed by
`scripts/ci/mypy_typed_surface.sh`. It does not treat the complete scientific
and visualization stack as statically typed.

Run the smoke commands from a single, freshly prepared development
environment. Formatter output can change between Black releases, so do not mix
results from different virtual environments when deciding whether a branch is
clean. If formatting results disagree, compare:

```bash
which python
python -m black --version
```

Run the typed-surface check directly whenever a changed file is listed in
`scripts/ci/mypy_typed_surface.sh`:

```bash
bash scripts/ci/mypy_typed_surface.sh
```

Check public-facing Markdown and Mermaid conventions with:

```bash
python scripts/ci/check_documentation.py
```

Run coverage explicitly when coverage is the goal:

```bash
python -m pytest --cov=generator --cov=shared --cov=visualizer --cov-report=term-missing
```

This single-process coverage command is not the macOS v0.1 release gate. Large
Visualizer test collections have shown native Qt teardown instability after
many tests share one process on Apple Silicon, even when the affected files
pass independently. On macOS, use the release smoke plus focused test files,
and split a coverage campaign into separate pytest processes when complete
coverage is required.

## Generated Interfaces And References

When a scenario-schema change affects the exact YAML reference, regenerate it
and verify that the checked-in page is current:

```bash
python scripts/docs/generate_parameter_reference.py --target scenario --write
python scripts/docs/generate_parameter_reference.py --target scenario --check
```

When the gRPC protocol changes, regenerate the checked-in protobuf modules and
verify the result with the pinned contributor dependencies:

```bash
python scripts/protobuf/generate_protobuf.py
python scripts/protobuf/generate_protobuf.py --check
```

Review generated changes together with the schema or protocol change that
produced them. Do not edit generated blocks or protobuf modules by hand.

## Pull Requests

1. Branch from `main`.
2. Keep changes focused and include tests or documentation updates where needed.
3. Confirm generated data, local caches, and restricted datasets are not committed.
4. Include the quality command you ran and its result in the pull request.
5. Open a pull request with a concise summary and verification notes.

## Reporting Issues

Open a GitHub issue with:

- Steps to reproduce.
- Expected and actual behavior.
- Python version, operating system, and GPU/CUDA information when relevant.
- Scenario path or a minimal scenario snippet when the issue involves generation
  or visualization.
