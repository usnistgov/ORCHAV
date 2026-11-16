# Generator tests

| Folder | Contents |
| --- | --- |
| `unit/` | Tests for individual components, including configuration, geometry, and command-line arguments. |
| `integration/` | Tests that exercise several components together, including the generation pipeline. |
| `fixtures/` | Scenario inputs, meshes, and other test data. |
| `scenarios/` | Python package marker. Runnable examples live in the [scenario catalog](../../scenarios/README.md). |

Put new unit tests in `unit/` and multi-component tests in `integration/`. Run a focused test from the repository root.

```bash
python -m pytest --no-cov tests/generator/unit/test_generator_cli.py -q
```

See the [test suite guide](../README.md) for environment setup and release checks.
