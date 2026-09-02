# Coding Style Guide

Use these conventions for code added or changed in a contribution. They define
the current standard for touched code, not a claim that every historical module
already follows every preference.

Keep each change focused. Improve nearby code when the result is low-risk and
directly related, but do not expand a change solely to normalize untouched
code.

## Change Scope

The following requirements apply when their surface is touched:

- Changed Python files must pass the configured Black and Ruff checks.
- Behavior changes need focused tests for the affected package or boundary.
- Files in the configured MyPy surface must pass its typed-surface check.
- Generated references and interfaces must be regenerated and checked when
  their source schema or protocol changes.
- Local paths, credentials, private addresses, and unreviewed runtime outputs
  must stay out of version control.

Broader type coverage, complete docstrings, logger adoption, and consistent
constant naming remain repository-wide goals. Untouched historical code does
not need a style-only rewrite as part of an unrelated change. The pull-request
and release checks in [Contributing](../../CONTRIBUTING.md#checks) still apply.

## Formatting And Linting

Use Black formatting and keep Ruff clean for changed Python files. Prefer
small, focused diffs over unrelated style churn.

The canonical checks are listed in `CONTRIBUTING.md`. When a change touches a
specific package, run the narrowest formatting, linting, type-checking, and test
commands that cover that package.

## Logging And User Output

Use module loggers for new or modified library, service, pipeline, and
long-running runtime behavior:

```python
from shared.logging import get_logger

logger = get_logger(__name__)
logger.info("Loaded %d frames", frame_count)
logger.debug("Frame metadata: %r", metadata)
```

Intentional stdout is acceptable for command-line interfaces, small maintenance
scripts, benchmark summaries, and tests. Examples include CLI catalog output,
`--help`-style messages, progress summaries, and benchmark tables.

When using `print()` intentionally in checked Python files, make the intent
clear near the call or with a targeted lint suppression:

```python
print(format_scenario_catalog())  # noqa: T201 - intentional CLI output
```

Avoid `print()` for debug traces inside reusable runtime code. Prefer structured
logger calls so tests, scripts, and applications can control verbosity.

## Type Hints

Add type hints for new exported APIs, dataclasses, protocol/data-contract code,
and new service boundaries. Prefer type hints for new helpers when they improve
readability.

Do not perform broad type-only rewrites across dynamic GUI,
Sionna/Mitsuba/Open3D, or payload code unless the module is part of the
configured MyPy target set or the typing directly supports the change.

Prefer Python 3.12-style built-in generics:

```python
def load_frames(
    path: Path,
    steps: list[int] | None = None,
) -> dict[int, StandardMPCFrame]:
    ...
```

Use `Any` sparingly, mainly for external libraries without stable types, dynamic
YAML/protobuf/HDF5 payloads, and renderer objects.

## Docstrings

Add or improve docstrings when creating or materially changing public modules,
public classes, exported functions, and non-obvious internal helpers. Avoid
boilerplate docstrings that restate the function name. Prefer domain intent,
units, side effects, and ownership boundaries.

```python
def build_coverage_request(
    scenario_path: Path,
    receiver_height_m: float,
) -> CoverageRequest:
    """Build a coverage-map request for one scenario.

    Args:
        scenario_path: Directory containing `scenario.yaml`.
        receiver_height_m: Receiver sampling height in meters.

    Returns:
        CoverageRequest ready for the Generator coverage service.
    """
```

## Domain Constants

Name new or modified values that encode domain policy, units, Sionna behavior,
data-contract versions, renderer conventions, or repeated thresholds:

```python
MAX_DELAY_NS = 500.0  # Maximum propagation delay retained in diagnostics.
NOISE_FLOOR_DB = -130.0  # Receiver noise-floor assumption for this model.
```

Do not extract every obvious one-off value if a constant would reduce
readability. The goal is to make policy visible, not to hide simple local
values behind names.

## Local Paths And Credentials

Do not commit local-only paths, private IPs, credentials, or machine-specific
environment assumptions. Use configuration, environment variables, or documented
scratch paths instead:

```python
server = os.environ.get("ORCHAV_SERVER", "localhost:50052")
path = Path(config.get("data_path", "./data"))
```

---

Up: [Documentation](../README.md) | Related: [Contributing](../../CONTRIBUTING.md) | [Documentation Style](documentation_style.md)
