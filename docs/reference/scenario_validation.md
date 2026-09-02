# Scenario Validation

ORCHAV can check scenario configuration before generation and apply stricter
file checks after generated output exists.

## Scenario Validation

`orchav-validate` checks scenario configuration without running a simulation.
Each argument may be a scenario directory or a YAML file, and more than one
scenario may be checked in one command.

```bash
orchav-validate path/to/scenario
orchav-validate scenario-a scenario-b
```

Ordinary validation parses the YAML, validates the shared scenario schema, and
checks referenced scene, actor, mobility, and extension inputs. Missing input
paths are reported as warnings, while malformed YAML and schema errors return
a nonzero exit status. A missing generated `frames/` directory is not checked
in this mode, so validation also works before the first Generator run.
The required [`schema_version: 2`](compatibility.md#scenario-yaml) is checked as
part of this schema validation.

Use strict validation after generating a scenario:

```bash
orchav-validate --strict path/to/generated/scenario
```

`--strict` treats every referenced-path warning as an error. For a scenario that
saves frames, it also requires the configured frame directory to exist. It does
not open the manifest or HDF5 chunks. Use `orchav-inspect` afterward to confirm
that an actual frame can be loaded.

On success, the validator names the groups it checked, including whether it
checked a saved-frame directory. To inspect the normalized configuration and
applicable defaults in more detail, use `--dump-config`.

To inspect the normalized scenario configuration derived from YAML without
running the Generator, print it as YAML:

```bash
orchav-validate --dump-config path/to/scenario
```

The dump is written to stdout, which makes it suitable for inspection or
redirection to a temporary diagnostic file. It expands defaults owned by the
scenario schema while omitting inactive optional sections and unset values.
Generator and application defaults, including choices that depend on the loaded
scene, are finalized at runtime. This diagnostic view is therefore not a
replacement scenario file or a fully resolved runtime configuration. See the
[Scenario YAML Reference](scenario_yaml.md) for those field-level behaviors.

## Before Sharing a Scenario

Use the checks in this order:

```bash
orchav-validate path/to/scenario
orchav-generator path/to/scenario
orchav-validate --strict path/to/scenario
orchav-inspect path/to/scenario --frame 0
```

This sequence validates the configuration, generates its output, applies the
strict referenced-path and output-directory policy, and confirms that the first
frame can be loaded without opening the Visualizer. For a Python-scripted
scenario, replace the Generator command with
`python path/to/scenario/generate.py`.

## Contributor Project Checks

Scenario validation is separate from the repository's formatting, linting,
typing, test, packaging, and generated-code checks. Contributors should use
the commands maintained in [Contributing](../../CONTRIBUTING.md#checks) and its
[pull-request checklist](../../CONTRIBUTING.md#pull-requests).

---

Home: [Documentation](../README.md) | Related: [Contributing](../../CONTRIBUTING.md) | [Troubleshooting](../help/troubleshooting.md)
