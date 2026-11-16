# Scenario Validation

ORCHAV can check scenario configuration before generation and apply stricter
file checks after generated output exists.

## Scenario Validation

`orchav-validate` checks scenario configuration without running a simulation.
Each argument may be a scenario directory or a YAML file, and more than one
scenario may be checked in one command.

```bash
orchav-validate path/to/scenario
```

Ordinary validation parses the YAML, validates the shared scenario schema, and
checks referenced scene, actor, mobility, and extension inputs. Missing input
paths are reported as warnings, while malformed YAML and schema errors return
a nonzero exit status. A missing generated `frames/` directory is not checked
in this mode, so validation also works before the first Generator run.
The required [`schema_version: 2`](compatibility.md#scenario-yaml) is checked as
part of this schema validation.

Before generation, require all referenced input assets without requiring an
output directory:

```bash
orchav-validate path/to/scenario --require-assets
```

`--require-assets` turns missing scene, mesh, motion and other referenced input
paths into errors. It does not require generated frames. Use this check before the first generation of a scenario.

Use strict validation after generating a scenario:

```bash
orchav-validate --strict path/to/generated/scenario
```

`--strict` treats every referenced-path warning as an error. For a scenario that
saves frames, it also requires the configured frame directory to exist. It does
not open the manifest or HDF5 chunks. Use `orchav-inspect` afterward to confirm
that an actual frame can be loaded.

On success, the validator names the groups it checked, including whether it
checked a saved-frame directory. To print the normalized YAML and schema
defaults before generation, run:

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

## Choose a validation mode

| Mode | Input assets | Generated frames |
| --- | --- | --- |
| Ordinary | Missing paths produce warnings. YAML/schema errors fail. | Not required. |
| `--require-assets` | Missing referenced inputs fail. | Not required. |
| `--strict` | Every referenced-path warning fails. | Configured frame directory must exist, but its contents are not decoded. |

After generation, use [frame inspection](../shared/frame_inspection.md) to
confirm that a saved frame can be loaded. Validation alone does not establish
that a recording is complete or scientifically correct.

## Contributor Project Checks

Scenario validation is separate from the repository's formatting, linting,
typing, test, packaging, and generated-code checks. Contributors should use
the commands maintained in [Contributing](../../CONTRIBUTING.md#checks) and its
[pull-request checklist](../../CONTRIBUTING.md#pull-requests).

---

Home: [Documentation](../README.md) | Related: [Contributing](../../CONTRIBUTING.md) | [Troubleshooting](../help/troubleshooting.md)
