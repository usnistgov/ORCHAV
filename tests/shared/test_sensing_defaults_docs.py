from __future__ import annotations

import re
from pathlib import Path

from generator.core.configuration import SensingConfig
from shared.scenarios.model import SensingModel


def _documented_runtime_fields(text: str) -> set[str]:
    """Read runtime-field columns, excluding grouped and summary-only tables."""
    documented: set[str] = set()
    column: int | None = None
    in_table = False
    for line in text.splitlines():
        if not line.startswith("|"):
            in_table = False
            column = None
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not in_table:
            in_table = True
            column = next(
                (
                    i
                    for i, cell in enumerate(cells)
                    if cell in {"Parameter", "Key", "Flat runtime/schema field"}
                ),
                None,
            )
            continue
        if column is not None and column < len(cells):
            match = re.fullmatch(r"`([a-z_][a-z_0-9]*)`", cells[column])
            if match:
                documented.add(match.group(1))
    return documented


def test_runtime_field_parser_ignores_grouped_and_summary_only_tables() -> None:
    text = """| Group | Meaning |
|---|---|
| `tracking` | Grouped authoring |

| Setting | Default |
|---|---|
| `history_time_axis` | `scene_frame` |

| Parameter | Default |
|---|---|
| `fft_size_range` | `256` |
| `new_runtime_field` | `1` |

| Key | Default |
|---|---|
| `range_response` | omitted |
| `range_response.model` | `point_bin_floor` |

| YAML under `sensing` | Flat runtime/schema field | Default |
|---|---|---|
| `tracking.jpda.max_joint_nodes` | `jpda_max_joint_nodes` | 100000 |
"""
    assert _documented_runtime_fields(text) == {
        "fft_size_range",
        "new_runtime_field",
        "range_response",
        "jpda_max_joint_nodes",
    }


def test_public_sensing_parameter_reference_matches_runtime_and_schema() -> None:
    params_doc = Path(__file__).resolve().parents[2] / "docs" / "sensing" / "parameters.md"
    text = params_doc.read_text(encoding="utf-8")
    documented = _documented_runtime_fields(text)
    runtime = {field.name for field in SensingConfig.__dataclass_fields__.values()}
    yaml_schema = set(SensingModel.model_fields)

    assert documented == runtime
    assert documented <= yaml_schema


def test_public_sensing_reference_documents_bounded_raw_payload_default() -> None:
    params_doc = Path(__file__).resolve().parents[2] / "docs" / "sensing" / "parameters.md"
    text = params_doc.read_text(encoding="utf-8")

    match = re.search(r"\| `raw_payload_policy` \| `([^`]+)` \|", text)

    assert match is not None
    assert match.group(1) == SensingConfig().raw_payload_policy == "none"
    assert "`none` keeps only processed products" in text


def test_scenario_and_runtime_models_enable_typed_detection_stage_products() -> None:
    scenario = SensingModel(enabled=True, persist_detection_stage_artifacts=True)
    runtime = SensingConfig(enabled=True, persist_detection_stage_artifacts=True)

    assert scenario.persist_detection_stage_artifacts is True
    assert runtime.persist_detection_stage_artifacts is True
