from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

from shared.scenarios import validate_scenario_data
from shared.scenarios.actors import (
    ActorsSpec,
    AlignMotionOrientationSpec,
    CatalogAssetSpec,
    CircularMobilitySpec,
    ConstantSpeedTraversalSpec,
    DirectoryAssetSpec,
    Figure8MobilitySpec,
    FileAssetSpec,
    FitDurationTraversalSpec,
    FixedOrientationSpec,
    GaussMarkovMobilitySpec,
    GridScanMobilitySpec,
    GroupDeviationSpec,
    GroupMemberMobilitySpec,
    GroupOffsetSpec,
    GroupSpec,
    KeyframesOrientationSpec,
    LinearMobilitySpec,
    LookAtOrientationSpec,
    ManhattanGridMobilitySpec,
    MeshSequenceMobilitySpec,
    NetworkRouteMobilitySpec,
    OrientationKeyframeSpec,
    OscillatingMobilitySpec,
    PendulumMobilitySpec,
    RandomOrientationSpec,
    RandomSamplingMobilitySpec,
    RandomWaypointMobilitySpec,
    RxActorSpec,
    SampledMobilitySpec,
    SpinOrientationSpec,
    SpiralMobilitySpec,
    StationaryMobilitySpec,
    SurveyMobilitySpec,
    TargetActorSpec,
    TimelineSpec,
    TxActorSpec,
    WaypointMobilitySpec,
)
from shared.scenarios.model import (
    AntennaArrayModel,
    CoverageModel,
    CoverageSaveDataModel,
    CoverageSaveFigureModel,
    DataConfigModel,
    FilesConfigModel,
    GeneratorSummaryModel,
    LiveGrpcConfigModel,
    MaterialOverrideModel,
    MpcVisibilityDefaultsModel,
    PathFilterModel,
    RayTracingModel,
    RemoteHdf5ConfigModel,
    ScenarioModel,
    SceneMaterialsModel,
    SceneModel,
    ViewDefaultsModel,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_REFERENCE = PROJECT_ROOT / "docs" / "reference" / "scenario_yaml.md"


def _extract_yaml_example(text: str, name: str) -> dict:
    match = re.search(
        rf"<!-- BEGIN EXAMPLE: {re.escape(name)} -->\s*"
        rf"```yaml\s*(.*?)\s*```\s*"
        rf"<!-- END EXAMPLE: {re.escape(name)} -->",
        text,
        flags=re.DOTALL,
    )
    assert match is not None, f"missing YAML example {name!r}"
    parsed = yaml.safe_load(match.group(1))
    assert isinstance(parsed, dict)
    return parsed


def _documented_code_terms(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\.<[^>]+>", "", text).replace("[]", "")
    return set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)`", text))


def _assert_model_fields_documented(
    path: Path,
    model_classes: list[type],
    *,
    field_excludes: dict[type, set[str]] | None = None,
) -> None:
    documented = _documented_code_terms(path)
    missing: dict[str, list[str]] = {}
    for model_class in model_classes:
        fields = set(model_class.model_fields) - (field_excludes or {}).get(model_class, set())
        not_documented = sorted(
            field
            for field in fields
            if field not in documented
            and not any(name.endswith(f".{field}") for name in documented)
        )
        if not_documented:
            missing[model_class.__name__] = not_documented

    assert missing == {}


def test_scenario_parameter_reference_mentions_public_yaml_schema_fields() -> None:
    _assert_model_fields_documented(
        SCENARIO_REFERENCE,
        [
            SceneModel,
            RayTracingModel,
            AntennaArrayModel,
            PathFilterModel,
            MaterialOverrideModel,
            SceneMaterialsModel,
            CoverageModel,
            CoverageSaveDataModel,
            CoverageSaveFigureModel,
            DataConfigModel,
            FilesConfigModel,
            RemoteHdf5ConfigModel,
            LiveGrpcConfigModel,
            ViewDefaultsModel,
            MpcVisibilityDefaultsModel,
        ],
        field_excludes={SceneModel: {"osm"}},
    )


def test_scenario_parameter_reference_covers_actor_motion_and_assets() -> None:
    parameter_reference = SCENARIO_REFERENCE
    _assert_model_fields_documented(
        parameter_reference,
        [ScenarioModel],
        field_excludes={ScenarioModel: {"sensing"}},
    )
    _assert_model_fields_documented(
        parameter_reference,
        [
            TimelineSpec,
            ActorsSpec,
            TxActorSpec,
            RxActorSpec,
            TargetActorSpec,
            GroupSpec,
            GroupDeviationSpec,
            GroupOffsetSpec,
            FitDurationTraversalSpec,
            ConstantSpeedTraversalSpec,
            StationaryMobilitySpec,
            LinearMobilitySpec,
            WaypointMobilitySpec,
            SampledMobilitySpec,
            CircularMobilitySpec,
            SurveyMobilitySpec,
            GridScanMobilitySpec,
            OscillatingMobilitySpec,
            PendulumMobilitySpec,
            Figure8MobilitySpec,
            SpiralMobilitySpec,
            RandomSamplingMobilitySpec,
            GaussMarkovMobilitySpec,
            RandomWaypointMobilitySpec,
            ManhattanGridMobilitySpec,
            NetworkRouteMobilitySpec,
            MeshSequenceMobilitySpec,
            GroupMemberMobilitySpec,
            FixedOrientationSpec,
            OrientationKeyframeSpec,
            KeyframesOrientationSpec,
            AlignMotionOrientationSpec,
            LookAtOrientationSpec,
            SpinOrientationSpec,
            RandomOrientationSpec,
            CatalogAssetSpec,
            FileAssetSpec,
            DirectoryAssetSpec,
        ],
    )

    text = parameter_reference.read_text(encoding="utf-8")
    assert "`actors.tx`" in text
    assert "`actors.rx`" in text
    assert "`actors.targets`" in text
    assert "`visualizer`" in text
    assert "`debug_level`" in text
    assert "### Devices" not in text
    assert "### Target definitions" not in text


def test_component_parameter_guides_link_to_canonical_reference() -> None:
    for relative_path in [
        "docs/generator/configuration.md",
        "docs/visualizer/scenario_defaults.md",
    ]:
        text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert "../reference/scenario_yaml.md" in text
        assert "Scenario YAML Reference" in text


def test_scenario_parameter_reference_generated_block_is_current() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/docs/generate_parameter_reference.py",
            "--target",
            "scenario",
            "--check",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_scenario_schema_version_is_documented_as_an_integer() -> None:
    text = SCENARIO_REFERENCE.read_text(encoding="utf-8")
    compatibility = (PROJECT_ROOT / "docs" / "reference" / "compatibility.md").read_text(
        encoding="utf-8"
    )

    assert (
        "| `schema_version` | integer; `2` | required | Required format identifier for "
        "`scenario.yaml`. ORCHAV 0.1 accepts only `2`. |"
    ) in text
    assert "## Scenario Schema Version" in text
    assert "[Versions and Compatibility](compatibility.md#scenario-yaml)" in text
    assert "rather than being guessed or converted" in " ".join(compatibility.split())


def test_scenario_parameter_reference_keeps_nullable_numeric_constraints() -> None:
    text = SCENARIO_REFERENCE.read_text(encoding="utf-8")

    for yaml_key in [
        "actors.<role>[].mobility.min_distance_m",
        "actors.<role>[].orientation.max_yaw_rate_deg_s",
        "actors.<role>[].orientation.max_pitch_rate_deg_s",
        "actors.<role>[].orientation.update_interval_s",
    ]:
        rows = [line for line in text.splitlines() if line.startswith(f"| `{yaml_key}` |")]
        assert rows
        assert all("> 0.0" in row for row in rows)


def test_scenario_parameter_reference_uses_public_headings_and_stable_anchors() -> None:
    text = SCENARIO_REFERENCE.read_text(encoding="utf-8")
    generated = text.split("<!-- BEGIN GENERATED: scenario-parameters -->", maxsplit=1)[1]

    assert re.search(r"\b[A-Za-z_]\w*(?:Model|Spec)\b", generated) is None
    assert re.search(r"\b(?:dict|list|tuple)\[", generated) is None
    assert re.search(r"\b(?:dict|tuple)\b", generated) is None
    assert "dictionary" not in generated.lower()
    for anchor in [
        "scenario-document",
        "mobility-stationary",
        "orientation-look-at",
        "target-asset-catalog",
        "raytracing-quality",
        "generator-summary-settings",
        "view-defaults",
    ]:
        assert f'<a id="{anchor}"></a>' in generated

    headings = [
        "## Common Scenario Fields",
        "## Actors And Groups",
        "## Mobility",
        "## Orientation",
        "## Target Assets",
        "## Ray Tracing",
        "## Coverage",
        "## Data Modes",
        "## Generator Summary",
        "## Visualizer Defaults",
    ]
    positions = [generated.index(heading) for heading in headings]
    assert positions == sorted(positions)

    anchors = re.findall(r'<a id="([^"]+)"></a>', generated)
    assert anchors
    assert len(anchors) == len(set(anchors))


def test_scenario_parameter_reference_documents_cross_field_constraints() -> None:
    text = SCENARIO_REFERENCE.read_text(encoding="utf-8")

    for expected in [
        "At least one of the three maximum bounds must be greater than zero.",
        "`poisson_disk` requires `min_distance_m`. With `uniform`, omit that field.",
        "`random_walk` requires `seed` and does not accept `start_node` or `end_node`",
        "`resolution_m` must contain exactly two positive values",
        "Enabling figures requires persisted coverage data",
        "An explicit pattern also requires `scattering_coefficient`",
    ]:
        assert expected in text


def test_scenario_parameter_reference_lists_coverage_metric_choices() -> None:
    text = SCENARIO_REFERENCE.read_text(encoding="utf-8")

    store_row = next(
        line for line in text.splitlines() if line.startswith("| `coverage.metrics.store` |")
    )
    derived_row = next(
        line for line in text.splitlines() if line.startswith("| `coverage.metrics.derived` |")
    )
    for value in ["path_gain_linear", "rss_w", "sinr_linear"]:
        assert f"`{value}`" in store_row
    for value in [
        "path_gain_db",
        "path_loss_db",
        "best_path_loss_db",
        "rss_dbm",
        "best_rss_dbm",
        "sum_rss_dbm",
        "sinr_db",
        "serving_tx",
        "tx_margin_db",
    ]:
        assert f"`{value}`" in derived_row


def test_scenario_parameter_reference_has_no_missing_field_descriptions() -> None:
    text = SCENARIO_REFERENCE.read_text(encoding="utf-8")
    generated = text.split("<!-- BEGIN GENERATED: scenario-parameters -->", maxsplit=1)[1]

    parameter_rows = [line for line in generated.splitlines() if line.startswith("| `")]
    assert parameter_rows
    assert all(not line.endswith("| - |") for line in parameter_rows)


def test_scenario_parameter_reference_examples_validate() -> None:
    text = SCENARIO_REFERENCE.read_text(encoding="utf-8")

    validate_scenario_data(_extract_yaml_example(text, "minimal-scenario"))
    LinearMobilitySpec.model_validate(_extract_yaml_example(text, "mobility")["mobility"])
    LookAtOrientationSpec.model_validate(_extract_yaml_example(text, "orientation")["orientation"])
    CatalogAssetSpec.model_validate(_extract_yaml_example(text, "target-asset")["asset"])
    RayTracingModel.model_validate(_extract_yaml_example(text, "raytracing")["raytracing"])
    CoverageModel.model_validate(_extract_yaml_example(text, "coverage")["coverage"])
    DataConfigModel.model_validate(_extract_yaml_example(text, "data-mode")["data"])
    GeneratorSummaryModel.model_validate(
        _extract_yaml_example(text, "generator-summary")["generator_summary"]
    )
    ViewDefaultsModel.model_validate(_extract_yaml_example(text, "view-defaults")["view_defaults"])


def test_scenario_parameter_reference_keeps_public_release_scope() -> None:
    text = SCENARIO_REFERENCE.read_text(encoding="utf-8")
    generated = text.split("<!-- BEGIN GENERATED: scenario-parameters -->", maxsplit=1)[1]

    assert "`sensing`" not in generated
