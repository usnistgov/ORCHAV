from __future__ import annotations

from pathlib import Path

from scripts.ci.check_documentation import check_markdown

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _check(tmp_path: Path, text: str) -> list[str]:
    path = tmp_path / "page.md"
    path.write_text(text, encoding="utf-8")
    return [finding.message for finding in check_markdown(path)]


def test_vertical_mermaid_flow_passes(tmp_path: Path) -> None:
    assert (
        _check(
            tmp_path,
            """```mermaid
flowchart TB
    Producer --> Consumer
```
""",
        )
        == []
    )


def test_html_break_in_mermaid_label_fails(tmp_path: Path) -> None:
    findings = _check(
        tmp_path,
        """```mermaid
flowchart TB
    Producer["ORCHAV Generator<br/>orchestrates Sionna RT"]
```
""",
    )
    assert any("must not use HTML <br>" in finding for finding in findings)


def test_unreviewed_horizontal_flow_fails(tmp_path: Path) -> None:
    findings = _check(
        tmp_path,
        """```mermaid
flowchart LR
    A --> B
```
""",
    )
    assert any("must use top-to-bottom direction" in finding for finding in findings)


def test_horizontal_subgraph_inside_vertical_flow_fails(tmp_path: Path) -> None:
    findings = _check(
        tmp_path,
        """```mermaid
flowchart TB
    subgraph Components
        direction LR
        Producer --> Consumer
    end
```
""",
    )
    assert any("must use top-to-bottom direction" in finding for finding in findings)


def test_marked_horizontal_comparison_passes(tmp_path: Path) -> None:
    assert (
        _check(
            tmp_path,
            """```mermaid
%% orchav-docs: allow-horizontal-comparison
flowchart LR
    subgraph First
        direction TB
        A --> B
    end
    subgraph Second
        direction TB
        C --> D
    end
```
""",
        )
        == []
    )


def test_scenario_authoring_python_example_is_single_step_safe() -> None:
    text = (PROJECT_ROOT / "docs/generator/scenario_authoring.md").read_text(encoding="utf-8")
    section = text.split("## Python-Scripted Actors", 1)[1].split("## Continue", 1)[0]

    assert "WaypointMobilitySpec" not in section
    assert "StationaryMobilitySpec(position_m=rx_position_m)" in section
