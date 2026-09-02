#!/usr/bin/env python3
"""Check public-facing Markdown conventions that ordinary link checks miss."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MERMAID_FENCE_RE = re.compile(
    r"^```mermaid\s*$\n(?P<body>.*?)^```\s*$",
    flags=re.MULTILINE | re.DOTALL,
)
FLOW_DIRECTION_RE = re.compile(r"^\s*(?:flowchart|graph)\s+([A-Z]{2})\b", re.MULTILINE)
SUBGRAPH_DIRECTION_RE = re.compile(r"^\s*direction\s+([A-Z]{2})\b", re.MULTILINE)
HTML_BREAK_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
GLUED_CLOSING_TAG_RE = re.compile(r"</[A-Za-z][^>]*>[A-Za-z0-9]")
HORIZONTAL_EXCEPTION = "%% orchav-docs: allow-horizontal-comparison"

DEFAULT_INPUTS = (
    Path("README.md"),
    Path("docs"),
    Path("examples"),
    Path("scenarios"),
)

# These private or historical surfaces are not part of the curated public
# documentation. Generated public exports are checked without these exclusions.
PRIVATE_PREFIXES = (
    "docs/" + "_internal/",
    "docs/propagation/",
    "docs/" + "sensing/",
    "scenarios/_archive/",
    "scenarios/_deprecated/",
    "scenarios/_internal/",
    "scenarios/benchmarks/",
    "scenarios/generator/private/",
    "scenarios/osm/",
    "scenarios/papers/",
    "scenarios/" + "sensing/",
    "scenarios/visualizer/private/",
)

PRIVATE_FILES = {
    "docs/concepts/architecture_overview.md",
    "docs/generator/architecture.md",
    "docs/license_inventory.md",
    "docs/visualizer/architecture.md",
}


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    message: str


def _is_private_source(path: Path, root: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    return relative in PRIVATE_FILES or relative.startswith(PRIVATE_PREFIXES)


def _markdown_files(root: Path, inputs: Iterable[Path], *, public_export: bool) -> list[Path]:
    files: set[Path] = set()
    for input_path in inputs:
        candidate = input_path if input_path.is_absolute() else root / input_path
        if candidate.is_file() and candidate.suffix.lower() == ".md":
            files.add(candidate)
        elif candidate.is_dir():
            files.update(candidate.rglob("*.md"))
    return sorted(path for path in files if public_export or not _is_private_source(path, root))


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def check_markdown(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8")
    findings: list[Finding] = []
    for match in MERMAID_FENCE_RE.finditer(text):
        body = match.group("body")
        block_line = _line_number(text, match.start("body"))

        if HTML_BREAK_RE.search(body):
            findings.append(
                Finding(
                    path,
                    block_line,
                    "Mermaid labels must not use HTML <br>; use separate nodes, "
                    "edge labels, or concise text.",
                )
            )
        if GLUED_CLOSING_TAG_RE.search(body):
            findings.append(
                Finding(
                    path,
                    block_line,
                    "Mermaid text contains an HTML closing tag joined directly to a word.",
                )
            )

        directions = FLOW_DIRECTION_RE.findall(body) + SUBGRAPH_DIRECTION_RE.findall(body)
        for direction in directions:
            if direction == "TB":
                continue
            if direction == "LR" and HORIZONTAL_EXCEPTION in body:
                continue
            findings.append(
                Finding(
                    path,
                    block_line,
                    "Public flowcharts and subgraphs must use top-to-bottom direction; "
                    "an intentionally horizontal comparison requires the documented "
                    "exception marker.",
                )
            )
    return findings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=list(DEFAULT_INPUTS),
        help="Markdown files or directories to check.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository or generated-export root (default: current directory).",
    )
    parser.add_argument(
        "--public-export",
        action="store_true",
        help="Treat every selected Markdown file as public; do not apply private-source exclusions.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    findings: list[Finding] = []
    for path in _markdown_files(root, args.paths, public_export=args.public_export):
        findings.extend(check_markdown(path))

    for finding in findings:
        relative = finding.path.relative_to(root)
        print(f"{relative}:{finding.line}: {finding.message}")
    if findings:
        print(f"Documentation check failed with {len(findings)} finding(s).")
        return 1
    print("Documentation check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
