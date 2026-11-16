#!/usr/bin/env python3
"""Check local Markdown destinations and heading anchors without network access.

Defaults to tracked/unignored README, docs and scenario Markdown. Generated
outputs and private archives are excluded; explicit file arguments are accepted.
External URLs are outside this check. Markdown examples inside fences are ignored.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {"_internal", "_archive", "_deprecated", "benchmarks", "private", "papers"}
LINK = re.compile(r"!?\[[^\]]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+['\"][^\n]*?['\"])?\s*\)")
REFERENCE = re.compile(r"^ {0,3}\[[^\]\n]+\]:\s*(<[^>]+>|\S+)", re.M)
HTML_LINK = re.compile(r"<[^>]+\b(?:src|href)=[\"']([^\"']+)[\"']", re.I)


def prose(text: str) -> str:
    """Blank fenced examples while retaining line numbers for diagnostics."""
    lines = []
    fence = None
    for line in text.splitlines(keepends=True):
        match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            lines.append("\n")
        else:
            lines.append("\n" if fence else line)
    return "".join(lines)


def anchors(text: str) -> set[str]:
    text = prose(text)
    found = set(re.findall(r'<[^>]+\b(?:id|name)=["\']([^"\']+)', text))
    counts: dict[str, int] = {}
    headings = re.findall(r"^ {0,3}#{1,6}\s+(.+?)(?:\s+#+)?\s*$", text, re.M)
    headings += re.findall(r"^([^\n]+)\n(?:={3,}|-{3,})\s*$", text, re.M)
    for heading in headings:
        heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
        heading = re.sub(r"<[^>]*>", "", heading).lower()
        slug = "".join(c for c in heading if c.isalnum() or c in "_ -").replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        found.add(slug if count == 0 else f"{slug}-{count}")
    return found


def check_file(path: Path) -> list[str]:
    text = prose(path.read_text(encoding="utf-8"))
    errors = []
    for match in [*LINK.finditer(text), *REFERENCE.finditer(text), *HTML_LINK.finditer(text)]:
        destination = match.group(1).strip("<>")
        parts = urlsplit(destination)
        if parts.scheme or parts.netloc or parts.path.startswith("/"):
            continue
        target = (path.parent / unquote(parts.path)).resolve() if parts.path else path.resolve()
        line = text.count("\n", 0, match.start()) + 1
        if not target.exists():
            errors.append(f"{path}:{line}: missing destination {destination}")
        elif parts.fragment and target.suffix.lower() == ".md":
            if unquote(parts.fragment) not in anchors(target.read_text(encoding="utf-8")):
                errors.append(f"{path}:{line}: missing anchor {destination}")
    return errors


def documentation_paths() -> list[Path]:
    names = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "README.md",
            "docs",
            "scenarios",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    return sorted(
        {ROOT / p for p in names if p.endswith(".md") and not EXCLUDED.intersection(Path(p).parts)}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="*")
    args = parser.parse_args()
    paths = args.paths or documentation_paths()
    errors = [error for path in paths for error in check_file(path)]
    for error in errors:
        print(error)
    print(f"Checked {len(paths)} Markdown files: {len(errors)} broken local links/anchors.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
