"""Discover saved sensing volumes without scanning their native trial recordings."""

from pathlib import Path


def volume_candidates(path):
    """Resolve one volume or a shallow experiment directory, never a trial tree."""
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.name == "sensing_volume.json":
            return [path]
        if path.suffix.lower() in {".yaml", ".yml"}:
            candidate = path.parent / "sensing_volume.json"
            return [candidate] if candidate.is_file() else []
        return []
    if not path.is_dir():
        return []
    candidate = path / "sensing_volume.json"
    if candidate.is_file():
        return [candidate]
    # An ordinary scenario may contain ancillary results. Do not replace it.
    if (path / "scenario.yaml").is_file():
        return []
    return sorted(p for p in path.glob("*/sensing_volume.json") if p.is_file())


def condition_sources(reader):
    """Offer neighbouring named conditions without asserting their comparability."""
    path = getattr(reader, "path", None)
    if path is None:
        return []
    path = Path(path).resolve()
    sources = {path, *volume_candidates(path.parent.parent)}
    declared = reader.metadata.get("comparison_volume")
    if declared:
        other = path.parent / declared
        other = other / "sensing_volume.json" if other.is_dir() else other
        if other.is_file():
            sources.add(other.resolve())
    return sorted(sources)


def resolve_volume_source(visualizer, path):
    """Choose an explicit source; multiple conditions are never silently selected."""
    candidates = volume_candidates(path)
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    from PySide6.QtWidgets import QInputDialog

    labels = [p.parent.name for p in candidates]
    selected, accepted = QInputDialog.getItem(
        visualizer, "Open sensing experiment", "Condition", labels, 0, False
    )
    return candidates[labels.index(selected)] if accepted else False
