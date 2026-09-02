"""Tests for side-effect-free installed Sionna scene discovery."""

from __future__ import annotations

from types import SimpleNamespace

from shared.scenarios import parsers


def test_available_sionna_scene_ids_uses_canonical_installed_layout(
    monkeypatch,
    tmp_path,
) -> None:
    package_root = tmp_path / "sionna"
    package_root.mkdir()
    origin = package_root / "__init__.py"
    origin.touch()
    scenes_root = package_root / "rt" / "scenes"
    for scene_id in ("zeta", "alpha"):
        scene_dir = scenes_root / scene_id
        scene_dir.mkdir(parents=True)
        (scene_dir / f"{scene_id}.xml").touch()
    incomplete = scenes_root / "missing_matching_xml"
    incomplete.mkdir()
    (incomplete / "different.xml").touch()

    monkeypatch.setattr(
        parsers.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(origin)),
    )

    assert parsers.available_sionna_scene_ids() == ("alpha", "zeta")


def test_available_sionna_scene_ids_is_empty_when_package_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(parsers.importlib.util, "find_spec", lambda _name: None)

    assert parsers.available_sionna_scene_ids() == ()
