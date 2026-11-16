"""Named YAML assets remain inspectable after CPU results are relocated."""

from types import SimpleNamespace


def test_relocated_inspection_uses_named_asset_without_rewriting_metadata(tmp_path):
    from visualizer.src.services.target_service import TargetService

    source = tmp_path / "source"
    source.mkdir()
    (source / "cube.ply").write_text("fixture")
    missing = str(tmp_path / "derived/libraries/targets/cube")
    actor = SimpleNamespace(
        name="Cube", asset=SimpleNamespace(source="directory", path=str(source))
    )
    viz = SimpleNamespace(scenario=SimpleNamespace(actors=SimpleNamespace(targets=(actor,))))
    owner = SimpleNamespace(visualizer=viz)
    resolve = TargetService._inspection_asset_directory
    assert resolve(owner, "Cube", "cube.ply", missing) == str(source)
    assert resolve(owner, "Other", "cube.ply", missing) == missing
    assert resolve(owner, "Cube", "../cube.ply", missing) == missing
    # Existing recorded assets remain authoritative.
    assert resolve(owner, "Cube", "cube.ply", str(source)) == str(source)
