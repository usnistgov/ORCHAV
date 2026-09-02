from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from generator.io.storage import summary_publication as publication_module
from generator.io.storage.summary_publication import (
    SUMMARY_HASH_MARKER,
    SUMMARY_OUTPUT_CONTRACT_VERSION,
    SummaryPublication,
    SummaryPublicationError,
)
from shared.frames.directory_ownership import destination_lock_path
from shared.scenarios.loader import _summary_yaml_hash

HASH_A = "a" * 64
HASH_B = "b" * 64


def _configuration(
    root: Path,
    *,
    yaml_hash: str = HASH_A,
    force: bool = False,
    requested: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        root=root,
        summary_yaml_hash=yaml_hash,
        generator_summary={
            "enabled": requested,
            "force": force,
            "create": ["scene2d"] if requested else [],
        },
        sensing={},
        coverage_cfg={},
    )


def _cache_key(yaml_hash: str) -> str:
    return publication_module._summary_cache_key(yaml_hash)


def _existing_summary(root: Path, marker: str | None = None) -> Path:
    summary = root / "summary"
    summary.mkdir()
    (summary / "old.png").write_bytes(b"old-summary")
    marker_value = _cache_key(HASH_A) if marker is None else marker
    (summary / SUMMARY_HASH_MARKER).write_text(marker_value + "\n", encoding="utf-8")
    return summary


def test_matching_yaml_reuses_complete_summary_and_warns_about_external_inputs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    summary = _existing_summary(tmp_path)
    before = {path.name: path.read_bytes() for path in summary.iterdir()}

    publication = SummaryPublication(_configuration(tmp_path))

    assert publication.begin() is False
    assert publication.skipped
    assert {path.name: path.read_bytes() for path in summary.iterdir()} == before
    assert "automatically skipped" in caplog.text
    assert "XML, meshes, textures, target catalogs" in caplog.text
    assert "generator_summary.force: true" in caplog.text
    assert not destination_lock_path(summary).exists()


def test_matching_yaml_is_checked_while_publication_lock_is_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _existing_summary(tmp_path)
    publication = SummaryPublication(_configuration(tmp_path))
    marker_checks: list[bool] = []
    original_marker_matches = publication._marker_matches

    def marker_matches() -> bool:
        lock = publication._lock
        marker_checks.append(bool(lock is not None and lock.acquired))
        return original_marker_matches()

    monkeypatch.setattr(publication, "_marker_matches", marker_matches)

    assert publication.begin() is False
    assert marker_checks == [True]


def test_changed_yaml_replaces_the_whole_summary_tree(tmp_path: Path) -> None:
    summary = _existing_summary(tmp_path)
    publication = SummaryPublication(_configuration(tmp_path, yaml_hash=HASH_B))

    assert publication.begin() is True
    (publication.staging_directory / "new.png").write_bytes(b"new-summary")
    publication.finalize()

    assert not (summary / "old.png").exists()
    assert (summary / "new.png").read_bytes() == b"new-summary"
    assert (summary / SUMMARY_HASH_MARKER).read_text(encoding="utf-8") == (
        _cache_key(HASH_B) + "\n"
    )


def test_force_rebuilds_even_when_yaml_marker_matches(tmp_path: Path) -> None:
    summary = _existing_summary(tmp_path)
    publication = SummaryPublication(_configuration(tmp_path, force=True))

    assert publication.begin() is True
    (publication.staging_directory / "forced.png").write_bytes(b"forced")
    publication.finalize()

    assert not (summary / "old.png").exists()
    assert (summary / "forced.png").read_bytes() == b"forced"
    assert (summary / SUMMARY_HASH_MARKER).read_text(encoding="utf-8") == (
        _cache_key(HASH_A) + "\n"
    )


def test_build_failure_retains_prior_tree_and_invalidates_marker(tmp_path: Path) -> None:
    summary = _existing_summary(tmp_path)
    publication = SummaryPublication(_configuration(tmp_path, force=True))
    publication.begin()
    staged = publication.staging_directory
    (staged / "partial.png").write_bytes(b"partial")

    publication.fail()

    assert (summary / "old.png").read_bytes() == b"old-summary"
    assert not (summary / SUMMARY_HASH_MARKER).exists()
    assert not staged.exists()
    assert not destination_lock_path(summary).exists()


def test_abort_after_begin_retains_prior_tree_and_invalidates_marker(tmp_path: Path) -> None:
    summary = _existing_summary(tmp_path)
    publication = SummaryPublication(_configuration(tmp_path, force=True))
    publication.begin()
    staged = publication.staging_directory
    (staged / "partial.png").write_bytes(b"partial")

    publication.abort()

    assert (summary / "old.png").read_bytes() == b"old-summary"
    assert not (summary / SUMMARY_HASH_MARKER).exists()
    assert not staged.exists()
    assert not destination_lock_path(summary).exists()


def test_promotion_failure_rolls_back_prior_tree_and_invalidates_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = _existing_summary(tmp_path)
    publication = SummaryPublication(_configuration(tmp_path, yaml_hash=HASH_B))
    publication.begin()
    staged = publication.staging_directory
    (staged / "new.png").write_bytes(b"new-summary")
    real_replace = publication_module.os.replace

    def fail_new_tree_promotion(source: Path, destination: Path) -> None:
        if Path(source) == staged and Path(destination) == summary:
            raise PermissionError("injected summary promotion failure")
        real_replace(source, destination)

    monkeypatch.setattr(publication_module.os, "replace", fail_new_tree_promotion)

    with pytest.raises(SummaryPublicationError, match="prior summary was retained"):
        publication.finalize()

    assert (summary / "old.png").read_bytes() == b"old-summary"
    assert not (summary / SUMMARY_HASH_MARKER).exists()
    assert not staged.exists()
    assert not destination_lock_path(summary).exists()


def test_no_requested_summary_products_leave_existing_tree_untouched(tmp_path: Path) -> None:
    summary = _existing_summary(tmp_path)
    publication = SummaryPublication(_configuration(tmp_path, requested=False))

    assert publication.begin() is False
    publication.finalize()

    assert (summary / "old.png").read_bytes() == b"old-summary"
    assert (summary / SUMMARY_HASH_MARKER).read_text(encoding="utf-8") == (
        _cache_key(HASH_A) + "\n"
    )


def test_legacy_yaml_only_marker_cannot_reuse_old_summary(tmp_path: Path) -> None:
    summary = _existing_summary(tmp_path, marker=HASH_A)
    publication = SummaryPublication(_configuration(tmp_path))

    assert publication.begin() is True
    assert publication.active
    assert not publication.skipped

    publication.abort()

    assert (summary / "old.png").read_bytes() == b"old-summary"
    assert not (summary / SUMMARY_HASH_MARKER).exists()


def test_summary_cache_key_includes_output_contract_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_key = _cache_key(HASH_A)

    monkeypatch.setattr(
        publication_module,
        "SUMMARY_OUTPUT_CONTRACT_VERSION",
        SUMMARY_OUTPUT_CONTRACT_VERSION + 1,
    )

    assert _cache_key(HASH_A) != current_key


def test_normalized_yaml_hash_ignores_key_order_and_force_value() -> None:
    left = {
        "schema_version": 2,
        "timeline": {"steps": 2, "duration_s": 1.0},
        "generator_summary": {"enabled": True, "force": False, "create": ["scene2d"]},
    }
    right = {
        "generator_summary": {"create": ["scene2d"], "force": True, "enabled": True},
        "timeline": {"duration_s": 1.0, "steps": 2},
        "schema_version": 2,
    }

    assert _summary_yaml_hash(left) == _summary_yaml_hash(right)


def test_coverage_figure_alone_requests_summary_publication(tmp_path: Path) -> None:
    configuration = SimpleNamespace(
        root=tmp_path,
        summary_yaml_hash=HASH_A,
        generator_summary={},
        sensing={},
        coverage_cfg={
            "enabled": True,
            "save": {"figure": {"enabled": True}},
        },
    )
    publication = SummaryPublication(configuration)

    assert publication.begin() is True
    publication.abort()


@pytest.mark.parametrize(
    ("sensing_enabled", "sensing_selected", "expected"),
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_sensing_summary_requests_output_only_when_enabled_and_selected(
    tmp_path: Path,
    sensing_enabled: bool,
    sensing_selected: bool,
    expected: bool,
) -> None:
    configuration = SimpleNamespace(
        root=tmp_path,
        summary_yaml_hash=HASH_A,
        generator_summary={
            "enabled": True,
            "create": ["sensing"] if sensing_selected else [],
        },
        sensing={"enabled": sensing_enabled},
        coverage_cfg={},
    )
    publication = SummaryPublication(configuration)

    assert publication.begin() is expected
    publication.abort()


def test_exact_summary_entry_cannot_be_an_indirect_directory(tmp_path: Path) -> None:
    target = tmp_path / "external-summary"
    target.mkdir()
    sentinel = target / "sentinel.png"
    sentinel.write_bytes(b"external")
    try:
        (tmp_path / "summary").symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    publication = SummaryPublication(_configuration(tmp_path, yaml_hash=HASH_B))

    with pytest.raises(SummaryPublicationError, match="must be a real directory"):
        publication.begin()
    assert sentinel.read_bytes() == b"external"
