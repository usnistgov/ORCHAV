from types import SimpleNamespace

from visualizer.src.services.object_entry_index import CanonicalEntryIndex


def _visualizer(**collections):
    defaults = {
        "mesh_entries": [],
        "target_entries": [],
        "tx_entries": [],
        "rx_entries": [],
    }
    defaults.update(collections)
    return SimpleNamespace(**defaults)


def test_same_length_in_place_rewrite_repairs_canonical_lookup() -> None:
    original = {"name": "Wall A", "object_key": "mesh:old"}
    entries = [original]
    index = CanonicalEntryIndex(_visualizer(mesh_entries=entries))

    assert index.resolve({"object_key": "mesh:old"}) is original

    replacement = {"name": "Wall B", "object_key": "mesh:new"}
    entries[0] = replacement

    assert index.resolve({"object_key": "mesh:new"}) is replacement
    assert index.index_for_entry(replacement, entry_type="mesh") == 0


def test_index_preserves_first_match_semantics_for_duplicate_names() -> None:
    first = {"name": "Shared"}
    second = {"name": "Shared"}
    index = CanonicalEntryIndex(_visualizer(mesh_entries=[first, second]))

    assert index.resolve({"name": "Shared"}) is first
    assert index.index_for_entry(second, entry_type="mesh") == 1
