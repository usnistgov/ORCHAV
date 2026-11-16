"""Tests for TX/RX node label resolution."""

from visualizer.src.scene.geometry_helpers import normalize_node_label_mode, resolve_node_label
from visualizer.src.state import AppState, create_initial_state


def test_resolve_node_label_role_mode_ignores_device_names():
    """Role mode ignores scenario/frame names."""
    assert (
        resolve_node_label(
            "TX",
            0,
            (),
            label_mode="role",
            device_names=("MainTransmitter",),
        )
        == "TX1"
    )


def test_resolve_node_label_custom_names_take_precedence():
    """Runtime Rename labels apply in both role and device-name modes."""
    assert (
        resolve_node_label(
            "TX",
            0,
            ("CustomTx",),
            label_mode="role",
            device_names=("MainTransmitter",),
        )
        == "CustomTx"
    )


def test_resolve_node_label_name_mode_uses_custom_then_device_then_role():
    """Name mode prefers custom labels, then scenario/frame names, then fallback."""
    assert (
        resolve_node_label(
            "TX",
            0,
            ("CustomTx",),
            label_mode="name",
            device_names=("MainTransmitter",),
        )
        == "CustomTx"
    )
    assert (
        resolve_node_label(
            "RX",
            0,
            (),
            label_mode="name",
            device_names=("MovingReceiver",),
        )
        == "MovingReceiver"
    )
    assert resolve_node_label("RX", 1, (), label_mode="name", device_names=("RX_A",)) == "RX2"


def test_normalize_node_label_mode_accepts_only_supported_modes():
    assert normalize_node_label_mode("role") == "role"
    assert normalize_node_label_mode("name") == "name"
    assert normalize_node_label_mode("unknown") == "role"


def test_app_state_round_trip_keeps_node_label_mode_and_device_names():
    """Node label settings are persisted through AppState serialization."""
    state = create_initial_state(
        node_label_mode="name",
        tx_device_names=("MainTransmitter",),
        rx_device_names=("WalkingReceiver",),
    )
    restored = AppState.from_dict(state.to_dict())

    assert restored.node_label_mode == "name"
    assert restored.tx_device_names == ("MainTransmitter",)
    assert restored.rx_device_names == ("WalkingReceiver",)


def test_sensing_mode_round_trip_and_non_sensing_fallback():
    assert normalize_node_label_mode(" SENSING ") == "sensing"
    state = create_initial_state(node_label_mode="sensing")
    assert AppState.from_dict(state.to_dict()).node_label_mode == "sensing"
    # Without scene context the base text helper uses conventional roles.
    assert resolve_node_label("RX", 0, label_mode="sensing", device_names=("RadarRX",)) == "RX1"


def test_monostatic_pairing_requires_unique_finite_colocation():
    from shared.node_labels import monostatic_label_pairs, sensing_role_label

    pairs = monostatic_label_pairs([[1e6, 0, 1], [5, 0, 1]], [[5, 0, 1], [1e6, 0, 1]])
    assert pairs == {0: 1, 1: 0}
    assert sensing_role_label("tx", 0, pairs) == "Monostatic radar 1"
    assert sensing_role_label("rx", 1, pairs) == "Monostatic radar 1"
    assert sensing_role_label("rx", 2, pairs) == "Sensing receiver 3"
    # A relative tolerance at geographic coordinates would merge these wrongly.
    assert monostatic_label_pairs([[1e6, 0, 1]], [[1e6 + 0.1, 0, 1]]) == {}
    assert monostatic_label_pairs([[0, 0, 0]], [[0, 0, 0], [0, 0, 0]]) == {}
    assert monostatic_label_pairs([[float("nan"), 0, 0]], [[float("nan"), 0, 0]]) == {}
    # Summary grouping checks the complete trajectory, not just a crossing.
    assert monostatic_label_pairs([[[0, 0, 0], [1, 0, 0]]], [[[0, 0, 0], [2, 0, 0]]]) == {}


def test_named_sensing_pair_labels_keep_explicit_names_and_generic_fallback():
    from shared.node_labels import sensing_role_label

    pairs = {0: 1}
    assert (
        sensing_role_label("tx", 0, pairs, device_names=("HarborTX", "HarborRX"))
        == "Harbor (monostatic radar)"
    )
    assert (
        sensing_role_label("rx", 1, pairs, device_names=("Harbor_TX", "Harbor_RX"))
        == "Harbor (monostatic radar)"
    )
    assert (
        sensing_role_label("tx", 0, pairs, device_names=("Beacon", "Array"))
        == "Beacon / Array (monostatic radar)"
    )
    assert sensing_role_label("tx", 0, pairs, device_names=("TX1", "RX2")) == "Monostatic radar 1"
    assert sensing_role_label("tx", 0, pairs, device_names=("", "Array")) == "Monostatic radar 1"
