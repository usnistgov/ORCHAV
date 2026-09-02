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
