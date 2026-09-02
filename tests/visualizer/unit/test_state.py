"""Unit tests for AppState (state.py)."""

from dataclasses import FrozenInstanceError, replace

import pytest

from visualizer.src.state import (
    DEFAULT_MPC_ALLOWED_ORDERS,
    DEFAULT_MPC_ALLOWED_TYPES,
    MpcVisibility,
    create_initial_state,
    get_beamforming_state_defaults,
    update_state,
)


class TestAppStateImmutability:
    """Test that AppState is properly immutable."""

    def test_state_is_frozen(self):
        """Test that AppState cannot be modified after creation."""
        state = create_initial_state()
        with pytest.raises(FrozenInstanceError):
            state.step = 5

    def test_state_replace_creates_new_instance(self):
        """Test that replace() creates a new instance."""
        state = create_initial_state(step=0)
        new_state = replace(state, step=5)

        assert new_state is not state
        assert new_state.step == 5
        assert state.step == 0

    def test_update_state_helper(self):
        """Test the update_state helper function."""
        state = create_initial_state(step=0)
        new_state = update_state(
            state,
            step=10,
            mpc_visibility=MpcVisibility(enabled=False),
        )

        assert new_state.step == 10
        assert new_state.mpc_visibility.enabled is False
        assert state.step == 0
        assert state.mpc_visibility.enabled is True


class TestAppStateCreation:
    """Test AppState creation and initialization."""

    def test_create_initial_state(self):
        """Test creating initial state with defaults."""
        state = create_initial_state()

        assert state.step == 0
        assert state.mpc_visibility == MpcVisibility()
        assert state.color_mode == "reflection_order"
        assert state.mpc_allowed_orders == DEFAULT_MPC_ALLOWED_ORDERS
        assert state.mpc_allowed_types == DEFAULT_MPC_ALLOWED_TYPES

    def test_create_state_with_custom_values(self):
        """Test creating state with custom values."""
        state = create_initial_state(
            step=5,
            selected_tx=1,
            selected_rx=2,
            mpc_layer_enabled=False,
            color_mode="delay",
        )

        assert state.step == 5
        assert state.selected_tx == 1
        assert state.selected_rx == 2
        assert state.mpc_visibility.enabled is False
        assert state.color_mode == "delay"

    def test_create_state_with_all_node_selection(self):
        """Test creating state with 'all' node selection."""
        state = create_initial_state(selected_tx="all", selected_rx="all")

        assert state.selected_tx == "all"
        assert state.selected_rx == "all"

    def test_beamforming_all_node_selection_normalizes_to_auto(self):
        """Beam-pattern selectors do not keep an effective All state."""
        state = create_initial_state()
        updated = update_state(state, beamforming_tx_node="all", beamforming_rx_node="all")

        assert updated.beamforming_tx_node == "auto"
        assert updated.beamforming_rx_node == "auto"

    def test_create_state_with_frozenset_filters(self):
        """Test creating state with frozenset filters."""
        orders = frozenset([0, 1, 2])
        types = frozenset([1, 2])

        state = create_initial_state(mpc_allowed_orders=orders, mpc_allowed_types=types)

        assert state.mpc_allowed_orders == orders
        assert state.mpc_allowed_types == types


class TestAppStateValidation:
    """Test AppState validation logic."""

    def test_negative_step_allowed(self):
        """Test that negative steps can be created (no validation yet)."""
        # Note: Validation should be added in the future
        state = create_initial_state(step=-1)
        assert state.step == -1

    def test_color_mode_values(self):
        """Test valid color mode values."""
        valid_modes = ["reflection_order", "mpc_type", "delay", "path_loss"]

        for mode in valid_modes:
            state = create_initial_state(color_mode=mode)
            assert state.color_mode == mode


class TestAppStateEquality:
    """Test AppState equality comparison."""

    def test_equal_states(self):
        """Test that identical states are equal."""
        state1 = create_initial_state(step=0)
        state2 = create_initial_state(step=0)

        assert state1 == state2

    def test_different_states(self):
        """Test that different states are not equal."""
        state1 = create_initial_state(step=0)
        state2 = create_initial_state(step=1)

        assert state1 != state2

    def test_states_with_different_filters(self):
        """Test that states with different filters are not equal."""
        state1 = create_initial_state(mpc_allowed_orders=frozenset([0, 1]))
        state2 = create_initial_state(mpc_allowed_orders=frozenset([0, 2]))

        assert state1 != state2


class TestAppStateFilters:
    """Test MPC filter-related state properties."""

    def test_allowed_orders_default(self):
        """Test default allowed reflection orders."""
        state = create_initial_state()
        assert state.mpc_allowed_orders == DEFAULT_MPC_ALLOWED_ORDERS

    def test_allowed_types_default(self):
        """Test default allowed interaction types."""
        state = create_initial_state()
        assert state.mpc_allowed_types == DEFAULT_MPC_ALLOWED_TYPES
        assert 99 in state.mpc_allowed_types

    def test_custom_filters(self):
        """Test setting custom filter values."""
        state = create_initial_state(
            mpc_allowed_orders=frozenset([0, 1]),
            mpc_allowed_types=frozenset([1, 2]),
        )

        assert state.mpc_allowed_orders == frozenset([0, 1])
        assert state.mpc_allowed_types == frozenset([1, 2])

    def test_filters_with_empty_frozenset(self):
        """Test that explicit empty allow-lists mean no orders or types."""
        state = create_initial_state(
            mpc_allowed_orders=frozenset(),
            mpc_allowed_types=frozenset(),
        )

        assert state.mpc_allowed_orders == frozenset()
        assert state.mpc_allowed_types == frozenset()
        assert len(state.mpc_allowed_orders) == 0
        assert len(state.mpc_allowed_types) == 0


class TestAppStateOptionalFields:
    """Test optional fields with default values."""

    def test_optional_fields_defaults(self):
        """Test that optional fields have correct defaults."""
        state = create_initial_state()

        assert state.fly_mode is False
        assert state.show_beamforming is False
        assert state.show_coverage is False
        assert state.show_rf_xray is False
        assert state.rf_xray_mode == "material_map"
        assert state.rf_xray_property == "scattering_coefficient"
        assert state.rf_xray_opacity == pytest.approx(0.85)
        assert state.rf_xray_show_top_paths is False
        assert state.rf_xray_max_top_paths == 12
        assert state.coverage_height_index == 0
        assert state.topk_render_enabled is False
        assert state.topk_render_max_paths == 20000
        assert state.beamforming_azimuth_samples == 72
        assert state.beamforming_elevation_samples == 37
        assert state.beamforming_tx_scale == 1.5
        assert state.beamforming_rx_scale == 1.5
        assert state.beamforming_tx_node == "auto"
        assert state.beamforming_rx_node == "auto"

    def test_standalone_beamforming_defaults(self):
        """Test standalone beamforming parameter defaults."""
        state = create_initial_state()

        assert state.standalone_beamforming_mode == "standalone"
        assert state.standalone_antenna_rows == 1
        assert state.standalone_antenna_cols == 1
        assert state.standalone_horizontal_spacing_m == pytest.approx(0.00535343675)
        assert state.standalone_vertical_spacing_m == pytest.approx(0.00535343675)
        assert state.standalone_carrier_frequency_ghz == 28.0
        assert state.standalone_steering_strategy == "svd"
        assert state.standalone_azimuth_deg == 0.0
        assert state.standalone_elevation_deg == 0.0
        assert state.beamforming_tx_element_pattern == "isotropic"
        assert state.beamforming_rx_element_pattern == "isotropic"

    def test_custom_optional_fields(self):
        """Test setting custom values for optional fields."""
        state = create_initial_state(
            show_beamforming=True,
            show_coverage=True,
            coverage_height_index=5,
            topk_render_enabled=True,
            topk_render_max_paths=5000,
            beamforming_azimuth_samples=100,
        )

        assert state.show_beamforming is True
        assert state.show_coverage is True
        assert state.coverage_height_index == 5
        assert state.topk_render_enabled is True
        assert state.topk_render_max_paths == 5000
        assert state.beamforming_azimuth_samples == 100

    def test_rf_xray_state_round_trips(self):
        """RF X-Ray controls are regular AppState fields."""
        state = create_initial_state(
            show_rf_xray=True,
            rf_xray_mode="material_properties",
            rf_xray_property="conductivity",
            rf_xray_opacity=0.4,
            rf_xray_show_top_paths=True,
            rf_xray_max_top_paths=4,
        )

        restored = type(state).from_dict(state.to_dict())

        assert restored.show_rf_xray is True
        assert restored.rf_xray_mode == "material_properties"
        assert restored.rf_xray_property == "conductivity"
        assert restored.rf_xray_opacity == pytest.approx(0.4)
        assert restored.rf_xray_show_top_paths is True
        assert restored.rf_xray_max_top_paths == 4

    def test_rf_xray_opacity_is_clamped(self):
        """RF X-Ray opacity stays within the supported UI range."""
        low = create_initial_state(rf_xray_opacity=-1.0)
        high = create_initial_state(rf_xray_opacity=2.0)
        data = create_initial_state().to_dict()
        data["rf_xray_opacity"] = "bad"
        restored = type(low).from_dict(data)

        assert low.rf_xray_opacity == pytest.approx(0.05)
        assert high.rf_xray_opacity == pytest.approx(1.0)
        assert restored.rf_xray_opacity == pytest.approx(0.85)

    def test_beamforming_defaults_merge_scenario_antenna_config(self):
        """Scenario antenna YAML initializes runtime beam controls."""
        defaults = get_beamforming_state_defaults(
            {
                "raytracing": {
                    "carrier_frequency_hz": 60e9,
                    "antenna": {
                        "tx": {"pattern": "tr38901", "num_rows": 4, "num_cols": 8},
                        "rx": {"pattern": "dipole"},
                    },
                }
            }
        )

        assert defaults["show_beamforming"] is False
        assert defaults["standalone_beamforming_mode"] == "standalone"
        assert defaults["standalone_steering_strategy"] == "svd"
        assert defaults["standalone_antenna_rows"] == 4
        assert defaults["standalone_antenna_cols"] == 8
        assert defaults["standalone_carrier_frequency_ghz"] == 60.0
        assert defaults["beamforming_tx_element_pattern"] == "tr38901"
        assert defaults["beamforming_rx_element_pattern"] == "dipole"


class TestAppStateUpdate:
    """Test updating AppState with update_state helper."""

    def test_update_single_field(self):
        """Test updating a single field."""
        state = create_initial_state(step=0)
        new_state = update_state(state, step=5)

        assert new_state.step == 5
        assert new_state.mpc_visibility == state.mpc_visibility

    def test_update_multiple_fields(self):
        """Test updating multiple fields."""
        state = create_initial_state(step=0)
        new_state = update_state(
            state,
            step=10,
            mpc_visibility=MpcVisibility(enabled=False),
            color_mode="delay",
            mpc_allowed_orders=frozenset([0, 1]),
        )

        assert new_state.step == 10
        assert new_state.mpc_visibility.enabled is False
        assert new_state.color_mode == "delay"
        assert new_state.mpc_allowed_orders == frozenset([0, 1])
        assert state.step == 0  # Original unchanged

    def test_update_filters(self):
        """Test updating filter fields."""
        state = create_initial_state()
        new_state = update_state(
            state,
            mpc_allowed_orders=frozenset([0, 1, 2]),
            mpc_allowed_types=frozenset([1, 2, 4]),
        )

        assert new_state.mpc_allowed_orders == frozenset([0, 1, 2])
        assert new_state.mpc_allowed_types == frozenset([1, 2, 4])
        assert state.mpc_allowed_orders == DEFAULT_MPC_ALLOWED_ORDERS  # Original unchanged

    def test_follow_mode_disables_fly_mode(self):
        """Test fly mode is normalized off outside overview mode."""
        state = create_initial_state(fly_mode=True)
        new_state = update_state(state, camera_mode="follow")

        assert new_state.camera_mode == "follow"
        assert new_state.fly_mode is False

    def test_overview_allows_fly_mode(self):
        """Test fly mode can be enabled in overview mode."""
        state = create_initial_state(camera_mode="follow")
        new_state = update_state(state, camera_mode="overview", fly_mode=True)

        assert new_state.camera_mode == "overview"
        assert new_state.fly_mode is True


class TestAppStateTrajectory:
    """Test 3D trajectory visualization state fields."""

    def test_trajectory_defaults_false(self):
        """Test that trajectory toggles default to False."""
        state = create_initial_state()
        assert state.show_tx_trajectory is False
        assert state.show_rx_trajectory is False

    def test_trajectory_toggle_via_update_state(self):
        """Test toggling trajectory fields via update_state."""
        state = create_initial_state()
        new_state = update_state(state, show_tx_trajectory=True, show_rx_trajectory=True)

        assert new_state.show_tx_trajectory is True
        assert new_state.show_rx_trajectory is True
        assert state.show_tx_trajectory is False  # Original unchanged

    def test_trajectory_in_to_dict(self):
        """Test that trajectory fields are serialized in to_dict."""
        state = create_initial_state(show_tx_trajectory=True, show_rx_trajectory=False)
        d = state.to_dict()

        assert d["show_tx_trajectory"] is True
        assert d["show_rx_trajectory"] is False

    def test_trajectory_independent_toggles(self):
        """Test that TX and RX trajectory can be toggled independently."""
        state = create_initial_state()
        tx_only = update_state(state, show_tx_trajectory=True)
        rx_only = update_state(state, show_rx_trajectory=True)

        assert tx_only.show_tx_trajectory is True
        assert tx_only.show_rx_trajectory is False
        assert rx_only.show_tx_trajectory is False
        assert rx_only.show_rx_trajectory is True


class TestAppStateSerialization:
    """Test AppState serialization round-trips."""

    def test_fly_mode_in_to_dict(self):
        state = create_initial_state(fly_mode=True)
        data = state.to_dict()

        assert data["fly_mode"] is True
