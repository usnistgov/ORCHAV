"""Unit tests for AppState serialization (to_dict/from_dict)."""

from visualizer.src.state import AppState, create_initial_state, update_state


class TestAppStateToDict:
    """Test AppState.to_dict() method."""

    def test_to_dict_returns_dict(self):
        """Test that to_dict returns a dictionary."""
        state = create_initial_state()
        result = state.to_dict()

        assert isinstance(result, dict)

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all AppState fields."""
        state = create_initial_state()
        result = state.to_dict()

        # Check for key fields
        assert "step" in result
        assert "selected_tx" in result
        assert "selected_rx" in result
        assert "mpc_visibility" in result
        assert "mpc_allowed_orders" in result
        assert "mpc_allowed_types" in result
        assert "color_mode" in result
        assert "camera_mode" in result
        assert "pov_axis" in result
        assert "show_camera_minimap" in result
        assert "show_beamforming" in result
        assert "show_coverage" in result
        assert "topk_render_enabled" in result
        assert "topk_render_max_paths" in result
        assert "viewport_hud_enabled" in result
        assert "viewport_hud_mode" in result

    def test_to_dict_converts_frozensets_to_lists(self):
        """Test that frozensets are converted to lists for JSON compatibility."""
        state = create_initial_state(
            mpc_allowed_orders=frozenset([0, 1, 2, 3]), mpc_allowed_types=frozenset([0, 1])
        )
        result = state.to_dict()

        assert isinstance(result["mpc_allowed_orders"], list)
        assert isinstance(result["mpc_allowed_types"], list)
        assert set(result["mpc_allowed_orders"]) == {0, 1, 2, 3}
        assert set(result["mpc_allowed_types"]) == {0, 1}

    def test_to_dict_converts_tuples_to_lists(self):
        """Test that tuples are converted to lists for JSON compatibility."""
        state = create_initial_state(pov_hidden_node=("tx", 0))
        result = state.to_dict()

        assert isinstance(result["pov_hidden_node"], list)
        assert result["pov_hidden_node"] == ["tx", 0]

    def test_to_dict_handles_none_values(self):
        """Test that None values are preserved."""
        state = create_initial_state(pov_hidden_node=None)
        result = state.to_dict()

        assert result["pov_hidden_node"] is None

    def test_to_dict_preserves_values(self):
        """Test that to_dict preserves all values correctly."""
        state = create_initial_state(
            step=42,
            selected_tx=1,
            selected_rx="all",
            mpc_layer_enabled=True,
            show_mpc_paths=True,
            show_mpc_bounce_points=False,
            color_mode="path_loss",
            camera_mode="pov",
            pov_axis="forward",
            show_camera_minimap=True,
        )
        result = state.to_dict()

        assert result["step"] == 42
        assert result["selected_tx"] == 1
        assert result["selected_rx"] == "all"
        assert result["mpc_visibility"] == {
            "enabled": True,
            "paths": True,
            "bounce_points": False,
        }
        assert result["color_mode"] == "path_loss"
        assert result["camera_mode"] == "pov"
        assert result["pov_axis"] == "forward"
        assert result["show_camera_minimap"] is True


class TestAppStateFromDict:
    """Test AppState.from_dict() method."""

    def test_from_dict_creates_app_state(self):
        """Test that from_dict creates an AppState instance."""
        data = {
            "step": 42,
            "selected_tx": "all",
            "selected_rx": "all",
            "mpc_visibility": {
                "enabled": True,
                "paths": True,
                "bounce_points": True,
            },
            "mpc_allowed_orders": [0, 1, 2],
            "mpc_allowed_types": [0, 1],
            "color_mode": "reflection_order",
            "show_labels": True,
            "sync_target_position": True,
            "camera_mode": "overview",
            "pov_axis": "forward",
            "pov_hidden_node": None,
            "show_camera_minimap": False,
            "show_beamforming": False,
            "show_coverage": False,
            "coverage_height_index": 0,
            "topk_render_enabled": False,
            "topk_render_max_paths": 20000,
            "beamforming_azimuth_samples": 72,
            "beamforming_elevation_samples": 37,
            "beamforming_tx_scale": 1.5,
            "beamforming_rx_scale": 1.5,
            "beamforming_tx_node": "auto",
            "beamforming_rx_node": "all",
            "standalone_beamforming_mode": "frame",
            "standalone_antenna_rows": 8,
            "standalone_antenna_cols": 8,
            "standalone_horizontal_spacing_m": 0.00536,
            "standalone_vertical_spacing_m": 0.00536,
            "standalone_carrier_frequency_ghz": 28.0,
            "standalone_steering_strategy": "manual",
            "standalone_azimuth_deg": 0.0,
            "standalone_elevation_deg": 0.0,
        }

        result = AppState.from_dict(data)

        assert isinstance(result, AppState)
        assert result.step == 42

    def test_from_dict_converts_lists_to_frozensets(self):
        """Test that lists are converted back to frozensets."""
        data = create_initial_state().to_dict()
        data["mpc_allowed_orders"] = [0, 1, 2, 3]
        data["mpc_allowed_types"] = [0, 1]

        result = AppState.from_dict(data)

        assert isinstance(result.mpc_allowed_orders, frozenset)
        assert isinstance(result.mpc_allowed_types, frozenset)
        assert result.mpc_allowed_orders == frozenset([0, 1, 2, 3])
        assert result.mpc_allowed_types == frozenset([0, 1])

    def test_from_dict_expands_legacy_full_type_set_with_virtual(self):
        """The former all-types set remains unfiltered after Virtual is exposed."""
        data = create_initial_state().to_dict()
        data.pop("mpc_type_filter_schema_version")
        data["mpc_allowed_types"] = [0, 1, 2, 4, 8]

        result = AppState.from_dict(data)

        assert result.mpc_allowed_types == frozenset({0, 1, 2, 4, 8, 99})

        data["mpc_allowed_types"] = []
        assert AppState.from_dict(data).mpc_allowed_types == frozenset()

    def test_current_type_filter_can_explicitly_hide_virtual(self):
        """Current workspaces preserve an explicit Virtual-off filter."""
        state = create_initial_state(
            mpc_allowed_types=frozenset({0, 1, 2, 4, 8}),
        )

        restored = AppState.from_dict(state.to_dict())

        assert restored.mpc_allowed_types == frozenset({0, 1, 2, 4, 8})

    def test_from_dict_converts_lists_to_tuples(self):
        """Test that lists are converted back to tuples where needed."""
        data = create_initial_state().to_dict()
        data["pov_hidden_node"] = ["tx", 0]

        result = AppState.from_dict(data)

        assert isinstance(result.pov_hidden_node, tuple)
        assert result.pov_hidden_node == ("tx", 0)

    def test_from_dict_handles_none_values(self):
        """Test that from_dict handles None values correctly."""
        data = create_initial_state().to_dict()
        data["pov_hidden_node"] = None

        result = AppState.from_dict(data)

        assert result.pov_hidden_node is None

    def test_from_dict_ignores_retired_dash_by_type_setting(self):
        """Workspace snapshots saved before dash removal remain loadable."""
        data = create_initial_state().to_dict()
        data["dash_by_itype"] = True

        result = AppState.from_dict(data)

        assert isinstance(result, AppState)
        assert "dash_by_itype" not in result.to_dict()

    def test_from_dict_migrates_legacy_hud_off_to_separate_master_state(self):
        """Legacy combined HUD state loads hidden with a valid detail mode."""
        data = create_initial_state().to_dict()
        data.pop("viewport_hud_enabled")
        data["viewport_hud_mode"] = "off"

        result = AppState.from_dict(data)

        assert result.viewport_hud_enabled is False
        assert result.viewport_hud_mode == "compact"
        assert result.to_dict()["viewport_hud_mode"] == "compact"

    def test_explicit_hud_enabled_wins_over_legacy_off_mode(self):
        """New workspace master state takes precedence during mixed-version loads."""
        data = create_initial_state().to_dict()
        data["viewport_hud_enabled"] = True
        data["viewport_hud_mode"] = "off"

        result = AppState.from_dict(data)

        assert result.viewport_hud_enabled is True
        assert result.viewport_hud_mode == "compact"


class TestSerializationRoundtrip:
    """Test to_dict/from_dict roundtrip."""

    def test_roundtrip_preserves_state(self):
        """Test that to_dict → from_dict preserves complete state."""
        original = create_initial_state(
            step=42,
            selected_tx=1,
            selected_rx=2,
            mpc_layer_enabled=True,
            show_mpc_paths=False,
            show_mpc_bounce_points=True,
            mpc_allowed_orders=frozenset([0, 1, 2]),
            mpc_allowed_types=frozenset([0, 1]),
            color_mode="delay",
            camera_mode="follow",
            pov_axis="x",
            pov_hidden_node=("rx", 3),
            show_camera_minimap=True,
            show_beamforming=True,
            show_coverage=True,
            coverage_height_index=5,
            topk_render_enabled=True,
            topk_render_max_paths=12345,
        )

        # Roundtrip
        dict_form = original.to_dict()
        restored = AppState.from_dict(dict_form)

        # Verify all fields match
        assert restored.step == original.step
        assert restored.selected_tx == original.selected_tx
        assert restored.selected_rx == original.selected_rx
        assert restored.mpc_visibility == original.mpc_visibility
        assert restored.mpc_allowed_orders == original.mpc_allowed_orders
        assert restored.mpc_allowed_types == original.mpc_allowed_types
        assert restored.color_mode == original.color_mode
        assert restored.camera_mode == original.camera_mode
        assert restored.pov_axis == original.pov_axis
        assert restored.pov_hidden_node == original.pov_hidden_node
        assert restored.show_camera_minimap == original.show_camera_minimap
        assert restored.show_beamforming == original.show_beamforming
        assert restored.show_coverage == original.show_coverage
        assert restored.coverage_height_index == original.coverage_height_index
        assert restored.topk_render_enabled == original.topk_render_enabled
        assert restored.topk_render_max_paths == original.topk_render_max_paths

    def test_roundtrip_with_default_state(self):
        """Test roundtrip with default initial state."""
        original = create_initial_state()

        dict_form = original.to_dict()
        restored = AppState.from_dict(dict_form)

        # Verify key fields match
        assert restored.step == original.step
        assert restored.color_mode == original.color_mode
        assert restored.camera_mode == original.camera_mode
        assert restored.mpc_visibility == original.mpc_visibility

    def test_hud_master_and_detail_roundtrip_independently(self):
        """Hiding the HUD does not discard its selected detail level."""
        original = create_initial_state(
            viewport_hud_enabled=False,
            viewport_hud_mode="detailed",
        )

        restored = AppState.from_dict(original.to_dict())

        assert restored.viewport_hud_enabled is False
        assert restored.viewport_hud_mode == "detailed"

    def test_legacy_update_off_preserves_existing_hud_detail(self):
        """Older callers can hide the HUD without erasing detailed mode."""
        state = create_initial_state(viewport_hud_mode="detailed")

        updated = update_state(state, viewport_hud_mode="off")

        assert updated.viewport_hud_enabled is False
        assert updated.viewport_hud_mode == "detailed"

    def test_roundtrip_with_complex_beamforming_state(self):
        """Test roundtrip with complex standalone beamforming parameters."""
        original = create_initial_state(
            standalone_beamforming_mode="standalone",
            standalone_antenna_rows=16,
            standalone_antenna_cols=16,
            standalone_carrier_frequency_ghz=60.0,
        )

        dict_form = original.to_dict()
        restored = AppState.from_dict(dict_form)

        assert restored.standalone_beamforming_mode == "standalone"
        assert restored.standalone_antenna_rows == 16
        assert restored.standalone_antenna_cols == 16
        assert restored.standalone_carrier_frequency_ghz == 60.0
