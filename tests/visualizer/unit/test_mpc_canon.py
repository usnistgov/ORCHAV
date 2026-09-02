"""Unit tests for MPC canonicalization (mpc_canon.py)."""

import numpy as np

from visualizer.src.metrics.mpc_canon import (
    CanonicalStepData,
    build_filter_mask,
    colorize,
    colorize_segments,
    ensure_luts,
    normalize_angle_deg,
    remap_lines,
)


class TestMPCFiltering:
    """Test MPC filtering by order, type, selection, and materials."""

    def test_build_filter_mask_all_allowed(self):
        """All segments pass when orders and types are unrestricted."""
        canon = CanonicalStepData(
            points=np.zeros((8, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int32),
            order=np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32),
            itype=np.ones(8, dtype=np.uint8),
            delay=np.zeros(8, dtype=np.float32),
            loss=np.zeros(8, dtype=np.float32),
        )

        _, segment_mask = build_filter_mask(canon, allowed_orders=[0, 1, 2, 3], allowed_types=[1])

        assert segment_mask.tolist() == [True, True, True, True]

    def test_build_filter_mask_order_filtering(self):
        """Only segments within allowed reflection orders remain."""
        canon = CanonicalStepData(
            points=np.zeros((8, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int32),
            order=np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32),
            itype=np.ones(8, dtype=np.uint8),
            delay=np.zeros(8, dtype=np.float32),
            loss=np.zeros(8, dtype=np.float32),
        )

        _, segment_mask = build_filter_mask(canon, allowed_orders=[0, 1], allowed_types=[1])

        assert segment_mask.tolist() == [True, True, False, False]

    def test_build_filter_mask_sparse_order_selection_stays_filtered(self):
        """Sparse order selections must not trigger the all-true fast path."""
        canon = CanonicalStepData(
            points=np.zeros((6, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
            order=np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
            itype=np.ones(6, dtype=np.uint8),
            delay=np.zeros(6, dtype=np.float32),
            loss=np.zeros(6, dtype=np.float32),
        )

        _, segment_mask = build_filter_mask(canon, allowed_orders=[0, 2], allowed_types=[1])

        assert segment_mask.tolist() == [True, False, True]

    def test_build_filter_mask_type_filtering(self):
        """Interaction type filtering should restrict segments correctly."""
        canon = CanonicalStepData(
            points=np.zeros((8, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int32),
            order=np.zeros(8, dtype=np.int32),
            itype=np.array([1, 1, 2, 2, 4, 4, 8, 8], dtype=np.uint8),
            delay=np.zeros(8, dtype=np.float32),
            loss=np.zeros(8, dtype=np.float32),
        )

        _, segment_mask = build_filter_mask(canon, allowed_orders=[0], allowed_types=[1, 2])

        assert segment_mask.tolist() == [True, True, False, False]

    def test_build_filter_mask_sparse_type_selection_stays_filtered(self):
        """Sparse type selections must not trigger the all-true fast path."""
        canon = CanonicalStepData(
            points=np.zeros((8, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int32),
            order=np.zeros(8, dtype=np.int32),
            itype=np.array([0, 0, 1, 1, 2, 2, 4, 4], dtype=np.uint8),
            delay=np.zeros(8, dtype=np.float32),
            loss=np.zeros(8, dtype=np.float32),
        )

        _, segment_mask = build_filter_mask(canon, allowed_orders=[0], allowed_types=[0, 4])

        assert segment_mask.tolist() == [True, False, False, True]

    def test_build_filter_mask_virtual_type_does_not_reenable_filtered_types(self):
        """Type 99 must not make the all-types fast path treat 1/2 as allowed."""
        canon = CanonicalStepData(
            points=np.zeros((6, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
            order=np.zeros(6, dtype=np.int32),
            itype=np.array([0, 0, 1, 1, 2, 2], dtype=np.uint8),
            delay=np.zeros(6, dtype=np.float32),
            loss=np.zeros(6, dtype=np.float32),
        )

        _, no_specular = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[0, 2, 4, 8, 99],
        )
        assert no_specular.tolist() == [True, False, True]

        _, no_diffuse = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[0, 1, 4, 8, 99],
        )
        assert no_diffuse.tolist() == [True, True, False]

    def test_type_zero_does_not_keep_non_los_canonical_paths(self):
        """LoS/type-0 filtering must not preserve TX segments from non-LoS paths."""
        canon = CanonicalStepData(
            points=np.zeros((8, 3), dtype=np.float32),
            lines=np.array(
                [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7]],
                dtype=np.int32,
            ),
            order=np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=np.uint8),
            itype=np.array([0, 0, 1, 1, 1, 8, 8, 8], dtype=np.uint8),
            delay=np.zeros(8, dtype=np.float32),
            loss=np.zeros(8, dtype=np.float32),
            segment_order=np.array([0, 1, 1, 1, 1], dtype=np.uint8),
            segment_itype=np.array([0, 1, 1, 8, 8], dtype=np.uint8),
            segment_path_id=np.array([0, 1, 1, 2, 2], dtype=np.int32),
        )

        np.testing.assert_array_equal(
            canon.segment_itype,
            np.array([0, 1, 1, 8, 8], dtype=np.uint8),
        )

        _, no_specular = build_filter_mask(
            canon,
            allowed_orders=[0, 1],
            allowed_types=[0, 2, 4, 8],
        )
        np.testing.assert_array_equal(
            canon.segment_path_id[no_specular],
            np.array([0, 2, 2], dtype=np.int32),
        )

        _, no_diffraction = build_filter_mask(
            canon,
            allowed_orders=[0, 1],
            allowed_types=[0, 1, 2, 4],
        )
        np.testing.assert_array_equal(
            canon.segment_path_id[no_diffraction],
            np.array([0, 1, 1], dtype=np.int32),
        )

        _, only_los = build_filter_mask(
            canon,
            allowed_orders=[0, 1],
            allowed_types=[0],
        )
        np.testing.assert_array_equal(
            canon.segment_path_id[only_los],
            np.array([0], dtype=np.int32),
        )

    def test_build_filter_mask_combined(self):
        """Order and type filters can be combined."""
        canon = CanonicalStepData(
            points=np.zeros((4, 3)),
            lines=np.array([[0, 1], [2, 3]], dtype=np.int32),
            order=np.array([0, 0, 1, 1], dtype=np.int32),
            itype=np.array([1, 1, 2, 2], dtype=np.uint8),
            delay=np.zeros(4, dtype=np.float32),
            loss=np.zeros(4, dtype=np.float32),
        )

        _, segment_mask = build_filter_mask(canon, allowed_orders=[0], allowed_types=[1])

        assert segment_mask.tolist() == [True, False]

    def test_build_filter_mask_tx_rx_selection(self):
        """Selecting a specific TX/RX pair should limit survivable points."""
        canon = CanonicalStepData(
            points=np.zeros((4, 3)),
            lines=np.array([[0, 1], [2, 3]], dtype=np.int32),
            order=np.zeros(4, dtype=np.int32),
            itype=np.ones(4, dtype=np.uint8),
            delay=np.zeros(4, dtype=np.float32),
            loss=np.zeros(4, dtype=np.float32),
            tx_id=np.array([0, 0, 1, 1], dtype=np.int16),
            rx_id=np.array([0, 0, 1, 1], dtype=np.int16),
        )

        point_mask, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[1],
            selected_tx=0,
            selected_rx="all",
        )

        assert point_mask.tolist() == [True, True, False, False]
        assert segment_mask.tolist() == [True, False]

    def test_build_filter_mask_material_filtering(self):
        """Material filters should restrict segments based on the start node material."""
        canon = CanonicalStepData(
            points=np.zeros((6, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
            order=np.zeros(6, dtype=np.int32),
            itype=np.ones(6, dtype=np.uint8),
            delay=np.zeros(6, dtype=np.float32),
            loss=np.zeros(6, dtype=np.float32),
            material_names=np.array(["glass", "rx", "wood", "rx", "metal", "rx"], dtype=object),
        )

        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[1],
            allowed_materials=["glass"],
        )

        assert segment_mask.tolist() == [True, False, False]

    def test_build_filter_mask_tx_segments_allowed_with_no_material(self):
        """Segments starting at TX nodes should be included when 'no-material' is allowed."""
        canon = CanonicalStepData(
            points=np.zeros((4, 3)),
            lines=np.array([[0, 1], [2, 3]], dtype=np.int32),
            order=np.zeros(4, dtype=np.int32),
            itype=np.array([0, 0, 1, 1], dtype=np.uint8),
            delay=np.zeros(4, dtype=np.float32),
            loss=np.zeros(4, dtype=np.float32),
            material_names=np.array(["", "", "glass", "glass"], dtype=object),
        )

        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[0, 1],
            allowed_materials=["no-material"],
            show_tx_segments=True,
        )

        assert segment_mask.tolist() == [True, False]

    def test_build_filter_mask_all_material_ids_plus_no_material_is_noop(self):
        """Selecting every known material plus no-material should keep all segments."""
        canon = CanonicalStepData(
            points=np.zeros((4, 3)),
            lines=np.array([[0, 1], [2, 3]], dtype=np.int32),
            order=np.zeros(4, dtype=np.int32),
            itype=np.array([0, 0, 1, 1], dtype=np.uint8),
            delay=np.zeros(4, dtype=np.float32),
            loss=np.zeros(4, dtype=np.float32),
            material_ids=np.array([0, 0, 1, 1], dtype=np.int16),
            material_id_to_name={0: "", 1: "glass"},
            material_id_to_bare={0: "", 1: "glass"},
            material_id_to_itu={0: "", 1: "glass"},
        )

        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[0, 1],
            allowed_materials=["glass", "no-material"],
            show_tx_segments=True,
        )

        assert segment_mask.tolist() == [True, True]

    def test_build_filter_mask_all_material_ids_without_no_material_hides_tx(self):
        """Selecting every known material without no-material should only hide TX segments."""
        canon = CanonicalStepData(
            points=np.zeros((4, 3)),
            lines=np.array([[0, 1], [2, 3]], dtype=np.int32),
            order=np.zeros(4, dtype=np.int32),
            itype=np.array([0, 0, 1, 1], dtype=np.uint8),
            delay=np.zeros(4, dtype=np.float32),
            loss=np.zeros(4, dtype=np.float32),
            material_ids=np.array([0, 0, 1, 1], dtype=np.int16),
            material_id_to_name={0: "", 1: "glass"},
            material_id_to_bare={0: "", 1: "glass"},
            material_id_to_itu={0: "", 1: "glass"},
        )

        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[0, 1],
            allowed_materials=["glass"],
            show_tx_segments=False,
        )

        assert segment_mask.tolist() == [False, True]

    def test_build_filter_mask_hide_no_material_with_selection_filter(self):
        """ID-backed hide-no-material mode must still work when fast-path early return is disabled."""
        canon = CanonicalStepData(
            points=np.zeros((4, 3)),
            lines=np.array([[0, 1], [2, 3]], dtype=np.int32),
            order=np.zeros(4, dtype=np.int32),
            itype=np.array([0, 0, 1, 1], dtype=np.uint8),
            delay=np.zeros(4, dtype=np.float32),
            loss=np.zeros(4, dtype=np.float32),
            tx_id=np.zeros(4, dtype=np.int16),
            material_ids=np.array([0, 0, 1, 1], dtype=np.int16),
            material_id_to_name={0: "", 1: "glass"},
            material_id_to_bare={0: "", 1: "glass"},
            material_id_to_itu={0: "", 1: "glass"},
        )

        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[0, 1],
            selected_tx=0,
            allowed_materials=["glass"],
            show_tx_segments=False,
        )

        assert segment_mask.tolist() == [False, True]


class TestLineRemapping:
    """Test line index remapping after filtering."""

    def test_remap_lines_basic(self):
        """Test basic line remapping."""
        lines = np.array([[0, 1], [1, 2], [2, 3]])
        mask = np.array([True, True, False, True])  # Remove point 2

        remapped = remap_lines(lines, mask)

        # Only the first segment survives because both endpoints are kept
        expected = np.array([[0, 1]], dtype=np.int32)
        np.testing.assert_array_equal(remapped, expected)

    def test_remap_lines_all_valid(self):
        """Test remapping when all points are kept."""
        lines = np.array([[0, 1], [1, 2], [2, 3]])
        mask = np.array([True, True, True, True])

        remapped = remap_lines(lines, mask)

        np.testing.assert_array_equal(remapped, lines)

    def test_remap_lines_empty_result(self):
        """Test remapping when no points are kept."""
        lines = np.array([[0, 1], [1, 2]])
        mask = np.array([False, False, False])

        remapped = remap_lines(lines, mask)

        # All lines should become invalid
        assert np.all(remapped == -1)


class TestMPCColorComputation:
    """Test MPC color computation for different modes."""

    def test_ensure_luts(self):
        """Test that color LUTs are created."""
        viridis = ensure_luts()

        assert isinstance(viridis, np.ndarray)
        assert viridis.shape == (256, 3)
        assert viridis.dtype == np.float32
        assert np.all((viridis >= 0) & (viridis <= 1))

    def test_colorize_by_reflection_order(self):
        """Test coloring MPCs by reflection order."""
        canon = CanonicalStepData(
            points=np.zeros((5, 3)),
            lines=np.array([[0, 1]]),
            order=np.array([0, 1, 2, 0, 1], dtype=np.int32),
            itype=np.ones(5, dtype=np.uint8),
            delay=np.zeros(5, dtype=np.float32),
            loss=np.zeros(5, dtype=np.float32),
        )

        mask = np.ones(5, dtype=bool)
        order_palette = np.array(
            [[0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32
        )  # Green, yellow, red
        type_palette = np.zeros((10, 3), dtype=np.float32)
        viridis = ensure_luts()

        colors = colorize(canon, mask, "reflection_order", order_palette, type_palette, viridis)

        assert colors.shape == (5, 3)
        assert colors.dtype == np.float32
        # Order 0 should be green
        np.testing.assert_array_almost_equal(colors[0], [0, 1, 0])
        # Order 1 should be yellow
        np.testing.assert_array_almost_equal(colors[1], [1, 1, 0])

    def test_mpc_type_colors_preserve_virtual_and_unknown_for_points_and_segments(self):
        """Point and segment colorization share canonical type semantics."""
        interaction_types = np.array([0, 1, 2, 4, 8, 99, 255], dtype=np.uint8)
        canon = CanonicalStepData(
            points=np.zeros((7, 3), dtype=np.float32),
            lines=np.column_stack(
                (
                    np.arange(7, dtype=np.int32),
                    np.arange(7, dtype=np.int32),
                )
            ),
            order=np.zeros(7, dtype=np.uint8),
            itype=interaction_types,
            delay=np.zeros(7, dtype=np.float32),
            loss=np.zeros(7, dtype=np.float32),
            segment_itype=interaction_types,
        )
        type_palette = np.array(
            [[index / 10.0, 0.1, 0.2] for index in range(9)],
            dtype=np.float32,
        )
        mask = np.ones(7, dtype=bool)

        point_colors = colorize(
            canon,
            mask,
            "mpc_type",
            np.zeros((1, 3), dtype=np.float32),
            type_palette,
            ensure_luts(),
        )
        segment_colors = colorize_segments(
            canon,
            mask,
            "mpc_type",
            np.zeros((1, 3), dtype=np.float32),
            type_palette,
            ensure_luts(),
        )

        expected = np.vstack(
            (
                type_palette[0],
                type_palette[1],
                type_palette[2],
                type_palette[4],
                type_palette[5],
                np.array([1.0, 0.6, 0.2], dtype=np.float32),
                np.array([0.5, 0.5, 0.5], dtype=np.float32),
            )
        )
        np.testing.assert_allclose(point_colors, expected)
        np.testing.assert_allclose(segment_colors, expected)

    def test_colorize_by_delay(self):
        """Test coloring MPCs by delay."""
        canon = CanonicalStepData(
            points=np.zeros((3, 3)),
            lines=np.array([[0, 1]]),
            order=np.zeros(3, dtype=np.int32),
            itype=np.ones(3, dtype=np.uint8),
            delay=np.array([0.0, 0.5, 1.0], dtype=np.float32),
            loss=np.zeros(3, dtype=np.float32),
            delay_min=0.0,
            delay_max=1.0,
        )

        mask = np.ones(3, dtype=bool)
        order_palette = np.zeros((5, 3), dtype=np.float32)
        type_palette = np.zeros((10, 3), dtype=np.float32)
        viridis = ensure_luts()

        colors = colorize(canon, mask, "delay", order_palette, type_palette, viridis)

        assert colors.shape == (3, 3)
        assert colors.dtype == np.float32
        # Colors should vary based on delay
        assert not np.array_equal(colors[0], colors[2])

    def test_colorize_by_path_loss(self):
        """Test coloring MPCs by path loss."""
        canon = CanonicalStepData(
            points=np.zeros((3, 3)),
            lines=np.array([[0, 1]]),
            order=np.zeros(3, dtype=np.int32),
            itype=np.ones(3, dtype=np.uint8),
            delay=np.zeros(3, dtype=np.float32),
            loss=np.array([0.0, 50.0, 100.0], dtype=np.float32),
            loss_min=0.0,
            loss_max=100.0,
        )

        mask = np.ones(3, dtype=bool)
        order_palette = np.zeros((5, 3), dtype=np.float32)
        type_palette = np.zeros((10, 3), dtype=np.float32)
        viridis = ensure_luts()

        colors = colorize(canon, mask, "path_loss", order_palette, type_palette, viridis)

        assert colors.shape == (3, 3)
        assert colors.dtype == np.float32

    def test_colorize_material_mode(self):
        """Material colorization uses provided palettes and handles missing materials."""
        canon = CanonicalStepData(
            points=np.zeros((4, 3)),
            lines=np.array([[0, 1], [2, 3]], dtype=np.int32),
            order=np.zeros(4, dtype=np.int32),
            itype=np.zeros(4, dtype=np.uint8),
            delay=np.zeros(4, dtype=np.float32),
            loss=np.zeros(4, dtype=np.float32),
            material_names=np.array(["", "mat-itu_glass", "mat-itu_steel", ""], dtype=object),
        )

        mask = np.ones(4, dtype=bool)
        order_palette = np.zeros((1, 3), dtype=np.float32)
        type_palette = np.zeros((10, 3), dtype=np.float32)
        viridis = ensure_luts()
        material_colors = {
            "mat-itu_glass": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "mat-itu_steel": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }

        colors = colorize(
            canon,
            mask,
            "material",
            order_palette,
            type_palette,
            viridis,
            material_colors=material_colors,
        )

        np.testing.assert_array_almost_equal(colors[1], [1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(colors[2], [0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(colors[0], [0.0, 0.0, 0.0])

    def test_colorize_material_id_based_with_target_suffix(self):
        """Material ID coloring matches target materials via ITU-type fallback.

        When HDF5 stores 'mat-itu_glass_drone' but the palette only has
        'mat-itu_glass', the ITU-type ('glass') is used for matching.
        """
        # Points: TX, bounce (building), bounce (target), RX
        canon = CanonicalStepData(
            points=np.zeros((4, 3)),
            lines=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32),
            order=np.array([0, 1, 1, 0], dtype=np.uint8),
            itype=np.array([0, 1, 1, 0], dtype=np.uint8),
            delay=np.zeros(4, dtype=np.float32),
            loss=np.zeros(4, dtype=np.float32),
            # ID-based material data
            material_ids=np.array([0, 1, 2, 0], dtype=np.int16),
            material_id_to_name={0: "", 1: "mat-itu_concrete", 2: "mat-itu_glass_drone"},
            material_id_to_itu={0: "", 1: "concrete", 2: "glass"},
            material_id_to_bare={0: "", 1: "concrete", 2: "glass_drone"},
        )

        mask = np.ones(4, dtype=bool)
        viridis = ensure_luts()
        # Palette has stripped names (no target suffix)
        material_colors = {
            "mat-itu_concrete": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "mat-itu_glass": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }

        colors = colorize(
            canon,
            mask,
            "material",
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((10, 3), dtype=np.float32),
            viridis,
            material_colors=material_colors,
        )

        # Building bounce (ID 1 = concrete) should be red
        np.testing.assert_array_almost_equal(colors[1], [1.0, 0.0, 0.0])
        # Target bounce (ID 2 = glass_drone → ITU fallback → glass) should be blue
        np.testing.assert_array_almost_equal(colors[2], [0.0, 0.0, 1.0])
        # TX start (ID 0, is_start=True) → black (intentional for path starts)
        np.testing.assert_array_almost_equal(colors[0], [0.0, 0.0, 0.0])

    def test_material_color_aliases_do_not_override_exact_frame_labels(self):
        """Exact frame material colors should win over scene-material aliases."""
        canon = CanonicalStepData(
            points=np.zeros((3, 3), dtype=np.float32),
            lines=np.array([[0, 1], [1, 2]], dtype=np.int32),
            order=np.array([0, 1, 0], dtype=np.uint8),
            itype=np.array([0, 1, 0], dtype=np.uint8),
            delay=np.zeros(3, dtype=np.float32),
            loss=np.zeros(3, dtype=np.float32),
            material_ids=np.array([0, 1, 0], dtype=np.int16),
            material_id_to_name={0: "", 1: "ground_asphalt"},
            material_id_to_itu={0: "", 1: "concrete"},
            material_id_to_bare={0: "", 1: "asphalt"},
            segment_material_ids=np.array([0, 1], dtype=np.int16),
        )
        viridis = ensure_luts()
        asphalt_blue = np.array([0.1, 0.2, 0.9], dtype=np.float32)
        scene_brown = np.array([0.55, 0.34, 0.29], dtype=np.float32)
        material_colors = {
            "mat-ground_asphalt": scene_brown,
            "ground_asphalt": asphalt_blue,
        }

        point_colors = colorize(
            canon,
            np.ones(3, dtype=bool),
            "material",
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((10, 3), dtype=np.float32),
            viridis,
            material_colors=material_colors,
        )
        segment_colors = colorize_segments(
            canon,
            np.ones(2, dtype=bool),
            "material",
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((10, 3), dtype=np.float32),
            viridis,
            material_colors=material_colors,
        )

        np.testing.assert_array_almost_equal(point_colors[1], asphalt_blue)
        np.testing.assert_array_almost_equal(segment_colors[1], asphalt_blue)

    def test_target_suffix_prefers_itu_family_over_exact_visual_color(self):
        """Target-specific material IDs should use EM family colors for MPCs."""
        canon = CanonicalStepData(
            points=np.zeros((2, 3)),
            lines=np.array([[0, 1]], dtype=np.int32),
            order=np.array([1, 0], dtype=np.uint8),
            itype=np.array([1, 0], dtype=np.uint8),
            delay=np.zeros(2, dtype=np.float32),
            loss=np.zeros(2, dtype=np.float32),
            material_ids=np.array([1, 0], dtype=np.int16),
            material_id_to_name={0: "", 1: "itu_glass_Pedestrian"},
            material_id_to_itu={0: "", 1: "glass"},
            material_id_to_bare={0: "", 1: "glass_pedestrian"},
            segment_material_ids=np.array([1], dtype=np.int16),
        )
        viridis = ensure_luts()
        glass_blue = np.array([0.1, 0.2, 0.8], dtype=np.float32)
        skin_color = np.array([0.82, 0.62, 0.49], dtype=np.float32)
        material_colors = {
            "glass": glass_blue,
            "mat-itu_glass_Pedestrian": skin_color,
        }

        point_colors = colorize(
            canon,
            np.array([True, True]),
            "material",
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((10, 3), dtype=np.float32),
            viridis,
            material_colors=material_colors,
        )
        segment_colors = colorize_segments(
            canon,
            np.array([True]),
            "material",
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((10, 3), dtype=np.float32),
            viridis,
            material_colors=material_colors,
        )

        np.testing.assert_array_almost_equal(point_colors[0], glass_blue)
        np.testing.assert_array_almost_equal(segment_colors[0], glass_blue)

    def test_colorize_material_str_based_with_target_suffix(self):
        """String-based material coloring matches target materials via ITU fallback."""
        canon = CanonicalStepData(
            points=np.zeros((4, 3)),
            lines=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32),
            order=np.array([0, 1, 1, 0], dtype=np.uint8),
            itype=np.array([0, 1, 1, 0], dtype=np.uint8),
            delay=np.zeros(4, dtype=np.float32),
            loss=np.zeros(4, dtype=np.float32),
            # String-only material data (no IDs)
            material_names=np.array(
                ["", "mat-itu_concrete", "mat-itu_glass_drone", ""], dtype=object
            ),
            material_itu_types=np.array(["", "concrete", "glass", ""], dtype=object),
        )

        mask = np.ones(4, dtype=bool)
        viridis = ensure_luts()
        material_colors = {
            "mat-itu_concrete": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "mat-itu_glass": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }

        colors = colorize(
            canon,
            mask,
            "material",
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((10, 3), dtype=np.float32),
            viridis,
            material_colors=material_colors,
        )

        # Building bounce should be red
        np.testing.assert_array_almost_equal(colors[1], [1.0, 0.0, 0.0])
        # Target bounce should be blue (via ITU-type fallback)
        np.testing.assert_array_almost_equal(colors[2], [0.0, 0.0, 1.0])


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_canonical_data(self):
        """Test handling of empty canonical data."""
        canon = CanonicalStepData(
            points=np.zeros((0, 3)),
            lines=np.zeros((0, 2), dtype=np.int32),
            order=np.zeros(0, dtype=np.int32),
            itype=np.zeros(0, dtype=np.uint8),
            delay=np.zeros(0, dtype=np.float32),
            loss=np.zeros(0, dtype=np.float32),
        )

        point_mask, segment_mask = build_filter_mask(canon, allowed_orders=[0], allowed_types=[1])

        assert len(point_mask) == 0
        assert len(segment_mask) == 0
        assert point_mask.dtype == bool
        assert segment_mask.dtype == bool

    def test_filter_with_empty_allowed_sets(self):
        """Test filtering with empty allowed sets."""
        canon = CanonicalStepData(
            points=np.zeros((5, 3)),
            lines=np.array([[0, 1], [2, 3]], dtype=np.int32),
            order=np.array([0, 1, 2, 0, 1], dtype=np.int32),
            itype=np.ones(5, dtype=np.uint8),
            delay=np.zeros(5, dtype=np.float32),
            loss=np.zeros(5, dtype=np.float32),
        )

        # Empty allowed sets should filter everything out
        point_mask, segment_mask = build_filter_mask(canon, allowed_orders=[], allowed_types=[])

        assert not segment_mask.any()  # All segments should be filtered out


class TestRangeFiltering:
    """Test range filtering for delay, power, and angles."""

    def test_delay_range_filter_min(self):
        """Test filtering by minimum delay."""
        canon = CanonicalStepData(
            points=np.zeros((6, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
            order=np.zeros(6, dtype=np.int32),
            itype=np.ones(6, dtype=np.uint8),
            delay=np.array([10.0, 10.0, 50.0, 50.0, 100.0, 100.0], dtype=np.float32),
            loss=np.zeros(6, dtype=np.float32),
        )

        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[1],
            delay_min_ns=40.0,
        )

        # Only segments with delay >= 40 should pass
        assert segment_mask.tolist() == [False, True, True]

    def test_delay_range_filter_max(self):
        """Test filtering by maximum delay."""
        canon = CanonicalStepData(
            points=np.zeros((6, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
            order=np.zeros(6, dtype=np.int32),
            itype=np.ones(6, dtype=np.uint8),
            delay=np.array([10.0, 10.0, 50.0, 50.0, 100.0, 100.0], dtype=np.float32),
            loss=np.zeros(6, dtype=np.float32),
        )

        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[1],
            delay_max_ns=60.0,
        )

        # Only segments with delay <= 60 should pass
        assert segment_mask.tolist() == [True, True, False]

    def test_delay_range_filter_both(self):
        """Test filtering by both min and max delay."""
        canon = CanonicalStepData(
            points=np.zeros((6, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
            order=np.zeros(6, dtype=np.int32),
            itype=np.ones(6, dtype=np.uint8),
            delay=np.array([10.0, 10.0, 50.0, 50.0, 100.0, 100.0], dtype=np.float32),
            loss=np.zeros(6, dtype=np.float32),
        )

        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[1],
            delay_min_ns=40.0,
            delay_max_ns=60.0,
        )

        # Only segments with 40 <= delay <= 60 should pass
        assert segment_mask.tolist() == [False, True, False]

    def test_power_range_filter(self):
        """Test filtering by power (path loss) range."""
        canon = CanonicalStepData(
            points=np.zeros((6, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
            order=np.zeros(6, dtype=np.int32),
            itype=np.ones(6, dtype=np.uint8),
            delay=np.zeros(6, dtype=np.float32),
            loss=np.array([30.0, 30.0, 60.0, 60.0, 90.0, 90.0], dtype=np.float32),
        )

        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[1],
            power_min_db=50.0,
            power_max_db=70.0,
        )

        # Only segments with 50 <= loss <= 70 should pass
        assert segment_mask.tolist() == [False, True, False]

    def test_aoa_azimuth_range_filter(self):
        """Test filtering by AoA azimuth range."""
        canon = CanonicalStepData(
            points=np.zeros((6, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
            order=np.zeros(6, dtype=np.int32),
            itype=np.ones(6, dtype=np.uint8),
            delay=np.zeros(6, dtype=np.float32),
            loss=np.zeros(6, dtype=np.float32),
            aoa_az=np.array([45.0, 45.0, 135.0, 135.0, 270.0, 270.0], dtype=np.float32),
            aoa_el=np.zeros(6, dtype=np.float32),
            aod_az=np.zeros(6, dtype=np.float32),
            aod_el=np.zeros(6, dtype=np.float32),
        )

        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[1],
            selected_rx=0,
            aoa_az_min_deg=90.0,
            aoa_az_max_deg=180.0,
        )

        # Only segments with 90 <= aoa_az <= 180 should pass
        assert segment_mask.tolist() == [False, True, False]

    def test_angle_filters_are_ignored_without_endpoint_selection(self):
        """AoA/AoD angle filters should not apply while TX/RX selectors are set to All."""
        canon = CanonicalStepData(
            points=np.zeros((6, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
            order=np.zeros(6, dtype=np.int32),
            itype=np.ones(6, dtype=np.uint8),
            delay=np.zeros(6, dtype=np.float32),
            loss=np.zeros(6, dtype=np.float32),
            aoa_az=np.array([45.0, 45.0, 135.0, 135.0, 270.0, 270.0], dtype=np.float32),
            aoa_el=np.zeros(6, dtype=np.float32),
            aod_az=np.array([45.0, 45.0, 135.0, 135.0, 270.0, 270.0], dtype=np.float32),
            aod_el=np.zeros(6, dtype=np.float32),
        )

        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[1],
            aoa_az_min_deg=90.0,
            aoa_az_max_deg=180.0,
            aod_az_min_deg=90.0,
            aod_az_max_deg=180.0,
        )

        assert segment_mask.tolist() == [True, True, True]

    def test_aod_azimuth_range_filter_handles_wraparound(self):
        """AOD azimuth filters should treat 350/360 degrees as -10/0 degrees."""
        canon = CanonicalStepData(
            points=np.zeros((8, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int32),
            order=np.zeros(8, dtype=np.int32),
            itype=np.ones(8, dtype=np.uint8),
            delay=np.zeros(8, dtype=np.float32),
            loss=np.zeros(8, dtype=np.float32),
            aod_az=np.array([350.0, 350.0, 360.0, 360.0, 10.0, 10.0, 35.0, 35.0]),
            aod_el=np.zeros(8, dtype=np.float32),
        )

        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[1],
            selected_tx=0,
            aod_az_min_deg=-20.0,
            aod_az_max_deg=20.0,
        )

        assert segment_mask.tolist() == [True, True, True, False]

    def test_combined_range_and_order_filter(self):
        """Test combining range filters with order filters."""
        canon = CanonicalStepData(
            points=np.zeros((8, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int32),
            order=np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int32),
            itype=np.ones(8, dtype=np.uint8),
            delay=np.array([10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float32),
            loss=np.zeros(8, dtype=np.float32),
        )

        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],  # Only order 0
            allowed_types=[1],
            delay_min_ns=50.0,  # Only delay >= 50
        )

        # Only order 0 AND delay >= 50 should pass
        # Segment 0: order=0, delay=10 -> False (delay too low)
        # Segment 1: order=1, delay=10 -> False (order wrong)
        # Segment 2: order=0, delay=100 -> True
        # Segment 3: order=1, delay=100 -> False (order wrong)
        assert segment_mask.tolist() == [False, False, True, False]

    def test_range_filter_with_none_values(self):
        """Test that None filter values mean no filtering."""
        canon = CanonicalStepData(
            points=np.zeros((6, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
            order=np.zeros(6, dtype=np.int32),
            itype=np.ones(6, dtype=np.uint8),
            delay=np.array([10.0, 10.0, 50.0, 50.0, 100.0, 100.0], dtype=np.float32),
            loss=np.zeros(6, dtype=np.float32),
        )

        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[1],
            delay_min_ns=None,  # No min filter
            delay_max_ns=None,  # No max filter
        )

        # All segments should pass since no range filters applied
        assert segment_mask.tolist() == [True, True, True]

    def test_angle_filter_without_angle_data(self):
        """Test that angle filters are safely ignored when no angle data exists."""
        canon = CanonicalStepData(
            points=np.zeros((4, 3)),
            lines=np.array([[0, 1], [2, 3]], dtype=np.int32),
            order=np.zeros(4, dtype=np.int32),
            itype=np.ones(4, dtype=np.uint8),
            delay=np.zeros(4, dtype=np.float32),
            loss=np.zeros(4, dtype=np.float32),
            # No angle data (aoa_az, etc. are None by default)
        )

        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[1],
            aoa_az_min_deg=90.0,  # Should be ignored since no angle data
            aoa_az_max_deg=180.0,
        )

        # All segments should pass since angle data is not available
        assert segment_mask.tolist() == [True, True]


class TestAngleNormalization:
    """Tests for normalize_angle_deg function."""

    def test_normalize_positive_in_range(self):
        """Angles already in [-180, 180] should not change."""
        angles = np.array([0.0, 45.0, 90.0, -90.0, -180.0, 179.9])
        result = normalize_angle_deg(angles)
        np.testing.assert_allclose(result, angles, atol=1e-6)

    def test_normalize_positive_wrapping(self):
        """Angles > 180 should wrap to negative."""
        angles = np.array([181.0, 270.0, 360.0, 540.0])
        result = normalize_angle_deg(angles)
        expected = np.array([-179.0, -90.0, 0.0, -180.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_normalize_negative_wrapping(self):
        """Angles < -180 should wrap appropriately."""
        angles = np.array([-181.0, -270.0, -360.0, -540.0])
        result = normalize_angle_deg(angles)
        # -540° wraps to -180° (equivalent to 180° but normalized to [-180, 180))
        expected = np.array([179.0, 90.0, 0.0, -180.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_normalize_scalar(self):
        """Should work with scalar input."""
        result = normalize_angle_deg(np.array(270.0))
        np.testing.assert_allclose(result, -90.0, atol=1e-6)


class TestDeviceOrientationFiltering:
    """Tests for angle filtering with device orientation transformation."""

    def test_aoa_azimuth_with_device_orientation(self):
        """Test AoA azimuth filtering transforms angles to device-local coordinates."""
        # Create paths with world-space AoA azimuth values
        canon = CanonicalStepData(
            points=np.zeros((6, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
            order=np.zeros(6, dtype=np.int32),
            itype=np.ones(6, dtype=np.uint8),
            delay=np.zeros(6, dtype=np.float32),
            loss=np.zeros(6, dtype=np.float32),
            # World-space AoA azimuth: 0°, 90°, 180° for three paths
            aoa_az=np.array([0.0, 0.0, 90.0, 90.0, 180.0, 180.0], dtype=np.float32),
            aoa_el=np.zeros(6, dtype=np.float32),
        )

        # Device with 90° yaw (facing world +Y direction)
        # In device-local: world 0° -> local -90°, world 90° -> local 0°, world 180° -> local 90°
        rx_orientation = (np.pi / 2, 0.0, 0.0)  # 90° yaw

        # Filter for local azimuth between -45° and 45° (should pass world 90°)
        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[1],
            selected_rx=0,
            aoa_az_min_deg=-45.0,
            aoa_az_max_deg=45.0,
            aoa_device_orientation=rx_orientation,
        )

        # Only the second segment (world 90° -> local 0°) should pass
        assert segment_mask.tolist() == [False, True, False]

    def test_aod_azimuth_with_device_orientation(self):
        """Test AoD azimuth filtering transforms angles to device-local coordinates."""
        canon = CanonicalStepData(
            points=np.zeros((6, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
            order=np.zeros(6, dtype=np.int32),
            itype=np.ones(6, dtype=np.uint8),
            delay=np.zeros(6, dtype=np.float32),
            loss=np.zeros(6, dtype=np.float32),
            # World-space AoD azimuth
            aod_az=np.array([45.0, 45.0, 135.0, 135.0, 225.0, 225.0], dtype=np.float32),
            aod_el=np.zeros(6, dtype=np.float32),
        )

        # Device with 45° yaw
        # World 45° -> local 0°, world 135° -> local 90°, world 225° -> local 180°
        tx_orientation = (np.pi / 4, 0.0, 0.0)  # 45° yaw

        # Filter for local azimuth between -10° and 10° (should pass world 45°)
        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[1],
            selected_tx=0,
            aod_az_min_deg=-10.0,
            aod_az_max_deg=10.0,
            aod_device_orientation=tx_orientation,
        )

        # Only the first segment (world 45° -> local 0°) should pass
        assert segment_mask.tolist() == [True, False, False]

    def test_no_device_orientation_uses_world_angles(self):
        """Test that without device orientation, world angles are used directly."""
        canon = CanonicalStepData(
            points=np.zeros((4, 3)),
            lines=np.array([[0, 1], [2, 3]], dtype=np.int32),
            order=np.zeros(4, dtype=np.int32),
            itype=np.ones(4, dtype=np.uint8),
            delay=np.zeros(4, dtype=np.float32),
            loss=np.zeros(4, dtype=np.float32),
            aoa_az=np.array([45.0, 45.0, 135.0, 135.0], dtype=np.float32),
            aoa_el=np.zeros(4, dtype=np.float32),
        )

        # Without device orientation, filter on world coordinates
        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[1],
            selected_rx=0,
            aoa_az_min_deg=40.0,
            aoa_az_max_deg=50.0,
            aoa_device_orientation=None,  # No orientation
        )

        # Only the first segment (world 45°) should pass
        assert segment_mask.tolist() == [True, False]

    def test_aoa_elevation_with_device_pitch(self):
        """AoA elevation filtering should use the same pitch convention as aperture geometry."""
        canon = CanonicalStepData(
            points=np.zeros((6, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
            order=np.zeros(6, dtype=np.int32),
            itype=np.ones(6, dtype=np.uint8),
            delay=np.zeros(6, dtype=np.float32),
            loss=np.zeros(6, dtype=np.float32),
            aoa_az=np.zeros(6, dtype=np.float32),
            aoa_el=np.array([-30.0, -30.0, 0.0, 0.0, 30.0, 30.0], dtype=np.float32),
        )

        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[1],
            selected_rx=0,
            aoa_el_min_deg=-5.0,
            aoa_el_max_deg=5.0,
            aoa_device_orientation=(0.0, np.radians(30.0), 0.0),
        )

        assert segment_mask.tolist() == [True, False, False]

    def test_aod_elevation_with_device_pitch(self):
        """AoD elevation filtering should match the local-frame aperture preview."""
        canon = CanonicalStepData(
            points=np.zeros((6, 3)),
            lines=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
            order=np.zeros(6, dtype=np.int32),
            itype=np.ones(6, dtype=np.uint8),
            delay=np.zeros(6, dtype=np.float32),
            loss=np.zeros(6, dtype=np.float32),
            aod_az=np.zeros(6, dtype=np.float32),
            aod_el=np.array([-30.0, -30.0, 0.0, 0.0, 30.0, 30.0], dtype=np.float32),
        )

        _, segment_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[1],
            selected_tx=0,
            aod_el_min_deg=-5.0,
            aod_el_max_deg=5.0,
            aod_device_orientation=(0.0, np.radians(30.0), 0.0),
        )

        assert segment_mask.tolist() == [True, False, False]

    def test_zero_orientation_same_as_no_orientation(self):
        """Test that zero orientation gives same result as no orientation."""
        canon = CanonicalStepData(
            points=np.zeros((4, 3)),
            lines=np.array([[0, 1], [2, 3]], dtype=np.int32),
            order=np.zeros(4, dtype=np.int32),
            itype=np.ones(4, dtype=np.uint8),
            delay=np.zeros(4, dtype=np.float32),
            loss=np.zeros(4, dtype=np.float32),
            aoa_az=np.array([45.0, 45.0, 135.0, 135.0], dtype=np.float32),
            aoa_el=np.zeros(4, dtype=np.float32),
        )

        # With zero orientation
        _, mask_with_zero = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[1],
            selected_rx=0,
            aoa_az_min_deg=40.0,
            aoa_az_max_deg=50.0,
            aoa_device_orientation=(0.0, 0.0, 0.0),
        )

        # Without orientation
        _, mask_without = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[1],
            selected_rx=0,
            aoa_az_min_deg=40.0,
            aoa_az_max_deg=50.0,
            aoa_device_orientation=None,
        )

        assert mask_with_zero.tolist() == mask_without.tolist()
