"""Unit tests for MPC Panel bulk material selection actions."""

import sys
from unittest.mock import Mock

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel

# Ensure QApplication exists for Qt widgets
if not QApplication.instance():
    app = QApplication(sys.argv)

from visualizer.src.panels.mpc_panel import MPCVisualizationPanel, _get_type_palette_colors
from visualizer.src.renderers.protocol import RendererCapabilities


def _make_panel():
    parent = Mock()
    parent.renderer.capabilities = RendererCapabilities(
        angular_preview=True,
        mpc_type_markers=True,
    )
    panel = MPCVisualizationPanel(parent)
    panel._test_group = panel.create_panel()
    return panel


def _material_rows(panel):
    model = panel.widgets["materials_model"]
    return [
        (model.item(row).text(), model.item(row).checkState() == Qt.Checked)
        for row in range(model.rowCount())
        if model.item(row).isCheckable()
    ]


def _legend_labels(panel):
    return [
        label.text()
        for label in panel.widgets["legend_items_container"].findChildren(QLabel)
        if label.text()
    ]


class TestMPCPanelBulkActions:
    """Tests for bulk material selection functionality."""

    def test_interaction_markers_are_capability_gated(self):
        """Only renderers that implement marker glyphs expose the control."""
        supported = _make_panel()
        assert supported.widgets["mpc_interaction_markers_cb"].isVisibleTo(supported._test_group)

        parent = Mock()
        parent.renderer.capabilities = RendererCapabilities()
        unsupported = MPCVisualizationPanel(parent)
        unsupported._test_group = unsupported.create_panel()

        assert not unsupported.widgets["mpc_interaction_markers_cb"].isVisibleTo(
            unsupported._test_group
        )

    def test_virtual_mpc_type_filter_is_visible(self):
        """Reconstructed paths have the same explicit filter control as native types."""
        panel = _make_panel()

        checkbox = panel.widgets["type_99_cb"]
        assert checkbox.text() == "Virtual"
        assert checkbox.isChecked() is True
        assert checkbox.isVisibleTo(panel._test_group) is True

    def test_mpc_type_legend_exposes_virtual_and_unknown_semantic_colors(self):
        """The control-panel legend uses the canonical special-type colors."""
        colors = _get_type_palette_colors()

        assert len(colors) == 7
        assert colors[-2:] == ["#ff9933", "#808080"]

    def test_mpc_type_legend_tracks_only_types_in_each_accepted_frame(self):
        """Known, virtual, unknown, and empty rows follow the presented tuple."""
        panel = _make_panel()

        panel.set_present_mpc_type_codes((0, 1))
        panel.update_color_legend("mpc_type")
        assert _legend_labels(panel) == ["LoS", "Specular"]

        panel.set_present_mpc_type_codes((1, 99, 42))
        assert _legend_labels(panel) == ["Specular", "Virtual", "Unknown"]

        panel.set_present_mpc_type_codes((8,))
        assert _legend_labels(panel) == ["Diffraction"]

        panel.set_present_mpc_type_codes(())
        assert _legend_labels(panel) == ["No visible path types"]

    def test_select_all_materials(self):
        """Test that 'All' button selects all material checkboxes."""
        panel = _make_panel()

        # Set up materials
        materials = ["concrete", "glass", "metal", "wood"]
        panel.set_materials(materials, checked=set())  # All unchecked initially

        # Verify all unchecked
        rows = _material_rows(panel)
        assert len(rows) == 4
        assert all(not checked for _name, checked in rows)

        # Click "All" button
        panel._on_select_all_materials()

        # Verify all checked
        assert all(checked for _name, checked in _material_rows(panel))

    def test_deselect_all_materials(self):
        """Test that 'None' button deselects all material checkboxes."""
        panel = _make_panel()

        # Set up materials (all checked initially)
        materials = ["concrete", "glass", "metal", "wood"]
        panel.set_materials(materials, checked=set(materials))

        # Verify all checked
        rows = _material_rows(panel)
        assert len(rows) == 4
        assert all(checked for _name, checked in rows)

        # Click "None" button
        panel._on_deselect_all_materials()

        # Verify all unchecked
        assert all(not checked for _name, checked in _material_rows(panel))

    def test_invert_material_selection(self):
        """Test that 'Invert' button inverts all material checkbox states."""
        panel = _make_panel()

        # Set up materials (mix of checked/unchecked)
        materials = ["concrete", "glass", "metal", "wood"]
        panel.set_materials(materials, checked={"concrete", "metal"})

        # Get initial states
        initial_states = [checked for _name, checked in _material_rows(panel)]
        # Expected: [True, False, True, False] (concrete=checked, glass=unchecked, metal=checked, wood=unchecked)
        # Order is sorted: concrete, glass, metal, wood

        # Click "Invert" button
        panel._on_invert_material_selection()

        # Verify states are inverted
        final_states = [checked for _name, checked in _material_rows(panel)]
        assert final_states == [not state for state in initial_states]

    def test_bulk_actions_with_no_materials(self):
        """Test that bulk actions handle empty material list gracefully."""
        panel = _make_panel()

        # Set up no materials
        panel.set_materials([], checked=set())

        # Verify no checkboxes
        assert _material_rows(panel) == []

        # Click all bulk action buttons (should not raise exceptions)
        panel._on_select_all_materials()
        panel._on_deselect_all_materials()
        panel._on_invert_material_selection()

    def test_bulk_actions_with_single_material(self):
        """Test bulk actions work with single material."""
        panel = _make_panel()

        # Set up single material
        materials = ["concrete"]
        panel.set_materials(materials, checked=set())

        assert _material_rows(panel) == [("concrete", False)]

        # Test select all
        panel._on_select_all_materials()
        assert _material_rows(panel) == [("concrete", True)]

        # Test invert
        panel._on_invert_material_selection()
        assert _material_rows(panel) == [("concrete", False)]

        # Test invert again
        panel._on_invert_material_selection()
        assert _material_rows(panel) == [("concrete", True)]

        # Test deselect all
        panel._on_deselect_all_materials()
        assert _material_rows(panel) == [("concrete", False)]

    def test_material_model_rows_stored_correctly(self):
        """Test that material rows are stored in the model."""
        panel = _make_panel()

        # Set up materials
        materials = ["concrete", "glass", "metal"]
        panel.set_materials(materials, checked={"glass"})

        # Verify checkboxes are stored
        assert _material_rows(panel) == [
            ("concrete", False),
            ("glass", True),
            ("metal", False),
        ]

    def test_multiple_bulk_operations(self):
        """Test multiple consecutive bulk operations."""
        panel = _make_panel()

        # Set up materials
        materials = ["concrete", "glass", "metal", "wood", "plastic"]
        panel.set_materials(materials, checked=set())

        # Operation 1: Select all
        panel._on_select_all_materials()
        assert all(checked for _name, checked in _material_rows(panel))

        # Operation 2: Deselect all
        panel._on_deselect_all_materials()
        assert all(not checked for _name, checked in _material_rows(panel))

        # Operation 3: Invert (all should become checked)
        panel._on_invert_material_selection()
        assert all(checked for _name, checked in _material_rows(panel))

        # Operation 4: Invert again (all should become unchecked)
        panel._on_invert_material_selection()
        assert all(not checked for _name, checked in _material_rows(panel))

        # Operation 5: Select all
        panel._on_select_all_materials()
        assert all(checked for _name, checked in _material_rows(panel))

        # Operation 6: Invert (all should become unchecked)
        panel._on_invert_material_selection()
        assert all(not checked for _name, checked in _material_rows(panel))
