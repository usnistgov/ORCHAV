"""Control-panel assembly entry point for the visualizer app shell.

Startup UI delegates here after services exist. This module creates
``UIPanelManager``, builds all panel widgets, connects them back to
``OrchavVisualizer``, and performs small post-build diagnostics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from PySide6.QtWidgets import QWidget

from shared.logging import get_logger

from .panel_manager import UIPanelManager

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = get_logger("orchav.ui_setup")


def build_control_panel(
    visualizer: OrchavVisualizer, total_steps: int
) -> Tuple[UIPanelManager, QWidget]:
    """Return the panel manager and root control widget for the Qt shell."""
    ui_manager = UIPanelManager(visualizer, total_steps)
    ctrl_panel = ui_manager.create_all_panels()
    ui_manager.connect_widgets_to_parent(visualizer)

    _log_colorbar_widget_state(visualizer)
    _refresh_mpc_panel_legend(ui_manager)

    return ui_manager, ctrl_panel


def _log_colorbar_widget_state(visualizer: OrchavVisualizer) -> None:
    """Log diagnostic details about the connected colorbar widget."""
    if hasattr(visualizer, "colorbar_widget"):
        logger.debug(f"Colorbar widget connected: {visualizer.colorbar_widget}")
        if visualizer.colorbar_widget is None:
            logger.warning("Colorbar widget is None after connection")
            return
        logger.debug(f"Colorbar widget type: {type(visualizer.colorbar_widget)}")
        logger.debug(f"Colorbar widget parent: {visualizer.colorbar_widget.parent()}")
    else:
        logger.warning("Colorbar widget attribute not found after connection")


def _refresh_mpc_panel_legend(ui_manager: UIPanelManager) -> None:
    """Ensure the MPC panel has its legend initialized."""
    mpc_panel = getattr(ui_manager, "panels", {}).get("mpc")
    if mpc_panel:
        mpc_panel.update_color_legend("reflection_order")
        logger.debug("MPC panel initialized and color legend updated")
    else:
        logger.warning("MPC panel not found during initialization")
