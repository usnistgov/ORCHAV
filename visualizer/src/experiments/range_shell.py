"""Selection-driven, display-only range shells owned by the sensing dashboard."""

from __future__ import annotations

import numpy as np

from .range_geometry import build_range_locus


class RangeShellController:
    """Keep geometry tied to one accepted scene and the exact selected product.

    No detector, tracker, source array or camera is changed. Rebuilding a shell
    requires a validated source/recipe/pair/cell and frozen reference pose.
    """

    def __init__(self, viz):
        self.viz = viz
        self.dashboard = None
        self.overlay = None
        self.identity = None
        self.locus = None
        self.accepted_step = None
        self.explanation = None

    def present(self, step):
        self.accepted_step = step
        self.sync()

    def clear(self, reason="No range bin selected"):
        if self.overlay is not None:
            self.overlay.clear()
        self.identity = None
        self.locus = None
        if self.dashboard is not None:
            try:
                self.dashboard.range_shell_status.setText(reason)
                self.dashboard.range_shell_status.setToolTip("")
            except RuntimeError:
                self.dashboard = None

    def sync(self, dashboard=None):
        if dashboard is not None:
            self.dashboard = dashboard
        dashboard = self.dashboard
        if dashboard is None:
            return
        try:
            if hasattr(dashboard, "scene_explanation"):
                if self.explanation is None:
                    from .scene_explanation import SceneExplanationController

                    self.explanation = SceneExplanationController(self.viz)
                self.explanation.sync(dashboard, self.accepted_step)
            if not dashboard.isVisible() or dashboard._last_request is None:
                self.clear("Range shell cleared: sensing view is closed or has no result")
                return
            if not dashboard.range_shell_enabled.isChecked():
                self.clear("Range shell off")
                return
            renderer = getattr(self.viz, "renderer", None)
            if getattr(renderer, "renderer_id", None) != "pygfx":
                self.clear("Range shells require the pygfx renderer")
                return
            request = dashboard._last_request
            selection = dashboard._linked_selection
            if selection is None or selection.cell is None:
                self.clear()
                return
            if request.effective_display_step != self.accepted_step:
                self.clear("Waiting for the selected acquisition's scene to be displayed")
                return
            panel = getattr(getattr(self.viz, "ui_manager", None), "panels", {}).get("sensing")
            session = getattr(panel, "_dashboard_session", None)
            if session is not None and session.source_identity != request.source_identity:
                raise ValueError("Selected result belongs to a retired recording")
            locus = build_range_locus(request, selection)
            cutaway = dashboard.range_shell_cutaway.isChecked()
            identity = (locus.identity, cutaway)
            if identity != self.identity:
                if self.overlay is None:
                    from ..renderers.pygfx.experiments import ExperimentOverlay

                    self.overlay = ExperimentOverlay(renderer)
                self.overlay.range_locus(locus, cutaway=cutaway)
                self.identity = identity
            self.locus = locus
            dashboard.range_shell_status.setText(
                locus.label
                + " · "
                + ("horizontal cutaway" if cutaway else "full shell")
                + " · range-bin geometry"
            )
            dashboard.range_shell_status.setToolTip("\n".join(locus.disclosures))
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            self.clear(f"Range shell unavailable: {exc}")
        except RuntimeError:
            # Qt may retire a dashboard before the queued accepted-frame callback.
            if self.overlay is not None:
                self.overlay.clear()
            self.dashboard = None
            self.identity = None

    def frame(self):
        """Fit the explanatory shell only on an explicit camera-control request."""
        self.sync()
        if self.locus is None:
            return False
        points = np.concatenate([vertices for vertices, _triangles in self.locus.surfaces])
        lo, hi = points.min(axis=0), points.max(axis=0)
        return self.viz.renderer.set_overview_camera("isometric", ((lo + hi) / 2, (hi - lo) * 1.15))

    def close(self):
        if self.explanation is not None:
            self.explanation.close()
            self.explanation = None
        self.clear("Range shell cleared")
        self.dashboard = None
        self.accepted_step = None
