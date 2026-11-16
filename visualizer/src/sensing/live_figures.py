"""Move existing sensing plots into live docks without another data reader."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..widgets.dashboard_host import dashboard_is_visible


@dataclass
class _PinnedFigure:
    wrapper: QWidget
    caption: QLabel
    placeholder: QLabel
    inspected: bool = False


class LiveFiguresMixin:
    """Placement and selective demand for the dashboard's persistent figures."""

    _LIVE_FIGURES = (
        ("rd_plot", "Range–Velocity", ("range_doppler",)),
        ("rp_plot", "Range profile", ("range_profile", "range_doppler")),
        ("gt_range_plot", "Range over time", ()),
        ("gt_speed_plot", "Radial velocity over time", ()),
        ("tracking_plot", "Tracks and ground truth", ()),
        ("range_azimuth_plot", "Range–Azimuth", ("range_doppler_angle",)),
        ("doppler_azimuth_plot", "Doppler–Azimuth", ("range_doppler_angle",)),
        ("angular_spectrum_plot", "Angular spectrum", ("range_doppler_angle",)),
    )
    _LIVE_HISTORY = frozenset({"gt_range_plot", "gt_speed_plot", "tracking_plot"})
    _LIVE_ANGULAR = frozenset(
        {"range_azimuth_plot", "doppler_azimuth_plot", "angular_spectrum_plot"}
    )

    def available_live_figures(self) -> tuple[tuple[str, str], ...]:
        """List figures supported by the current recorded result metadata."""
        request = self._last_request
        if request is None:
            return self._live_catalog
        sensing = request.display_frame.get("sensing") or {}
        kinds = request.available_product_kinds
        if kinds is None:
            kinds = {d.get("kind") for d in sensing.get("products", {}).values()}
        config = sensing.get("config_resolved") or {}
        choices = []
        for key, title, products in self._LIVE_FIGURES:
            if products and not set(products).intersection(kinds):
                continue
            if key == "tracking_plot" and not getattr(self, "_tracking_spatial_available", False):
                continue
            if key == "angular_spectrum_plot" and config.get("antenna_rows") == 1:
                continue
            choices.append((key, title))
        return tuple(choices)

    def _publish_live_figure_catalog(self) -> None:
        choices = self.available_live_figures()
        request = self._last_request
        if request is not None:
            self._live_product_kinds = request.available_product_kinds
            if self._live_product_kinds is None:
                self._live_product_kinds = frozenset(
                    descriptor.get("kind")
                    for descriptor in request.display_frame["sensing"].get("products", {}).values()
                )
        if choices != self._live_catalog:
            self._live_catalog = choices
            self.live_figures_changed.emit()

    def _live_only(self) -> bool:
        return bool(self._live_figures) and not dashboard_is_visible(self)

    def _live_plot_needs_update(self, plot) -> bool:
        request = self._last_request
        return (
            not self._live_figures and (request is None or not request.recording_managed)
        ) or dashboard_is_visible(plot)

    def _live_plot_is_inspected(self, plot) -> bool:
        return any(
            pinned.inspected and getattr(self, key) is plot
            for key, pinned in self._live_figures.items()
        )

    def live_figure_request_suffix(self) -> tuple:
        """Extend request identity only while a dock needs its own products."""
        return (self.live_figure_products(),) if self.has_visible_live_figures() else ()

    def acquire_live_figure(self, key: str, parent: QWidget) -> QWidget:
        """Lend one plot to a dock, preserving its data, zoom and click handlers."""
        if key not in dict(self.available_live_figures()):
            raise ValueError("This sensing figure was not recorded")
        if key in self._live_figures:
            raise ValueError("This sensing figure is already open")
        wrapper = QWidget(parent)
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(4, 4, 4, 4)
        caption = QLabel(wrapper)
        caption.setWordWrap(True)
        caption.setProperty("role", "secondary")
        layout.addWidget(caption)
        plot = getattr(self, key)
        inspected = plot.has_manual_range()
        layout.addWidget(plot, 1)
        placeholder = QLabel(
            f"{dict(self.available_live_figures())[key]} is open in a live figure."
        )
        placeholder.setWordWrap(True)
        self._live_figures[key] = _PinnedFigure(wrapper, caption, placeholder, inspected)
        plot.getViewBox().sigRangeChangedManually.connect(self._live_range_inspected)
        self._repopulate_grid(self._selected_view_mode())
        plot.show()
        self._update_live_figure_captions()
        self.view_changed.emit()
        return wrapper

    def release_live_figure(self, key: str) -> None:
        """Return the same plot to its normal dashboard layout."""
        pinned = self._live_figures.pop(key, None)
        if pinned is None:
            return
        plot = getattr(self, key)
        plot.getViewBox().sigRangeChangedManually.disconnect(self._live_range_inspected)
        pinned.wrapper.layout().removeWidget(plot)
        plot.hide()
        plot.setParent(self._parked_plots)
        pinned.placeholder.deleteLater()
        pinned.wrapper.hide()
        pinned.wrapper.deleteLater()
        self._repopulate_grid(self._selected_view_mode())
        self._recording_render_key = None
        self._recorded_scatter_state = None
        self.view_changed.emit()

    def _live_range_inspected(self, *_args) -> None:
        view = self.sender()
        for key, pinned in self._live_figures.items():
            if getattr(self, key).getViewBox() is view:
                pinned.inspected = True

    def has_visible_live_figures(self) -> bool:
        """Hidden or minimized figure windows request no data or plotting work."""
        return bool(self._visible_live_figure_keys())

    def _visible_live_figure_keys(self) -> set[str]:
        return {
            key
            for key, pinned in self._live_figures.items()
            if dashboard_is_visible(pinned.wrapper)
        }

    def refresh_live_figure_visibility(self) -> None:
        """Reconsider product demand after a host is shown, hidden or restored."""
        self._recording_render_key = None
        self._recorded_scatter_state = None
        self.view_changed.emit()

    def live_figure_products(self) -> tuple[str, ...]:
        """Request only products used by visible pinned figures."""
        visible = self._visible_live_figure_keys()
        products = set()
        for key, _title, required in self._LIVE_FIGURES:
            if key not in visible:
                continue
            if key == "rp_plot":
                required = (
                    (
                        "range_profile"
                        if "range_profile" in self._live_product_kinds
                        else "range_doppler"
                    ),
                )
            products.update(required)
        return tuple(sorted(products))

    def _live_history_visible(self) -> bool:
        return bool(self._LIVE_HISTORY.intersection(self._visible_live_figure_keys()))

    def _tracking_figures_active(self) -> bool:
        return self._selected_view_mode() == "tracking" or self._live_history_visible()

    def _update_live_figure_captions(self, unavailable: str = "") -> None:
        request = self._last_request
        if unavailable:
            text = unavailable
        elif request is None:
            text = "Loading sensing result…"
        else:
            time_s = request.display_frame["sensing"]["temporal"]["result"]["reference_time_s"]
            time_label = (
                f"Result {time_s:.3f} s · " if isinstance(time_s, Real) and isfinite(time_s) else ""
            )
            text = f"{time_label}scene {request.effective_display_step}"
            if request.display_from_cache:
                text += f" · held for scene {request.state_step}"
        for pinned in self._live_figures.values():
            pinned.caption.setText(text)

    def _update_live_figures(self, request) -> None:
        """Render visible pins through the existing plot methods and prepared data."""
        visible = self._visible_live_figure_keys()
        if not visible:
            return
        sensing = request.display_frame["sensing"]
        config = sensing["config_resolved"]
        variant = "primary" if request.variant == "clean" else request.variant
        suffix = " (alternate saved)" if variant == "raw" else ""
        self._update_target_names(request.state_frame)
        self._update_target_names(request.display_frame)
        if "rd_plot" in visible:
            self._update_ground_truth(request.display_frame, render=False)
            self._update_range_velocity(
                request.rd_result, request.display_frame, suffix, config["carrier_freq_hz"]
            )
        if "rp_plot" in visible:
            self._update_range_profile(request.rp_result, suffix, rd_result=request.rd_result)
        if visible.intersection(self._LIVE_ANGULAR):
            self._update_angular_products(sensing, variant, suffix, config["carrier_freq_hz"])
        if visible.intersection(self._LIVE_HISTORY):
            history = request.tracking_history or request.recording_history
            self._render_tracking_history(history, self.result_time_s(), apply_overlays=False)
            if history is not None and not history.error:
                self._fit_history_axes(history, self.result_time_s())
        self._apply_overlay_visibility()
        self.apply_linked_selection()
        self._update_live_figure_captions()
