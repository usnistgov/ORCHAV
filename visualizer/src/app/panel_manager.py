"""Tabbed panel composition and widget binding for the visualizer shell.

``UIPanelManager`` creates the high-level tab/section layout, owns panel
instances, exposes panel visibility helpers, and connects generated widgets
back to callbacks on ``OrchavVisualizer``. Detailed control semantics stay in
the individual panel modules.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from shared.logging import get_logger

from ..extensions import RuntimeFeatureExtension, registered_runtime_extensions
from ..panels import (
    AnimationControlsPanel,
    BeamPatternPanel,
    CameraControlPanel,
    CoverageMapPanel,
    DataSourcePanel,
    ExportPanel,
    GlobalContextPanel,
    MaterialsPanel,
    MPCVisualizationPanel,
    NodesSelectionPanel,
    ObjectManagementPanel,
    PerformancePanel,
    RenderPanel,
    StatisticsPanel,
    TrajectoryPreviewPanel,
)
from ..panels.collapsible_section import CollapsibleSection
from ..renderers.protocol import renderer_capabilities
from ..state import (
    DEFAULT_MPC_ALLOWED_ORDERS,
    DEFAULT_MPC_ALLOWED_TYPES,
    MPC_ORDER_VALUES,
    MPC_TYPE_VALUES,
)

logger = get_logger("orchav.panel_manager")


class _RangeFiltersProxy:
    """Expose MPC range filters through the same section API as full panels."""

    def __init__(self, mpc_panel: MPCVisualizationPanel) -> None:
        """Store the MPC panel whose range-filter section is being proxied."""
        self._mpc_panel = mpc_panel
        self.widgets = mpc_panel.widgets

    def create_panel(self) -> QWidget:
        """Create the range-filter section using the wrapped MPC panel."""
        return self._mpc_panel.create_range_filters_panel()


class _RenderSubPanelProxy:
    """Expose one ``RenderPanel`` section through the common panel interface."""

    def __init__(self, render_panel: RenderPanel, method_name: str) -> None:
        """Store the render panel and sub-section factory method name."""
        self._render_panel = render_panel
        self._method_name = method_name
        self.widgets = render_panel.widgets

    def create_panel(self) -> QWidget:
        """Create the configured RenderPanel sub-section."""
        return getattr(self._render_panel, self._method_name)()


class _SharedPanelSectionProxy:
    """Expose one section factory from a panel already mounted elsewhere."""

    def __init__(self, panel: object, method_name: str) -> None:
        """Store the shared panel and the section factory to invoke."""
        self._panel = panel
        self._method_name = method_name
        self.widgets = getattr(panel, "widgets", {})

    def create_panel(self) -> QWidget:
        """Create the configured section without cloning panel state."""
        return getattr(self._panel, self._method_name)()


class _StatisticsSubPanelProxy:
    """Proxy that exposes StatisticsPanel sub-sections as tab sections."""

    def __init__(self, statistics_panel: StatisticsPanel, method_name: str) -> None:
        """Store the statistics panel and sub-section factory method name."""
        self._statistics_panel = statistics_panel
        self._method_name = method_name
        self.widgets = statistics_panel.widgets

    def create_panel(self) -> QWidget:
        """Create the configured StatisticsPanel sub-section."""
        return getattr(self._statistics_panel, self._method_name)()

    def bind_section(self, section: CollapsibleSection) -> None:
        """Give graph sections back to StatisticsPanel for visibility gating."""
        if self._method_name == "create_graphs_panel":
            self._statistics_panel.bind_graphs_section(section)


class UIPanelManager:
    """Create panels, arrange them into workflow tabs, and bind widgets.

    The manager owns the Qt layout shell only. Individual panel classes create
    their own widgets and controller callbacks own user-intent behavior; this
    class records enough section and widget references for startup, session
    restore, and scenario transitions to find the generated controls.
    """

    # Scene-only scenarios hide frame-dependent workflows but keep scene,
    # rendering, and system controls available.
    _FRAME_DATA_TABS = ("Paths", "Analysis", "Antennas")
    _FRAME_DATA_SECTIONS = ("nodes",)
    _LAZY_SECTION_KEYS = frozenset(
        {
            "statistics_graphs",
            "trajectory",
            "export",
        }
    )
    _SECTION_TITLE_OVERRIDES = {
        "statistics_graphs": "Graphs",
        "trajectory": "Trajectory Analysis",
        "export": "Export",
        "viewport_hud": "Viewport HUD",
    }

    def __init__(self, parent_widget, total_steps=60):
        """Store the visualizer parent and initialize panel registries."""
        self.parent = parent_widget
        self.total_steps = total_steps
        self.panels = {}
        self.widgets = {}
        self.sections = {}
        self.panel_sequence: List = []
        self.ctrl_panel = None
        self.ctrl_layout = None
        self._frame_data_available = True
        self._coverage_data_available = False
        self._cleanup_complete = False
        self._deferred_render_sync_timer: QTimer | None = None
        self._runtime_extensions: dict[str, RuntimeFeatureExtension] = {
            extension.key: extension for extension in registered_runtime_extensions()
        }
        conditional_tabs: dict[str, list[tuple[str, bool]]] = {}
        for extension in self._runtime_extensions.values():
            conditional_tabs.setdefault(extension.tab_label, []).append(
                (extension.key, extension.start_open)
            )
        self._conditional_tabs = list(conditional_tabs.items())

    def _is_statistics_enabled(self) -> bool:
        """Return whether YAML configuration enables the statistics panel."""
        return self._panel_enabled("statistics", default=True)

    def _panel_enabled(self, key: str, default: bool = True) -> bool:
        """Return True if a panel is enabled in visualizer_cfg (or default if unset)."""
        try:
            config = getattr(self.parent, "scenario_config", None)
            if not config:
                return default
            viz_config = getattr(config, "visualizer_cfg", {}) or {}
            panels_cfg = viz_config.get("panels", {}) if isinstance(viz_config, dict) else {}
            panel_cfg = panels_cfg.get(key, None)
            if isinstance(panel_cfg, dict):
                return bool(panel_cfg.get("enabled", default))
            if isinstance(panel_cfg, bool):
                return panel_cfg
        except (KeyError, AttributeError) as e:
            logger.debug("Could not check panel config for %s: %s", key, e)
        return default

    def update_total_steps(self, new_total_steps):
        """Propagate a new frame count to panels that mirror timeline length."""
        self.total_steps = new_total_steps

        if "animation" in self.panels:
            self.panels["animation"].update_total_steps(new_total_steps)
        if "export" in self.panels:
            self.panels["export"].update_total_steps(new_total_steps)

    def set_panel_visible(self, key: str, visible: bool) -> None:
        """Toggle visibility for a panel's collapsible section if it exists."""
        if key == "coverage":
            self.set_coverage_data_available(visible)
            return
        section = self.sections.get(key)
        if section:
            section.setVisible(bool(visible))

    def set_coverage_data_available(self, available: bool) -> None:
        """Show the Coverage workflow tab only when coverage data is loaded."""
        self._coverage_data_available = bool(available)
        section = self.sections.get("coverage")
        if section is not None:
            section.setVisible(self._coverage_data_available)
        self.notify_coverage_selection_changed()
        if not hasattr(self, "_tab_widget") or self._tab_widget is None:
            return
        self._set_tab_available(
            "Coverage",
            self._coverage_data_available,
            unavailable_reason="Unavailable: no coverage data loaded",
        )
        self._set_tab_available(
            "Analysis",
            self._frame_data_available
            or (self._coverage_data_available and self.panels.get("statistics_graphs") is not None),
            unavailable_reason="Unavailable: no frame or coverage data loaded",
        )
        if self._coverage_data_available and not self._frame_data_available:
            coverage_index = self.find_tab_index_by_label("Coverage")
            if coverage_index >= 0:
                self._tab_widget.setCurrentIndex(coverage_index)
        self._ensure_active_tab_available()

    def notify_coverage_selection_changed(self, *, render: bool = True) -> None:
        """Notify dynamic analysis figures after coverage UI state changes."""
        statistics_panel = self.panels.get("statistics")
        callback = getattr(statistics_panel, "coverage_selection_changed", None)
        if callable(callback):
            callback(render=render)

    def set_frame_data_available(self, available: bool) -> None:
        """Show frame-dependent UI only when MPC frame data is available.

        Scenario loading can produce scene-only sessions. In that mode the
        layout keeps scene/rendering/system controls usable while hiding tabs
        whose controls assume an MPC frame source.
        """
        self._frame_data_available = bool(available)
        context = self.panels.get("context")
        if context is not None:
            context.set_frame_data_available(self._frame_data_available)
        if not hasattr(self, "_tab_widget") or self._tab_widget is None:
            return

        extension_tabs = {
            extension.tab_label
            for extension in self._runtime_extensions.values()
            if extension.frame_data
        }
        for tab_label in (*self._FRAME_DATA_TABS, *sorted(extension_tabs)):
            self._set_tab_available(tab_label, self._frame_data_available)
        self._set_tab_available(
            "Analysis",
            self._frame_data_available
            or (self._coverage_data_available and self.panels.get("statistics_graphs") is not None),
            unavailable_reason="Unavailable: no frame or coverage data loaded",
        )

        for panel_key in self._FRAME_DATA_SECTIONS:
            self.set_panel_visible(panel_key, self._frame_data_available)

        self._ensure_active_tab_available()

    def _set_tab_available(
        self,
        label: str,
        available: bool,
        *,
        unavailable_reason: str = "Unavailable: no MPC frame data loaded",
    ) -> None:
        """Hide a tab when supported, otherwise disable it."""
        idx = self.find_tab_index_by_label(label)
        if idx < 0:
            return
        if hasattr(self._tab_widget, "setTabVisible"):
            self._tab_widget.setTabVisible(idx, bool(available))
        else:
            self._tab_widget.setTabEnabled(idx, bool(available))
            self._tab_widget.setTabToolTip(
                idx,
                "" if available else unavailable_reason,
            )

    def _tab_available_at(self, index: int) -> bool:
        """Return whether a tab index can become the active tab."""
        if index < 0 or not hasattr(self, "_tab_widget") or self._tab_widget is None:
            return False
        if hasattr(self._tab_widget, "isTabVisible") and not self._tab_widget.isTabVisible(index):
            return False
        return bool(self._tab_widget.isTabEnabled(index))

    def _ensure_active_tab_available(self) -> None:
        """Move focus away from a tab that was just hidden or disabled."""
        if not hasattr(self, "_tab_widget") or self._tab_widget is None:
            return
        current = self._tab_widget.currentIndex()
        if self._tab_available_at(current):
            return
        for idx in range(self._tab_widget.count()):
            if self._tab_available_at(idx):
                self._tab_widget.setCurrentIndex(idx)
                return

    def _collapse_all_sections(self) -> None:
        """Collapse all collapsible sections in the active tab."""
        for section in self._active_tab_sections():
            section.collapse()

    def _expand_all_sections(self) -> None:
        """Expand all collapsible sections in the active tab."""
        for section in self._active_tab_sections():
            section.expand()

    def _active_tab_sections(self) -> list:
        """Return CollapsibleSection widgets in the currently active tab."""
        if not hasattr(self, "_tab_widget") or self._tab_widget is None:
            return []
        current = self._tab_widget.currentWidget()
        if current is None:
            return []
        page = current.widget() if hasattr(current, "widget") else current
        if page is None:
            return []
        from ..panels.collapsible_section import CollapsibleSection

        return [child for child in page.findChildren(CollapsibleSection)]

    @staticmethod
    def _normalize_tab_label(label: str) -> str:
        """Collapse dynamic tab label variants to their stable logical label."""
        text = str(label or "").strip()
        if text.startswith("Paths"):
            return "Paths"
        return text

    def get_active_tab_label(self) -> Optional[str]:
        """Return the stable label for the currently active tab."""
        if not hasattr(self, "_tab_widget") or self._tab_widget is None:
            return None
        idx = self._tab_widget.currentIndex()
        if idx < 0:
            return None
        return self._normalize_tab_label(self._tab_widget.tabText(idx))

    def find_tab_index_by_label(self, label: str) -> int:
        """Return the current index of a stable tab label, or -1 if absent."""
        if not hasattr(self, "_tab_widget") or self._tab_widget is None:
            return -1
        target = self._normalize_tab_label(label)
        if not target:
            return -1
        for i in range(self._tab_widget.count()):
            if self._normalize_tab_label(self._tab_widget.tabText(i)) == target:
                return i
        return -1

    def restore_active_tab(
        self,
        *,
        label: Optional[str] = None,
        index: Optional[int] = None,
    ) -> bool:
        """Select a saved tab only when it still exists and is available."""
        normalized_label = self._normalize_tab_label(label or "")
        target = self.find_tab_index_by_label(normalized_label) if normalized_label else -1
        if target < 0 and not normalized_label and index is not None:
            try:
                candidate = int(index)
            except (TypeError, ValueError):
                candidate = -1
            if self._tab_available_at(candidate):
                target = candidate
        if not self._tab_available_at(target):
            return False
        self._tab_widget.setCurrentIndex(target)
        return True

    def get_panel_tab_label(self, panel_key: str) -> Optional[str]:
        """Return the stable tab label that owns a panel key, if known."""
        label = self._tab_map.get(panel_key)
        if label is None:
            return None
        return self._normalize_tab_label(label)

    def is_panel_in_active_tab(self, panel_key: str) -> bool:
        """Return whether a panel key belongs to the currently active tab page."""
        if not hasattr(self, "_tab_widget") or self._tab_widget is None:
            return True
        panel_tab_widget = self._tab_widget_by_panel.get(panel_key)
        if panel_tab_widget is None:
            return False
        current = self._tab_widget.currentWidget()
        if current is None:
            return True
        return current is panel_tab_widget

    def _range_filter_is_active(self, key: str, use_minimum: bool) -> bool:
        """Return whether a range-filter spinbox is away from its unfiltered bound."""
        mpc_panel = self.panels.get("mpc")
        if mpc_panel is None:
            return False
        widget = mpc_panel.widgets.get(key)
        if widget is None or not all(
            hasattr(widget, attr) for attr in ("value", "minimum", "maximum")
        ):
            return False
        current = float(widget.value())
        bound = float(widget.minimum() if use_minimum else widget.maximum())
        return abs(current - bound) > 1e-9

    def _has_active_path_filters(self) -> bool:
        """Return True when any Paths-tab MPC filter is currently active."""
        state = getattr(self.parent, "app_state", None)
        if state is not None:
            if frozenset(getattr(state, "mpc_allowed_orders", ())) != DEFAULT_MPC_ALLOWED_ORDERS:
                return True
            if frozenset(getattr(state, "mpc_allowed_types", ())) != DEFAULT_MPC_ALLOWED_TYPES:
                return True
            if bool(getattr(state, "topk_render_enabled", False)):
                return True
            if any(
                getattr(state, field, None) is not None
                for field in (
                    "delay_filter_min_ns",
                    "delay_filter_max_ns",
                    "power_filter_min_db",
                    "power_filter_max_db",
                    "aoa_az_filter_min_deg",
                    "aoa_az_filter_max_deg",
                    "aoa_el_filter_min_deg",
                    "aoa_el_filter_max_deg",
                    "aod_az_filter_min_deg",
                    "aod_az_filter_max_deg",
                    "aod_el_filter_min_deg",
                    "aod_el_filter_max_deg",
                )
            ):
                return True

        mpc_panel = self.panels.get("mpc")
        if mpc_panel is None:
            return False

        widgets = mpc_panel.widgets

        for i in MPC_ORDER_VALUES:
            cb = widgets.get(f"order_{i}_cb")
            if cb is not None and not cb.isChecked():
                return True

        for t in MPC_TYPE_VALUES:
            cb = widgets.get(f"type_{t}_cb")
            if cb is not None and not cb.isChecked():
                return True

        topk = widgets.get("topk_render_cb")
        if topk is not None and topk.isChecked():
            return True

        material_filter_active = getattr(mpc_panel, "material_filter_active", None)
        if callable(material_filter_active) and material_filter_active():
            return True

        range_filters = (
            ("delay_filter_min", True),
            ("delay_filter_max", False),
            ("power_filter_min", True),
            ("power_filter_max", False),
            ("aoa_az_filter_min", True),
            ("aoa_az_filter_max", False),
            ("aoa_el_filter_min", True),
            ("aoa_el_filter_max", False),
            ("aod_az_filter_min", True),
            ("aod_az_filter_max", False),
            ("aod_el_filter_min", True),
            ("aod_el_filter_max", False),
        )
        return any(
            self._range_filter_is_active(key, use_minimum) for key, use_minimum in range_filters
        )

    def _on_screenshot_clicked(self) -> None:
        """Take a screenshot of the 3D view."""
        export_panel = self.panels.get("export")
        if export_panel is not None and hasattr(export_panel, "_on_screenshot_clicked"):
            export_panel._on_screenshot_clicked()

    def update_paths_tab_badge(self) -> None:
        """Show a stable ``Paths`` tab label plus a transient filter badge."""
        filters_active = self._has_active_path_filters()
        context = self.panels.get("context")
        if context is not None:
            context.set_filters_active(filters_active)

        if not hasattr(self, "_tab_widget") or self._tab_widget is None:
            return

        label = "Paths (filtered)" if filters_active else "Paths"
        idx = self.find_tab_index_by_label("Paths")
        if idx >= 0:
            self._tab_widget.setTabText(idx, label)

    def refresh_global_context(self, state: object | None = None) -> None:
        """Refresh persistent context controls from canonical application state."""
        context = self.panels.get("context")
        if context is None:
            return
        context.sync_from_state(state)
        context.set_filters_active(self._has_active_path_filters())
        nodes = self.panels.get("nodes")
        if nodes is not None and hasattr(nodes, "update_node_rename_visibility"):
            nodes.update_node_rename_visibility()

    # Tab schema order is user-visible and persisted by session restore.
    # Each entry is (tab_label, [(panel_key, start_open), ...]).
    _CORE_TABS: List[Tuple[str, List[Tuple[str, bool]]]] = [
        ("Scene", [("nodes", True), ("objects", False), ("materials", False)]),
        ("Paths", [("mpc", True), ("range_filters", False)]),
        ("Coverage", [("coverage", True)]),
        (
            "Analysis",
            [
                ("statistics", True),
                ("statistics_graphs", False),
                ("trajectory", False),
                ("rf_xray", False),
            ],
        ),
        ("Edit", [("interactive_preview", True)]),
        (
            "Rendering",
            [
                ("scene_style", True),
                ("scene_view", False),
                ("lighting", False),
                ("viewport_hud", False),
            ],
        ),
        (
            "Capture & Export",
            [
                ("figure_capture", True),
                ("export", False),
            ],
        ),
        ("Antennas", [("beam_pattern", True)]),
        (
            "System",
            [
                ("data_source", True),
                ("performance", False),
            ],
        ),
    ]

    def create_all_panels(self):
        """Create all panel instances and arrange them in the app shell.

        Panels are instantiated before layout assembly so shared sub-section
        proxies can refer to the same widget dictionaries and post-build
        connection code can mirror widgets onto the parent visualizer.
        """
        self.panels["animation"] = AnimationControlsPanel(self.parent, self.total_steps)
        self.panels["context"] = GlobalContextPanel(self.parent)
        self.panels["nodes"] = NodesSelectionPanel(self.parent)
        capabilities = renderer_capabilities(getattr(self.parent, "renderer", None))
        self.panels["interactive_preview"] = _SharedPanelSectionProxy(
            self.panels["nodes"],
            "create_interactive_preview_panel",
        )
        self.panels["mpc"] = MPCVisualizationPanel(self.parent)
        self.panels["range_filters"] = _RangeFiltersProxy(self.panels["mpc"])
        self.panels["beam_pattern"] = BeamPatternPanel(self.parent)
        self.panels["coverage"] = CoverageMapPanel(self.parent)
        self.panels["camera"] = CameraControlPanel(self.parent)
        self.panels["objects"] = ObjectManagementPanel(self.parent)
        self.panels["performance"] = PerformancePanel(self.parent)
        self.panels["render"] = RenderPanel(self.parent)
        self.panels["scene_style"] = _RenderSubPanelProxy(
            self.panels["render"], "create_scene_style_panel"
        )
        self.panels["scene_view"] = _RenderSubPanelProxy(
            self.panels["render"], "create_scene_view_panel"
        )
        self.panels["lighting"] = _RenderSubPanelProxy(
            self.panels["render"], "create_lighting_panel"
        )
        self.panels["figure_capture"] = _RenderSubPanelProxy(
            self.panels["render"], "create_figure_capture_panel"
        )
        self.panels["viewport_hud"] = (
            _RenderSubPanelProxy(self.panels["render"], "create_viewport_hud_panel")
            if capabilities.viewport_hud
            else None
        )
        self.panels["materials"] = MaterialsPanel(self.parent)
        self.panels["rf_xray"] = (
            _SharedPanelSectionProxy(
                self.panels["materials"],
                "create_rf_xray_panel",
            )
            if capabilities.rf_xray_overlay
            else None
        )
        self.panels["data_source"] = DataSourcePanel(self.parent)
        self.panels["export"] = ExportPanel(self.parent, self.total_steps)

        for extension in self._runtime_extensions.values():
            self.panels[extension.key] = (
                extension.panel_factory(self.parent)
                if bool(extension.enabled(self.parent))
                else None
            )

        self.panels["trajectory"] = TrajectoryPreviewPanel(self.parent)
        if self._panel_enabled("statistics", default=True):
            stats_panel = StatisticsPanel(self.parent)
            self.panels["statistics"] = stats_panel
            self.panels["statistics_graphs"] = _StatisticsSubPanelProxy(
                stats_panel, "create_graphs_panel"
            )
        else:
            self.panels["statistics"] = None
            self.panels["statistics_graphs"] = None

        # ``ensure_panel`` uses this compatibility order when adding lazy panels
        # after the initial tab layout already exists.
        self.panel_sequence = [
            ("animation", True),
            ("camera", True),
            ("nodes", True),
            ("objects", False),
            ("materials", False),
            ("interactive_preview", True),
            ("mpc", True),
            ("range_filters", False),
            ("coverage", False),
            ("statistics", True),
            ("trajectory", False),
            ("rf_xray", False),
            ("scene_style", True),
            ("scene_view", False),
            ("lighting", False),
            ("viewport_hud", False),
            ("figure_capture", False),
            ("beam_pattern", True),
            ("data_source", True),
            ("performance", False),
            ("export", False),
        ]
        self.panel_sequence.extend(
            (extension.key, extension.start_open) for extension in self._runtime_extensions.values()
        )

        ctrl_panel = QWidget()
        root_layout = QVBoxLayout(ctrl_panel)
        root_layout.setSpacing(4)
        root_layout.setContentsMargins(6, 6, 6, 6)
        self.ctrl_panel = ctrl_panel
        self.ctrl_layout = root_layout

        # Animation and camera stay outside workflow tabs so the active tab can
        # change without hiding transport or viewpoint controls.
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setSpacing(8)
        top_layout.setContentsMargins(0, 0, 0, 0)

        anim_widget = self.panels["animation"].create_panel()
        if hasattr(anim_widget, "setTitle"):
            anim_widget.setTitle("")
        top_layout.addWidget(anim_widget, stretch=1)

        camera_widget = self.panels["camera"].create_panel()
        if hasattr(camera_widget, "setTitle"):
            camera_widget.setTitle("")
        top_layout.addWidget(camera_widget, stretch=0)

        root_layout.addWidget(top_bar)

        context_widget = self.panels["context"].create_panel()
        root_layout.addWidget(context_widget)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Plain)
        separator.setFixedHeight(1)
        root_layout.addWidget(separator)

        tab_toolbar = QHBoxLayout()
        tab_toolbar.setContentsMargins(0, 0, 0, 0)
        tab_toolbar.setSpacing(4)
        collapse_btn = QPushButton("\u25b2 Collapse All")
        collapse_btn.setFlat(True)
        collapse_btn.setFixedHeight(20)
        collapse_btn.setStyleSheet("QPushButton { font-size: 10px; padding: 1px 4px; }")
        collapse_btn.clicked.connect(self._collapse_all_sections)
        tab_toolbar.addWidget(collapse_btn)
        expand_btn = QPushButton("\u25bc Expand All")
        expand_btn.setFlat(True)
        expand_btn.setFixedHeight(20)
        expand_btn.setStyleSheet("QPushButton { font-size: 10px; padding: 1px 4px; }")
        expand_btn.clicked.connect(self._expand_all_sections)
        tab_toolbar.addWidget(expand_btn)
        tab_toolbar.addStretch()
        screenshot_btn = QPushButton("\U0001f4f7 Screenshot")
        screenshot_btn.setFlat(True)
        screenshot_btn.setFixedHeight(20)
        screenshot_btn.setStyleSheet("QPushButton { font-size: 10px; padding: 1px 4px; }")
        screenshot_btn.clicked.connect(self._on_screenshot_clicked)
        tab_toolbar.addWidget(screenshot_btn)
        root_layout.addLayout(tab_toolbar)

        self._tab_widget = QTabWidget()
        self._tab_map: Dict[str, str] = {}
        self._tab_widget_by_panel: Dict[str, QWidget] = {}
        self.sections = {}

        for tab_label, panel_defs in self._CORE_TABS:
            if any(self.panels.get(panel_key) is not None for panel_key, _ in panel_defs):
                self._add_tab(tab_label, panel_defs)

        for tab_label, panel_defs in self._conditional_tabs:
            has_any = any(self.panels.get(pk) is not None for pk, _ in panel_defs)
            if has_any:
                self._add_tab(tab_label, panel_defs)

        root_layout.addWidget(self._tab_widget, stretch=1)
        self.set_coverage_data_available(self._coverage_data_available)
        self.refresh_global_context()

        # Render sub-panels share one RenderPanel; sync after every widget has
        # been created so capability-dependent visibility can settle once.
        render_panel = self.panels.get("render")
        if render_panel is not None:
            timer = QTimer(ctrl_panel)
            timer.setSingleShot(True)
            timer.timeout.connect(render_panel._sync_from_visualizer)
            timer.start(0)
            self._deferred_render_sync_timer = timer

        return ctrl_panel

    def cleanup(self) -> None:
        """Stop panel-owned work and detach process-wide subscriptions once."""
        if self._cleanup_complete:
            return
        self._cleanup_complete = True

        timer = self._deferred_render_sync_timer
        self._deferred_render_sync_timer = None
        if timer is not None:
            timer.stop()
            timer.deleteLater()

        cleaned: set[int] = set()
        for panel in self.panels.values():
            if panel is None or id(panel) in cleaned:
                continue
            cleaned.add(id(panel))
            cleanup = getattr(panel, "cleanup", None)
            if not callable(cleanup):
                continue
            try:
                cleanup()
            except Exception as exc:  # cleanup must continue across independent panels
                logger.warning(
                    "Panel cleanup failed for %s: %s",
                    type(panel).__name__,
                    exc,
                    exc_info=True,
                )

    def _add_tab(
        self,
        label: str,
        panel_defs: List[Tuple[str, bool]],
    ) -> None:
        """Create a tab page and register each panel section by key."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setSpacing(6)
        page_layout.setContentsMargins(4, 4, 4, 4)

        for panel_key, start_open in panel_defs:
            panel = self.panels.get(panel_key)
            if panel is None:
                continue

            lazy = self._should_lazy_build_section(panel_key, start_open)
            panel_widget = None if lazy else self._create_panel_widget(panel_key)
            section_title = self._section_title(panel_key, panel_widget)
            section = CollapsibleSection(section_title, start_open=start_open)
            if panel_widget is not None:
                section.content_layout().addWidget(panel_widget)
                section._lazy_content_created = True
            else:
                section._lazy_content_created = False
                section.toggled.connect(
                    lambda checked, key=panel_key, sec=section: (
                        self._realize_lazy_section(key, sec) if checked else None
                    )
                )
            page_layout.addWidget(section)
            self.sections[panel_key] = section
            self._tab_map[panel_key] = label
            self._tab_widget_by_panel[panel_key] = scroll
            bind_section = getattr(panel, "bind_section", None)
            if callable(bind_section):
                bind_section(section)

        page_layout.addStretch()
        scroll.setWidget(page)
        self._tab_widget.addTab(scroll, label)

    def _should_lazy_build_section(self, panel_key: str, start_open: bool) -> bool:
        """Return whether a collapsed panel body can be built on first expansion."""
        extension = self._runtime_extensions.get(panel_key)
        if extension is not None:
            return not start_open and extension.lazy
        return not start_open and panel_key in self._LAZY_SECTION_KEYS

    def _section_title(self, panel_key: str, panel_widget: Optional[QWidget]) -> str:
        """Return a section title without forcing lazy panel construction."""
        if panel_widget is not None and hasattr(panel_widget, "title"):
            title = panel_widget.title()
            if title:
                return title
        panel = self.panels.get(panel_key)
        title_method = getattr(panel, "title", None)
        if callable(title_method):
            title = title_method()
            if title:
                return str(title)
        extension = self._runtime_extensions.get(panel_key)
        if extension is not None:
            return extension.panel_title
        return self._SECTION_TITLE_OVERRIDES.get(panel_key, panel_key.replace("_", " ").title())

    def _create_panel_widget(self, panel_key: str) -> QWidget:
        """Create a panel widget and normalize embedded group titles."""
        panel = self.panels[panel_key]
        panel_widget = panel.create_panel()
        if hasattr(panel_widget, "setTitle"):
            panel_widget.setTitle("")
        return panel_widget

    def _realize_lazy_section(self, panel_key: str, section: CollapsibleSection) -> None:
        """Build the body for a lazy section the first time it is expanded."""
        if bool(getattr(section, "_lazy_content_created", False)):
            return
        panel = self.panels.get(panel_key)
        if panel is None:
            return
        panel_widget = self._create_panel_widget(panel_key)
        section.content_layout().addWidget(panel_widget)
        section._lazy_content_created = True

    def _panel_start_open(self, key: str, default: bool = True) -> bool:
        """Return the default open state for a panel based on panel_sequence."""
        for seq_key, start_open in self.panel_sequence:
            if seq_key == key:
                return bool(start_open)
        return default

    def _create_section(self, key: str, start_open: bool) -> Optional[CollapsibleSection]:
        """Create one section for a lazily inserted optional panel."""
        panel = self.panels.get(key)
        if panel is None:
            return None

        panel_widget = panel.create_panel()
        section_title = ""
        if hasattr(panel_widget, "title"):
            section_title = panel_widget.title()
        if not section_title:
            section_title = key.replace("_", " ").title()
        if hasattr(panel_widget, "setTitle"):
            panel_widget.setTitle("")

        section = CollapsibleSection(section_title, start_open=start_open)
        section.content_layout().addWidget(panel_widget)
        return section

    def ensure_panel(self, key: str) -> None:
        """Create and insert an optional panel that became available later.

        Scenario loading may enable an external panel after the initial UI
        shell exists. This method preserves the initial tab/section structure.
        """
        if key in self.panels and self.panels[key] is not None:
            return

        extension = self._runtime_extensions.get(key)
        if extension is None:
            return
        self.panels[key] = extension.panel_factory(self.parent)

        if not hasattr(self, "_tab_widget") or self._tab_widget is None:
            return

        start_open = self._panel_start_open(key, default=extension.start_open)
        tab_label = extension.tab_label

        for i in range(self._tab_widget.count()):
            if self._tab_widget.tabText(i) == tab_label:
                section = self._create_section(key, start_open=start_open)
                if section is None:
                    return
                scroll = self._tab_widget.widget(i)
                page = scroll.widget()
                page_layout = page.layout()
                page_layout.insertWidget(max(page_layout.count() - 1, 0), section)
                self.sections[key] = section
                self._tab_map[key] = tab_label
                self._tab_widget_by_panel[key] = scroll
                return

        self._add_tab(tab_label, [(key, start_open)])

    def _connect_event_handlers(self, parent):
        """Wire generated widgets to controller handlers."""
        controller = getattr(parent, "ui_controller", None)
        animation_controller = getattr(parent, "animation_controller", None)

        if hasattr(parent, "play_btn") and parent.play_btn:
            handler = getattr(animation_controller, "toggle_animation", None)
            if callable(handler):
                parent.play_btn.clicked.connect(
                    lambda _checked=False, handler=handler: handler(direction=1)
                )
        if hasattr(parent, "reverse_play_btn") and parent.reverse_play_btn:
            handler = getattr(animation_controller, "play_backward", None)
            if callable(handler):
                parent.reverse_play_btn.clicked.connect(
                    lambda _checked=False, handler=handler: handler()
                )
        if hasattr(parent, "prev_btn") and parent.prev_btn:
            handler = getattr(animation_controller, "previous_frame", None)
            if callable(handler):
                parent.prev_btn.clicked.connect(lambda _checked=False, handler=handler: handler())
        if hasattr(parent, "next_btn") and parent.next_btn:
            handler = getattr(animation_controller, "next_frame", None)
            if callable(handler):
                parent.next_btn.clicked.connect(lambda _checked=False, handler=handler: handler())
        if hasattr(parent, "reset_btn") and parent.reset_btn:
            handler = getattr(animation_controller, "reset_animation", None)
            if callable(handler):
                parent.reset_btn.clicked.connect(lambda _checked=False, handler=handler: handler())
        if (
            controller
            and hasattr(controller, "handle_step_changed")
            and hasattr(parent, "step_slider")
            and parent.step_slider
        ):
            parent.step_slider.valueChanged.connect(controller.handle_step_changed)
            release_handler = getattr(controller, "handle_step_slider_released", None)
            if callable(release_handler):
                parent.step_slider.sliderReleased.connect(release_handler)
        if (
            controller
            and hasattr(controller, "handle_frame_input_changed")
            and hasattr(parent, "frame_input")
            and parent.frame_input
        ):
            parent.frame_input.valueChanged.connect(controller.handle_frame_input_changed)

        # The loop checkbox is read by frame-step methods instead of emitting
        # an immediate intent.

        if getattr(parent, "tx_dropdown", None) and controller:
            handler = getattr(controller, "handle_tx_selection_changed", None)
            if handler:
                parent.tx_dropdown.currentTextChanged.connect(handler)
        if getattr(parent, "rx_dropdown", None) and controller:
            handler = getattr(controller, "handle_rx_selection_changed", None)
            if handler:
                parent.rx_dropdown.currentTextChanged.connect(handler)
        if getattr(parent, "node_label_mode_combo", None) and controller:
            handler = getattr(controller, "handle_node_label_mode_changed", None)
            if handler:
                parent.node_label_mode_combo.currentIndexChanged.connect(handler)
        for widget_name, handler_name in [
            ("labels_cb", "handle_labels_toggled"),
            ("target_labels_cb", "handle_target_labels_toggled"),
            ("tx_orient_cb", "handle_tx_orientation_toggled"),
            ("rx_orient_cb", "handle_rx_orientation_toggled"),
            ("target_orient_cb", "handle_target_orientation_toggled"),
            ("tx_trajectory_cb", "handle_tx_trajectory_toggled"),
            ("rx_trajectory_cb", "handle_rx_trajectory_toggled"),
            ("target_trajectory_cb", "handle_target_trajectory_toggled"),
        ]:
            widget = getattr(parent, widget_name, None)
            handler = getattr(controller, handler_name, None) if controller else None
            if widget and handler:
                widget.stateChanged.connect(handler)

        color_handler = (
            getattr(controller, "handle_trajectory_color_mode_changed", None)
            if controller
            else None
        )
        if color_handler:
            for mode_id in ("node_color", "speed", "altitude", "time", "angular_speed"):
                rb = getattr(parent, f"trajectory_color_{mode_id}_rb", None)
                if rb:
                    rb.toggled.connect(color_handler)

        line_w_spin = getattr(parent, "trajectory_line_width_spin", None)
        line_w_handler = (
            getattr(controller, "handle_trajectory_line_width_changed", None)
            if controller
            else None
        )
        if line_w_spin and line_w_handler:
            line_w_spin.valueChanged.connect(line_w_handler)

        pt_size_spin = getattr(parent, "trajectory_point_size_spin", None)
        pt_size_handler = (
            getattr(controller, "handle_trajectory_point_size_changed", None)
            if controller
            else None
        )
        if pt_size_spin and pt_size_handler:
            pt_size_spin.valueChanged.connect(pt_size_handler)

        font_size_spin = getattr(parent, "label_font_size_spin", None)
        font_size_handler = getattr(controller, "handle_label_font_size_changed", None)
        if font_size_spin and font_size_handler:
            font_size_spin.valueChanged.connect(font_size_handler)

        for widget_name in ("per_node_type_rb", "individual_nodes_rb"):
            widget = getattr(parent, widget_name, None)
            handler = (
                getattr(controller, "handle_node_coloring_changed", None) if controller else None
            )
            if widget and handler:
                widget.toggled.connect(handler)

        if controller and hasattr(controller, "handle_label_offset_changed"):
            for widget_name in ("x_offset_spinbox", "y_offset_spinbox", "z_offset_spinbox"):
                widget = getattr(parent, widget_name, None)
                if widget:
                    widget.valueChanged.connect(controller.handle_label_offset_changed)

        if (
            controller
            and hasattr(controller, "handle_tx_marker_size_changed")
            and getattr(parent, "tx_marker_size_spin", None)
        ):
            parent.tx_marker_size_spin.valueChanged.connect(
                controller.handle_tx_marker_size_changed
            )
        if (
            controller
            and hasattr(controller, "handle_rx_marker_size_changed")
            and getattr(parent, "rx_marker_size_spin", None)
        ):
            parent.rx_marker_size_spin.valueChanged.connect(
                controller.handle_rx_marker_size_changed
            )

        if (
            controller
            and hasattr(controller, "handle_orientation_scale_changed")
            and getattr(parent, "orientation_scale_spin", None)
        ):
            parent.orientation_scale_spin.valueChanged.connect(
                controller.handle_orientation_scale_changed
            )

        if (
            controller
            and hasattr(controller, "handle_live_preview_toggled")
            and getattr(parent, "live_preview_cb", None)
        ):
            parent.live_preview_cb.stateChanged.connect(controller.handle_live_preview_toggled)
        if (
            controller
            and hasattr(controller, "handle_live_preview_recompute")
            and getattr(parent, "live_preview_recompute_btn", None)
        ):
            parent.live_preview_recompute_btn.clicked.connect(
                controller.handle_live_preview_recompute
            )
        if (
            controller
            and hasattr(controller, "handle_live_preview_reset_selected")
            and getattr(parent, "live_preview_reset_selected_btn", None)
        ):
            parent.live_preview_reset_selected_btn.clicked.connect(
                controller.handle_live_preview_reset_selected
            )
        if (
            controller
            and hasattr(controller, "handle_live_preview_reset_all")
            and getattr(parent, "live_preview_reset_all_btn", None)
        ):
            parent.live_preview_reset_all_btn.clicked.connect(
                controller.handle_live_preview_reset_all
            )

        if controller and getattr(parent, "mpc_layer_cb", None):
            parent.mpc_layer_cb.stateChanged.connect(controller.handle_mpc_layer_toggled)
        if (
            controller
            and hasattr(controller, "handle_viewport_hud_enabled_toggled")
            and getattr(parent, "viewport_hud_cb", None)
        ):
            parent.viewport_hud_cb.toggled.connect(controller.handle_viewport_hud_enabled_toggled)
        if controller and getattr(parent, "mpc_paths_cb", None):
            parent.mpc_paths_cb.stateChanged.connect(controller.handle_mpc_paths_toggled)
        if (
            controller
            and hasattr(controller, "toggle_mpc_explorer")
            and getattr(parent, "mpc_explorer_btn", None)
        ):
            parent.mpc_explorer_btn.clicked.connect(controller.toggle_mpc_explorer)
        if controller and getattr(parent, "mpc_bounce_points_cb", None):
            parent.mpc_bounce_points_cb.stateChanged.connect(
                controller.handle_mpc_bounce_points_toggled
            )
        if (
            controller
            and hasattr(controller, "handle_mpc_interaction_markers_toggled")
            and getattr(parent, "mpc_interaction_markers_cb", None)
        ):
            parent.mpc_interaction_markers_cb.stateChanged.connect(
                controller.handle_mpc_interaction_markers_toggled
            )
        if (
            controller
            and hasattr(controller, "handle_topk_render_toggled")
            and getattr(parent, "topk_render_cb", None)
        ):
            parent.topk_render_cb.stateChanged.connect(controller.handle_topk_render_toggled)
        if (
            controller
            and hasattr(controller, "handle_topk_render_max_paths_changed")
            and getattr(parent, "topk_render_max_spin", None)
        ):
            parent.topk_render_max_spin.valueChanged.connect(
                controller.handle_topk_render_max_paths_changed
            )
        if (
            controller
            and hasattr(controller, "handle_beamforming_toggled")
            and getattr(parent, "beamforming_cb", None)
        ):
            parent.beamforming_cb.stateChanged.connect(controller.handle_beamforming_toggled)
        for widget_name, handler_name in [
            ("beam_azimuth_spin", "handle_beamforming_resolution_azimuth_changed"),
            ("beam_elevation_spin", "handle_beamforming_resolution_elevation_changed"),
            ("beam_tx_scale_spin", "handle_beamforming_tx_scale_changed"),
            ("beam_rx_scale_spin", "handle_beamforming_rx_scale_changed"),
        ]:
            widget = getattr(parent, widget_name, None)
            handler = getattr(controller, handler_name, None) if controller else None
            if widget and handler:
                widget.valueChanged.connect(handler)
        for widget_name, handler_name in [
            ("beam_tx_selector", "handle_beamforming_tx_node_changed"),
            ("beam_rx_selector", "handle_beamforming_rx_node_changed"),
        ]:
            widget = getattr(parent, widget_name, None)
            handler = getattr(controller, handler_name, None) if controller else None
            if widget and handler:
                signal = getattr(widget, "currentTextChanged", None)
                if signal is not None:
                    signal.connect(handler)

        # Standalone beamforming controls share one mode handler because the
        # selected antenna/source mode determines which sub-controls are active.
        if controller:
            mode_handler = getattr(controller, "handle_standalone_mode_changed", None)
            if mode_handler:
                for widget_name in ("standalone_mode_frame", "standalone_mode_standalone"):
                    widget = getattr(parent, widget_name, None)
                    if widget:
                        widget.toggled.connect(mode_handler)
                for widget in getattr(parent, "standalone_optional_modes", {}).values():
                    if widget:
                        widget.toggled.connect(mode_handler)

            antenna_handler = getattr(controller, "handle_standalone_antenna_changed", None)
            if antenna_handler:
                for widget_name in ("standalone_rows", "standalone_cols"):
                    widget = getattr(parent, widget_name, None)
                    if widget:
                        widget.valueChanged.connect(antenna_handler)

            freq_handler = getattr(controller, "handle_standalone_frequency_changed", None)
            if freq_handler:
                widget = getattr(parent, "standalone_freq", None)
                if widget:
                    widget.valueChanged.connect(freq_handler)

            spacing_handler = getattr(controller, "handle_standalone_spacing_changed", None)
            if spacing_handler:
                for widget_name in ("standalone_h_spacing", "standalone_v_spacing"):
                    widget = getattr(parent, widget_name, None)
                    if widget:
                        widget.valueChanged.connect(spacing_handler)

            strategy_handler = getattr(controller, "handle_standalone_strategy_changed", None)
            if strategy_handler:
                widget = getattr(parent, "standalone_strategy", None)
                if widget:
                    widget.currentTextChanged.connect(strategy_handler)

            angles_handler = getattr(controller, "handle_standalone_angles_changed", None)
            if angles_handler:
                for widget_name in ("standalone_azimuth", "standalone_elevation"):
                    widget = getattr(parent, widget_name, None)
                    if widget:
                        widget.valueChanged.connect(angles_handler)

            for widget_name, handler_name in [
                ("beam_db_scale_cb", "handle_beamforming_db_scale_changed"),
                ("beam_dynamic_range", "handle_beamforming_dynamic_range_changed"),
                ("beam_colormap", "handle_beamforming_colormap_changed"),
                ("beam_element_pattern", "handle_beamforming_element_pattern_changed"),
                (
                    "beam_tx_element_pattern",
                    "handle_beamforming_tx_element_pattern_changed",
                ),
                (
                    "beam_rx_element_pattern",
                    "handle_beamforming_rx_element_pattern_changed",
                ),
            ]:
                handler = getattr(controller, handler_name, None)
                widget = getattr(parent, widget_name, None)
                if handler and widget:
                    if hasattr(widget, "stateChanged"):
                        widget.stateChanged.connect(handler)
                    elif hasattr(widget, "currentTextChanged"):
                        widget.currentTextChanged.connect(handler)
                    elif hasattr(widget, "valueChanged"):
                        widget.valueChanged.connect(handler)

        if controller and hasattr(controller, "handle_color_mode_changed"):
            for widget in (
                getattr(parent, "reflection_order_rb", None),
                getattr(parent, "mpc_type_rb", None),
                getattr(parent, "delay_rb", None),
                getattr(parent, "path_loss_rb", None),
                getattr(parent, "material_rb", None),
                getattr(parent, "reconstruction_type_rb", None),
            ):
                if widget:
                    widget.toggled.connect(controller.handle_color_mode_changed)

        # Capture loop variables as lambda defaults so each checkbox keeps its
        # own order/type/material identifier.
        if controller and hasattr(controller, "handle_mpc_order_filter_changed"):
            for i in MPC_ORDER_VALUES:
                checkbox = getattr(parent, f"order_{i}_cb", None)
                if checkbox:
                    handler = controller.handle_mpc_order_filter_changed
                    checkbox.stateChanged.connect(
                        lambda checked, order=i, handler=handler: handler(order, checked)
                    )
        if controller and hasattr(controller, "handle_mpc_type_filter_changed"):
            for type_val in MPC_TYPE_VALUES:
                checkbox = getattr(parent, f"type_{type_val}_cb", None)
                if checkbox:
                    handler = controller.handle_mpc_type_filter_changed
                    checkbox.stateChanged.connect(
                        lambda checked, type_val=type_val, handler=handler: handler(
                            type_val, checked
                        )
                    )
        if controller and hasattr(controller, "handle_mpc_material_filter_changed"):
            mpc_panel = self.panels.get("mpc")
            if mpc_panel and hasattr(mpc_panel, "widgets"):
                for key, w in mpc_panel.widgets.items():
                    if (
                        key.startswith("material_")
                        and key.endswith("_cb")
                        and hasattr(w, "stateChanged")
                    ):
                        material_id = key[len("material_") : -3]
                        handler = controller.handle_mpc_material_filter_changed
                        w.stateChanged.connect(
                            lambda checked, mid=material_id, handler=handler: handler(mid, checked)
                        )

        self._connect_filter_badge_updates()

        if controller and hasattr(controller, "handle_distinct_material_colors_toggled"):
            mpc_panel = self.panels.get("mpc")
            if mpc_panel and hasattr(mpc_panel, "widgets"):
                distinct_cb = mpc_panel.widgets.get("distinct_material_colors_cb")
                if distinct_cb and hasattr(distinct_cb, "stateChanged"):
                    distinct_cb.stateChanged.connect(
                        controller.handle_distinct_material_colors_toggled
                    )

        if hasattr(parent, "reset_camera_btn") and parent.reset_camera_btn:
            parent.reset_camera_btn.clicked.connect(parent.reset_camera_to_overview)

        if hasattr(parent, "view_top_btn") and parent.view_top_btn:
            parent.view_top_btn.clicked.connect(lambda: parent.apply_camera_view("top"))
        if hasattr(parent, "view_side_btn") and parent.view_side_btn:
            parent.view_side_btn.clicked.connect(lambda: parent.apply_camera_view("side"))
        if hasattr(parent, "view_iso_btn") and parent.view_iso_btn:
            parent.view_iso_btn.clicked.connect(lambda: parent.apply_camera_view("isometric"))
        if hasattr(parent, "view_front_btn") and parent.view_front_btn:
            parent.view_front_btn.clicked.connect(lambda: parent.apply_camera_view("front"))

        if hasattr(parent, "camera_preset_save_btn") and parent.camera_preset_save_btn:
            parent.camera_preset_save_btn.toggled.connect(parent._set_camera_preset_save_mode)
        if hasattr(parent, "camera_preset_buttons") and parent.camera_preset_buttons:
            for btn in parent.camera_preset_buttons:
                preset_num = btn.property("preset_num")
                if preset_num:
                    btn.clicked.connect(
                        lambda checked, pn=preset_num: parent._handle_camera_preset_clicked(pn)
                    )
        if controller and hasattr(controller, "handle_camera_mode_changed"):
            if hasattr(parent, "overview_mode_rb") and parent.overview_mode_rb:
                parent.overview_mode_rb.toggled.connect(
                    lambda checked: controller.handle_camera_mode_changed("overview", checked)
                )
            if hasattr(parent, "follow_mode_rb") and parent.follow_mode_rb:
                parent.follow_mode_rb.toggled.connect(
                    lambda checked: controller.handle_camera_mode_changed("follow", checked)
                )
            if hasattr(parent, "pov_mode_rb") and parent.pov_mode_rb:
                parent.pov_mode_rb.toggled.connect(
                    lambda checked: controller.handle_camera_mode_changed("pov", checked)
                )

        fly_cb = getattr(parent, "fly_mode_cb", None)
        if fly_cb:
            if controller and hasattr(controller, "handle_fly_mode_toggled"):
                fly_cb.toggled.connect(controller.handle_fly_mode_toggled)
            else:
                fly_cb.toggled.connect(
                    lambda checked: (
                        parent.renderer.set_fly_mode(checked)
                        if hasattr(parent, "renderer")
                        else None
                    )
                )

        minimap_cb = getattr(parent, "camera_minimap_cb", None)
        if controller and hasattr(controller, "handle_camera_minimap_toggled") and minimap_cb:
            minimap_cb.toggled.connect(controller.handle_camera_minimap_toggled)

        if controller and hasattr(controller, "handle_pov_axis_combo_changed"):
            combo = getattr(parent, "pov_axis_combo", None)
            if combo:
                combo.currentIndexChanged.connect(
                    lambda idx, c=combo: controller.handle_pov_axis_combo_changed(c.itemData(idx))
                )

        if controller and hasattr(controller, "handle_target_focus_changed"):
            widget = getattr(parent, "target_focus_dropdown", None)
            if widget:
                widget.currentTextChanged.connect(controller.handle_target_focus_changed)

        if controller and hasattr(controller, "handle_building_labels_toggled"):
            widget = getattr(parent, "building_labels_cb", None)
            if widget:
                widget.stateChanged.connect(controller.handle_building_labels_toggled)

        if (
            controller
            and hasattr(controller, "handle_scene_toggled")
            and getattr(parent, "scene_cb", None)
        ):
            parent.scene_cb.stateChanged.connect(controller.handle_scene_toggled)
        if (
            controller
            and hasattr(controller, "handle_target_toggled")
            and getattr(parent, "target_cb", None)
        ):
            parent.target_cb.stateChanged.connect(controller.handle_target_toggled)

        performance_widgets = getattr(self.panels.get("performance"), "widgets", {})
        restart_btn = performance_widgets.get("restart_preload_btn")
        if controller and hasattr(controller, "handle_restart_preload") and restart_btn:
            restart_btn.clicked.connect(controller.handle_restart_preload)
        clear_cache_btn = performance_widgets.get("clear_cache_btn")
        if (
            controller
            and hasattr(controller, "handle_clear_performance_caches")
            and clear_cache_btn
        ):
            clear_cache_btn.clicked.connect(controller.handle_clear_performance_caches)
        clear_asset_cache_btn = performance_widgets.get("clear_asset_cache_btn")
        if (
            controller
            and hasattr(controller, "handle_clear_asset_caches")
            and clear_asset_cache_btn
        ):
            clear_asset_cache_btn.clicked.connect(controller.handle_clear_asset_caches)
        if (
            controller
            and hasattr(controller, "handle_log_level_changed")
            and getattr(parent, "log_level_combo", None)
        ):
            parent.log_level_combo.currentTextChanged.connect(controller.handle_log_level_changed)

        animation_widgets = getattr(self.panels.get("animation"), "widgets", {})
        playback_mode_combo = animation_widgets.get("playback_mode_combo")
        playback_fps_spinbox = animation_widgets.get("playback_fps_spinbox")
        playback_stride_combo = animation_widgets.get("stride_combo")
        if (
            controller
            and hasattr(controller, "handle_playback_timing_changed")
            and playback_mode_combo is not None
        ):
            playback_mode_combo.currentIndexChanged.connect(
                controller.handle_playback_timing_changed
            )

        if (
            controller
            and hasattr(controller, "handle_playback_timing_changed")
            and playback_fps_spinbox is not None
        ):
            playback_fps_spinbox.valueChanged.connect(controller.handle_playback_timing_changed)

        if (
            controller
            and hasattr(controller, "handle_playback_timing_changed")
            and playback_stride_combo is not None
        ):
            playback_stride_combo.currentTextChanged.connect(
                controller.handle_playback_timing_changed
            )

        self._connect_object_material_sync()

        logger.debug("Event handlers connected")

    def _connect_filter_badge_updates(self) -> None:
        """Connect path filter widgets to the transient Paths-tab badge."""
        mpc_panel = self.panels.get("mpc")
        if mpc_panel is None:
            return
        widgets = mpc_panel.widgets
        for i in MPC_ORDER_VALUES:
            cb = widgets.get(f"order_{i}_cb")
            if cb is not None:
                cb.stateChanged.connect(lambda *_: self.update_paths_tab_badge())
        for t in MPC_TYPE_VALUES:
            cb = widgets.get(f"type_{t}_cb")
            if cb is not None:
                cb.stateChanged.connect(lambda *_: self.update_paths_tab_badge())
        topk = widgets.get("topk_render_cb")
        if topk is not None:
            topk.stateChanged.connect(lambda *_: self.update_paths_tab_badge())
        preset_combo = widgets.get("preset_combo")
        if preset_combo is not None:
            preset_combo.currentTextChanged.connect(lambda *_: self.update_paths_tab_badge())
        for key in (
            "delay_filter_min",
            "delay_filter_max",
            "power_filter_min",
            "power_filter_max",
            "aoa_az_filter_min",
            "aoa_az_filter_max",
            "aoa_el_filter_min",
            "aoa_el_filter_max",
            "aod_az_filter_min",
            "aod_az_filter_max",
            "aod_el_filter_min",
            "aod_el_filter_max",
        ):
            spin = widgets.get(key)
            if spin is not None:
                spin.valueChanged.connect(lambda *_: self.update_paths_tab_badge())

    def _connect_object_material_sync(self) -> None:
        """Mirror object-tree material selection into the Materials panel."""
        objects_panel = self.panels.get("objects")
        materials_panel = self.panels.get("materials")
        if objects_panel is None or materials_panel is None:
            return
        tree = objects_panel.widgets.get("object_tree")
        mat_combo = materials_panel.widgets.get("material_combo")
        if tree is None or mat_combo is None:
            return

        def _on_tree_selection_changed(current, _previous):
            """Mirror selected object material into the material editor combo."""
            if not current.isValid():
                return
            model = current.model()
            if model is None:
                return
            # Proxy models hide the source-item payload that stores material IDs.
            source_index = current
            if hasattr(model, "mapToSource"):
                source_index = model.mapToSource(current)
                model = model.sourceModel()
            if model is None or not hasattr(model, "itemFromIndex"):
                return
            item = model.itemFromIndex(source_index)
            if item is None:
                return
            entry = item.data(Qt.UserRole)
            if not isinstance(entry, dict):
                return
            material_id = entry.get("material_id")
            if material_id and mat_combo.findText(material_id) >= 0:
                mat_combo.setCurrentText(material_id)

        sel_model = tree.selectionModel()
        if sel_model is not None:
            sel_model.currentChanged.connect(_on_tree_selection_changed)

    def get_widgets(self):
        """Return a merged widget dictionary from instantiated panels."""
        all_widgets = {}
        for panel_name, panel in self.panels.items():
            if panel is None:
                continue
            all_widgets.update(panel.widgets)
        return all_widgets

    def connect_widgets_to_parent(self, parent):
        """Mirror panel widgets onto the parent visualizer, then wire signals.

        Several controllers and compatibility methods still read widgets as
        attributes on ``OrchavVisualizer``. This method keeps that surface
        stable while panel classes remain the actual widget owners.
        """
        if hasattr(self.panels["animation"], "widgets"):
            parent.play_btn = self.panels["animation"].widgets.get("play_btn")
            parent.reverse_play_btn = self.panels["animation"].widgets.get("reverse_play_btn")
            parent.prev_btn = self.panels["animation"].widgets.get("prev_btn")
            parent.next_btn = self.panels["animation"].widgets.get("next_btn")
            parent.reset_btn = self.panels["animation"].widgets.get("reset_btn")
            parent.step_slider = self.panels["animation"].widgets.get("step_slider")
            parent.step_label = self.panels["animation"].widgets.get("step_label")
            parent.frame_input = self.panels["animation"].widgets.get("frame_input")
            parent.total_steps_label = self.panels["animation"].widgets.get("total_steps_label")
            parent.loop_cb = self.panels["animation"].widgets.get("loop_cb")
            parent.stride_combo = self.panels["animation"].widgets.get("stride_combo")

        if hasattr(self.panels["context"], "widgets"):
            parent.tx_dropdown = self.panels["context"].widgets.get("tx_dropdown")
            parent.rx_dropdown = self.panels["context"].widgets.get("rx_dropdown")
            parent.mpc_layer_cb = self.panels["context"].widgets.get("mpc_layer_cb")
            parent.viewport_hud_cb = self.panels["context"].widgets.get("viewport_hud_cb")

        if hasattr(self.panels["nodes"], "widgets"):
            parent.labels_cb = self.panels["nodes"].widgets.get("labels_cb")
            parent.node_label_mode_combo = self.panels["nodes"].widgets.get("node_label_mode_combo")
            parent.target_labels_cb = self.panels["nodes"].widgets.get("target_labels_cb")
            parent.target_cb = self.panels["nodes"].widgets.get("target_cb")
            parent.label_font_size_spin = self.panels["nodes"].widgets.get("label_font_size_spin")
            parent.tx_orient_cb = self.panels["nodes"].widgets.get("tx_orient_cb")
            parent.rx_orient_cb = self.panels["nodes"].widgets.get("rx_orient_cb")
            parent.target_orient_cb = self.panels["nodes"].widgets.get("target_orient_cb")
            parent.node_coloring_group = self.panels["nodes"].widgets.get("node_coloring_group")
            parent.per_node_type_rb = self.panels["nodes"].widgets.get("per_node_type_rb")
            parent.individual_nodes_rb = self.panels["nodes"].widgets.get("individual_nodes_rb")
            parent.x_offset_spinbox = self.panels["nodes"].widgets.get("x_offset_spinbox")
            parent.y_offset_spinbox = self.panels["nodes"].widgets.get("y_offset_spinbox")
            parent.z_offset_spinbox = self.panels["nodes"].widgets.get("z_offset_spinbox")
            parent.tx_marker_size_spin = self.panels["nodes"].widgets.get("tx_marker_size_spin")
            parent.rx_marker_size_spin = self.panels["nodes"].widgets.get("rx_marker_size_spin")
            parent.orientation_scale_spin = self.panels["nodes"].widgets.get(
                "orientation_scale_spin"
            )
            parent.tx_legend_label = self.panels["nodes"].widgets.get("tx_legend_label")
            parent.rx_legend_label = self.panels["nodes"].widgets.get("rx_legend_label")
            parent.tx_rx_legend_layout = self.panels["nodes"].widgets.get("tx_rx_legend_layout")
            parent.tx_trajectory_cb = self.panels["nodes"].widgets.get("tx_trajectory_cb")
            parent.rx_trajectory_cb = self.panels["nodes"].widgets.get("rx_trajectory_cb")
            parent.target_trajectory_cb = self.panels["nodes"].widgets.get("target_trajectory_cb")
            parent.live_preview_cb = self.panels["nodes"].widgets.get("live_preview_cb")
            parent.live_preview_recompute_btn = self.panels["nodes"].widgets.get(
                "live_preview_recompute_btn"
            )
            parent.live_preview_reset_selected_btn = self.panels["nodes"].widgets.get(
                "live_preview_reset_selected_btn"
            )
            parent.live_preview_reset_all_btn = self.panels["nodes"].widgets.get(
                "live_preview_reset_all_btn"
            )
            parent.live_preview_status_label = self.panels["nodes"].widgets.get(
                "live_preview_status_label"
            )
            parent.trajectory_status_label = self.panels["nodes"].widgets.get(
                "trajectory_status_label"
            )
            for mode_id in ("node_color", "speed", "altitude", "time", "angular_speed"):
                key = f"trajectory_color_{mode_id}_rb"
                setattr(parent, key, self.panels["nodes"].widgets.get(key))
            parent.trajectory_line_width_spin = self.panels["nodes"].widgets.get(
                "trajectory_line_width_spin"
            )
            parent.trajectory_point_size_spin = self.panels["nodes"].widgets.get(
                "trajectory_point_size_spin"
            )

        if hasattr(self.panels["mpc"], "widgets"):
            parent.mpc_paths_cb = self.panels["mpc"].widgets.get("mpc_paths_cb")
            parent.mpc_bounce_points_cb = self.panels["mpc"].widgets.get("mpc_bounce_points_cb")
            parent.mpc_interaction_markers_cb = self.panels["mpc"].widgets.get(
                "mpc_interaction_markers_cb"
            )
            parent.topk_render_cb = self.panels["mpc"].widgets.get("topk_render_cb")
            parent.topk_render_max_spin = self.panels["mpc"].widgets.get("topk_render_max_spin")
            parent.mpc_info_label = self.panels["mpc"].widgets.get("mpc_info_label")
            parent.mpc_explorer_btn = self.panels["mpc"].widgets.get("mpc_explorer_btn")
            parent.color_mode_group = self.panels["mpc"].widgets.get("color_mode_group")
            parent.reflection_order_rb = self.panels["mpc"].widgets.get("reflection_order_rb")
            parent.mpc_type_rb = self.panels["mpc"].widgets.get("mpc_type_rb")
            parent.delay_rb = self.panels["mpc"].widgets.get("delay_rb")
            parent.path_loss_rb = self.panels["mpc"].widgets.get("path_loss_rb")
            parent.material_rb = self.panels["mpc"].widgets.get("material_rb")
            parent.reconstruction_type_rb = self.panels["mpc"].widgets.get("reconstruction_type_rb")
            parent.color_legend_label = self.panels["mpc"].widgets.get("color_legend_label")
            parent.colorbar_widget = self.panels["mpc"].widgets.get("colorbar_widget")
            parent.color_legend_layout = self.panels["mpc"].widgets.get("color_legend_layout")

            for order in MPC_ORDER_VALUES:
                setattr(
                    parent, f"order_{order}_cb", self.panels["mpc"].widgets.get(f"order_{order}_cb")
                )

            for type_val in MPC_TYPE_VALUES:
                setattr(
                    parent,
                    f"type_{type_val}_cb",
                    self.panels["mpc"].widgets.get(f"type_{type_val}_cb"),
                )

        if hasattr(self.panels["beam_pattern"], "widgets"):
            parent.beamforming_cb = self.panels["beam_pattern"].widgets.get("beamforming_cb")
            parent.beam_tx_selector = self.panels["beam_pattern"].widgets.get("beam_tx_selector")
            parent.beam_rx_selector = self.panels["beam_pattern"].widgets.get("beam_rx_selector")
            parent.beam_azimuth_spin = self.panels["beam_pattern"].widgets.get("beam_azimuth_spin")
            parent.beam_elevation_spin = self.panels["beam_pattern"].widgets.get(
                "beam_elevation_spin"
            )
            parent.beam_tx_scale_spin = self.panels["beam_pattern"].widgets.get(
                "beam_tx_scale_spin"
            )
            parent.beam_rx_scale_spin = self.panels["beam_pattern"].widgets.get(
                "beam_rx_scale_spin"
            )
            parent.standalone_mode_frame = self.panels["beam_pattern"].widgets.get("mode_frame")
            parent.standalone_mode_standalone = self.panels["beam_pattern"].widgets.get(
                "mode_standalone"
            )
            parent.standalone_optional_modes = {
                key[len("mode_optional_") :]: widget
                for key, widget in self.panels["beam_pattern"].widgets.items()
                if key.startswith("mode_optional_")
            }
            parent.standalone_rows = self.panels["beam_pattern"].widgets.get("standalone_rows")
            parent.standalone_cols = self.panels["beam_pattern"].widgets.get("standalone_cols")
            parent.standalone_freq = self.panels["beam_pattern"].widgets.get("standalone_freq")
            parent.standalone_h_spacing = self.panels["beam_pattern"].widgets.get(
                "standalone_h_spacing"
            )
            parent.standalone_v_spacing = self.panels["beam_pattern"].widgets.get(
                "standalone_v_spacing"
            )
            parent.standalone_strategy = self.panels["beam_pattern"].widgets.get(
                "standalone_strategy"
            )
            parent.standalone_azimuth = self.panels["beam_pattern"].widgets.get(
                "standalone_azimuth"
            )
            parent.standalone_elevation = self.panels["beam_pattern"].widgets.get(
                "standalone_elevation"
            )
            parent.beam_db_scale_cb = self.panels["beam_pattern"].widgets.get("beam_db_scale_cb")
            parent.beam_dynamic_range = self.panels["beam_pattern"].widgets.get(
                "beam_dynamic_range"
            )
            parent.beam_complexity_note = self.panels["beam_pattern"].widgets.get(
                "beam_complexity_note"
            )
            parent.beam_colormap = self.panels["beam_pattern"].widgets.get("beam_colormap")
            parent.beam_element_pattern = self.panels["beam_pattern"].widgets.get(
                "beam_element_pattern"
            )
            parent.beam_tx_element_pattern = self.panels["beam_pattern"].widgets.get(
                "beam_tx_element_pattern"
            )
            parent.beam_rx_element_pattern = self.panels["beam_pattern"].widgets.get(
                "beam_rx_element_pattern"
            )
            parent.beam_gain_label = self.panels["beam_pattern"].widgets.get("beam_gain_label")
            parent.beam_status_label = self.panels["beam_pattern"].widgets.get("beam_status_label")

        if hasattr(self.panels["camera"], "widgets"):
            parent.reset_camera_btn = self.panels["camera"].widgets.get("reset_camera_btn")

            parent.view_top_btn = self.panels["camera"].widgets.get("view_top_btn")
            parent.view_side_btn = self.panels["camera"].widgets.get("view_side_btn")
            parent.view_iso_btn = self.panels["camera"].widgets.get("view_iso_btn")
            parent.view_front_btn = self.panels["camera"].widgets.get("view_front_btn")
            parent.overview_mode_rb = self.panels["camera"].widgets.get("overview_mode_rb")
            parent.follow_mode_rb = self.panels["camera"].widgets.get("follow_mode_rb")
            parent.pov_mode_rb = self.panels["camera"].widgets.get("pov_mode_rb")
            parent.pov_axis_combo = self.panels["camera"].widgets.get("pov_axis_combo")
            parent.fly_mode_cb = self.panels["camera"].widgets.get("fly_mode_cb")
            parent.camera_minimap_cb = self.panels["camera"].widgets.get("camera_minimap_cb")
            parent.target_focus_dropdown = self.panels["camera"].widgets.get(
                "target_focus_dropdown"
            )
            parent.track_group = self.panels["camera"].widgets.get("track_group")
            parent.pov_axis_container = self.panels["camera"].widgets.get("pov_axis_container")
            parent.camera_preset_save_btn = self.panels["camera"].widgets.get(
                "camera_preset_save_btn"
            )
            parent.camera_preset_buttons = self.panels["camera"].widgets.get(
                "camera_preset_buttons", []
            )

        if hasattr(self.panels["objects"], "widgets"):
            parent.building_labels_cb = self.panels["objects"].widgets.get("building_labels_cb")
            parent.object_search_filter = self.panels["objects"].widgets.get("object_search_filter")
            parent.group_by_combo = self.panels["objects"].widgets.get("group_by_combo")
            parent.object_tree = self.panels["objects"].widgets.get("object_tree")
            parent.scene_cb = self.panels["objects"].widgets.get("scene_cb")

        if hasattr(self.panels["coverage"], "widgets"):
            parent.coverage_toggle = self.panels["coverage"].widgets.get("coverage_toggle")
            parent.coverage_opacity_slider = self.panels["coverage"].widgets.get("coverage_opacity")
            parent.coverage_status_label = self.panels["coverage"].widgets.get("coverage_status")

        if hasattr(self.panels["performance"], "widgets"):
            parent.preload_status_label = self.panels["performance"].widgets.get(
                "preload_status_label"
            )
            parent.log_level_combo = self.panels["performance"].widgets.get("log_level_combo")

        self._connect_event_handlers(parent)

        logger.debug("Widget connection completed")
