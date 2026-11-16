"""Task navigation and persistent result browsing for sensing analysis.

The existing view tokens remain the reader's product-selection contract. This
shell groups those views without requesting products for hidden workspaces.
"""

from PySide6.QtCore import QSettings, QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabBar,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

from ..widgets.wrapping_layout import WrappingLayout

_RESULT_TABS = (
    ("Detections", "current", None),
    ("Detector", "diagnostics", "cfar"),
    ("Angle", "angular", None),
    ("Inputs", "diagnostics", "inputs"),
)


class WorkspaceShellMixin:
    """Keep navigation and selection outside the scrollable evidence area."""

    def _build_workspace_shell(self, root, navigation, overlay_controls):
        from .current_result_summary import CurrentResultSummary

        header = root.itemAt(0).layout()
        while root.count():
            item = root.takeAt(0)
            if item.layout() is not None:
                layout = item.layout()
                layout.setParent(None)
                if layout not in (header, navigation, overlay_controls):
                    while layout.count():
                        unused = layout.takeAt(0).widget()
                        if isinstance(unused, QLabel):
                            unused.hide()
                            unused.setParent(self._parked_plots)
        root.addLayout(header)
        self._view_mode_box.hide()
        self._support_box.hide()
        self.view_description.hide()
        self.summary_label.hide()
        self._product_note.hide()

        tools = WrappingLayout()
        self.workspace_box = QComboBox()
        for label, value in (
            ("Run summary", "summary"),
            ("Current result", "current"),
            ("Motion", "micro-doppler"),
            ("Tracking", "tracking"),
        ):
            self.workspace_box.addItem(label, value)
        self.workspace_box.currentIndexChanged.connect(self._choose_workspace)
        tools.addWidget(self.workspace_box)
        display_tools = WrappingLayout()
        display_tools.addWidget(self.table_button)
        display_tools.addWidget(QLabel("Graph height:"))
        self._graph_height_box.setMaximumWidth(85)
        display_tools.addWidget(self._graph_height_box)
        overlay_controls.removeWidget(self._zoom_data_cb)
        display_tools.addWidget(self._zoom_data_cb)
        display_footer = QWidget()
        display_footer.setLayout(display_tools)
        self._advanced_widget.layout().addWidget(display_footer, 16, 0, 1, 6)
        root.addLayout(tools)
        navigation.addWidget(self.provenance_label)
        root.addLayout(navigation)

        self.workspace_tabs = QTabBar()
        self.workspace_tabs.setExpanding(False)
        self.workspace_tabs.setUsesScrollButtons(True)
        self.workspace_tabs.currentChanged.connect(self._choose_workspace_tab)
        self._workspace_tab_group = None
        root.addWidget(self.workspace_tabs)
        self.input_kind_box = QComboBox()
        self.input_kind_box.addItem("CIR samples", "inputs")
        self.input_kind_box.addItem("Frequency response", "frequency")
        self.input_kind_box.currentIndexChanged.connect(self._choose_input_kind)
        local_controls = WrappingLayout()
        local_controls.addWidget(self.input_kind_box)
        local_controls.addWidget(self._advanced_motion_cb)
        self.tracking_projection = QComboBox()
        for plane in ("XY", "XZ", "YZ"):
            self.tracking_projection.addItem(plane)
        self.tracking_projection.setToolTip("Spatial projection of saved world positions.")
        self.tracking_projection.currentIndexChanged.connect(self._rerender_last_request)
        local_controls.addWidget(self.tracking_projection)
        self.tracking_measurements = QCheckBox("Measurements")
        self.tracking_measurements.setChecked(True)
        self.tracking_measurements.toggled.connect(self._apply_overlay_visibility)
        local_controls.addWidget(self.tracking_measurements)
        from .truth_zoom import TruthZoomController

        self.truth_zoom = TruthZoomController(self)
        local_controls.addWidget(self.truth_zoom.combo)
        root.addLayout(local_controls)
        overlays = QWidget()
        overlays.setLayout(overlay_controls)
        self._advanced_widget.layout().addWidget(overlays, 15, 0, 1, 6)

        self.evidence_scroll = QScrollArea()
        self.evidence_scroll.setWidgetResizable(True)
        self.evidence_scroll.setFrameShape(QScrollArea.NoFrame)
        self.evidence_scroll.setMinimumHeight(100)
        self.evidence_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        body = QWidget()
        evidence = QVBoxLayout(body)
        evidence.setContentsMargins(0, 0, 0, 0)
        # Expanded controls change height without changing the available width.
        # Avoid QWidgetItemV2 retaining their earlier collapsed size hint.
        self._advanced_widget.setParent(body)
        evidence.addItem(QWidgetItem(self._advanced_widget))
        evidence.addWidget(self.motion_controls)
        self.current_result_summary = CurrentResultSummary()
        evidence.addWidget(self.current_result_summary)
        self.current_product_notice = QLabel()
        self.current_product_notice.setWordWrap(True)
        self.current_product_notice.hide()
        evidence.addWidget(self.current_product_notice)
        for expanding in (self.tracking_inspector, self.association_figure):
            expanding.setParent(body)
            evidence.addItem(QWidgetItem(expanding))
        evidence.addWidget(self._plots_widget, 1)
        evidence.addWidget(self._unavailable_label)
        evidence.addWidget(self.detection_stage_summary)
        evidence.addWidget(self.bin_spacing_label)
        self.evidence_scroll.setWidget(body)
        self.result_splitter = QSplitter(Qt.Vertical)
        self.result_splitter.setChildrenCollapsible(False)
        self.result_splitter.addWidget(self.evidence_scroll)
        self.result_scroll = QScrollArea()
        self.result_scroll.setWidgetResizable(True)
        self.result_scroll.setFrameShape(QScrollArea.NoFrame)
        self.result_scroll.setMinimumHeight(120)
        results_body = QWidget()
        results = QVBoxLayout(results_body)
        results.setContentsMargins(0, 0, 0, 0)
        self.result_browser.setMinimumHeight(120)
        self.result_browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results.addWidget(self.result_browser, 1)
        results.addWidget(self.selection_tools_toggle, alignment=Qt.AlignLeft)
        # Expanded details can exceed the table pane's height. Keep them below
        # the selected rows and scroll them without shrinking the graph pane.
        # An uncached item follows changes in the card's wrapped height.
        self.selection_tools.setParent(results_body)
        results.addItem(QWidgetItem(self.selection_tools))
        self.result_scroll.setWidget(results_body)
        self.result_splitter.addWidget(self.result_scroll)
        self.result_splitter.setStretchFactor(0, 1)
        self.result_splitter.setStretchFactor(1, 0)
        root.addWidget(self.result_splitter, 1)
        self._browser_settings = QSettings()
        height = self._browser_settings.value("visualizer/sensing_browser_height", 190, type=int)
        self.result_splitter.setSizes([650, max(120, min(height, 500))])
        self.result_splitter.splitterMoved.connect(self._remember_browser_size)
        opened = self._browser_settings.value("visualizer/sensing_browser_open", True, type=bool)
        self.table_button.setText("Results table")
        with QSignalBlocker(self.table_button):
            self.table_button.setChecked(opened)
        self._table_requested = opened
        self.result_browser.setVisible(opened)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _update_result_browser(self, request):
        from .tracking_presentation import tracking_snapshot

        sensing = request.display_frame["sensing"]
        key = (id(sensing), id(request.saved_tracking), self.result_time_s())
        if (
            request.display_data_token is None
            or getattr(self, "_browser_snapshot_key", None) != key
            or getattr(self, "_browser_snapshot_payload", None) is not sensing
        ):
            self._browser_snapshot = tracking_snapshot(
                sensing, saved_tracking=request.saved_tracking, time_s=self.result_time_s()
            )
            self._browser_snapshot_key = key
            self._browser_snapshot_payload = sensing
        self.result_browser.update_from_request(
            request,
            tracks=self._browser_snapshot.states,
            result_label=f"Selected result · {self.result_time_s():.3f} s",
            tracks_note=self._browser_snapshot.note,
            complete_track_states=self._browser_snapshot.complete_states,
        )

    def _update_browser_cursor(self, record):
        from .tracking_presentation import tracking_snapshot

        metadata = record.sensing_metadata or {}
        snapshot = (
            tracking_snapshot(
                metadata, saved_tracking=self._last_request.saved_tracking, time_s=record.time_s
            )
            if self._last_request.saved_tracking is not None
            else None
        )
        self.result_browser.update_result(
            metadata.get("detections", record.detections),
            snapshot.states if snapshot is not None else record.tracking_states,
            context_key=(self._last_request.source_identity, *record.identity[:3]),
            acquisition_id=record.identity[3],
            detection_collection=metadata.get("_display_detection_collection", "detections"),
            elevation_available=metadata.get("config_resolved", {}).get("antenna_rows") != 1,
            result_label=f"Selected result · {record.time_s:.3f} s",
            tracks_note=(
                snapshot.note
                if snapshot is not None
                else (
                    "Saved tracker states"
                    if getattr(record, "tracking_states_complete", False)
                    else "Published track states"
                )
            ),
            complete_track_states=(
                snapshot.complete_states
                if snapshot is not None
                else getattr(record, "tracking_states_complete", False)
            ),
        )

    def _refresh_current_summary(self):
        if self._last_request is None or self._selected_view_mode() not in {"current", "analysis"}:
            return
        sensing = self._last_request.display_frame["sensing"]
        event = self._last_request.evaluation
        summary = (
            self.evaluation_panel.current_summary() if event and not event.get("reason") else event
        )
        if summary is not None:
            summary = dict(summary)
            model = self.evaluation_settings
            cohort = "Confirmed" if model.confirmed_only else "All outputs"
            summary["scope_label"] = (
                f"{cohort} · ±{model.range_gate_m:g} m / ±{model.velocity_gate_m_s:g} m/s"
            )
        self.current_result_summary.update_result(
            sensing,
            result_label=self.provenance_label.text(),
            evaluation=summary,
            tracking_snapshot=getattr(self, "_browser_snapshot", None),
        )

    def _select_browser_detection(self, index):
        if getattr(self, "_recording_cursor", None) is not None:
            self._pending_browser_selection = (
                self._last_request.source_identity,
                self._recording_cursor.identity,
                (self._recording_cursor.sensing_metadata or {}).get(
                    "_display_detection_collection", "detections"
                ),
                "detection",
                index,
            )
            self.view_changed.emit()
        else:
            self.select_detection_index(index)

    def _select_browser_track(self, track_id):
        if getattr(self, "_recording_cursor", None) is not None:
            self._pending_browser_selection = (
                self._last_request.source_identity,
                self._recording_cursor.identity,
                (self._recording_cursor.sensing_metadata or {}).get(
                    "_display_detection_collection", "detections"
                ),
                "track",
                track_id,
            )
            self.view_changed.emit()
        else:
            self.select_track(track_id)

    def _finish_browser_selection(self, sensing):
        from .selection import selection_scope

        pending = getattr(self, "_pending_browser_selection", None)
        if pending is None:
            return
        self._pending_browser_selection = None
        identity = (*selection_scope(sensing), sensing["temporal"]["acquisition"]["acquisition_id"])
        if (
            pending[0] == self._last_request.source_identity
            and pending[1] == identity
            and pending[2] == sensing.get("_display_detection_collection", "detections")
        ):
            if pending[3] == "detection":
                self.select_detection_index(pending[4])
            else:
                self.select_track(pending[4])

    def _remember_browser_size(self, *_):
        sizes = self.result_splitter.sizes()
        if self.result_browser.isVisible() and sizes[1] > 0:
            self._browser_settings.setValue("visualizer/sensing_browser_height", sizes[1])

    def _set_browser_visible(self, visible):
        self._table_requested = bool(visible)
        self._sync_browser_visibility()
        self._browser_settings.setValue("visualizer/sensing_browser_open", bool(visible))

    def _sync_browser_visibility(self):
        per_result = self._selected_view_mode() not in {"summary", "evaluation"} and not (
            self._last_request is not None and self._last_request.recording_overview_only
        )
        self.table_button.setVisible(per_result)
        self.result_browser.setVisible(per_result and self._table_requested)
        self.result_scroll.setVisible(
            per_result and (self._table_requested or not self.selection_tools_toggle.isHidden())
        )

    def _choose_workspace(self, *_):
        self._set_view_mode(self.workspace_box.currentData())

    def _choose_workspace_tab(self, index):
        if index < 0:
            return
        if self._workspace_tab_group == "summary":
            self._set_view_mode("summary" if index == 0 else "evaluation")
        elif self._workspace_tab_group == "current":
            _, mode, stage = _RESULT_TABS[index]
            self._open_result_detail(mode, stage or "")

    def _choose_input_kind(self, *_):
        item = self.input_kind_box.model().item(self.input_kind_box.currentIndex())
        if item is None or not item.isEnabled():
            return
        self._open_result_detail("diagnostics", self.input_kind_box.currentData())

    def _open_result_detail(self, mode, stage=""):
        if stage:
            self._support_choices["diagnostics"] = stage
        self._set_view_mode(mode)
        if stage:
            self._support_box.setCurrentIndex(self._support_box.findData(stage))
        self._on_view_mode_changed()

    def _sync_workspace_navigation(self):
        if not hasattr(self, "workspace_box"):
            return
        mode = self._selected_view_mode()
        group = (
            "current"
            if mode in {"analysis", "diagnostics", "angular"}
            else "summary" if mode == "evaluation" else mode
        )
        with QSignalBlocker(self.workspace_box):
            self.workspace_box.setCurrentIndex(self.workspace_box.findData(group))
        tabs = (
            ("Overview",)
            if group == "summary"
            else (tuple(t[0] for t in _RESULT_TABS) if group == "current" else ())
        )
        with QSignalBlocker(self.workspace_tabs):
            if self._workspace_tab_group != group:
                while self.workspace_tabs.count():
                    self.workspace_tabs.removeTab(0)
                for label in tabs:
                    self.workspace_tabs.addTab(label)
                self._workspace_tab_group = group
            if group == "summary":
                selected = 0
            elif mode in {"current", "analysis"}:
                selected = 0
            elif mode == "angular":
                selected = 2
            else:
                selected = {"cfar": 1, "clutter": 1, "inputs": 3, "frequency": 3}.get(
                    self._support_box.currentData(), 1
                )
            if tabs:
                self.workspace_tabs.setCurrentIndex(selected)
        self.workspace_tabs.setVisible(bool(tabs))
        self.current_result_summary.setVisible(mode in {"current", "analysis"})
        missing_maps = not self.rd_plot.isEnabled()
        missing_profile = not self.rp_plot.isEnabled()
        self.current_product_notice.setText(
            "Range–Doppler and range profiles were not recorded."
            if missing_maps and missing_profile
            else (
                "Range–Doppler maps were not recorded."
                if missing_maps
                else "Range profiles were not recorded."
            )
        )
        self.current_product_notice.setVisible(
            self._last_request is not None
            and mode in {"current", "analysis"}
            and (missing_maps or missing_profile)
        )
        self._sync_selection_details()
        self._view_mode_box.hide()
        self._support_box.hide()
        self._product_note.hide()
        self.view_description.hide()
        is_input = mode == "diagnostics" and self._support_box.currentData() in {
            "inputs",
            "frequency",
        }
        self.input_kind_box.setVisible(is_input)
        if is_input:
            with QSignalBlocker(self.input_kind_box):
                self.input_kind_box.setCurrentIndex(
                    self.input_kind_box.findData(self._support_box.currentData())
                )
        for token in ("tracking", "micro-doppler"):
            internal = self._view_mode_box.model().item(self._view_mode_box.findData(token))
            public = self.workspace_box.model().item(self.workspace_box.findData(token))
            public.setEnabled(internal.isEnabled())
            public.setToolTip(internal.toolTip())
        if group == "current":
            angular = self._view_mode_box.model().item(self._view_mode_box.findData("angular"))
            with QSignalBlocker(self.workspace_tabs):
                self.workspace_tabs.setTabEnabled(2, angular.isEnabled())
                self.workspace_tabs.setTabToolTip(2, angular.toolTip())
                capabilities = getattr(self, "_recorded_capabilities", {})
                for index, key in ((1, "cfar"), (3, "inputs")):
                    if key in capabilities:
                        enabled, reason = capabilities[key]
                        if key == "inputs" and not enabled:
                            enabled = capabilities.get("frequency", (False, ""))[0]
                        self.workspace_tabs.setTabEnabled(index, enabled)
                        self.workspace_tabs.setTabToolTip(index, "" if enabled else reason)
                self.workspace_tabs.setCurrentIndex(selected)
        history = None
        if mode == "tracking" and self._last_request is not None:
            history = self._last_request.tracking_history or self._last_request.recording_history
        prepared = history.tracking_plot_data if history is not None else None
        if mode == "tracking":
            self._set_tracking_projection_capability(prepared)
        self.tracking_projection.setVisible(
            mode == "tracking"
            and not self.tracking_plot.isHidden()
            and getattr(prepared, "spatial_dimensions", 0) == 3
        )
