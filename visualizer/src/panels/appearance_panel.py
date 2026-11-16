"""Compact pygfx appearance selector, reusing existing customization controls."""

from PySide6.QtCore import QSignalBlocker, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..materials.view_styles import DESCRIPTIONS, PRESETS
from .base import BasePanel


class AppearancePanel(BasePanel):
    """Select global looks while keeping surface and lighting controls accessible."""

    def __init__(self, parent, manager):
        super().__init__(parent)
        self.manager = manager
        self._demo_service = None
        self._demo_render_sync = None

    def create_panel(self):
        self.service = self.parent.appearance_preset_service
        self.group = self.create_group_box("Appearance")
        layout = QVBoxLayout(self.group)
        self.combo = QComboBox()
        self.combo.setToolTip(
            "Apply a scene look. MPC colors, sizes and visibility remain controlled under Paths. "
            "Camera, frame, path filters, strongest-path limit, coverage and sensing settings are preserved."
        )
        self.combo.activated.connect(
            lambda: self.run(lambda: self.service.apply(self.combo.currentText()))
        )
        layout.addWidget(self.combo)
        self.description = QLabel()
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.plan_service = getattr(getattr(self.parent, "scene_service", None), "plan_view", None)
        self.plan_status = QLabel()
        self.plan_status.setWordWrap(True)
        layout.addWidget(self.plan_status)
        if self.plan_service is not None:
            self.plan_service.changed.connect(self.plan_changed)
        self.sync_plan_status()
        row = QHBoxLayout()
        for text, callback in (
            ("Save as…", self.save),
            ("Reset", self.reset),
            ("Previous look", self.undo),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)
            self.widgets[text] = button
        layout.addLayout(row)
        self.customize = QCheckBox("Customize appearance")
        self.customize.setChecked(True)
        self.customize.toggled.connect(self.show_customization)
        layout.addWidget(self.customize)
        self.details = QWidget(self.group)
        details = QVBoxLayout(self.details)
        details.setContentsMargins(0, 0, 0, 0)
        surface_row = QHBoxLayout()
        surface_label = QLabel("Surface colors")
        surface_row.addWidget(surface_label)
        self.surface = QComboBox()
        for label, value in (
            ("Authored", "original"),
            ("Uniform color", "geometry"),
            ("Radio materials", "radio"),
            ("Dark", "dark"),
            ("Glow", "glow"),
        ):
            self.surface.addItem(label, value)
        self.surface.setToolTip(
            "Authored keeps the scene's visual materials and enabled textures. Uniform color uses "
            "a uniform scene color; Radio materials uses colors from the radio-material catalog. "
            "Color modes preserve target colors and textures."
        )
        surface_label.setBuddy(self.surface)
        self.widgets["surface_colors_label"] = surface_label
        self.surface.activated.connect(self.edit_style)
        surface_row.addWidget(self.surface, 1)
        self.unlit = QCheckBox("Unlit")
        self.unlit.setToolTip("Show surface colors without lighting or shading.")
        self.unlit.clicked.connect(self.edit_style)
        surface_row.addWidget(self.unlit)
        details.addLayout(surface_row)
        bloom_row = QHBoxLayout()
        self.bloom = QCheckBox("Bloom")
        self.bloom.setToolTip("Add a soft glow around bright scene details and paths.")
        self.bloom_strength = QDoubleSpinBox()
        self.bloom_strength.setRange(0, 3)
        self.bloom_strength.setDecimals(3)
        self.bloom_strength.setSingleStep(0.01)
        self.bloom_strength.setToolTip("Bloom strength. Small values keep path cores readable.")
        bloom_row.addWidget(self.bloom)
        bloom_row.addWidget(self.bloom_strength)
        self.bloom.clicked.connect(self.edit_bloom)
        self.bloom_strength.valueChanged.connect(self.edit_bloom)
        details.addLayout(bloom_row)
        # Move, rather than duplicate, the existing global texture/theme controls.
        materials = self.manager.panels["materials"]
        details.addWidget(materials.surface_controls)
        self.details.hide()
        self.service.changed.connect(self.refresh)
        self._edit_timer = QTimer(self.group)
        self._edit_timer.setSingleShot(True)
        self._edit_timer.setInterval(250)  # Follow opacity-slider debounce.
        self._edit_timer.timeout.connect(self.service.mark_if_custom)
        self.reset_name = None
        self.refresh()
        return self.group

    def connect_customization(self):
        """Observe UI edits once, never poll during playback."""
        panels = self.manager.panels
        for key in ("render", "materials"):
            for widget in panels[key].widgets.values():
                for signal_name in ("valueChanged", "currentIndexChanged", "toggled"):
                    signal = getattr(widget, signal_name, None)
                    if signal is not None:
                        signal.connect(self.queue_custom_check)
                        break
        download = panels["render"].widgets.get("environment_download")
        if download is not None:
            download.ready.connect(self.queue_custom_check)
        # Demo can change just the environment after applying a look. Mirror
        # that final state without marking the temporary settings as Custom.
        self._demo_service = getattr(self.parent, "demo_service", None)
        if self._demo_service is not None:
            self._demo_render_sync = panels["render"]._sync_from_visualizer
            self._demo_service.changed.connect(self._demo_render_sync)
        # Reuse the texture controls in Surfaces, alongside opacity and edges.
        surface_panel = self.manager.sections["scene_style"].content_layout().itemAt(0).widget()
        surface_panel.layout().insertWidget(0, self.details)
        mipmaps = panels["render"].widgets["texture_mipmaps_cb"]
        mipmaps.setText("Mipmaps")
        panels["materials"].texture_maps_row.addWidget(mipmaps)
        self.show_customization(self.customize.isChecked())

    def queue_custom_check(self, *_):
        if not self.service.applying:
            self._edit_timer.start()

    def show_customization(self, visible):
        self.details.setVisible(visible)
        for key in ("scene_style", "lighting"):
            section = self.manager.sections.get(key)
            if section is not None:
                section.setVisible(visible)

    def refresh(self):
        name = self.service.active_name
        plan_available = self.plan_service is not None and self.plan_service.available
        with QSignalBlocker(self.combo):
            names = [key for key in PRESETS if key != "Plan" or plan_available]
            names += self.service.names() + ["Custom"]
            if [self.combo.itemText(i) for i in range(self.combo.count())] != names:
                self.combo.clear()
                self.combo.addItems(names)
            self.combo.setCurrentText(name)
            self.combo.model().item(self.combo.findText("Custom")).setEnabled(False)
        if name != "Custom":
            self.reset_name = name
        self.widgets["Reset"].setEnabled(self.reset_name in names)
        self.widgets["Previous look"].setEnabled(self.service.previous is not None)
        self.description.setText(
            self.service.notice
            or DESCRIPTIONS.get(name, "Your current global appearance settings.")
        )
        with (
            QSignalBlocker(self.surface),
            QSignalBlocker(self.unlit),
            QSignalBlocker(self.bloom),
            QSignalBlocker(self.bloom_strength),
        ):
            self.surface.setCurrentIndex(self.surface.findData(self.service.surface))
            self.unlit.setChecked(self.service.unlit)
            bloom = self.parent.renderer.get_bloom_settings()
            self.bloom.setChecked(bloom["enabled"])
            self.bloom_strength.setValue(bloom["bloom_strength"])
            self.bloom_strength.setEnabled(bloom["enabled"])
        panels = self.manager.panels
        panels["render"]._sync_from_visualizer()
        panels["materials"].sync_procedural_controls()

    def plan_changed(self):
        self.sync_plan_status()
        self.refresh()

    def sync_plan_status(self):
        service = self.plan_service
        self.plan_status.setText(service.status if service is not None else "")
        self.plan_status.setVisible(
            bool(
                service is not None
                and service.status
                and (service.enabled or not service.available)
            )
        )

    def edit_style(self, *_):
        data = self.service.capture()
        data.update(
            surface=self.surface.currentData(),
            unlit=self.unlit.isChecked(),
            wireframe=self.parent.renderer.get_wireframe(),
        )
        self.run(lambda: self.service.restore(data))

    def edit_bloom(self, *_):
        settings = self.parent.renderer.get_bloom_settings()
        settings.update(enabled=self.bloom.isChecked(), bloom_strength=self.bloom_strength.value())

        def apply():
            self.parent.renderer.set_bloom_settings(settings)
            self.service.mark_if_custom()
            self.refresh()

        self.run(apply)

    def run(self, action):
        self.group.setEnabled(False)
        self.details.setEnabled(False)
        try:
            action()
        except (ValueError, OSError, RuntimeError) as exc:
            QMessageBox.warning(self.group, "Appearance", str(exc))
            self.refresh()
        finally:
            self.group.setEnabled(True)
            self.details.setEnabled(True)

    def reset(self):
        if self.reset_name:
            self.run(lambda: self.service.apply(self.reset_name))

    def undo(self):
        if self.service.previous is not None:
            self.run(lambda: self.service.restore(self.service.previous))

    def save(self):
        name, accepted = QInputDialog.getText(self.group, "Save appearance", "Preset name:")
        if not accepted:
            return
        name = name.strip()
        if (
            name in self.service.names()
            and QMessageBox.question(self.group, "Replace preset", f"Replace '{name}'?")
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.run(lambda: self.service.save(name))

    def cleanup(self):
        self._edit_timer.stop()
        if self.plan_service is not None:
            self.plan_service.changed.disconnect(self.plan_changed)
        self.service.changed.disconnect(self.refresh)
        if self._demo_service is not None and self._demo_render_sync is not None:
            self._demo_service.changed.disconnect(self._demo_render_sync)
            self._demo_service = None
            self._demo_render_sync = None
