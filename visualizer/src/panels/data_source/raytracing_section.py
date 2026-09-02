"""Raytracing parameter subsection for live-gRPC data sources.

The controls mirror generator quality presets plus a custom override path. On
apply, parameter updates go through the live gRPC provider and then invalidate
frame-data caches so the current frame can be recomputed.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from generator.core.configuration import QUALITY_PRESETS
from shared.logging import get_logger

from ...services.raytracing_settings_service import (
    DEFAULT_RAYTRACING_PRESET,
    RaytracingSettingsService,
)
from ..ui_theme import configure_label, set_widget_role

logger = get_logger("orchav.raytracing_section")

RT_DEFAULT_SEED = 42
RT_SEED_MIN = 0
RT_SEED_MAX = 999_999

RT_SAMPLES_PER_SRC_MIN = 1_000
RT_SAMPLES_PER_SRC_MAX = 100_000_000
RT_SAMPLES_PER_SRC_STEP = 10_000
RT_SAMPLES_PER_SRC_DEFAULT = 100_000

RT_MAX_NUM_PATHS_PER_SRC_MIN = 1_000
RT_MAX_NUM_PATHS_PER_SRC_MAX = 100_000_000
RT_MAX_NUM_PATHS_PER_SRC_STEP = 10_000
RT_MAX_NUM_PATHS_PER_SRC_DEFAULT = 100_000

RT_PRESET_OPTIONS = ("ultra-low", "low", "medium", "high", "ultra", "custom")


class RaytracingControlSection:
    """Build and apply live raytracing quality/phenomenon controls.

    Args:
        parent: The visualizer instance.
        widgets: Shared widget registry dict from DataSourcePanel.
        button_style_fn: Callable returning QPushButton stylesheet string.
        group_title: Heading shown above the controls.
        show_apply_button: Whether to include the live-gRPC apply action.
        status_text: Initial helper text shown below the controls.
        initial_preset: Preset selected when the controls are created.
        sync_initial_preset: Whether construction also updates shared solver
            settings. User changes always update the shared settings.
    """

    def __init__(
        self,
        parent: Any,
        widgets: Dict[str, Any],
        button_style_fn: Callable[[], str],
        *,
        group_title: str = "Raytracing Control",
        show_apply_button: bool = True,
        status_text: str = "",
        initial_preset: str = DEFAULT_RAYTRACING_PRESET,
        sync_initial_preset: bool = True,
    ) -> None:
        """Initialize preset/custom-parameter state and shared widgets."""
        self.parent = parent
        self.widgets = widgets
        self._get_button_style = button_style_fn
        self._group_title = str(group_title)
        self._show_apply_button = bool(show_apply_button)
        self._initial_status_text = str(status_text)
        requested_preset = str(initial_preset or DEFAULT_RAYTRACING_PRESET)
        self._initial_preset = (
            requested_preset if requested_preset in RT_PRESET_OPTIONS else DEFAULT_RAYTRACING_PRESET
        )
        self._sync_initial_preset = bool(sync_initial_preset)
        self._rt_current_preset = self._initial_preset
        self._rt_custom_params: Dict[str, Any] = {}
        self._loading_rt_parameters = False
        self._local_settings_service: RaytracingSettingsService | None = None

    def create_content(self) -> QWidget:
        """Create raytracing control content widget.

        Returns:
            A QWidget containing the raytracing control group.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._create_raytracing_control_group())

        return container

    def sync_from_scenario(self, scenario: Any) -> None:
        """Load the authored raytracing quality into the visible controls."""
        raytracing = getattr(scenario, "raytracing", None)
        if not isinstance(raytracing, Mapping):
            raytracing = {}
        quality = raytracing.get("quality") or {}
        if not isinstance(quality, Mapping):
            quality = {}

        requested_preset = str(quality.get("preset") or DEFAULT_RAYTRACING_PRESET)
        if requested_preset not in RT_PRESET_OPTIONS:
            requested_preset = DEFAULT_RAYTRACING_PRESET
        custom = quality.get("custom") or {}
        if not isinstance(custom, Mapping):
            custom = {}

        if custom or requested_preset == "custom":
            base_name = (
                requested_preset
                if requested_preset in QUALITY_PRESETS
                else DEFAULT_RAYTRACING_PRESET
            )
            merged = dict(QUALITY_PRESETS[base_name])
            merged.update(dict(custom))
            selected_preset = "custom"
            self._rt_custom_params = merged
        else:
            selected_preset = requested_preset
            self._rt_custom_params = {}

        self._initial_preset = selected_preset
        combo = self.widgets.get("rt_preset_combo")
        if combo is None:
            self._rt_current_preset = selected_preset
            if selected_preset == "custom":
                self._settings_service().set_custom(self._rt_custom_params)
            else:
                self._settings_service().set_preset(selected_preset)
            return

        with QSignalBlocker(combo):
            combo.setCurrentText(selected_preset)
        self._on_rt_preset_changed(selected_preset)

    def _create_raytracing_control_group(self) -> QGroupBox:
        """Create preset, custom parameter, phenomenon, and apply controls."""
        group = QGroupBox(self._group_title)
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        preset_layout = QHBoxLayout()
        preset_label = QLabel("Preset:")
        configure_label(preset_label, role="secondary", min_width=100)
        preset_layout.addWidget(preset_label)

        self.widgets["rt_preset_combo"] = QComboBox()
        self.widgets["rt_preset_combo"].addItems(RT_PRESET_OPTIONS)
        self.widgets["rt_preset_combo"].setCurrentText(self._initial_preset)
        self.widgets["rt_preset_combo"].currentTextChanged.connect(self._on_rt_preset_changed)
        preset_layout.addWidget(self.widgets["rt_preset_combo"])
        preset_layout.addStretch()
        layout.addLayout(preset_layout)

        params_layout = QGridLayout()
        params_layout.setSpacing(4)

        depth_label = QLabel("Max Depth:")
        configure_label(depth_label, role="secondary")
        self.widgets["rt_max_depth"] = QSpinBox()
        self.widgets["rt_max_depth"].setRange(1, 10)
        self.widgets["rt_max_depth"].setValue(4)
        self.widgets["rt_max_depth"].valueChanged.connect(self._on_rt_custom_parameter_changed)
        params_layout.addWidget(depth_label, 0, 0)
        params_layout.addWidget(self.widgets["rt_max_depth"], 0, 1)

        samples_label = QLabel("Samples/Source:")
        configure_label(samples_label, role="secondary")
        self.widgets["rt_samples_per_src"] = QSpinBox()
        self.widgets["rt_samples_per_src"].setRange(RT_SAMPLES_PER_SRC_MIN, RT_SAMPLES_PER_SRC_MAX)
        self.widgets["rt_samples_per_src"].setValue(RT_SAMPLES_PER_SRC_DEFAULT)
        self.widgets["rt_samples_per_src"].setSingleStep(RT_SAMPLES_PER_SRC_STEP)
        self.widgets["rt_samples_per_src"].valueChanged.connect(
            self._on_rt_custom_parameter_changed
        )
        params_layout.addWidget(samples_label, 0, 2)
        params_layout.addWidget(self.widgets["rt_samples_per_src"], 0, 3)

        paths_label = QLabel("Max Paths/Source:")
        configure_label(paths_label, role="secondary")
        self.widgets["rt_max_num_paths_per_src"] = QSpinBox()
        self.widgets["rt_max_num_paths_per_src"].setRange(
            RT_MAX_NUM_PATHS_PER_SRC_MIN, RT_MAX_NUM_PATHS_PER_SRC_MAX
        )
        self.widgets["rt_max_num_paths_per_src"].setValue(RT_MAX_NUM_PATHS_PER_SRC_DEFAULT)
        self.widgets["rt_max_num_paths_per_src"].setSingleStep(RT_MAX_NUM_PATHS_PER_SRC_STEP)
        self.widgets["rt_max_num_paths_per_src"].valueChanged.connect(
            self._on_rt_custom_parameter_changed
        )
        params_layout.addWidget(paths_label, 1, 0)
        params_layout.addWidget(self.widgets["rt_max_num_paths_per_src"], 1, 1)

        seed_label = QLabel("Seed:")
        configure_label(seed_label, role="secondary")
        self.widgets["rt_seed"] = QSpinBox()
        self.widgets["rt_seed"].setRange(RT_SEED_MIN, RT_SEED_MAX)
        self.widgets["rt_seed"].setValue(RT_DEFAULT_SEED)
        self.widgets["rt_seed"].valueChanged.connect(self._on_rt_custom_parameter_changed)
        params_layout.addWidget(seed_label, 1, 2)
        params_layout.addWidget(self.widgets["rt_seed"], 1, 3)

        layout.addLayout(params_layout)

        checkboxes_layout = QHBoxLayout()
        for name, key, default in [
            ("LOS", "rt_los", True),
            ("Specular", "rt_specular", True),
            ("Diffuse", "rt_diffuse", True),
            ("Refraction", "rt_refraction", False),
            ("Diffraction", "rt_diffraction", False),
        ]:
            cb = QCheckBox(name)
            cb.setChecked(default)
            cb.toggled.connect(self._on_rt_custom_parameter_changed)
            self.widgets[key] = cb
            checkboxes_layout.addWidget(cb)
        checkboxes_layout.addStretch()
        layout.addLayout(checkboxes_layout)

        if self._show_apply_button:
            apply_btn = QPushButton("Apply Raytracing Changes")
            apply_btn.setStyleSheet(self._get_button_style())
            apply_btn.clicked.connect(self._on_rt_apply_clicked)
            self.widgets["rt_apply_btn"] = apply_btn
            layout.addWidget(apply_btn)

        self.widgets["rt_status_label"] = QLabel(self._initial_status_text)
        configure_label(
            self.widgets["rt_status_label"], role="secondary", font_size=10, word_wrap=True
        )
        layout.addWidget(self.widgets["rt_status_label"])

        # Apply preset defaults once so custom controls start disabled.
        self._on_rt_preset_changed(
            self._initial_preset,
            sync_settings=self._sync_initial_preset,
        )

        return group

    # -- Preset Handlers ------------------------------------------------------

    def _on_rt_preset_changed(
        self,
        preset_name: str,
        *,
        sync_settings: bool = True,
    ) -> None:
        """Load generator preset values or enable custom raytracing controls."""
        self._rt_current_preset = preset_name

        if preset_name == "custom":
            self._set_rt_parameters_enabled(True)
            if self._rt_custom_params:
                self._load_rt_parameters(self._rt_custom_params)
            if sync_settings:
                self._sync_custom_settings_from_widgets()
        else:
            preset = QUALITY_PRESETS.get(preset_name, QUALITY_PRESETS["medium"])
            self._load_rt_parameters(preset)
            self._set_rt_parameters_enabled(False)
            if sync_settings:
                self._settings_service().set_preset(preset_name)

    def _load_rt_parameters(self, params: Dict[str, Any]) -> None:
        """Load raytracing parameter values into widgets."""
        self._loading_rt_parameters = True
        try:
            self.widgets["rt_max_depth"].setValue(params.get("max_depth", 4))
            self.widgets["rt_samples_per_src"].setValue(
                params.get("samples_per_src", RT_SAMPLES_PER_SRC_DEFAULT)
            )
            self.widgets["rt_max_num_paths_per_src"].setValue(
                params.get("max_num_paths_per_src", RT_MAX_NUM_PATHS_PER_SRC_DEFAULT)
            )
            self.widgets["rt_seed"].setValue(params.get("seed", RT_DEFAULT_SEED))
            self.widgets["rt_los"].setChecked(params.get("los", True))
            self.widgets["rt_specular"].setChecked(params.get("specular_reflection", True))
            self.widgets["rt_diffuse"].setChecked(params.get("diffuse_reflection", True))
            self.widgets["rt_refraction"].setChecked(params.get("refraction", False))
            self.widgets["rt_diffraction"].setChecked(params.get("diffraction", False))
        finally:
            self._loading_rt_parameters = False

    def _set_rt_parameters_enabled(self, enabled: bool) -> None:
        """Enable parameters only for the custom preset path."""
        for key in [
            "rt_max_depth",
            "rt_samples_per_src",
            "rt_max_num_paths_per_src",
            "rt_seed",
            "rt_los",
            "rt_specular",
            "rt_diffuse",
            "rt_refraction",
            "rt_diffraction",
        ]:
            self.widgets[key].setEnabled(enabled)

    def _on_rt_custom_parameter_changed(self, *_args: Any) -> None:
        """Mirror visible custom edits into the shared preview settings."""
        if self._loading_rt_parameters or self._rt_current_preset != "custom":
            return
        self._sync_custom_settings_from_widgets()

    def current_raytracing_config(self) -> tuple[str, dict[str, Any]]:
        """Return the preset label and normalized settings currently shown."""
        preset = self._rt_current_preset
        if preset == "custom":
            config = self._current_widget_config()
            self._rt_custom_params = dict(config)
            normalized = self._settings_service().set_custom(config)
            return "custom", normalized
        normalized = self._settings_service().set_preset(preset)
        return preset, normalized

    def _sync_custom_settings_from_widgets(self) -> None:
        """Update shared settings from enabled custom controls."""
        self._rt_custom_params = self._current_widget_config()
        self._settings_service().set_custom(self._rt_custom_params)

    def _current_widget_config(self) -> dict[str, Any]:
        """Read the raytracing widgets into a solver-settings mapping."""
        return {
            "max_depth": self.widgets["rt_max_depth"].value(),
            "samples_per_src": self.widgets["rt_samples_per_src"].value(),
            "max_num_paths_per_src": self.widgets["rt_max_num_paths_per_src"].value(),
            "seed": self.widgets["rt_seed"].value(),
            "los": self.widgets["rt_los"].isChecked(),
            "specular_reflection": self.widgets["rt_specular"].isChecked(),
            "diffuse_reflection": self.widgets["rt_diffuse"].isChecked(),
            "refraction": self.widgets["rt_refraction"].isChecked(),
            "diffraction": self.widgets["rt_diffraction"].isChecked(),
            "synthetic_array": True,
        }

    def _settings_service(self) -> RaytracingSettingsService:
        """Return the app-level settings service, with a local test fallback."""
        service = getattr(self.parent, "raytracing_settings_service", None)
        required = ("set_preset", "set_custom", "release_settings", "drag_settings")
        if service is not None and all(callable(getattr(service, name, None)) for name in required):
            return service
        if self._local_settings_service is None:
            self._local_settings_service = RaytracingSettingsService()
        return self._local_settings_service

    def _on_rt_apply_clicked(self) -> None:
        """Send a ParameterUpdate via gRPC and invalidate frame-data caches."""
        try:
            from ...io.frame_sources import LiveGrpcSource

            if not isinstance(self.parent.frame_source, LiveGrpcSource):
                self.widgets["rt_status_label"].setText("Error: Not in live gRPC mode")
                set_widget_role(self.widgets["rt_status_label"], "error")
                return

            provider = self.parent.frame_source.provider
            if not provider:
                self.widgets["rt_status_label"].setText("Error: No gRPC provider available")
                set_widget_role(self.widgets["rt_status_label"], "error")
                return

            preset, config = self.current_raytracing_config()

            self.widgets["rt_apply_btn"].setEnabled(False)
            self.widgets["rt_status_label"].setText("Applying changes...")
            set_widget_role(self.widgets["rt_status_label"], "accent")

            if hasattr(provider, "update_raytracing_params"):
                response = provider.update_raytracing_params(preset, config, flush_cache=True)
                if response and getattr(response, "success", False):
                    final_message = response.message or f"Applied {preset} preset successfully"
                    self.widgets["rt_status_label"].setText(final_message)
                    set_widget_role(self.widgets["rt_status_label"], "success")

                    self._reload_current_frame()
                else:
                    error_message = "Failed to apply parameters"
                    if response is None:
                        error_message = "Timed out waiting for parameter update response"
                    elif getattr(response, "message", ""):
                        error_message = response.message
                    self.widgets["rt_status_label"].setText(f"Warning: {error_message}")
                    set_widget_role(self.widgets["rt_status_label"], "warning")
            else:
                self.widgets["rt_status_label"].setText(
                    "Warning: Parameter update not yet implemented on server"
                )
                set_widget_role(self.widgets["rt_status_label"], "warning")

            self.widgets["rt_apply_btn"].setEnabled(True)

        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Error applying raytracing parameters: %s", e)
            self.widgets["rt_status_label"].setText(f"Error: {e}")
            set_widget_role(self.widgets["rt_status_label"], "error")
            self.widgets["rt_apply_btn"].setEnabled(True)

    def _reload_current_frame(self) -> bool:
        """Ask the frame-refresh service to reload after raytracing changes."""
        refresh_service = getattr(self.parent, "frame_refresh_service", None)
        refresh = getattr(refresh_service, "refresh_current_frame_after_data_change", None)
        if not callable(refresh):
            logger.warning("Frame refresh service unavailable after raytracing parameter update")
            return False
        return bool(refresh(reason="PARAM_UPDATE"))
