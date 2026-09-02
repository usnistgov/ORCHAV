"""Application state definitions for the ORCHAV visualizer."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Union

from .utils.antenna_utils import beamforming_defaults_from_scenario_config

ColorMode = Literal[
    "reflection_order", "mpc_type", "delay", "path_loss", "material", "reconstruction_type"
]
NodeSelection = Union[int, Literal["all"]]
CameraMode = Literal["overview", "follow", "pov"]
POVAxis = Literal["forward", "x", "y", "z", "-x", "-y", "-z"]
TrajectoryColorMode = Literal["node_color", "speed", "altitude", "time", "angular_speed"]
NodeLabelMode = Literal["role", "name"]
HoverInfoMode = Literal["off", "essential", "inspect_all"]
ViewportHudMode = Literal["compact", "detailed"]
RFXRayMode = Literal["material_map", "mpc_usage", "material_properties"]
RFXRayProperty = Literal[
    "relative_permittivity",
    "conductivity",
    "scattering_coefficient",
    "xpd_coefficient",
    "thickness",
]
DEFAULT_RF_XRAY_OPACITY = 0.85
MIN_RF_XRAY_OPACITY = 0.05
MAX_RF_XRAY_OPACITY = 1.0
DEFAULT_RF_XRAY_PROPERTY: RFXRayProperty = "scattering_coefficient"
RF_XRAY_PROPERTY_OPTIONS: tuple[RFXRayProperty, ...] = (
    "relative_permittivity",
    "conductivity",
    "scattering_coefficient",
    "xpd_coefficient",
    "thickness",
)
HOVER_INFO_MODE_OPTIONS: tuple[HoverInfoMode, ...] = ("off", "essential", "inspect_all")
VIEWPORT_HUD_MODE_OPTIONS: tuple[ViewportHudMode, ...] = (
    "compact",
    "detailed",
)

# Canonical MPC filter values. The UI, presets, session synchronization, and
# initial state all use these tuples so supported interaction types cannot
# silently disappear in one workflow.
MPC_ORDER_VALUES: tuple[int, ...] = tuple(range(7))
MPC_TYPE_VALUES: tuple[int, ...] = (0, 1, 2, 4, 8, 99)
DEFAULT_MPC_ALLOWED_ORDERS = frozenset(MPC_ORDER_VALUES)
DEFAULT_MPC_ALLOWED_TYPES = frozenset(MPC_TYPE_VALUES)
_LEGACY_DEFAULT_MPC_ALLOWED_TYPES = frozenset((0, 1, 2, 4, 8))
_MPC_TYPE_FILTER_SCHEMA_VERSION = 2

BEAMFORMING_STATE_KEYS: tuple[str, ...] = (
    "show_beamforming",
    "beamforming_azimuth_samples",
    "beamforming_elevation_samples",
    "beamforming_tx_scale",
    "beamforming_rx_scale",
    "beamforming_tx_node",
    "beamforming_rx_node",
    "standalone_beamforming_mode",
    "standalone_antenna_rows",
    "standalone_antenna_cols",
    "standalone_horizontal_spacing_m",
    "standalone_vertical_spacing_m",
    "standalone_carrier_frequency_ghz",
    "standalone_steering_strategy",
    "standalone_azimuth_deg",
    "standalone_elevation_deg",
    "beamforming_db_scale",
    "beamforming_dynamic_range_db",
    "beamforming_colormap",
    "beamforming_element_pattern",
    "beamforming_tx_element_pattern",
    "beamforming_rx_element_pattern",
    "beamforming_pattern_status",
)


def normalize_rf_xray_opacity(value: Any) -> float:
    """Clamp RF X-Ray overlay opacity to the UI-supported range."""
    try:
        opacity = float(value)
    except (TypeError, ValueError):
        opacity = DEFAULT_RF_XRAY_OPACITY
    return max(MIN_RF_XRAY_OPACITY, min(MAX_RF_XRAY_OPACITY, opacity))


def normalize_rf_xray_mode(value: Any) -> RFXRayMode:
    """Return a supported RF X-Ray mode."""
    mode = str(value or "").strip().lower()
    if mode in {"mpc_usage", "mpc-material-usage", "mpc_material_usage"}:
        return "mpc_usage"
    if mode in {"material_properties", "material-property", "material-properties"}:
        return "material_properties"
    return "material_map"


def normalize_rf_xray_property(value: Any) -> RFXRayProperty:
    """Return a supported RF X-Ray material-property field."""
    prop = str(value or "").strip().lower()
    aliases = {
        "epsilon": "relative_permittivity",
        "epsilon_r": "relative_permittivity",
        "eps_r": "relative_permittivity",
        "permittivity": "relative_permittivity",
        "sigma": "conductivity",
        "scattering": "scattering_coefficient",
        "scatter": "scattering_coefficient",
        "xpd": "xpd_coefficient",
        "xpd_coeff": "xpd_coefficient",
        "material_thickness": "thickness",
    }
    prop = aliases.get(prop, prop)
    if prop in RF_XRAY_PROPERTY_OPTIONS:
        return prop  # type: ignore[return-value]
    return DEFAULT_RF_XRAY_PROPERTY


def normalize_hover_info_mode(value: Any) -> HoverInfoMode:
    """Return a supported renderer hover-tooltip policy."""
    mode = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "none": "off",
        "disabled": "off",
        "disable": "off",
        "minimal": "essential",
        "essentials": "essential",
        "rays": "essential",
        "ray": "essential",
        "all": "inspect_all",
        "inspect": "inspect_all",
        "full": "inspect_all",
        "debug": "inspect_all",
    }
    mode = aliases.get(mode, mode)
    if mode in HOVER_INFO_MODE_OPTIONS:
        return mode  # type: ignore[return-value]
    return "essential"


def normalize_viewport_hud_mode(value: Any) -> ViewportHudMode:
    """Return a supported viewport-HUD detail policy."""
    mode = str(value or "").strip().lower()
    if mode in VIEWPORT_HUD_MODE_OPTIONS:
        return mode  # type: ignore[return-value]
    return "compact"


def _legacy_viewport_hud_mode_is_off(value: Any) -> bool:
    """Return whether the pre-split combined HUD setting requested hidden state."""
    return str(value or "").strip().lower() == "off"


@dataclass(frozen=True, slots=True)
class MpcVisibility:
    """User intent for the MPC layer and its independently selectable parts."""

    enabled: bool = True
    paths: bool = True
    bounce_points: bool = True

    @property
    def effective_paths(self) -> bool:
        """Return whether MPC path segments should be presented."""
        return self.enabled and self.paths

    @property
    def effective_bounce_points(self) -> bool:
        """Return whether physical MPC bounce points should be presented."""
        return self.enabled and self.bounce_points

    def to_dict(self) -> dict[str, bool]:
        """Return the JSON representation stored in application sessions."""
        return {
            "enabled": self.enabled,
            "paths": self.paths,
            "bounce_points": self.bounce_points,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MpcVisibility":
        """Build MPC visibility from its current serialized representation."""
        return cls(
            enabled=bool(data.get("enabled", True)),
            paths=bool(data.get("paths", True)),
            bounce_points=bool(data.get("bounce_points", True)),
        )


@dataclass(frozen=True)
class AppState:
    """Immutable snapshot describing what the visualizer should display."""

    step: int
    selected_tx: NodeSelection
    selected_rx: NodeSelection
    mpc_visibility: MpcVisibility
    mpc_allowed_orders: frozenset[int]
    mpc_allowed_types: frozenset[int]
    color_mode: ColorMode
    show_labels: bool
    sync_target_position: bool
    camera_mode: CameraMode = "overview"
    fly_mode: bool = False
    pov_axis: POVAxis = "forward"
    pov_hidden_node: tuple[str, int] | None = None  # (type, index) of hidden node in POV mode
    show_camera_minimap: bool = False
    show_beamforming: bool = False
    show_coverage: bool = False
    show_rf_xray: bool = False
    rf_xray_mode: RFXRayMode = "material_map"
    rf_xray_property: RFXRayProperty = DEFAULT_RF_XRAY_PROPERTY
    rf_xray_opacity: float = DEFAULT_RF_XRAY_OPACITY
    rf_xray_show_top_paths: bool = False
    rf_xray_max_top_paths: int = 12
    coverage_height_index: int = 0
    topk_render_enabled: bool = False
    topk_render_max_paths: int = 20000
    beamforming_azimuth_samples: int = 72
    beamforming_elevation_samples: int = 37
    beamforming_tx_scale: float = 1.5
    beamforming_rx_scale: float = 1.5
    beamforming_tx_node: str = "auto"
    beamforming_rx_node: str = "auto"
    # Standalone beamforming parameters
    standalone_beamforming_mode: str = "standalone"  # "frame" or "standalone"
    standalone_antenna_rows: int = 1
    standalone_antenna_cols: int = 1
    standalone_horizontal_spacing_m: float = 0.00535343675  # 0.5*lambda at 28 GHz
    standalone_vertical_spacing_m: float = 0.00535343675
    standalone_carrier_frequency_ghz: float = 28.0
    standalone_steering_strategy: str = "svd"  # "manual", "los", "svd"
    standalone_azimuth_deg: float = 0.0
    standalone_elevation_deg: float = 0.0
    # Beam pattern display options
    beamforming_db_scale: bool = False
    beamforming_dynamic_range_db: float = 40.0
    beamforming_colormap: str = "jet"  # "jet", "viridis", "hot", "coolwarm"
    beamforming_element_pattern: str = "isotropic"  # Backward-compatible alias
    beamforming_tx_element_pattern: str = "isotropic"
    beamforming_rx_element_pattern: str = "isotropic"
    beamforming_pattern_status: str = ""

    # Range filters (None = no bound)
    delay_filter_min_ns: float | None = None
    delay_filter_max_ns: float | None = None
    power_filter_min_db: float | None = None  # path loss dB (lower = stronger signal)
    power_filter_max_db: float | None = None

    # Angle filters (degrees, None = no bound)
    aoa_az_filter_min_deg: float | None = None  # Azimuth AoA
    aoa_az_filter_max_deg: float | None = None
    aoa_el_filter_min_deg: float | None = None  # Elevation AoA
    aoa_el_filter_max_deg: float | None = None
    aod_az_filter_min_deg: float | None = None  # Azimuth AoD
    aod_az_filter_max_deg: float | None = None
    aod_el_filter_min_deg: float | None = None  # Elevation AoD
    aod_el_filter_max_deg: float | None = None

    # Aperture visualization
    show_aoa_aperture: bool = False
    show_aod_aperture: bool = False
    aperture_radius_m: float = 5.0
    show_global_angular_reference: bool = False
    show_local_angular_reference: bool = False

    # 3D trajectory visualization
    show_tx_trajectory: bool = False
    show_rx_trajectory: bool = False
    show_target_trajectory: bool = False
    trajectory_color_mode: TrajectoryColorMode = "node_color"
    use_distinct_material_colors: bool = False

    # TX/RX label display. Runtime custom labels always win; role mode then
    # uses TX1/RX1, while name mode uses scenario/frame names before fallback.
    node_label_mode: NodeLabelMode = "role"
    tx_labels: tuple[str, ...] = ()
    rx_labels: tuple[str, ...] = ()
    tx_device_names: tuple[str, ...] = ()
    rx_device_names: tuple[str, ...] = ()
    target_labels: tuple[str, ...] = ()
    show_target_labels: bool = True
    # When True, pygfx labels use screen-space sizing (stable pixel size
    # regardless of camera distance). Has no effect on the Open3D renderer.
    label_screen_space: bool = True

    # Optional features keep transient values in their own namespace. These
    # values are intentionally excluded from session serialization.
    extension_state: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        compare=True,
        hash=False,
        repr=False,
    )

    # Cutaway / clipping planes (pygfx only).
    # Each axis has an enable flag, a world-space position in meters, and a
    # "flip" bool that reverses which half-space is kept.
    # flip=False keeps the negative side (e.g. z <= position); flip=True keeps
    # the positive side.
    clip_x_enabled: bool = False
    clip_x_position: float = 0.0
    clip_x_flip: bool = False
    clip_y_enabled: bool = False
    clip_y_position: float = 0.0
    clip_y_flip: bool = False
    clip_z_enabled: bool = False
    clip_z_position: float = 0.0
    clip_z_flip: bool = False

    # Pygfx display controls for interactive antialiasing and the ground grid.
    antialiasing_mode: str = "off"  # off | fxaa | ddaa | ppaa
    show_ground_grid: bool = False
    hover_info_mode: HoverInfoMode = "essential"
    viewport_hud_enabled: bool = True
    viewport_hud_mode: ViewportHudMode = "compact"
    viewport_hud_show_status: bool = True
    viewport_hud_show_legends: bool = True
    viewport_hud_show_filters: bool = True
    viewport_hud_show_annotations: bool = True
    # Distinguish MPC interaction points with canonical per-type glyphs.
    # The workspace schema fixes this persisted key name for snapshot compatibility.
    # Pygfx only; the data path uses canonical itype end-to-end.
    show_mpc_type_markers: bool = False

    def to_dict(self) -> dict:
        """Convert AppState to JSON-serializable dict.

        Returns:
            Dict with all AppState fields, with frozensets converted to lists
            and tuples converted to lists for JSON compatibility.
        """
        return {
            "step": self.step,
            "selected_tx": self.selected_tx,
            "selected_rx": self.selected_rx,
            "mpc_visibility": self.mpc_visibility.to_dict(),
            "mpc_allowed_orders": sorted(list(self.mpc_allowed_orders)),
            "mpc_allowed_types": sorted(list(self.mpc_allowed_types)),
            # Distinguish explicit "Virtual off" intent from workspaces written
            # before type 99 and the filter-schema marker were introduced.
            "mpc_type_filter_schema_version": _MPC_TYPE_FILTER_SCHEMA_VERSION,
            "color_mode": self.color_mode,
            "show_labels": self.show_labels,
            "sync_target_position": self.sync_target_position,
            "camera_mode": self.camera_mode,
            "fly_mode": self.fly_mode,
            "pov_axis": self.pov_axis,
            "pov_hidden_node": list(self.pov_hidden_node) if self.pov_hidden_node else None,
            "show_camera_minimap": self.show_camera_minimap,
            "show_beamforming": self.show_beamforming,
            "show_coverage": self.show_coverage,
            "show_rf_xray": self.show_rf_xray,
            "rf_xray_mode": self.rf_xray_mode,
            "rf_xray_property": self.rf_xray_property,
            "rf_xray_opacity": self.rf_xray_opacity,
            "rf_xray_show_top_paths": self.rf_xray_show_top_paths,
            "rf_xray_max_top_paths": self.rf_xray_max_top_paths,
            "coverage_height_index": self.coverage_height_index,
            "topk_render_enabled": self.topk_render_enabled,
            "topk_render_max_paths": self.topk_render_max_paths,
            "beamforming_azimuth_samples": self.beamforming_azimuth_samples,
            "beamforming_elevation_samples": self.beamforming_elevation_samples,
            "beamforming_tx_scale": self.beamforming_tx_scale,
            "beamforming_rx_scale": self.beamforming_rx_scale,
            "beamforming_tx_node": self.beamforming_tx_node,
            "beamforming_rx_node": self.beamforming_rx_node,
            "standalone_beamforming_mode": self.standalone_beamforming_mode,
            "standalone_antenna_rows": self.standalone_antenna_rows,
            "standalone_antenna_cols": self.standalone_antenna_cols,
            "standalone_horizontal_spacing_m": self.standalone_horizontal_spacing_m,
            "standalone_vertical_spacing_m": self.standalone_vertical_spacing_m,
            "standalone_carrier_frequency_ghz": self.standalone_carrier_frequency_ghz,
            "standalone_steering_strategy": self.standalone_steering_strategy,
            "standalone_azimuth_deg": self.standalone_azimuth_deg,
            "standalone_elevation_deg": self.standalone_elevation_deg,
            "beamforming_db_scale": self.beamforming_db_scale,
            "beamforming_dynamic_range_db": self.beamforming_dynamic_range_db,
            "beamforming_colormap": self.beamforming_colormap,
            "beamforming_element_pattern": self.beamforming_element_pattern,
            "beamforming_tx_element_pattern": self.beamforming_tx_element_pattern,
            "beamforming_rx_element_pattern": self.beamforming_rx_element_pattern,
            "beamforming_pattern_status": self.beamforming_pattern_status,
            # Range filters
            "delay_filter_min_ns": self.delay_filter_min_ns,
            "delay_filter_max_ns": self.delay_filter_max_ns,
            "power_filter_min_db": self.power_filter_min_db,
            "power_filter_max_db": self.power_filter_max_db,
            # Angle filters
            "aoa_az_filter_min_deg": self.aoa_az_filter_min_deg,
            "aoa_az_filter_max_deg": self.aoa_az_filter_max_deg,
            "aoa_el_filter_min_deg": self.aoa_el_filter_min_deg,
            "aoa_el_filter_max_deg": self.aoa_el_filter_max_deg,
            "aod_az_filter_min_deg": self.aod_az_filter_min_deg,
            "aod_az_filter_max_deg": self.aod_az_filter_max_deg,
            "aod_el_filter_min_deg": self.aod_el_filter_min_deg,
            "aod_el_filter_max_deg": self.aod_el_filter_max_deg,
            # Aperture visualization
            "show_aoa_aperture": self.show_aoa_aperture,
            "show_aod_aperture": self.show_aod_aperture,
            "aperture_radius_m": self.aperture_radius_m,
            "show_global_angular_reference": self.show_global_angular_reference,
            "show_local_angular_reference": self.show_local_angular_reference,
            # 3D trajectory visualization
            "show_tx_trajectory": self.show_tx_trajectory,
            "show_rx_trajectory": self.show_rx_trajectory,
            "show_target_trajectory": self.show_target_trajectory,
            "trajectory_color_mode": self.trajectory_color_mode,
            "use_distinct_material_colors": self.use_distinct_material_colors,
            # Node labels
            "node_label_mode": self.node_label_mode,
            "tx_labels": list(self.tx_labels),
            "rx_labels": list(self.rx_labels),
            "tx_device_names": list(self.tx_device_names),
            "rx_device_names": list(self.rx_device_names),
            "target_labels": list(self.target_labels),
            "show_target_labels": self.show_target_labels,
            "label_screen_space": self.label_screen_space,
            # Cutaway planes
            "clip_x_enabled": self.clip_x_enabled,
            "clip_x_position": self.clip_x_position,
            "clip_x_flip": self.clip_x_flip,
            "clip_y_enabled": self.clip_y_enabled,
            "clip_y_position": self.clip_y_position,
            "clip_y_flip": self.clip_y_flip,
            "clip_z_enabled": self.clip_z_enabled,
            "clip_z_position": self.clip_z_position,
            "clip_z_flip": self.clip_z_flip,
            # Capture mode
            "antialiasing_mode": self.antialiasing_mode,
            "show_ground_grid": self.show_ground_grid,
            "hover_info_mode": self.hover_info_mode,
            "viewport_hud_enabled": bool(self.viewport_hud_enabled),
            "viewport_hud_mode": normalize_viewport_hud_mode(self.viewport_hud_mode),
            "viewport_hud_show_status": self.viewport_hud_show_status,
            "viewport_hud_show_legends": self.viewport_hud_show_legends,
            "viewport_hud_show_filters": self.viewport_hud_show_filters,
            "viewport_hud_show_annotations": self.viewport_hud_show_annotations,
            "show_mpc_type_markers": self.show_mpc_type_markers,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AppState":
        """Create AppState from dict (inverse of to_dict).

        Args:
            data: Dict with AppState fields (from JSON)

        Returns:
            New AppState instance

        Note:
            Lists are converted back to frozensets/tuples where needed.
        """
        # Convert lists back to frozensets and tuples
        kwargs = data.copy()
        kwargs.pop("extension_state", None)

        visibility = kwargs.get("mpc_visibility")
        if isinstance(visibility, dict):
            kwargs["mpc_visibility"] = MpcVisibility.from_dict(visibility)
        elif not isinstance(visibility, MpcVisibility):
            kwargs["mpc_visibility"] = MpcVisibility()
        kwargs["mpc_allowed_orders"] = frozenset(kwargs.get("mpc_allowed_orders", []))
        type_filter_schema = int(kwargs.pop("mpc_type_filter_schema_version", 1))
        allowed_types = frozenset(kwargs.get("mpc_allowed_types", []))
        if (
            type_filter_schema < _MPC_TYPE_FILTER_SCHEMA_VERSION
            and allowed_types == _LEGACY_DEFAULT_MPC_ALLOWED_TYPES
        ):
            allowed_types = DEFAULT_MPC_ALLOWED_TYPES
        kwargs["mpc_allowed_types"] = allowed_types

        # Convert pov_hidden_node from list to tuple if present
        if kwargs.get("pov_hidden_node") is not None:
            kwargs["pov_hidden_node"] = tuple(kwargs["pov_hidden_node"])

        # Convert label lists back to tuples
        if "tx_labels" in kwargs and isinstance(kwargs["tx_labels"], list):
            kwargs["tx_labels"] = tuple(kwargs["tx_labels"])
        if "rx_labels" in kwargs and isinstance(kwargs["rx_labels"], list):
            kwargs["rx_labels"] = tuple(kwargs["rx_labels"])
        if "tx_device_names" in kwargs and isinstance(kwargs["tx_device_names"], list):
            kwargs["tx_device_names"] = tuple(kwargs["tx_device_names"])
        if "rx_device_names" in kwargs and isinstance(kwargs["rx_device_names"], list):
            kwargs["rx_device_names"] = tuple(kwargs["rx_device_names"])
        if "target_labels" in kwargs and isinstance(kwargs["target_labels"], list):
            kwargs["target_labels"] = tuple(kwargs["target_labels"])
        kwargs["node_label_mode"] = (
            "name"
            if str(kwargs.get("node_label_mode", "role")).strip().lower() == "name"
            else "role"
        )

        # Drop vestigial keys that may exist in saved sessions
        for old_key in [
            "standalone_tx_position",
            "standalone_tx_orientation",
            "standalone_rx_position",
            "standalone_rx_orientation",
            "coverage_stack_mode",
            "dash_by_itype",
        ]:
            kwargs.pop(old_key, None)

        legacy_pattern = kwargs.get("beamforming_element_pattern", "isotropic")
        kwargs.setdefault("beamforming_tx_element_pattern", legacy_pattern)
        kwargs.setdefault("beamforming_rx_element_pattern", legacy_pattern)
        kwargs.setdefault("beamforming_pattern_status", "")
        if kwargs.get("beamforming_tx_node") == "all":
            kwargs["beamforming_tx_node"] = "auto"
        if kwargs.get("beamforming_rx_node") == "all":
            kwargs["beamforming_rx_node"] = "auto"

        if kwargs.get("camera_mode", "overview") != "overview":
            kwargs["fly_mode"] = False

        kwargs.setdefault("show_rf_xray", False)
        kwargs["rf_xray_mode"] = normalize_rf_xray_mode(kwargs.get("rf_xray_mode"))
        kwargs["rf_xray_property"] = normalize_rf_xray_property(
            kwargs.get("rf_xray_property", DEFAULT_RF_XRAY_PROPERTY)
        )
        kwargs["rf_xray_opacity"] = normalize_rf_xray_opacity(
            kwargs.get("rf_xray_opacity", DEFAULT_RF_XRAY_OPACITY)
        )
        kwargs.pop("rf_xray_show_bounces", None)
        kwargs.setdefault("rf_xray_show_top_paths", False)
        kwargs["rf_xray_max_top_paths"] = max(1, int(kwargs.get("rf_xray_max_top_paths", 12)))
        kwargs["hover_info_mode"] = normalize_hover_info_mode(kwargs.get("hover_info_mode"))
        raw_hud_mode = kwargs.get("viewport_hud_mode", "compact")
        if "viewport_hud_enabled" in kwargs:
            kwargs["viewport_hud_enabled"] = bool(kwargs["viewport_hud_enabled"])
        else:
            kwargs["viewport_hud_enabled"] = not _legacy_viewport_hud_mode_is_off(raw_hud_mode)
        kwargs["viewport_hud_mode"] = normalize_viewport_hud_mode(raw_hud_mode)
        kwargs.setdefault("viewport_hud_show_status", True)
        kwargs.setdefault("viewport_hud_show_legends", True)
        kwargs.setdefault("viewport_hud_show_filters", True)
        kwargs.setdefault("viewport_hud_show_annotations", True)

        return cls(**kwargs)


def create_initial_state(
    *,
    step: int = 0,
    selected_tx: NodeSelection = "all",
    selected_rx: NodeSelection = "all",
    mpc_layer_enabled: bool = True,
    show_mpc_paths: bool = True,
    show_mpc_bounce_points: bool = True,
    mpc_allowed_orders: frozenset[int] | None = None,
    mpc_allowed_types: frozenset[int] | None = None,
    color_mode: ColorMode = "reflection_order",
    show_labels: bool = True,
    sync_target_position: bool = True,
    camera_mode: CameraMode = "overview",
    fly_mode: bool = False,
    pov_axis: POVAxis = "forward",
    pov_hidden_node: tuple[str, int] | None = None,
    show_camera_minimap: bool = False,
    show_beamforming: bool = False,
    show_coverage: bool = False,
    show_rf_xray: bool = False,
    rf_xray_mode: RFXRayMode = "material_map",
    rf_xray_property: RFXRayProperty = DEFAULT_RF_XRAY_PROPERTY,
    rf_xray_opacity: float = DEFAULT_RF_XRAY_OPACITY,
    rf_xray_show_top_paths: bool = False,
    rf_xray_max_top_paths: int = 12,
    coverage_height_index: int = 0,
    topk_render_enabled: bool = False,
    topk_render_max_paths: int = 20000,
    beamforming_azimuth_samples: int = 72,
    beamforming_elevation_samples: int = 37,
    beamforming_tx_scale: float = 1.5,
    beamforming_rx_scale: float = 1.5,
    beamforming_tx_node: str = "auto",
    beamforming_rx_node: str = "auto",
    standalone_beamforming_mode: str = "standalone",  # "frame" or "standalone"
    standalone_antenna_rows: int = 1,
    standalone_antenna_cols: int = 1,
    standalone_horizontal_spacing_m: float = 0.00535343675,
    standalone_vertical_spacing_m: float = 0.00535343675,
    standalone_carrier_frequency_ghz: float = 28.0,
    standalone_steering_strategy: str = "svd",
    standalone_azimuth_deg: float = 0.0,
    standalone_elevation_deg: float = 0.0,
    beamforming_db_scale: bool = False,
    beamforming_dynamic_range_db: float = 40.0,
    beamforming_colormap: str = "jet",
    beamforming_element_pattern: str = "isotropic",
    beamforming_tx_element_pattern: str = "isotropic",
    beamforming_rx_element_pattern: str = "isotropic",
    beamforming_pattern_status: str = "",
    # Range filters
    delay_filter_min_ns: float | None = None,
    delay_filter_max_ns: float | None = None,
    power_filter_min_db: float | None = None,
    power_filter_max_db: float | None = None,
    # Angle filters
    aoa_az_filter_min_deg: float | None = None,
    aoa_az_filter_max_deg: float | None = None,
    aoa_el_filter_min_deg: float | None = None,
    aoa_el_filter_max_deg: float | None = None,
    aod_az_filter_min_deg: float | None = None,
    aod_az_filter_max_deg: float | None = None,
    aod_el_filter_min_deg: float | None = None,
    aod_el_filter_max_deg: float | None = None,
    # Aperture visualization
    show_aoa_aperture: bool = False,
    show_aod_aperture: bool = False,
    aperture_radius_m: float = 5.0,
    show_global_angular_reference: bool = False,
    show_local_angular_reference: bool = False,
    # 3D trajectory visualization
    show_tx_trajectory: bool = False,
    show_rx_trajectory: bool = False,
    show_target_trajectory: bool = False,
    trajectory_color_mode: TrajectoryColorMode = "node_color",
    use_distinct_material_colors: bool = False,
    # Node labels
    node_label_mode: NodeLabelMode = "role",
    tx_labels: tuple[str, ...] = (),
    rx_labels: tuple[str, ...] = (),
    tx_device_names: tuple[str, ...] = (),
    rx_device_names: tuple[str, ...] = (),
    target_labels: tuple[str, ...] = (),
    show_target_labels: bool = True,
    label_screen_space: bool = True,
    # Cutaway planes
    clip_x_enabled: bool = False,
    clip_x_position: float = 0.0,
    clip_x_flip: bool = False,
    clip_y_enabled: bool = False,
    clip_y_position: float = 0.0,
    clip_y_flip: bool = False,
    clip_z_enabled: bool = False,
    clip_z_position: float = 0.0,
    clip_z_flip: bool = False,
    antialiasing_mode: str = "off",
    show_ground_grid: bool = False,
    hover_info_mode: HoverInfoMode = "essential",
    viewport_hud_enabled: bool | None = None,
    viewport_hud_mode: ViewportHudMode = "compact",
    viewport_hud_show_status: bool = True,
    viewport_hud_show_legends: bool = True,
    viewport_hud_show_filters: bool = True,
    viewport_hud_show_annotations: bool = True,
    show_mpc_type_markers: bool = False,
) -> AppState:
    """Return the default AppState configuration."""

    return AppState(
        step=step,
        selected_tx=selected_tx,
        selected_rx=selected_rx,
        mpc_visibility=MpcVisibility(
            enabled=mpc_layer_enabled,
            paths=show_mpc_paths,
            bounce_points=show_mpc_bounce_points,
        ),
        mpc_allowed_orders=(
            DEFAULT_MPC_ALLOWED_ORDERS
            if mpc_allowed_orders is None
            else frozenset(mpc_allowed_orders)
        ),
        mpc_allowed_types=(
            DEFAULT_MPC_ALLOWED_TYPES if mpc_allowed_types is None else frozenset(mpc_allowed_types)
        ),
        color_mode=color_mode,
        show_labels=show_labels,
        sync_target_position=sync_target_position,
        camera_mode=camera_mode,
        fly_mode=bool(fly_mode) if camera_mode == "overview" else False,
        pov_axis=pov_axis,
        pov_hidden_node=pov_hidden_node,
        show_camera_minimap=show_camera_minimap,
        show_beamforming=show_beamforming,
        show_coverage=show_coverage,
        show_rf_xray=show_rf_xray,
        rf_xray_mode=normalize_rf_xray_mode(rf_xray_mode),
        rf_xray_property=normalize_rf_xray_property(rf_xray_property),
        rf_xray_opacity=normalize_rf_xray_opacity(rf_xray_opacity),
        rf_xray_show_top_paths=rf_xray_show_top_paths,
        rf_xray_max_top_paths=max(1, int(rf_xray_max_top_paths)),
        coverage_height_index=coverage_height_index,
        topk_render_enabled=topk_render_enabled,
        topk_render_max_paths=topk_render_max_paths,
        beamforming_azimuth_samples=beamforming_azimuth_samples,
        beamforming_elevation_samples=beamforming_elevation_samples,
        beamforming_tx_scale=beamforming_tx_scale,
        beamforming_rx_scale=beamforming_rx_scale,
        beamforming_tx_node=beamforming_tx_node,
        beamforming_rx_node=beamforming_rx_node,
        standalone_beamforming_mode=standalone_beamforming_mode,
        standalone_antenna_rows=standalone_antenna_rows,
        standalone_antenna_cols=standalone_antenna_cols,
        standalone_horizontal_spacing_m=standalone_horizontal_spacing_m,
        standalone_vertical_spacing_m=standalone_vertical_spacing_m,
        standalone_carrier_frequency_ghz=standalone_carrier_frequency_ghz,
        standalone_steering_strategy=standalone_steering_strategy,
        standalone_azimuth_deg=standalone_azimuth_deg,
        standalone_elevation_deg=standalone_elevation_deg,
        beamforming_db_scale=beamforming_db_scale,
        beamforming_dynamic_range_db=beamforming_dynamic_range_db,
        beamforming_colormap=beamforming_colormap,
        beamforming_element_pattern=beamforming_element_pattern,
        beamforming_tx_element_pattern=beamforming_tx_element_pattern,
        beamforming_rx_element_pattern=beamforming_rx_element_pattern,
        beamforming_pattern_status=beamforming_pattern_status,
        # Range filters
        delay_filter_min_ns=delay_filter_min_ns,
        delay_filter_max_ns=delay_filter_max_ns,
        power_filter_min_db=power_filter_min_db,
        power_filter_max_db=power_filter_max_db,
        # Angle filters
        aoa_az_filter_min_deg=aoa_az_filter_min_deg,
        aoa_az_filter_max_deg=aoa_az_filter_max_deg,
        aoa_el_filter_min_deg=aoa_el_filter_min_deg,
        aoa_el_filter_max_deg=aoa_el_filter_max_deg,
        aod_az_filter_min_deg=aod_az_filter_min_deg,
        aod_az_filter_max_deg=aod_az_filter_max_deg,
        aod_el_filter_min_deg=aod_el_filter_min_deg,
        aod_el_filter_max_deg=aod_el_filter_max_deg,
        # Aperture visualization
        show_aoa_aperture=show_aoa_aperture,
        show_aod_aperture=show_aod_aperture,
        aperture_radius_m=aperture_radius_m,
        show_global_angular_reference=show_global_angular_reference,
        show_local_angular_reference=show_local_angular_reference,
        # 3D trajectory visualization
        show_tx_trajectory=show_tx_trajectory,
        show_rx_trajectory=show_rx_trajectory,
        show_target_trajectory=show_target_trajectory,
        trajectory_color_mode=trajectory_color_mode,
        use_distinct_material_colors=use_distinct_material_colors,
        # Node labels
        node_label_mode=node_label_mode,
        tx_labels=tx_labels,
        rx_labels=rx_labels,
        tx_device_names=tx_device_names,
        rx_device_names=rx_device_names,
        target_labels=target_labels,
        show_target_labels=show_target_labels,
        label_screen_space=label_screen_space,
        clip_x_enabled=clip_x_enabled,
        clip_x_position=clip_x_position,
        clip_x_flip=clip_x_flip,
        clip_y_enabled=clip_y_enabled,
        clip_y_position=clip_y_position,
        clip_y_flip=clip_y_flip,
        clip_z_enabled=clip_z_enabled,
        clip_z_position=clip_z_position,
        clip_z_flip=clip_z_flip,
        antialiasing_mode=antialiasing_mode,
        show_ground_grid=show_ground_grid,
        hover_info_mode=normalize_hover_info_mode(hover_info_mode),
        viewport_hud_enabled=(
            not _legacy_viewport_hud_mode_is_off(viewport_hud_mode)
            if viewport_hud_enabled is None
            else bool(viewport_hud_enabled)
        ),
        viewport_hud_mode=normalize_viewport_hud_mode(viewport_hud_mode),
        viewport_hud_show_status=viewport_hud_show_status,
        viewport_hud_show_legends=viewport_hud_show_legends,
        viewport_hud_show_filters=viewport_hud_show_filters,
        viewport_hud_show_annotations=viewport_hud_show_annotations,
        show_mpc_type_markers=show_mpc_type_markers,
    )


def get_beamforming_state_defaults(scenario_config: Any | None = None) -> dict[str, Any]:
    """Return the default runtime-only beamforming state."""
    initial = create_initial_state()
    defaults = {key: getattr(initial, key) for key in BEAMFORMING_STATE_KEYS}
    if scenario_config is not None:
        defaults.update(beamforming_defaults_from_scenario_config(scenario_config))
    return defaults


def strip_beamforming_state(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *data* without runtime-only beamforming fields."""
    stripped = dict(data)
    for key in BEAMFORMING_STATE_KEYS:
        stripped.pop(key, None)
    return stripped


def update_state(state: AppState, **changes) -> AppState:
    """Return a new AppState with the requested changes applied."""
    visibility = changes.get("mpc_visibility")
    if isinstance(visibility, dict):
        changes["mpc_visibility"] = MpcVisibility.from_dict(visibility)
    if "viewport_hud_mode" in changes:
        requested_hud_mode = changes["viewport_hud_mode"]
        if _legacy_viewport_hud_mode_is_off(requested_hud_mode):
            # Compatibility for callers that still use the retired combined
            # setting: hiding the HUD must not erase the user's detail choice.
            changes["viewport_hud_mode"] = normalize_viewport_hud_mode(state.viewport_hud_mode)
            changes.setdefault("viewport_hud_enabled", False)
        else:
            changes["viewport_hud_mode"] = normalize_viewport_hud_mode(requested_hud_mode)
    elif _legacy_viewport_hud_mode_is_off(state.viewport_hud_mode):
        # Normalize an AppState constructed directly with the retired value.
        changes["viewport_hud_mode"] = "compact"
        changes.setdefault("viewport_hud_enabled", False)
    if "viewport_hud_enabled" in changes:
        changes["viewport_hud_enabled"] = bool(changes["viewport_hud_enabled"])
    updated = replace(state, **changes)
    normalized_mode = normalize_rf_xray_mode(updated.rf_xray_mode)
    if normalized_mode != updated.rf_xray_mode:
        updated = replace(updated, rf_xray_mode=normalized_mode)
    normalized_property = normalize_rf_xray_property(updated.rf_xray_property)
    if normalized_property != updated.rf_xray_property:
        updated = replace(updated, rf_xray_property=normalized_property)
    normalized_opacity = normalize_rf_xray_opacity(updated.rf_xray_opacity)
    if normalized_opacity != updated.rf_xray_opacity:
        updated = replace(updated, rf_xray_opacity=normalized_opacity)
    if updated.beamforming_tx_node == "all" or updated.beamforming_rx_node == "all":
        updated = replace(
            updated,
            beamforming_tx_node=(
                "auto" if updated.beamforming_tx_node == "all" else updated.beamforming_tx_node
            ),
            beamforming_rx_node=(
                "auto" if updated.beamforming_rx_node == "all" else updated.beamforming_rx_node
            ),
        )
    if updated.camera_mode != "overview" and updated.fly_mode:
        updated = replace(updated, fly_mode=False)
    normalized_hover_mode = normalize_hover_info_mode(updated.hover_info_mode)
    if normalized_hover_mode != updated.hover_info_mode:
        updated = replace(updated, hover_info_mode=normalized_hover_mode)
    normalized_hud_mode = normalize_viewport_hud_mode(updated.viewport_hud_mode)
    if normalized_hud_mode != updated.viewport_hud_mode:
        updated = replace(updated, viewport_hud_mode=normalized_hud_mode)
    return updated
