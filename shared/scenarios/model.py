"""Pydantic schema models for ORCHAV scenario YAML files.

This package-level schema is the authority for validating scenario documents
before generator, visualizer, or shared tooling adapts them into runtime
objects. The models describe user-facing YAML sections, keep typo detection
strict, and leave generator-specific dataclass conversion to generator core.
"""

from pathlib import PureWindowsPath
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shared.grpc_transport import format_grpc_endpoint, parse_grpc_endpoint
from shared.scenarios.actors import (
    ActorsSpec,
    AlignMotionOrientationSpec,
    GroupMemberMobilitySpec,
    GroupSpec,
    KeyframesOrientationSpec,
    LookAtOrientationSpec,
    MeshSequenceMobilitySpec,
    RandomOrientationSpec,
    RandomSamplingMobilitySpec,
    SpinOrientationSpec,
    StationaryMobilitySpec,
    TimelineSpec,
)
from shared.scenarios.defaults import DEFAULT_COVERAGE_TX_MODE
from shared.scenarios.extensions import registered_scene_source_keys
from shared.scenarios.frame_paths import (
    DEFAULT_FRAMES_DIRECTORY,
    DEFAULT_FRAMES_PATTERN,
    validate_frames_directory,
    validate_frames_pattern,
)

# ---------------------------------------------------------------------------
# Quality preset names (from generator/core/configuration/presets.py QUALITY_PRESETS)
# ---------------------------------------------------------------------------
QualityPreset = Literal["ultra-low", "low", "medium", "high", "ultra", "custom"]


class BaseStrictModel(BaseModel):
    """Base model that forbids extra fields to catch typos."""

    model_config = ConfigDict(extra="forbid")


class SceneModel(BaseStrictModel):
    """Scene selection for library, local XML, Sionna built-in, or registered sources."""

    id: str = Field(
        default="default",
        description=(
            "Scene identifier, library path, local XML path, or [Sionna RT]"
            "(https://nvlabs.github.io/sionna/) scene name. Local paths may use "
            "`${PROJECT_ROOT}`."
        ),
    )
    source: str = Field(
        default="library",
        description="Scene source resolver used to interpret `scene.id`.",
    )

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        source = str(value or "").strip()
        allowed = {"library", "local", "sionna"} | registered_scene_source_keys()
        if source not in allowed:
            allowed_text = ", ".join(sorted(allowed))
            raise ValueError(f"scene source must be one of: {allowed_text}")
        return source


# ---------------------------------------------------------------------------
# Ray Tracing
# ---------------------------------------------------------------------------
class AntennaArrayModel(BaseStrictModel):
    """Configuration for a single antenna array (TX or RX)."""

    pattern: str = Field(
        default="iso",
        description="Sionna RT antenna pattern name, such as `iso`, `dipole`, or `tr38901`.",
    )
    polarization: Literal["V", "H"] = Field(
        default="V",
        description="Element polarization: vertical `V` or horizontal `H`.",
    )
    num_rows: int = Field(
        default=1,
        ge=1,
        description="Number of rows in the rectangular antenna array.",
    )
    num_cols: int = Field(
        default=1,
        ge=1,
        description="Number of columns in the rectangular antenna array.",
    )
    vertical_spacing: float = Field(
        default=0.5,
        gt=0,
        description="Vertical element spacing in wavelengths.",
    )
    horizontal_spacing: float = Field(
        default=0.5,
        gt=0,
        description="Horizontal element spacing in wavelengths.",
    )

    @model_validator(mode="after")
    def validate_pattern(self):
        pattern = (self.pattern or "").strip()
        if not pattern:
            raise ValueError("pattern must be a non-empty string")
        self.pattern = pattern
        return self


class AntennaModel(BaseStrictModel):
    """Antenna configuration with independent TX and RX arrays."""

    tx: Optional[AntennaArrayModel] = Field(
        default=None,
        description="Transmitter antenna array settings.",
    )
    rx: Optional[AntennaArrayModel] = Field(
        default=None,
        description="Receiver antenna array settings.",
    )


class RayTracingQualityModel(BaseStrictModel):
    """Quality preset or custom ray-tracing parameters."""

    preset: Optional[QualityPreset] = Field(
        default=None,
        description="Named ray-tracing quality preset.",
    )
    custom: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-key Sionna RT solver overrides merged on top of the preset.",
    )


class PathFilterModel(BaseStrictModel):
    """Configuration for filtering paths before writing to files."""

    relative_threshold_db: Optional[float] = Field(
        default=None,
        description="Keep paths within this many dB of the strongest path for each TX/RX pair.",
    )
    max_path_loss_db: Optional[float] = Field(
        default=None,
        description="Drop paths whose path loss is above this absolute dB threshold.",
    )
    max_paths_per_pair: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of paths retained per TX/RX pair after filtering.",
    )
    log_filtering_stats: bool = Field(
        default=True,
        description="Log path-filtering counts and thresholds during generation.",
    )
    generate_diagnostic: bool = Field(
        default=False,
        description="Generate diagnostic plots for filtered paths.",
    )


class MaterialOverrideModel(BaseStrictModel):
    """Per-material scattering and thickness overrides."""

    relative_permittivity: Optional[float] = Field(
        default=None,
        gt=0,
        description="Relative permittivity assigned to the matched material.",
    )
    conductivity: Optional[float] = Field(
        default=None,
        ge=0,
        description="Electrical conductivity in siemens per meter.",
    )
    scattering_coefficient: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Fraction of reflected energy scattered diffusely.",
    )
    scattering_pattern: Optional[
        Literal[
            "lambertian",
            "directive",
            "backscattering",
            "g-rer",
            "iso",
            "isotropic",
            "er-isotropic",
        ]
    ] = Field(
        default=None,
        description="Diffuse scattering pattern used when scattering is enabled.",
    )
    alpha_r: Optional[float] = Field(
        default=None,
        gt=0,
        description="Roughness parameter for directive or backscattering patterns.",
    )
    alpha_g: Optional[float] = Field(
        default=None,
        ge=0,
        description="Gaussian RER parameter for the `g-rer` scattering pattern.",
    )
    xpd_coefficient: Optional[float] = Field(
        default=None,
        description="Cross-polarization discrimination coefficient.",
    )
    thickness: Optional[float] = Field(
        default=None,
        gt=0,
        description="Material thickness in meters, used by refraction calculations.",
    )

    @model_validator(mode="after")
    def validate_scattering_fields(self):
        """Validate scattering override consistency.

        Coefficient-only overrides (no pattern, no alpha) are valid — the
        coefficient is applied to the material's existing scattering pattern.
        When alpha_r/alpha_g or an explicit scattering_pattern is provided,
        both pattern and coefficient are required.
        """
        has_pattern = self.scattering_pattern is not None
        has_coeff = self.scattering_coefficient is not None
        has_alpha = self.alpha_r is not None or self.alpha_g is not None

        if not has_pattern and not has_coeff and not has_alpha:
            return self

        # Coefficient-only: default to lambertian (matches Sionna default)
        if has_coeff and not has_pattern and not has_alpha:
            self.scattering_pattern = "lambertian"
            return self

        # When pattern or alpha params are involved, require full block
        if (has_alpha or has_pattern) and not has_pattern:
            raise ValueError("scattering_pattern is required when alpha_r or alpha_g is defined")
        if has_pattern and not has_coeff:
            raise ValueError(
                "scattering_coefficient is required when scattering_pattern is defined"
            )

        pattern = self.scattering_pattern
        if self.alpha_r is not None and self.alpha_g is not None:
            raise ValueError("alpha_r and alpha_g cannot both be set for one material override")

        if pattern == "g-rer":
            if self.alpha_g is None:
                raise ValueError("alpha_g is required when scattering_pattern is 'g-rer'")
            if self.alpha_r is not None:
                raise ValueError("alpha_r is not allowed when scattering_pattern is 'g-rer'")
            return self

        if pattern in {"directive", "backscattering"}:
            if self.alpha_r is None:
                raise ValueError(f"alpha_r is required when scattering_pattern is '{pattern}'")
            if self.alpha_g is not None:
                raise ValueError(f"alpha_g is not allowed when scattering_pattern is '{pattern}'")
            return self

        if self.alpha_r is not None:
            raise ValueError(f"alpha_r is not allowed when scattering_pattern is '{pattern}'")
        if self.alpha_g is not None:
            raise ValueError(f"alpha_g is not allowed when scattering_pattern is '{pattern}'")

        return self


class SceneMaterialsModel(BaseStrictModel):
    """Policy for ORCHAV-managed defaults on Sionna-loaded scene materials."""

    scattering_coefficient_preset: Literal["none", "itu"] = Field(
        default="none",
        description=(
            "Scene material scattering preset. `none` leaves Sionna-loaded values "
            "unchanged. `itu` assigns ORCHAV's known coefficients by material family."
        ),
    )


class RayTracingModel(BaseStrictModel):
    """Ray-tracing settings independent of the scenario timeline and actors."""

    enabled: bool = Field(default=False, description="Enable Generator-side ray tracing.")
    view: Optional[str] = Field(
        default=None,
        description="Summary/camera view hint such as `top`, `side`, `front`, or `isometric`.",
    )
    export_path_metrics: bool = Field(
        default=False,
        description="Store per-path delay, path loss, AoA, and AoD metrics in generated frames.",
    )
    quality: Optional[RayTracingQualityModel] = Field(
        default=None,
        description="Quality preset and optional custom ray-tracing settings.",
    )
    carrier_frequency_hz: Optional[float] = Field(
        default=None,
        gt=0,
        description="RF carrier frequency in hertz.",
    )
    bandwidth_hz: Optional[float] = Field(
        default=None,
        gt=0,
        description="Channel bandwidth in hertz.",
    )
    temperature_k: Optional[float] = Field(
        default=None,
        gt=0,
        description="Scene noise temperature in kelvin.",
    )
    path_filter: Optional[PathFilterModel] = Field(
        default=None,
        description="Generator-side path filtering thresholds applied before writing frames.",
    )
    antenna: Optional[AntennaModel] = Field(
        default=None,
        description="TX/RX antenna array configuration used during ray tracing.",
    )
    start_step: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "First absolute scenario step written to a fresh frame set. A value "
            "above zero writes only `[start_step, timeline.steps)`. It does not "
            "resume or merge an earlier frame set."
        ),
    )
    materials: Optional[Dict[str, MaterialOverrideModel]] = Field(
        default=None,
        description="Per-material electromagnetic property overrides keyed by material name.",
    )
    scene_materials: Optional[SceneMaterialsModel] = Field(
        default=None,
        description="Policy for default properties assigned to loaded scene materials.",
    )
    mesh_update_interval_s: Optional[float] = Field(
        default=None,
        gt=0,
        description="Animated mesh update interval in seconds.",
    )
    cir_time_steps: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of Sionna-native CIR time samples per ray-tracing call.",
    )
    cir_sampling_frequency_hz: Optional[float] = Field(
        default=None,
        gt=0,
        description="CIR sampling frequency in hertz when multiple CIR time samples are used.",
    )


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
def _validate_figure_basename(value: str) -> str:
    """Keep configurable figure names inside their fixed output directory."""
    if (
        not value
        or value != value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or bool(PureWindowsPath(value).drive)
    ):
        raise ValueError("figure filename must be a non-empty basename")
    return value


class CoverageSaveDataModel(BaseStrictModel):
    """Whether to persist the canonical scenario coverage map."""

    enabled: bool = Field(default=True, description="Write coverage-map arrays to HDF5.")


class CoverageDistributionFigureModel(BaseStrictModel):
    """Histogram/CDF output for selected metrics at every coverage height."""

    enabled: bool = Field(
        default=False,
        description=(
            "Save one histogram/CDF figure per configured height for the selected "
            "coverage metrics."
        ),
    )
    metrics: List[str] = Field(
        default_factory=list,
        description="Coverage metric names included in the distribution figure.",
    )
    filename: str = Field(
        default="coverage_distributions",
        description="Base filename for distribution figures.",
    )
    bins: int = Field(
        default=40,
        ge=5,
        description="Histogram bin count for scalar distribution panels.",
    )

    _filename_is_basename = field_validator("filename")(_validate_figure_basename)


class CoverageSaveFigureModel(BaseStrictModel):
    """How to render every coverage height under fixed `summary/coverage`."""

    enabled: bool = Field(
        default=False,
        description="Save one rendered coverage-map figure per configured height.",
    )
    format: Literal["png", "svg", "pdf"] = Field(
        default="png",
        description="Coverage figure file format.",
    )
    filename: str = Field(
        default="coverage_maps",
        description="Base filename for saved coverage-map figures.",
    )
    overlay_scene: bool = Field(
        default=False,
        description="Overlay scene geometry on coverage-map figures.",
    )
    overlay_alpha: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Scene overlay opacity.",
    )
    overlay_style: str = Field(
        default="outline",
        description="Scene overlay drawing style.",
    )
    interpolation: str = Field(
        default="nearest",
        description="Image interpolation mode for rendered coverage arrays.",
    )
    metrics: List[str] = Field(
        default_factory=list,
        description=(
            "Metric names or metric specs included in one multi-panel guide per "
            "configured height."
        ),
    )
    metric_filename: str = Field(
        default="coverage_metrics",
        description="Base filename for metric-guide figures.",
    )
    columns: int = Field(
        default=3,
        ge=1,
        description="Number of columns in multi-panel coverage figures.",
    )
    show_tx: bool = Field(
        default=True,
        description="Draw transmitter markers on coverage figures.",
    )
    distribution: Optional[CoverageDistributionFigureModel] = Field(
        default=None,
        description="Optional histogram/CDF distribution figure settings.",
    )

    _filenames_are_basenames = field_validator("filename", "metric_filename")(
        _validate_figure_basename
    )

    @model_validator(mode="after")
    def _enabled_output_filenames_are_distinct(self) -> "CoverageSaveFigureModel":
        """Prevent enabled figure families from overwriting the same files."""
        if not self.enabled:
            return self

        enabled_families = [("coverage map", self.filename)]
        if self.metrics:
            enabled_families.append(("metric guide", self.metric_filename))
        if (
            self.distribution is not None
            and self.distribution.enabled
            and self.distribution.metrics
        ):
            enabled_families.append(("distribution", self.distribution.filename))

        filenames_by_casefold: Dict[str, tuple[str, str]] = {}
        for family, filename in enabled_families:
            key = filename.casefold()
            existing = filenames_by_casefold.get(key)
            if existing is not None:
                existing_family, existing_filename = existing
                raise ValueError(
                    "enabled coverage figure families must use distinct base "
                    "filenames (case-insensitive for Windows portability): "
                    f"{existing_family} {existing_filename!r} conflicts with "
                    f"{family} {filename!r}"
                )
            filenames_by_casefold[key] = (family, filename)

        return self


class CoverageSaveModel(BaseStrictModel):
    """Coverage output settings for data files and figures."""

    compression: Optional[Literal["lzf", "gzip", "none"]] = Field(
        default="lzf",
        description="Compression used for coverage HDF5 datasets.",
    )
    data: Optional[CoverageSaveDataModel] = Field(
        default=None,
        description="Coverage-map persistence settings for the fixed HDF5 output.",
    )
    figure: Optional[CoverageSaveFigureModel] = Field(
        default=None,
        description="Coverage figure settings for the fixed summary location.",
    )

    @model_validator(mode="after")
    def _figures_require_persisted_data(self) -> "CoverageSaveModel":
        data_enabled = self.data is None or self.data.enabled
        if self.figure is not None and self.figure.enabled and not data_enabled:
            raise ValueError("coverage figures require save.data.enabled: true")
        return self


class CoverageQualityModel(BaseStrictModel):
    """Quality preset or custom settings for coverage computation."""

    preset: Optional[QualityPreset] = Field(
        default=None,
        description="Named quality preset for coverage-map computation.",
    )
    custom: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-key coverage solver overrides merged on top of the preset.",
    )


class CoverageGridModel(BaseStrictModel):
    """Regular planar coverage grid definition."""

    bbox_xy: Union[List[List[float]], Literal["auto"], None] = Field(
        default="auto",
        description=(
            "Coverage XY bounds as `[[x_min, x_max], [y_min, y_max]]` or `auto`. "
            "Auto requires scene geometry bounds."
        ),
    )
    resolution_m: List[float] = Field(
        default_factory=lambda: [1.0, 1.0],
        description="Coverage grid spacing `[dx, dy]` in meters.",
    )
    heights_m: Union[List[float], float, Literal["auto"], None] = Field(
        default=None,
        description="Coverage sampling height or heights in meters.",
    )

    @model_validator(mode="after")
    def validate_grid(self):
        if self.bbox_xy not in (None, "auto"):
            if not isinstance(self.bbox_xy, list) or len(self.bbox_xy) != 2:
                raise ValueError("coverage.grid.bbox_xy must be [[x_min, x_max], [y_min, y_max]]")
            for pair in self.bbox_xy:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError(
                        "coverage.grid.bbox_xy must be [[x_min, x_max], [y_min, y_max]]"
                    )
        if len(self.resolution_m) != 2:
            raise ValueError("coverage.grid.resolution_m must be [dx, dy]")
        if any(float(v) <= 0.0 for v in self.resolution_m):
            raise ValueError("coverage.grid.resolution_m values must be positive")
        return self


class CoverageSolverModel(BaseStrictModel):
    """Sionna RT radio-map solver settings."""

    preset: Optional[QualityPreset] = Field(
        default=None,
        description="Named radio-map quality preset.",
    )
    custom: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-key radio-map solver overrides merged on top of the preset.",
    )
    samples_per_tx: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of coverage samples launched per transmitter.",
    )
    max_depth: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum ray interaction depth for coverage computation.",
    )
    los: Optional[bool] = Field(default=None, description="Enable line-of-sight paths.")
    specular_reflection: Optional[bool] = Field(
        default=None,
        description="Enable specular reflection paths.",
    )
    diffuse_reflection: Optional[bool] = Field(
        default=None,
        description="Enable diffuse reflection paths.",
    )
    refraction: Optional[bool] = Field(default=None, description="Enable refracted paths.")
    diffraction: Optional[bool] = Field(default=None, description="Enable diffracted paths.")
    seed: Optional[int] = Field(default=None, description="Random seed for coverage sampling.")
    rr_depth: Optional[int] = Field(
        default=None,
        ge=0,
        description="Russian-roulette start depth for path termination.",
    )
    rr_prob: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Russian-roulette continuation probability.",
    )
    stop_threshold: Optional[float] = Field(
        default=None,
        description="Path power threshold for early termination.",
    )


class CoverageMetricsModel(BaseStrictModel):
    """Coverage metric selection for the HDF5 logical metric menu.

    ``store`` and ``derived`` describe metrics that should be available to
    readers. The compact coverage writer may still materialize only canonical
    path gain and recipe metadata when those metrics are derivable.
    """

    store: List[Literal["path_gain_linear", "rss_w", "sinr_linear"]] = Field(
        default_factory=lambda: ["path_gain_linear"],
        description="Physical metric families made available from canonical coverage data.",
    )
    derived: List[
        Literal[
            "path_gain_db",
            "path_loss_db",
            "best_path_loss_db",
            "rss_dbm",
            "best_rss_dbm",
            "sum_rss_dbm",
            "sinr_db",
            "serving_tx",
            "tx_margin_db",
        ]
    ] = Field(
        default_factory=lambda: [
            "path_loss_db",
            "rss_dbm",
            "sinr_db",
            "serving_tx",
            "tx_margin_db",
        ],
        description="Derived coverage metrics made available through compact layers or recipes.",
    )


class CoverageTxModel(BaseStrictModel):
    """How transmitter layers are exposed for coverage analysis."""

    mode: Literal["per_tx", "best_server", "sum_power", "selected", "margin"] = Field(
        default=DEFAULT_COVERAGE_TX_MODE,
        description="How transmitter layers are combined or selected for coverage output.",
    )
    selected: Optional[Union[int, str]] = Field(
        default=None,
        description=("Zero-based transmitter index or actor name used when `mode` is `selected`."),
    )
    precoding_vec: Optional[List[float]] = Field(
        default=None,
        description="Optional precoding vector passed to the coverage solver.",
    )


class CoverageModel(BaseStrictModel):
    """Coverage map computation settings."""

    enabled: bool = Field(default=False, description="Enable coverage-map computation.")

    # Structured coverage sections.
    grid: Optional[CoverageGridModel] = Field(
        default=None,
        description="Structured coverage grid definition.",
    )
    solver: Optional[CoverageSolverModel] = Field(
        default=None,
        description="Structured coverage solver settings.",
    )
    metrics: Optional[CoverageMetricsModel] = Field(
        default=None,
        description="Coverage metric storage and derived-layer settings.",
    )
    tx: Optional[CoverageTxModel] = Field(
        default=None,
        description="Transmitter layer selection and combination settings.",
    )

    stride: int = Field(
        default=1,
        ge=1,
        description="Compute coverage every N-th simulation step.",
    )
    save: Optional[CoverageSaveModel] = Field(
        default=None,
        description="Coverage data and figure output settings.",
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class FilesConfigModel(BaseStrictModel):
    """HDF5 frame-directory, read-filter, compression, and chunk settings."""

    format: Literal["h5", "hdf5"] = Field(
        default="h5",
        description="HDF5 file format label.",
    )
    directory: str = Field(
        default=DEFAULT_FRAMES_DIRECTORY,
        description=(
            "Directory containing the complete HDF5 frame set. Relative paths "
            "are resolved from the scenario directory. Read-only profiles may "
            "use `${PROJECT_ROOT}`."
        ),
    )
    pattern: str = Field(
        default=DEFAULT_FRAMES_PATTERN,
        description=(
            "Optional basename-only frame-file filter for read-only tools. "
            "Generated chunk names are owned by the frame manifest."
        ),
    )
    chunk_size: int = Field(
        default=100,
        ge=1,
        description="Frames per generated HDF5 chunk.",
    )
    compression: str = Field(
        default="lzf",
        description=(
            "HDF5 compression profile used for generated frame chunks "
            "(lzf/balanced, gzip-4/compact, or none/fast)."
        ),
    )

    @field_validator("directory")
    @classmethod
    def validate_directory(cls, value: str) -> str:
        return validate_frames_directory(value)

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, value: str) -> str:
        return validate_frames_pattern(value)


class RemoteHdf5ConfigModel(BaseStrictModel):
    """Remote HDF5 server connection settings for gRPC-based file access."""

    server: Optional[str] = Field(
        default=None,
        description="Remote frame-file server address as `host:port`.",
    )
    cache_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of remote frames cached locally by the Visualizer.",
    )
    connect_timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description="Connection timeout in seconds.",
    )
    frame_index_ttl_s: Optional[float] = Field(
        default=None,
        ge=0,
        description="Frame-index refresh interval in seconds. `0` keeps the startup snapshot.",
    )

    @field_validator("server")
    @classmethod
    def validate_server(cls, value: Optional[str]) -> Optional[str]:
        """Validate and normalize the documented bare remote server address."""
        if value is None:
            return None
        normalized = value.strip()
        if "://" in normalized:
            raise ValueError("remote_hdf5.server must use the bare host:port form")
        host, port = parse_grpc_endpoint(normalized)
        return str(format_grpc_endpoint(host, port))


class LiveGrpcConfigModel(BaseStrictModel):
    """Live Generator frame-stream connection settings."""

    endpoint: Optional[str] = Field(
        default=None,
        description=(
            "Direct Live Generator endpoint used by the Visualizer and advertised "
            "when this scenario starts live generation. This value overrides endpoint "
            "aliases for the `sionna` service."
        ),
        json_schema_extra={
            "values": "`host:port` or `grpc://host:port`; port 1-65535",
        },
    )
    buffer_size: int = Field(
        default=50,
        ge=1,
        description="Number of streamed frames buffered by the Visualizer.",
    )

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        parse_grpc_endpoint(value)
        return value.strip()


class DataConfigModel(BaseStrictModel):
    """Data mode selection (files, live_grpc, remote_hdf5) and per-mode settings."""

    mode: Literal["files", "live_grpc", "remote_hdf5"] = Field(
        default="files",
        description="Frame delivery route selected for this scenario.",
    )
    files: FilesConfigModel = Field(
        default_factory=FilesConfigModel,
        description="Local HDF5 frame loading and output settings.",
    )
    live_grpc: Optional[LiveGrpcConfigModel] = Field(
        default=None,
        description="Live Generator streaming settings.",
    )
    remote_hdf5: Optional[RemoteHdf5ConfigModel] = Field(
        default=None,
        description="Remote HDF5 frame-file server settings.",
    )
    live_grpc_endpoints: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Endpoint aliases keyed by service name. Matching keys override the root "
            "`live_grpc_endpoints` mapping."
        ),
    )

    @field_validator("live_grpc_endpoints")
    @classmethod
    def validate_live_grpc_endpoints(
        cls,
        value: Optional[Dict[str, str]],
    ) -> Optional[Dict[str, str]]:
        if value and value.get("sionna"):
            parse_grpc_endpoint(value["sionna"])
        return value


# ---------------------------------------------------------------------------
# View Defaults
# ---------------------------------------------------------------------------
class MpcVisibilityDefaultsModel(BaseStrictModel):
    """Initial visibility for the MPC layer and its independently selectable parts."""

    enabled: Optional[bool] = Field(
        default=None,
        description="Enable the MPC layer. When false, paths and bounce points are both hidden.",
    )
    paths: Optional[bool] = Field(
        default=None,
        description="Show MPC path segments when the MPC layer is enabled.",
    )
    bounce_points: Optional[bool] = Field(
        default=None,
        description="Show physical MPC interaction points when the MPC layer is enabled.",
    )


class ViewDefaultsModel(BaseStrictModel):
    """Default Visualizer view settings loaded from scenario YAML."""

    color_mode: Optional[
        Literal["reflection_order", "mpc_type", "delay", "path_loss", "reconstruction_type"]
    ] = Field(
        default=None,
        description="Initial MPC color mode. `reconstruction_type` requires compatible extension data.",
    )
    selected_tx: Union[str, int, List[str], List[int], Literal["all"], None] = Field(
        default=None,
        description="Initial transmitter filter: `all`, one index/name, or a list of indices/names.",
    )
    selected_rx: Union[str, int, List[str], List[int], Literal["all"], None] = Field(
        default=None,
        description="Initial receiver filter: `all`, one index/name, or a list of indices/names.",
    )
    mpc_visibility: Optional[MpcVisibilityDefaultsModel] = Field(
        default=None,
        description="Initial MPC layer, path, and physical bounce-point visibility.",
    )
    resolution: Optional[List[int]] = Field(
        default=None,
        description="Initial viewport size hint as `[width, height]`.",
    )
    camera_dist: Optional[float] = Field(
        default=None,
        description="Initial camera distance from the scene focus point in meters.",
    )
    fov: Optional[float] = Field(
        default=None,
        description="Initial camera field of view in degrees.",
    )
    camera_view: Optional[Literal["top", "side", "front", "isometric", "iso"]] = Field(
        default=None,
        description="Initial named camera orientation.",
    )
    auto_generate_uvs: Optional[bool] = Field(
        default=None,
        description="Generate box-projection UVs for meshes without embedded UV coordinates.",
    )
    merge_scene_meshes: Optional[bool] = Field(
        default=None,
        description=(
            "Merge scene meshes by material for renderer performance. Omission uses "
            "automatic behavior."
        ),
    )
    visual_profiles: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional renderer profiles consumed by Visualizer workflows.",
    )


# ---------------------------------------------------------------------------
# Generator Summary
# ---------------------------------------------------------------------------
GeneratorSummaryProduct = Literal[
    "scene2d",
    "scene3d",
    "speed",
    "orientation",
    "angular_velocity",
]


class GeneratorSummaryOutputModel(BaseStrictModel):
    """Image format for figures written below fixed `summary/`."""

    format: Literal["png", "svg", "pdf"] = Field(
        default="png",
        description="Image format for Generator summary figures.",
    )


class VisualizationConfigModel(BaseStrictModel):
    """Visualization rendering configuration for 2D and 3D scene summaries."""

    scene2d_mode: str = Field(
        default="rasterized",
        description="2D scene summary rendering mode.",
    )
    scene2d_resolution: float = Field(
        default=0.05,
        gt=0,
        description="Rasterized 2D scene resolution in meters.",
    )
    scene2d_material_legend: bool = Field(
        default=False,
        description="Include detected scene material families in the 2D summary legend.",
    )
    actor_label_mode: Literal["role", "name"] = Field(
        default="role",
        description="Actor label mode: role labels such as TX1/RX1 or scenario names.",
    )
    scene3d_mode: str = Field(
        default="floor_plan",
        description="3D scene summary rendering mode.",
    )
    scene3d_alpha: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Scene opacity in 3D summary figures.",
    )
    scene3d_z_exaggeration: Optional[Union[float, Literal["auto"]]] = Field(
        default=None,
        description="Visual-only Z-axis exaggeration for 3D summaries.",
    )
    scene3d_bounds: Literal[
        "union", "scene", "geometry", "actors", "actor", "trajectory", "trajectories"
    ] = Field(
        default="union",
        description="Bounds source used to frame the 3D summary figure.",
    )
    scene3d_limits: Optional[Union[List[float], Dict[str, Any]]] = Field(
        default=None,
        description="Explicit 3D summary limits.",
    )
    scene3d_camera: Optional[Dict[str, float]] = Field(
        default=None,
        description="Camera settings for 3D summary figures.",
    )


class GeneratorSummaryModel(BaseStrictModel):
    """Generator post-run topology, motion, and extension diagnostics."""

    enabled: bool = Field(default=False, description="Generate post-run summary figures.")
    force: bool = Field(
        default=False,
        description=(
            "Regenerate summary figures even when the normalized scenario YAML is unchanged."
        ),
    )
    create: List[GeneratorSummaryProduct] = Field(
        default_factory=list,
        description="Summary figure groups to create. Unlisted groups are not generated.",
    )
    output: Optional[GeneratorSummaryOutputModel] = Field(
        default=None,
        description="Image format for the fixed scenario summary output.",
    )
    visualization: Optional[VisualizationConfigModel] = Field(
        default=None,
        description="2D and 3D rendering options for summary figures.",
    )


# ---------------------------------------------------------------------------
# Live Generator Endpoints
# ---------------------------------------------------------------------------
class LiveGrpcEndpointsModel(BaseStrictModel):
    """Endpoint addresses used by a Live Generator session."""

    model_config = ConfigDict(extra="allow")

    sionna: Optional[str] = Field(
        default=None,
        description="Live Generator gRPC endpoint.",
    )
    http: Optional[str] = Field(
        default=None,
        description="Optional HTTP endpoint associated with live streaming.",
    )

    @field_validator("sionna")
    @classmethod
    def validate_sionna_endpoint(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        parse_grpc_endpoint(value)
        return value.strip()


# ---------------------------------------------------------------------------
# Root Scenario
# ---------------------------------------------------------------------------
class ScenarioModel(BaseStrictModel):
    """Immutable scenario configuration.

    Actor and group reference validation happens here because it requires the
    complete document.  Empty ``actors`` is valid for scenarios whose Python
    entry point supplies actors programmatically.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    @model_validator(mode="before")
    @classmethod
    def reject_unavailable_sensing_authoring(cls, value: Any) -> Any:
        """Reject scenario-authored sensing while preserving frame interoperability."""

        if isinstance(value, dict) and "sensing" in value:
            raise ValueError(
                "The `sensing` section is not supported in ORCHAV 0.1 scenario authoring. "
                "This release can still read optional sensing extension data stored in "
                "StandardMPCFrame HDF5 files."
            )
        return value

    schema_version: Literal[2] = Field(
        description=("Required format identifier for `scenario.yaml`. ORCHAV 0.1 accepts only `2`.")
    )
    timeline: TimelineSpec = Field(description="Endpoint-inclusive scenario timeline.")
    actors: ActorsSpec = Field(
        default_factory=ActorsSpec,
        description="Optional YAML-authored transmitter, receiver, and target actors.",
    )
    groups: tuple[GroupSpec, ...] = Field(
        default=(),
        description="Optional shared mobility groups referenced by actors.",
    )
    scene: Optional[SceneModel] = Field(default=None, description="Scene selection block.")
    data: DataConfigModel = Field(
        default_factory=DataConfigModel,
        description="Frame data mode block.",
    )
    raytracing: Optional[RayTracingModel] = Field(
        default=None,
        description="Generator ray-tracing settings.",
    )
    coverage: Optional[CoverageModel] = Field(
        default=None,
        description="Optional coverage-map computation settings.",
    )
    view_defaults: Optional[ViewDefaultsModel] = Field(
        default=None,
        description="Visualizer view defaults.",
    )
    visualizer: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional component-owned Visualizer settings. The shared schema accepts this "
            "mapping without defining its subkeys. Use `view_defaults` for portable initial "
            "camera, coloring, and visibility preferences."
        ),
    )
    generator_summary: Optional[GeneratorSummaryModel] = Field(
        default=None,
        description="Post-run Generator summary figure settings.",
    )
    debug_level: Optional[str] = Field(
        default=None,
        description=(
            "Scenario logging threshold. Recognized names are DEBUG, INFO, WARNING, ERROR, "
            "and CRITICAL. Omission or an unrecognized value uses WARNING unless "
            "`ORCHAV_LOG_LEVEL` overrides it."
        ),
    )

    live_grpc_endpoints: Optional[LiveGrpcEndpointsModel] = Field(
        default=None,
        description=(
            "Root Live Generator endpoint aliases. Matching aliases under `data` override "
            "these values."
        ),
    )

    @model_validator(mode="after")
    def validate_actor_graph(self) -> "ScenarioModel":
        actors = self.actors.all()
        actor_names = [actor.name for actor in actors]
        duplicate_actor_names = sorted(
            name for name in set(actor_names) if actor_names.count(name) > 1
        )
        if duplicate_actor_names:
            raise ValueError(
                "actor names must be globally unique; duplicates: "
                + ", ".join(duplicate_actor_names)
            )

        group_names = [group.name for group in self.groups]
        duplicate_group_names = sorted(
            name for name in set(group_names) if group_names.count(name) > 1
        )
        if duplicate_group_names:
            raise ValueError(
                "group names must be unique; duplicates: " + ", ".join(duplicate_group_names)
            )

        groups_by_name = {group.name: group for group in self.groups}
        member_counts = {name: 0 for name in groups_by_name}
        member_roles: dict[str, set[str]] = {name: set() for name in groups_by_name}
        for actor in actors:
            mobility = actor.mobility
            if isinstance(mobility, GroupMemberMobilitySpec):
                if mobility.group not in groups_by_name:
                    raise ValueError(
                        f"actor '{actor.name}' references missing group '{mobility.group}'"
                    )
                member_counts[mobility.group] += 1
                member_roles[mobility.group].add(actor.role.value)

            if isinstance(mobility, MeshSequenceMobilitySpec) and actor.role.value != "target":
                raise ValueError("mesh_sequence mobility is only valid for target actors")

            orientation = actor.orientation
            effective_mobility = (
                groups_by_name[mobility.group].mobility
                if isinstance(mobility, GroupMemberMobilitySpec)
                else mobility
            )
            if isinstance(orientation, AlignMotionOrientationSpec) and isinstance(
                effective_mobility,
                (StationaryMobilitySpec, RandomSamplingMobilitySpec),
            ):
                raise ValueError(
                    f"actor '{actor.name}' align_motion requires mobility with physical velocity"
                )
            if isinstance(orientation, LookAtOrientationSpec) and orientation.actor is not None:
                if orientation.actor not in actor_names:
                    raise ValueError(
                        f"actor '{actor.name}' look_at references missing actor "
                        f"'{orientation.actor}'"
                    )
                if orientation.actor == actor.name:
                    raise ValueError(f"actor '{actor.name}' cannot look_at itself")

            if isinstance(orientation, KeyframesOrientationSpec):
                if orientation.keyframes[-1].time_s > self.timeline.duration_s:
                    raise ValueError(
                        f"actor '{actor.name}' orientation keyframe exceeds timeline duration_s"
                    )

        invalid_groups = sorted(name for name, count in member_counts.items() if count < 2)
        if invalid_groups:
            raise ValueError(
                "groups require at least two actor members; invalid groups: "
                + ", ".join(invalid_groups)
            )

        for group in self.groups:
            if isinstance(group.mobility, MeshSequenceMobilitySpec) and member_roles[
                group.name
            ] != {"target"}:
                raise ValueError(
                    f"group '{group.name}' uses target-only mesh_sequence mobility but has "
                    "a non-target member"
                )

        has_motion = any(
            not isinstance(actor.mobility, (StationaryMobilitySpec, GroupMemberMobilitySpec))
            for actor in actors
        ) or any(not isinstance(group.mobility, StationaryMobilitySpec) for group in self.groups)
        has_time_varying_orientation = any(
            isinstance(
                actor.orientation,
                (KeyframesOrientationSpec, SpinOrientationSpec, RandomOrientationSpec),
            )
            for actor in actors
        )
        if has_motion or has_time_varying_orientation:
            if self.timeline.steps < 2:
                raise ValueError("moving scenarios require timeline.steps >= 2")
            if self.timeline.duration_s <= 0.0:
                raise ValueError("moving scenarios require timeline.duration_s > 0")

        return self
