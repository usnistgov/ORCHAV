"""Named generator configuration defaults.

Shared scenario models remain the YAML schema authority.  These constants name
the generator-side runtime defaults used after scenario parsing, especially
where a value carries units, domain policy, or is reused across adapter code.
"""

from __future__ import annotations

from typing import Final

from shared.scenarios import defaults as scenario_defaults

# Simulation/runtime output defaults.
DEFAULT_SCENE_NAME: Final = "etoile"
DEFAULT_DURATION_S: Final = 10.0
DEFAULT_NUM_STEPS: Final = 10
DEFAULT_START_STEP: Final = 0
DEFAULT_QUALITY_PRESET: Final = scenario_defaults.DEFAULT_RAYTRACING_QUALITY_PRESET
CUSTOM_QUALITY_BASE_PRESET: Final = "medium"
DEFAULT_VIEW: Final = "top"
DEFAULT_GENERATE_VIDEO: Final = False
DEFAULT_DEBUG_LEVEL: Final = scenario_defaults.DEFAULT_DEBUG_LEVEL
DEFAULT_FILE_FORMAT: Final = "hdf5"
SCENARIO_HDF5_FILE_FORMAT_ALIAS: Final = "h5"
DEFAULT_OUTPUT_MODE: Final = "local"
DEFAULT_GRPC_SERVER_ADDRESS: Final = "localhost:50051"
DEFAULT_SCENE_MATERIAL_SCATTERING_COEFFICIENT_PRESET: Final = "none"
DEFAULT_EXPORT_PATH_METRICS: Final = False

# RF defaults. ``None`` for temperature leaves Sionna's scene default in force.
DEFAULT_CARRIER_FREQUENCY_HZ: Final = 28e9
DEFAULT_BANDWIDTH_HZ: Final = 2e9
DEFAULT_TEMPERATURE_K: Final[float | None] = None

# Sionna CIR evolution and mesh update policy.
DEFAULT_MESH_UPDATE_INTERVAL_S: Final[float | None] = None
DEFAULT_CIR_TIME_STEPS: Final = 1
DEFAULT_CIR_SAMPLING_FREQUENCY_HZ: Final[float | None] = None

# Coverage defaults.
DEFAULT_COVERAGE_ENABLED: Final = False
DEFAULT_COVERAGE_BBOX: Final = None
DEFAULT_COVERAGE_RESOLUTION_M: Final = (1.0, 1.0)
DEFAULT_COVERAGE_HEIGHTS_M: Final = (0.0, 5.0)
DEFAULT_COVERAGE_METRIC: Final = "path_loss_db"
DEFAULT_COVERAGE_STRIDE: Final = 1
DEFAULT_COVERAGE_QUALITY_PRESET: Final = "medium"
DEFAULT_COVERAGE_MAX_PATHS: Final = 1000
DEFAULT_COVERAGE_TX_COMBINATION: Final = "best_server"
DEFAULT_COVERAGE_TX_MODE: Final = scenario_defaults.DEFAULT_COVERAGE_TX_MODE
DEFAULT_COVERAGE_METRICS_STORE: Final = ("path_gain_linear",)
DEFAULT_COVERAGE_METRICS_DERIVED: Final = (
    "path_loss_db",
    "rss_dbm",
    "sinr_db",
    "serving_tx",
    "tx_margin_db",
)
DEFAULT_COVERAGE_SAVE_PATH: Final = "coverage/coverage_maps.h5"
DEFAULT_COVERAGE_SAVE_COMPRESSION: Final = "lzf"

# Sensing waveform and processing defaults.
DEFAULT_SENSING_ENABLED: Final = False
DEFAULT_SENSING_BANDWIDTH_HZ: Final = 100e6
DEFAULT_SENSING_CHIRPS_PER_FRAME: Final = 128
DEFAULT_SENSING_PRF_HZ: Final = 2000.0
DEFAULT_SENSING_RANGE_MODE: Final = "bistatic"
VALID_SENSING_RANGE_MODES: Final = frozenset({"bistatic", "monostatic"})
DEFAULT_SENSING_FFT_SIZE_RANGE: Final = 256
DEFAULT_SENSING_FFT_SIZE_DOPPLER: Final = 128
DEFAULT_SENSING_WINDOW: Final = "hann"

# Sensing detection defaults.
DEFAULT_SENSING_CFAR_TYPE: Final = "ca"
DEFAULT_SENSING_CFAR_GUARD_CELLS: Final = 4
DEFAULT_SENSING_CFAR_REF_CELLS: Final = 16
DEFAULT_SENSING_CFAR_THRESHOLD_SCALE: Final = 3.0
DEFAULT_SENSING_CFAR_OS_RANK: Final = 0.75
DEFAULT_SENSING_CFAR_RD_GUARD_CELLS_RANGE: Final = 2
DEFAULT_SENSING_CFAR_RD_GUARD_CELLS_DOPPLER: Final = 1
DEFAULT_SENSING_CFAR_RD_REF_CELLS_RANGE: Final = 8
DEFAULT_SENSING_CFAR_RD_REF_CELLS_DOPPLER: Final = 4
DEFAULT_SENSING_CFAR_MIN_SNR_DB: Final = 6.0
DEFAULT_SENSING_PRE_CFAR_DOPPLER_EDGE_GUARD_BINS: Final = 2
DEFAULT_SENSING_PRE_CFAR_DOPPLER_DC_GUARD_BINS: Final = 0

# Sensing tracker and evaluation gates.
DEFAULT_SENSING_TRACK_CONFIRMATION_ENABLED: Final = False
DEFAULT_SENSING_TRACK_CONFIRMATION_M: Final = 2
DEFAULT_SENSING_TRACK_CONFIRMATION_N: Final = 3
DEFAULT_SENSING_TRACK_MAX_MISSED_FRAMES: Final = 1
DEFAULT_SENSING_TRACK_ASSOCIATION_RANGE_GATE_M: Final = 0.6
DEFAULT_SENSING_TRACK_ASSOCIATION_VELOCITY_GATE_M_S: Final = 0.3
DEFAULT_SENSING_TRACK_OUTPUT_TENTATIVE: Final = False
DEFAULT_SENSING_TRACKER_TYPE: Final = "m_of_n"
VALID_SENSING_TRACKER_TYPES: Final = frozenset({"m_of_n", "kalman", "jpdaf"})
DEFAULT_SENSING_EVAL_ASSOCIATION_RANGE_GATE_M: Final = 10.0
DEFAULT_SENSING_EVAL_ASSOCIATION_VELOCITY_GATE_M_S: Final = 5.0

# Sensing output and raw payload policy.
DEFAULT_SENSING_CLUTTER_REMOVAL_ENABLED: Final = False
DEFAULT_SENSING_CLUTTER_REMOVAL_WINDOW: Final = 8
DEFAULT_SENSING_OUTPUT_RANGE_DOPPLER: Final = True
DEFAULT_SENSING_OUTPUT_RANGE_PROFILE: Final = False
DEFAULT_SENSING_OUTPUT_DETECTIONS: Final = True
DEFAULT_SENSING_PERSIST_DETECTION_STAGE_ARTIFACTS: Final = False
DEFAULT_SENSING_RAW_PAYLOAD_POLICY: Final = "none"
VALID_SENSING_RAW_PAYLOAD_POLICIES: Final = frozenset({"none", "diagnostic", "always"})
DEFAULT_SENSING_NOISE_ENABLED: Final = False
DEFAULT_SENSING_AOA_FILTER_ENABLED: Final = False
DEFAULT_SENSING_PROCESSING_MODE: Final = "auto"
VALID_SENSING_PROCESSING_MODES: Final = frozenset(
    {"auto", "coherent_cpi", "snapshot_direct_rd", "sequential_stft"}
)

# Antenna defaults and values accepted by generator-side Sionna adapters.
DEFAULT_ANTENNA_PATTERN: Final = "iso"
DEFAULT_ANTENNA_POLARIZATION: Final = "V"
DEFAULT_ANTENNA_NUM_ROWS: Final = 1
DEFAULT_ANTENNA_NUM_COLS: Final = 1
DEFAULT_ANTENNA_VERTICAL_SPACING: Final = 0.5
DEFAULT_ANTENNA_HORIZONTAL_SPACING: Final = 0.5
VALID_ANTENNA_PATTERNS: Final = frozenset({"iso", "dipole", "hw_dipole", "tr38901"})
VALID_POLARIZATIONS: Final = frozenset({"V", "H"})

VALID_VIEWS: Final = ("top", "side", "isometric", "front")
VALID_FILE_FORMATS: Final = ("hdf5", "h5")
VALID_OUTPUT_MODES: Final = ("local", "grpc")
VALID_SCENE_MATERIAL_SCATTERING_COEFFICIENT_PRESETS: Final = ("none", "itu")
