"""
Tests for generator configuration constants and pure helper functions.

These tests verify QUALITY_PRESETS, AVAILABLE_SCENES, and pure config logic.
Heavy dependencies (mitsuba, sionna) are not required for these tests.
The full config classes (SimulationConfig, TransmitterConfig) are skipped
when dependencies are unavailable.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# Find the generator path
_generator_root = Path(__file__).resolve().parents[3] / "generator"
_core_root = _generator_root / "core"
_configuration_root = _core_root / "configuration"


def _load_pure_module(name: str, filepath: Path):
    """Load a module directly from filepath."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the presets module (no heavy deps)
_constants_module = _load_pure_module(
    "generator.core.configuration.presets_test", _configuration_root / "presets.py"
)
QUALITY_PRESETS = _constants_module.QUALITY_PRESETS
AVAILABLE_SCENES = _constants_module.AVAILABLE_SCENES

_defaults_module = _load_pure_module(
    "generator.core.configuration.defaults_test", _configuration_root / "defaults.py"
)

# Load exceptions module (no heavy deps)
_exceptions_module = _load_pure_module(
    "generator.core.exceptions_test", _core_root / "exceptions.py"
)
ConfigurationError = _exceptions_module.ConfigurationError


class TestQualityPresets:
    """Test the QUALITY_PRESETS constant."""

    def test_all_presets_have_required_keys(self):
        """All quality presets should have required keys."""
        required_keys = [
            "max_depth",
            "samples_per_src",
            "max_num_paths_per_src",
            "los",
            "specular_reflection",
            "diffuse_reflection",
        ]
        for preset_name, preset in QUALITY_PRESETS.items():
            for key in required_keys:
                assert key in preset, f"Preset '{preset_name}' missing key '{key}'"

    def test_common_presets_exist(self):
        """Common preset names should exist."""
        common_presets = ["low", "medium", "high"]
        for preset in common_presets:
            assert preset in QUALITY_PRESETS

    def test_low_preset_values(self):
        """Test specific values in the low preset."""
        low = QUALITY_PRESETS["low"]
        assert low["max_depth"] == 3
        assert low["samples_per_src"] == 1000000
        assert low["los"] is True
        assert low["specular_reflection"] is True
        assert low["diffuse_reflection"] is False

    def test_medium_preset_values(self):
        """Test specific values in the medium preset."""
        medium = QUALITY_PRESETS["medium"]
        assert medium["max_depth"] == 4
        assert medium["samples_per_src"] == 10000000
        assert medium["los"] is True

    def test_high_preset_values(self):
        """Test specific values in the high preset."""
        high = QUALITY_PRESETS["high"]
        assert high["max_depth"] == 5
        assert high["samples_per_src"] == 10000000
        assert high["refraction"] is True
        assert high["diffraction"] is True

    def test_ultra_preset_values(self):
        """Test specific values in the ultra preset."""
        ultra = QUALITY_PRESETS["ultra"]
        assert ultra["max_depth"] == 6
        assert ultra["refraction"] is True
        assert ultra["diffraction"] is True

    def test_presets_have_seed(self):
        """All presets should have a seed value."""
        for preset_name, preset in QUALITY_PRESETS.items():
            assert "seed" in preset, f"Preset '{preset_name}' missing 'seed'"


class TestConfigurationDefaults:
    """Test named generator-side configuration defaults."""

    def test_simulation_and_rf_defaults_match_runtime_contract(self):
        assert _defaults_module.DEFAULT_SCENE_NAME == "etoile"
        assert _defaults_module.DEFAULT_DURATION_S == 10.0
        assert _defaults_module.DEFAULT_NUM_STEPS == 10
        assert _defaults_module.DEFAULT_START_STEP == 0
        assert _defaults_module.DEFAULT_QUALITY_PRESET == "low"
        assert _defaults_module.CUSTOM_QUALITY_BASE_PRESET == "medium"
        assert _defaults_module.DEFAULT_VIEW == "top"
        assert _defaults_module.DEFAULT_FILE_FORMAT == "hdf5"
        assert _defaults_module.DEFAULT_OUTPUT_MODE == "local"
        assert _defaults_module.DEFAULT_CARRIER_FREQUENCY_HZ == 28e9
        assert _defaults_module.DEFAULT_BANDWIDTH_HZ == 2e9
        assert _defaults_module.DEFAULT_TEMPERATURE_K is None

    def test_coverage_defaults_match_legacy_values(self):
        assert _defaults_module.DEFAULT_COVERAGE_RESOLUTION_M == (1.0, 1.0)
        assert _defaults_module.DEFAULT_COVERAGE_METRIC == "path_loss_db"
        assert _defaults_module.DEFAULT_COVERAGE_STRIDE == 1
        assert _defaults_module.DEFAULT_COVERAGE_QUALITY_PRESET == "medium"
        assert _defaults_module.DEFAULT_COVERAGE_TX_COMBINATION == "best_server"
        assert _defaults_module.DEFAULT_COVERAGE_TX_MODE == "per_tx"
        assert _defaults_module.DEFAULT_COVERAGE_METRICS_STORE == ("path_gain_linear",)
        assert _defaults_module.DEFAULT_COVERAGE_SAVE_COMPRESSION == "lzf"

    def test_sensing_defaults_match_runtime_contract(self):
        assert _defaults_module.DEFAULT_SENSING_BANDWIDTH_HZ == 100e6
        assert _defaults_module.DEFAULT_SENSING_CHIRPS_PER_FRAME == 128
        assert _defaults_module.DEFAULT_SENSING_PRF_HZ == 2000.0
        assert _defaults_module.DEFAULT_SENSING_RANGE_MODE == "bistatic"
        assert _defaults_module.DEFAULT_SENSING_FFT_SIZE_RANGE == 256
        assert _defaults_module.DEFAULT_SENSING_FFT_SIZE_DOPPLER == 128
        assert _defaults_module.DEFAULT_SENSING_RAW_PAYLOAD_POLICY == "none"
        assert _defaults_module.DEFAULT_SENSING_PROCESSING_MODE == "auto"
        assert _defaults_module.DEFAULT_SENSING_TRACKER_TYPE == "m_of_n"

    def test_antenna_defaults_and_valid_values_match_legacy_values(self):
        assert _defaults_module.DEFAULT_ANTENNA_PATTERN == "iso"
        assert _defaults_module.DEFAULT_ANTENNA_POLARIZATION == "V"
        assert _defaults_module.DEFAULT_ANTENNA_NUM_ROWS == 1
        assert _defaults_module.DEFAULT_ANTENNA_NUM_COLS == 1
        assert _defaults_module.DEFAULT_ANTENNA_VERTICAL_SPACING == 0.5
        assert _defaults_module.DEFAULT_ANTENNA_HORIZONTAL_SPACING == 0.5
        assert _defaults_module.VALID_ANTENNA_PATTERNS == frozenset(
            {"iso", "dipole", "hw_dipole", "tr38901"}
        )
        assert _defaults_module.VALID_POLARIZATIONS == frozenset({"V", "H"})


class TestAvailableScenes:
    """Test the AVAILABLE_SCENES constant."""

    def test_common_scenes_exist(self):
        """Common scene names should exist."""
        common_scenes = ["etoile", "box", "munich"]
        for scene in common_scenes:
            assert scene in AVAILABLE_SCENES

    def test_scenes_have_descriptions(self):
        """All scenes should have string descriptions."""
        for scene_name, description in AVAILABLE_SCENES.items():
            assert isinstance(description, str)
            assert len(description) > 0

    def test_etoile_description(self):
        """Etoile scene should have its description."""
        assert "toile" in AVAILABLE_SCENES["etoile"]  # "Étoile" or "Etoile"

    def test_box_scene_exists(self):
        """Box scene should exist."""
        assert "box" in AVAILABLE_SCENES
        assert "box_one_screen" in AVAILABLE_SCENES
        assert "box_two_screens" in AVAILABLE_SCENES

    def test_reflector_scenes_exist(self):
        """Reflector scenes should exist."""
        reflector_scenes = ["simple_reflector", "double_reflector", "triple_reflector"]
        for scene in reflector_scenes:
            assert scene in AVAILABLE_SCENES


class TestConfigurationError:
    """Test the ConfigurationError exception."""

    def test_can_raise_configuration_error(self):
        """ConfigurationError should be raisable."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("test error")

    def test_error_message_preserved(self):
        """Error message should be preserved."""
        try:
            raise ConfigurationError("custom message")
        except ConfigurationError as e:
            assert "custom message" in str(e)

    def test_is_exception_subclass(self):
        """ConfigurationError should be an Exception subclass."""
        assert issubclass(ConfigurationError, Exception)


# Parametrized tests for quality presets
@pytest.mark.parametrize(
    "preset_name",
    ["ultra-low", "low", "medium", "high", "ultra"],
)
def test_preset_exists(preset_name):
    """Test that all expected presets exist."""
    assert preset_name in QUALITY_PRESETS


@pytest.mark.parametrize(
    "preset_name,expected_depth",
    [
        ("low", 3),
        ("medium", 4),
        ("high", 5),
        ("ultra", 6),
        ("ultra-low", 2),
    ],
)
def test_preset_max_depth(preset_name, expected_depth):
    """Test max_depth values for specific presets."""
    assert QUALITY_PRESETS[preset_name]["max_depth"] == expected_depth


@pytest.mark.parametrize(
    "preset_name,has_refraction",
    [
        ("low", False),
        ("medium", False),
        ("high", True),
        ("ultra", True),
    ],
)
def test_preset_refraction(preset_name, has_refraction):
    """Test refraction setting for specific presets."""
    assert QUALITY_PRESETS[preset_name]["refraction"] is has_refraction


def test_device_configs_default_to_the_shared_fixed_orientation_spec() -> None:
    from generator.core.configuration import ReceiverConfig, TransmitterConfig
    from generator.core.mobility import StationaryMobility
    from shared.scenarios.actors import FixedOrientationSpec

    transmitter = TransmitterConfig(
        name="TX1",
        mobility=StationaryMobility(start_pos=(0.0, 0.0, 0.0)),
    )
    receiver = ReceiverConfig(
        name="RX1",
        mobility=StationaryMobility(start_pos=(1.0, 0.0, 0.0)),
    )

    assert transmitter.orientation == FixedOrientationSpec()
    assert receiver.orientation == FixedOrientationSpec()
    assert type(transmitter.orientation) is FixedOrientationSpec
    assert type(receiver.orientation) is FixedOrientationSpec


def test_device_configs_preserve_direct_shared_orientation_specs() -> None:
    from generator.core.configuration import TransmitterConfig
    from generator.core.mobility import StationaryMobility
    from shared.scenarios.actors import SpinOrientationSpec

    authored = SpinOrientationSpec(axis="yaw", rate_deg_s=30.0, yaw_deg=5.0)
    transmitter = TransmitterConfig(
        name="TX1",
        mobility=StationaryMobility(start_pos=(0.0, 0.0, 0.0)),
        orientation=authored,
    )

    assert transmitter.orientation is authored


@pytest.mark.parametrize(
    "scene_name",
    [
        "etoile",
        "simple_street_canyon",
        "simple_street_canyon_with_cars",
        "munich",
        "san_francisco",
        "florence",
        "box",
        "floor_wall",
        "simple_wedge",
    ],
)
def test_scene_exists(scene_name):
    """Test that all expected scenes exist."""
    assert scene_name in AVAILABLE_SCENES


class TestPresetConsistency:
    """Test consistency across presets."""

    def test_all_presets_have_same_keys(self):
        """All presets should have the same set of keys."""
        preset_keys = [set(p.keys()) for p in QUALITY_PRESETS.values()]
        first_keys = preset_keys[0]
        for i, keys in enumerate(preset_keys[1:], 1):
            preset_names = list(QUALITY_PRESETS.keys())
            assert (
                keys == first_keys
            ), f"Preset '{preset_names[i]}' has different keys than '{preset_names[0]}'"

    def test_depth_increases_with_quality(self):
        """Higher quality presets should generally have higher max_depth."""
        low_depth = QUALITY_PRESETS["low"]["max_depth"]
        medium_depth = QUALITY_PRESETS["medium"]["max_depth"]
        high_depth = QUALITY_PRESETS["high"]["max_depth"]

        assert medium_depth >= low_depth
        assert high_depth >= medium_depth

    def test_rays_increase_with_quality(self):
        """Higher quality presets should have more rays."""
        low_rays = QUALITY_PRESETS["low"]["samples_per_src"]
        medium_rays = QUALITY_PRESETS["medium"]["samples_per_src"]
        high_rays = QUALITY_PRESETS["high"]["samples_per_src"]

        assert medium_rays >= low_rays
        assert high_rays >= medium_rays
