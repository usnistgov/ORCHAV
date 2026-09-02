"""
Tests for OSS audit changes across multiple agents.

Covers:
- Dead code removal verification (NIST, standalone_*, observables/tier1/tier2)
- Debug quality preset (G1)
- Config validation at pipeline start (G2)
- FileProvider schema-skip optimization (P12)
- make_lookup_mask correctness (P6 np.unique replacement)
- AppState vestigial key handling (code-cleanup)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------- project roots ------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_core_root = PROJECT_ROOT / "generator" / "core"
_configuration_root = _core_root / "configuration"


def _load_pure_module(name: str, filepath: Path):
    """Load a module directly from filepath (avoids heavy generator deps)."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Pure-Python modules (no GPU / mitsuba / sionna deps)
_constants = _load_pure_module(
    "generator.core.configuration.presets_oss", _configuration_root / "presets.py"
)
_exceptions = _load_pure_module("generator.core.exceptions_oss", _core_root / "exceptions.py")

QUALITY_PRESETS = _constants.QUALITY_PRESETS
ConfigurationError = _exceptions.ConfigurationError


# ========================================================================
# 1. Dead-code removal verification
# ========================================================================


class TestDeadCodeRemoval:
    """Verify that removed code is actually gone."""

    def test_nist_not_in_generator_all(self):
        """NIST classes should not appear in generator.__all__."""
        # Load the actual generator __init__.py directly to avoid test package shadowing
        import importlib.util as ilu

        gen_init = PROJECT_ROOT / "generator" / "__init__.py"
        spec = ilu.spec_from_file_location("generator_init_check", gen_init)
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)

        nist_names = [n for n in mod.__all__ if "NIST" in n.upper()]
        assert nist_names == [], f"NIST symbols still exported: {nist_names}"

    def test_standalone_tx_position_not_in_appstate(self):
        """standalone_tx_position field should not exist on AppState."""
        from visualizer.src.state import AppState

        assert not hasattr(AppState, "standalone_tx_position")

    def test_standalone_rx_position_not_in_appstate(self):
        """standalone_rx_position field should not exist on AppState."""
        from visualizer.src.state import AppState

        assert not hasattr(AppState, "standalone_rx_position")

    def test_standalone_tx_orientation_not_in_appstate(self):
        """standalone_tx_orientation field should not exist on AppState."""
        from visualizer.src.state import AppState

        assert not hasattr(AppState, "standalone_tx_orientation")

    def test_standalone_rx_orientation_not_in_appstate(self):
        """standalone_rx_orientation field should not exist on AppState."""
        from visualizer.src.state import AppState

        assert not hasattr(AppState, "standalone_rx_orientation")

    def test_observables_tier1_module_removed(self):
        """shared/statistics/observables/tier1.py should not exist."""
        tier1 = PROJECT_ROOT / "shared" / "statistics" / "observables" / "tier1.py"
        assert not tier1.exists(), f"tier1.py should be deleted: {tier1}"

    def test_observables_tier2_module_removed(self):
        """shared/statistics/observables/tier2.py should not exist."""
        tier2 = PROJECT_ROOT / "shared" / "statistics" / "observables" / "tier2.py"
        assert not tier2.exists(), f"tier2.py should be deleted: {tier2}"

    def test_scenario_loader_utils_removed(self):
        """shared/scenarios/utils.py should not exist."""
        utils = PROJECT_ROOT / "shared" / "scenarios" / "utils.py"
        assert not utils.exists(), f"utils.py should be deleted: {utils}"


# ========================================================================
# 2. Debug quality preset (G1)
# ========================================================================


class TestDebugQualityPreset:
    """Verify the 'debug' quality preset exists and has expected values."""

    def test_debug_preset_exists(self):
        """Debug preset must be in QUALITY_PRESETS."""
        assert "debug" in QUALITY_PRESETS

    def test_debug_preset_has_max_depth_1(self):
        """Debug preset should have max_depth=1 for fastest runs."""
        assert QUALITY_PRESETS["debug"]["max_depth"] == 1

    def test_debug_preset_has_fixed_seed(self):
        """Debug preset should use deterministic seed=42."""
        assert QUALITY_PRESETS["debug"]["seed"] == 42

    def test_debug_preset_has_los(self):
        """Debug preset should have LoS enabled."""
        assert QUALITY_PRESETS["debug"]["los"] is True

    def test_debug_preset_disables_specular(self):
        """Debug preset should disable specular reflection for speed."""
        assert QUALITY_PRESETS["debug"]["specular_reflection"] is False

    def test_debug_preset_disables_diffuse(self):
        """Debug preset should disable diffuse reflection for speed."""
        assert QUALITY_PRESETS["debug"]["diffuse_reflection"] is False

    def test_debug_preset_disables_refraction(self):
        """Debug preset should disable refraction for speed."""
        assert QUALITY_PRESETS["debug"]["refraction"] is False

    def test_debug_preset_disables_diffraction(self):
        """Debug preset should disable diffraction for speed."""
        assert QUALITY_PRESETS["debug"]["diffraction"] is False

    def test_debug_preset_has_required_keys(self):
        """Debug preset should have all required quality keys."""
        required = [
            "max_depth",
            "samples_per_src",
            "max_num_paths_per_src",
            "los",
            "specular_reflection",
            "diffuse_reflection",
        ]
        for key in required:
            assert key in QUALITY_PRESETS["debug"], f"Debug preset missing '{key}'"


# ========================================================================
# 3. Config validation at pipeline start (G2)
# ========================================================================


class TestConfigValidationAtPipelineStart:
    """Test that PipelineContext calls validate() on entry."""

    def test_pipeline_context_calls_validate(self):
        """PipelineContext.__enter__ should call config.validate()."""
        from generator.core.pipeline.context import PipelineContext

        mock_config = MagicMock()
        mock_config.validate = MagicMock()

        ctx = PipelineContext(mock_config)
        ctx.__enter__()
        mock_config.validate.assert_called_once()

    def test_pipeline_context_skips_validate_if_missing(self):
        """PipelineContext should not fail if config has no validate()."""
        from generator.core.pipeline.context import PipelineContext

        mock_config = MagicMock(spec=[])  # no .validate attribute
        ctx = PipelineContext(mock_config)
        # Should not raise
        ctx.__enter__()


# ========================================================================
# 4. make_lookup_mask correctness (P6)
# ========================================================================


class TestMakeLookupMask:
    """Test the LUT-based filter mask (np.unique replacement)."""

    def test_basic_lookup(self):
        """Basic allowed orders should produce correct mask."""
        from visualizer.src.metrics.mpc_canon import make_lookup_mask

        lut = make_lookup_mask(5, np.array([0, 2, 4]))
        assert lut[0] is np.True_
        assert lut[1] is np.False_
        assert lut[2] is np.True_
        assert lut[3] is np.False_
        assert lut[4] is np.True_
        assert lut[5] is np.False_

    def test_empty_allowed(self):
        """Empty allowed array should produce all-False mask."""
        from visualizer.src.metrics.mpc_canon import make_lookup_mask

        lut = make_lookup_mask(3, np.array([], dtype=np.int32))
        assert lut.sum() == 0

    def test_allowed_exceeds_max_value(self):
        """Allowed values exceeding max_value should extend the LUT."""
        from visualizer.src.metrics.mpc_canon import make_lookup_mask

        lut = make_lookup_mask(2, np.array([5, 8]))
        assert len(lut) == 9  # 0..8 inclusive
        assert lut[5] is np.True_
        assert lut[8] is np.True_
        assert lut[0] is np.False_
        assert lut[2] is np.False_

    def test_negative_values_ignored(self):
        """Negative values in allowed should be safely ignored."""
        from visualizer.src.metrics.mpc_canon import make_lookup_mask

        lut = make_lookup_mask(3, np.array([-1, 0, 2]))
        assert lut[0] is np.True_
        assert lut[2] is np.True_
        assert lut[1] is np.False_

    def test_zero_max_value(self):
        """max_value=0 with allowed=[0] should work."""
        from visualizer.src.metrics.mpc_canon import make_lookup_mask

        lut = make_lookup_mask(0, np.array([0]))
        assert len(lut) == 1
        assert lut[0] is np.True_


# ========================================================================
# 6. AppState from_dict vestigial key handling
# ========================================================================


class TestAppStateVestigialKeys:
    """Test that AppState.from_dict drops removed standalone_* keys."""

    def test_from_dict_drops_standalone_tx_position(self):
        """from_dict should silently drop standalone_tx_position."""
        from visualizer.src.state import AppState, create_initial_state

        state = create_initial_state()
        d = state.to_dict()
        d["standalone_tx_position"] = [1.0, 2.0, 3.0]
        restored = AppState.from_dict(d)
        assert restored.step == state.step

    def test_from_dict_drops_standalone_rx_orientation(self):
        """from_dict should silently drop standalone_rx_orientation."""
        from visualizer.src.state import AppState, create_initial_state

        state = create_initial_state()
        d = state.to_dict()
        d["standalone_rx_orientation"] = [0.0, 0.0, 0.0]
        restored = AppState.from_dict(d)
        assert restored.step == state.step

    def test_from_dict_drops_all_vestigial_keys(self):
        """from_dict should handle all four vestigial keys at once."""
        from visualizer.src.state import AppState, create_initial_state

        state = create_initial_state(step=42, mpc_layer_enabled=False)
        d = state.to_dict()
        d["standalone_tx_position"] = [1.0, 2.0, 3.0]
        d["standalone_tx_orientation"] = [0.0, 0.0, 0.0]
        d["standalone_rx_position"] = [4.0, 5.0, 6.0]
        d["standalone_rx_orientation"] = [0.1, 0.2, 0.3]
        restored = AppState.from_dict(d)
        assert restored.step == 42
        assert restored.mpc_visibility.enabled is False


# ========================================================================
# 7. FileProvider edge cases
# ========================================================================


class TestFileProviderEdgeCases:
    """Additional edge cases for FileProvider beyond test_file_provider.py."""

    def test_info_name_reflects_subclass(self, tmp_path: Path):
        """Subclass info.name should reflect the actual subclass name."""
        from shared.frames.base import FormatHandler
        from shared.frames.providers import FileProvider
        from shared.frames.types import StandardMPCFrame

        class CustomProvider(FileProvider):
            pass

        class StubHandler(FormatHandler):
            def can_handle(self) -> bool:
                return True

            def list_frames(self) -> list[int]:
                return [0]

            def has_frame(self, step: int) -> bool:
                return step == 0

            def load_frame(self, step: int) -> StandardMPCFrame:
                raise AssertionError("provider metadata must not load frame data")

        handler = StubHandler(tmp_path)
        provider = CustomProvider(tmp_path, handler)
        assert provider.info.name == "CustomProvider"


# ========================================================================
# 8. Config validation errors
# ========================================================================


class TestConfigValidation:
    """Test SimulationConfig.validate() catches invalid settings."""

    def test_invalid_quality_raises(self):
        """Invalid quality preset should raise ConfigurationError."""
        from generator.core.configuration import SimulationConfig
        from generator.core.exceptions import ConfigurationError as CfgError

        sc = SimulationConfig()
        sc.quality = "nonexistent_preset"
        with pytest.raises(CfgError):
            sc.validate()

    def test_debug_quality_validates(self):
        """'debug' quality should pass validation."""
        from generator.core.configuration import SimulationConfig

        sc = SimulationConfig()
        sc.quality = "debug"
        # Should not raise
        sc.validate()

    def test_invalid_num_steps_raises(self):
        """num_steps < 1 should raise ConfigurationError."""
        from generator.core.configuration import SimulationConfig
        from generator.core.exceptions import ConfigurationError as CfgError

        sc = SimulationConfig()
        sc.num_steps = 0
        with pytest.raises(CfgError):
            sc.validate()

    def test_invalid_file_format_raises(self):
        """Non-HDF5 file format should raise ConfigurationError."""
        from generator.core.configuration import SimulationConfig
        from generator.core.exceptions import ConfigurationError as CfgError

        sc = SimulationConfig()
        sc.file_format = "csv"
        with pytest.raises(CfgError):
            sc.validate()


# ========================================================================
# 9. Streaming examples moved to examples/streaming/
# ========================================================================


class TestStreamingExamplesLocation:
    """Verify streaming_examples.py is in examples/streaming/."""

    def test_examples_streaming_exists(self):
        """examples/streaming/streaming_examples.py should exist."""
        path = PROJECT_ROOT / "examples" / "streaming" / "streaming_examples.py"
        assert path.exists(), f"Expected streaming example at {path}"

    def test_old_location_removed(self):
        """streaming_examples.py should NOT be in the project root."""
        old_path = PROJECT_ROOT / "streaming_examples.py"
        assert not old_path.exists(), f"Old location should be removed: {old_path}"


# ========================================================================
# 10. Shared package exports
# ========================================================================


class TestSharedExports:
    """Verify shared package exports remain intentional."""

    def test_file_provider_importable(self):
        """FileProvider should be importable from shared.frames.providers."""
        from shared.frames.providers import FileProvider

        assert FileProvider is not None

    def test_hdf5_provider_importable(self):
        """Hdf5Provider should be importable from shared.frames.providers."""
        from shared.frames.providers import Hdf5Provider

        assert Hdf5Provider is not None

    def test_frame_format_version_importable(self):
        """FRAME_FORMAT_VERSION should be importable from shared.frames.types."""
        from shared.frames.types import FRAME_FORMAT_VERSION

        assert isinstance(FRAME_FORMAT_VERSION, int)
        assert FRAME_FORMAT_VERSION > 0

    def test_shared_init_all_is_empty_facade(self):
        """shared.__init__.py should not re-export subpackage APIs."""
        # Load the actual shared __init__.py to avoid test package shadowing
        import importlib.util as ilu

        shared_init = PROJECT_ROOT / "shared" / "__init__.py"
        spec = ilu.spec_from_file_location("shared_init_check", shared_init)
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert mod.__all__ == []
