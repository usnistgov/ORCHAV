"""
Tests for antenna array configuration.

Covers:
- AntennaConfig dataclass defaults and validation
- AntennaArrayModel / AntennaModel Pydantic validation
- YAML round-trip: absent antenna → None, full antenna → populated, tx-only → rx None
- Config loader antenna extraction
- create_planar_array helper (skipped if sionna not available)
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Load pure modules without heavy deps (mitsuba, sionna)
# ---------------------------------------------------------------------------
_generator_root = Path(__file__).resolve().parents[3] / "generator"
_core_root = _generator_root / "core"


def _load_pure_module(name: str, filepath: Path):
    """Load a module directly from filepath."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load exceptions module (no heavy deps)
_exceptions_module = _load_pure_module(
    "generator.core.exceptions_test_ant", _core_root / "exceptions.py"
)
ConfigurationError = _exceptions_module.ConfigurationError


# ---------------------------------------------------------------------------
# AntennaConfig dataclass tests (requires mitsuba stub for config.py import)
# ---------------------------------------------------------------------------
_has_mitsuba = importlib.util.find_spec("mitsuba") is not None


@pytest.fixture()
def antenna_config_class():
    """Import AntennaConfig, mocking mitsuba if needed."""
    if _has_mitsuba:
        from generator.core.configuration import AntennaConfig

        return AntennaConfig
    else:
        # Provide a mitsuba stub so config.py can import
        mi_mock = MagicMock()
        mi_mock.Point3f = MagicMock()
        sys.modules.setdefault("mitsuba", mi_mock)
        try:
            from generator.core.configuration import AntennaConfig

            return AntennaConfig
        finally:
            pass


class TestAntennaConfigDefaults:
    """Test AntennaConfig default values."""

    def test_defaults(self, antenna_config_class):
        cfg = antenna_config_class()
        assert cfg.pattern == "iso"
        assert cfg.polarization == "V"
        assert cfg.num_rows == 1
        assert cfg.num_cols == 1
        assert cfg.vertical_spacing == 0.5
        assert cfg.horizontal_spacing == 0.5

    def test_custom_values(self, antenna_config_class):
        cfg = antenna_config_class(
            pattern="tr38901",
            polarization="H",
            num_rows=4,
            num_cols=4,
            vertical_spacing=0.7,
            horizontal_spacing=0.3,
        )
        assert cfg.pattern == "tr38901"
        assert cfg.polarization == "H"
        assert cfg.num_rows == 4
        assert cfg.num_cols == 4
        assert cfg.vertical_spacing == 0.7
        assert cfg.horizontal_spacing == 0.3


class TestAntennaConfigValidation:
    """Test AntennaConfig __post_init__ validation."""

    def test_invalid_pattern(self, antenna_config_class):
        with pytest.raises(Exception, match="Invalid antenna pattern"):
            antenna_config_class(pattern="bad")

    def test_invalid_polarization(self, antenna_config_class):
        with pytest.raises(Exception, match="Invalid polarization"):
            antenna_config_class(polarization="X")

    def test_zero_rows(self, antenna_config_class):
        with pytest.raises(Exception, match="num_rows must be >= 1"):
            antenna_config_class(num_rows=0)

    def test_negative_cols(self, antenna_config_class):
        with pytest.raises(Exception, match="num_cols must be >= 1"):
            antenna_config_class(num_cols=-1)

    def test_zero_vertical_spacing(self, antenna_config_class):
        with pytest.raises(Exception, match="vertical_spacing must be > 0"):
            antenna_config_class(vertical_spacing=0.0)

    def test_negative_horizontal_spacing(self, antenna_config_class):
        with pytest.raises(Exception, match="horizontal_spacing must be > 0"):
            antenna_config_class(horizontal_spacing=-0.1)

    def test_all_valid_patterns(self, antenna_config_class):
        for p in ("iso", "dipole", "hw_dipole", "tr38901"):
            cfg = antenna_config_class(pattern=p)
            assert cfg.pattern == p

    def test_both_polarizations(self, antenna_config_class):
        for pol in ("V", "H"):
            cfg = antenna_config_class(polarization=pol)
            assert cfg.polarization == pol


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------
class TestAntennaArrayModel:
    """Test Pydantic AntennaArrayModel."""

    def test_defaults(self):
        from shared.scenarios.model import AntennaArrayModel

        m = AntennaArrayModel()
        assert m.pattern == "iso"
        assert m.polarization == "V"
        assert m.num_rows == 1
        assert m.num_cols == 1
        assert m.vertical_spacing == 0.5
        assert m.horizontal_spacing == 0.5

    def test_custom_values(self):
        from shared.scenarios.model import AntennaArrayModel

        m = AntennaArrayModel(pattern="tr38901", polarization="H", num_rows=4, num_cols=2)
        assert m.pattern == "tr38901"
        assert m.polarization == "H"
        assert m.num_rows == 4
        assert m.num_cols == 2

    def test_extra_field_rejected(self):
        from pydantic import ValidationError

        from shared.scenarios.model import AntennaArrayModel

        with pytest.raises(ValidationError):
            AntennaArrayModel(pattern="iso", bogus_field=True)

    def test_invalid_pattern_rejected(self):
        from shared.scenarios.model import AntennaArrayModel

        # Pattern is now a free-form string to support custom .mat antenna
        # patterns, so any string is accepted.
        model = AntennaArrayModel(pattern="custom_pattern")
        assert model.pattern == "custom_pattern"

    def test_invalid_polarization_rejected(self):
        from pydantic import ValidationError

        from shared.scenarios.model import AntennaArrayModel

        with pytest.raises(ValidationError):
            AntennaArrayModel(polarization="X")


class TestAntennaModel:
    """Test Pydantic AntennaModel."""

    def test_defaults_none(self):
        from shared.scenarios.model import AntennaModel

        m = AntennaModel()
        assert m.tx is None
        assert m.rx is None

    def test_tx_only(self):
        from shared.scenarios.model import AntennaModel

        m = AntennaModel(tx={"pattern": "dipole", "num_rows": 2, "num_cols": 2})
        assert m.tx is not None
        assert m.tx.pattern == "dipole"
        assert m.tx.num_rows == 2
        assert m.rx is None

    def test_both(self):
        from shared.scenarios.model import AntennaModel

        m = AntennaModel(
            tx={"pattern": "tr38901", "num_rows": 4, "num_cols": 4},
            rx={"pattern": "iso"},
        )
        assert m.tx.pattern == "tr38901"
        assert m.tx.num_rows == 4
        assert m.rx.pattern == "iso"
        assert m.rx.num_rows == 1  # default


class TestRayTracingModelAntenna:
    """Test antenna field on RayTracingModel."""

    def test_absent_antenna(self):
        from shared.scenarios.model import RayTracingModel

        m = RayTracingModel()
        assert m.antenna is None

    def test_with_antenna(self):
        from shared.scenarios.model import RayTracingModel

        m = RayTracingModel(antenna={"tx": {"pattern": "tr38901", "num_rows": 4, "num_cols": 4}})
        assert m.antenna is not None
        assert m.antenna.tx.pattern == "tr38901"
        assert m.antenna.rx is None


class TestScenarioModelAntenna:
    """Test full scenario model with antenna."""

    def test_full_scenario_with_antenna(self):
        from shared.scenarios.model import ScenarioModel

        m = ScenarioModel(
            schema_version=2,
            timeline={"steps": 10, "duration_s": 1.0},
            raytracing={
                "antenna": {
                    "tx": {"pattern": "tr38901", "polarization": "V", "num_rows": 4, "num_cols": 4},
                    "rx": {"pattern": "iso"},
                },
            },
        )
        assert m.raytracing.antenna.tx.num_rows == 4
        assert m.raytracing.antenna.rx.pattern == "iso"

    def test_scenario_without_antenna(self):
        from shared.scenarios.model import ScenarioModel

        m = ScenarioModel(
            schema_version=2,
            timeline={"steps": 10, "duration_s": 1.0},
            raytracing={},
        )
        assert m.raytracing.antenna is None

    def test_scenario_accepts_custom_antenna_pattern(self):
        from shared.scenarios.model import ScenarioModel

        # Pattern identifiers may refer to custom antenna resources.
        model = ScenarioModel(
            schema_version=2,
            timeline={"steps": 1, "duration_s": 0.0},
            raytracing={"antenna": {"tx": {"pattern": "custom_mat_file"}}},
        )
        assert model.raytracing.antenna.tx.pattern == "custom_mat_file"


# ---------------------------------------------------------------------------
# Section parser round-trip
# ---------------------------------------------------------------------------
class TestSectionParserAntenna:
    """Test that parse_raytracing_config passes antenna through."""

    def test_absent_antenna(self):
        from shared.scenarios.parsers import parse_raytracing_config

        result = parse_raytracing_config({})
        assert result.get("antenna") is None

    def test_present_antenna(self):
        from shared.scenarios.parsers import parse_raytracing_config

        data = {
            "raytracing": {
                "antenna": {
                    "tx": {"pattern": "tr38901", "num_rows": 4, "num_cols": 4},
                    "rx": {"pattern": "iso"},
                }
            }
        }
        result = parse_raytracing_config(data)
        ant = result["antenna"]
        assert ant["tx"]["pattern"] == "tr38901"
        assert ant["rx"]["pattern"] == "iso"

    def test_tx_only_antenna(self):
        from shared.scenarios.parsers import parse_raytracing_config

        data = {
            "raytracing": {
                "antenna": {
                    "tx": {"pattern": "dipole", "num_rows": 2, "num_cols": 2},
                }
            }
        }
        result = parse_raytracing_config(data)
        ant = result["antenna"]
        assert ant["tx"]["pattern"] == "dipole"
        assert "rx" not in ant


# ---------------------------------------------------------------------------
# Config loader round-trip
# ---------------------------------------------------------------------------
class TestConfigLoaderAntenna:
    """Test that load_simulation_config populates tx_antenna / rx_antenna."""

    @pytest.fixture()
    def _sim_config_class(self):
        """Ensure SimulationConfig is importable."""
        if not _has_mitsuba:
            mi_mock = MagicMock()
            mi_mock.Point3f = MagicMock()
            mi_mock.Float = MagicMock()
            sys.modules.setdefault("mitsuba", mi_mock)
        from generator.core.configuration import SimulationConfig

        return SimulationConfig

    def _make_scenario(self, antenna_dict=None):
        """Create a minimal scenario-like object with raytracing.antenna."""

        class FakeScenario:
            scene_source = "sionna"
            scene_id = "etoile"
            debug_level = None
            frames_format = "h5"
            data_mode = "files"
            live_grpc_endpoints = {}

        sc = FakeScenario()
        rt = {"steps": 5, "quality": {"preset": "low"}}
        if antenna_dict is not None:
            rt["antenna"] = antenna_dict
        sc.raytracing = rt
        sc.sensing = {}
        return sc

    def test_no_antenna_yields_none(self, _sim_config_class):
        from generator.core.configuration import load_simulation_config

        sc = self._make_scenario()
        cfg = load_simulation_config(sc)
        assert cfg.tx_antenna is None
        assert cfg.rx_antenna is None

    def test_full_antenna(self, _sim_config_class):
        from generator.core.configuration import load_simulation_config

        sc = self._make_scenario(
            antenna_dict={
                "tx": {"pattern": "tr38901", "num_rows": 4, "num_cols": 4},
                "rx": {"pattern": "dipole", "polarization": "H"},
            }
        )
        cfg = load_simulation_config(sc)
        assert cfg.tx_antenna is not None
        assert cfg.tx_antenna.pattern == "tr38901"
        assert cfg.tx_antenna.num_rows == 4
        assert cfg.rx_antenna is not None
        assert cfg.rx_antenna.pattern == "dipole"
        assert cfg.rx_antenna.polarization == "H"

    def test_tx_only(self, _sim_config_class):
        from generator.core.configuration import load_simulation_config

        sc = self._make_scenario(
            antenna_dict={"tx": {"pattern": "hw_dipole", "num_rows": 2, "num_cols": 2}}
        )
        cfg = load_simulation_config(sc)
        assert cfg.tx_antenna is not None
        assert cfg.tx_antenna.pattern == "hw_dipole"
        assert cfg.rx_antenna is None

    def test_empty_role_mappings_apply_explicit_defaults(self, _sim_config_class):
        from generator.core.configuration import AntennaConfig, load_simulation_config

        base = _sim_config_class(
            tx_antenna=AntennaConfig(pattern="tr38901", num_rows=4),
            rx_antenna=AntennaConfig(pattern="dipole", num_cols=2),
        )
        cfg = load_simulation_config(
            self._make_scenario(antenna_dict={"tx": {}, "rx": {}}),
            base=base,
        )

        assert cfg.tx_antenna == AntennaConfig()
        assert cfg.rx_antenna == AntennaConfig()

    @pytest.mark.parametrize(
        "antenna_dict",
        [None, {"tx": None, "rx": None}],
        ids=["omitted", "null"],
    )
    def test_omitted_or_null_roles_do_not_apply_overrides(
        self,
        _sim_config_class,
        antenna_dict,
    ):
        from generator.core.configuration import AntennaConfig, load_simulation_config

        base = _sim_config_class(
            tx_antenna=AntennaConfig(pattern="tr38901"),
            rx_antenna=AntennaConfig(pattern="dipole"),
        )
        cfg = load_simulation_config(
            self._make_scenario(antenna_dict=antenna_dict),
            base=base,
        )

        assert cfg.tx_antenna == base.tx_antenna
        assert cfg.rx_antenna == base.rx_antenna

    def test_pattern_whitespace_is_normalized(self, _sim_config_class):
        from generator.core.configuration import load_simulation_config

        cfg = load_simulation_config(
            self._make_scenario(
                antenna_dict={
                    "tx": {"pattern": "  tr38901\t"},
                    "rx": {"pattern": "\n dipole  "},
                }
            )
        )

        assert cfg.tx_antenna is not None
        assert cfg.tx_antenna.pattern == "tr38901"
        assert cfg.rx_antenna is not None
        assert cfg.rx_antenna.pattern == "dipole"

    def test_base_copy_preserves_antenna(self, _sim_config_class):
        from generator.core.configuration import AntennaConfig, load_simulation_config

        base = _sim_config_class(
            tx_antenna=AntennaConfig(pattern="tr38901", num_rows=8, num_cols=8),
        )
        cfg = load_simulation_config(None, base=base)
        assert cfg.tx_antenna is not None
        assert cfg.tx_antenna is not base.tx_antenna
        assert cfg.tx_antenna.pattern == "tr38901"
        assert cfg.tx_antenna.num_rows == 8

        cfg.tx_antenna.num_rows = 2
        assert base.tx_antenna.num_rows == 8


# ---------------------------------------------------------------------------
# create_planar_array (skip if sionna not available)
# ---------------------------------------------------------------------------
_has_sionna = importlib.util.find_spec("sionna") is not None


def _has_usable_planar_array() -> bool:
    if not _has_sionna:
        return False
    try:
        # Importing Sionna RT selects the backend appropriate for the host.
        # Do not force LLVM here: that masks a working CUDA runtime on Windows
        # and makes this availability probe disagree with real generators.
        from generator.core.scenario_entities.antenna_arrays import create_planar_array

        create_planar_array(None)
    except Exception:
        return False
    return True


_has_planar_array_runtime = _has_usable_planar_array()


@pytest.mark.skipif(
    not _has_planar_array_runtime, reason="usable Sionna PlanarArray runtime unavailable"
)
@pytest.mark.optional_runtime
class TestCreatePlanarArray:
    """Test create_planar_array helper with real Sionna.

    Note: Sionna's PlanarArray exposes ``num_ant`` (total antenna count)
    but not ``num_rows``/``num_cols`` as readable attributes.
    """

    def test_default_none(self):
        from generator.core.scenario_entities.antenna_arrays import create_planar_array

        arr = create_planar_array(None)
        assert arr.num_ant == 1  # 1x1

    def test_default_no_args(self):
        from generator.core.scenario_entities.antenna_arrays import create_planar_array

        arr = create_planar_array()
        assert arr.num_ant == 1  # 1x1

    def test_default_none_uses_sionna_defaults(self, monkeypatch):
        import generator.core.scenario_entities.antenna_arrays as antenna_arrays

        call_kwargs = {}

        class FakePlanarArray:
            num_ant = 1

            def __init__(self, **kwargs):
                call_kwargs.update(kwargs)

        monkeypatch.setattr(antenna_arrays, "PlanarArray", FakePlanarArray)

        arr = antenna_arrays.create_planar_array(None)

        assert arr.num_ant == 1
        assert call_kwargs == {
            "num_rows": 1,
            "num_cols": 1,
            "pattern": "iso",
            "polarization": "V",
        }

    def test_custom_config(self):
        from generator.core.configuration import AntennaConfig
        from generator.core.scenario_entities.antenna_arrays import create_planar_array

        cfg = AntennaConfig(
            pattern="tr38901",
            polarization="V",
            num_rows=4,
            num_cols=4,
            vertical_spacing=0.7,
            horizontal_spacing=0.3,
        )
        arr = create_planar_array(cfg)
        assert arr.num_ant == 16  # 4x4
