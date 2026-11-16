"""Compatibility and import boundaries for the shared sensing package."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _fresh_interpreter(code: str) -> subprocess.CompletedProcess[str]:
    """Use this checkout without inheriting a runner's imported optional stacks."""
    bootstrap = f"import sys\nsys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
    result = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", "-B", "-c", bootstrap + textwrap.dedent(code)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        # Isolated children ignore PYTHONUTF8; keep pipe encoding explicit on Windows.
        encoding="utf-8",
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_sensing_package_and_numerical_api_keep_optional_stacks_lazy() -> None:
    """Browsing namespaces and using numerical contracts must not load UI or solvers."""
    _fresh_interpreter("""
        forbidden = {
            "matplotlib", "scipy", "sklearn", "PySide6", "PyQt5", "PyQt6",
            "sionna", "mitsuba", "drjit", "tensorflow", "torch", "open3d", "pygfx",
        }

        class RejectOptionalStack:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in forbidden:
                    raise AssertionError("Eager optional dependency: " + fullname)
                return None

        sys.meta_path.insert(0, RejectOptionalStack())
        import shared.sensing
        import shared.sensing.detection
        import shared.sensing.fusion
        import shared.sensing.plotting
        import shared.sensing.volume

        assert "shared.sensing.signal_processing" not in sys.modules
        from shared.sensing import RangeDopplerResult, compute_range_profile
        from shared.sensing.signal_processing import RangeDopplerResult as CanonicalResult

        assert RangeDopplerResult is CanonicalResult
        assert callable(compute_range_profile)
        assert not forbidden.intersection(name.split(".", 1)[0] for name in sys.modules)
        """)


def test_existing_numerical_imports_resolve_to_the_canonical_objects() -> None:
    """Existing callers and saved Python class references retain one type identity."""
    _fresh_interpreter("""
        import pickle

        import shared.sensing as sensing
        from shared.sensing import signal_processing

        for name in (
            "C0",
            "RangeProfileResult",
            "RangeDopplerResult",
            "DopplerProfileResult",
            "DetectionVelocityDisplay",
            "compute_range_profile",
            "cir_to_samples",
            "doppler_to_velocity_m_s",
            "sensing_window",
        ):
            assert getattr(sensing, name) is getattr(signal_processing, name), name

        historical_type = pickle.loads(b"cshared.sensing\\nRangeProfileResult\\n.")
        assert historical_type is signal_processing.RangeProfileResult
        """)


def test_existing_postprocessor_module_command_preserves_cli_help() -> None:
    """The retained module launcher must run the same parser as the canonical CLI."""
    outputs = []
    for module in ("shared.sensing_postprocessor", "shared.sensing.postprocessor"):
        result = _fresh_interpreter(f"""
            import runpy

            sys.argv = ["orchav-sensing", "--help"]
            runpy.run_module({module!r}, run_name="__main__")
            """)
        assert "--output-dir" in result.stdout
        assert "--recipes" in result.stdout
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
