"""Regression tests for visualizer startup import boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("package", ["controllers", "services"])
def test_layer_package_import_does_not_eagerly_import_implementations(package: str) -> None:
    """A package-orientation import must not construct its application graph."""
    qualified = f"visualizer.src.{package}"
    script = textwrap.dedent(f"""
        import importlib
        import json
        import sys

        module = importlib.import_module({qualified!r})
        loaded = sorted(
            name for name in sys.modules
            if name.startswith({qualified + '.'!r})
        )
        print(json.dumps({{"all": list(module.__all__), "loaded": loaded}}))
        """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout.splitlines()[-1])
    assert report == {"all": [], "loaded": []}


@pytest.mark.headless
def test_visualizer_import_does_not_eagerly_load_sionna_rt_runtime(tmp_path) -> None:
    script = textwrap.dedent("""
        import json
        import sys

        import visualizer.visualizer  # noqa: F401

        runtime_modules = sorted(
            name
            for name in sys.modules
            if name == "drjit"
            or name.startswith("drjit.")
            or name == "mitsuba"
            or name.startswith("mitsuba.")
            or name == "sionna.rt"
            or name.startswith("sionna.rt.")
        )
        sys.stdout.write(json.dumps(runtime_modules) + "\\n")
        """)

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    env["QT_QPA_PLATFORM"] = "offscreen"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert stdout_lines, f"expected module report in stdout, stderr was: {result.stderr}"
    assert json.loads(stdout_lines[-1]) == []


@pytest.mark.headless
def test_visualizer_import_defers_optional_feature_modules(tmp_path) -> None:
    """Video, live-gRPC, and metrics UI dependencies load only on demand."""
    script = textwrap.dedent("""
        import json
        import sys

        import visualizer.visualizer  # noqa: F401

        deferred = {
            "imageio": any(name == "imageio" or name.startswith("imageio.") for name in sys.modules),
            "grpc_provider": "visualizer.src.io.grpc_provider" in sys.modules,
            "metrics_window": "visualizer.src.metrics.viz_metrics" in sys.modules,
            "mpc_explorer_window": "visualizer.src.panels.mpc_explorer_window" in sys.modules,
            "mpc_explorer_service": "visualizer.src.services.mpc_explorer_service" in sys.modules,
            "mpc_selection_service": "visualizer.src.services.mpc_selection_service" in sys.modules,
            "mpc_explorer_model": "visualizer.src.model.mpc_explorer_model" in sys.modules,
            "mpc_path_catalog": "visualizer.src.metrics.mpc_path_catalog" in sys.modules,
        }
        print(json.dumps(deferred))
        """)
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    env["QT_QPA_PLATFORM"] = "offscreen"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout.splitlines()[-1]) == {
        "imageio": False,
        "grpc_provider": False,
        "metrics_window": False,
        "mpc_explorer_window": False,
        "mpc_explorer_service": False,
        "mpc_selection_service": False,
        "mpc_explorer_model": False,
        "mpc_path_catalog": False,
    }


@pytest.mark.headless
def test_authoring_compiler_boundary_defers_optional_native_stack(tmp_path) -> None:
    """Validating authored YAML must not initialize optional native dependencies."""
    script = textwrap.dedent("""
        import json
        import sys

        from generator.core.scenario_actors import prepare_scenario
        from visualizer.src.authoring.compiler import canonical_scenario_mapping

        assert callable(prepare_scenario)
        assert callable(canonical_scenario_mapping)

        forbidden_roots = {
            "drjit",
            "geopandas",
            "mitsuba",
            "pandas",
            "pyarrow",
            "sionna",
        }
        loaded = sorted(
            name
            for name in sys.modules
            if name.partition(".")[0] in forbidden_roots
        )
        print(json.dumps(loaded))
        """)
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    env["QT_QPA_PLATFORM"] = "offscreen"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout.splitlines()[-1]) == []


def test_visualizer_startup_logs_do_not_claim_stale_path_policy_import() -> None:
    """Keep startup logging aligned with actual adjacent imports."""
    source = Path("visualizer/visualizer.py").read_text(encoding="utf-8")

    assert "Imported path policy module" not in source
