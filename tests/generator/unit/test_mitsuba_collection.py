"""Runtime test imports must not replace the backend owned by the caller."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.optional_runtime
def test_mesh_test_collection_preserves_selected_mitsuba_backend():
    mi = pytest.importorskip("mitsuba")
    variant = mi.variant() or os.environ.get("ORCHAV_NATIVE_RCS_VARIANT", "llvm_ad_mono_polarized")
    root = Path(__file__).resolve().parents[3]
    code = """
import runpy
import sys
import mitsuba as mi

variant, root = sys.argv[1:]
mi.set_variant(variant)
for module in ("test_mesh_update_timing.py", "test_target_mesh_sequence.py"):
    runpy.run_path(f"{root}/tests/generator/unit/{module}")
    assert mi.variant() == variant, (module, variant, mi.variant())
"""
    subprocess.run(
        [sys.executable, "-c", code, variant, str(root)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
