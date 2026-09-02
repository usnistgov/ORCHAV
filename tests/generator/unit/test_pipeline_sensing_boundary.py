from __future__ import annotations

import subprocess
import sys


def test_offline_pipeline_import_does_not_load_sensing_internals() -> None:
    code = """
import sys
import generator.core.pipeline.offline_pipeline  # noqa: F401
for name in (
    "generator.core.sensing.processor",
    "generator.core.sensing.hooks",
    "generator.core.sensing.timing",
):
    if name in sys.modules:
        raise SystemExit(f"{name} loaded during pipeline import")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
