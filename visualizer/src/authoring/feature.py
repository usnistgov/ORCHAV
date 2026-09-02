"""Feature policy for the Scenario Builder workspace."""

from __future__ import annotations

import os
from collections.abc import Mapping

SCENARIO_BUILDER_ENV = "ORCHAV_ENABLE_SCENARIO_BUILDER"


def scenario_builder_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether the Scenario Builder workspace is explicitly enabled.

    The launch contract requires the exact value ``1`` so loosely interpreted
    environment values cannot enable the workspace.
    """

    source = os.environ if environ is None else environ
    return source.get(SCENARIO_BUILDER_ENV, "") == "1"
