#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TYPED_SURFACE=(
  shared/frames/types.py
  shared/frames/directory_ownership.py
  shared/scenarios/paths.py
  shared/scenarios/frame_paths.py
  shared/logging/config.py
  shared/logging/filters.py
  shared/logging/formatters.py
  shared/frames/schema.py
  shared/scenarios/actors.py
  shared/scenarios/model.py
  generator/core/configuration/presets.py
  generator/core/exceptions.py
  generator/core/orientation/adapters.py
  generator/core/orientation/base.py
  generator/core/scenario_actors/__init__.py
  generator/core/scenario_actors/_adapters.py
  generator/core/scenario_actors/errors.py
  generator/core/scenario_actors/mobility.py
  generator/core/scenario_actors/orientation.py
  generator/core/scenario_actors/preparation.py
  generator/core/scenario_actors/quaternion.py
  generator/core/scenario_actors/resources.py
  generator/core/scenario_actors/runtime.py
  generator/core/scenario_actors/state.py
  generator/core/scenario_actors/types.py
  generator/core/utils/angle_utils.py
  generator/core/utils/tensor_utils.py
  generator/io/storage/hdf5_frame_output.py
  visualizer/src/authoring/assets.py
  visualizer/src/authoring/domain.py
  visualizer/src/authoring/mobility_models.py
  visualizer/src/authoring/mobility_control_rig.py
  visualizer/src/authoring/document.py
  visualizer/src/authoring/undo.py
  visualizer/src/authoring/compiler.py
  visualizer/src/authoring/compilation_scheduler.py
  visualizer/src/authoring/persistence.py
  visualizer/src/authoring/generation.py
  visualizer/src/authoring/mobility_editor.py
  visualizer/src/authoring/model_capabilities.py
  visualizer/src/authoring/orientation_editor.py
  visualizer/src/authoring/orientation_models.py
  visualizer/src/authoring/viewport_port.py
)

python -m mypy --follow-imports=skip --no-warn-unused-configs "${TYPED_SURFACE[@]}"
