#!/usr/bin/env bash
# Validate an ORCHAV checkout with the checks expected for release review. This
# is intentionally broader than a quick import smoke, but still
# avoids display/GPU-heavy renderer benchmarks unless explicitly requested.
set -euo pipefail

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

RUN_HELLO_WORLD="${RUN_HELLO_WORLD:-1}"
RUN_VISUALIZER_BENCHMARK="${RUN_VISUALIZER_BENCHMARK:-0}"
RUN_SCENARIO_BUILDER_DISPLAY_SMOKE="${RUN_SCENARIO_BUILDER_DISPLAY_SMOKE:-0}"
PRESERVE_SMOKE_OUTPUT="${PRESERVE_SMOKE_OUTPUT:-$RUN_SCENARIO_BUILDER_DISPLAY_SMOKE}"
AUTHORING_CAPTURE_OUTPUT=""
if ! PYTHON_BIN="$(command -v python)"; then
    echo "ERROR: python is required for release smoke." >&2
    exit 2
fi

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if ! REPO_ROOT="$(git -C "$SCRIPT_ROOT" rev-parse --show-toplevel 2>/dev/null)"; then
    echo "ERROR: release_smoke.sh must run from a Git checkout." >&2
    exit 2
fi
REPO_ROOT_PHYSICAL="$(cd "$REPO_ROOT" && pwd -P)"
cd "$REPO_ROOT_PHYSICAL"

BASELINE_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
if [[ -n "$BASELINE_STATUS" ]]; then
    echo "ERROR: release smoke requires a clean tracked/untracked worktree." >&2
    echo "$BASELINE_STATUS" >&2
    exit 2
fi

# ORCHAV_TMP_DIR names a pre-existing scratch parent, never a cleanup target.
# The default is a sibling of the checkout so generated data cannot land in
# the source tree. Each invocation owns only its atomically created child.
SMOKE_PARENT_INPUT="${ORCHAV_TMP_DIR:-$(dirname "$REPO_ROOT_PHYSICAL")}"
if [[ ! -d "$SMOKE_PARENT_INPUT" ]]; then
    echo "ERROR: scratch parent does not exist: $SMOKE_PARENT_INPUT" >&2
    exit 2
fi
if [[ -L "$SMOKE_PARENT_INPUT" ]]; then
    echo "ERROR: scratch parent may not be a symlink or junction: $SMOKE_PARENT_INPUT" >&2
    exit 2
fi
if TMPDIR="$SMOKE_PARENT_INPUT" \
    TMP="$SMOKE_PARENT_INPUT" \
    TEMP="$SMOKE_PARENT_INPUT" \
    PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON_BIN" -I -B -c '
from pathlib import Path
import sys

path = Path(sys.argv[1])
is_junction = getattr(path, "is_junction", lambda: False)
raise SystemExit(42 if path.is_symlink() or is_junction() else 0)
' "$SMOKE_PARENT_INPUT"; then
    :
else
    link_check_status=$?
    if [[ "$link_check_status" == "42" ]]; then
        echo "ERROR: scratch parent may not be a symlink or junction: $SMOKE_PARENT_INPUT" >&2
    else
        echo "ERROR: could not verify scratch-parent link safety: $SMOKE_PARENT_INPUT" >&2
    fi
    exit 2
fi
SMOKE_PARENT_LOGICAL="$(cd -L "$SMOKE_PARENT_INPUT" && pwd -L)"
SMOKE_PARENT_PHYSICAL="$(cd -P "$SMOKE_PARENT_INPUT" && pwd -P)"
if [[ "$SMOKE_PARENT_LOGICAL" != "$SMOKE_PARENT_PHYSICAL" ]]; then
    echo "ERROR: scratch parent may not be a symlink or junction: $SMOKE_PARENT_INPUT" >&2
    exit 2
fi
case "$SMOKE_PARENT_PHYSICAL/" in
    "$REPO_ROOT_PHYSICAL/"*)
        echo "ERROR: scratch parent must be outside the checkout: $SMOKE_PARENT_PHYSICAL" >&2
        exit 2
        ;;
esac
if SCRATCH_GIT_ROOT="$(git -C "$SMOKE_PARENT_PHYSICAL" rev-parse --show-toplevel 2>/dev/null)"; then
    echo "ERROR: scratch parent may not be inside a Git checkout: $SCRATCH_GIT_ROOT" >&2
    exit 2
fi
if [[ ! -w "$SMOKE_PARENT_PHYSICAL" ]]; then
    echo "ERROR: scratch parent is not writable: $SMOKE_PARENT_PHYSICAL" >&2
    exit 2
fi

SMOKE_RUN_ROOT="$(
    mktemp -d "$SMOKE_PARENT_PHYSICAL/orchav-release-smoke.XXXXXX"
)"
SMOKE_RUN_ROOT_PHYSICAL="$(cd "$SMOKE_RUN_ROOT" && pwd -P)"
SMOKE_RUN_TOKEN="orchav-release-smoke:$SMOKE_RUN_ROOT_PHYSICAL:$$:$RANDOM"
SMOKE_OWNER_MARKER="$SMOKE_RUN_ROOT_PHYSICAL/.orchav-release-smoke-owner"
printf "%s\n" "$SMOKE_RUN_TOKEN" >"$SMOKE_OWNER_MARKER"

cleanup_smoke_run() {
    local exit_status=$?
    local cleanup_error=0
    local current_root=""
    local marker_value=""
    local root_kind=""
    local root_kind_status=0

    trap - EXIT HUP INT TERM
    if [[ "$PRESERVE_SMOKE_OUTPUT" == "1" ]]; then
        echo "Preserved smoke output: $SMOKE_RUN_ROOT_PHYSICAL"
        exit "$exit_status"
    fi

    # Bash's -e/-L checks do not classify every Windows reparse point,
    # especially a dangling junction. Native Python provides a fail-closed
    # four-way classification before any cleanup decision.
    if TMPDIR="$SMOKE_PARENT_PHYSICAL" \
        TMP="$SMOKE_PARENT_PHYSICAL" \
        TEMP="$SMOKE_PARENT_PHYSICAL" \
        PYTHONDONTWRITEBYTECODE=1 \
        "$PYTHON_BIN" -I -B -c '
from pathlib import Path
import sys

path = Path(sys.argv[1])
is_junction = getattr(path, "is_junction", lambda: False)
if path.is_symlink() or is_junction():
    raise SystemExit(20)
if not path.exists():
    raise SystemExit(0)
raise SystemExit(10 if path.is_dir() else 30)
' "$SMOKE_RUN_ROOT_PHYSICAL"; then
        root_kind="absent"
    else
        root_kind_status=$?
        case "$root_kind_status" in
            10) root_kind="directory" ;;
            20) root_kind="link" ;;
            30) root_kind="other" ;;
            *)
                root_kind="error"
                cleanup_error=1
                ;;
        esac
    fi

    if [[ "$root_kind" != "absent" ]]; then
        if [[ "$root_kind" == "directory" ]] \
            && current_root="$(
                cd "$SMOKE_RUN_ROOT_PHYSICAL" 2>/dev/null && pwd -P
            )"; then
            marker_value="$(cat "$SMOKE_OWNER_MARKER" 2>/dev/null || true)"
            if [[ "$current_root" != "$SMOKE_RUN_ROOT_PHYSICAL" ]] \
                || [[ "$marker_value" != "$SMOKE_RUN_TOKEN" ]] \
                || [[ "$(basename "$current_root")" != orchav-release-smoke.* ]] \
                || [[ "$(dirname "$current_root")" != "$SMOKE_PARENT_PHYSICAL" ]]; then
                cleanup_error=1
            fi
            if [[ "$current_root" == "/" ]] \
                || [[ "$current_root" == "$REPO_ROOT_PHYSICAL" ]] \
                || [[ -n "${HOME:-}" && "$current_root" == "$HOME" ]]; then
                cleanup_error=1
            fi
            case "$current_root/" in
                "$REPO_ROOT_PHYSICAL/"*) cleanup_error=1 ;;
            esac
        else
            cleanup_error=1
        fi

        # Recheck the deletion target immediately before the only recursive
        # removal. Any link/junction or probe failure makes cleanup refuse.
        if ! TMPDIR="$SMOKE_PARENT_PHYSICAL" \
            TMP="$SMOKE_PARENT_PHYSICAL" \
            TEMP="$SMOKE_PARENT_PHYSICAL" \
            PYTHONDONTWRITEBYTECODE=1 \
            "$PYTHON_BIN" -I -B -c '
from pathlib import Path
import sys

path = Path(sys.argv[1])
is_junction = getattr(path, "is_junction", lambda: False)
raise SystemExit(1 if path.is_symlink() or is_junction() else 0)
' "$SMOKE_RUN_ROOT_PHYSICAL" "cleanup-final-link-check"; then
            cleanup_error=1
        fi

        if [[ "$cleanup_error" == "0" ]]; then
            rm -rf -- "$SMOKE_RUN_ROOT_PHYSICAL"
        else
            echo "ERROR: refusing to remove an unverified smoke path: $SMOKE_RUN_ROOT_PHYSICAL" >&2
            if [[ "$exit_status" == "0" ]]; then
                exit_status=1
            fi
        fi
    fi
    exit "$exit_status"
}
trap cleanup_smoke_run EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p \
    "$SMOKE_RUN_ROOT_PHYSICAL/tmp" \
    "$SMOKE_RUN_ROOT_PHYSICAL/wheels" \
    "$SMOKE_RUN_ROOT_PHYSICAL/cache/black" \
    "$SMOKE_RUN_ROOT_PHYSICAL/cache/ruff" \
    "$SMOKE_RUN_ROOT_PHYSICAL/cache/mypy" \
    "$SMOKE_RUN_ROOT_PHYSICAL/cache/pip" \
    "$SMOKE_RUN_ROOT_PHYSICAL/cache/xdg" \
    "$SMOKE_RUN_ROOT_PHYSICAL/cache/matplotlib" \
    "$SMOKE_RUN_ROOT_PHYSICAL/cache/numba" \
    "$SMOKE_RUN_ROOT_PHYSICAL/cache/orchav" \
    "$SMOKE_RUN_ROOT_PHYSICAL/logs" \
    "$SMOKE_RUN_ROOT_PHYSICAL/pytest"

export TMPDIR="$SMOKE_RUN_ROOT_PHYSICAL/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PYTHONPYCACHEPREFIX="$SMOKE_RUN_ROOT_PHYSICAL/cache/python"
export BLACK_CACHE_DIR="$SMOKE_RUN_ROOT_PHYSICAL/cache/black"
export RUFF_CACHE_DIR="$SMOKE_RUN_ROOT_PHYSICAL/cache/ruff"
export MYPY_CACHE_DIR="$SMOKE_RUN_ROOT_PHYSICAL/cache/mypy"
export PIP_CACHE_DIR="$SMOKE_RUN_ROOT_PHYSICAL/cache/pip"
export XDG_CACHE_HOME="$SMOKE_RUN_ROOT_PHYSICAL/cache/xdg"
export XDG_CONFIG_HOME="$SMOKE_RUN_ROOT_PHYSICAL/cache/xdg-config"
export MPLCONFIGDIR="$SMOKE_RUN_ROOT_PHYSICAL/cache/matplotlib"
export NUMBA_CACHE_DIR="$SMOKE_RUN_ROOT_PHYSICAL/cache/numba"
export ORCHAV_CACHE_DIR="$SMOKE_RUN_ROOT_PHYSICAL/cache/orchav"
export ORCHAV_SUMMARY_CACHE_DIR="$SMOKE_RUN_ROOT_PHYSICAL/cache/orchav/summary"
export ORCHAV_PYGFX_MESH_BUFFER_CACHE_DIR="$SMOKE_RUN_ROOT_PHYSICAL/cache/orchav/pygfx-mesh"
export ORCHAV_UV_CACHE_DIR="$SMOKE_RUN_ROOT_PHYSICAL/cache/orchav/uv"
export ORCHAV_SCENE_PAYLOAD_CACHE_DIR="$SMOKE_RUN_ROOT_PHYSICAL/cache/orchav/scene-payload"
export ORCHAV_LOG_FILE="$SMOKE_RUN_ROOT_PHYSICAL/logs/orchav.log"

WHEEL_DIR="$SMOKE_RUN_ROOT_PHYSICAL/wheels"
BENCHMARK_OUTPUT="$SMOKE_RUN_ROOT_PHYSICAL/orchav-release-ci-hello-world-benchmark.json"
export WHEEL_DIR BENCHMARK_OUTPUT

HELLO_WORLD_REL="scenarios/getting_started/hello_world"
SMOKE_SCENARIO="$SMOKE_RUN_ROOT_PHYSICAL/source/$HELLO_WORLD_REL"
mkdir -p "$SMOKE_SCENARIO"
SCENARIO_MODE="$(
    git ls-tree HEAD -- "$HELLO_WORLD_REL/scenario.yaml" | awk '{print $1}'
)"
if [[ "$SCENARIO_MODE" != "100644" ]] && [[ "$SCENARIO_MODE" != "100755" ]]; then
    echo "ERROR: committed hello_world/scenario.yaml is missing or not a regular file." >&2
    exit 2
fi
git show "HEAD:$HELLO_WORLD_REL/scenario.yaml" >"$SMOKE_SCENARIO/scenario.yaml"

PACKAGE_SOURCE="$SMOKE_RUN_ROOT_PHYSICAL/package-source"
mkdir -p "$PACKAGE_SOURCE"
PACKAGE_ARCHIVE_PATHS=(
    pyproject.toml
    README.md
    LICENSE
    NOTICE
    NIST_SOFTWARE_DISCLAIMER.md
    THIRD_PARTY_NOTICES.md
    generator
    shared
    visualizer
)
git archive --format=tar HEAD -- "${PACKAGE_ARCHIVE_PATHS[@]}" \
    | tar -xf - -C "$PACKAGE_SOURCE"

"$PYTHON_BIN" -I -B - "$SMOKE_RUN_ROOT_PHYSICAL" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
for name in (
    "TMPDIR",
    "TMP",
    "TEMP",
    "PYTHONPYCACHEPREFIX",
    "BLACK_CACHE_DIR",
    "RUFF_CACHE_DIR",
    "MYPY_CACHE_DIR",
    "PIP_CACHE_DIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "MPLCONFIGDIR",
    "NUMBA_CACHE_DIR",
    "ORCHAV_CACHE_DIR",
    "ORCHAV_SUMMARY_CACHE_DIR",
    "ORCHAV_PYGFX_MESH_BUFFER_CACHE_DIR",
    "ORCHAV_UV_CACHE_DIR",
    "ORCHAV_SCENE_PAYLOAD_CACHE_DIR",
    "ORCHAV_LOG_FILE",
    "WHEEL_DIR",
    "BENCHMARK_OUTPUT",
):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set")
    candidate = Path(value).resolve()
    if candidate != root and root not in candidate.parents:
        raise SystemExit(f"{name} escapes the smoke run root: {candidate}")
PY

announce_step() {
    echo ""
    echo "==> $1"
}

run_step() {
    local title="$1"
    shift
    announce_step "$title"
    "$@"
    echo "PASS: $title"
}

run_step_to_file() {
    local title="$1"
    local output_path="$2"
    shift 2
    announce_step "$title"
    echo "Output saved to: $output_path"
    "$@" >"$output_path"
    echo "PASS: $title"
}

# Release smoke is the reviewer-facing contract. Renderer-internal
# architecture checks are maintained separately because they verify backend
# implementation boundaries rather than generator readiness.
PYTEST_TARGETS=(
    tests/test_path_policy.py
    tests/integration/test_external_raytracer_import.py
    tests/shared
    tests/statistics_shared
    tests/generator
    tests/visualizer/authoring
    tests/visualizer/unit/test_authoring_generation.py
    tests/visualizer/unit/test_capture_scenario_builder_workflow.py
    tests/visualizer/unit/test_mobility_editor.py
    tests/visualizer/unit/test_orientation_editor.py
    tests/visualizer/unit/test_scenario_authoring_entry.py
    tests/visualizer/unit/test_scenario_authoring_generation_controller.py
    tests/visualizer/unit/test_scenario_authoring_viewport.py
    tests/visualizer/unit/test_scenario_authoring_workspace.py
    tests/visualizer/unit/test_startup_workflow.py
    tests/visualizer/unit/test_scenario_workflow.py
    tests/visualizer/unit/test_scenario_loader_service.py
    tests/visualizer/unit/test_session_service.py
    tests/visualizer/unit/test_window_manager.py
    tests/visualizer/unit/test_viewport_workspace.py
    tests/visualizer/unit/test_frame_retry_policy.py
    tests/visualizer/unit/test_renderer_protocol.py
    tests/visualizer/unit/test_pygfx_renderer_helpers.py
    tests/visualizer/unit/test_frame_types.py
    tests/visualizer/unit/test_visualizer_imports.py
)

echo "ORCHAV release smoke"
echo "Repo: $REPO_ROOT_PHYSICAL"
echo "Scratch run root: $SMOKE_RUN_ROOT_PHYSICAL"
echo "Qt platform: $QT_QPA_PLATFORM"
echo "Display: ${DISPLAY:-<unset>}"
echo ""

run_step "Release smoke safety contract: owned scratch and sentinel preservation" \
    bash tests/ci/test_release_smoke_safety.sh

run_step "Black formatting check: generator, shared, scripts, visualizer, and tests" \
    python scripts/ci/black_check.py --check --workers 1 generator shared scripts visualizer tests

run_step "Ruff lint check: generator, shared, scripts, visualizer, and tests" \
    python -m ruff check generator shared scripts visualizer tests

run_step "Public documentation and Mermaid conventions" \
    python scripts/ci/check_documentation.py

run_step "Committed-tree archive scenario inventory" \
    python scripts/ci/check_git_archive.py

run_step "MyPy typed-surface check: staged ORCHAV-owned typed modules" \
    bash scripts/ci/mypy_typed_surface.sh

run_step "Python bytecode compile check: generator, shared, scripts, visualizer, and tests" \
    python -m compileall -q generator shared scripts visualizer tests

run_step_to_file "Public test collection: all retained tests import cleanly" \
    "$SMOKE_RUN_ROOT_PHYSICAL/pytest-collection.txt" \
    python -m pytest --collect-only -q tests

run_step "Wheel build check: package metadata and build backend" \
    python -m pip wheel --no-build-isolation --no-deps "$PACKAGE_SOURCE" -w "$WHEEL_DIR"

run_step "Import check: generator, shared, and visualizer packages" \
    python -c "import generator, shared, visualizer; print('ORCHAV package imports OK')"

run_step_to_file "Generator CLI help check: verify python -m generator --help exits successfully" \
    "$SMOKE_RUN_ROOT_PHYSICAL/orchav-generator-help.txt" \
    python -m generator --help

run_step_to_file "Generator scenario listing check: verify python -m generator exits successfully" \
    "$SMOKE_RUN_ROOT_PHYSICAL/orchav-generator-catalog.txt" \
    python -m generator

run_step_to_file "Shared inspect CLI help check: verify python -m shared.cli.inspect --help exits successfully" \
    "$SMOKE_RUN_ROOT_PHYSICAL/orchav-inspect-help.txt" \
    python -m shared.cli.inspect --help

run_step_to_file "Visualizer CLI help check: verify python -m visualizer --help exits successfully" \
    "$SMOKE_RUN_ROOT_PHYSICAL/orchav-visualizer-help.txt" \
    python -m visualizer --help

announce_step "Scenario validation check: getting-started scenarios"
printf "Scenarios:\n"
printf "  - %s\n" \
  scenarios/getting_started/hello_world \
  scenarios/getting_started/hello_world_scripted
python -m shared.cli.validate \
  scenarios/getting_started/hello_world \
  scenarios/getting_started/hello_world_scripted
echo "PASS: Scenario validation check"

if [[ "${ORCHAV_RUN_OPTIONAL_TESTS:-0}" != "1" ]]; then
    echo "Optional runtime/socket/private/soak tests are deselected by default."
fi

announce_step "Pytest check: generator tests plus focused shared/visualizer tests"
printf "Targets:\n"
printf "  - %s\n" "${PYTEST_TARGETS[@]}"
python -m pytest --no-cov -q \
    -o "cache_dir=$SMOKE_RUN_ROOT_PHYSICAL/pytest/cache" \
    --basetemp "$SMOKE_RUN_ROOT_PHYSICAL/pytest/tmp" \
    "${PYTEST_TARGETS[@]}"
echo "PASS: Pytest check"

if [[ "$RUN_HELLO_WORLD" == "1" ]]; then
    announce_step "Generator smoke run: getting-started hello_world scenario"
    python -m generator "$SMOKE_SCENARIO/"
    echo "PASS: Generator smoke run"
else
    announce_step "Generator smoke run skipped: RUN_HELLO_WORLD=$RUN_HELLO_WORLD"
fi

if [[ "$RUN_VISUALIZER_BENCHMARK" == "1" ]]; then
    announce_step "Optional visualizer benchmark: pygfx renderer, one benchmark frame"
    python -m visualizer \
      --scenario "$SMOKE_SCENARIO" \
      --renderer pygfx \
      --benchmark 1 \
      --viewport-mode embedded \
      --benchmark-warmup 0 \
      --benchmark-output "$BENCHMARK_OUTPUT" \
      --no-resume
    echo "PASS: Optional visualizer benchmark"
else
    announce_step "Optional visualizer benchmark skipped: RUN_VISUALIZER_BENCHMARK=$RUN_VISUALIZER_BENCHMARK"
fi

if [[ "$RUN_SCENARIO_BUILDER_DISPLAY_SMOKE" == "1" ]]; then
    announce_step "Optional Scenario Builder display workflow: author, save, generate, preview, and capture"
    AUTHORING_CAPTURE_OUTPUT="$(
        mktemp -d "$SMOKE_RUN_ROOT_PHYSICAL/orchav-authoring-display.XXXXXX"
    )"
    QT_QPA_PLATFORM="${ORCHAV_AUTHORING_QPA_PLATFORM:-xcb}" \
        ORCHAV_ENABLE_SCENARIO_BUILDER=1 \
        python scripts/ci/run_with_timeout.py \
        --timeout-s "${ORCHAV_AUTHORING_PROCESS_TIMEOUT_S:-420}" \
        -- python scripts/capture_scenario_builder_workflow.py \
        --output-dir "$AUTHORING_CAPTURE_OUTPUT" \
        --timeout-s "${ORCHAV_AUTHORING_STAGE_TIMEOUT_S:-300}"
    echo "Capture evidence: $AUTHORING_CAPTURE_OUTPUT"
    echo "PASS: Optional Scenario Builder display workflow"
else
    announce_step "Optional Scenario Builder display workflow skipped: RUN_SCENARIO_BUILDER_DISPLAY_SMOKE=$RUN_SCENARIO_BUILDER_DISPLAY_SMOKE"
fi

announce_step "Final worktree check: tracked and untracked status is unchanged"
FINAL_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
if [[ "$FINAL_STATUS" != "$BASELINE_STATUS" ]]; then
    echo "Smoke changed the checkout:" >&2
    echo "Before:" >&2
    printf "%s\n" "$BASELINE_STATUS" >&2
    echo "After:" >&2
    printf "%s\n" "$FINAL_STATUS" >&2
    exit 1
fi
echo "PASS: Final worktree check"

echo ""
echo "ORCHAV release smoke passed"
