#!/usr/bin/env bash
# Exercise release_smoke.sh with a real disposable Git clone and fake, fast
# tool commands. This proves path ownership independently of the expensive
# release checks that the production script normally runs.
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
CANDIDATE="$SOURCE_ROOT/scripts/ci/release_smoke.sh"
REAL_BASH="$BASH"
REAL_PYTHON="$(command -v python || true)"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

create_directory_link() {
    local link_path="$1"
    local target_path="$2"
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            local link_windows
            local target_windows
            link_windows="$(cygpath -w "$link_path")"
            target_windows="$(cygpath -w "$target_path")"
            MSYS2_ARG_CONV_EXCL='*' \
                ORCHAV_LINK_PATH="$link_windows" \
                ORCHAV_LINK_TARGET="$target_windows" \
                powershell.exe \
                -NoProfile \
                -NonInteractive \
                -Command \
                'New-Item -ItemType Junction -Path $env:ORCHAV_LINK_PATH -Target $env:ORCHAV_LINK_TARGET -ErrorAction Stop | Out-Null' \
                >/dev/null 2>&1
            ;;
        *)
            ln -s "$target_path" "$link_path"
            ;;
    esac
}

remove_directory_link() {
    local link_path="$1"
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            local link_windows
            link_windows="$(cygpath -w "$link_path")"
            MSYS2_ARG_CONV_EXCL='*' \
                ORCHAV_LINK_PATH="$link_windows" \
                powershell.exe \
                -NoProfile \
                -NonInteractive \
                -Command \
                '[System.IO.Directory]::Delete($env:ORCHAV_LINK_PATH)' \
                >/dev/null 2>&1
            ;;
        *)
            rm -- "$link_path"
            ;;
    esac
}

"$REAL_BASH" -n "$CANDIDATE"
[[ -n "$REAL_PYTHON" ]] || fail "python is required for junction-path checks"
if grep -Eq 'git[[:space:]]+clean' "$CANDIDATE"; then
    fail "release smoke still invokes repository-wide Git cleanup"
fi
if [[ "$(grep -Fc 'rm -rf -- "$SMOKE_RUN_ROOT_PHYSICAL"' "$CANDIDATE")" != "1" ]]; then
    fail "release smoke must have exactly one owned-root recursive removal"
fi
grep -F 'is_junction' "$CANDIDATE" >/dev/null \
    || fail "release smoke lacks a Windows junction guard"

TEST_PARENT="${TMPDIR:-$(dirname "$SOURCE_ROOT")}"
TEST_PARENT_PHYSICAL="$(cd "$TEST_PARENT" && pwd -P)"
TEST_ROOT="$(
    mktemp -d "$TEST_PARENT_PHYSICAL/orchav release smoke safety.XXXXXX"
)"
TEST_ROOT_PHYSICAL="$(cd "$TEST_ROOT" && pwd -P)"
TEST_OWNER_TOKEN="orchav-release-smoke-safety:$TEST_ROOT_PHYSICAL:$$:$RANDOM"
TEST_OWNER_MARKER="$TEST_ROOT_PHYSICAL/.orchav-release-smoke-safety-owner"
printf "%s\n" "$TEST_OWNER_TOKEN" >"$TEST_OWNER_MARKER"
cleanup_test_root() {
    local exit_status=$?
    local current_root=""
    local marker_value=""
    local unsafe=0

    trap - EXIT HUP INT TERM
    if [[ -e "$TEST_ROOT_PHYSICAL" || -L "$TEST_ROOT_PHYSICAL" ]]; then
        if [[ -L "$TEST_ROOT_PHYSICAL" ]]; then
            unsafe=1
        elif current_root="$(cd "$TEST_ROOT_PHYSICAL" 2>/dev/null && pwd -P)"; then
            marker_value="$(cat "$TEST_OWNER_MARKER" 2>/dev/null || true)"
            if [[ "$current_root" != "$TEST_ROOT_PHYSICAL" ]] \
                || [[ "$(dirname "$current_root")" != "$TEST_PARENT_PHYSICAL" ]] \
                || [[ "$marker_value" != "$TEST_OWNER_TOKEN" ]]; then
                unsafe=1
            fi
            case "$(basename "$current_root")" in
                "orchav release smoke safety."*) ;;
                *) unsafe=1 ;;
            esac
            if ! "$REAL_PYTHON" -I -B -c '
from pathlib import Path
import sys

path = Path(sys.argv[1])
is_junction = getattr(path, "is_junction", lambda: False)
raise SystemExit(1 if path.is_symlink() or is_junction() else 0)
' "$TEST_ROOT_PHYSICAL"; then
                unsafe=1
            fi
        else
            unsafe=1
        fi

        if [[ "$unsafe" == "0" ]]; then
            rm -rf -- "$TEST_ROOT_PHYSICAL"
        else
            echo "ERROR: refusing to remove unverified test root: $TEST_ROOT_PHYSICAL" >&2
            exit_status=1
        fi
    fi
    exit "$exit_status"
}
trap cleanup_test_root EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

SOURCE_REPO="$TEST_ROOT/source repo"
CLONE="$TEST_ROOT/disposable clone"
SCRATCH_PARENT="$TEST_ROOT/shared scratch parent"
ESCAPE_ROOT="$TEST_ROOT/must not be touched"
FAKE_BIN="$TEST_ROOT/fake bin"
LOG_ROOT="$TEST_ROOT/logs"
mkdir -p \
    "$SOURCE_REPO/scripts/ci" \
    "$SOURCE_REPO/scenarios/getting_started/hello_world" \
    "$SOURCE_REPO/generator" \
    "$SOURCE_REPO/shared" \
    "$SOURCE_REPO/visualizer" \
    "$SCRATCH_PARENT/orchav-release-smoke.preexisting" \
    "$ESCAPE_ROOT/tmp" \
    "$ESCAPE_ROOT/wheels" \
    "$FAKE_BIN" \
    "$LOG_ROOT" \
    "$TEST_ROOT/no-hooks"

cp "$CANDIDATE" "$SOURCE_REPO/scripts/ci/release_smoke.sh"
printf "schema_version: 2\n" \
    >"$SOURCE_REPO/scenarios/getting_started/hello_world/scenario.yaml"
printf "# Disposable ORCHAV package\n" >"$SOURCE_REPO/README.md"
printf "test license\n" >"$SOURCE_REPO/LICENSE"
printf "test notice\n" >"$SOURCE_REPO/NOTICE"
printf "test disclaimer\n" >"$SOURCE_REPO/NIST_SOFTWARE_DISCLAIMER.md"
printf "test third-party notices\n" >"$SOURCE_REPO/THIRD_PARTY_NOTICES.md"
cat >"$SOURCE_REPO/pyproject.toml" <<'EOF'
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"
EOF
printf "\n" >"$SOURCE_REPO/generator/__init__.py"
printf "\n" >"$SOURCE_REPO/shared/__init__.py"
printf "\n" >"$SOURCE_REPO/visualizer/__init__.py"
cat >"$SOURCE_REPO/.gitignore" <<'EOF'
.venv/
.smoke-parent/
build/
*.egg-info/
scenarios/**/frames/
scenarios/**/summary/
EOF
cat >"$SOURCE_REPO/.gitattributes" <<'EOF'
*.sh text eol=lf
EOF

git -c init.templateDir= -C "$SOURCE_REPO" init -q
git -C "$SOURCE_REPO" add .
git -C "$SOURCE_REPO" \
    -c user.name="ORCHAV smoke test" \
    -c user.email="smoke-test@example.invalid" \
    -c commit.gpgSign=false \
    -c core.hooksPath="$TEST_ROOT/no-hooks" \
    commit -q --no-verify -m "Disposable release smoke fixture"
git clone -q "$SOURCE_REPO" "$CLONE"

mkdir -p \
    "$CLONE/.venv" \
    "$CLONE/.smoke-parent" \
    "$CLONE/build" \
    "$CLONE/orchav.egg-info" \
    "$CLONE/scenarios/getting_started/hello_world/frames" \
    "$CLONE/scenarios/getting_started/hello_world/summary"
printf "environment sentinel\n" >"$CLONE/.venv/KEEP_ME.txt"
printf "build sentinel\n" >"$CLONE/build/KEEP_ME.txt"
printf "egg-info sentinel\n" >"$CLONE/orchav.egg-info/KEEP_ME.txt"
printf "frame sentinel\n" \
    >"$CLONE/scenarios/getting_started/hello_world/frames/KEEP_ME.txt"
printf "summary sentinel\n" \
    >"$CLONE/scenarios/getting_started/hello_world/summary/KEEP_ME.txt"
printf "scratch parent sentinel\n" >"$SCRATCH_PARENT/KEEP_ME.txt"
printf "pre-existing sibling sentinel\n" \
    >"$SCRATCH_PARENT/orchav-release-smoke.preexisting/KEEP_ME.txt"
printf "ambient tmp sentinel\n" >"$ESCAPE_ROOT/tmp/KEEP_ME.txt"
printf "ambient wheel sentinel\n" >"$ESCAPE_ROOT/wheels/KEEP_ME.txt"
printf "ambient benchmark sentinel\n" >"$ESCAPE_ROOT/benchmark.json"
mkdir -p \
    "$ESCAPE_ROOT/orchav-cache" \
    "$ESCAPE_ROOT/summary-cache" \
    "$ESCAPE_ROOT/mesh-cache" \
    "$ESCAPE_ROOT/uv-cache" \
    "$ESCAPE_ROOT/scene-payload-cache" \
    "$ESCAPE_ROOT/junction-target" \
    "$ESCAPE_ROOT/dangling-junction-target"
printf "ambient ORCHAV log sentinel\n" >"$ESCAPE_ROOT/orchav.log"
printf "junction target sentinel\n" >"$ESCAPE_ROOT/junction-target/KEEP_ME.txt"

cat >"$FAKE_BIN/python" <<'EOF'
#!/bin/bash
set -u
joined=" $* "
if [[ "$joined" == *"is_junction"* ]]; then
    {
        if [[ "$joined" == *"cleanup-final-link-check"* ]]; then
            printf "FINAL_LINK_CHECK"
        else
            printf "PATH_LINK_CHECK"
        fi
        for arg in "$@"; do
            printf "\tARG=%s" "$arg"
        done
        printf "\n"
    } >>"$FAKE_TOOL_LOG"
    exec "$REAL_LINK_CHECK_PYTHON" "$@"
fi
if [[ "${1:-}" == "-I" && "${2:-}" == "-B" && "${3:-}" == "-" ]]; then
    exec "$REAL_LINK_CHECK_PYTHON" "$@"
fi
{
    printf "TMPDIR=%s" "${TMPDIR:-}"
    for name in \
        ORCHAV_CACHE_DIR \
        ORCHAV_SUMMARY_CACHE_DIR \
        ORCHAV_PYGFX_MESH_BUFFER_CACHE_DIR \
        ORCHAV_UV_CACHE_DIR \
        ORCHAV_SCENE_PAYLOAD_CACHE_DIR \
        ORCHAV_LOG_FILE; do
        printf "\t%s=%s" "$name" "${!name:-}"
    done
    for arg in "$@"; do
        printf "\tARG=%s" "$arg"
    done
    printf "\n"
} >>"$FAKE_TOOL_LOG"

if [[ -n "${FAKE_PYTHON_DELAY:-}" ]]; then
    sleep "$FAKE_PYTHON_DELAY"
fi
if [[ -n "${FAKE_PYTHON_TAMPER_MODE:-}" ]] \
    && [[ "$joined" == *" -m generator "* ]] \
    && [[ "$joined" == *"/source/scenarios/getting_started/hello_world/"* ]]; then
    run_root="${TMPDIR%/tmp}"
    case "$FAKE_PYTHON_TAMPER_MODE" in
        marker_missing)
            rm -f -- "$run_root/.orchav-release-smoke-owner"
            ;;
        marker_mismatch)
            printf "wrong owner\n" >"$run_root/.orchav-release-smoke-owner"
            ;;
        root_link|root_dangling|root_backup_collision)
            "$FAKE_TAMPER_HELPER" \
                "$run_root" \
                "$FAKE_TAMPER_TARGET" \
                "$FAKE_PYTHON_TAMPER_MODE" \
                || exit 94
            ;;
        *)
            exit 92
            ;;
    esac
fi
if [[ -n "${FAKE_PYTHON_SIGNAL_PARENT:-}" ]] \
    && [[ "$joined" == *" -m generator "* ]] \
    && [[ "$joined" == *"/source/scenarios/getting_started/hello_world/"* ]]; then
    case "$FAKE_PYTHON_SIGNAL_PARENT" in
        HUP|INT|TERM) ;;
        *) exit 93 ;;
    esac
    kill "-${FAKE_PYTHON_SIGNAL_PARENT}" "$PPID" || exit 95
    # If the candidate lacks the signal trap, this child returns success. The
    # expected 129/130/143 status must therefore come from the candidate.
    sleep 0.2
    exit 0
fi
if [[ -n "${FAKE_PYTHON_BARRIER_DIR:-}" ]] \
    && [[ "$joined" == *" -m generator "* ]] \
    && [[ "$joined" == *"/source/scenarios/getting_started/hello_world/"* ]]; then
    printf "ready\n" \
        >"$FAKE_PYTHON_BARRIER_DIR/$FAKE_PYTHON_BARRIER_ID.ready"
    for _ in {1..200}; do
        [[ -f "$FAKE_PYTHON_BARRIER_DIR/release" ]] && exit 0
        sleep 0.02
    done
    exit 96
fi
if [[ -n "${FAKE_PYTHON_FAIL_MATCH:-}" ]] \
    && [[ "$joined" == *"$FAKE_PYTHON_FAIL_MATCH"* ]]; then
    exit 41
fi
exit 0
EOF
cat >"$FAKE_BIN/bash" <<'EOF'
#!/bin/bash
set -u
printf "CHILD_BASH" >>"$FAKE_TOOL_LOG"
for arg in "$@"; do
    printf "\tARG=%s" "$arg" >>"$FAKE_TOOL_LOG"
done
printf "\n" >>"$FAKE_TOOL_LOG"
exit 0
EOF
cat >"$FAKE_BIN/tamper-root" <<'EOF'
#!/bin/bash
set -euo pipefail

link_path="$1"
target_path="$2"
tamper_mode="${3:-root_link}"
original_path="$link_path.original"
if [[ "$tamper_mode" == "root_backup_collision" ]]; then
    mkdir "$original_path"
    printf "backup collision sentinel\n" >"$original_path/KEEP_ME.txt"
fi
if [[ -e "$original_path" || -L "$original_path" ]]; then
    echo "refusing to replace existing backup path: $original_path" >&2
    exit 97
fi
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        link_windows="$(cygpath -w "$link_path")"
        original_windows="$(cygpath -w "$original_path")"
        target_windows="$(cygpath -w "$target_path")"
        MSYS2_ARG_CONV_EXCL='*' \
            ORCHAV_LINK_PATH="$link_windows" \
            ORCHAV_LINK_ORIGINAL="$original_windows" \
            ORCHAV_LINK_TARGET="$target_windows" \
            ORCHAV_TAMPER_MODE="$tamper_mode" \
            powershell.exe \
            -NoProfile \
            -NonInteractive \
            -Command \
            'Move-Item -LiteralPath $env:ORCHAV_LINK_PATH -Destination $env:ORCHAV_LINK_ORIGINAL -ErrorAction Stop; New-Item -ItemType Junction -Path $env:ORCHAV_LINK_PATH -Target $env:ORCHAV_LINK_TARGET -ErrorAction Stop | Out-Null; if ($env:ORCHAV_TAMPER_MODE -eq "root_dangling") { [System.IO.Directory]::Delete($env:ORCHAV_LINK_TARGET) }' \
            >/dev/null
        ;;
    *)
        mv "$link_path" "$original_path"
        ln -s "$target_path" "$link_path"
        if [[ "$tamper_mode" == "root_dangling" ]]; then
            rmdir -- "$target_path"
        fi
        ;;
esac
EOF
chmod +x "$FAKE_BIN/python" "$FAKE_BIN/bash" "$FAKE_BIN/tamper-root"

assert_sentinels() {
    [[ "$(cat "$CLONE/.venv/KEEP_ME.txt")" == "environment sentinel" ]] \
        || fail "environment sentinel changed"
    [[ "$(cat "$CLONE/build/KEEP_ME.txt")" == "build sentinel" ]] \
        || fail "build sentinel changed"
    [[ "$(cat "$CLONE/orchav.egg-info/KEEP_ME.txt")" == "egg-info sentinel" ]] \
        || fail "egg-info sentinel changed"
    [[ "$(
        cat "$CLONE/scenarios/getting_started/hello_world/frames/KEEP_ME.txt"
    )" == "frame sentinel" ]] || fail "frame sentinel changed"
    [[ "$(
        cat "$CLONE/scenarios/getting_started/hello_world/summary/KEEP_ME.txt"
    )" == "summary sentinel" ]] || fail "summary sentinel changed"
    [[ "$(cat "$SCRATCH_PARENT/KEEP_ME.txt")" == "scratch parent sentinel" ]] \
        || fail "scratch-parent sentinel changed"
    [[ "$(
        cat "$SCRATCH_PARENT/orchav-release-smoke.preexisting/KEEP_ME.txt"
    )" == "pre-existing sibling sentinel" ]] \
        || fail "pre-existing scratch sibling changed"
    [[ "$(cat "$ESCAPE_ROOT/tmp/KEEP_ME.txt")" == "ambient tmp sentinel" ]] \
        || fail "ambient TMPDIR content changed"
    [[ "$(cat "$ESCAPE_ROOT/wheels/KEEP_ME.txt")" == "ambient wheel sentinel" ]] \
        || fail "ambient WHEEL_DIR content changed"
    [[ "$(cat "$ESCAPE_ROOT/benchmark.json")" == "ambient benchmark sentinel" ]] \
        || fail "ambient BENCHMARK_OUTPUT content changed"
    [[ "$(cat "$ESCAPE_ROOT/orchav.log")" == "ambient ORCHAV log sentinel" ]] \
        || fail "ambient ORCHAV_LOG_FILE content changed"
    [[ "$(
        cat "$ESCAPE_ROOT/junction-target/KEEP_ME.txt"
    )" == "junction target sentinel" ]] \
        || fail "junction target sentinel changed"
    [[ -z "$(git -C "$CLONE" status --porcelain=v1 --untracked-files=all)" ]] \
        || fail "disposable clone status changed"
}

extract_run_root() {
    local output_path="$1"
    sed -n 's/^Scratch run root: //p' "$output_path" | head -n 1
}

assert_owned_run() {
    local output_path="$1"
    local log_path="$2"
    local run_root
    run_root="$(extract_run_root "$output_path")"
    [[ -n "$run_root" ]] || fail "smoke did not report its run root"
    case "$run_root" in
        "$SCRATCH_PARENT"/orchav-release-smoke.*) ;;
        *) fail "run root escaped scratch parent: $run_root" ;;
    esac
    [[ ! -e "$run_root" ]] || fail "owned run root was not removed: $run_root"
    if ! grep -F "$run_root/source/scenarios/getting_started/hello_world/" \
        "$log_path" >/dev/null; then
        echo "Smoke output:" >&2
        cat "$output_path" >&2
        echo "Tool log:" >&2
        cat "$log_path" >&2
        fail "generator did not use the scratch scenario"
    fi
    grep -F "TMPDIR=$run_root/tmp" "$log_path" >/dev/null \
        || fail "ambient TMPDIR was not replaced"
    grep -F "$run_root/package-source" "$log_path" >/dev/null \
        || fail "wheel build did not use the scratch package source"
    if grep -F "$ESCAPE_ROOT" "$log_path" >/dev/null; then
        fail "an ambient output override escaped the owned run root"
    fi
    printf "%s\n" "$run_root"
}

assert_retained_refusal() {
    local output_path="$1"
    local run_root
    run_root="$(extract_run_root "$output_path")"
    case "$run_root" in
        "$SCRATCH_PARENT"/orchav-release-smoke.*) ;;
        *) fail "retained run root escaped scratch parent: $run_root" ;;
    esac
    if [[ ! -e "$run_root" && ! -L "$run_root" ]] \
        && ! "$REAL_PYTHON" -I -B -c '
from pathlib import Path
import sys

path = Path(sys.argv[1])
is_junction = getattr(path, "is_junction", lambda: False)
raise SystemExit(0 if path.is_symlink() or is_junction() else 1)
' "$run_root"; then
        fail "unsafe cleanup did not retain its run root"
    fi
    grep -F "refusing to remove an unverified smoke path" "$output_path" >/dev/null \
        || fail "unsafe cleanup did not report its refusal"
    printf "%s\n" "$run_root"
}

remove_retained_run_root() {
    local run_root="$1"
    case "$run_root" in
        "$SCRATCH_PARENT"/orchav-release-smoke.*) ;;
        *) fail "refusing test cleanup outside scratch parent: $run_root" ;;
    esac
    if ! "$REAL_PYTHON" -I -B -c '
from pathlib import Path
import sys

path = Path(sys.argv[1])
is_junction = getattr(path, "is_junction", lambda: False)
raise SystemExit(1 if path.is_symlink() or is_junction() else 0)
' "$run_root"; then
        fail "normal retained-root cleanup received a directory link"
    fi
    rm -rf -- "$run_root"
}

run_candidate() {
    local label="$1"
    local fail_match="${2:-}"
    local signal_parent="${3:-}"
    local delay="${4:-}"
    local preserve_output="${5:-0}"
    local display_smoke="${6:-0}"
    local tamper_mode="${7:-}"
    local tamper_target="${8:-}"
    local barrier_dir="${9:-}"
    local barrier_id="${10:-}"
    local output_path="$LOG_ROOT/$label.out"
    local log_path="$LOG_ROOT/$label.tools"

    (
        cd "$CLONE"
        PATH="$FAKE_BIN:$PATH" \
        FAKE_TOOL_LOG="$log_path" \
        REAL_LINK_CHECK_PYTHON="$REAL_PYTHON" \
        FAKE_PYTHON_FAIL_MATCH="$fail_match" \
        FAKE_PYTHON_SIGNAL_PARENT="$signal_parent" \
        FAKE_PYTHON_DELAY="$delay" \
        FAKE_PYTHON_TAMPER_MODE="$tamper_mode" \
        FAKE_TAMPER_HELPER="$FAKE_BIN/tamper-root" \
        FAKE_TAMPER_TARGET="$tamper_target" \
        FAKE_PYTHON_BARRIER_DIR="$barrier_dir" \
        FAKE_PYTHON_BARRIER_ID="$barrier_id" \
        ORCHAV_TMP_DIR="$SCRATCH_PARENT" \
        TMPDIR="$ESCAPE_ROOT/tmp" \
        WHEEL_DIR="$ESCAPE_ROOT/wheels" \
        BENCHMARK_OUTPUT="$ESCAPE_ROOT/benchmark.json" \
        ORCHAV_CACHE_DIR="$ESCAPE_ROOT/orchav-cache" \
        ORCHAV_SUMMARY_CACHE_DIR="$ESCAPE_ROOT/summary-cache" \
        ORCHAV_PYGFX_MESH_BUFFER_CACHE_DIR="$ESCAPE_ROOT/mesh-cache" \
        ORCHAV_UV_CACHE_DIR="$ESCAPE_ROOT/uv-cache" \
        ORCHAV_SCENE_PAYLOAD_CACHE_DIR="$ESCAPE_ROOT/scene-payload-cache" \
        ORCHAV_LOG_FILE="$ESCAPE_ROOT/orchav.log" \
        RUN_HELLO_WORLD=1 \
        RUN_VISUALIZER_BENCHMARK=1 \
        RUN_SCENARIO_BUILDER_DISPLAY_SMOKE="$display_smoke" \
        PRESERVE_SMOKE_OUTPUT="$preserve_output" \
        "$REAL_BASH" scripts/ci/release_smoke.sh
    ) >"$output_path" 2>&1
}

success_one_status=0
run_candidate success_one || success_one_status=$?
if [[ "$success_one_status" != "0" ]]; then
    cat "$LOG_ROOT/success_one.out" >&2
    fail "first successful smoke fixture returned $success_one_status"
fi
root_one="$(assert_owned_run "$LOG_ROOT/success_one.out" "$LOG_ROOT/success_one.tools")"
assert_sentinels

run_candidate success_two
root_two="$(assert_owned_run "$LOG_ROOT/success_two.out" "$LOG_ROOT/success_two.tools")"
[[ "$root_one" != "$root_two" ]] || fail "sequential runs reused a run root"
assert_sentinels

failure_status=0
run_candidate \
    injected_failure \
    "/source/scenarios/getting_started/hello_world/" \
    || failure_status=$?
[[ "$failure_status" == "41" ]] \
    || fail "injected generator failure returned $failure_status instead of 41"
assert_owned_run \
    "$LOG_ROOT/injected_failure.out" \
    "$LOG_ROOT/injected_failure.tools" >/dev/null
assert_sentinels

term_status=0
run_candidate injected_term "" TERM || term_status=$?
[[ "$term_status" == "143" ]] \
    || fail "injected TERM returned $term_status instead of 143"
assert_owned_run \
    "$LOG_ROOT/injected_term.out" \
    "$LOG_ROOT/injected_term.tools" >/dev/null
assert_sentinels

hup_status=0
run_candidate injected_hup "" HUP || hup_status=$?
[[ "$hup_status" == "129" ]] \
    || fail "injected HUP returned $hup_status instead of 129"
assert_owned_run \
    "$LOG_ROOT/injected_hup.out" \
    "$LOG_ROOT/injected_hup.tools" >/dev/null
assert_sentinels

int_status=0
run_candidate injected_int "" INT || int_status=$?
[[ "$int_status" == "130" ]] \
    || fail "injected INT returned $int_status instead of 130"
assert_owned_run \
    "$LOG_ROOT/injected_int.out" \
    "$LOG_ROOT/injected_int.tools" >/dev/null
assert_sentinels

CONCURRENCY_BARRIER="$TEST_ROOT/concurrency barrier"
mkdir -p "$CONCURRENCY_BARRIER"
run_candidate \
    concurrent_one "" "" "" 0 0 "" "" "$CONCURRENCY_BARRIER" one &
pid_one=$!
run_candidate \
    concurrent_two "" "" "" 0 0 "" "" "$CONCURRENCY_BARRIER" two &
pid_two=$!
for _ in {1..200}; do
    if [[ -f "$CONCURRENCY_BARRIER/one.ready" ]] \
        && [[ -f "$CONCURRENCY_BARRIER/two.ready" ]]; then
        break
    fi
    sleep 0.02
done
if [[ ! -f "$CONCURRENCY_BARRIER/one.ready" ]] \
    || [[ ! -f "$CONCURRENCY_BARRIER/two.ready" ]]; then
    touch "$CONCURRENCY_BARRIER/release"
    wait "$pid_one" || true
    wait "$pid_two" || true
    fail "concurrent smoke runs did not reach their shared barrier"
fi
concurrent_root_one="$(extract_run_root "$LOG_ROOT/concurrent_one.out")"
concurrent_root_two="$(extract_run_root "$LOG_ROOT/concurrent_two.out")"
[[ -n "$concurrent_root_one" && -n "$concurrent_root_two" ]] \
    || fail "concurrent smoke runs did not report both run roots"
[[ "$concurrent_root_one" != "$concurrent_root_two" ]] \
    || fail "concurrent runs reused a run root"
[[ -d "$concurrent_root_one" && -d "$concurrent_root_two" ]] \
    || fail "concurrent smoke run roots did not exist at the same time"
touch "$CONCURRENCY_BARRIER/release"
wait "$pid_one" || fail "first concurrent smoke failed"
wait "$pid_two" || fail "second concurrent smoke failed"
concurrent_root_one="$(
    assert_owned_run "$LOG_ROOT/concurrent_one.out" "$LOG_ROOT/concurrent_one.tools"
)"
concurrent_root_two="$(
    assert_owned_run "$LOG_ROOT/concurrent_two.out" "$LOG_ROOT/concurrent_two.tools"
)"
assert_sentinels

run_candidate preserved_capture "" "" "" 1 1
preserved_root="$(extract_run_root "$LOG_ROOT/preserved_capture.out")"
case "$preserved_root" in
    "$SCRATCH_PARENT"/orchav-release-smoke.*) ;;
    *) fail "preserved run root escaped scratch parent: $preserved_root" ;;
esac
[[ -f "$preserved_root/.orchav-release-smoke-owner" ]] \
    || fail "preserved run root lacks its ownership marker"
grep -F "Preserved smoke output: $preserved_root" \
    "$LOG_ROOT/preserved_capture.out" >/dev/null \
    || fail "preserved run did not report its retained path"
[[ -n "$(find "$preserved_root" -maxdepth 1 -type d -name 'orchav-authoring-display.*')" ]] \
    || fail "display-capture directory was not retained"
remove_retained_run_root "$preserved_root"
assert_sentinels

missing_marker_status=0
run_candidate marker_missing "" "" "" 0 0 marker_missing \
    || missing_marker_status=$?
[[ "$missing_marker_status" == "1" ]] \
    || fail "missing marker returned $missing_marker_status instead of 1"
missing_marker_root="$(assert_retained_refusal "$LOG_ROOT/marker_missing.out")"
remove_retained_run_root "$missing_marker_root"
assert_sentinels

mismatch_marker_status=0
run_candidate marker_mismatch "" "" "" 0 0 marker_mismatch \
    || mismatch_marker_status=$?
[[ "$mismatch_marker_status" == "1" ]] \
    || fail "mismatched marker returned $mismatch_marker_status instead of 1"
mismatch_marker_root="$(assert_retained_refusal "$LOG_ROOT/marker_mismatch.out")"
remove_retained_run_root "$mismatch_marker_root"
assert_sentinels

backup_collision_status=0
run_candidate \
    root_backup_collision \
    "" \
    "" \
    "" \
    0 \
    0 \
    root_backup_collision \
    "$ESCAPE_ROOT/junction-target" \
    || backup_collision_status=$?
[[ "$backup_collision_status" == "94" ]] \
    || fail "backup collision returned $backup_collision_status instead of 94"
backup_collision_root="$(extract_run_root "$LOG_ROOT/root_backup_collision.out")"
case "$backup_collision_root" in
    "$SCRATCH_PARENT"/orchav-release-smoke.*) ;;
    *) fail "backup collision run root escaped scratch parent: $backup_collision_root" ;;
esac
[[ ! -e "$backup_collision_root" && ! -L "$backup_collision_root" ]] \
    || fail "backup collision left the owned run root in place"
backup_collision_path="$backup_collision_root.original"
[[ "$(cat "$backup_collision_path/KEEP_ME.txt")" == "backup collision sentinel" ]] \
    || fail "backup collision overwrote its pre-existing destination"
grep -F "refusing to replace existing backup path" \
    "$LOG_ROOT/root_backup_collision.out" >/dev/null \
    || fail "backup collision did not report its destination-safety refusal"
rm -- "$backup_collision_path/KEEP_ME.txt"
rmdir -- "$backup_collision_path"
assert_sentinels

root_link_status=0
run_candidate \
    root_link_replacement \
    "" \
    "" \
    "" \
    0 \
    0 \
    root_link \
    "$ESCAPE_ROOT/junction-target" \
    || root_link_status=$?
[[ "$root_link_status" == "1" ]] \
    || {
        echo "Junction replacement output:" >&2
        cat "$LOG_ROOT/root_link_replacement.out" >&2
        echo "Junction replacement tool log:" >&2
        cat "$LOG_ROOT/root_link_replacement.tools" >&2
        fail "junction replacement returned $root_link_status instead of 1"
    }
root_link_path="$(assert_retained_refusal "$LOG_ROOT/root_link_replacement.out")"
if ! grep -F "FINAL_LINK_CHECK" \
    "$LOG_ROOT/root_link_replacement.tools" >/dev/null; then
    cat "$LOG_ROOT/root_link_replacement.tools" >&2
    fail "cleanup did not run its native link guard on the replaced root"
fi
if ! "$REAL_PYTHON" -I -B -c '
from pathlib import Path
import sys

path = Path(sys.argv[1])
is_junction = getattr(path, "is_junction", lambda: False)
raise SystemExit(0 if path.is_symlink() or is_junction() else 1)
' "$root_link_path"; then
    fail "run-root replacement was not a recognized directory link"
fi
remove_directory_link "$root_link_path"
remove_retained_run_root "$root_link_path.original"
assert_sentinels

dangling_link_status=0
run_candidate \
    dangling_root_link_replacement \
    "" \
    "" \
    "" \
    0 \
    0 \
    root_dangling \
    "$ESCAPE_ROOT/dangling-junction-target" \
    || dangling_link_status=$?
[[ "$dangling_link_status" == "1" ]] \
    || fail "dangling junction replacement returned $dangling_link_status instead of 1"
dangling_link_path="$(
    assert_retained_refusal "$LOG_ROOT/dangling_root_link_replacement.out"
)"
if ! grep -F "FINAL_LINK_CHECK" \
    "$LOG_ROOT/dangling_root_link_replacement.tools" >/dev/null; then
    cat "$LOG_ROOT/dangling_root_link_replacement.tools" >&2
    fail "cleanup did not probe the dangling replacement root"
fi
if ! "$REAL_PYTHON" -I -B -c '
from pathlib import Path
import sys

path = Path(sys.argv[1])
is_junction = getattr(path, "is_junction", lambda: False)
raise SystemExit(0 if path.is_symlink() or is_junction() else 1)
' "$dangling_link_path"; then
    fail "dangling run-root replacement was not recognized as a directory link"
fi
remove_directory_link "$dangling_link_path"
remove_retained_run_root "$dangling_link_path.original"
assert_sentinels

printf "dirty checkout sentinel\n" >"$CLONE/DIRTY_WORKTREE.txt"
dirty_status=0
(
    cd "$CLONE"
    PATH="$FAKE_BIN:$PATH" \
    FAKE_TOOL_LOG="$LOG_ROOT/dirty.tools" \
    REAL_LINK_CHECK_PYTHON="$REAL_PYTHON" \
    ORCHAV_TMP_DIR="$SCRATCH_PARENT" \
    "$REAL_BASH" scripts/ci/release_smoke.sh
) >"$LOG_ROOT/dirty.out" 2>&1 || dirty_status=$?
[[ "$dirty_status" == "2" ]] \
    || fail "dirty-worktree rejection returned $dirty_status instead of 2"
grep -F "release smoke requires a clean tracked/untracked worktree" \
    "$LOG_ROOT/dirty.out" >/dev/null \
    || fail "dirty-worktree rejection did not explain the error"
if grep -F "Scratch run root:" "$LOG_ROOT/dirty.out" >/dev/null; then
    fail "dirty-worktree rejection created a smoke run root"
fi
rm -f -- "$CLONE/DIRTY_WORKTREE.txt"
assert_sentinels

if (
    cd "$CLONE"
    PATH="$FAKE_BIN:$PATH" \
    FAKE_TOOL_LOG="$LOG_ROOT/inside_repo.tools" \
    REAL_LINK_CHECK_PYTHON="$REAL_PYTHON" \
    ORCHAV_TMP_DIR="$CLONE/.smoke-parent" \
    "$REAL_BASH" scripts/ci/release_smoke.sh
) >"$LOG_ROOT/inside_repo.out" 2>&1; then
    fail "scratch parent inside the checkout was accepted"
fi
grep -F "scratch parent must be outside the checkout" \
    "$LOG_ROOT/inside_repo.out" >/dev/null \
    || fail "inside-checkout rejection did not explain the error"
assert_sentinels

LINK_PARENT="$TEST_ROOT/linked scratch parent"
create_directory_link "$LINK_PARENT" "$SCRATCH_PARENT" \
    || fail "could not create a directory link for the platform safety test"
if ! "$REAL_PYTHON" -I -B -c '
from pathlib import Path
import sys

path = Path(sys.argv[1])
is_junction = getattr(path, "is_junction", lambda: False)
raise SystemExit(0 if path.is_symlink() or is_junction() else 1)
' "$LINK_PARENT"; then
    fail "the platform link was not recognized as a symlink or junction"
fi
if (
    cd "$CLONE"
    PATH="$FAKE_BIN:$PATH" \
    FAKE_TOOL_LOG="$LOG_ROOT/linked_parent.tools" \
    REAL_LINK_CHECK_PYTHON="$REAL_PYTHON" \
    ORCHAV_TMP_DIR="$LINK_PARENT" \
    "$REAL_BASH" scripts/ci/release_smoke.sh
) >"$LOG_ROOT/linked_parent.out" 2>&1; then
    fail "linked scratch parent was accepted"
fi
grep -F "scratch parent may not be a symlink or junction" \
    "$LOG_ROOT/linked_parent.out" >/dev/null \
    || fail "linked-parent rejection did not explain the error"
remove_directory_link "$LINK_PARENT"

assert_sentinels
echo "PASS: release smoke owns only unique external scratch paths"
