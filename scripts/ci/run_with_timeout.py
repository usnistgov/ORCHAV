#!/usr/bin/env python3
"""Run one command with a cross-platform whole-process timeout."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from collections.abc import Sequence

TIMEOUT_EXIT_CODE = 124


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate the child process group without adding a psutil dependency."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def run_with_timeout(command: Sequence[str], timeout_s: float) -> int:
    """Run the command and return 124 after terminating it on timeout."""
    if not command:
        raise ValueError("a command is required")
    popen_options: dict[str, object] = {}
    if os.name == "nt":
        popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_options["start_new_session"] = True
    process = subprocess.Popen(list(command), **popen_options)
    try:
        return process.wait(timeout=float(timeout_s))
    except subprocess.TimeoutExpired:
        print(
            f"Command exceeded {float(timeout_s):g} seconds; terminating process tree.",
            file=sys.stderr,
        )
        _terminate_process_tree(process)
        process.wait()
        return TIMEOUT_EXIT_CODE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-s", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    return args


def main() -> int:
    args = parse_args()
    return run_with_timeout(args.command, args.timeout_s)


if __name__ == "__main__":
    raise SystemExit(main())
