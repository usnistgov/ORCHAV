"""Run Black checks with immediate process exit after Black returns.

Some local Sionna/Open3D/Qt environments can hang during Python interpreter
shutdown after broad tooling imports. This wrapper preserves Black's exit code
and skips teardown after the check has completed.
"""

from __future__ import annotations

import os
import sys

from black import patched_main


def main() -> None:
    """Run Black's CLI entry point and terminate with its status code."""
    sys.argv = ["black", *sys.argv[1:]]
    try:
        patched_main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    else:
        code = 0
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
