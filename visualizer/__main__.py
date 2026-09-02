"""Entry point for ``python -m visualizer``.

The import-light renderer-support gate runs before Qt or Open3D is imported.
Desktop graphics selection remains under the active operating-system display
stack.
"""

from visualizer.gpu_preflight import (
    reject_unsupported_macos_open3d as _reject_unsupported_macos_open3d,
)

_reject_unsupported_macos_open3d()

from visualizer.visualizer import main  # noqa: E402  (import after platform gate)

if __name__ == "__main__":
    main()
