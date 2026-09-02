#!/usr/bin/env python3
"""Regenerate or verify the checked-in ORCHAV protobuf/gRPC stubs.

The generated modules stay faithful to ``grpcio-tools`` output except for the
package-relative import required by ``shared.protos``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTO_DIR = PROJECT_ROOT / "shared" / "protos"
PROTO_FILE = PROTO_DIR / "visualizer.proto"
GENERATED_FILENAMES = ("visualizer_pb2.py", "visualizer_pb2_grpc.py")
GRPC_TOOLS_VERSION = "1.81.1"


def _toolchain_is_supported() -> bool:
    """Require the compiler version that defines checked-in generated output."""
    try:
        installed = version("grpcio-tools")
    except PackageNotFoundError:
        installed = None
    if installed == GRPC_TOOLS_VERSION:
        return True

    found = installed if installed is not None else "not installed"
    print(
        f"grpcio-tools=={GRPC_TOOLS_VERSION} is required for protobuf generation "
        f'(found {found}). Install it with: python -m pip install -e ".[dev]"'
    )
    return False


def _patch_generated_imports(proto_dir: Path) -> None:
    """Ensure generated Python stubs use package-relative imports."""
    grpc_file = proto_dir / "visualizer_pb2_grpc.py"
    content = grpc_file.read_text(encoding="utf-8")
    target = "import visualizer_pb2 as visualizer__pb2"
    replacement = "from . import visualizer_pb2 as visualizer__pb2"
    if replacement not in content:
        if target not in content:
            raise RuntimeError(f"generated import not found in {grpc_file}")
        content = content.replace(target, replacement)

    # Explicit UTF-8 bytes keep the byte-for-byte check independent of the
    # host platform's default newline translation.
    grpc_file.write_bytes(content.encode("utf-8"))


def _generate_into(output_dir: Path) -> bool:
    """Run protoc into *output_dir* and apply ORCHAV's package import policy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={PROTO_DIR}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        str(PROTO_FILE),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print("Error generating protobuf files:")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return False

    try:
        _patch_generated_imports(output_dir)
    except (OSError, RuntimeError) as exc:
        print(f"Error patching generated protobuf imports: {exc}")
        return False
    return True


def generate_protobuf(*, check: bool = False) -> bool:
    """Generate stubs, or verify that regeneration would not change them."""
    if not _toolchain_is_supported():
        return False
    if not PROTO_FILE.exists():
        print(f"Error: {PROTO_FILE} not found")
        return False

    if not check:
        print(f"Generating protobuf files in: {PROTO_DIR}")
        if not _generate_into(PROTO_DIR):
            return False
        print("Protobuf files generated successfully")
        return True

    with tempfile.TemporaryDirectory(prefix="orchav-protobuf-check-") as temporary:
        generated_dir = Path(temporary)
        if not _generate_into(generated_dir):
            return False

        changed = [
            name
            for name in GENERATED_FILENAMES
            if not (PROTO_DIR / name).exists()
            or (PROTO_DIR / name).read_bytes() != (generated_dir / name).read_bytes()
        ]
    if changed:
        print("Protobuf regeneration would change:")
        for name in changed:
            print(f"  {PROTO_DIR / name}")
        return False

    print("Checked-in protobuf files match regenerated output")
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify checked-in stubs without modifying the source tree",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    success = generate_protobuf(check=args.check)
    sys.exit(0 if success else 1)
