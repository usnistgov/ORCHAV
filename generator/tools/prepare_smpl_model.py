"""Write a Chumpy-free copy of a trusted, separately obtained SMPL model.

The original SMPL Python v1.0.0 download stores some arrays as Chumpy objects.
Convert these once before using the SMPL/AMASS importer with modern Python.
No body geometry, shape coefficients, or model parameters are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from generator.tools.smpl_to_ply import _install_chumpy_compat


def _numeric_copy(value: Any) -> Any:
    """Evaluate legacy Chumpy arrays while retaining ordinary model values."""
    if type(value).__module__.split(".")[0] == "chumpy":
        return np.asarray(value.r)
    if isinstance(value, dict):
        return {key: _numeric_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_numeric_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_numeric_copy(item) for item in value)
    return value


def prepare_model(source: Path, output: Path, *, trusted_model: bool = False) -> Path:
    """Convert one trusted pickle, preserving the source and existing outputs."""
    if not trusted_model:
        raise ValueError("--trusted-model is required to deserialize the supplied SMPL pickle")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new file: {output}")
    source_bytes = source.read_bytes()
    _install_chumpy_compat()
    try:
        model = pickle.loads(source_bytes, encoding="latin1")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "chumpy":
            raise RuntimeError(
                "This legacy model needs Chumpy for one-time conversion. Run "
                "'python -m pip install setuptools wheel', then "
                "'python -m pip install --no-build-isolation chumpy==0.70', "
                "and retry. Chumpy is not needed to import the converted model."
            ) from exc
        raise
    if not isinstance(model, dict):
        raise ValueError("The SMPL model must contain a parameter dictionary")
    payload = pickle.dumps(_numeric_copy(model), protocol=4)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)
    print(f"Source SHA256: {hashlib.sha256(source_bytes).hexdigest()}")  # noqa: T201
    print(f"Output SHA256: {hashlib.sha256(payload).hexdigest()}")  # noqa: T201
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path, required=True, help="Original SMPL pickle")
    parser.add_argument("--output", type=Path, required=True, help="New model pickle to write")
    parser.add_argument(
        "--trusted-model", action="store_true", help="Allow loading this trusted model pickle"
    )
    args = parser.parse_args(argv)
    try:
        print(  # noqa: T201
            prepare_model(args.model_file, args.output, trusted_model=args.trusted_model)
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
