"""Installed CLI for already-calibrated measurement comparison evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from shared.sensing.measurement import MeasurementContractError
from shared.sensing.measurement_workflow import (
    publish_measurement_comparison,
    publish_synthetic_measurement_workflow,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchav-sensing-compare",
        description=(
            "Import already-calibrated sensing products and publish aligned "
            "measurement/simulation/residual evidence without calibration code."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo",
        help="publish the ORCHAV-authored synthetic measurement comparison",
    )
    demo.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory to publish atomically; it must not already exist",
    )

    compare = subparsers.add_parser(
        "compare",
        help="compare two ingress manifests with an exact hashed alignment recipe",
    )
    compare.add_argument("--measurement-manifest", type=Path, required=True)
    compare.add_argument("--simulation-manifest", type=Path, required=True)
    compare.add_argument("--alignment-recipe", type=Path, required=True)
    compare.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory to publish atomically; it must not already exist",
    )
    return parser


def _summary(publication) -> str:
    return json.dumps(
        {
            "output_dir": str(publication.root),
            "comparison_json": str(publication.comparison_json),
            "residual_csv": str(publication.residual_csv),
            "synchronized_png": str(publication.synchronized_png),
            "hash_inventory": str(publication.hash_inventory),
            "inventory_sha256": publication.inventory_sha256,
        },
        allow_nan=False,
        sort_keys=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the installed already-calibrated comparison workflow."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "demo":
            publication = publish_synthetic_measurement_workflow(args.output_dir)
        else:
            publication = publish_measurement_comparison(
                args.measurement_manifest,
                args.simulation_manifest,
                args.alignment_recipe,
                args.output_dir,
            )
    except (MeasurementContractError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    sys.stdout.write(_summary(publication) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - installed entry point owns dispatch
    raise SystemExit(main())
