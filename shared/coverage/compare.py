"""CPU coverage comparison: python -m shared.coverage.compare DEFINITION --output PREFIX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .comparison import analyze_definition
from .comparison_export import export_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("definition", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--row", type=int, default=0, help="Curve row; all sources remain in summary"
    )
    parser.add_argument("--threshold", type=float, default=-75.0, help="Displayed threshold in dBm")
    args = parser.parse_args(argv)
    definition = json.loads(args.definition.read_text())
    definition["thresholds_dbm"] = sorted(set(definition["thresholds_dbm"] + [args.threshold]))
    result = analyze_definition(definition, base_dir=args.definition.parent)
    export_comparison(result, args.output, row_index=args.row, threshold_dbm=args.threshold)


if __name__ == "__main__":
    main()
