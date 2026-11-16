"""Export a 3D coverage slice using saved data, without starting the Visualizer."""

import argparse
import sys
from pathlib import Path

from generator.core.configuration.defaults import DEFAULT_COVERAGE_SAVE_PATH
from generator.figures.coverage.spatial import create_coverage_3d_figure
from shared.scenarios import load_scenario_configuration


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True, help="Scenario directory or YAML")
    parser.add_argument(
        "--coverage-file",
        type=Path,
        help="Existing coverage HDF5; defaults to the scenario's coverage/coverage_maps.h5",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination PNG, PDF or SVG")
    parser.add_argument(
        "--metric", default="best_path_loss_db", help="Metric, optionally qualified by /TX_name"
    )
    parser.add_argument(
        "--height-index",
        type=int,
        default=0,
        help="Zero-based index into the saved height list (default: 0)",
    )
    parser.add_argument("--elevation", type=float, default=40.0, help="Camera elevation in degrees")
    parser.add_argument("--azimuth", type=float, default=-60.0, help="Camera azimuth in degrees")
    args = parser.parse_args(argv)
    config = load_scenario_configuration(args.scenario)
    scenario_root = args.scenario if args.scenario.is_dir() else args.scenario.parent
    coverage_file = args.coverage_file or scenario_root / DEFAULT_COVERAGE_SAVE_PATH
    path = create_coverage_3d_figure(
        coverage_file,
        args.output,
        scenario_context=config,
        metric=args.metric,
        height_index=args.height_index,
        elevation=args.elevation,
        azimuth=args.azimuth,
    )
    sys.stdout.write(f"Saved {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
