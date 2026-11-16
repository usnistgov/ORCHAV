"""Prepare saved sensing views before a demo: python -m visualizer.sensing_prepare SCENARIO."""

import argparse
import json
import sys
from pathlib import Path
from threading import Event
from time import monotonic

from shared.frames.providers import Hdf5Provider

from .src.sensing.cache_policy import add_sensing_cache_arguments, policy_from_options
from .src.sensing.preparation_status import prepared_coverage_messages
from .src.sensing.recording_cache import RecordingCache
from .src.sensing.startup_cache import prepare_motion_cache, prepare_startup_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", help="Scenario directory or scenario.yaml with saved frames")
    parser.add_argument(
        "--sensing-figures",
        choices=("all", "motion"),
        default="all",
        help="Prepare all saved figure types, or only the chosen Motion view (default: all).",
    )
    parser.add_argument(
        "--sensing-motion",
        choices=("all-ranges", "saved-stft"),
        default=None,
        help="Motion view to prepare: range-summed RD or saved STFT (default: all-ranges).",
    )
    parser.add_argument(
        "--sensing-3d",
        choices=("on", "off"),
        default=None,
        help="Include saved 3D sensing preparation (default: on for all, off for motion).",
    )
    add_sensing_cache_arguments(parser)
    args = parser.parse_args(argv)
    if args.sensing_motion is not None and args.sensing_figures != "motion":
        parser.error("--sensing-motion requires --sensing-figures motion")
    include_scene = (
        args.sensing_3d == "on" if args.sensing_3d is not None else args.sensing_figures == "all"
    )
    source = Path(args.scenario)
    if source.suffix.lower() in {".yaml", ".yml"}:
        source = source.parent
    provider = Hdf5Provider(source)
    cancelled = Event()
    start = monotonic()
    last = (None, 0.0)

    def progress(stage, done, total):
        nonlocal last
        now = monotonic()
        if stage != last[0] or now - last[1] > 1:
            sys.stdout.write(f"{stage}: {done}/{total}\n")
            sys.stdout.flush()
            last = stage, now

    try:
        policy = policy_from_options(args, prepare=True)
        if args.sensing_figures == "motion":
            capabilities = prepare_motion_cache(
                provider,
                cancelled,
                progress,
                motion_mode="saved-stft" if args.sensing_motion == "saved-stft" else "auto",
                include_scene=include_scene,
                cache_policy=policy,
            )
            sys.stdout.write(f"Motion disk cache ready in {monotonic() - start:.2f} s.\n")
        else:
            built = prepare_startup_cache(
                provider,
                cancelled,
                progress,
                retain_session=False,
                include_scene=include_scene,
                cache_policy=policy,
            )
            sys.stdout.write(
                f"Saved-figure disk cache {'built' if built else 'reused'} "
                f"in {monotonic() - start:.2f} s.\n"
            )
            cache = RecordingCache.for_provider(provider)
            marker = json.loads((cache._prepared_root / "complete.json").read_text())
            capabilities = marker["capabilities"]
        sys.stdout.write(
            f"3D sensing {'included' if include_scene else 'deferred'}. "
            "Complex Inputs and explicit signal reconstruction load when requested.\n"
        )
        for message in prepared_coverage_messages(capabilities):
            sys.stdout.write(message + "\n")
        return 0
    except KeyboardInterrupt:
        cancelled.set()
        sys.stderr.write("Sensing preparation cancelled.\n")
        return 130
    finally:
        provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
