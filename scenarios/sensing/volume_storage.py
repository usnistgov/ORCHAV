"""Explain campaign storage and release only explicitly disposable trial evidence."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

MODES = ("full", "volume")


def configure_trial_summary(configuration, previous_trials):
    """Keep native plots on the first eligible trial, retained under either policy.

    Other recordings still support the ordinary saved-summary CLI. Selecting
    once during preparation avoids thousands of repeated figure jobs and binds
    the display choice into each trial's authored configuration hash.
    """
    summary = configuration.setdefault("generator_summary", {})
    reference = not any(trial["status"] == "pending" for trial in previous_trials)
    summary["enabled"] = bool(summary.get("enabled", False) and reference)
    return summary["enabled"]


def retention_spec(trials, mode):
    """Freeze the policy before acquisition; retain one native metadata reference."""
    if mode not in MODES:
        raise ValueError(f"Unknown retention mode: {mode}")
    first = next((t["id"] for t in trials if t["status"] == "pending"), None)
    return {"mode": mode, "reference_trial_ids": [first] if mode == "volume" and first else []}


def retention_mode(spec):
    mode = spec.get("retention", {}).get("mode", "full")
    if mode not in MODES:
        raise ValueError(f"Unsupported frozen retention mode: {mode}")
    return mode


def keeps_native(spec, trial):
    return retention_mode(spec) == "full" or trial["id"] in spec["retention"]["reference_trial_ids"]


def storage_plan(roots, *, worker=0, workers=1, limit=None):
    """Estimate remaining work, explicitly labelling assumptions rather than caps."""
    plans = []
    for root in roots:
        root = Path(root)
        spec = json.loads((root / "volume_recipe.json").read_text())
        mode = retention_mode(spec)
        trials = [t for t in spec["trials"] if t["status"] == "pending"]
        if worker is None:
            selected = [
                t
                for i in range(workers)
                for t in (trials[i::workers][:limit] if limit is not None else trials[i::workers])
            ]
        else:
            selected = trials[worker::workers]
            if limit is not None:
                selected = selected[:limit]
        completed = [t for t in selected if (root / t["native"] / "../outcome.json").is_file()]
        remaining = len(selected) - len(completed)
        # Scenario-specific estimates are planning inputs, not universal bounds.
        mb = float(spec.get("estimated_native_mb_per_trial", 10.0))
        if not math.isfinite(mb) or mb <= 0:
            raise ValueError("Expected positive per-trial storage estimate")
        retained = len(trials) if mode == "full" else len(spec["retention"]["reference_trial_ids"])
        observed = []
        for trial in completed:
            outcome = root / trial["native"] / "../outcome.json"
            row = json.loads(outcome.read_text())
            if row.get("native_evidence") == "not_included":
                continue
            frames = root / trial["native"] / "frames"
            files = [frames / name for name in row["source_hashes"]]
            if files and all(f.is_file() for f in files):
                observed.append(sum(f.stat().st_size for f in files) / 1e6)
            if len(observed) >= 16:
                break
        if observed:
            mb = sum(observed) / len(observed)
        plans.append(
            dict(
                root=str(root),
                mode=mode,
                trials=len(trials),
                remaining=remaining,
                native_recordings=retained,
                native_mb_per_trial=mb,
                estimated_final_gb=(retained * mb + len(trials) * 0.032) / 1000,
                free_gb=shutil.disk_usage(root).free / 1e9,
                observed_recordings=len(observed),
            )
        )
    return plans


def confirm_storage(roots, *, yes=False, worker=0, workers=1, limit=None, compact_supported=True):
    """CLI-only confirmation, before workers start or the ray tracer is imported."""
    plans = storage_plan(roots, worker=worker, workers=workers, limit=limit)
    remaining = sum(p["remaining"] for p in plans)
    print(f"\nSensing volume generation | {remaining:,} acquisitions remaining", flush=True)
    for p in plans:
        print(
            f"\n  {p['root']}\n"
            f"    Trials      {p['trials']:,} planned; {p['remaining']:,} remaining in this run\n"
            f"    Storage     retention={p['mode']}; {p['native_recordings']:,} native recordings kept\n"
            f"    Estimate    {p['estimated_final_gb']:.2f} GB for the complete campaign; "
            f"{p['free_gb']:.1f} GB free\n"
            f"                {p['native_mb_per_trial']:.2f} MB/native trial "
            f"({'observed sample mean' if p['observed_recordings'] else 'planning assumption'})",
            flush=True,
        )
    if any(p["mode"] == "full" for p in plans):
        print(
            "\n  Full retention keeps every trial's frames, paths and sensing products.\n"
            + (
                "  For smaller results, prepare a NEW directory with --retention volume."
                if compact_supported
                else "  This experiment retains both native sources for fusion replay."
            ),
            flush=True,
        )
    if any(p["mode"] == "volume" for p in plans):
        print(
            "\n  Volume retention keeps configurations, truth, reports and one native reference.\n"
            "  Other native frames are deleted after verified extraction; volume inspection\n"
            "  and gate rescoring remain available, but RD/CIR inspection and signal\n"
            "  reprocessing require retained recordings. Every trial still runs ray tracing.",
            flush=True,
        )
    print(
        "  Estimates include 32 kB/trial for summaries/configurations; actual sizes vary.\n"
        "  Worker logs and temporary recordings are additional. Existing retention is unchanged.\n",
        flush=True,
    )
    if not remaining or yes:
        return
    if not sys.stdin.isatty():
        raise SystemExit(
            "Acquisition not started: interactive confirmation required; use --yes for an intentional batch run."
        )
    try:
        answer = input("Start these acquisitions with the displayed retention policy? [y/N] ")
    except EOFError:
        answer = ""
    if answer.strip().lower() not in {"y", "yes"}:
        raise SystemExit("Acquisition cancelled; prepared configurations remain available.")


def release_native_frames(root, spec, trial):
    """Release only hash-verified files after an atomic compact outcome exists.

    A resumed cleanup tolerates files already removed after a prior interruption.
    Unlisted files and scenario YAML are never deleted. Full-retention recipes
    do nothing, even if an outcome happens to carry a compact availability flag.
    """
    if keeps_native(spec, trial):
        return
    root = Path(root).resolve()
    native = root / trial["native"]
    if not native.resolve().is_relative_to(root) or native.is_symlink():
        raise ValueError("Native cleanup path escapes its campaign")
    outcome = json.loads((native.parent / "outcome.json").read_text())
    if outcome.get("native_evidence") != "not_included" or "reports" not in outcome:
        raise ValueError("Persist verified compact reports before releasing native frames")
    if (
        outcome.get("id") != trial["id"]
        or outcome.get("configuration_sha256") != trial["configuration_sha256"]
    ):
        raise ValueError("Compact outcome is not bound to this trial")
    digest = hashlib.sha256((native / "scenario.yaml").read_bytes()).hexdigest()
    if digest != trial["configuration_sha256"]:
        raise ValueError("Native configuration changed before cleanup")
    frames = native / "frames"
    if frames.is_symlink():
        raise ValueError("Refuse to remove files through a symlinked frames directory")
    files = []
    for name, expected in outcome["source_hashes"].items():
        if Path(name).name != name or Path(name).suffix not in {".json", ".h5"}:
            raise ValueError("Unsafe source fingerprint path")
        path = frames / name
        if path.is_symlink():
            raise ValueError("Refuse to remove a symlinked source file")
        if path.exists():
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError("Native evidence changed; cleanup refused")
            files.append(path)
    # Check every remaining file before removing any; preserve unexpected files.
    for path in files:
        path.unlink()
    if frames.is_dir() and not any(frames.iterdir()):
        frames.rmdir()
