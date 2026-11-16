"""Small CLI conveniences shared by authored sensing-volume campaigns."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from scenarios.sensing.volume_storage import retention_mode


def prepare_volume(root, prepare, *args, recipe_roots=None, **kwargs):
    """Skip repeated CLI preparation without changing the saved experiment.

    Underlying preparation functions still refuse overwrites. Recognize an
    existing directory only when all expected recipes and their referenced
    configurations exist; partial preparation needs a new destination.
    """
    if not root.exists():
        return prepare(root, *args, **kwargs)
    recipes = []
    try:
        for case in recipe_roots if recipe_roots is not None else [root]:
            spec = json.loads((case / "volume_recipe.json").read_text())
            if not (case / spec["scene_scenario"]).is_file() or not isinstance(
                spec["trials"], list
            ):
                raise ValueError("missing scene or trial list")
            for trial in spec["trials"]:
                if (
                    trial["status"] == "pending"
                    and not (case / trial["native"] / "scenario.yaml").is_file()
                ):
                    raise ValueError("missing trial configuration")
            recipes.append((case, spec, retention_mode(spec)))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"Directory already exists but preparation is incomplete or unrecognized:\n"
            f"  {root}\n"
            "Existing files were kept. Choose a new output directory for --prepare."
        ) from exc

    print(f"Already prepared: {root}", flush=True)
    for case, spec, mode in recipes:
        trials = [t for t in spec["trials"] if t["status"] == "pending"]
        saved = sum((case / t["native"]).parent.joinpath("outcome.json").is_file() for t in trials)
        print(
            f"  {case.name}: grid={spec.get('grid_preset', 'custom')}, retention={mode}; "
            f"{saved:,}/{len(trials):,} trials saved.",
            flush=True,
        )
    script = Path(sys.argv[0]).resolve()
    repository = Path(__file__).resolve().parents[2]
    if script.is_relative_to(repository) and script.suffix == ".py":
        command = ["python", "-m", ".".join(script.relative_to(repository).with_suffix("").parts)]
    else:
        command = ["python", str(script)]
    destination = (
        str(root.relative_to(Path.cwd())) if root.is_relative_to(Path.cwd()) else str(root)
    )
    print(
        "Preparation skipped; existing settings and recordings were kept.\n"
        "Continue generation (completed trials are skipped):\n"
        f"  {shlex.join([*command, destination, '--yes'])}\n"
        "To use different preparation options, choose a new output directory.",
        flush=True,
    )
