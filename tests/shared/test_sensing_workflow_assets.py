"""Every new workflow illustration is bound to recorded provenance and current bytes."""

import hashlib
import json
from pathlib import Path


def test_workflow_illustration_inventory_and_hashes():
    root = Path(__file__).resolve().parents[2] / "docs/assets/sensing_workflow"
    manifest = json.loads((root / "provenance.json").read_text())
    assets = manifest["assets"]
    assert {a["image"] for a in assets} == {p.name for p in root.glob("*.png")}
    assert len({a["image"] for a in assets}) == len(assets)
    for asset in assets:
        assert asset["source"]
        assert asset.get("command") or asset.get("reproduction_status", "").startswith(
            "Historical instructional capture;"
        )
        assert hashlib.sha256((root / asset["image"]).read_bytes()).hexdigest() == asset["sha256"]
