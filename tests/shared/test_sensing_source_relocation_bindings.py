"""Source relocation receipts must authenticate unchanged scientific statements."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.docs.refresh_sensing_authoring_receipts import (
    SOURCE_RELOCATION_BINDINGS,
    bound_authoring_bytes,
)

COMPARE = "fixture/compare.py"


def _binding_fixture(root: Path, name: str = COMPARE):
    previous = "from shared.sensing_range_response import C0\nVALUE = 1\n"
    current = previous.replace("shared.sensing_range_response", "shared.sensing.range_response")
    document = {
        "schema": "orchav.sensing-source-relocation-bindings.v1",
        "sources": {
            name: {
                "previous_authored_sha256": hashlib.sha256(previous.encode()).hexdigest(),
                "current_authored_sha256": hashlib.sha256(current.encode()).hexdigest(),
                "replacements": [
                    {
                        "previous": "from shared.sensing_range_response import C0",
                        "current": "from shared.sensing.range_response import C0",
                    }
                ],
            }
        },
    }
    source = root / name
    source.parent.mkdir(parents=True)
    source.write_text(current)
    inventory = root / SOURCE_RELOCATION_BINDINGS
    inventory.parent.mkdir(parents=True)
    inventory.write_text(json.dumps(document))
    return source, inventory, document


def test_declared_source_relocation_preserves_original_receipt_identity(tmp_path):
    source, _, document = _binding_fixture(tmp_path)
    current = source.read_bytes()
    row = document["sources"][COMPARE]
    previous = bound_authoring_bytes(source, tmp_path)
    assert hashlib.sha256(previous).hexdigest() == row["previous_authored_sha256"]
    assert hashlib.sha256(current).hexdigest() == row["current_authored_sha256"]
    assert previous != current
    assert source.read_bytes() == current


def test_source_binding_rejects_changed_current_input(tmp_path):
    source, _, _ = _binding_fixture(tmp_path)
    source.write_bytes(source.read_bytes() + b"\nSCIENTIFIC_FACTOR = 2\n")
    with pytest.raises(ValueError, match="Current source input changed"):
        bound_authoring_bytes(source, tmp_path)


def test_source_binding_rejects_scientific_edits_even_with_updated_hash_and_blocks(tmp_path):
    source, inventory, document = _binding_fixture(tmp_path)
    addition = "\nSCIENTIFIC_FACTOR = 2\n"
    source.write_bytes(source.read_bytes() + addition.encode())
    row = document["sources"][COMPARE]
    row["current_authored_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    row["replacements"].append({"current": addition, "previous": ""})
    inventory.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="changes scientific statements"):
        bound_authoring_bytes(source, tmp_path)


def test_source_binding_rejects_unapproved_import_retargeting(tmp_path):
    source, inventory, document = _binding_fixture(tmp_path)
    source.write_bytes(
        source.read_bytes().replace(b"shared.sensing.range_response", b"unreviewed.range_response")
    )
    row = document["sources"][COMPARE]
    row["current_authored_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    row["replacements"][0]["current"] = row["replacements"][0]["current"].replace(
        "shared.sensing.range_response", "unreviewed.range_response"
    )
    inventory.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="changes scientific statements"):
        bound_authoring_bytes(source, tmp_path)


def test_unlisted_python_source_requires_its_exact_current_bytes(tmp_path):
    _, _, _ = _binding_fixture(tmp_path)
    source = tmp_path / "unlisted.py"
    source.write_bytes(b"VALUE = 1\n")
    assert bound_authoring_bytes(source, tmp_path) == source.read_bytes()
