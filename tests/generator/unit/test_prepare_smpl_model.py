"""Conversion must preserve model values and require explicit pickle trust."""

import pickle

import numpy as np
import pytest
from scipy import sparse

from generator.tools.prepare_smpl_model import prepare_model


def test_requires_trust_before_reading_model(tmp_path, monkeypatch):
    def forbidden_load(*args, **kwargs):
        pytest.fail("Untrusted model must not be deserialized")

    monkeypatch.setattr(pickle, "loads", forbidden_load)
    with pytest.raises(ValueError, match="--trusted-model"):
        prepare_model(tmp_path / "missing.pkl", tmp_path / "output.pkl")
    assert not (tmp_path / "output.pkl").exists()


def test_preserves_numeric_and_sparse_parameters(tmp_path):
    original = {
        "v_template": np.arange(12, dtype=np.float64).reshape(4, 3),
        "J_regressor": sparse.csc_matrix(np.eye(4)),
        "metadata": {"name": "synthetic", "values": [1, 2, 3]},
    }
    source = tmp_path / "source.pkl"
    source_bytes = pickle.dumps(original, protocol=2)
    source.write_bytes(source_bytes)
    output = tmp_path / "models" / "SMPL_MALE.pkl"

    assert prepare_model(source, output, trusted_model=True) == output

    converted = pickle.loads(output.read_bytes())
    np.testing.assert_array_equal(converted["v_template"], original["v_template"])
    assert (converted["J_regressor"] != original["J_regressor"]).nnz == 0
    assert converted["metadata"] == original["metadata"]
    assert source.read_bytes() == source_bytes


def test_refuses_existing_destination_before_loading_source(tmp_path):
    output = tmp_path / "existing.pkl"
    output.write_bytes(b"preserve existing model")
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_model(tmp_path / "missing.pkl", output, trusted_model=True)
    assert output.read_bytes() == b"preserve existing model"
