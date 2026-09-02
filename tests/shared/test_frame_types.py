"""Tests for canonical frame construction and explicit validation."""

import numpy as np
import pytest

from shared.frames import StandardMPCFrame, standard_mpc_frame_from_pair_data
from shared.frames.schema import (
    is_valid_standard_mpc_frame,
    validate_standard_mpc_frame,
)
from shared.frames.types import FRAME_FORMAT_VERSION


def _frame() -> StandardMPCFrame:
    return standard_mpc_frame_from_pair_data(
        frame_index=0,
        tx_rx_pairs=[[0, 0]],
        tx_positions=[[1.0, 2.0, 3.0]],
        rx_positions=[[4.0, 5.0, 6.0]],
        vertices_by_pair=[[[[2.0, 2.0, 2.0]]]],
        interactions_by_pair=[[[1]]],
        provenance={"provider": "test"},
    )


def test_frame_format_version_identifies_compact_contract() -> None:
    assert FRAME_FORMAT_VERSION == 2


def test_constructed_frame_is_valid() -> None:
    frame = _frame()

    assert is_valid_standard_mpc_frame(frame)
    assert validate_standard_mpc_frame(frame, raise_on_error=False) == []


def test_validation_rejects_mapping_instead_of_compatibility_conversion() -> None:
    candidate = {"frame_index": 0}

    assert not is_valid_standard_mpc_frame(candidate)
    with pytest.raises(ValueError, match="complete StandardMPCFrame"):
        validate_standard_mpc_frame(candidate)


def test_explicit_validation_detects_mutated_numpy_buffers() -> None:
    frame = _frame()
    frame.interactions[0] = np.uint8(0)

    errors = validate_standard_mpc_frame(frame, raise_on_error=False)

    assert errors == ["interactions must contain positive physical-bounce codes"]


def test_direct_constructor_requires_canonical_dtypes() -> None:
    frame = _frame()
    values = {
        field: getattr(frame, field) for field in frame.__dataclass_fields__ if field != "version"
    }
    values["tx_positions"] = frame.tx_positions.astype(np.float32)

    with pytest.raises(ValueError, match="tx_positions must use dtype float64"):
        StandardMPCFrame(**values)
