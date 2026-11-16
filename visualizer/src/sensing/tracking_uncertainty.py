"""Pure saved Cartesian model-covariance geometry; no tracking or file access.

Principal-axis scales are standard deviations, not calibrated confidence levels
or measured error. Missing dimensions and covariance are never synthesized.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

_POSITION_LABELS = ("x_m", "y_m", "z_m")
_EPOCH_TOLERANCE_S = 1e-9


@dataclass(frozen=True)
class CovarianceEllipsoid:
    """World-coordinate principal axes of one saved model covariance."""

    center: np.ndarray
    basis: np.ndarray
    radii: np.ndarray
    covariance: np.ndarray
    track_id: object
    status: str


def _array(value):
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError("Expected real numeric values")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Expected finite values")
    return array


def _frozen(array):
    """Own immutable bytes, rather than sharing mutable saved-state buffers."""
    array = np.ascontiguousarray(array, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _symmetric_psd(matrix):
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10):
        raise ValueError("Saved covariance is not symmetric")
    symmetric = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    tolerance = max(
        1e-9,
        32 * np.finfo(np.float64).eps * len(matrix) * float(np.max(np.abs(eigenvalues))),
    )
    if eigenvalues.min() < -tolerance:
        raise ValueError("Saved covariance is not positive semidefinite")
    return symmetric, np.maximum(eigenvalues, 0.0), eigenvectors


def covariance_ellipsoid(state, *, sigma_scale=1.0, receiver_position=None, receiver_rotation=None):
    """Return ``(ellipsoid, reason)``; invalid/unavailable states fail closed.

    The positional marginal is selected by explicit coordinate labels. Local
    states require the receiver pose bound to that exact saved result. World
    states are already transformed and must not be rotated a second time.
    ``sigma_scale`` multiplies each principal standard deviation. Zero axes
    remain zero, including singular but valid positive-semidefinite covariances.
    """
    try:
        if not isinstance(state, Mapping):
            raise ValueError("Saved track state is unavailable")
        scale = _array(sigma_scale)
        if scale.shape != () or float(scale) <= 0:
            raise ValueError("Principal-axis scale must be a positive finite scalar")
        labels = tuple(state.get("state_labels", ()))
        if any(not isinstance(label, str) for label in labels) or len(set(labels)) != len(labels):
            raise ValueError("Saved state coordinate labels must be unique")
        if not all(label in labels for label in _POSITION_LABELS):
            raise ValueError("Complete saved Cartesian x/y/z state is unavailable")
        if state.get("covariance") is None:
            raise ValueError("Saved track covariance was not recorded")
        mean = _array(state.get("mean"))
        covariance = _array(state["covariance"])
        if mean.shape != (len(labels),) or covariance.shape != (len(labels), len(labels)):
            raise ValueError("Saved state and covariance dimensions do not match their labels")
        # Validate the complete retained covariance, not just its diagonal or
        # positional corner. Check the positional marginal independently too.
        covariance, _, _ = _symmetric_psd(covariance)
        indices = [labels.index(label) for label in _POSITION_LABELS]
        center = mean[indices]
        positional, _, _ = _symmetric_psd(covariance[np.ix_(indices, indices)])
        frame = state.get("position_frame")
        if frame == "receiver local":
            if receiver_position is None or receiver_rotation is None:
                raise ValueError("The saved result's receiver pose is unavailable")
            origin, rotation = _array(receiver_position), _array(receiver_rotation)
            if origin.shape != (3,) or rotation.shape != (3, 3):
                raise ValueError("Receiver pose requires a 3D position and 3×3 rotation")
            if not np.allclose(
                rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0
            ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9, rtol=0):
                raise ValueError("Receiver orientation is not a proper rotation")
            center = origin + rotation @ center
            positional = rotation @ positional @ rotation.T
        elif frame != "world":
            raise ValueError("Saved Cartesian coordinate frame is unavailable or unsupported")
        positional, values, basis = _symmetric_psd(positional)
        # A proper basis is convenient for mesh transforms. Eigenvector signs
        # have no scientific significance and do not alter the ellipsoid.
        if np.linalg.det(basis) < 0:
            basis[:, -1] *= -1
        radii = float(scale) * np.sqrt(values)
        if not np.isfinite(center).all() or not np.isfinite(radii).all():
            raise ValueError("World covariance geometry is not finite")
        return (
            CovarianceEllipsoid(
                center=_frozen(center),
                basis=_frozen(basis),
                radii=_frozen(radii),
                covariance=_frozen(positional),
                track_id=state.get("track_id"),
                status=str(state.get("status", "observation unknown")),
            ),
            "",
        )
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError) as exc:
        return None, str(exc)


def native_epoch_valid(sensing, time_s):
    """Require diagnostics, result and requested epoch to agree exactly.

    Source identity remains the controller's responsibility. This check never
    replaces acquisition/pair/run binding and does not select a nearest epoch.
    """
    try:
        if not isinstance(sensing, Mapping):
            raise ValueError("Saved sensing result is unavailable")
        diagnostics = (
            sensing.get("detection_filter_metadata", {}).get("tracker", {}).get("decisions")
        )
        if not isinstance(diagnostics, Mapping) or diagnostics.get("schema_version") != 1:
            raise ValueError("Saved tracking diagnostics were not recorded or are unsupported")
        reference = sensing.get("temporal", {}).get("result", {}).get("reference_time_s")
        values = [_array(value) for value in (time_s, reference, diagnostics.get("update_time_s"))]
        if any(value.shape != () for value in values):
            raise ValueError("Saved tracking epoch must be a finite scalar")
        requested, result, update = (float(value) for value in values)
        if (
            abs(requested - result) > _EPOCH_TOLERANCE_S
            or abs(update - result) > _EPOCH_TOLERANCE_S
        ):
            raise ValueError("Saved tracker covariance is not bound to this exact result epoch")
        return True, ""
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        return False, str(exc)
