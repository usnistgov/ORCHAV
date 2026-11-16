"""Strict schema-v4 selection of persisted range-Doppler-angle products.

Angular consumers use this module instead of compatibility arrays such as
``range_doppler_azimuth_cube``.  The selector binds a producer-authored
``[pair, elevation, azimuth, Doppler, range]`` power product to its exact
direction-cosine and broadside-angle coordinates without reconstructing axes
from configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from shared.sensing.arrays import forward_hemisphere_angles
from shared.sensing.products import select_product_by_contract

_PRODUCT_KIND = "range_doppler_angle"
_POWER_REPRESENTATION = "power_linear"
_AXIS_NAMES = (
    "pair",
    "elevation_direction_cosine",
    "azimuth_direction_cosine",
    "doppler_frequency",
    "range",
)


@dataclass(frozen=True)
class RangeDopplerAngleProduct:
    """One exact single-pair angular power product and its persisted axes."""

    data_key: str
    product_id: str
    pair_id: str
    variant: str
    power_linear_by_pair: np.ndarray
    elevation_direction_cosine: np.ndarray
    azimuth_direction_cosine: np.ndarray
    elevation_broadside_angle_deg: np.ndarray
    azimuth_broadside_angle_deg: np.ndarray
    doppler_axis_hz: np.ndarray
    range_axis_m: np.ndarray
    range_coordinate_semantics: str
    descriptor: Mapping[str, Any]

    @property
    def power_linear(self) -> np.ndarray:
        """Return the single pair as ``[elevation, azimuth, Doppler, range]``."""

        return self.power_linear_by_pair[0]

    @property
    def forward_power_linear(self) -> np.ndarray:
        """Return display power with nonphysical and exact-endfire bins masked.

        The persisted tensor and axes remain intact for numerical replay.
        """
        visible, _, _ = forward_hemisphere_angles(
            self.elevation_direction_cosine, self.azimuth_direction_cosine
        )
        return np.where(visible[:, :, None, None], self.power_linear, 0.0)


def _descriptors_for_variant(sensing: Mapping[str, Any], variant: str) -> list[Mapping[str, Any]]:
    products = sensing.get("products")
    if not isinstance(products, Mapping):
        return []
    return [
        descriptor
        for descriptor in products.values()
        if isinstance(descriptor, Mapping)
        and descriptor.get("kind") == _PRODUCT_KIND
        and descriptor.get("variant") == variant
    ]


def has_range_doppler_angle_variant(sensing: Mapping[str, Any], variant: str) -> bool:
    """Check saved descriptor availability without loading an angular cube.

    Presence does not validate the dense contract. The ordinary selector still
    checks the advertised product when its data is loaded.
    """
    return bool(_descriptors_for_variant(sensing, variant))


def _validate_axis_contract(descriptor: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    axes = descriptor.get("axes")
    if not isinstance(axes, list) or len(axes) != len(_AXIS_NAMES):
        raise ValueError("Typed range_doppler_angle has an invalid ordered axis contract")
    if not all(isinstance(axis, Mapping) for axis in axes):
        raise ValueError("Typed range_doppler_angle axes must be mappings")
    typed_axes = list(axes)

    expected_core = (
        ("pair", "1", "catalog", "pair_axis"),
        (
            "elevation_direction_cosine",
            "1",
            "dense",
            "elevation_direction_cosine",
        ),
        (
            "azimuth_direction_cosine",
            "1",
            "dense",
            "azimuth_direction_cosine",
        ),
        ("doppler_frequency", "Hz", "dense", "doppler_axis_hz"),
        ("range", "m", "dense", "range_axis_m"),
    )
    observed_core = tuple(
        (
            axis.get("name"),
            axis.get("unit"),
            axis.get("coordinate_kind"),
            axis.get("coordinate_key"),
        )
        for axis in typed_axes
    )
    if observed_core != expected_core:
        raise ValueError(
            "Typed range_doppler_angle axes must use the exact persisted pair, "
            "direction-cosine, Doppler-Hz, and range-m coordinates"
        )

    expected_semantics = (
        ("ordered_pair_identity", "none", ()),
        (
            "rx_array_local_z_direction_cosine",
            "rx_array_local",
            ("elevation_broadside_angle_deg",),
        ),
        (
            "rx_array_local_y_direction_cosine",
            "rx_array_local",
            ("azimuth_broadside_angle_deg",),
        ),
        ("fftshifted_doppler_frequency", "none", ()),
    )
    observed_semantics = tuple(
        (
            axis.get("coordinate_semantics"),
            axis.get("reference_frame"),
            tuple(axis.get("alternate_coordinate_keys", ())),
        )
        for axis in typed_axes[:4]
    )
    if observed_semantics != expected_semantics:
        raise ValueError(
            "Typed range_doppler_angle angular axes must declare exact RX-array-local "
            "direction-cosine semantics and persisted broadside-degree alternates"
        )

    range_axis = typed_axes[4]
    range_semantics = range_axis.get("coordinate_semantics")
    if (
        range_semantics not in {"monostatic_range", "bistatic_path_length"}
        or range_axis.get("reference_frame") != "tx_target_rx_path"
        or tuple(range_axis.get("alternate_coordinate_keys", ()))
    ):
        raise ValueError(
            "Typed range_doppler_angle range coordinates require exact monostatic-range "
            "or bistatic-path-length semantics"
        )
    return typed_axes


def _validate_pair_identity(
    sensing: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    pair_coordinates: np.ndarray | list[Any],
    *,
    requested_pair_id: str | None,
) -> str:
    if not isinstance(pair_coordinates, list) or len(pair_coordinates) != 1:
        raise ValueError("Typed range_doppler_angle requires exactly one pair axis row")
    pair_row = pair_coordinates[0]
    if not isinstance(pair_row, Mapping):
        raise ValueError("Typed range_doppler_angle pair coordinates must be mappings")
    if pair_row.get("pair_index") != 0 or pair_row.get("frame_pair_index") != 0:
        raise ValueError("Typed range_doppler_angle requires exact pair index zero")

    pair_id = pair_row.get("pair_id")
    tx_sensor_id = pair_row.get("tx_sensor_id")
    rx_sensor_id = pair_row.get("rx_sensor_id")
    if not all(isinstance(value, str) and value for value in (pair_id, tx_sensor_id, rx_sensor_id)):
        raise ValueError("Typed range_doppler_angle pair coordinates lack sensor identities")
    if requested_pair_id is not None and pair_id != requested_pair_id:
        raise ValueError(
            "Typed range_doppler_angle pair does not match the requested pair identity"
        )
    if descriptor.get("pair_ids") != [pair_id]:
        raise ValueError("Typed range_doppler_angle descriptor pair_ids mismatch its pair axis")
    descriptor_pair = descriptor.get("pair")
    expected_pair = {
        "pair_id": pair_id,
        "tx_sensor_id": tx_sensor_id,
        "rx_sensor_id": rx_sensor_id,
    }
    if not isinstance(descriptor_pair, Mapping) or any(
        descriptor_pair.get(key) != value for key, value in expected_pair.items()
    ):
        raise ValueError("Typed range_doppler_angle descriptor pair identity is inconsistent")

    temporal = sensing.get("temporal")
    acquisition = temporal.get("acquisition") if isinstance(temporal, Mapping) else None
    if not isinstance(acquisition, Mapping) or any(
        acquisition.get(key) != value for key, value in expected_pair.items()
    ):
        raise ValueError(
            "Typed range_doppler_angle pair identity differs from temporal acquisition"
        )
    return pair_id


def _dense_coordinate(
    value: Any,
    *,
    expected_size: int,
    label: str,
    allow_descending: bool = False,
) -> np.ndarray:
    coordinate = np.asarray(value)
    if (
        coordinate.ndim != 1
        or coordinate.size != expected_size
        or coordinate.dtype.kind not in "fiu"
        or not np.all(np.isfinite(coordinate))
    ):
        raise ValueError(
            f"Typed range_doppler_angle {label} coordinates must be a finite real "
            f"vector of length {expected_size}"
        )
    differences = np.diff(coordinate.astype(np.float64))
    if coordinate.size > 1 and not (
        np.all(differences > 0.0) or (allow_descending and np.all(differences < 0.0))
    ):
        raise ValueError(
            f"Typed range_doppler_angle {label} coordinates must be strictly "
            + ("monotonic" if allow_descending else "increasing")
        )
    return np.array(coordinate, copy=True)


def _broadside_coordinate(
    value: Any, direction_cosine: np.ndarray, *, label: str, allow_descending: bool = False
) -> np.ndarray:
    """Read exact saved degrees, allowing NaN only at nonphysical FFT bins."""
    coordinate = np.asarray(value)
    if coordinate.shape != direction_cosine.shape or coordinate.dtype.kind not in "fiu":
        raise ValueError(f"Typed range_doppler_angle {label} must match its direction-cosine axis")
    physical = np.abs(direction_cosine) <= 1.0
    if not np.all(np.isnan(coordinate[~physical])):
        raise ValueError(f"Typed range_doppler_angle {label} must be NaN at nonphysical FFT bins")
    _dense_coordinate(
        coordinate[physical],
        expected_size=int(np.count_nonzero(physical)),
        label=label,
        allow_descending=allow_descending,
    )
    return np.array(coordinate, copy=True)


def select_range_doppler_angle_product(
    sensing: Mapping[str, Any],
    *,
    variant: str = "primary",
    pair_id: str | None = None,
) -> RangeDopplerAngleProduct | None:
    """Select one exact angular product, or ``None`` when the variant is absent.

    Once a descriptor advertises the requested kind and variant, every
    representation, axis, coordinate, dimension, and pair mismatch is an
    explicit error.  Legacy arrays and reconstructed coordinates are never
    consulted.
    """

    if not isinstance(variant, str) or not variant:
        raise ValueError("Typed angular-product selection requires a non-empty variant")
    if pair_id is not None and (not isinstance(pair_id, str) or not pair_id):
        raise ValueError("Typed angular-product pair_id must be a non-empty string")
    if not has_range_doppler_angle_variant(sensing, variant):
        return None
    try:
        data_key, values, coordinates, descriptor = select_product_by_contract(
            sensing,
            kind=_PRODUCT_KIND,
            variant=variant,
            axis_names=_AXIS_NAMES,
            representations=(_POWER_REPRESENTATION,),
        )
    except KeyError as exc:
        raise ValueError(
            "Persisted range_doppler_angle exists but does not satisfy the exact "
            "P,E,A,D,R power contract"
        ) from exc

    axes = _validate_axis_contract(descriptor)
    if values.ndim != 5 or values.shape[0] != 1:
        raise ValueError(
            "Typed range_doppler_angle must have shape [1, elevation, azimuth, "
            f"Doppler, range]; received {values.shape}"
        )
    if values.dtype.kind not in "fiu" or not np.all(np.isfinite(values)):
        raise ValueError("Typed range_doppler_angle values must be finite real power")
    if np.any(values < 0):
        raise ValueError("Typed range_doppler_angle power cannot be negative")

    pair_identity = _validate_pair_identity(
        sensing,
        descriptor,
        coordinates[0],
        requested_pair_id=pair_id,
    )
    elevation_u = _dense_coordinate(
        coordinates[1],
        expected_size=int(values.shape[1]),
        label="elevation direction-cosine",
        # The producer negates the row FFT coordinate because Sionna rows
        # run downwards in local z. Preserve that decreasing coordinate order.
        allow_descending=True,
    )
    azimuth_u = _dense_coordinate(
        coordinates[2],
        expected_size=int(values.shape[2]),
        label="azimuth direction-cosine",
    )
    doppler_axis_hz = _dense_coordinate(
        coordinates[3],
        expected_size=int(values.shape[3]),
        label="Doppler",
    )
    range_axis_m = _dense_coordinate(
        coordinates[4],
        expected_size=int(values.shape[4]),
        label="range",
    )

    # Compact arrays retain the full FFT grid, including nonphysical bins.
    # Broadside alternates identify those bins with NaN; consumers mask them.
    elevation_degrees = _broadside_coordinate(
        sensing.get("elevation_broadside_angle_deg"),
        elevation_u,
        label="elevation broadside-angle",
        allow_descending=True,
    )
    azimuth_degrees = _broadside_coordinate(
        sensing.get("azimuth_broadside_angle_deg"),
        azimuth_u,
        label="azimuth broadside-angle",
    )

    product_id = descriptor.get("product_id")
    if not isinstance(product_id, str) or not product_id:
        raise ValueError("Typed range_doppler_angle descriptor has no product_id")
    return RangeDopplerAngleProduct(
        data_key=data_key,
        product_id=product_id,
        pair_id=pair_identity,
        variant=variant,
        power_linear_by_pair=np.array(values, copy=True),
        elevation_direction_cosine=elevation_u,
        azimuth_direction_cosine=azimuth_u,
        elevation_broadside_angle_deg=elevation_degrees,
        azimuth_broadside_angle_deg=azimuth_degrees,
        doppler_axis_hz=doppler_axis_hz,
        range_axis_m=range_axis_m,
        range_coordinate_semantics=str(axes[4]["coordinate_semantics"]),
        descriptor=deepcopy(descriptor),
    )


__all__ = [
    "RangeDopplerAngleProduct",
    "has_range_doppler_angle_variant",
    "select_range_doppler_angle_product",
]
