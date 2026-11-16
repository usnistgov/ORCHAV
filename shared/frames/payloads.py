"""Stable names for optional ``StandardMPCFrame.sensing`` array products.

The packed HDF5 codec separates top-level numeric extension values from small
JSON metadata and restores them to the same ``sensing`` mapping. The dense-key
inventory documents ORCHAV's standard product vocabulary; it is not a
restrictive whitelist for additional numeric products.
"""

from __future__ import annotations

FRAME_EXTENSION_DENSE_HDF5_MIN_SCHEMA_VERSION = 2

# Long-form frame payload key -> short detection-stage artifact field.
FRAME_EXTENSION_STAGE_ARTIFACT_DATASET_ALIASES = {
    "range_doppler_cfar_input_power_db": "cfar_input_power_db",
    "range_doppler_cfar_mask": "cfar_mask",
    "range_doppler_cfar_noise_floor": "cfar_noise_floor",
    "range_doppler_nms_mask": "nms_mask",
    "range_doppler_snr_map": "snr_map",
    "range_doppler_cluster_labels": "cluster_labels",
}

# Recognized top-level ndarray products in sensing payloads.
FRAME_EXTENSION_DENSE_DATASET_KEYS = (
    "cir",
    "cir_by_pair_rx_element",
    "slow_time_matrix",
    "slow_time_by_pair_rx_element",
    "slow_time_axis_s",
    "range_profile",
    "range_profile_by_pair",
    "range_profile_raw_by_pair",
    "range_profile_raw",
    "range_profile_baseline",
    "range_doppler_map",
    "range_doppler_by_pair",
    "range_doppler_raw_by_pair",
    "range_doppler_power_by_pair",
    "range_doppler_raw_power_by_pair",
    "range_doppler_angle_by_pair",
    "range_doppler_map_raw",
    "range_doppler_aoa_az_map",
    "range_doppler_aoa_el_map",
    "range_doppler_azimuth_cube",
    "range_profile_cfar_mask",
    "range_doppler_detection_input_power_by_pair",
    "range_doppler_cfar_noise_power_by_pair",
    "range_doppler_cfar_threshold_power_by_pair",
    "range_doppler_cfar_training_cell_count_by_pair",
    "range_doppler_cfar_eligible_mask_by_pair",
    "range_doppler_cfar_mask_by_pair",
    "range_doppler_nms_mask_by_pair",
    "range_doppler_snr_db_by_pair",
    "range_doppler_snr_valid_mask_by_pair",
    "range_doppler_cluster_labels_by_pair",
    # Legacy untyped detector aliases remain recognized only for sidecar and
    # pre-v4 compatibility. Native v4 publication strips them unless a typed
    # descriptor explicitly binds the array.
    "range_doppler_cfar_mask",
    "range_doppler_cfar_input_power_db",
    "range_doppler_cfar_noise_floor",
    "range_doppler_nms_mask",
    "range_doppler_snr_map",
    "range_doppler_cluster_labels",
    "range_axis_m",
    "doppler_axis_hz",
    "azimuth_direction_cosine",
    "elevation_direction_cosine",
    "azimuth_broadside_angle_deg",
    "elevation_broadside_angle_deg",
)

__all__ = [
    "FRAME_EXTENSION_DENSE_DATASET_KEYS",
    "FRAME_EXTENSION_DENSE_HDF5_MIN_SCHEMA_VERSION",
    "FRAME_EXTENSION_STAGE_ARTIFACT_DATASET_ALIASES",
]
