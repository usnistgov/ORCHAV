"""Site-subset comparisons on explicit observation supports, without RF imports.

Values are received power in dBm, not SINR. Authenticated empty observations
(-infinity) fail finite thresholds. Unknown and invalid values retain separate
codes and contribute lower/upper bounds over the same eligible denominator.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .availability import CoverageAvailability

MAX_ENUMERATED_SUBSETS = 255


def site_subsets(site_names, subsets=None):
    """Resolve explicit named subsets; enumerate only a bounded small site set."""
    names = tuple(str(name) for name in site_names)
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("Site names must be unique and nonempty")
    if subsets is None:
        if len(names) > 8:
            raise ValueError("More than eight sites requires explicit selected subsets")
        result = [
            tuple(s) for n in range(1, len(names) + 1) for s in itertools.combinations(names, n)
        ]
    else:
        result = [tuple(str(name) for name in subset) for subset in subsets]
        if not result or len(result) > MAX_ENUMERATED_SUBSETS:
            raise ValueError("Select between 1 and 255 subsets")
        if any(not s or len(set(s)) != len(s) or not set(s) <= set(names) for s in result):
            raise ValueError("Each subset needs distinct known site identities")
        if len(set(result)) != len(result):
            raise ValueError("Repeated subsets are ambiguous")
    return result


def weighted_subset_summary(
    rss_dbm,
    site_names,
    *,
    subsets=None,
    thresholds_dbm,
    weights=None,
    eligible=None,
    availability=None,
):
    """Return named subset × threshold bounds, counts and explicit support.

    Weights normalize on eligible samples. The denominator never silently drops
    unknowns. A subset passes if any known site passes; its upper bound also
    passes if at least one selected site is unknown/invalid. No interpolation
    between samples or independence of samples is implied.
    """
    values = np.asarray(rss_dbm, dtype=float)
    names = tuple(str(n) for n in site_names)
    selected = site_subsets(names, subsets)
    thresholds = np.asarray(thresholds_dbm, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("Received powers must have sample × site shape")
    if thresholds.ndim != 1 or not thresholds.size or not np.isfinite(thresholds).all():
        raise ValueError("Thresholds must be a nonempty finite vector in dBm")
    if np.isposinf(values).any():
        raise ValueError("Positive infinite received power is invalid")
    codes = (
        np.where(np.isnan(values), 0, np.where(np.isneginf(values), 2, 1))
        if availability is None
        else np.asarray(availability)
    )
    if codes.shape != values.shape or not np.isin(codes, [0, 1, 2, 3]).all():
        raise ValueError("Availability must use coverage codes 0, 1, 2, 3")
    if np.any((codes == 1) & ~np.isfinite(values)) or np.any((codes == 2) & ~np.isneginf(values)):
        raise ValueError("Positive and empty availability must agree with power values")
    mask = np.ones(len(values), dtype=bool) if eligible is None else np.asarray(eligible)
    if mask.shape != (len(values),) or mask.dtype.kind != "b":
        raise ValueError("Eligibility must be a Boolean sample vector")
    w = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != mask.shape or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Weights must be finite nonnegative sample weights")
    mass = float(w[mask].sum())
    if mask.any() and mass <= 0:
        raise ValueError("Eligible observations have zero weight")
    normalized = np.where(mask, w / mass, 0) if mass else np.zeros(len(values))
    lower, upper, counts, upper_counts, unknown_counts = [], [], [], [], []
    known = np.isin(codes, [1, 2])
    safe = np.where(known, values, -np.inf)
    for subset in selected:
        ix = [names.index(name) for name in subset]
        passing = np.max(safe[:, ix], axis=1)[:, None] >= thresholds
        unknown = (~known[:, ix]).any(axis=1)
        possible = passing | unknown[:, None]
        lower.append(normalized @ passing if mass else np.full(len(thresholds), np.nan))
        upper.append(normalized @ possible if mass else np.full(len(thresholds), np.nan))
        counts.append((passing & mask[:, None]).sum(axis=0))
        upper_counts.append((possible & mask[:, None]).sum(axis=0))
        unknown_counts.append(int(np.sum(unknown & mask)))
    return {
        "subsets": [list(s) for s in selected],
        "thresholds_dbm": thresholds,
        "fraction_lower": np.asarray(lower),
        "fraction_upper": np.asarray(upper),
        "count_lower": np.asarray(counts),
        "count_upper": np.asarray(upper_counts),
        "unknown_samples_by_subset": np.asarray(unknown_counts),
        "eligible_samples": int(mask.sum()),
        "total_samples": len(values),
        "eligible_weight": mass,
        "availability_counts_by_site": {
            str(int(code)): ((codes == code) & mask[:, None]).sum(axis=0)
            for code in CoverageAvailability
        },
        "policy": "Fixed eligible denominator; zero fails finite thresholds; unknown/invalid bracket lower and upper bounds",
    }


def file_identity(path):
    """Stream a source digest without loading native recordings into memory."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def json_value(value):
    """Convert NumPy values and unavailable scalar values into strict JSON."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def _validate_bundle(data):
    rss = np.asarray(data["rss_dbm"], float)
    names, groups = data["site_names"].tolist(), data["group_names"].tolist()
    if rss.ndim != 3 or rss.shape[0] != len(groups) or rss.shape[2] != len(names):
        raise ValueError("Bundle requires group × sample × site power and named axes")
    if len(set(groups)) != len(groups):
        raise ValueError("Group names must be unique")
    for key in ("eligible", "weights"):
        if np.shape(data[key]) != rss.shape[:2]:
            raise ValueError(f"{key} must use group × sample axes")
    if np.shape(data["availability"]) != rss.shape:
        raise ValueError("Availability shape differs from received powers")
    for i in range(len(groups)):
        weighted_subset_summary(
            rss[i],
            names,
            subsets=[names],
            thresholds_dbm=[0],
            weights=data["weights"][i],
            eligible=data["eligible"][i],
            availability=data["availability"][i],
        )
    metadata = json.loads(str(data["metadata_json"]))
    for field in ("quantity", "receiver_model", "support", "weighting", "provenance"):
        if field not in metadata or not metadata[field]:
            raise ValueError(f"Bundle requires explicit {field}")
    if "common_mask" in data:
        common = data["common_mask"]
        if common.dtype.kind != "b" or common.shape != rss.shape[1:2]:
            raise ValueError("Common mask must align identical sample columns across groups")
        if not metadata.get("common_columns_aligned"):
            raise ValueError("Common support requires explicit aligned-column semantics")
        if np.any(common & ~np.all(data["eligible"], axis=0)):
            raise ValueError("Common mask includes ineligible columns")
    return metadata


def save_sample_bundle(
    path,
    *,
    rss_dbm,
    site_names,
    group_names,
    eligible,
    weights,
    availability,
    metadata,
    common_mask=None,
):
    """Write a small canonical, portable analysis input from a native provider."""
    data = {
        "rss_dbm": np.asarray(rss_dbm, float),
        "site_names": np.asarray(site_names),
        "group_names": np.asarray(group_names),
        "eligible": np.asarray(eligible),
        "weights": np.asarray(weights, float),
        "availability": np.asarray(availability),
        "metadata_json": np.asarray(json.dumps(metadata, allow_nan=False)),
    }
    if common_mask is not None:
        data["common_mask"] = np.asarray(common_mask)
    _validate_bundle(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_sample_bundle(path):
    """Load the portable input without Sionna, CUDA, widgets or pickle."""
    with np.load(path, allow_pickle=False) as source:
        data = {k: np.array(source[k]) for k in source.files}
    _validate_bundle(data)
    return data


def native_coverage_bundle(path, *, eligible=None, common_mask=None, step_index=0, roi_bounds=None):
    """Read a saved native map; optional masks refer to recorded grid columns."""
    import h5py

    from .availability import validate_availability
    from .hdf5 import CoverageHDF5Reader

    gains = CoverageHDF5Reader(path).read_path_gain(step_index=step_index)[0].astype(float)
    with h5py.File(path, "r") as h5:
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in h5["tx/names"][:]]
        powers = np.asarray(h5["tx/powers_dbm"], float)
        heights = np.asarray(h5["grid/heights_m"], float)
        origin = np.asarray(h5["grid/origin"], float) if roi_bounds is not None else None
        spacing = np.asarray(h5["grid/spacing_xy"], float) if roi_bounds is not None else None
        if "availability/path_gain" in h5:
            codes = validate_availability(
                np.asarray(h5["availability/path_gain"][step_index]), gains
            )
        else:
            codes = np.where(np.isfinite(gains) & (gains > 0), 1, 0)
    gain = np.where(codes == 2, 0.0, gains)
    with np.errstate(divide="ignore", invalid="ignore"):
        rss = powers[None, :, None, None] + 10 * np.log10(gain)
    nz, nt, ny, nx = rss.shape
    data = {
        "rss_dbm": rss.transpose(0, 2, 3, 1).reshape(nz, ny * nx, nt),
        "availability": codes.transpose(0, 2, 3, 1).reshape(nz, ny * nx, nt),
        "site_names": np.asarray(names),
        "group_names": np.asarray([f"{h:g} m" for h in heights]),
        "eligible": (
            np.ones((nz, ny * nx), bool)
            if eligible is None
            else np.asarray(eligible).reshape(nz, ny * nx)
        ),
        "weights": np.ones((nz, ny * nx)),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "quantity": "Native area-mean incoherent path power at saved TX power (dBm)",
                    "receiver_model": "Native radio-map ideal total-transverse-field response; not the point RX array",
                    "support": "Recorded grid cells at discrete heights; equal eligible cell weights",
                    "weighting": "Equal samples within each selected plane",
                    "common_columns_aligned": True,
                    "provenance": {
                        "path": str(Path(path).resolve()),
                        **file_identity(path),
                        "step_index": step_index,
                    },
                }
            )
        ),
    }
    if roi_bounds is not None:
        from .spatial import coverage_roi_mask

        axes = (
            origin[0] + (np.arange(nx) + 0.5) * spacing[0],
            origin[1] + (np.arange(ny) + 0.5) * spacing[1],
            heights,
        )
        selected = coverage_roi_mask(axes, roi_bounds).reshape(nz, ny * nx)
        data["eligible"] = data["eligible"] & selected
        # Keep column identity while omitting heights outside the chosen band.
        heights_selected = selected.any(axis=1)
        for key in ("rss_dbm", "availability", "eligible", "weights", "group_names"):
            data[key] = data[key][heights_selected]
        metadata = json.loads(str(data["metadata_json"]))
        metadata["roi_bounds_m"] = np.asarray(roi_bounds).tolist()
        metadata["support"] += "; restricted to inclusive XYZ cell-centre bounds"
        data["metadata_json"] = np.asarray(json.dumps(metadata))
        if common_mask is not None:
            common_mask = np.asarray(common_mask).reshape(ny * nx) & np.all(
                data["eligible"], axis=0
            )
    if common_mask is not None:
        data["common_mask"] = np.asarray(common_mask).reshape(ny * nx)
    _validate_bundle(data)
    return data


def analyze_definition(definition, *, base_dir="."):
    """Evaluate a versioned saved study from authenticated source descriptors."""
    if definition.get("schema_version") != 1 or definition.get("kind") != "coverage_comparison":
        raise ValueError("Unsupported coverage comparison definition")
    if not definition.get("sources"):
        raise ValueError("Select at least one comparison source")
    rows, identities = [], []
    subsets = definition.get("subsets")
    common_sites = None
    labels = set()
    for source in definition["sources"]:
        path = (Path(base_dir) / source["path"]).resolve()
        identity = file_identity(path)
        if source.get("sha256") and source["sha256"] != identity["sha256"]:
            raise ValueError(f"Comparison source changed: {path}")
        kind = source.get("kind", "sample_bundle")
        if kind == "sample_bundle":
            data = load_sample_bundle(path)
        elif kind == "native_coverage":
            mask_args = {}
            if source.get("masks_path"):
                mask_path = (Path(base_dir) / source["masks_path"]).resolve()
                mask_identity = file_identity(mask_path)
                if source.get("masks_sha256") != mask_identity["sha256"]:
                    raise ValueError("External support mask changed or is not authenticated")
                with np.load(mask_path, allow_pickle=False) as masks:
                    mask_args = {
                        k: np.array(masks[k]) for k in ("eligible", "common_mask") if k in masks
                    }
            data = native_coverage_bundle(
                path,
                step_index=source.get("step_index", 0),
                roi_bounds=source.get("roi_bounds_m"),
                **mask_args,
            )
        else:
            raise ValueError(f"Unsupported source kind: {kind}")
        names = data["site_names"].tolist()
        if common_sites is not None and set(names) != set(common_sites):
            raise ValueError("Sources must identify the same transmitter set")
        common_sites = names
        selected = site_subsets(names, subsets)
        if subsets is None:
            subsets = selected
        groups = data["group_names"].tolist()
        chosen = source.get("groups", groups)
        if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(groups):
            raise ValueError("Select known, distinct source groups")
        support = source.get("support", "per_group")
        if support not in ("per_group", "common_columns"):
            raise ValueError("Unknown support policy")
        if support == "common_columns" and "common_mask" not in data:
            raise ValueError("Source has no qualified common-column mask")
        group_results = []
        metadata = json.loads(str(data["metadata_json"]))
        for group in chosen:
            i = groups.index(group)
            mask = data["common_mask"] if support == "common_columns" else data["eligible"][i]
            result = weighted_subset_summary(
                data["rss_dbm"][i],
                names,
                subsets=selected,
                thresholds_dbm=definition["thresholds_dbm"],
                weights=data["weights"][i],
                eligible=mask,
                availability=data["availability"][i],
            )
            group_results.append(result)
            if not source.get("pool_groups", False):
                label = f"{source['label']} / {group}"
                if label in labels:
                    raise ValueError("Comparison labels must be distinct")
                labels.add(label)
                rows.append(
                    {
                        "label": label,
                        "source": source["label"],
                        "group": group,
                        "support": support,
                        "metadata": metadata,
                        **result,
                    }
                )
        if source.get("pool_groups", False):
            w = np.asarray(source.get("group_weights", np.ones(len(chosen))), float)
            if (
                w.shape != (len(chosen),)
                or not np.isfinite(w).all()
                or np.any(w < 0)
                or w.sum() <= 0
            ):
                raise ValueError("Group weights must be nonnegative with positive total")
            w /= w.sum()
            if any(r["eligible_samples"] == 0 for r in group_results):
                raise ValueError("Cannot pool a group with empty support")
            label = source["label"]
            if label in labels:
                raise ValueError("Comparison labels must be distinct")
            labels.add(label)
            rows.append(
                {
                    "label": label,
                    "source": label,
                    "groups": chosen,
                    "group_weights": w,
                    "support": support,
                    "metadata": metadata,
                    "subsets": [list(s) for s in selected],
                    "thresholds_dbm": np.asarray(definition["thresholds_dbm"]),
                    "fraction_lower": np.tensordot(
                        w, [r["fraction_lower"] for r in group_results], axes=1
                    ),
                    "fraction_upper": np.tensordot(
                        w, [r["fraction_upper"] for r in group_results], axes=1
                    ),
                    "eligible_samples": sum(r["eligible_samples"] for r in group_results),
                    "group_summaries": group_results,
                    "policy": "Normalize weights within each group, then explicit group weights; counts are per-group, not implied pooled independent trials",
                }
            )
        identities.append({"path": str(path), **identity, "metadata": metadata})
    return {
        "schema_version": 1,
        "kind": "coverage_comparison_result",
        "definition": definition,
        "sources": identities,
        "rows": rows,
        "comparison": "Sources retain distinct support and receiver estimands; common units alone do not establish equal measurement models",
    }
