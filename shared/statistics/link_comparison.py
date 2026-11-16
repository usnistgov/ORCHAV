"""Identity-selected link traces and matched-state communication comparisons.

This CPU analysis uses the existing frame provider and measured path-loss
contract. It keeps authenticated empty pairs (zero gain) separate from absent
pairs and invalid path metrics. It never infers communication service quality.
Run ``python -m shared.statistics.link_comparison --help`` for native extraction
and saved comparison exports. Plot composition is in ``link_plots``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from shared.statistics.core.metrics import pathloss_db_to_power_linear

SCHEMA = "orchav.link_recording.v1"
COMPARISON_SCHEMA = "orchav.link_comparison.v1"
# Present empty pairs are observed zeros; missing/invalid values remain unknown.
POSITIVE, EMPTY, MISSING, INVALID = 0, 1, 2, 3
STATUS_NAMES = ("positive", "empty", "missing", "invalid")


def file_identity(path: Path) -> dict[str, Any]:
    """Hash one source without retaining its complete contents in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


@dataclass(frozen=True)
class LinkRecording:
    """Arrays in sample × transmitter × receiver order, with stable names."""

    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    @property
    def tx_names(self) -> tuple[str, ...]:
        return tuple(self.metadata["tx_names"])

    @property
    def rx_names(self) -> tuple[str, ...]:
        return tuple(self.metadata["rx_names"])

    def pair(self, tx: str, rx: str) -> tuple[np.ndarray, np.ndarray]:
        """Select by recorded names; unknown identities raise explicitly."""
        ti, ri = self.tx_names.index(tx), self.rx_names.index(rx)
        return self.arrays["rss_dbm"][:, ti, ri], self.arrays["status"][:, ti, ri]


def reduce_link_frames(
    frames, frame_ids, tx_power_dbm: Mapping[str, float], *, time_binding=None
) -> LinkRecording:
    """Reduce canonical frames/projections without storage or Qt dependencies.

    ``time_binding`` is an explicit mapping of frame IDs to authored seconds.
    Missing native clocks otherwise fail; the unavailable clock remains stored.
    Any unavailable path loss invalidates its pair rather than reducing a partial
    sum. A represented zero-path pair has gain 0 and RSS -inf.
    """
    from shared.frames.contracts import PathMetric

    ids = np.asarray(frame_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0 or len(set(ids.tolist())) != len(ids):
        raise ValueError("Frame IDs must be nonempty and unique")
    binding = None if time_binding is None else {int(k): float(v) for k, v in time_binding.items()}
    if binding is not None and set(binding) != set(ids.tolist()):
        raise ValueError("Authored time binding must identify every selected frame exactly")
    rows = []
    tx_names = rx_names = None
    powers = None
    for fid, item in zip(ids, frames, strict=True):
        frame = getattr(item, "frame", item)
        if int(frame.frame_index) != int(fid):
            raise ValueError("Projection frame identity differs from requested state")
        names = tuple(frame.tx_names), tuple(frame.rx_names)
        if tx_names is None:
            tx_names, rx_names = names
            if (
                any(not n for n in tx_names + rx_names)
                or len(set(tx_names)) != len(tx_names)
                or len(set(rx_names)) != len(rx_names)
            ):
                raise ValueError("Nonempty unique TX and RX names are required")
            if not tx_names or not rx_names or set(tx_power_dbm) != set(tx_names):
                raise ValueError("Explicit TX power must match all recorded transmitter identities")
            powers = np.asarray([tx_power_dbm[n] for n in tx_names], float)
            if not np.isfinite(powers).all():
                raise ValueError("TX powers must be finite dBm")
        elif names != (tx_names, rx_names):
            raise ValueError("Device identity/order changes between frames")
        nt, nr = len(tx_names), len(rx_names)
        gain = np.full((nt, nr), np.nan)
        status = np.full((nt, nr), MISSING, np.uint8)
        counts = np.full((nt, nr), -1, np.int64)
        losses = np.asarray(frame.path_loss_db, float)
        valid = np.asarray(frame.metric_is_valid(PathMetric.PATH_LOSS_DB), bool)
        for p, (tx, rx) in enumerate(frame.tx_rx_pairs):
            tx, rx = int(tx), int(rx)
            if not 0 <= tx < nt or not 0 <= rx < nr or status[tx, rx] != MISSING:
                raise ValueError("Duplicate or invalid TX/RX pair")
            a, b = map(int, frame.pair_path_offsets[p : p + 2])
            counts[tx, rx] = b - a
            if a == b:
                gain[tx, rx], status[tx, rx] = 0.0, EMPTY
            elif not valid[a:b].all() or not np.isfinite(losses[a:b]).all():
                status[tx, rx] = INVALID
            else:
                value = float(pathloss_db_to_power_linear(losses[a:b]).sum())
                if not np.isfinite(value) or value <= 0:
                    status[tx, rx] = INVALID
                else:
                    gain[tx, rx], status[tx, rx] = value, POSITIVE
        clock = float(frame.timestamp_s) if frame.timestamp_s is not None else np.nan
        when = clock if binding is None else binding[int(fid)]
        if not np.isfinite(when):
            raise ValueError(
                "Native timestamps unavailable; provide explicit authored frame/time binding"
            )
        if (
            binding is not None
            and np.isfinite(clock)
            and not np.isclose(clock, when, rtol=0, atol=1e-8)
        ):
            raise ValueError("Recorded and authored times disagree")
        rows.append(
            (
                gain,
                status,
                counts,
                np.asarray(frame.tx_positions, float),
                np.asarray(frame.rx_positions, float),
                clock,
                when,
            )
        )
    keys = (
        "gain_linear",
        "status",
        "path_count",
        "tx_positions_m",
        "rx_positions_m",
        "recorded_time_s",
        "time_s",
    )
    arrays = {key: np.asarray([r[i] for r in rows]) for i, key in enumerate(keys)}
    if len(ids) > 1 and np.any(np.diff(arrays["time_s"]) <= 0):
        raise ValueError("Sample times must be strictly increasing")
    if (
        not np.isfinite(arrays["tx_positions_m"]).all()
        or not np.isfinite(arrays["rx_positions_m"]).all()
    ):
        raise ValueError("Recorded device positions must be finite")
    arrays["frame_ids"] = ids
    with np.errstate(divide="ignore"):
        arrays["rss_dbm"] = powers[None, :, None] + 10 * np.log10(arrays["gain_linear"])
    distance = np.linalg.norm(np.diff(arrays["rx_positions_m"], axis=0), axis=-1)
    arrays["distance_m"] = np.concatenate(
        [np.zeros((1, len(rx_names))), np.cumsum(distance, axis=0)]
    )
    return LinkRecording(
        arrays,
        {
            "schema": SCHEMA,
            "tx_names": list(tx_names),
            "rx_names": list(rx_names),
            "tx_power_dbm": dict(tx_power_dbm),
            "quantity": "TX dBm + 10log10(sum retained path gains)",
            "reduction": "incoherent path-power sum",
            "status_codes": dict(enumerate(STATUS_NAMES)),
            "time_reference": (
                "explicit authored frame binding" if binding is not None else "recorded timestamps"
            ),
            "time_binding": binding,
        },
    )


def validate_recording(record: LinkRecording) -> None:
    """Reject malformed compact evidence before it can enter a comparison."""
    arrays = record.arrays
    tx, rx = record.tx_names, record.rx_names
    if not tx or not rx or any(not isinstance(n, str) or not n for n in tx + rx):
        raise ValueError("Recording requires nonempty TX/RX names")
    if len(set(tx)) != len(tx) or len(set(rx)) != len(rx):
        raise ValueError("Recording device names must be unique")
    required = {
        "frame_ids",
        "time_s",
        "recorded_time_s",
        "gain_linear",
        "rss_dbm",
        "status",
        "path_count",
        "tx_positions_m",
        "rx_positions_m",
        "distance_m",
    }
    if required - arrays.keys():
        raise ValueError("Recording arrays are incomplete")
    ids = arrays["frame_ids"]
    if (
        ids.ndim != 1
        or not len(ids)
        or ids.dtype.kind not in "iu"
        or len(set(ids.tolist())) != len(ids)
    ):
        raise ValueError("Recording frame IDs must be unique integers")
    n = len(ids)
    for key in ("gain_linear", "rss_dbm", "status", "path_count"):
        if arrays[key].shape != (n, len(tx), len(rx)):
            raise ValueError(f"Recording {key} must have sample × TX × RX shape")
    for key in ("time_s", "recorded_time_s"):
        if arrays[key].shape != (n,):
            raise ValueError("Recording clock shape differs from frame IDs")
    if not np.isfinite(arrays["time_s"]).all() or np.any(np.diff(arrays["time_s"]) <= 0):
        raise ValueError("Recording times must be finite and strictly increasing")
    for role, names in (("tx", tx), ("rx", rx)):
        positions = arrays[f"{role}_positions_m"]
        if positions.shape != (n, len(names), 3) or not np.isfinite(positions).all():
            raise ValueError("Recording endpoint positions are malformed")
    if arrays["distance_m"].shape != (n, len(rx)):
        raise ValueError("Recording distance shape differs from receiver identities")
    expected_distance = np.concatenate(
        [
            np.zeros((1, len(rx))),
            np.cumsum(np.linalg.norm(np.diff(arrays["rx_positions_m"], axis=0), axis=-1), axis=0),
        ]
    )
    if not np.allclose(arrays["distance_m"], expected_distance, rtol=0, atol=1e-9):
        raise ValueError("Recording distances disagree with endpoint samples")
    codes, gain, rss, counts = (
        arrays[k] for k in ("status", "gain_linear", "rss_dbm", "path_count")
    )
    if (
        codes.dtype.kind not in "iu"
        or not np.isin(codes, (POSITIVE, EMPTY, MISSING, INVALID)).all()
    ):
        raise ValueError("Recording has unsupported status codes")
    positive, empty, unknown = codes == POSITIVE, codes == EMPTY, codes >= MISSING
    if not (
        np.isfinite(gain[positive]).all()
        and (gain[positive] > 0).all()
        and np.isfinite(rss[positive]).all()
    ):
        raise ValueError("Positive observations require positive finite gains and dBm")
    if not (
        np.all(gain[empty] == 0)
        and np.isneginf(rss[empty]).all()
        and np.isnan(rss[unknown]).all()
        and np.isnan(gain[unknown]).all()
    ):
        raise ValueError("Status and empty/unknown observation values disagree")
    if (
        counts.dtype.kind not in "iu"
        or np.any(counts[positive] < 1)
        or np.any(counts[empty] != 0)
        or np.any(counts[codes == MISSING] != -1)
        or np.any(counts[codes == INVALID] < 0)
    ):
        raise ValueError("Path counts disagree with observation status")
    powers = record.metadata["tx_power_dbm"]
    if set(powers) != set(tx) or not np.isfinite(list(powers.values())).all():
        raise ValueError("Recording powers differ from named transmitters")
    with np.errstate(divide="ignore"):
        expected_rss = np.asarray([powers[name] for name in tx])[None, :, None] + 10 * np.log10(
            gain
        )
    if not np.allclose(rss, expected_rss, rtol=0, atol=1e-10, equal_nan=True):
        raise ValueError("Stored dBm disagrees with gain and transmitter power")


def load_link_recording(
    source: str | Path, *, time_binding=None, tx_power_dbm=None
) -> LinkRecording:
    """Read a native recording or an exported compact NPZ/JSON observation.

    Native power defaults come only from explicit named actor power fields in
    the recording's scenario. The clock never defaults to a guessed timeline.
    """
    source = Path(source).resolve()
    if source.suffix in (".npz", ".json"):
        metadata = json.loads(source.with_suffix(".json").read_text())
        if metadata.get("schema") != SCHEMA:
            raise ValueError("Not an ORCHAV link observation export")
        path = source.with_suffix(".npz")
        if file_identity(path)["sha256"] != metadata["arrays_sha256"]:
            raise ValueError("Link arrays do not match exported identity")
        with np.load(path, allow_pickle=False) as stored:
            arrays = {k: stored[k] for k in stored.files}
        record = LinkRecording(arrays, metadata)
        validate_recording(record)
        return record
    import yaml

    from shared.frames.contracts import FrameReadRequest, PathMetric
    from shared.frames.providers import Hdf5Provider

    root = (
        source.parent if source.name == "frames" or source.suffix in (".yaml", ".yml") else source
    )
    config_path = root / "scenario.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    if tx_power_dbm is None:
        actors = config.get("actors", {}).get("tx", [])
        if not actors or any("power_dbm" not in actor for actor in actors):
            raise ValueError("Supply explicit TX powers when not recorded in scenario.yaml")
        tx_power_dbm = {a["name"]: float(a["power_dbm"]) for a in actors}
    request = FrameReadRequest.for_metrics((PathMetric.PATH_LOSS_DB,), include_devices=True)
    with Hdf5Provider(root) as provider:
        ids = provider.list_frames()
        record = reduce_link_frames(
            provider.iter_frame_projections(ids, request),
            ids,
            tx_power_dbm,
            time_binding=time_binding,
        )
    # Native files are authenticated in the export rather than copied into it.
    files = [p for p in (root / "frames").rglob("*") if p.suffix in (".h5", ".hdf5", ".json")]
    rt = config.get("raytracing", {})
    record.metadata.update(
        source=str(root),
        source_files=[file_identity(p) for p in sorted(files)],
        scenario=file_identity(config_path) if config_path.exists() else None,
        carrier_frequency_hz=rt.get("carrier_frequency_hz"),
        receiver_model=rt.get("antenna", {}).get("rx"),
        transmitter_model=rt.get("antenna", {}).get("tx"),
        scene=config.get("scene"),
    )
    return record


def export_recording(record: LinkRecording, stem: str | Path) -> Path:
    """Write compact observations and a source/quantity manifest, no raw paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stem.with_suffix(".npz"), **record.arrays)
    meta = dict(record.metadata, arrays_sha256=file_identity(stem.with_suffix(".npz"))["sha256"])
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
    return stem.with_suffix(".npz")


def support_weights(coordinate: np.ndarray) -> np.ndarray:
    """Trapezoidal node weights without interpolating threshold crossings."""
    x = np.asarray(coordinate, float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or np.any(np.diff(x) < 0):
        raise ValueError("Support must be a nonempty finite nondecreasing vector")
    if len(x) == 1:
        return np.ones(1)
    dx = np.diff(x)
    w = np.r_[dx[0] / 2, (dx[:-1] + dx[1:]) / 2, dx[-1] / 2]
    if w.sum() <= 0:
        raise ValueError("Stationary support has no distance-weighted summary")
    return w / w.sum()


def site_selection(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strongest known site and margin; unknown sites preclude a winner.

    -1 means no observed site, -2 means unknown/invalid input. A lone positive
    site has infinite observed margin; it is not a finite interference margin.
    """
    values = np.asarray(values, float)
    if values.ndim != 2 or not values.shape[1]:
        raise ValueError("Expected sample by selected site values")
    unknown = np.isnan(values).any(axis=1) | np.isposinf(values).any(axis=1)
    ranked = np.sort(values, axis=1)
    positive = np.isfinite(ranked[:, -1])
    winner = np.where(positive, np.argmax(values, axis=1), -1)
    winner[unknown] = -2
    margin = np.full(len(values), np.nan)
    margin[positive & ~unknown] = np.inf
    if values.shape[1] > 1:
        two = np.isfinite(ranked[:, -2]) & ~unknown
        margin[two] = ranked[two, -1] - ranked[two, -2]
    return winner, margin


def paired_difference(
    left: LinkRecording, right: LinkRecording, tx: str, rx: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return right-minus-left dB on explicitly matched states and geometry."""
    for key in ("carrier_frequency_hz", "receiver_model", "transmitter_model", "scene"):
        if left.metadata.get(key) is None or left.metadata.get(key) != right.metadata.get(key):
            raise ValueError(f"Paired observations require matching declared {key}")
    if left.metadata["tx_power_dbm"][tx] != right.metadata["tx_power_dbm"][tx]:
        raise ValueError("Paired observations require identical TX powers")
    for key in ("frame_ids", "time_s"):
        if not np.array_equal(left.arrays[key], right.arrays[key]):
            raise ValueError("Comparison requires explicitly identical frame/time sample states")
    for role, name in (("tx", tx), ("rx", rx)):
        il, ir = getattr(left, f"{role}_names").index(name), getattr(right, f"{role}_names").index(
            name
        )
        if not np.allclose(
            left.arrays[f"{role}_positions_m"][:, il],
            right.arrays[f"{role}_positions_m"][:, ir],
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError("Paired sample positions differ")
    lv, ls = left.pair(tx, rx)
    rv, rs = right.pair(tx, rx)
    finite = (ls == POSITIVE) & (rs == POSITIVE)
    delta = np.full(lv.shape, np.nan)
    delta[finite] = rv[finite] - lv[finite]
    return delta, finite


def load_definition(path: str | Path) -> tuple[dict, list[LinkRecording]]:
    """Load explicit labelled source selections relative to a saved definition."""
    path = Path(path).resolve()
    definition = json.loads(path.read_text())
    if definition.get("schema") != COMPARISON_SCHEMA or not definition.get("sources"):
        raise ValueError("Expected a versioned nonempty link comparison definition")
    labels = [s["label"] for s in definition["sources"]]
    if len(labels) != len(set(labels)):
        raise ValueError("Comparison source labels must be unique")
    records = []
    for source in definition["sources"]:
        source_path = path.parent / source["path"]
        if source.get("sha256") and file_identity(source_path)["sha256"] != source["sha256"]:
            raise ValueError("Comparison source identity changed")
        if (
            source.get("metadata_sha256")
            and file_identity(source_path.with_suffix(".json"))["sha256"]
            != source["metadata_sha256"]
        ):
            raise ValueError("Comparison source metadata changed")
        records.append(
            load_link_recording(
                path.parent / source["path"],
                time_binding=source.get("time_binding"),
                tx_power_dbm=source.get("tx_power_dbm"),
            )
        )
    return definition, records


def analyze_comparison(definition: dict, records: list[LinkRecording]) -> dict:
    """Produce identity-selected traces and optional matched-state differences."""
    tx_names = tuple(definition["transmitters"])
    rx = definition["receiver"]
    axis = definition.get("axis", "time")
    if axis not in ("time", "distance") or not tx_names or len(set(tx_names)) != len(tx_names):
        raise ValueError("Select unique transmitters and time or distance axis")
    if len(records) != len(definition["sources"]):
        raise ValueError("Source count differs from definition")
    traces = []
    for info, record in zip(definition["sources"], records, strict=True):
        pairs = [record.pair(tx, rx) for tx in tx_names]
        values = np.stack([p[0] for p in pairs], axis=1)
        status = np.stack([p[1] for p in pairs], axis=1)
        x = (
            record.arrays["time_s"]
            if axis == "time"
            else record.arrays["distance_m"][:, record.rx_names.index(rx)]
        )
        winner, margin = site_selection(values)
        # A stationary distance trace is drawable but has no distance-weighted mean.
        weights = support_weights(x) if len(x) == 1 or x[-1] > x[0] else None
        traces.append(
            dict(
                label=info["label"],
                x=x,
                frame_ids=record.arrays["frame_ids"],
                values=values,
                status=status,
                winner=winner,
                margin_db=margin,
                weights=weights,
            )
        )
    result = dict(traces=traces, receiver=rx, transmitters=tx_names, axis=axis, differences=[])
    if definition.get("paired", False):
        for i in range(1, len(records)):
            deltas = [paired_difference(records[0], records[i], tx, rx)[0] for tx in tx_names]
            result["differences"].append(
                dict(
                    label=f"{traces[i]['label']} minus {traces[0]['label']}",
                    delta_db=np.stack(deltas, axis=1),
                )
            )
    return result


def export_comparison(definition: dict, records: list[LinkRecording], output: str | Path) -> dict:
    """Export the same analysis used by the native dialog, plus portable inputs."""
    import csv

    from .link_plots import comparison_figure

    result = analyze_comparison(definition, records)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for i, trace in enumerate(result["traces"]):
        arrays.update({f"source{i}_{k}": v for k, v in trace.items() if isinstance(v, np.ndarray)})
    for i, diff in enumerate(result["differences"]):
        arrays[f"difference{i}_db"] = diff["delta_db"]
    np.savez_compressed(output / "values.npz", **arrays)
    with (output / "traces.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "source",
                "frame_id",
                f"{result['axis']}_{'s' if result['axis']=='time' else 'm'}",
                "tx",
                "rx",
                "rss_dbm",
                "status",
                "best_site",
                "observed_margin_db",
            ]
        )
        for trace in result["traces"]:
            for row, fid in enumerate(trace["frame_ids"]):
                for col, tx in enumerate(result["transmitters"]):
                    winner = trace["winner"][row]
                    writer.writerow(
                        [
                            trace["label"],
                            int(fid),
                            trace["x"][row],
                            tx,
                            result["receiver"],
                            trace["values"][row, col],
                            STATUS_NAMES[int(trace["status"][row, col])],
                            (
                                result["transmitters"][winner]
                                if winner >= 0
                                else ("unknown" if winner == -2 else "none")
                            ),
                            trace["margin_db"][row],
                        ]
                    )
    portable = dict(definition, sources=[])
    for i, (item, record) in enumerate(zip(definition["sources"], records, strict=True)):
        path = export_recording(record, output / f"source{i}")
        portable["sources"].append(
            {
                "label": item["label"],
                "path": path.name,
                "sha256": file_identity(path)["sha256"],
                "metadata_sha256": file_identity(path.with_suffix(".json"))["sha256"],
            }
        )
    (output / "definition.json").write_text(json.dumps(portable, indent=2) + "\n")
    summary = {
        "schema": COMPARISON_SCHEMA,
        "definition": portable,
        "quantity": records[0].metadata["quantity"],
        "differences": [],
        "unknown_policy": "No paired dB difference for zero/missing/invalid pairs; no winner when any selected site is unknown",
        "sources": [r.metadata for r in records],
    }
    summary["trace_summaries"] = []
    for trace in result["traces"]:
        weights = trace["weights"]
        entries = []
        for col, tx in enumerate(result["transmitters"]):
            codes = trace["status"][:, col]
            known = (codes == POSITIVE) | (codes == EMPTY)
            mean = None
            if weights is not None and known.all():
                mean = float(weights @ (10.0 ** (trace["values"][:, col] / 10.0) * 0.001))
            entries.append(
                {
                    "tx": tx,
                    "counts": {
                        name: int(np.sum(codes == code)) for code, name in enumerate(STATUS_NAMES)
                    },
                    "unknown_weight": float(weights @ ~known) if weights is not None else None,
                    "mean_received_power_w": mean,
                }
            )
        summary["trace_summaries"].append(
            {
                "label": trace["label"],
                "weighting": f"Trapezoidal node weights on {result['axis']}; no interpolation of threshold crossings",
                "weighting_available": weights is not None,
                "pairs": entries,
            }
        )
    for item in result["differences"]:
        a = item["delta_db"]
        finite = a[np.isfinite(a)]
        summary["differences"].append(
            {
                "label": item["label"],
                "finite_pairs": int(len(finite)),
                "minimum_db": float(finite.min()) if len(finite) else None,
                "maximum_db": float(finite.max()) if len(finite) else None,
                "median_db": float(np.median(finite)) if len(finite) else None,
            }
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    fig = comparison_figure(result)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"comparison.{suffix}", dpi=170, bbox_inches="tight")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract", help="Extract a native identity-selected observation")
    extract.add_argument("source")
    extract.add_argument("output")
    extract.add_argument(
        "--time-binding", help="Explicit JSON frame-ID to authored-seconds mapping"
    )
    extract.add_argument("--tx-powers", help="JSON name-to-dBm mapping when scenario lacks powers")
    compare = sub.add_parser("compare", help="Run a saved native/portable comparison definition")
    compare.add_argument("definition")
    compare.add_argument("output")
    args = parser.parse_args(argv)
    if args.command == "extract":

        def read(path):
            return json.loads(Path(path).read_text()) if path else None

        record = load_link_recording(
            args.source, time_binding=read(args.time_binding), tx_power_dbm=read(args.tx_powers)
        )
        export_recording(record, args.output)
    else:
        definition, records = load_definition(args.definition)
        export_comparison(definition, records, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
