"""Supervise disjoint native trial workers; retain their logs beside the results."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scenarios.sensing.volume_progress import VolumeProgress
from scenarios.sensing.volume_tools import write_json


def run_parallel(root, gpus, jobs_per_gpu, condition=None, limit=None, *, script=None):
    """Assign every trial to exactly one process, including on resumed runs."""
    workers = len(gpus) * jobs_per_gpu
    logs = root / "worker_logs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    logs.mkdir(parents=True)
    if (root / "volume_recipe.json").is_file():
        cases = [root]
    elif (root / "study.json").is_file():
        study = json.loads((root / "study.json").read_text())
        cases = [root / row["name"] / c for row in study["conditions"] for c in ("wall", "open")]
    else:
        cases = [root / c for c in ((condition,) if condition else ("wall", "open"))]
    outcomes = []
    for case in cases:
        spec = json.loads((case / "volume_recipe.json").read_text())
        pending = [t for t in spec["trials"] if t["status"] == "pending"]
        for worker in range(workers):
            selected = pending[worker::workers]
            if limit is not None:
                selected = selected[:limit]
            outcomes.extend((case / t["native"]).parent / "outcome.json" for t in selected)
    saved = sum(path.is_file() for path in outcomes)
    with VolumeProgress("All workers", len(outcomes), completed=saved) as progress:
        children = []
        start = time.perf_counter()
        try:
            for worker in range(workers):
                gpu = gpus[worker % len(gpus)]
                command = [
                    sys.executable,
                    str(script or Path(__file__).with_name("generate.py")),
                    str(root),
                    "--worker",
                    str(worker),
                    "--workers",
                    str(workers),
                    "--yes",
                ]
                if condition:
                    command += ["--condition", condition]
                if limit is not None:
                    command += ["--limit", str(limit)]
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                    environment[name] = "1"
                log = logs / f"worker_{worker:02d}_gpu{gpu}.log"
                with log.open("w") as output:
                    process = subprocess.Popen(
                        command, env=environment, stdout=output, stderr=subprocess.STDOUT
                    )
                children.append((process, worker, gpu, log))
            progress.message(f"Started {workers} native workers; logs: {logs}")
            last_check = float("-inf")
            while any(process.poll() is None for process, *_ in children):
                failed = [
                    (worker, process.returncode, log)
                    for process, worker, _, log in children
                    if process.poll() not in (None, 0)
                ]
                if failed:
                    raise RuntimeError(
                        f"Native worker failed; completed acquisitions are preserved: {failed}"
                    )
                if time.monotonic() - last_check >= 5.0:
                    active = sum(process.poll() is None for process, *_ in children)
                    progress.update_workers(sum(path.is_file() for path in outcomes), active)
                    last_check = time.monotonic()
                time.sleep(0.5)
            progress.update_workers(sum(path.is_file() for path in outcomes), 0)
            if any(process.returncode for process, *_ in children):
                raise RuntimeError(f"Native worker failed; inspect {logs}")
        finally:
            for process, *_ in children:
                if process.poll() is None:
                    process.terminate()
            for process, *_ in children:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            write_json(
                logs / "workers.json",
                {
                    "elapsed_seconds": time.perf_counter() - start,
                    "workers": [
                        {
                            "worker": worker,
                            "gpu": gpu,
                            "returncode": process.returncode,
                            "log": log.name,
                        }
                        for process, worker, gpu, log in children
                    ],
                    "condition": condition,
                    "limit_per_worker_per_condition": limit,
                },
            )
