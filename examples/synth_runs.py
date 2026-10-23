#!/usr/bin/env python3
"""Write synthetic finished runs straight to Parquet for UI stress tests.

    python3 examples/synth_runs.py --root stress --runs 1000 --keys 100 --points 1000
"""
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_run(args):
    root, name, n_keys, n_points, seed, gpus = args
    rng = np.random.default_rng(seed)
    d = Path(root) / name
    (d / "segments").mkdir(parents=True, exist_ok=True)
    warmup = rng.uniform(5, 120)  # variable weight-load time before training starts
    t = warmup + np.arange(n_points, dtype=np.float64) * rng.uniform(0.5, 2.0)
    step = np.arange(n_points, dtype=np.float64)
    lr = 10 ** rng.uniform(-4.5, -2)
    names, values, times, gpu_col = [], [], [], []

    def add(metric, v, ts, gpu=""):
        names.append(np.full(len(v), metric, dtype=object)); values.append(v); times.append(ts)
        gpu_col.append(np.full(len(v), gpu, dtype=object))

    base_keys = 3
    floor = 0.3 + 0.2 * (np.log10(lr) + 3.2) ** 2
    loss = floor + 2.5 * np.exp(-step * lr * 5) + rng.normal(0, 0.03, n_points).cumsum() * 0.01 + np.abs(rng.normal(0, 0.05, n_points))
    add("train_step", step, t)
    add("train_loss", loss, t)
    add("train_grad_norm", 0.5 + rng.lognormal(-2, 1, n_points), t)
    pre = np.linspace(0, warmup, 8)
    add("weights_loaded_bytes", np.linspace(0, 1.4e11, 8), pre)  # only during warmup
    for k in range(max(0, n_keys - base_keys - 1)):
        g = k % gpus
        phase, freq = rng.uniform(0, 6.28), rng.uniform(0.002, 0.05)
        kind = k // gpus
        v = 50 + 40 * np.sin(t * freq + phase) + rng.normal(0, 5, n_points)
        if kind % 3 == 1:
            v = np.cumsum(np.abs(v))
        add(f"synthetic_metric_{kind:05d}", v, t, str(g))
    table = pa.table({
        "scraper_endpoint": pa.array(np.full(sum(len(x) for x in values), "trainer", dtype=object), pa.string()),
        "metric_name": pa.array(np.concatenate(names), pa.string()),
        "metric_value": pa.array(np.concatenate(values), pa.float64()),
        "histogram_bucket_lower": pa.nulls(sum(len(x) for x in values), pa.float64()),
        "histogram_bucket_upper": pa.nulls(sum(len(x) for x in values), pa.float64()),
        "histogram_sum": pa.nulls(sum(len(x) for x in values), pa.float64()),
        "histogram_count": pa.nulls(sum(len(x) for x in values), pa.float64()),
        "time_since_start": pa.array(np.concatenate(times), pa.float64()),
        "timestamp_ns": pa.array((np.concatenate(times) * 1e9).astype(np.int64), pa.int64()),
        "gpu": pa.array(np.concatenate(gpu_col), pa.string()),
    })
    table = table.sort_by([("metric_name", "ascending"), ("gpu", "ascending"), ("time_since_start", "ascending")])
    pq.write_table(table, d / "final.parquet", compression="zstd", row_group_size=1 << 20)
    now = int(time.time() * 1000)
    meta = {"name": name, "labels": {"lr": f"{lr:.2e}", "seed": str(seed), "optimizer": ["adamw", "lion", "sgd"][seed % 3],
            "model": ["tiny", "small", "base"][seed % 5 % 3]}, "status": "finished", "started_at_ms": now - n_points * 1000 - seed,
            "heartbeat_ms": now, "finished_at_ms": now, "save_interval_secs": 2, "host": "synth", "targets": []}
    (d / "run.json").write_text(json.dumps(meta))
    return table.num_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="stress")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--keys", type=int, default=100, help="series per run")
    ap.add_argument("--points", type=int, default=1000, help="samples per series")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--prefix", default="synth")
    a = ap.parse_args()
    jobs = [(a.root, f"{a.prefix}-{i:05d}", a.keys, a.points, i, a.gpus) for i in range(a.runs)]
    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        rows = sum(ex.map(write_run, jobs, chunksize=4))
    print(f"{a.runs} runs, {rows:,} rows in {time.time() - t0:.1f}s")
