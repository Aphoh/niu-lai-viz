#!/usr/bin/env python3
"""Spin up a zoo of fake jobs, scrape each into its own run, and serve the viewer.

Some runs finish, one crashes (goes stale), the rest stay live until Ctrl+C.
    python3 examples/demo.py --bin target/release/niu-lai-viz --root runs
"""
import argparse, os, signal, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fake_exporter import FakeJob, serve  # noqa: E402

RUNS = [
    # name, lr, seed, speed, lifetime_s (None = live, negative = SIGKILL)
    ("baseline-lr3e-4", 3e-4, 1, 40, 50),
    ("lowlr-1e-4", 1e-4, 2, 40, 50),
    ("hotlr-3e-3", 3e-3, 3, 40, 50),
    ("crashy-lr6e-4", 6e-4, 4, 40, -30),
    ("live-lr3e-4-bs128", 3e-4, 5, 15, None),
    ("live-lr1e-3", 1e-3, 6, 15, None),
    ("live-yolo-1e-2", 1e-2, 7, 15, None),
]

ap = argparse.ArgumentParser()
ap.add_argument("--bin", default="target/debug/niu-lai-viz")
ap.add_argument("--root", default="runs")
ap.add_argument("--base-port", type=int, default=9101)
ap.add_argument("--addr", default="127.0.0.1:8737", help="where the viewer listens")
ap.add_argument("--no-serve", action="store_true", help="only scrape; run `niu-lai-viz serve` yourself")
a = ap.parse_args()
cfg = Path(__file__).parent / "demo.toml"

server = None
if not a.no_serve:
    server = subprocess.Popen([a.bin, "serve", "--root", a.root, "--addr", a.addr], stderr=subprocess.DEVNULL)

procs = []
for i, (name, lr, seed, speed, life) in enumerate(RUNS):
    port = a.base_port + i
    serve(port, FakeJob(lr, seed, speed))
    cmd = [a.bin, "scrape", "--config", str(cfg), "--root", a.root, "--run", name,
           "--target", f"trainer=http://127.0.0.1:{port}/metrics", "--interval-ms", "500",
           "--label", f"lr={lr:g}", "--label", f"seed={seed}",
           "--label", f"batch={128 if 'bs128' in name else 64}",
           "--label", f"optimizer={'adamw' if i % 2 == 0 else 'lion'}"]
    procs.append((name, life, time.time(), subprocess.Popen(cmd, stderr=subprocess.DEVNULL)))
    print(f"{name}: :{port}", flush=True)

if server:
    bar = "=" * 60
    print(f"\n{bar}\n  niu-lai-viz is live:  http://{a.addr}\n  open it in a browser. Ctrl+C stops everything.\n{bar}\n", flush=True)

try:
    while any(life is not None for _, life, _, _ in procs):
        time.sleep(1)
        for entry in list(procs):
            name, life, t0, p = entry
            if life is not None and time.time() - t0 > abs(life):
                p.send_signal(signal.SIGKILL if life < 0 else signal.SIGTERM)
                p.wait()
                print(f"{name}: {'killed' if life < 0 else 'finished'}", flush=True)
                procs.remove(entry)
    if server:
        print(f"timed runs done; live runs keep going. viewer: http://{a.addr}", flush=True)
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    for _, _, _, p in procs:
        p.send_signal(signal.SIGTERM)
    for _, _, _, p in procs:
        p.wait()
finally:
    if server:
        server.terminate()
