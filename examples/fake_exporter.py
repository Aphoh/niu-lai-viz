#!/usr/bin/env python3
"""Fake training job exposing Prometheus /metrics with noisy, spiky, plausible curves.

    python3 fake_exporter.py --port 9101 --lr 3e-4 --seed 1 --speed 40
"""
import argparse, bisect, math, random, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STEP_BUCKETS = [float(f"{0.05 * 1.35**k:.3g}") for k in range(16)]  # 50ms .. ~4.5s
# vLLM's default buckets (vllm/v1/metrics/buckets.py), so the rollout histograms look like a real server's.
TTFT_BUCKETS = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
                20.0, 40.0, 80.0, 160.0, 640.0, 2560.0]
ITL_BUCKETS = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0]


class Hist:
    def __init__(self, bounds):
        self.bounds, self.counts, self.sum = bounds, [0] * (len(bounds) + 1), 0.0

    def observe(self, v):
        self.counts[bisect.bisect_left(self.bounds, v)] += 1
        self.sum += v

    def render(self, name, help_):
        out, cum = [f"# HELP {name} {help_}\n# TYPE {name} histogram"], 0
        for b, c in zip(self.bounds + ["+Inf"], self.counts):
            cum += c
            out.append(f'{name}_bucket{{le="{b}"}} {cum}')
        out.append(f"{name}_sum {self.sum:.4f}\n{name}_count {cum}")
        return out


class FakeJob:
    def __init__(self, lr, seed, speed, gpus=4, batch=64, extra=0):
        self.lr, self.speed, self.gpus, self.batch = lr, speed, gpus, batch
        # Stress mode: `extra` more gauges, like a big model's per-layer stats.
        self.extra = [(f'layer_stat{{layer="{i // 8}",stat="s{i % 8}"}}', 1 + i % 7, i * 0.37) for i in range(extra)]
        self.rng = random.Random(seed)
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.step = 0
        self.loss = 2.8 + self.rng.random() * 0.4
        self.temp = [55 + self.rng.random() * 10 for _ in range(gpus)]
        self.step_h, self.ttft_h, self.itl_h = Hist(STEP_BUCKETS), Hist(TTFT_BUCKETS), Hist(ITL_BUCKETS)
        self.step_base = 0.18 * (batch / 64) ** 0.8 * (1 + 0.15 * self.rng.random())
        self.straggler = seed % 3 == 1  # one slow rank every so often: a second band in the step-time heatmap
        self.slowdown_at = self.rng.uniform(25, 45) if lr > 1e-3 else None  # sim seconds
        self.req_t = 0.0
        self.tokens = 0
        self.eval_acc = 0.0
        self.spike = 0.0
        self.diverge_at = 1500 + self.rng.randint(0, 800) if lr > 1e-3 else None

    def schedule(self, step):
        warm = 200
        if step < warm:
            return self.lr * step / warm
        p = min(1.0, (step - warm) / 6000)
        return self.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    def advance(self):
        target = int((time.time() - self.t0) * self.speed)
        r = self.rng
        while self.step < target:
            self.step += 1
            s = self.step
            floor = 0.35 + 0.15 * math.log10(3e-4 / self.lr) ** 2
            self.loss += (floor - self.loss) * self.schedule(s) * 12 + r.gauss(0, 0.004)
            if r.random() < 0.002:
                self.spike = r.uniform(0.3, 1.2)
            self.spike *= 0.97
            if self.diverge_at and s > self.diverge_at:
                self.loss *= 1.0 + 0.002 * r.random()
            self.step_h.observe(self.step_time(s))
            self.tokens += self.batch * 2048
            if s % 500 == 0:
                self.eval_acc = min(0.92, max(0.0, 1.05 - self.loss * 0.6 + r.gauss(0, 0.02)))
        self.rollouts(time.time() - self.t0)
        return self.step

    def step_time(self, s):
        """Seconds for step s: drifting base, eval stalls, checkpoint tail, stragglers, a late slowdown."""
        r, t = self.rng, s / self.speed  # simulated seconds, so the shape doesn't depend on --speed
        mu, sig = self.step_base * (1 + 0.25 * min(1.0, t / 120)), 0.18
        if self.slowdown_at and t > self.slowdown_at:
            mu, sig = mu * 1.7, 0.4
        dt = r.lognormvariate(math.log(mu), sig)
        if s % 300 < 15:
            dt *= 3.5  # eval
        if s % 1000 < 3:
            dt *= 12  # checkpoint
        if self.straggler and t % 40 < 12 and r.random() < 0.5:
            dt *= 2.2
        if r.random() < 0.01:
            dt += r.expovariate(1 / 0.4)  # data loader hiccup
        return dt

    def rollouts(self, now):
        """RL-style generation requests: load swings over a ~60s cycle; TTFT queues up under load."""
        r = self.rng
        while self.req_t < now:
            load = 0.5 + 0.45 * math.sin(self.req_t * 2 * math.pi / 60 + self.lr * 1e4)
            rate = 6 + 18 * load
            self.req_t += r.expovariate(rate)
            queue = r.expovariate(1 / (0.02 + 4 * load**6)) if r.random() < 0.3 + 0.6 * load else 0.0
            self.ttft_h.observe(r.lognormvariate(math.log(0.03 * (self.batch / 64)), 0.5) + queue)
            itl = 0.012 * (1 + 1.5 * load) * (self.batch / 64) ** 0.5
            for _ in range(12):
                v = r.lognormvariate(math.log(itl), 0.25)
                if r.random() < 0.004 * (1 + 4 * load):
                    v *= r.uniform(10, 60)  # preemption
                self.itl_h.observe(v)

    def render(self):
        with self.lock:
            s = self.advance()
            r, t = self.rng, time.time() - self.t0
            evaluating = s % 300 < 15
            out = []

            def m(name, typ, help_, samples):
                out.append(f"# HELP {name} {help_}\n# TYPE {name} {typ}")
                for labels, v in samples:
                    lab = ",".join(f'{k}="{v_}"' for k, v_ in labels.items())
                    out.append(f"{name}{{{lab}}} {v}" if lab else f"{name} {v}")

            loss = self.loss + self.spike + abs(r.gauss(0, 0.02))
            m("train_loss", "gauge", "Training loss", [({}, f"{loss:.5f}")])
            m("train_step", "counter", "Optimizer steps", [({}, s)])
            m("train_learning_rate", "gauge", "LR", [({}, f"{self.schedule(s):.3e}")])
            gn = 0.6 + 2 * math.exp(-s / 800) + r.lognormvariate(-2, 1.0) + self.spike * 8
            m("train_grad_norm", "gauge", "Grad norm", [({}, f"{gn:.4f}")])
            m("train_tokens_total", "counter", "Tokens seen", [({}, self.tokens)])
            m("eval_accuracy", "gauge", "Eval accuracy", [({"split": "val"}, f"{self.eval_acc:.4f}")])
            util, mem, temp, power = [], [], [], []
            for g in range(self.gpus):
                u = 10 + r.random() * 5 if evaluating else 78 + 15 * math.sin(t / 6 + g) + r.gauss(0, 4)
                util.append(({"gpu": str(g)}, f"{max(0, min(100, u)):.1f}"))
                mem.append(({"gpu": str(g)}, int((40 + 30 * ((t / 20 + g * 0.25) % 1)) * 2**30)))
                self.temp[g] += (60 + u * 0.25 - self.temp[g]) * 0.1 + r.gauss(0, 0.5)
                temp.append(({"gpu": str(g)}, f"{self.temp[g]:.1f}"))
                power.append(({"gpu": str(g)}, f"{150 + u * 4 + r.gauss(0, 20):.0f}"))
            m("accel_utilization_percent", "gauge", "Accelerator busy %", util)
            m("accel_memory_used_bytes", "gauge", "Accelerator memory", mem)
            m("accel_temperature_celsius", "gauge", "Accelerator temp", temp)
            m("accel_power_watts", "gauge", "Accelerator power", power)
            out += self.step_h.render("train_step_seconds", "Optimizer step time")
            out += self.ttft_h.render("rollout_time_to_first_token_seconds", "Rollout TTFT")
            out += self.itl_h.render("rollout_inter_token_latency_seconds", "Rollout ITL (12 tokens sampled per request)")
            out.append(f"process_resident_memory_bytes {int(2e9 + r.random() * 1e8)}")
            if self.extra:
                out.append("# TYPE layer_stat gauge")
                out += [f"{n} {sc * (1 + 0.3 * math.sin(t / 5 + ph)) + r.random() * 0.05:.4f}" for n, sc, ph in self.extra]
            return "\n".join(out) + "\n"


def serve(port, job):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = job.render().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9101)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=20, help="optimizer steps per wall second")
    a = ap.parse_args()
    serve(a.port, FakeJob(a.lr, a.seed, a.speed))
    print(f"fake job on http://127.0.0.1:{a.port}/metrics", flush=True)
    while True:
        time.sleep(3600)
