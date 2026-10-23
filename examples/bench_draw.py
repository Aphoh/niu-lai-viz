#!/usr/bin/env python3
"""Per-chart draw-time benchmark at realistic run counts, with a phase breakdown.

Needs `pip install playwright` and a Chrome with CDP (default 127.0.0.1:29229),
plus `serve --root stress` with runs from synth_runs.py --prefix fifty.

    python3 examples/bench_draw.py --runs 20,50 --reps 200
"""
import argparse, asyncio, json, statistics
from playwright.async_api import async_playwright

KEYS = ["trainer/train_loss", "trainer/train_grad_norm", "trainer/train_step",
        "trainer/synthetic_metric_00000 [gpu=0]", "trainer/synthetic_metric_00001 [gpu=1]", "trainer/synthetic_metric_00002 [gpu=2]"]

MEASURE = r"""async ({reps}) => {
  const P = ["rasterGrid", "rasterLines", "blit", "labels"], acc = {draw: []};
  for (const p of P) acc[p] = [];
  const proto = Object.getPrototypeOf(niu.charts.values().next().value), orig = {};
  for (const p of P) { orig[p] = proto[p]; proto[p] = function (...a) { const t = performance.now(); const r = orig[p].apply(this, a); acc[p].push(performance.now() - t); return r; }; }
  const cs = [...niu.charts.values()].filter((c) => c.data);
  for (let i = 0; i < 20; i++) for (const c of cs) c.drawNow();       // warm up JIT
  for (const k in acc) acc[k].length = 0;
  for (let i = 0; i < reps; i++) for (const c of cs) { const t = performance.now(); c.drawNow(); acc.draw.push(performance.now() - t); }
  // live-update storm: every visible chart redraws at once through the frame-budgeted queue
  const frames = [];
  for (let i = 0; i < 30; i++) {
    niu.frameWork.length = 0;
    for (const c of cs) c.draw();
    await new Promise((res) => { const poll = () => (niu.frameWork.length && !document.hidden && cs.every(() => true) && niu.frameWork.length >= 1 ? setTimeout(res, 60) : requestAnimationFrame(poll)); poll(); });
    frames.push(...niu.frameWork);
  }
  for (const p of P) proto[p] = orig[p];
  // hover: run the rAF callback synchronously and time it
  const c = cs[0], hov = [], r = c.over.getBoundingClientRect(), raf = window.requestAnimationFrame;
  window.requestAnimationFrame = (cb) => { cb(performance.now()); return 0; };
  for (let i = 0; i < 300; i++) {
    c.mouse = { x: c.W * 0.15 + (c.W * 0.8) * (i % 100) / 100, y: c.H / 2 }; c.hoverRaf = 0;
    const t = performance.now(); c.hover(); hov.push(performance.now() - t);
  }
  window.requestAnimationFrame = raf; c.mouse = null;
  const pts = cs.map((c) => c.data.series.reduce((n, s) => n + s.xs.length, 0));
  return { acc, frames, hov, charts: cs.length, size: [cs[0].W, cs[0].H], pts };
}"""

def dist(xs):
    xs = sorted(xs)
    if not xs: return "-"
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return f"{q(.5):6.2f} {q(.95):6.2f} {xs[-1]:6.2f}"

async def bench(browser, url, runs, reps, vw, vh):
    ctx = await browser.new_context(viewport={"width": vw, "height": vh}, device_scale_factor=2)
    page = await ctx.new_page()
    await page.goto(url + "/api/runs")
    names = sorted(r["name"] for r in json.loads(await page.inner_text("body")) if r["name"].startswith("fifty"))[:runs]
    await page.evaluate("""([p, k]) => { localStorage.setItem('niu.picked', JSON.stringify(p)); localStorage.setItem('niu.panels', JSON.stringify(k)); }""", [names, KEYS])
    await page.goto(url)
    await page.wait_for_function("niu.charts.size === 6 && [...niu.charts.values()].every((c) => c.data && c.data.series.length > 0)", timeout=60000)
    res = await page.evaluate(MEASURE, {"reps": reps})
    await ctx.close()
    print(f"\n== {runs} runs, {res['charts']} charts, canvas {res['size'][0]}x{res['size'][1]} px (viewport {vw}x{vh}@2x), points/chart {min(res['pts'])}-{max(res['pts'])}")
    print(f"{'phase':12s}    p50    p95    max  (ms per chart)")
    for k, v in res["acc"].items():
        print(f"{k:12s} {dist(v)}")
    print(f"{'frame work':12s} {dist(res['frames'])}  (chart work per rAF when all charts redraw at once)")
    print(f"{'hover':12s} {dist(res['hov'])}  (one hover update)")

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdp", default="http://127.0.0.1:29229")
    ap.add_argument("--url", default="http://127.0.0.1:8738")
    ap.add_argument("--runs", default="20,50")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--viewport", default="1440x900,1920x1080,2560x1440")
    a = ap.parse_args()
    async with async_playwright() as pw:
        browser = await pw.chromium.connect_over_cdp(a.cdp)
        for vp in a.viewport.split(","):
            vw, vh = map(int, vp.split("x"))
            for n in map(int, a.runs.split(",")):
                await bench(browser, a.url, n, a.reps, vw, vh)

asyncio.run(main())
