#!/usr/bin/env python3
"""Real-browser stress benchmark. Requires `pip install playwright`, existing CDP.

python examples/bench_ui.py --label baseline --out /tmp/niu-results
No browser download, mocked API, server restart, or repository commits.
Frames use synthetic DOM wheel+scroll events (one scroll increment per rAF);
they measure main-thread responsiveness, not compositor/trusted-wheel latency.
windowMouseup includes Playwright's single hit-target-interceptor listener;
appWindowMouseup excludes it. Read clearSyncMs, not clearSettledMs, for work.
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from playwright.async_api import async_playwright


INIT = r"""
localStorage.clear();
window.B = {phase:'navigation', notes:[], draws:[], tasks:[], hover:[], errors:[],
  active:0,maxActive:0, events:[], keyChanges:[]};
addEventListener('error',e=>B.errors.push(e.message));
new PerformanceObserver(l=>B.tasks.push(...l.getEntries().map(e=>({
  phase:B.phase,start:e.startTime,ms:e.duration})))).observe({type:'longtask'});
const raf = window.requestAnimationFrame;
window.requestAnimationFrame = function(cb) {
  const hover = cb.toString().includes('this.hoverRaf');
  return raf.call(window,t=>{const st=performance.now();cb(t);
    if(hover) B.hover.push({phase:B.phase,ms:performance.now()-st});});
};
addEventListener('DOMContentLoaded',()=>{
  const draw=Chart.prototype.draw, load=Chart.prototype.load, note=perf.note;
  Chart.prototype.draw=function(...args){const st=performance.now();
    const r=draw.apply(this,args);
    if(this.data) B.draws.push({phase:B.phase,key:this.key,ms:performance.now()-st,
      at:performance.now(),series:this.data.series.length,
      points:this.data.series.reduce((n,s)=>n+s.xs.length,0)});
    return r;};
  Chart.prototype.load=async function(...args){B.active++;B.maxActive=Math.max(B.maxActive,B.active);
    try{return await load.apply(this,args)}finally{B.active--}};
  perf.note=function(key,net,draw,bytes){B.notes.push({
    phase:B.phase,key,net,draw,bytes,at:performance.now()});
    return note.call(this,key,net,draw,bytes)};
  const kc=document.querySelector('#keycount');
  new MutationObserver(()=>B.keyChanges.push({at:performance.now(),text:kc.textContent}))
    .observe(kc,{childList:true,subtree:true,characterData:true});
});
"""

STATE = r"""() => ({
  at:performance.now(),picked:S.picked.size,keys:S.keys.length,shown:S.shownKeys.length,
  panels:charts.size,loaded:[...charts.values()].filter(c=>c.data).length,
  visible:[...charts.values()].filter(c=>c.visible).length,running:SCHED.running,
  maxActive:B.maxActive,hud:$('#perf').textContent,
  rows:$('#keys').querySelectorAll('.vrow').length,keyHeight:$('#keys').clientHeight,
  charts:[...charts.values()].filter(c=>c.visible).map(c=>({
    key:c.key,x0:c.data?.x0,x1:c.data?.x1,series:c.data?.series.length,
    width:c.W-M.l-M.r,maxSamples:c.data?Math.max(...c.data.series.map(s=>s.xs.length)):0,
    firstXs:c.data?.series.map(s=>s.xs[0]),
    firstYs:c.data?.series.slice(0,3).map(s=>Array.from(s.ys.slice(0,10))),
    dirty:c.dirty,loading:c.loading
  }))
})"""

SWEEP = r"""async ({kind,frames=180})=>{
  B.phase=kind;
  const el=document.querySelector(kind==='keys-scroll'?'#keys':'#panels');
  const gaps=[],work=[],labels=[];
  let prev;
  for(let i=0;i<frames;i++){
    const t=await new Promise(requestAnimationFrame);
    if(prev!==undefined)gaps.push(t-prev);prev=t;
    const st=performance.now();
    if(kind==='hover'){
      const c=[...charts.values()].find(c=>c.visible&&c.data);
      const r=c.over.getBoundingClientRect();
      c.over.dispatchEvent(new MouseEvent('mousemove',{bubbles:true,
        clientX:r.left+r.width*(.2+.75*i/(frames-1)),
        clientY:r.top+r.height*(.3+.35*Math.sin(i/10))}));
    } else {
      const delta=kind==='keys-scroll'?1800:350;
      el.dispatchEvent(new WheelEvent('wheel',{bubbles:true,deltaY:delta}));
      el.scrollTop+=delta;
      if(el.scrollTop+el.clientHeight>=el.scrollHeight-1)el.scrollTop=0;
      if(i%30===0)labels.push(el.querySelector('.vrow')?.textContent);
    }
    work.push(performance.now()-st);
  }
  return {gaps,work,labels,scrollTop:el.scrollTop};
}"""


def dist(xs):
    xs = sorted(xs)
    return {
        "n": len(xs),
        "p50": round(statistics.median(xs), 3) if xs else None,
        "p95": round(xs[min(len(xs)-1, int(len(xs)*.95))], 3) if xs else None,
        "max": round(max(xs), 3) if xs else None,
    }


def cpu_top(profile):
    nodes = {n["id"]: n for n in profile["nodes"]}
    times = {}
    for node, delta in zip(profile.get("samples", []), profile.get("timeDeltas", [])):
        call = nodes[node]["callFrame"]
        key = (call["functionName"] or "(anonymous)", call["url"], call["lineNumber"]+1)
        times[key] = times.get(key, 0) + delta
    total = sum(times.values()) or 1
    return [{"function": k[0], "url": k[1], "line": k[2],
             "self_ms": round(v/1000, 2), "pct": round(100*v/total, 2)}
            for k, v in sorted(times.items(), key=lambda kv: -kv[1])[:25]]


class Bench:
    def __init__(self, browser, args):
        self.browser, self.args = browser, args
        self.results = {"label": args.label, "url": args.url, "scenarios": {},
                        "environment": {"chrome": browser.version,
                                        "viewport": [1600, 1060],
                                        "cdp": args.cdp,
                                        "scroll": "synthetic DOM wheel + scrollTop per rAF"}}
        self.output = Path(args.out)
        self.output.mkdir(parents=True, exist_ok=True)

    async def fresh(self, name, url=None):
        self.context = await self.browser.new_context(viewport={"width":1600,"height":1060})
        await self.context.add_init_script(INIT)
        self.page = await self.context.new_page()
        self.cdp = await self.context.new_cdp_session(self.page)
        await self.cdp.send("Performance.enable")
        await self.page.goto(url or self.args.url, wait_until="domcontentloaded")
        await self.page.wait_for_function("typeof charts !== 'undefined' && charts.size>0")
        await self.ready()
        self.current = {"navigationReadyMs": await self.page.evaluate("performance.now()"),
                        "initial": await self.page.evaluate(STATE), "checks": []}
        self.results["scenarios"][name] = self.current
        await self.page.evaluate("(n)=>B.phase=n", name)

    async def ready(self):
        await self.page.wait_for_function("""()=>{
          const cs=[...charts.values()].filter(c=>c.visible);
          return cs.length && cs.every(c=>c.data&&!c.dirty&&!c.loading)&&!SCHED.running
        }""", timeout=120000)
        await self.page.wait_for_timeout(80)

    async def input(self, selector, value):
        return await self.page.evaluate("""({selector,value})=>{
          const el=document.querySelector(selector);el.value=value;
          const t=performance.now();el.dispatchEvent(new Event('input',{bubbles:true}));return t
        }""", {"selector":selector,"value":value})

    async def click(self, selector):
        return await self.page.evaluate("""s=>{
          const t=performance.now();document.querySelector(s).click();
          B.events.push({selector:s,start:t,syncMs:performance.now()-t});return t
        }""", selector)

    async def pick(self, text, count):
        await self.input("#runfilter", text)
        await self.page.wait_for_timeout(90)
        t = await self.click("#selall")
        await self.page.wait_for_function("(n)=>S.picked.size===n", arg=count)
        await self.page.wait_for_timeout(220)
        await self.page.wait_for_function("!keysCtl || !keysCtl.signal.aborted")
        return t

    async def panels(self, regex, expected):
        await self.click("#clear")
        await self.input("#keyfilter", regex)
        await self.page.wait_for_timeout(100)
        t = await self.click("#addall")
        await self.page.wait_for_function("(n)=>charts.size===n", arg=expected)
        await self.ready()
        return t

    def check(self, name, actual, expected):
        passed = actual == expected
        self.current["checks"].append({"name":name,"actual":actual,"expected":expected,"passed":passed})

    async def heap(self):
        await self.cdp.send("HeapProfiler.collectGarbage")
        heap = await self.cdp.send("Runtime.getHeapUsage")
        dom = await self.cdp.send("Memory.getDOMCounters")
        listener = await self.cdp.send("Runtime.evaluate", {
            "expression":"""(()=>{
              const ls=getEventListeners(window).mouseup||[];
              return {all:ls.length,app:ls.filter(e=>
                !e.listener.toString().includes('_hitTargetInterceptor')).length}
            })()""",
            "includeCommandLineAPI":True,"returnByValue":True})
        listeners = listener["result"].get("value", {})
        return {**heap, **dom, "windowMouseup":listeners.get("all"),
                "appWindowMouseup":listeners.get("app")}

    async def profile_start(self):
        await self.cdp.send("Profiler.enable")
        await self.cdp.send("Profiler.setSamplingInterval", {"interval":1000})
        await self.cdp.send("Profiler.start")

    async def profile_stop(self, name):
        profile = (await self.cdp.send("Profiler.stop"))["profile"]
        path = self.output / f"{self.args.label}-{name}.cpuprofile"
        path.write_text(json.dumps(profile))
        self.current["cpuProfile"] = str(path)
        self.current["cpuTopSelf"] = cpu_top(profile)

    async def sweep(self, kind):
        result = await self.page.evaluate(SWEEP, {"kind":kind,"frames":180})
        self.current[kind] = {"frameMs": dist(result["gaps"]),
                              "eventWorkMs":dist(result["work"]), "labels":result["labels"]}
        await self.ready()

    async def finish(self, shot):
        self.current["final"] = await self.page.evaluate(STATE)
        metrics = await self.page.evaluate("B")
        self.current["raw"] = metrics
        self.current["summary"] = {
            "networkMs":dist([n["net"] for n in metrics["notes"]]),
            "drawMs":dist([n["ms"] for n in metrics["draws"]]),
            "parseDrawMs":dist([n["draw"] for n in metrics["notes"]]),
            "bytes":dist([n["bytes"] for n in metrics["notes"]]),
            "hoverCallbackMs":dist([n["ms"] for n in metrics["hover"] if n["phase"]=="hover"]),
            "longTaskMs":dist([n["ms"] for n in metrics["tasks"]]),
            "maxActive":metrics["maxActive"], "errors":metrics["errors"]}
        self.current["screenshot"] = str(self.output/f"{self.args.label}-{shot}.png")
        await self.page.screenshot(path=self.current["screenshot"])
        print(json.dumps({"scenario":shot, **self.current["summary"],
                          "checks":self.current["checks"]}), flush=True)
        self.save()
        await self.context.close()

    def save(self):
        (self.output/f"{self.args.label}.json").write_text(json.dumps(self.results, indent=2))

    async def all_runs(self):
        await self.fresh("all1004")
        t = await self.pick("", 1004)
        await self.ready()
        self.current["pickToDrawMs"] = await self.page.evaluate("performance.now()")-t
        t = await self.panels("train_loss$|train_grad_norm$|synthetic_metric_0000[0-3] .*gpu=0",6)
        self.current["sixPanelsReadyMs"] = await self.page.evaluate("performance.now()")-t
        self.check("six panels", await self.page.evaluate("charts.size"),6)
        await self.profile_start()
        await self.sweep("hover")
        await self.profile_stop("all1004")
        await self.finish("all1004")

    async def wide(self):
        await self.fresh("wide400")
        await self.pick("wide",1)
        await self.click("#clear")
        await self.page.wait_for_function("S.keys.length===20000")
        self.current["heapBefore"] = await self.heap()
        self.current["filters"] = []
        for regex, count in [("",20000),("synthetic_metric_00000",8),("[",0),(".*",20000)]:
            t = await self.input("#keyfilter",regex)
            await self.page.wait_for_function("""({count,t})=>
              S.shownKeys.length===count && B.keyChanges.some(c=>c.at>=t)
            """,arg={"count":count,"t":t})
            at = await self.page.evaluate("""()=>{
              const n=B.keyChanges.at(-1);return n?.at || performance.now()
            }""")
            self.current["filters"].append({"regex":regex,"matches":count,"ms":at-t})
            self.check("regex "+regex, await self.page.evaluate("S.shownKeys.length"),count)
        await self.sweep_empty_keys()
        await self.profile_start()
        t = await self.click("#addall")
        await self.ready()
        self.current["createReadyMs"] = await self.page.evaluate("performance.now()")-t
        self.current["afterCreate"] = await self.page.evaluate(STATE)
        self.current["heapAfterCreate"] = await self.heap()
        self.check("400 cap", await self.page.evaluate("charts.size"),400)
        await self.sweep("panels-scroll")
        await self.page.evaluate("document.querySelector('#panels').scrollTop=1e9")
        await self.ready()
        self.current["afterScroll"] = await self.page.evaluate(STATE)
        self.current["heapAfterScroll"] = await self.heap()
        shot = str(self.output/f"{self.args.label}-wide400-populated.png")
        await self.page.screenshot(path=shot)
        self.current["populatedScreenshot"] = shot
        self.current["cycles"] = []
        for i in range(3):
            t = await self.click("#clear")
            await self.page.wait_for_timeout(150)
            self.current["cycles"].append({"cycle":i+1,
                                          "clearSettledMs":await self.page.evaluate("performance.now()")-t,
                                          "clearSyncMs":await self.page.evaluate("B.events.at(-1).syncMs"),
                                          "heapAfterClear":await self.heap()})
            if i<2:
                await self.click("#addall")
                await self.ready()
        await self.profile_stop("wide400")
        self.check("clear leaves no panels", await self.page.evaluate("charts.size"),0)
        await self.finish("wide400")

    async def sweep_empty_keys(self):
        result = await self.page.evaluate(SWEEP, {"kind":"keys-scroll","frames":180})
        self.current["keys-scroll"] = {"frameMs":dist(result["gaps"]),
                                      "eventWorkMs":dist(result["work"]),
                                      "labels":result["labels"]}
        self.current["keyRowCount"] = await self.page.evaluate("""()=>({
          count:$('#keys').querySelectorAll('.vrow').length,height:$('#keys').clientHeight
        })""")

    async def long(self):
        await self.fresh("long")
        await self.pick("long-",3)
        await self.panels("train_loss$|train_grad_norm$",2)
        self.current["full"] = await self.page.evaluate(STATE)
        await self.page.evaluate("""()=>{const d=[...charts.values()][0].data;
          setView({x0:d.x0+(d.x1-d.x0)*.3,x1:d.x0+(d.x1-d.x0)*.6})}""")
        await self.ready()
        self.current["zoom"] = await self.page.evaluate(STATE)
        self.current["zoomScreenshot"] = str(self.output/f"{self.args.label}-long-zoom.png")
        await self.page.screenshot(path=self.current["zoomScreenshot"])
        await self.page.evaluate("setView(null)")
        await self.ready()
        self.current["unzoom"] = await self.page.evaluate(STATE)
        for key in ["full","zoom","unzoom"]:
            cs = self.current[key]["charts"]
            self.check(key+" chart count",len(cs),2)
            self.check(key+" common x range",len(set((c["x0"],c["x1"]) for c in cs)),1)
            self.check(key+" sample bound",all(c["maxSamples"]<=4*c["width"]+4 for c in cs),True)
            self.check(key+" runs",all(c["series"]==3 for c in cs),True)
        self.check("unzoom restores full range",
                   [(c["x0"],c["x1"]) for c in self.current["unzoom"]["charts"]],
                   [(c["x0"],c["x1"]) for c in self.current["full"]["charts"]])
        # Real event path verifies the fix did not break drag dispatch.
        await self.page.evaluate("""()=>{
          const c=[...charts.values()][0],r=c.over.getBoundingClientRect();
          const ev=(type,x)=>new MouseEvent(type,{bubbles:true,clientX:r.left+r.width*x,
            clientY:r.top+r.height*.5});
          c.over.dispatchEvent(ev('mousedown',.3));c.over.dispatchEvent(ev('mousemove',.6));
        }""")
        await self.page.wait_for_timeout(50)
        self.current["dragHeld"] = await self.page.evaluate("[...charts.values()][0].drag")
        held = str(self.output/f"{self.args.label}-drag-held.png")
        await self.page.screenshot(path=held)
        self.current["dragScreenshot"] = held
        await self.page.evaluate("window.dispatchEvent(new MouseEvent('mouseup',{bubbles:true}))")
        await self.ready()
        self.current["dragReleased"] = await self.page.evaluate(STATE)
        self.check("drag changes view",await self.page.evaluate("S.view!==null"),True)
        await self.page.evaluate("[...charts.values()][0].over.dispatchEvent(new MouseEvent('dblclick',{bubbles:true}))")
        await self.ready()
        self.check("doubleclick clears view",await self.page.evaluate("S.view===null"),True)
        await self.finish("long")

    async def alignment(self):
        await self.fresh("alignment")
        await self.pick("many-",1000)
        await self.panels("train_loss$|train_grad_norm$",2)
        self.current["start"] = await self.page.evaluate(STATE)
        for mode,key in [("above","train_step$"),("first","train_loss$")]:
            await self.input("#akey",key)
            await self.input("#aval","0")
            t = await self.input("#amode",mode)
            await self.page.wait_for_timeout(350)
            await self.ready()
            self.current[mode] = await self.page.evaluate(STATE)
            self.current[mode]["inputToReadyMs"] = self.current[mode]["at"]-t
        await self.input("#xkey","trainer/train_step")
        await self.input("#xmode","metric")
        await self.page.wait_for_timeout(350)
        await self.ready()
        self.current["metricX"] = await self.page.evaluate(STATE)
        self.check("metric x key",await self.page.evaluate("S.xkey"),"trainer/train_step")
        for key in ["start","above","first","metricX"]:
            self.check(key+" runs",all(c["series"]==1000 for c in self.current[key]["charts"]),True)
        for key in ["above","first"]:
            loss=next(c for c in self.current[key]["charts"] if c["key"].endswith("train_loss"))
            bucket=(loss["x1"]-loss["x0"])/loss["width"]
            self.check(key+" aligned within first bucket",all(abs(x)<=bucket+2 for x in loss["firstXs"]),True)
        await self.finish("alignment")

    async def smooth(self):
        await self.fresh("smooth")
        await self.pick("",1004)
        await self.panels("train_loss$|train_grad_norm$|synthetic_metric_0000[0-3] .*gpu=0",6)
        self.current["before"] = await self.page.evaluate(STATE)
        await self.profile_start()
        t = await self.input("#smooth","0.9")
        await self.page.wait_for_timeout(160)
        await self.ready()
        self.current["inputToReadyMs"] = await self.page.evaluate("performance.now()")-t
        self.current["after"] = await self.page.evaluate(STATE)
        await self.sweep("hover")
        await self.profile_stop("smooth")
        self.check("smooth setting",await self.page.evaluate("S.smooth"),.9)
        before={c["key"]:c for c in self.current["before"]["charts"]}
        self.check("all visible series samples change",all(
            c["firstYs"]!=before[c["key"]]["firstYs"] for c in self.current["after"]["charts"]),True)
        await self.finish("smooth")

    async def demo(self):
        await self.fresh("demo","http://127.0.0.1:8737")
        self.current["before"] = await self.page.evaluate(STATE)
        await self.page.wait_for_timeout(5500)
        await self.ready()
        self.current["after"] = await self.page.evaluate(STATE)
        self.check("live badge",await self.page.locator("#live").text_content(),"● 3 LIVE")
        self.check("live charts grow",self.current["after"]["charts"][0]["x1"]>
                   self.current["before"]["charts"][0]["x1"],True)
        self.current["headers"] = []
        for width in [1600,1024]:
            await self.page.set_viewport_size({"width":width,"height":1060})
            await self.page.wait_for_timeout(100)
            self.current["headers"].append(await self.page.evaluate("""()=>{
              const a=$('h1').getBoundingClientRect(),b=$('.tag').getBoundingClientRect();
              return {width:innerWidth,title:a.toJSON(),subtitle:b.toJSON(),
                overlap:!(a.right<b.left||a.left>b.right||a.bottom<b.top||a.top>b.bottom)}
            }"""))
            path = str(self.output/f"{self.args.label}-demo-{width}.png")
            await self.page.screenshot(path=path)
            self.current["headers"][-1]["screenshot"] = path
        await self.finish("demo")

    async def run(self):
        for method in [self.all_runs,self.wide,self.long,self.alignment,self.smooth,self.demo]:
            if self.args.scenarios and method.__name__ not in self.args.scenarios.split(","):
                continue
            try:
                await method()
            except Exception as exc:
                self.current["harnessError"] = repr(exc)
                self.save()
                print(f"SCENARIO ERROR: {exc}",flush=True)
                await self.context.close()
        self.save()


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cdp",default="http://127.0.0.1:29229")
    p.add_argument("--url",default="http://127.0.0.1:8738")
    p.add_argument("--label",required=True)
    p.add_argument("--out",default="/tmp/niu-results")
    p.add_argument("--scenarios",help="Optional comma-separated method names: all_runs,wide,long,alignment,smooth,demo")
    args = p.parse_args()
    async with async_playwright() as pw:
        browser = await pw.chromium.connect_over_cdp(args.cdp)
        await Bench(browser,args).run()


if __name__ == "__main__":
    asyncio.run(main())
