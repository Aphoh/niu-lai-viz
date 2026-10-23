//! `serve`: HTTP API + UI over a directory of runs.
//!
//! Every chart request is answered at pixel resolution: the server aligns each
//! run (T=0 anchor or a metric as x-axis), smooths, and min/max-decimates per
//! pixel column, so response size and draw cost are bounded by chart width
//! times run count, never by run length. Responses are packed f32 arrays.
use crate::run::{data_files, now_ms, RunMeta, META_FILE};
use arrow::array::{Array, AsArray, Float64Array, StringArray};
use arrow::datatypes::{DataType, Float64Type};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use tokio::sync::broadcast;

const INDEX: &str = include_str!("../ui/index.html");
const MARKED_JS: &str = include_str!("../ui/vendor/marked.min.js");
const PURIFY_JS: &str = include_str!("../ui/vendor/purify.min.js");
/// Writer columns plus `metric_name_clean`, which compaction derives.
const FIXED_COLUMNS: [&str; 10] = [
    "scraper_endpoint",
    "metric_name",
    "metric_value",
    "histogram_bucket_lower",
    "histogram_bucket_upper",
    "histogram_sum",
    "histogram_count",
    "time_since_start",
    "timestamp_ns",
    "metric_name_clean",
];

#[derive(Default)]
pub struct Series {
    pub t: Vec<f32>,
    pub v: Vec<f32>,
}

/// All series of one run, keyed `endpoint/metric_name [col=val,...]`.
/// Segments are append-only, so only new files are read on refresh.
#[derive(Default)]
pub struct RunData {
    files: Vec<(PathBuf, u64)>,
    index: HashMap<String, usize>,
    pub keys: Vec<String>,
    pub series: Vec<Series>,
    anchors: HashMap<String, Option<f64>>,
}

impl RunData {
    pub fn get(&self, key: &str) -> Option<&Series> {
        self.index.get(key).map(|&i| &self.series[i])
    }

    pub fn refresh(&mut self, run_dir: &Path) -> anyhow::Result<()> {
        let files: Vec<(PathBuf, u64)> = data_files(run_dir)
            .into_iter()
            .filter_map(|p| Some((p.clone(), std::fs::metadata(&p).ok()?.len())))
            .collect();
        let n = self.files.len();
        if files.len() < n || files[..n] != self.files[..] {
            *self = RunData::default();
        }
        if files.len() == self.files.len() {
            return Ok(());
        }
        let mut touched = HashSet::new();
        for (p, _) in &files[self.files.len()..] {
            if let Err(e) = self.read_file(p, &mut touched) {
                *self = RunData::default();
                return Err(e);
            }
        }
        for i in touched {
            let s = &mut self.series[i];
            if !s.t.windows(2).all(|w| w[0] <= w[1]) {
                let mut idx: Vec<usize> = (0..s.t.len()).collect();
                idx.sort_by(|&a, &b| s.t[a].total_cmp(&s.t[b]));
                s.t = idx.iter().map(|&i| s.t[i]).collect();
                s.v = idx.iter().map(|&i| s.v[i]).collect();
            }
        }
        self.anchors.clear();
        self.files = files;
        Ok(())
    }

    fn read_file(&mut self, path: &Path, touched: &mut HashSet<usize>) -> anyhow::Result<()> {
        let reader = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(path)?)?
            .with_batch_size(65536)
            .build()?;
        let mut key = String::with_capacity(256);
        for batch in reader {
            let batch = batch?;
            let schema = batch.schema();
            let utf8 = |name: &str| -> anyhow::Result<StringArray> {
                let col = batch
                    .column_by_name(name)
                    .ok_or_else(|| anyhow::anyhow!("{} lacks {name}", path.display()))?;
                Ok(arrow::compute::cast(col, &DataType::Utf8)?
                    .as_string::<i32>()
                    .clone())
            };
            let f64s = |name: &str| -> anyhow::Result<Float64Array> {
                let col = batch
                    .column_by_name(name)
                    .ok_or_else(|| anyhow::anyhow!("{} lacks {name}", path.display()))?;
                Ok(arrow::compute::cast(col, &DataType::Float64)?
                    .as_primitive::<Float64Type>()
                    .clone())
            };
            let (ep, name, value, t) = (
                utf8("scraper_endpoint")?,
                utf8("metric_name")?,
                f64s("metric_value")?,
                f64s("time_since_start")?,
            );
            let extras: Vec<(String, StringArray)> = schema
                .fields()
                .iter()
                .filter(|f| !FIXED_COLUMNS.contains(&f.name().as_str()))
                .map(|f| Ok((f.name().clone(), utf8(f.name())?)))
                .collect::<anyhow::Result<_>>()?;
            let mut last: Option<usize> = None;
            for i in 0..batch.num_rows() {
                key.clear();
                key.push_str(ep.value(i));
                key.push('/');
                key.push_str(name.value(i));
                let mut first = true;
                for (n, a) in &extras {
                    if a.is_valid(i) && !a.value(i).is_empty() {
                        key.push_str(if first { " [" } else { "," });
                        first = false;
                        key.push_str(n);
                        key.push('=');
                        key.push_str(a.value(i));
                    }
                }
                if !first {
                    key.push(']');
                }
                // final.parquet is sorted by metric, so consecutive rows usually share a series.
                let idx = match last.filter(|&l| self.keys[l] == key) {
                    Some(l) => l,
                    None => match self.index.get(key.as_str()) {
                        Some(&l) => l,
                        None => {
                            let l = self.series.len();
                            self.index.insert(key.clone(), l);
                            self.keys.push(key.clone());
                            self.series.push(Series::default());
                            l
                        }
                    },
                };
                if last != Some(idx) {
                    touched.insert(idx);
                    last = Some(idx);
                }
                let s = &mut self.series[idx];
                s.t.push(t.value(i) as f32);
                s.v.push(value.value(i) as f32);
            }
        }
        Ok(())
    }

    /// Time of the anchor event: first sample of any series matching `re`,
    /// or the first one whose value exceeds `above`.
    fn anchor(&mut self, spec: &Align, re: &Regex) -> Option<f64> {
        let id = format!("{}\u{0}{:?}", spec.key, spec.above);
        if let Some(v) = self.anchors.get(&id) {
            return *v;
        }
        let mut best: Option<f64> = None;
        for (k, s) in self.keys.iter().zip(&self.series) {
            if !re.is_match(k) {
                continue;
            }
            let hit = match spec.above {
                None => s.t.first().copied(),
                Some(th) => s.v.iter().position(|&v| v as f64 > th).map(|i| s.t[i]),
            };
            if let Some(h) = hit {
                best = Some(best.map_or(h as f64, |b: f64| b.min(h as f64)));
            }
        }
        self.anchors.insert(id, best);
        best
    }
}

type Shared = Arc<App>;

struct App {
    root: PathBuf,
    cache: Mutex<HashMap<String, Arc<RwLock<RunData>>>>,
    dirty: Mutex<HashSet<String>>,
    tx: broadcast::Sender<String>,
}

impl App {
    /// A run's data, refreshed if the watcher saw it change.
    fn run(&self, name: &str) -> anyhow::Result<Arc<RwLock<RunData>>> {
        if !valid_name(name) || !self.root.join(name).join(META_FILE).is_file() {
            anyhow::bail!("no such run");
        }
        let (data, fresh) = {
            let mut cache = self.cache.lock().unwrap();
            let fresh = !cache.contains_key(name);
            (cache.entry(name.to_string()).or_default().clone(), fresh)
        };
        let stale = self.dirty.lock().unwrap().remove(name);
        if fresh || stale {
            data.write().unwrap().refresh(&self.root.join(name))?;
        }
        Ok(data)
    }
}

fn valid_name(name: &str) -> bool {
    !name.is_empty() && !name.starts_with('.') && !name.contains(['/', '\\'])
}

#[derive(Serialize)]
struct RunInfo {
    #[serde(flatten)]
    meta: RunMeta,
    live: bool,
}

fn list_runs(root: &Path) -> Vec<RunInfo> {
    let now = now_ms();
    let dirs: Vec<_> = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    let mut runs: Vec<RunInfo> = dirs
        .par_iter()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if !valid_name(&name) {
                return None;
            }
            let mut meta = RunMeta::read(&e.path())?;
            meta.name = name;
            let live = meta.is_live(now);
            if meta.status == "running" && !live {
                meta.status = "stale".into();
            }
            Some(RunInfo { meta, live })
        })
        .collect();
    runs.sort_by(|a, b| {
        b.meta
            .started_at_ms
            .cmp(&a.meta.started_at_ms)
            .then(a.meta.name.cmp(&b.meta.name))
    });
    runs
}

async fn runs(State(app): State<Shared>) -> Response {
    let root = app.root.clone();
    let list = tokio::task::spawn_blocking(move || list_runs(&root))
        .await
        .unwrap_or_default();
    Json(list).into_response()
}

#[derive(Deserialize)]
struct KeysReq {
    runs: Vec<String>,
}

#[derive(Serialize, Default)]
struct KeysResp {
    keys: Vec<String>,
    /// Histogram metrics (see `hist::bucket_of`); their bucket series are not in `keys`.
    hists: Vec<String>,
    errors: BTreeMap<String, String>,
}

async fn keys(State(app): State<Shared>, Json(req): Json<KeysReq>) -> Response {
    let out = tokio::task::spawn_blocking(move || {
        let per_run: Vec<(String, anyhow::Result<Vec<String>>)> = req
            .runs
            .par_iter()
            .map(|r| {
                (
                    r.clone(),
                    app.run(r).map(|d| d.read().unwrap().keys.clone()),
                )
            })
            .collect();
        let (mut keys, mut hists) = (BTreeSet::new(), BTreeSet::new());
        let mut out = KeysResp::default();
        for (r, res) in per_run {
            match res {
                Ok(ks) => {
                    for k in ks {
                        match crate::hist::bucket_of(&k) {
                            Some((base, _)) => hists.insert(base),
                            None => keys.insert(k),
                        };
                    }
                }
                Err(e) => {
                    out.errors.insert(r, e.to_string());
                }
            }
        }
        out.keys = keys.into_iter().collect();
        out.hists = hists.into_iter().collect();
        out
    })
    .await
    .unwrap_or_default();
    Json(out).into_response()
}

#[derive(Deserialize, Clone, Debug)]
pub struct Align {
    /// Regex over series keys.
    pub key: String,
    /// Anchor on the first value above this instead of the first sample.
    #[serde(default)]
    pub above: Option<f64>,
}

#[derive(Deserialize)]
struct PlotReq {
    runs: Vec<String>,
    key: String,
    /// Pixel columns to decimate to.
    width: usize,
    #[serde(default)]
    x0: Option<f64>,
    #[serde(default)]
    x1: Option<f64>,
    #[serde(default)]
    align: Option<Align>,
    /// Plot against this series' value (e.g. a step counter) instead of time.
    #[serde(default)]
    x_key: Option<String>,
    /// Custom-plot formula (see `expr`); replaces `key` when set.
    #[serde(default)]
    expr: Option<String>,
    /// EMA factor in [0, 1).
    #[serde(default)]
    smooth: f64,
}

#[derive(Serialize)]
struct PlotRun {
    name: String,
    n: usize,
    offset: Option<f64>,
}

#[derive(Serialize)]
struct PlotMeta {
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    runs: Vec<PlotRun>,
    errors: BTreeMap<String, String>,
}

/// One run mapped onto the plot's x-axis: x[i] for sample i of the series.
struct Mapped {
    x: Vec<f32>,
    v: Vec<f32>,
}

fn map_run(
    data: &mut RunData,
    req: &PlotReq,
    anchor_re: Option<&Regex>,
) -> Result<Option<(Mapped, f64)>, String> {
    let offset = match (&req.align, anchor_re) {
        (Some(a), Some(re)) => match data.anchor(a, re) {
            Some(o) => o,
            None => return Ok(None),
        },
        _ => 0.0,
    };
    let owned;
    let s = match &req.expr {
        Some(e) => {
            let scope = crate::expr::Scope {
                keys: &data.keys,
                series: &data.series,
            };
            owned = crate::expr::eval(e, &scope)?;
            match &owned {
                Some(s) => s,
                None => return Ok(None),
            }
        }
        None => match data.get(&req.key) {
            Some(s) => s,
            None => return Ok(None),
        },
    };
    let mut v = s.v.clone();
    if req.smooth > 0.0 {
        let a = req.smooth.min(0.999) as f32;
        let mut acc: Option<f32> = None;
        for y in v.iter_mut() {
            if y.is_finite() {
                let next = acc.map_or(*y, |p| a * p + (1.0 - a) * *y);
                acc = Some(next);
                *y = next;
            }
        }
    }
    let x = match &req.x_key {
        None => s.t.iter().map(|&t| (t as f64 - offset) as f32).collect(),
        Some(xk) => {
            let Some(xs) = data.get(xk) else {
                return Ok(None);
            };
            // Step-hold join: x at time t is the latest x sample at or before t.
            let mut j = 0;
            let mut out = Vec::with_capacity(s.t.len());
            for &t in &s.t {
                while j < xs.t.len() && xs.t[j] <= t {
                    j += 1;
                }
                out.push(if j == 0 { f32::NAN } else { xs.v[j - 1] });
            }
            out
        }
    };
    Ok(Some((Mapped { x, v }, offset)))
}

/// Min/max (or M4 when `m4`) decimation of points with x in [x0, x1] into
/// `width` columns, keeping one point beyond each edge so lines reach it.
fn decimate(
    m: &Mapped,
    x0: f64,
    x1: f64,
    width: usize,
    m4: bool,
    xs: &mut Vec<f32>,
    ys: &mut Vec<f32>,
) {
    let n = m.x.len();
    let monotone = m.x.windows(2).all(|w| !(w[1] < w[0]));
    let (lo, hi) = if monotone {
        let lo = m.x.partition_point(|&x| (x as f64) < x0).saturating_sub(1);
        let hi = (m.x.partition_point(|&x| (x as f64) <= x1) + 1).min(n);
        (lo, hi)
    } else {
        (0, n)
    };
    let scale = width as f64 / (x1 - x0).max(1e-12);
    let mut col = i64::MIN;
    let mut picks: [usize; 4] = [0; 4];
    let flush = |picks: &[usize; 4], xs: &mut Vec<f32>, ys: &mut Vec<f32>| {
        let mut p = if m4 {
            picks.to_vec()
        } else {
            vec![picks[1], picks[2]]
        };
        p.sort_unstable();
        p.dedup();
        for i in p {
            xs.push(m.x[i]);
            ys.push(m.v[i]);
        }
    };
    for i in lo..hi {
        let x = m.x[i];
        if !x.is_finite() || !m.v[i].is_finite() {
            continue;
        }
        if !monotone && ((x as f64) < x0 || (x as f64) > x1) {
            continue;
        }
        let c = ((x as f64 - x0) * scale).floor() as i64;
        if c != col {
            if col != i64::MIN {
                flush(&picks, xs, ys);
            }
            col = c;
            picks = [i, i, i, i];
        } else {
            if m.v[i] < m.v[picks[1]] {
                picks[1] = i;
            }
            if m.v[i] > m.v[picks[2]] {
                picks[2] = i;
            }
            picks[3] = i;
        }
    }
    if col != i64::MIN {
        flush(&picks, xs, ys);
    }
}

fn plot_blocking(app: &App, req: PlotReq) -> Vec<u8> {
    let anchor_re = req.align.as_ref().and_then(|a| Regex::new(&a.key).ok());
    let mapped: Vec<(String, Result<Option<(Mapped, f64)>, String>)> = req
        .runs
        .par_iter()
        .map(|r| {
            let res = app
                .run(r)
                .map_err(|e| e.to_string())
                .and_then(|d| map_run(&mut d.write().unwrap(), &req, anchor_re.as_ref()));
            (r.clone(), res)
        })
        .collect();
    let (mut x0, mut x1) = (f64::INFINITY, f64::NEG_INFINITY);
    for (_, res) in &mapped {
        if let Ok(Some((m, _))) = res {
            for &x in
                m.x.iter()
                    .filter(|x| x.is_finite())
                    .take(1)
                    .chain(m.x.iter().rev().filter(|x| x.is_finite()).take(1))
            {
                x0 = x0.min(x as f64);
                x1 = x1.max(x as f64);
            }
        }
    }
    let x0 = req.x0.unwrap_or(if x0.is_finite() { x0 } else { 0.0 });
    let x1 = req.x1.unwrap_or(if x1.is_finite() { x1 } else { 1.0 });
    let x1 = if x1 > x0 { x1 } else { x0 + 1.0 };
    let width = req.width.clamp(8, 8192);
    let m4 = req.runs.len() <= 64;
    let decimated: Vec<(Vec<f32>, Vec<f32>)> = mapped
        .par_iter()
        .map(|(_, res)| {
            let (mut xs, mut ys) = (Vec::new(), Vec::new());
            if let Ok(Some((m, _))) = res {
                decimate(m, x0, x1, width, m4, &mut xs, &mut ys);
            }
            (xs, ys)
        })
        .collect();
    let (mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY);
    for ((xs, ys), _) in decimated.iter().zip(&mapped) {
        for (&x, &y) in xs.iter().zip(ys) {
            if (x as f64) >= x0 && (x as f64) <= x1 {
                y0 = y0.min(y as f64);
                y1 = y1.max(y as f64);
            }
        }
    }
    if !y0.is_finite() {
        (y0, y1) = (0.0, 1.0);
    }
    let mut meta = PlotMeta {
        x0,
        x1,
        y0,
        y1,
        runs: Vec::new(),
        errors: BTreeMap::new(),
    };
    for ((name, res), (xs, _)) in mapped.iter().zip(&decimated) {
        let offset = match res {
            Ok(Some((_, o))) => Some(*o),
            Ok(None) => None,
            Err(e) => {
                meta.errors.insert(name.clone(), e.clone());
                None
            }
        };
        meta.runs.push(PlotRun {
            name: name.clone(),
            n: xs.len(),
            offset,
        });
    }
    let mut json = serde_json::to_vec(&meta).unwrap_or_default();
    while json.len() % 4 != 0 {
        json.push(b' ');
    }
    let total: usize = decimated.iter().map(|(x, _)| x.len() * 8).sum();
    let mut buf = Vec::with_capacity(4 + json.len() + total);
    buf.extend_from_slice(&(json.len() as u32).to_le_bytes());
    buf.extend_from_slice(&json);
    for (xs, ys) in &decimated {
        for x in xs {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        for y in ys {
            buf.extend_from_slice(&y.to_le_bytes());
        }
    }
    buf
}

async fn plot(State(app): State<Shared>, Json(req): Json<PlotReq>) -> Response {
    let buf = tokio::task::spawn_blocking(move || plot_blocking(&app, req))
        .await
        .unwrap_or_default();
    (
        [(header::CONTENT_TYPE, "application/octet-stream")],
        Bytes::from(buf),
    )
        .into_response()
}

#[derive(Deserialize)]
struct StatsReq {
    runs: Vec<String>,
    /// Regex over series keys.
    #[serde(default)]
    key: String,
    /// Custom-plot formula; summarised as one row per run instead of `key`.
    #[serde(default)]
    expr: Option<String>,
    /// Only samples with aligned time in [t0, t1].
    #[serde(default)]
    t0: Option<f64>,
    #[serde(default)]
    t1: Option<f64>,
    #[serde(default)]
    align: Option<Align>,
    /// Max matched series returned.
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Serialize)]
struct StatRow {
    run: String,
    key: String,
    n: usize,
    t_first: f64,
    t_last: f64,
    first: f64,
    last: f64,
    min: f64,
    max: f64,
    mean: f64,
}

#[derive(Serialize, Default)]
struct StatsResp {
    rows: Vec<StatRow>,
    truncated: bool,
    error: Option<String>,
}

/// Per run x series summary: the cheap aggregate an agent (or a human) reads
/// before deciding what to chart.
fn summarize(run: &str, key: &str, s: &Series, offset: f64, t0: f64, t1: f64) -> Option<StatRow> {
    let mut row = StatRow {
        run: run.to_string(),
        key: key.to_string(),
        n: 0,
        t_first: f64::NAN,
        t_last: f64::NAN,
        first: f64::NAN,
        last: f64::NAN,
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
        mean: 0.0,
    };
    for (&t, &v) in s.t.iter().zip(&s.v) {
        let (t, v) = (t as f64 - offset, v as f64);
        if t < t0 || t > t1 || !v.is_finite() {
            continue;
        }
        if row.n == 0 {
            row.t_first = t;
            row.first = v;
        }
        row.n += 1;
        row.t_last = t;
        row.last = v;
        row.min = row.min.min(v);
        row.max = row.max.max(v);
        row.mean += v;
    }
    (row.n > 0).then(|| {
        row.mean /= row.n as f64;
        row
    })
}

fn stats_blocking(app: &App, req: StatsReq) -> StatsResp {
    let re = match Regex::new(&req.key) {
        Ok(r) => r,
        Err(e) => {
            return StatsResp {
                error: Some(e.to_string()),
                ..Default::default()
            }
        }
    };
    let anchor_re = req.align.as_ref().and_then(|a| Regex::new(&a.key).ok());
    let limit = req.limit.unwrap_or(200).min(5000);
    let err = std::sync::Mutex::new(None::<String>);
    let (t0, t1) = (
        req.t0.unwrap_or(f64::NEG_INFINITY),
        req.t1.unwrap_or(f64::INFINITY),
    );
    let mut rows: Vec<StatRow> = req
        .runs
        .par_iter()
        .flat_map_iter(|r| {
            let mut out = Vec::new();
            let Ok(d) = app.run(r) else { return out };
            let mut d = d.write().unwrap();
            let offset = match (&req.align, anchor_re.as_ref()) {
                (Some(a), Some(are)) => match d.anchor(a, are) {
                    Some(o) => o,
                    None => return out,
                },
                _ => 0.0,
            };
            if let Some(e) = &req.expr {
                let scope = crate::expr::Scope {
                    keys: &d.keys,
                    series: &d.series,
                };
                match crate::expr::eval(e, &scope) {
                    Ok(Some(s)) => out.extend(summarize(r, e, &s, offset, t0, t1)),
                    Ok(None) => {}
                    Err(e) => *err.lock().unwrap() = Some(e),
                }
                return out;
            }
            for (key, s) in d.keys.iter().zip(&d.series) {
                if re.is_match(key) {
                    out.extend(summarize(r, key, s, offset, t0, t1));
                }
            }
            out
        })
        .collect();
    rows.sort_by(|a, b| a.key.cmp(&b.key).then(a.run.cmp(&b.run)));
    let truncated = rows.len() > limit;
    rows.truncate(limit);
    StatsResp {
        rows,
        truncated,
        error: err.into_inner().unwrap(),
    }
}

#[derive(Deserialize)]
struct HistReq {
    runs: Vec<String>,
    /// Histogram base key from `/api/keys` `hists`.
    key: String,
    /// Number of equal time windows across [x0, x1].
    slices: usize,
    #[serde(default)]
    x0: Option<f64>,
    #[serde(default)]
    x1: Option<f64>,
    #[serde(default)]
    align: Option<Align>,
}

#[derive(Serialize)]
struct HistRun {
    name: String,
    offset: f64,
    /// Bucket upper bounds, ascending; `null` is +Inf.
    le: Vec<Option<f64>>,
    /// Per window, observations per bucket (`null`: no data or counter reset).
    counts: Vec<Option<Vec<f32>>>,
}

#[derive(Serialize, Default)]
struct HistResp {
    x0: f64,
    x1: f64,
    /// Windows actually used: at most `slices`, each >= 2 mean scrape intervals.
    slices: usize,
    runs: Vec<HistRun>,
    errors: BTreeMap<String, String>,
}

fn hist_buckets<'a>(d: &'a RunData, key: &str) -> Vec<(f64, &'a Series)> {
    let mut b: Vec<(f64, &Series)> = d
        .keys
        .iter()
        .zip(&d.series)
        .filter_map(|(k, s)| {
            crate::hist::bucket_of(k)
                .filter(|(base, _)| base == key)
                .map(|(_, le)| (le, s))
        })
        .collect();
    b.sort_by(|x, y| x.0.total_cmp(&y.0));
    b.dedup_by(|x, y| x.0 == y.0);
    b
}

fn hist_blocking(app: &App, req: HistReq) -> HistResp {
    let anchor_re = req.align.as_ref().and_then(|a| Regex::new(&a.key).ok());
    let found: Vec<(String, Result<Option<(f64, f64, f64, f64)>, String>)> = req
        .runs
        .par_iter()
        .map(|r| {
            let res = app.run(r).map_err(|e| e.to_string()).map(|d| {
                let mut d = d.write().unwrap();
                let offset = match (&req.align, anchor_re.as_ref()) {
                    (Some(a), Some(re)) => d.anchor(a, re)?,
                    _ => 0.0,
                };
                let (mut lo, mut hi, mut gap) = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64);
                for (_, s) in hist_buckets(&d, &req.key) {
                    if let (Some(&a), Some(&b)) = (s.t.first(), s.t.last()) {
                        lo = lo.min(a as f64 - offset);
                        hi = hi.max(b as f64 - offset);
                        if s.t.len() > 1 {
                            gap = gap.max((b - a) as f64 / (s.t.len() - 1) as f64);
                        }
                    }
                }
                lo.is_finite().then_some((offset, lo, hi, gap))
            });
            (r.clone(), res)
        })
        .collect();
    let mut out = HistResp::default();
    let (mut lo, mut hi, mut gap) = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64);
    for (r, res) in &found {
        match res {
            Ok(Some((_, a, b, g))) => {
                lo = lo.min(*a);
                hi = hi.max(*b);
                gap = gap.max(*g);
            }
            Ok(None) => {}
            Err(e) => {
                out.errors.insert(r.clone(), e.clone());
            }
        }
    }
    out.x0 = req.x0.unwrap_or(if lo.is_finite() { lo } else { 0.0 });
    out.x1 = req.x1.unwrap_or(if hi.is_finite() { hi } else { 1.0 });
    if out.x1 <= out.x0 {
        out.x1 = out.x0 + 1.0;
    }
    let n = if gap > 0.0 {
        let fit = ((out.x1 - out.x0) / (2.0 * gap)).floor().max(1.0) as usize;
        req.slices.clamp(1, 4096).min(fit)
    } else {
        req.slices.clamp(1, 4096)
    };
    out.slices = n;
    // Scrape gaps up to this long are jitter (counts spread across them); longer ones are outages.
    let max_gap = (10.0 * gap).max(5.0);
    let edges: Vec<f64> = (0..=n)
        .map(|k| out.x0 + (out.x1 - out.x0) * k as f64 / n as f64)
        .collect();
    out.runs = found
        .par_iter()
        .filter_map(|(r, res)| {
            let &Ok(Some((offset, _, _, _))) = res else {
                return None;
            };
            let d = app.run(r).ok()?;
            let d = d.read().unwrap();
            let b = hist_buckets(&d, &req.key);
            Some(HistRun {
                name: r.clone(),
                offset,
                le: b
                    .iter()
                    .map(|(le, _)| le.is_finite().then_some(*le))
                    .collect(),
                counts: crate::hist::slices(&b, offset, &edges, max_gap),
            })
        })
        .collect();
    out
}

async fn hist(State(app): State<Shared>, Json(req): Json<HistReq>) -> Response {
    let out = tokio::task::spawn_blocking(move || hist_blocking(&app, req))
        .await
        .unwrap_or_default();
    Json(out).into_response()
}

fn js(src: &'static str) -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "max-age=86400"),
        ],
        src,
    )
        .into_response()
}

async fn stats(State(app): State<Shared>, Json(req): Json<StatsReq>) -> Response {
    let out = tokio::task::spawn_blocking(move || stats_blocking(&app, req))
        .await
        .unwrap_or_default();
    Json(out).into_response()
}

async fn events(State(app): State<Shared>) -> impl IntoResponse {
    let rx = app.tx.subscribe();
    let stream = futures::stream::unfold(rx, |mut rx| async move {
        loop {
            match rx.recv().await {
                Ok(run) => {
                    return Some((
                        Ok::<_, Infallible>(Event::default().event("run").data(run)),
                        rx,
                    ))
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => return None,
            }
        }
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Cheap fingerprint of a run: data file names/sizes and run.json mtime.
fn signature(run_dir: &Path) -> Vec<(String, u64)> {
    data_files(run_dir)
        .into_iter()
        .chain([run_dir.join(META_FILE)])
        .filter_map(|p| {
            let md = std::fs::metadata(&p).ok()?;
            let mtime = md
                .modified()
                .ok()?
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?
                .as_millis() as u64;
            Some((
                p.file_name()?.to_string_lossy().to_string(),
                md.len() ^ mtime,
            ))
        })
        .collect()
}

async fn watch(app: Shared) {
    let mut seen: HashMap<String, Vec<(String, u64)>> = HashMap::new();
    loop {
        let root = app.root.clone();
        let sigs = tokio::task::spawn_blocking(move || {
            let dirs: Vec<String> = std::fs::read_dir(&root)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| valid_name(n))
                .collect();
            dirs.into_par_iter()
                .map(|n| {
                    let s = signature(&root.join(&n));
                    (n, s)
                })
                .collect::<Vec<_>>()
        })
        .await
        .unwrap_or_default();
        let names: HashSet<String> = sigs.iter().map(|(n, _)| n.clone()).collect();
        for (name, sig) in sigs {
            if seen.get(&name) != Some(&sig) {
                let first = !seen.contains_key(&name);
                seen.insert(name.clone(), sig);
                app.dirty.lock().unwrap().insert(name.clone());
                if !first || app.tx.receiver_count() > 0 {
                    let _ = app.tx.send(name);
                }
            }
        }
        seen.retain(|n, _| names.contains(n));
        app.cache.lock().unwrap().retain(|n, _| names.contains(n));
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}

pub fn router(root: PathBuf) -> Router {
    let (tx, _) = broadcast::channel(4096);
    let app = Arc::new(App {
        root,
        cache: Mutex::new(HashMap::new()),
        dirty: Mutex::new(HashSet::new()),
        tx,
    });
    tokio::spawn(watch(app.clone()));
    Router::new()
        .route("/", get(|| async { Html(INDEX) }))
        .route("/vendor/marked.js", get(|| async { js(MARKED_JS) }))
        .route("/vendor/purify.js", get(|| async { js(PURIFY_JS) }))
        .route("/api/runs", get(runs))
        .route("/api/keys", post(keys))
        .route("/api/plot", post(plot))
        .route("/api/stats", post(stats))
        .route("/api/hist", post(hist))
        .route("/api/events", get(events))
        .with_state(app)
}

pub async fn serve(root: PathBuf, addr: &str) -> anyhow::Result<()> {
    std::fs::create_dir_all(&root)?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    log::info!(
        "niu-lai-viz serving {} on http://{}",
        root.display(),
        listener.local_addr()?
    );
    axum::serve(listener, router(root)).await?;
    Ok(())
}
