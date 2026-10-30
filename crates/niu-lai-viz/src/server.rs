//! `serve`: HTTP API + UI over a directory of runs.
//!
//! Every chart request is answered at pixel resolution: the server aligns each
//! run (T=0 anchor or a metric as x-axis), smooths, and min/max-decimates per
//! pixel column, so response size and draw cost are bounded by chart width
//! times run count, never by run length. Responses are packed f32 arrays.
use crate::run::{data_files, final_files, is_locked, now_ms, segment_files, RunMeta, META_FILE};
use arrow::array::{Array, AsArray, Float64Array, StringArray};
use arrow::datatypes::{DataType, Float64Type};
use axum::body::Bytes;
use axum::extract::{Path as UrlPath, State};
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder};
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

/// A data file as loaded. Row counts come from the Parquet footer and let a
/// compacted file stand in for the segments it replaced without re-reading.
#[derive(Clone, Debug)]
struct FileInfo {
    path: PathBuf,
    len: u64,
    rows: u64,
    groups: usize,
}

impl FileInfo {
    fn same(&self, o: &FileInfo) -> bool {
        self.path == o.path && self.len == o.len
    }

    /// Compaction output: replaces the segments it was merged from.
    fn merged(&self) -> bool {
        self.path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("incomplete-") || n == crate::run::FINAL_FILE)
    }
}

/// How to bring loaded data in line with the files now on disk.
#[derive(Debug, PartialEq)]
enum Plan {
    Same,
    /// Read these new segments and append them.
    Append(Vec<usize>),
    /// Compaction merged files we hold into new ones with the same rows:
    /// adopt the new file list, then append the listed new segments.
    Swap(Vec<usize>),
    /// Compaction is mid-flight (output written, sources not yet deleted):
    /// keep serving what we have.
    Wait,
    Rebuild,
}

/// Indices are into `cur`.
fn plan(old: &[FileInfo], cur: &[FileInfo]) -> Plan {
    if old.is_empty() {
        return if cur.is_empty() {
            Plan::Same
        } else {
            Plan::Rebuild
        };
    }
    let gone: Vec<&FileInfo> = old
        .iter()
        .filter(|f| !cur.iter().any(|c| c.same(f)))
        .collect();
    let added: Vec<usize> = (0..cur.len())
        .filter(|&i| !old.iter().any(|o| o.same(&cur[i])))
        .collect();
    let (merged, fresh): (Vec<usize>, Vec<usize>) = added.iter().partition(|&&i| cur[i].merged());
    if gone.is_empty() && merged.is_empty() {
        return if fresh.is_empty() {
            Plan::Same
        } else {
            Plan::Append(fresh)
        };
    }
    let gone_rows: u64 = gone.iter().map(|f| f.rows).sum();
    let merged_rows: u64 = merged.iter().map(|&i| cur[i].rows).sum();
    let rows = |fs: &[FileInfo]| fs.iter().map(|f| f.rows).sum::<u64>();
    if !gone.is_empty() && merged_rows == gone_rows {
        Plan::Swap(fresh)
    } else if rows(cur) < rows(old) {
        // Files vanished without a replacement yet: never drop data on a guess.
        Plan::Wait
    } else if added.len() == cur.len() {
        // Nothing we hold is still on disk, so nothing is pending deletion.
        Plan::Rebuild
    } else if merged_rows > gone_rows {
        Plan::Wait
    } else {
        Plan::Rebuild
    }
}

/// How long `Plan::Wait` may last before rebuilding from whatever is on disk.
const MAX_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

/// The run's data files with sizes and footers; reuses `old` for unchanged
/// files. Files that vanish or have no valid footer yet are skipped.
/// A final.parquet still being written (by scrapers before atomic compaction
/// output) is passed over in favour of the segments it will replace.
fn list_files(run_dir: &Path, old: &[FileInfo]) -> Vec<FileInfo> {
    let info = |path: PathBuf| -> Option<FileInfo> {
        let len = std::fs::metadata(&path).ok()?.len();
        if let Some(o) = old.iter().find(|o| o.path == path && o.len == len) {
            return Some(o.clone());
        }
        let file = std::fs::File::open(&path).ok()?;
        let meta = ArrowReaderMetadata::load(&file, Default::default()).ok()?;
        let m = meta.metadata();
        Some(FileInfo {
            path,
            len,
            rows: m.file_metadata().num_rows().max(0) as u64,
            groups: m.num_row_groups(),
        })
    };
    if let Some(f) = final_files(run_dir).into_iter().find_map(info) {
        return vec![f];
    }
    segment_files(run_dir)
        .into_iter()
        .filter_map(info)
        .collect()
}

/// Series parsed from part of a file, in first-seen key order.
#[derive(Default)]
struct Chunk {
    keys: Vec<String>,
    series: Vec<Series>,
}

/// Parse every row group of `files` on scoped OS threads (not rayon: callers
/// hold a run's refresh lock, and a rayon wait could steal a job that needs it).
/// Chunks are handed to `sink` in file/row-group order as they finish, so only
/// a few parsed chunks exist beside the data they are merged into.
fn read_files(files: &[FileInfo], mut sink: impl FnMut(Chunk)) -> anyhow::Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let tasks: Vec<(&Path, usize)> = files
        .iter()
        .flat_map(|f| (0..f.groups).map(move |g| (f.path.as_path(), g)))
        .collect();
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(tasks.len());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::scope(|s| {
        for _ in 0..threads {
            let tx = tx.clone();
            let (tasks, next) = (&tasks, &next);
            s.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(path, group)) = tasks.get(i) else {
                    break;
                };
                if tx.send((i, read_group(path, group))).is_err() {
                    break;
                }
            });
        }
        drop(tx);
        let mut pending = BTreeMap::new();
        let mut want = 0;
        for (i, chunk) in rx {
            pending.insert(i, chunk);
            while let Some(chunk) = pending.remove(&want) {
                want += 1;
                match chunk {
                    Ok(c) => sink(c),
                    Err(e) => {
                        next.store(usize::MAX / 2, Ordering::Relaxed);
                        return Err(e);
                    }
                }
            }
        }
        Ok(())
    })
}

fn label(a: &StringArray, i: usize) -> Option<&str> {
    a.is_valid(i).then(|| a.value(i))
}

fn read_group(path: &Path, group: usize) -> anyhow::Result<Chunk> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(path)?)?
        .with_row_groups(vec![group])
        .with_batch_size(65536)
        .build()?;
    let mut chunk = Chunk::default();
    let mut index: HashMap<String, usize> = HashMap::new();
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
            // Compacted files are sorted by metric, so consecutive rows usually
            // share a series: compare the raw columns before building a key.
            let same = last.is_some()
                && ep.value(i) == ep.value(i - 1)
                && name.value(i) == name.value(i - 1)
                && extras.iter().all(|(_, a)| label(a, i) == label(a, i - 1));
            let idx = match last.filter(|_| same) {
                Some(l) => l,
                None => {
                    key.clear();
                    key.push_str(ep.value(i));
                    key.push('/');
                    key.push_str(name.value(i));
                    let mut first = true;
                    for (n, a) in &extras {
                        if let Some(v) = label(a, i).filter(|v| !v.is_empty()) {
                            key.push_str(if first { " [" } else { "," });
                            first = false;
                            key.push_str(n);
                            key.push('=');
                            key.push_str(v);
                        }
                    }
                    if !first {
                        key.push(']');
                    }
                    match index.get(key.as_str()) {
                        Some(&l) => l,
                        None => {
                            let l = chunk.series.len();
                            index.insert(key.clone(), l);
                            chunk.keys.push(key.clone());
                            chunk.series.push(Series::default());
                            l
                        }
                    }
                }
            };
            last = Some(idx);
            let s = &mut chunk.series[idx];
            s.t.push(t.value(i) as f32);
            s.v.push(value.value(i) as f32);
        }
    }
    Ok(chunk)
}

/// Bumped whenever any run's key list may have changed (see `RunData::gen`).
static KEY_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_gen() -> u64 {
    KEY_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// All series of one run, keyed `endpoint/metric_name [col=val,...]`.
#[derive(Default)]
pub struct RunData {
    files: Vec<FileInfo>,
    index: HashMap<String, usize>,
    pub keys: Vec<String>,
    pub series: Vec<Series>,
    anchors: Mutex<HashMap<String, Option<f64>>>,
    /// Changes whenever `keys` may have; clients skip refetching key lists otherwise.
    gen: u64,
}

impl RunData {
    pub fn get(&self, key: &str) -> Option<&Series> {
        self.index.get(key).map(|&i| &self.series[i])
    }

    /// Read `files` into the series as they parse.
    fn absorb(&mut self, files: &[FileInfo]) -> anyhow::Result<()> {
        let before = self.keys.len();
        let mut touched = HashSet::new();
        read_files(files, |c| self.merge(c, &mut touched))?;
        self.finish(before, touched);
        Ok(())
    }

    fn absorb_chunks(&mut self, chunks: Vec<Chunk>) {
        let before = self.keys.len();
        let mut touched = HashSet::new();
        for c in chunks {
            self.merge(c, &mut touched);
        }
        self.finish(before, touched);
    }

    fn merge(&mut self, c: Chunk, touched: &mut HashSet<usize>) {
        for (k, s) in c.keys.into_iter().zip(c.series) {
            let i = match self.index.get(&k) {
                Some(&i) => i,
                None => {
                    let i = self.series.len();
                    self.index.insert(k.clone(), i);
                    self.keys.push(k);
                    self.series.push(Series::default());
                    i
                }
            };
            let d = &mut self.series[i];
            if d.t.is_empty() {
                *d = s;
            } else {
                d.t.extend(s.t);
                d.v.extend(s.v);
            }
            touched.insert(i);
        }
    }

    /// Restore time order in merged series and drop their growth slack
    /// (thousands of series otherwise hold ~2x their data).
    fn finish(&mut self, keys_before: usize, touched: HashSet<usize>) {
        for i in touched {
            let s = &mut self.series[i];
            if !s.t.windows(2).all(|w| w[0] <= w[1]) {
                let mut idx: Vec<usize> = (0..s.t.len()).collect();
                idx.sort_by(|&a, &b| s.t[a].total_cmp(&s.t[b]));
                s.t = idx.iter().map(|&i| s.t[i]).collect();
                s.v = idx.iter().map(|&i| s.v[i]).collect();
            }
            s.t.shrink_to_fit();
            s.v.shrink_to_fit();
        }
        if self.keys.len() != keys_before || self.gen == 0 {
            self.gen = next_gen();
        }
        self.anchors.lock().unwrap().clear();
    }

    /// Load `run_dir` from scratch (tests and one-off tools).
    pub fn load(run_dir: &Path) -> anyhow::Result<RunData> {
        let files = list_files(run_dir, &[]);
        let mut d = RunData::default();
        d.absorb(&files)?;
        d.files = files;
        Ok(d)
    }

    /// Time of the anchor event: first sample of any series matching `re`,
    /// or the first one whose value exceeds `above`.
    fn anchor(&self, spec: &Align, re: &Regex) -> Option<f64> {
        let id = format!("{}\u{0}{:?}", spec.key, spec.above);
        if let Some(v) = self.anchors.lock().unwrap().get(&id) {
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
        self.anchors.lock().unwrap().insert(id, best);
        best
    }
}

/// Bring `data` up to date with `run_dir`. Files are parsed without holding
/// the data lock, so readers keep the previous snapshot meanwhile; a failure
/// leaves that snapshot untouched. Returns false if it should be retried soon.
fn refresh(
    data: &RwLock<RunData>,
    run_dir: &Path,
    waiting: &mut Option<std::time::Instant>,
) -> anyhow::Result<bool> {
    let old = data.read().unwrap().files.clone();
    let cur = list_files(run_dir, &old);
    let mut p = plan(&old, &cur);
    if p == Plan::Wait {
        let since = *waiting.get_or_insert_with(std::time::Instant::now);
        if since.elapsed() < MAX_WAIT {
            return Ok(false);
        }
        p = Plan::Rebuild;
    }
    *waiting = None;
    match p {
        Plan::Same | Plan::Wait => {}
        Plan::Append(fresh) | Plan::Swap(fresh) => {
            let new: Vec<FileInfo> = fresh.iter().map(|&i| cur[i].clone()).collect();
            let mut chunks = Vec::new();
            read_files(&new, |c| chunks.push(c))?;
            let mut d = data.write().unwrap();
            d.absorb_chunks(chunks);
            d.files = cur;
        }
        Plan::Rebuild => {
            let mut d = RunData::default();
            d.absorb(&cur)?;
            d.files = cur;
            let mut cur = data.write().unwrap();
            if d.keys == cur.keys {
                d.gen = cur.gen;
            }
            let old = std::mem::replace(&mut *cur, d);
            drop(cur);
            drop(old);
            release_memory();
        }
    }
    Ok(true)
}

/// Hand freed parse buffers back to the OS. glibc keeps them in per-thread
/// arenas otherwise, nearly doubling RSS after loading big runs.
fn release_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
}

type Shared = Arc<App>;

#[derive(Default)]
struct Entry {
    data: Arc<RwLock<RunData>>,
    loaded: std::sync::atomic::AtomicBool,
    /// Held while refreshing; when a compaction wait began.
    refresh: Mutex<Option<std::time::Instant>>,
}

struct App {
    root: PathBuf,
    cache: Mutex<HashMap<String, Arc<Entry>>>,
    dirty: Mutex<HashSet<String>>,
    tx: broadcast::Sender<String>,
}

impl App {
    fn new(root: PathBuf) -> App {
        App {
            root,
            cache: Mutex::new(HashMap::new()),
            dirty: Mutex::new(HashSet::new()),
            tx: broadcast::channel(4096).0,
        }
    }

    /// A run's data, refreshed if the watcher saw it change. Once a run has
    /// loaded, requests never wait on (or see the middle of) a refresh: they
    /// read the last complete snapshot.
    fn run(&self, name: &str) -> anyhow::Result<Arc<RwLock<RunData>>> {
        use std::sync::atomic::Ordering::{Acquire, Release};
        if !valid_name(name) || !self.root.join(name).join(META_FILE).is_file() {
            anyhow::bail!("no such run");
        }
        let entry = self
            .cache
            .lock()
            .unwrap()
            .entry(name.to_string())
            .or_default()
            .clone();
        let stale = self.dirty.lock().unwrap().remove(name);
        let loaded = entry.loaded.load(Acquire);
        if loaded && !stale {
            return Ok(entry.data.clone());
        }
        let mut waiting = if loaded {
            match entry.refresh.try_lock() {
                Ok(g) => g,
                Err(_) => {
                    // Another request is refreshing; it may predate this change.
                    self.dirty.lock().unwrap().insert(name.to_string());
                    return Ok(entry.data.clone());
                }
            }
        } else {
            let g = entry.refresh.lock().unwrap();
            if entry.loaded.load(Acquire) && !stale {
                return Ok(entry.data.clone());
            }
            g
        };
        match refresh(&entry.data, &self.root.join(name), &mut waiting) {
            Ok(done) => {
                entry.loaded.store(true, Release);
                if !done {
                    self.dirty.lock().unwrap().insert(name.to_string());
                }
            }
            Err(e) if loaded => {
                log::warn!("refresh {name}: {e}; serving the previous snapshot");
                self.dirty.lock().unwrap().insert(name.to_string());
            }
            Err(e) => return Err(e),
        }
        Ok(entry.data.clone())
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
            // A held flock means a scraper is alive even if its heartbeat lags.
            let live = meta.is_live(now)
                || (meta.status == "running" && is_locked(&e.path()) == Some(true));
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
    /// `etag` of the client's last response; unchanged keys come back as `same`.
    #[serde(default)]
    etag: Option<String>,
}

#[derive(Serialize, Default)]
struct KeysResp {
    keys: Vec<String>,
    /// Histogram metrics (see `hist::bucket_of`); their bucket series are not in `keys`.
    hists: Vec<String>,
    errors: BTreeMap<String, String>,
    etag: String,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    same: bool,
}

async fn keys(State(app): State<Shared>, Json(req): Json<KeysReq>) -> Response {
    let out = tokio::task::spawn_blocking(move || {
        let per_run: Vec<(String, anyhow::Result<Arc<RwLock<RunData>>>)> = req
            .runs
            .par_iter()
            .map(|r| (r.clone(), app.run(r)))
            .collect();
        let mut out = KeysResp::default();
        let mut guards = Vec::new();
        let mut tag = std::collections::hash_map::DefaultHasher::new();
        for (r, res) in &per_run {
            std::hash::Hash::hash(r, &mut tag);
            match res {
                Ok(d) => {
                    let g = d.read().unwrap();
                    std::hash::Hash::hash(&(g.gen, g.keys.len()), &mut tag);
                    guards.push(g);
                }
                Err(e) => {
                    std::hash::Hash::hash(&e.to_string(), &mut tag);
                    out.errors.insert(r.clone(), e.to_string());
                }
            }
        }
        out.etag = format!("{:016x}", std::hash::Hasher::finish(&tag));
        if req.etag.as_deref() == Some(out.etag.as_str()) {
            out.same = true;
            out.errors.clear();
            return out;
        }
        let (mut keys, mut hists) = (BTreeSet::new(), BTreeSet::new());
        for g in &guards {
            for k in &g.keys {
                match crate::hist::bucket_of(k) {
                    Some((base, _)) => hists.insert(base),
                    None => keys.insert(k.as_str()),
                };
            }
        }
        out.keys = keys.into_iter().map(str::to_string).collect();
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
    data: &RunData,
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
    let Some(s) = data.get(&req.key) else {
        return Ok(None);
    };
    Ok(map_series(data, s, offset, req).map(|m| (m, offset)))
}

/// Smooth `s` and put it on the plot's x-axis (aligned time, or `x_key`'s value).
fn map_series(data: &RunData, s: &Series, offset: f64, req: &PlotReq) -> Option<Mapped> {
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
            let xs = data.get(xk)?;
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
    Some(Mapped { x, v })
}

/// A run named in run('...'): exact, else the one run whose name contains it.
fn resolve_run(app: &App, pat: &str) -> Result<String, String> {
    let is_run = |n: &str| valid_name(n) && app.root.join(n).join(META_FILE).is_file();
    if is_run(pat) {
        return Ok(pat.to_string());
    }
    let mut hits: Vec<String> = std::fs::read_dir(&app.root)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.contains(pat) && is_run(n))
        .collect();
    hits.sort();
    match hits.len() {
        0 => Err(format!("run('{pat}'): no run has that in its name")),
        1 => Ok(hits.remove(0)),
        n => Err(format!(
            "run('{pat}') matches {n} runs ({}, {}, ...); use more of the name",
            hits[0], hits[1]
        )),
    }
}

/// Evaluate a formula in each of `runs`, or once when it only reads series
/// inside run('name', ...) (then it's one line, named by the formula, already
/// on the aligned axis with offset 0). `f(line name, run data, T=0 offset, series)`.
#[allow(clippy::type_complexity)]
fn eval_formula<T: Send>(
    app: &App,
    runs: &[String],
    src: &str,
    align: Option<&Align>,
    f: impl Fn(&str, &RunData, f64, Series) -> T + Sync,
) -> Result<Vec<(String, Result<Option<T>, String>)>, String> {
    let e = crate::expr::parse(src)?;
    let cross = !crate::expr::reads_own_run(&e);
    let ext: Vec<(String, String)> = crate::expr::run_refs(&e)
        .into_iter()
        .map(|p| resolve_run(app, &p).map(|n| (p, n)))
        .collect::<Result<_, _>>()?;
    let anchor_re = align.and_then(|a| Regex::new(&a.key).ok());
    let offset_of = |d: &RunData| match (align, anchor_re.as_ref()) {
        (Some(a), Some(re)) => d.anchor(a, re),
        _ => Some(0.0),
    };
    if ext.is_empty() {
        let none = HashMap::new();
        return Ok(runs
            .par_iter()
            .map(|r| {
                let res = app.run(r).map_err(|e| e.to_string()).and_then(|d| {
                    let d = d.read().unwrap();
                    let Some(o) = offset_of(&d) else {
                        return Ok(None);
                    };
                    let scope = crate::expr::Scope {
                        keys: &d.keys,
                        series: &d.series,
                        offset: o,
                        ext: &none,
                    };
                    let s = crate::expr::eval_parsed(&e, &scope)?;
                    Ok(s.map(|s| f(r, &d, o, s)))
                });
                (r.clone(), res)
            })
            .collect());
    }
    let mut names: Vec<String> = if cross { Vec::new() } else { runs.to_vec() };
    names.extend(ext.iter().map(|(_, n)| n.clone()));
    names.sort();
    names.dedup();
    // Refresh and anchor under each run's write lock, then take one read guard
    // per run in name order (a fixed order, so concurrent requests can't
    // deadlock) and evaluate on this thread: no rayon while holding guards, as
    // a stolen job could wait for a write lock on a run we hold.
    type Loaded = Result<(Arc<RwLock<RunData>>, Option<f64>), String>;
    let loaded: HashMap<String, Loaded> = names
        .par_iter()
        .map(|n| {
            let r = app.run(n).map_err(|e| e.to_string()).map(|d| {
                let o = offset_of(&d.read().unwrap());
                (d, o)
            });
            (n.clone(), r)
        })
        .collect();
    for (p, n) in &ext {
        match &loaded[n] {
            Err(e) => return Err(format!("run('{p}'): {e}")),
            Ok((_, None)) => return Err(format!("run('{p}'): {n} has no T=0 anchor")),
            Ok(_) => {}
        }
    }
    let guards: HashMap<&str, (std::sync::RwLockReadGuard<RunData>, f64)> = names
        .iter()
        .filter_map(|n| {
            let (d, o) = loaded[n].as_ref().ok()?;
            Some((n.as_str(), (d.read().unwrap(), (*o)?)))
        })
        .collect();
    let ext_map: HashMap<String, crate::expr::Ext> = ext
        .iter()
        .map(|(p, n)| {
            let (g, o) = &guards[n.as_str()];
            let x = crate::expr::Ext {
                keys: &g.keys,
                series: &g.series,
                offset: *o,
            };
            (p.clone(), x)
        })
        .collect();
    if cross {
        let empty = RunData::default();
        let scope = crate::expr::Scope {
            keys: &[],
            series: &[],
            offset: 0.0,
            ext: &ext_map,
        };
        let res = match crate::expr::eval_parsed(&e, &scope) {
            Ok(None) => {
                Err("the runs named in run(...) don't have the series this formula reads".into())
            }
            r => r.map(|s| s.map(|s| f(src, &empty, 0.0, s))),
        };
        return Ok(vec![(src.to_string(), res)]);
    }
    Ok(runs
        .iter()
        .map(|r| {
            let res = match (loaded.get(r), guards.get(r.as_str())) {
                (Some(Err(e)), _) => Err(e.clone()),
                (_, None) => Ok(None),
                (_, Some((g, o))) => {
                    let scope = crate::expr::Scope {
                        keys: &g.keys,
                        series: &g.series,
                        offset: *o,
                        ext: &ext_map,
                    };
                    crate::expr::eval_parsed(&e, &scope).map(|s| s.map(|s| f(r, g, *o, s)))
                }
            };
            (r.clone(), res)
        })
        .collect())
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
    type Line = (String, Result<Option<(Mapped, f64)>, String>);
    let mapped: Vec<Line> = match &req.expr {
        Some(src) => {
            let cross = crate::expr::parse(src).is_ok_and(|e| !crate::expr::reads_own_run(&e));
            let res = if cross && req.x_key.is_some() {
                Err("a formula across runs needs time on the x axis".to_string())
            } else {
                eval_formula(app, &req.runs, src, req.align.as_ref(), |_, d, o, s| {
                    map_series(d, &s, o, &req).map(|m| (m, o))
                })
            };
            match res {
                Ok(v) => v
                    .into_iter()
                    .map(|(n, r)| (n, r.map(Option::flatten)))
                    .collect(),
                Err(e) => vec![(src.clone(), Err(e))],
            }
        }
        None => req
            .runs
            .par_iter()
            .map(|r| {
                let res = app
                    .run(r)
                    .map_err(|e| e.to_string())
                    .and_then(|d| map_run(&d.read().unwrap(), &req, anchor_re.as_ref()));
                (r.clone(), res)
            })
            .collect(),
    };
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
    let limit = req.limit.unwrap_or(200).min(5000);
    let (t0, t1) = (
        req.t0.unwrap_or(f64::NEG_INFINITY),
        req.t1.unwrap_or(f64::INFINITY),
    );
    if let Some(src) = &req.expr {
        let (mut rows, mut error) = (Vec::new(), None);
        match eval_formula(app, &req.runs, src, req.align.as_ref(), |r, _, o, s| {
            summarize(r, src, &s, o, t0, t1)
        }) {
            Ok(v) => {
                for (_, res) in v {
                    match res {
                        Ok(Some(Some(row))) => rows.push(row),
                        Ok(_) => {}
                        Err(e) => error = Some(e),
                    }
                }
            }
            Err(e) => error = Some(e),
        }
        rows.sort_by(|a: &StatRow, b| a.run.cmp(&b.run));
        let truncated = rows.len() > limit;
        rows.truncate(limit);
        return StatsResp {
            rows,
            truncated,
            error,
        };
    }
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
    let err = std::sync::Mutex::new(None::<String>);
    let mut rows: Vec<StatRow> = req
        .runs
        .par_iter()
        .flat_map_iter(|r| {
            let mut out = Vec::new();
            let Ok(d) = app.run(r) else { return out };
            let d = d.read().unwrap();
            let offset = match (&req.align, anchor_re.as_ref()) {
                (Some(a), Some(are)) => match d.anchor(a, are) {
                    Some(o) => o,
                    None => return out,
                },
                _ => 0.0,
            };
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
                let d = d.read().unwrap();
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

#[derive(Deserialize)]
struct HistStatsReq {
    runs: Vec<String>,
    /// Histogram base key (optional `#`), exact or a regex.
    key: String,
    /// Window in aligned time; default the whole run.
    #[serde(default)]
    t0: Option<f64>,
    #[serde(default)]
    t1: Option<f64>,
    #[serde(default)]
    align: Option<Align>,
    /// Default p50, p90, p99.
    #[serde(default)]
    quantiles: Option<Vec<f64>>,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Serialize)]
struct HistStatRow {
    run: String,
    key: String,
    t0: f64,
    t1: f64,
    observations: f64,
    per_s: f64,
    /// `p50` -> value; `null` without observations.
    quantiles: BTreeMap<String, Option<f64>>,
    /// Bucket upper bounds and observations in each, over the window.
    le: Vec<Option<f64>>,
    counts: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize, Default)]
struct HistStatsResp {
    rows: Vec<HistStatRow>,
    truncated: bool,
    error: Option<String>,
}

fn qname(q: f64) -> String {
    format!("p{}", (q * 1e4).round() / 1e2)
}

/// Per run x histogram: observations, rate and quantiles over one window.
fn hist_stats_blocking(app: &App, req: HistStatsReq) -> HistStatsResp {
    let pat = req.key.strip_prefix('#').unwrap_or(&req.key).to_string();
    let re = match Regex::new(&pat) {
        Ok(r) => r,
        Err(e) => {
            return HistStatsResp {
                error: Some(e.to_string()),
                ..Default::default()
            }
        }
    };
    let qs = req.quantiles.clone().unwrap_or(vec![0.5, 0.9, 0.99]);
    if let Some(q) = qs.iter().find(|q| !(0.0..=1.0).contains(*q)) {
        return HistStatsResp {
            error: Some(format!("quantile {q} must be in [0, 1]")),
            ..Default::default()
        };
    }
    let anchor_re = req.align.as_ref().and_then(|a| Regex::new(&a.key).ok());
    let limit = req.limit.unwrap_or(200).min(5000);
    let mut rows: Vec<HistStatRow> = req
        .runs
        .par_iter()
        .flat_map_iter(|r| {
            let mut out = Vec::new();
            let Ok(d) = app.run(r) else { return out };
            let d = d.read().unwrap();
            let offset = match (&req.align, anchor_re.as_ref()) {
                (Some(a), Some(are)) => match d.anchor(a, are) {
                    Some(o) => o,
                    None => return out,
                },
                _ => 0.0,
            };
            let mut bases: Vec<String> = d
                .keys
                .iter()
                .filter_map(|k| crate::hist::bucket_of(k).map(|(b, _)| b))
                .collect();
            bases.sort();
            bases.dedup();
            let exact = bases.contains(&pat);
            for base in bases
                .iter()
                .filter(|b| if exact { **b == pat } else { re.is_match(b) })
            {
                let b = hist_buckets(&d, base);
                let Some((_, inf)) = b.last().filter(|(le, _)| le.is_infinite()) else {
                    continue;
                };
                let (Some(&first), Some(&last)) = (inf.t.first(), inf.t.last()) else {
                    continue;
                };
                let w0 = req
                    .t0
                    .unwrap_or(f64::NEG_INFINITY)
                    .max(first as f64 - offset);
                let w1 = req.t1.unwrap_or(f64::INFINITY).min(last as f64 - offset);
                if w1 <= w0 {
                    continue;
                }
                // One window, gaps included: totals, not a heatmap.
                let delta = |bs: &[(f64, &Series)]| {
                    crate::hist::slices(bs, offset, &[w0, w1], f64::INFINITY)
                        .pop()
                        .flatten()
                };
                let counts = delta(&b);
                let mut cum = 0.0;
                let cums: Vec<f64> = counts
                    .iter()
                    .flatten()
                    .map(|&c| {
                        cum += c as f64;
                        cum
                    })
                    .collect();
                let les: Vec<f64> = b.iter().map(|(le, _)| *le).collect();
                out.push(HistStatRow {
                    run: r.clone(),
                    key: format!("#{base}"),
                    t0: w0,
                    t1: w1,
                    observations: cum,
                    per_s: cum / (w1 - w0),
                    quantiles: qs
                        .iter()
                        .map(|&q| (qname(q), crate::hist::quantile(&les, &cums, q)))
                        .collect(),
                    le: les.iter().map(|le| le.is_finite().then_some(*le)).collect(),
                    counts: counts
                        .unwrap_or_default()
                        .into_iter()
                        .map(|c| c as f64)
                        .collect(),
                    error: cums
                        .is_empty()
                        .then(|| "counter reset inside the window; narrow t0/t1".into()),
                });
            }
            out
        })
        .collect();
    rows.sort_by(|a, b| a.key.cmp(&b.key).then(a.run.cmp(&b.run)));
    let truncated = rows.len() > limit;
    rows.truncate(limit);
    HistStatsResp {
        rows,
        truncated,
        error: None,
    }
}

async fn hist_stats(State(app): State<Shared>, Json(req): Json<HistStatsReq>) -> Response {
    let out = tokio::task::spawn_blocking(move || hist_stats_blocking(&app, req))
        .await
        .unwrap_or_default();
    Json(out).into_response()
}

async fn hist(State(app): State<Shared>, Json(req): Json<HistReq>) -> Response {
    let out = tokio::task::spawn_blocking(move || hist_blocking(&app, req))
        .await
        .unwrap_or_default();
    Json(out).into_response()
}

// ---------- dashboards: <root>/.dashboards/<name>.json, versioned, written under flock ----------
const DASH_DIR: &str = ".dashboards";

fn dash_name_ok(n: &str) -> bool {
    !n.is_empty()
        && n.len() <= 80
        && !n.starts_with('.')
        && n.chars().all(|c| c.is_alphanumeric() || "-_. ".contains(c))
}

#[derive(Serialize, Deserialize)]
struct Dash {
    version: u64,
    saved_at_ms: i64,
    state: serde_json::Value,
}

#[derive(Deserialize)]
struct DashPut {
    /// Version the client loaded (`null`: creating a new dashboard).
    base_version: Option<u64>,
    /// Overwrite even if someone saved since `base_version`.
    #[serde(default)]
    force: bool,
    state: serde_json::Value,
}

fn read_dash(p: &Path) -> Option<Dash> {
    serde_json::from_slice(&std::fs::read(p).ok()?).ok()
}

/// Run `f` holding an exclusive flock on `.dashboards/.lock`.
fn with_dash_lock<T>(root: &Path, f: impl FnOnce(&Path) -> T) -> std::io::Result<T> {
    let dir = root.join(DASH_DIR);
    std::fs::create_dir_all(&dir)?;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join(".lock"))?;
    lock.lock()?;
    Ok(f(&dir))
}

fn err(code: StatusCode, msg: impl ToString) -> Response {
    (code, Json(serde_json::json!({ "error": msg.to_string() }))).into_response()
}

async fn dash_list(State(app): State<Shared>) -> Response {
    let dir = app.root.join(DASH_DIR);
    let mut out: Vec<serde_json::Value> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let file = e.file_name().to_string_lossy().to_string();
            let name = file.strip_suffix(".json")?;
            let d = dash_name_ok(name).then(|| read_dash(&e.path()))??;
            Some(serde_json::json!({ "name": name, "version": d.version, "saved_at_ms": d.saved_at_ms }))
        })
        .collect();
    out.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    Json(out).into_response()
}

async fn dash_get(State(app): State<Shared>, UrlPath(name): UrlPath<String>) -> Response {
    if !dash_name_ok(&name) {
        return err(StatusCode::BAD_REQUEST, "bad dashboard name");
    }
    match read_dash(&app.root.join(DASH_DIR).join(format!("{name}.json"))) {
        Some(d) => Json(serde_json::json!({
            "name": name, "version": d.version, "saved_at_ms": d.saved_at_ms, "state": d.state
        }))
        .into_response(),
        None => err(StatusCode::NOT_FOUND, "no such dashboard"),
    }
}

async fn dash_put(
    State(app): State<Shared>,
    UrlPath(name): UrlPath<String>,
    Json(req): Json<DashPut>,
) -> Response {
    if !dash_name_ok(&name) {
        return err(
            StatusCode::BAD_REQUEST,
            "dashboard names: letters, digits, space, - _ . (max 80)",
        );
    }
    let root = app.root.clone();
    let res = tokio::task::spawn_blocking(move || {
        with_dash_lock(&root, |dir| -> std::io::Result<Result<u64, Option<u64>>> {
            let path = dir.join(format!("{name}.json"));
            let cur = read_dash(&path).map(|d| d.version);
            if !req.force && req.base_version != cur {
                return Ok(Err(cur));
            }
            let d = Dash {
                version: cur.unwrap_or(0) + 1,
                saved_at_ms: now_ms(),
                state: req.state,
            };
            let tmp = dir.join(format!(".{name}.json.tmp"));
            std::fs::write(&tmp, serde_json::to_vec_pretty(&d)?)?;
            std::fs::rename(&tmp, &path)?;
            Ok(Ok(d.version))
        })
    })
    .await;
    match res {
        Ok(Ok(Ok(Ok(version)))) => Json(serde_json::json!({ "version": version })).into_response(),
        Ok(Ok(Ok(Err(cur)))) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": match cur {
                    Some(_) => "saved by someone else since you loaded it",
                    None => "it was deleted since you loaded it",
                },
                "version": cur,
            })),
        )
            .into_response(),
        Ok(Ok(Err(e))) | Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

async fn dash_del(State(app): State<Shared>, UrlPath(name): UrlPath<String>) -> Response {
    if !dash_name_ok(&name) {
        return err(StatusCode::BAD_REQUEST, "bad dashboard name");
    }
    let root = app.root.clone();
    let res = tokio::task::spawn_blocking(move || {
        with_dash_lock(&root, |dir| {
            std::fs::remove_file(dir.join(format!("{name}.json")))
        })
    })
    .await;
    match res {
        Ok(Ok(Ok(()))) => StatusCode::NO_CONTENT.into_response(),
        Ok(Ok(Err(e))) if e.kind() == std::io::ErrorKind::NotFound => {
            err(StatusCode::NOT_FOUND, "no such dashboard")
        }
        Ok(Ok(Err(e))) | Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
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
        let scan = tokio::task::spawn_blocking(move || {
            let dirs: Vec<String> = std::fs::read_dir(&root)
                .ok()?
                .flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| valid_name(n))
                .collect();
            let sigs = dirs
                .into_par_iter()
                .map(|n| {
                    let s = signature(&root.join(&n));
                    (n, s)
                })
                .collect::<Vec<_>>();
            Some(sigs)
        })
        .await;
        // A failed scan (e.g. an NFS hiccup) says nothing about the runs:
        // keep their caches rather than dropping and reloading them.
        let Ok(Some(sigs)) = scan else {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            continue;
        };
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
        app.cache
            .lock()
            .unwrap()
            .retain(|n, _| names.contains(n) || app.root.join(n).join(META_FILE).is_file());
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}

/// The UI page with its default theme: plain, or the niu-lai look.
fn index_page(niu_lai: bool) -> String {
    if niu_lai {
        INDEX.replacen(
            "<html lang=\"en\">",
            "<html lang=\"en\" data-theme=\"niu\">",
            1,
        )
    } else {
        INDEX
            .replacen(
                "<html lang=\"en\">",
                "<html lang=\"en\" data-theme=\"plain\">",
                1,
            )
            .replacen("<body>", "<body class=\"plain\">", 1)
    }
}

pub fn router(root: PathBuf, niu_lai: bool) -> Router {
    let page: &'static str = index_page(niu_lai).leak();
    let app = Arc::new(App::new(root));
    tokio::spawn(watch(app.clone()));
    Router::new()
        .route("/", get(move || async move { Html(page) }))
        .route("/vendor/marked.js", get(|| async { js(MARKED_JS) }))
        .route("/vendor/purify.js", get(|| async { js(PURIFY_JS) }))
        .route("/api/runs", get(runs))
        .route("/api/keys", post(keys))
        .route("/api/plot", post(plot))
        .route("/api/stats", post(stats))
        .route("/api/hist", post(hist))
        .route("/api/hist_stats", post(hist_stats))
        .route("/api/dashboards", get(dash_list))
        .route(
            "/api/dashboards/{name}",
            get(dash_get).put(dash_put).delete(dash_del),
        )
        .route("/api/events", get(events))
        .with_state(app)
}

pub async fn serve(root: PathBuf, addr: &str, niu_lai: bool) -> anyhow::Result<()> {
    std::fs::create_dir_all(&root)?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    log::info!(
        "niu-lai-viz serving {} on http://{}",
        root.display(),
        listener.local_addr()?
    );
    axum::serve(listener, router(root, niu_lai)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Float64Array, StringArray};
    use arrow::record_batch::RecordBatch;

    /// `n` samples each of `metrics` series, starting at time `t0`.
    fn write(path: &Path, metrics: usize, t0: usize, n: usize) {
        let mut ep = Vec::new();
        let mut name = Vec::new();
        let mut v = Vec::new();
        let mut t = Vec::new();
        for m in 0..metrics {
            for i in t0..t0 + n {
                ep.push("trainer".to_string());
                name.push(format!("m{m}"));
                v.push(i as f64);
                t.push(i as f64);
            }
        }
        let batch = RecordBatch::try_from_iter([
            (
                "scraper_endpoint",
                Arc::new(StringArray::from(ep)) as ArrayRef,
            ),
            ("metric_name", Arc::new(StringArray::from(name)) as ArrayRef),
            ("metric_value", Arc::new(Float64Array::from(v)) as ArrayRef),
            (
                "time_since_start",
                Arc::new(Float64Array::from(t)) as ArrayRef,
            ),
        ])
        .unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let tmp = path.with_extension("tmp");
        let mut w = parquet::arrow::ArrowWriter::try_new(
            std::fs::File::create(&tmp).unwrap(),
            batch.schema(),
            None,
        )
        .unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
        std::fs::rename(tmp, path).unwrap();
    }

    fn setup() -> (tempfile::TempDir, App) {
        let root = tempfile::tempdir().unwrap();
        let run = root.path().join("r");
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(run.join(META_FILE), "{}").unwrap();
        let app = App::new(root.path().to_path_buf());
        (root, app)
    }

    fn total(app: &App) -> (usize, usize) {
        let d = app.run("r").unwrap();
        let d = d.read().unwrap();
        (d.keys.len(), d.series.iter().map(|s| s.t.len()).sum())
    }

    fn touch(app: &App) {
        app.dirty.lock().unwrap().insert("r".into());
    }

    #[test]
    fn compaction_never_hides_or_duplicates_data() {
        let (root, app) = setup();
        let seg = root.path().join("r/segments");
        write(&seg.join("out-0.parquet"), 50, 0, 10);
        write(&seg.join("out-1.parquet"), 50, 10, 10);
        assert_eq!(total(&app), (50, 1000));

        // Merged output appears before its sources are deleted.
        write(&seg.join("incomplete-0.parquet"), 50, 0, 20);
        touch(&app);
        assert_eq!(total(&app), (50, 1000));
        // Sources deleted one at a time, a new segment arriving meanwhile.
        std::fs::remove_file(seg.join("out-0.parquet")).unwrap();
        write(&seg.join("out-2.parquet"), 50, 20, 10);
        touch(&app);
        assert_eq!(total(&app), (50, 1000));
        std::fs::remove_file(seg.join("out-1.parquet")).unwrap();
        touch(&app);
        assert_eq!(total(&app), (50, 1500));
        let gen = app.run("r").unwrap().read().unwrap().gen;

        // A half-written segment is ignored, not fatal.
        std::fs::write(seg.join("out-3.parquet"), b"PAR1garbage").unwrap();
        touch(&app);
        assert_eq!(total(&app), (50, 1500));
        std::fs::remove_file(seg.join("out-3.parquet")).unwrap();

        // An old scraper's half-written final.parquet is passed over.
        std::fs::write(seg.join("final.parquet"), b"PAR1garbage").unwrap();
        touch(&app);
        assert_eq!(total(&app), (50, 1500));

        // Final compaction: segments/final.parquet replaces everything, then moves up.
        write(&seg.join("final.parquet"), 50, 0, 30);
        touch(&app);
        assert_eq!(total(&app), (50, 1500));
        for f in ["incomplete-0.parquet", "out-2.parquet"] {
            std::fs::remove_file(seg.join(f)).unwrap();
        }
        touch(&app);
        assert_eq!(total(&app), (50, 1500));
        std::fs::rename(
            seg.join("final.parquet"),
            root.path().join("r/final.parquet"),
        )
        .unwrap();
        touch(&app);
        assert_eq!(total(&app), (50, 1500));
        // Same keys throughout: clients' key lists stay valid.
        assert_eq!(app.run("r").unwrap().read().unwrap().gen, gen);
        let d = app.run("r").unwrap();
        let d = d.read().unwrap();
        let s = d.get("trainer/m7").unwrap();
        assert!(s.t.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn plan_waits_for_compaction_then_swaps() {
        let f = |name: &str, rows| FileInfo {
            path: PathBuf::from(name),
            len: rows,
            rows,
            groups: 1,
        };
        let old = vec![f("out-0.parquet", 5), f("out-1.parquet", 5)];
        assert_eq!(plan(&old, &old), Plan::Same);
        let mut cur = old.clone();
        cur.push(f("out-2.parquet", 5));
        assert_eq!(plan(&old, &cur), Plan::Append(vec![2]));
        cur.insert(0, f("incomplete-0.parquet", 10));
        assert_eq!(plan(&old, &cur), Plan::Wait);
        cur.remove(1);
        assert_eq!(plan(&old, &cur), Plan::Wait);
        cur.remove(1);
        assert_eq!(plan(&old, &cur), Plan::Swap(vec![1]));
        // Rows lost without a merged replacement: wait (then `MAX_WAIT` reloads).
        let cur = vec![f("incomplete-0.parquet", 8)];
        assert_eq!(plan(&old, &cur), Plan::Wait);
        assert_eq!(plan(&old, &[]), Plan::Wait);
        // Merged output with extra rows and nothing left to delete: reload.
        let cur = vec![f("final.parquet", 12)];
        assert_eq!(plan(&old, &cur), Plan::Rebuild);
    }

    #[test]
    fn concurrent_requests_never_see_an_empty_run() {
        let (root, app) = setup();
        let seg = root.path().join("r/segments");
        for i in 0..20 {
            write(&seg.join(format!("out-{i}.parquet")), 200, i * 10, 10);
        }
        let app = Arc::new(app);
        std::thread::scope(|s| {
            for _ in 0..8 {
                let app = app.clone();
                s.spawn(move || {
                    for _ in 0..50 {
                        touch(&app);
                        assert_eq!(total(&app), (200, 40_000));
                    }
                });
            }
        });
    }
}
