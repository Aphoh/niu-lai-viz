// Derived from tachometer-scraper via NVIDIA/srt-slurm (Apache-2.0); modified for niu-lai-viz. See NOTICE.
use clap::{Parser, Subcommand};
use log::{error, info, warn};
use niu_lai_viz::config::{Config, TargetConfig};
use niu_lai_viz::run::{lock_run, now_ms, RunMeta, Target, FINAL_FILE, SEGMENTS_DIR};
use niu_lai_viz::scrape_endpoint;
use niu_lai_writer::{compact_and_upload, periodic_compact_and_sync, DatasetWriter};
use object_store::ObjectStore;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

#[derive(Parser)]
#[command(
    name = "nlv",
    version,
    about = "Pounds & Prejudice: a poor man's W&B over /metrics",
    after_help = "Typical use: one `scrape` per experiment, one `serve` for all of them:\n  \
                  nlv scrape --root runs --run my-exp --target trainer=http://localhost:8000/metrics --label lr=3e-4\n  \
                  nlv serve --root runs   # then open http://127.0.0.1:8737"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Scrape /metrics endpoints into a new run until Ctrl+C / SIGTERM.
    #[command(after_help = "Examples:\n  \
        nlv scrape --run my-exp --target trainer=http://localhost:8000/metrics\n  \
        nlv scrape --run my-exp --target trainer=http://node1:8000/metrics --target gpus=http://node1:9400/metrics --label lr=3e-4 --label model=8b\n  \
        nlv scrape --config examples/dcgm.toml --run my-exp   # filters/renames per target\n\n\
        The run lives in <root>/<run>/ (a -2, -3... suffix if taken) and is locked (flock on <run>/.lock) while scraping.\n\
        Ctrl+C or SIGTERM ends the run and compacts it into final.parquet.")]
    Scrape {
        /// TOML config (see examples/).
        #[arg(long)]
        config: Option<PathBuf>,
        /// Directory holding runs (default ./runs).
        #[arg(long)]
        root: Option<PathBuf>,
        /// Run name (default: timestamp).
        #[arg(long)]
        run: Option<String>,
        /// Extra target, repeatable.
        #[arg(long = "target", value_name = "NAME=URL")]
        targets: Vec<String>,
        /// Run label (experiment parameter), repeatable.
        #[arg(long = "label", value_name = "KEY=VALUE")]
        labels: Vec<String>,
        /// Scrape interval for --target endpoints.
        #[arg(long, default_value = "1000")]
        interval_ms: u64,
        /// Seconds between sealed segments (UI latency).
        #[arg(long)]
        save_interval: Option<u64>,
        /// Also upload final.parquet to s3://bucket/prefix.
        #[arg(long)]
        storage: Option<String>,
    },
    /// Compact a run's segments into final.parquet (e.g. after a crash).
    Compact { run_dir: PathBuf },
    /// Serve the live UI over a directory of runs.
    Serve {
        #[arg(long, default_value = "./runs")]
        root: PathBuf,
        #[arg(long, default_value = "127.0.0.1:8737")]
        addr: String,
        /// Open in the niu-lai look instead of the plain one (each browser can still toggle it).
        #[arg(long)]
        niu_lai: bool,
    },
}

type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

fn split_pair(s: &str) -> AnyResult<(String, String)> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| format!("expected KEY=VALUE, got {s:?}"))?;
    Ok((k.to_string(), v.to_string()))
}

/// Store that final.parquet is uploaded to: S3 if asked, else the segments dir itself.
fn remote_store(
    storage: Option<&str>,
    seg_dir: &Path,
    run: &str,
) -> AnyResult<(Arc<dyn ObjectStore>, String)> {
    match storage {
        Some(s) if s.starts_with("s3://") => {
            let rest = &s["s3://".len()..];
            let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
            let store = object_store::aws::AmazonS3Builder::from_env()
                .with_bucket_name(bucket)
                .build()?;
            let path = [prefix.trim_matches('/'), run]
                .iter()
                .filter(|p| !p.is_empty())
                .cloned()
                .collect::<Vec<_>>()
                .join("/");
            Ok((Arc::new(store), path))
        }
        Some(s) => Err(format!(
            "unsupported storage {s:?}: only s3:// (the run directory is always local)"
        )
        .into()),
        None => Ok((
            Arc::new(object_store::local::LocalFileSystem::new_with_prefix(
                seg_dir,
            )?),
            String::new(),
        )),
    }
}

async fn finish(run_dir: &Path, storage: Option<&str>) -> AnyResult<()> {
    let seg = run_dir.join(SEGMENTS_DIR);
    let name = run_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("run")
        .to_string();
    let (store, path) = remote_store(storage, &seg, &name)?;
    compact_and_upload(&seg, store, &path).await?;
    let local_final = seg.join(FINAL_FILE);
    if local_final.is_file() {
        std::fs::rename(&local_final, run_dir.join(FINAL_FILE))?;
    }
    Ok(())
}

async fn scrape(
    config: Option<PathBuf>,
    root: Option<PathBuf>,
    run: Option<String>,
    targets: Vec<String>,
    labels: Vec<String>,
    interval_ms: u64,
    save_interval: Option<u64>,
    storage: Option<String>,
) -> AnyResult<()> {
    let mut cfg = match &config {
        Some(p) => Config::from_toml(&std::fs::read_to_string(p)?)?,
        None => Config::default(),
    };
    for t in &targets {
        let (name, url) = split_pair(t)?;
        cfg.targets.push(TargetConfig {
            name,
            url,
            interval_ms: Some(interval_ms),
            rules: Default::default(),
        });
    }
    for l in &labels {
        let (k, v) = split_pair(l)?;
        cfg.run.labels.insert(k, v);
    }
    if cfg.targets.is_empty() {
        return Err("no targets: pass --config or --target NAME=URL".into());
    }
    let storage = storage.or(cfg.storage.clone());
    let save_interval = save_interval.or(cfg.save_interval_secs).unwrap_or(2);
    let root = root
        .or(cfg.run.root.clone().map(PathBuf::from))
        .unwrap_or_else(|| "./runs".into());
    let base = run
        .or(cfg.run.name.clone())
        .unwrap_or_else(|| chrono::Local::now().format("%Y%m%d-%H%M%S").to_string());
    // create_dir is atomic: two scrapers asking for the same name get distinct runs.
    std::fs::create_dir_all(&root)?;
    let mut name = base.clone();
    let mut n = 2;
    let run_dir = loop {
        let dir = root.join(&name);
        match std::fs::create_dir(&dir) {
            Ok(()) => break dir,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                name = format!("{base}-{n}");
                n += 1;
            }
            Err(e) => return Err(e.into()),
        }
    };
    let mut lock = None;
    for _ in 0..20 {
        // `serve` probes the lock for a moment to show LIVE; retry past that.
        lock = lock_run(&run_dir)?;
        if lock.is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let _lock = lock.ok_or("run directory is locked by another process")?;
    let seg_dir = run_dir.join(SEGMENTS_DIR);
    std::fs::create_dir_all(&seg_dir)?;
    if let Some(p) = &config {
        let _ = std::fs::copy(p, run_dir.join("config.toml"));
    }

    let mut meta = RunMeta {
        name: name.clone(),
        labels: cfg.run.labels.clone(),
        status: "running".into(),
        started_at_ms: now_ms(),
        heartbeat_ms: now_ms(),
        finished_at_ms: None,
        save_interval_secs: save_interval,
        host: gethostname::gethostname().to_string_lossy().to_string(),
        targets: cfg
            .targets
            .iter()
            .map(|t| Target {
                name: t.name.clone(),
                url: t.url.clone(),
                interval_ms: t.interval_ms.unwrap_or(1000),
            })
            .collect(),
    };
    meta.write(&run_dir)?;
    info!("run {} -> {}", name, run_dir.display());

    let writer = Arc::new(DatasetWriter::streaming(
        seg_dir.clone(),
        cfg.rows_per_parquet.unwrap_or(1_000_000),
        save_interval,
        cfg.extra_column_names(),
    )?);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;

    let mut tasks = Vec::new();
    for t in &cfg.targets {
        let rules = t.rules.merged(&cfg.defaults).compile()?;
        let (writer, client, t) = (writer.clone(), client.clone(), t.clone());
        tasks.push(tokio::spawn(async move {
            let mut tick =
                tokio::time::interval(Duration::from_millis(t.interval_ms.unwrap_or(1000).max(50)));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tick.tick().await;
                match scrape_endpoint(&client, &t.url, &t.name, &rules).await {
                    Ok(rows) if !rows.is_empty() => {
                        if let Err(e) = writer.append_rows(rows).await {
                            error!("{}: append failed: {e}", t.name);
                        }
                    }
                    Ok(_) => {}
                    Err(e) => warn!("{}: scrape failed: {e}", t.name),
                }
            }
        }));
    }
    {
        let (run_dir, mut meta) = (run_dir.clone(), meta.clone());
        tasks.push(tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(save_interval.max(1))).await;
                meta.heartbeat_ms = now_ms();
                let _ = meta.write(&run_dir);
            }
        }));
    }
    let sync_secs = cfg.sync_interval_secs.unwrap_or(300);
    if sync_secs > 0 {
        let (store, path) = remote_store(storage.as_deref(), &seg_dir, &name)?;
        let seg_dir = seg_dir.clone();
        tasks.push(tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(sync_secs)).await;
                if let Err(e) = periodic_compact_and_sync(&seg_dir, store.clone(), &path).await {
                    error!("periodic sync failed: {e}");
                }
            }
        }));
    }

    wait_for_shutdown().await;
    for t in &tasks {
        t.abort();
    }
    for t in tasks {
        let _ = t.await;
    }
    writer.shutdown().await?;
    meta.status = "finished".into();
    if let Err(e) = finish(&run_dir, storage.as_deref()).await {
        error!("compaction failed, segments kept: {e}");
        meta.status = "failed".into();
    }
    meta.heartbeat_ms = now_ms();
    meta.finished_at_ms = Some(now_ms());
    meta.write(&run_dir)?;
    info!("run {} {}", name, meta.status);
    Ok(())
}

async fn wait_for_shutdown() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    let _ = tokio::signal::ctrl_c().await;
    info!("shutting down");
}

#[tokio::main]
async fn main() -> AnyResult<()> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    match Cli::parse().cmd {
        Cmd::Scrape {
            config,
            root,
            run,
            targets,
            labels,
            interval_ms,
            save_interval,
            storage,
        } => {
            scrape(
                config,
                root,
                run,
                targets,
                labels,
                interval_ms,
                save_interval,
                storage,
            )
            .await
        }
        Cmd::Compact { run_dir } => {
            let _lock = lock_run(&run_dir)?
                .ok_or("a scraper is still writing this run (its .lock is held); stop it first")?;
            finish(&run_dir, None).await?;
            if let Some(mut meta) = RunMeta::read(&run_dir) {
                if meta.status == "running" {
                    meta.status = "finished".into();
                    meta.finished_at_ms = Some(now_ms());
                    meta.write(&run_dir)?;
                }
            }
            Ok(())
        }
        Cmd::Serve {
            root,
            addr,
            niu_lai,
        } => Ok(niu_lai_viz::server::serve(root, &addr, niu_lai).await?),
    }
}
