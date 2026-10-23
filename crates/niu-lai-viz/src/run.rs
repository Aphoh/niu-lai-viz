//! On-disk layout of a run: `<root>/<name>/run.json`, `segments/out-N.parquet`
//! while live, `final.parquet` once compacted.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const META_FILE: &str = "run.json";
pub const SEGMENTS_DIR: &str = "segments";
pub const FINAL_FILE: &str = "final.parquet";
/// flock'd by the scraper writing the run; the OS drops it if that process dies.
pub const LOCK_FILE: &str = ".lock";

/// Exclusive flock on `<run>/.lock`, held while the returned file lives.
/// `Ok(None)`: another process holds it.
pub fn lock_run(run_dir: &Path) -> std::io::Result<Option<std::fs::File>> {
    let f = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(run_dir.join(LOCK_FILE))?;
    match f.try_lock() {
        Ok(()) => Ok(Some(f)),
        Err(std::fs::TryLockError::WouldBlock) => Ok(None),
        Err(std::fs::TryLockError::Error(e)) => Err(e),
    }
}

/// Whether a process holds the run's lock; `None` if unknown (no lock file,
/// or a filesystem without flock).
pub fn is_locked(run_dir: &Path) -> Option<bool> {
    let f = std::fs::File::open(run_dir.join(LOCK_FILE)).ok()?;
    match f.try_lock_shared() {
        Ok(()) => Some(false),
        Err(std::fs::TryLockError::WouldBlock) => Some(true),
        Err(_) => None,
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Target {
    pub name: String,
    pub url: String,
    pub interval_ms: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct RunMeta {
    pub name: String,
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
    /// "running", "finished" or "failed".
    pub status: String,
    pub started_at_ms: i64,
    #[serde(default)]
    pub heartbeat_ms: i64,
    #[serde(default)]
    pub finished_at_ms: Option<i64>,
    #[serde(default)]
    pub save_interval_secs: u64,
    #[serde(default)]
    pub host: String,
    #[serde(default)]
    pub targets: Vec<Target>,
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

impl RunMeta {
    pub fn read(run_dir: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(run_dir.join(META_FILE)).ok()?;
        serde_json::from_str(&text).ok()
    }

    /// Atomic replace so readers never see a torn file.
    pub fn write(&self, run_dir: &Path) -> std::io::Result<()> {
        let tmp = run_dir.join(format!(".{META_FILE}.tmp"));
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(tmp, run_dir.join(META_FILE))
    }

    /// Running with a recent heartbeat; a crashed scraper goes stale.
    pub fn is_live(&self, now: i64) -> bool {
        let grace = (self.save_interval_secs.max(1) as i64 * 3 + 15) * 1000;
        self.status == "running" && now - self.heartbeat_ms.max(self.started_at_ms) < grace
    }
}

/// Data files for a run, in read order: final.parquet supersedes segments.
pub fn data_files(run_dir: &Path) -> Vec<PathBuf> {
    let final_path = run_dir.join(FINAL_FILE);
    if final_path.is_file() {
        return vec![final_path];
    }
    let seg = run_dir.join(SEGMENTS_DIR);
    let index = |name: &str, prefix: &str| -> Option<usize> {
        name.strip_prefix(prefix)?
            .strip_suffix(".parquet")?
            .parse()
            .ok()
    };
    let mut incomplete = Vec::new();
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&seg) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if let Some(i) = index(&name, "incomplete-") {
                incomplete.push((i, e.path()));
            } else if let Some(i) = index(&name, "out-") {
                out.push((i, e.path()));
            }
        }
    }
    incomplete.sort();
    out.sort();
    incomplete.into_iter().chain(out).map(|(_, p)| p).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_lock_is_exclusive_and_released_on_drop() {
        let dir = std::env::temp_dir().join(format!("niu-lock-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let held = lock_run(&dir).unwrap().expect("first lock");
        assert!(
            lock_run(&dir).unwrap().is_none(),
            "second writer is refused"
        );
        assert_eq!(is_locked(&dir), Some(true));
        drop(held);
        assert_eq!(is_locked(&dir), Some(false));
        assert!(lock_run(&dir).unwrap().is_some());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
