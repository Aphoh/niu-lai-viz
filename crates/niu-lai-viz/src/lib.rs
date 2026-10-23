// Derived from tachometer-scraper via NVIDIA/srt-slurm (Apache-2.0); modified for niu-lai-viz. See NOTICE.
pub mod config;
pub mod expr;
pub mod parse;
pub mod rules;
pub mod run;
pub mod server;

pub use config::Config;
pub use parse::ParsedSample;
pub use rules::{RuleConfig, Rules};

use niu_lai_writer::Row;

/// Fetch one Prometheus text exposition (`http(s)://` or `file://`).
pub async fn fetch(client: &reqwest::Client, url: &str) -> anyhow::Result<String> {
    if let Some(path) = url.strip_prefix("file://") {
        return std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("read {path}: {e}"));
    }
    let resp = client.get(url).send().await?.error_for_status()?;
    Ok(resp.text().await?)
}

/// Scrape a target and return rows after applying its rules.
pub async fn scrape_endpoint(
    client: &reqwest::Client,
    url: &str,
    target: &str,
    rules: &Rules,
) -> anyhow::Result<Vec<Row>> {
    let text = fetch(client, url).await?;
    let samples = parse::parse_prometheus_samples(&text)?;
    Ok(parse::samples_to_rows_with_rules(samples, target, rules))
}
