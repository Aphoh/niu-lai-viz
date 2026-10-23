use crate::rules::RuleConfig;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize, Debug, Clone, Default)]
pub struct Config {
    #[serde(default)]
    pub run: RunConfig,
    /// Optional remote copy of final.parquet (`s3://bucket/prefix`); the run directory always keeps one.
    #[serde(default)]
    pub storage: Option<String>,
    #[serde(default)]
    pub rows_per_parquet: Option<usize>,
    /// Seconds between sealed segments; this is how "live" the UI is.
    #[serde(default)]
    pub save_interval_secs: Option<u64>,
    /// Seconds between merging segments into incomplete-N.parquet; 0 disables.
    #[serde(default)]
    pub sync_interval_secs: Option<u64>,
    #[serde(default)]
    pub defaults: RuleConfig,
    #[serde(default)]
    pub targets: Vec<TargetConfig>,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct RunConfig {
    pub name: Option<String>,
    /// Directory holding one subdirectory per run.
    pub root: Option<String>,
    /// Experiment parameters (lr, model, ...): shown and filterable in the UI.
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct TargetConfig {
    pub name: String,
    pub url: String,
    #[serde(default)]
    pub interval_ms: Option<u64>,
    #[serde(flatten)]
    pub rules: RuleConfig,
}

impl Config {
    pub fn from_toml(src: &str) -> anyhow::Result<Self> {
        Ok(toml::from_str(src)?)
    }

    /// Sorted union of the extra columns every target can emit.
    pub fn extra_column_names(&self) -> Vec<String> {
        let mut names = BTreeSet::new();
        for t in &self.targets {
            names.extend(t.rules.merged(&self.defaults).column_names());
        }
        names.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_config() {
        let c = Config::from_toml(
            r#"
save_interval_secs = 2
[run]
name = "exp1"
labels = { lr = "1e-4" }
[defaults]
exclude = ["^go_"]
[[targets]]
name = "gpu"
url = "http://localhost:9400/metrics"
interval_ms = 500
label_columns = ["gpu"]
[targets.lookup.gpu."0"]
role = "prefill"
"#,
        )
        .unwrap();
        assert_eq!(c.run.labels["lr"], "1e-4");
        assert_eq!(c.targets[0].interval_ms, Some(500));
        assert_eq!(c.extra_column_names(), vec!["gpu", "role"]);
    }
}
