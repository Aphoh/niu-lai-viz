//! Generic, config-driven metric relabeling. Nothing here knows about any
//! particular exporter: DCGM, node-exporter, etc. are just example configs.
use crate::parse::{MetricType, ParsedSample};
use regex::Regex;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Deserialize, Debug, Clone, Default)]
pub struct RenameRule {
    pub pattern: String,
    pub replace: String,
}

/// Rules as written in TOML. Shared `[defaults]` are merged into every target.
#[derive(Deserialize, Debug, Clone, Default)]
pub struct RuleConfig {
    /// Keep only metrics whose name matches one of these regexes (empty = keep all).
    #[serde(default)]
    pub include: Vec<String>,
    /// Drop metrics whose name matches any of these regexes.
    #[serde(default)]
    pub exclude: Vec<String>,
    /// Prefixes stripped from metric names, first match wins.
    #[serde(default)]
    pub strip_prefix: Vec<String>,
    /// Regex rewrites of the metric name, applied in order (`$1` capture syntax).
    #[serde(default)]
    pub rename: Vec<RenameRule>,
    /// Prometheus labels removed entirely.
    #[serde(default)]
    pub drop_labels: Vec<String>,
    /// Prometheus labels moved out of the metric name into their own column.
    #[serde(default)]
    pub label_columns: Vec<String>,
    /// `lookup.<label>.<value> = { column = "value" }`: extra columns keyed by a label's value.
    #[serde(default)]
    pub lookup: BTreeMap<String, BTreeMap<String, BTreeMap<String, String>>>,
    /// Static columns attached to every row.
    #[serde(default)]
    pub columns: BTreeMap<String, String>,
}

impl RuleConfig {
    pub fn merged(&self, defaults: &RuleConfig) -> RuleConfig {
        let cat = |a: &Vec<String>, b: &Vec<String>| a.iter().chain(b).cloned().collect::<Vec<_>>();
        let mut lookup = defaults.lookup.clone();
        lookup.extend(self.lookup.clone());
        let mut columns = defaults.columns.clone();
        columns.extend(self.columns.clone());
        RuleConfig {
            include: cat(&defaults.include, &self.include),
            exclude: cat(&defaults.exclude, &self.exclude),
            strip_prefix: cat(&defaults.strip_prefix, &self.strip_prefix),
            rename: defaults
                .rename
                .iter()
                .chain(&self.rename)
                .cloned()
                .collect(),
            drop_labels: cat(&defaults.drop_labels, &self.drop_labels),
            label_columns: cat(&defaults.label_columns, &self.label_columns),
            lookup,
            columns,
        }
    }

    /// Every extra column these rules can emit.
    pub fn column_names(&self) -> BTreeSet<String> {
        let mut names: BTreeSet<String> = self.columns.keys().cloned().collect();
        names.extend(self.label_columns.iter().cloned());
        for table in self.lookup.values() {
            for cols in table.values() {
                names.extend(cols.keys().cloned());
            }
        }
        names
    }

    pub fn compile(&self) -> anyhow::Result<Rules> {
        let re = |p: &String| Regex::new(p).map_err(|e| anyhow::anyhow!("bad regex {p:?}: {e}"));
        Ok(Rules {
            include: self.include.iter().map(re).collect::<Result<_, _>>()?,
            exclude: self.exclude.iter().map(re).collect::<Result<_, _>>()?,
            strip_prefix: self.strip_prefix.clone(),
            rename: self
                .rename
                .iter()
                .map(|r| Ok((re(&r.pattern)?, r.replace.clone())))
                .collect::<anyhow::Result<_>>()?,
            drop_labels: self.drop_labels.iter().cloned().collect(),
            label_columns: self.label_columns.clone(),
            lookup: self.lookup.clone(),
            columns: self.columns.clone(),
        })
    }
}

/// Compiled rules applied to every scraped sample of one target.
#[derive(Debug, Clone, Default)]
pub struct Rules {
    include: Vec<Regex>,
    exclude: Vec<Regex>,
    strip_prefix: Vec<String>,
    rename: Vec<(Regex, String)>,
    drop_labels: BTreeSet<String>,
    label_columns: Vec<String>,
    lookup: BTreeMap<String, BTreeMap<String, BTreeMap<String, String>>>,
    columns: BTreeMap<String, String>,
}

/// Suffix the exposition format puts on a sample of this type.
pub fn type_suffix(t: &MetricType) -> &'static str {
    match t {
        MetricType::HistogramBucket => "_bucket",
        MetricType::HistogramCount | MetricType::SummaryCount => "_count",
        MetricType::HistogramSum | MetricType::SummarySum => "_sum",
        _ => "",
    }
}

impl Rules {
    /// The metric name and extra columns for a sample, or None to drop it.
    /// Histogram `le` and summary `quantile` stay in the name so buckets and
    /// quantiles remain distinct series; `le` is also stored as bucket bounds.
    pub fn apply(&self, sample: &ParsedSample) -> Option<(String, Vec<(String, String)>)> {
        let suffix = type_suffix(&sample.metric_type);
        let exposed = format!("{}{}", sample.metric_name, suffix);
        let hit = |r: &Regex| r.is_match(&sample.metric_name) || r.is_match(&exposed);
        if !self.include.is_empty() && !self.include.iter().any(hit) {
            return None;
        }
        if self.exclude.iter().any(hit) {
            return None;
        }

        let mut base = sample.metric_name.as_str();
        for p in &self.strip_prefix {
            if let Some(rest) = base.strip_prefix(p.as_str()) {
                base = rest;
                break;
            }
        }
        let mut base = base.to_string();
        for (re, rep) in &self.rename {
            base = re.replace_all(&base, rep.as_str()).into_owned();
        }

        let mut extras: BTreeMap<String, String> = self.columns.clone();
        for (label, table) in &self.lookup {
            if let Some(cols) = sample.labels.get(label).and_then(|v| table.get(v)) {
                extras.extend(cols.clone());
            }
        }
        let mut labels: HashMap<&str, &str> = sample
            .labels
            .iter()
            .filter(|(k, _)| !self.drop_labels.contains(k.as_str()))
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        for col in &self.label_columns {
            if let Some(v) = labels.remove(col.as_str()) {
                extras.insert(col.clone(), v.to_string());
            }
        }

        let mut pairs: Vec<(&str, &str)> = labels.into_iter().collect();
        pairs.sort();
        let name = if pairs.is_empty() {
            format!("{base}{suffix}")
        } else {
            let body: Vec<String> = pairs.iter().map(|(k, v)| format!("{k}=\"{v}\"")).collect();
            format!("{base}{suffix}{{{}}}", body.join(","))
        };
        Some((name, extras.into_iter().collect()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_prometheus_samples;

    fn rules(toml_src: &str) -> Rules {
        toml::from_str::<RuleConfig>(toml_src)
            .unwrap()
            .compile()
            .unwrap()
    }

    #[test]
    fn default_rules_keep_name_and_labels() {
        let s = parse_prometheus_samples("# TYPE g gauge\ng{b=\"2\",a=\"1\"} 3").unwrap();
        let (name, extras) = Rules::default().apply(&s[0]).unwrap();
        assert_eq!(name, "g{a=\"1\",b=\"2\"}");
        assert!(extras.is_empty());
    }

    #[test]
    fn include_exclude_strip_rename() {
        let r = rules(
            r#"
include = ["^DCGM_"]
exclude = ["CLOCK"]
strip_prefix = ["DCGM_FI_DEV_"]
rename = [{ pattern = "^GPU_UTIL$", replace = "gpu_util" }]
"#,
        );
        let s = parse_prometheus_samples("DCGM_FI_DEV_GPU_UTIL 5\nDCGM_FI_DEV_SM_CLOCK 1\nother 1")
            .unwrap();
        let out: Vec<_> = s
            .iter()
            .filter_map(|s| r.apply(s))
            .map(|(n, _)| n)
            .collect();
        assert_eq!(out, vec!["gpu_util"]);
    }

    #[test]
    fn label_columns_lookup_and_static_columns() {
        let r = rules(
            r#"
drop_labels = ["UUID"]
label_columns = ["gpu"]
columns = { host = "node1" }
[lookup.gpu."0"]
role = "prefill"
"#,
        );
        let s = parse_prometheus_samples("util{gpu=\"0\",UUID=\"x\",model=\"H\"} 7").unwrap();
        let (name, extras) = r.apply(&s[0]).unwrap();
        assert_eq!(name, "util{model=\"H\"}");
        assert_eq!(
            extras,
            vec![
                ("gpu".into(), "0".into()),
                ("host".into(), "node1".into()),
                ("role".into(), "prefill".into())
            ]
        );
    }

    #[test]
    fn defaults_merge_into_target() {
        let d: RuleConfig =
            toml::from_str("exclude = [\"^go_\"]\ncolumns = { a = \"1\" }").unwrap();
        let t: RuleConfig =
            toml::from_str("exclude = [\"^process_\"]\ncolumns = { b = \"2\" }").unwrap();
        let m = t.merged(&d);
        assert_eq!(m.exclude, vec!["^go_", "^process_"]);
        assert_eq!(
            m.column_names().into_iter().collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }
}
