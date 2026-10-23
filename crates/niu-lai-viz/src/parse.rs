// Derived from tachometer-scraper via NVIDIA/srt-slurm (Apache-2.0); modified for niu-lai-viz. See NOTICE.
use niu_lai_writer::Row;
use std::collections::HashMap;

/// Represents a parsed Prometheus metric sample
#[derive(Debug, Clone)]
pub struct ParsedSample {
    pub metric_name: String, // Base metric name (without suffix)
    pub labels: HashMap<String, String>,
    pub value: f64,
    pub metric_type: MetricType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    HistogramBucket,
    HistogramCount,
    HistogramSum,
    Summary,
    SummaryQuantile,
    SummaryCount,
    SummarySum,
    Untyped,
}

/// Parse Prometheus text format and return ParsedSample objects
/// This parser preserves _count and _sum fields for histograms and summaries
pub fn parse_prometheus_samples(text: &str) -> Result<Vec<ParsedSample>, anyhow::Error> {
    let mut samples = Vec::new();
    let mut metric_types: HashMap<String, MetricType> = HashMap::new();

    // First pass: collect metric types from TYPE comments
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("# TYPE ") {
            if let Some(metric_type) = parse_type_line(line) {
                metric_types.insert(metric_type.0, metric_type.1);
            }
        }
    }

    // Second pass: parse metrics
    for line in text.lines() {
        let line = line.trim();

        // Skip empty lines and comments (except TYPE which we already handled)
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse metric line
        match parse_metric_line(line, &metric_types) {
            Ok(sample) => {
                samples.push(sample);
            }
            Err(_e) => {
                // Silently skip lines that fail to parse
                continue;
            }
        }
    }

    Ok(samples)
}

/// Convert samples to rows, applying a target's rules. Histogram `_sum` and
/// `_count` are folded into every bucket row of their histogram, and each
/// bucket gets its lower bound from the next-smaller `le` of the same series.
pub fn samples_to_rows_with_rules(
    samples: Vec<ParsedSample>,
    scraper_endpoint: &str,
    rules: &crate::rules::Rules,
) -> Vec<Row> {
    let mut stats: HashMap<String, (Option<f64>, Option<f64>)> = HashMap::new();
    let mut uppers: HashMap<String, Vec<f64>> = HashMap::new();
    for sample in &samples {
        let key = make_histogram_key(&sample.metric_name, &sample.labels);
        match sample.metric_type {
            MetricType::HistogramSum => stats.entry(key).or_default().0 = Some(sample.value),
            MetricType::HistogramCount => stats.entry(key).or_default().1 = Some(sample.value),
            MetricType::HistogramBucket => {
                uppers.entry(key).or_default().push(bucket_upper(sample))
            }
            _ => {}
        }
    }
    for bounds in uppers.values_mut() {
        bounds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    let mut rows = Vec::with_capacity(samples.len());
    for sample in samples {
        if matches!(
            sample.metric_type,
            MetricType::HistogramSum | MetricType::HistogramCount
        ) {
            continue;
        }
        let Some((metric_name, extras)) = rules.apply(&sample) else {
            continue;
        };
        let mut row = Row {
            scraper_endpoint: scraper_endpoint.to_string(),
            metric_name,
            metric_value: sample.value,
            histogram_bucket_lower: None,
            histogram_bucket_upper: None,
            histogram_sum: None,
            histogram_count: None,
            extras,
        };
        if sample.metric_type == MetricType::HistogramBucket {
            let key = make_histogram_key(&sample.metric_name, &sample.labels);
            let upper = bucket_upper(&sample);
            let lower = uppers
                .get(&key)
                .and_then(|b| b.iter().rev().find(|&&u| u < upper).copied())
                .unwrap_or(0.0);
            row.histogram_bucket_lower = Some(lower);
            row.histogram_bucket_upper = upper.is_finite().then_some(upper);
            if let Some((sum, count)) = stats.get(&key) {
                row.histogram_sum = *sum;
                row.histogram_count = *count;
            }
        }
        rows.push(row);
    }
    rows
}

/// Convert samples to rows with no relabeling.
pub fn samples_to_rows(samples: Vec<ParsedSample>, scraper_endpoint: &str) -> Vec<Row> {
    samples_to_rows_with_rules(samples, scraper_endpoint, &crate::rules::Rules::default())
}

fn bucket_upper(sample: &ParsedSample) -> f64 {
    match sample.labels.get("le").map(String::as_str) {
        None | Some("+Inf") => f64::INFINITY,
        Some(le) => le.parse().unwrap_or(f64::INFINITY),
    }
}

/// Convenience function to parse and convert in one step
pub fn parse_prometheus_text(
    text: &str,
    scraper_endpoint: &str,
) -> Result<Vec<Row>, anyhow::Error> {
    let samples = parse_prometheus_samples(text)?;
    Ok(samples_to_rows(samples, scraper_endpoint))
}

fn parse_type_line(line: &str) -> Option<(String, MetricType)> {
    // Format: # TYPE metric_name type
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() >= 4 {
        let metric_name = parts[2].to_string();
        let metric_type = match parts[3] {
            "counter" => MetricType::Counter,
            "gauge" => MetricType::Gauge,
            "histogram" => MetricType::Histogram,
            "summary" => MetricType::Summary,
            "untyped" => MetricType::Untyped,
            _ => return None,
        };
        Some((metric_name, metric_type))
    } else {
        None
    }
}

fn parse_metric_line(
    line: &str,
    metric_types: &HashMap<String, MetricType>,
) -> Result<ParsedSample, anyhow::Error> {
    // Format: metric_name{labels} value or metric_name value
    // Need to handle labels that may contain spaces, so find the split point correctly
    let metric_and_labels: &str;
    let value_str: &str;

    if line.contains('{') {
        // Metric has labels: find the closing brace and split after it
        let brace_end = line
            .rfind('}')
            .ok_or_else(|| anyhow::anyhow!("Unclosed brace in labels"))?;
        metric_and_labels = &line[..=brace_end];
        // Value starts after whitespace following the brace
        value_str = line[brace_end + 1..]
            .split_whitespace()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Missing value after labels"))?;
    } else {
        // No labels: split on first whitespace
        let mut parts = line.split_whitespace();
        metric_and_labels = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("Empty metric line"))?;
        value_str = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("Missing value"))?;
    }

    let value = value_str
        .parse::<f64>()
        .map_err(|e| anyhow::anyhow!("Failed to parse value '{}': {}", value_str, e))?;

    let (metric_name, labels, metric_type) = if metric_and_labels.contains('{') {
        parse_metric_with_labels(metric_and_labels, metric_types)?
    } else {
        let metric_type = determine_metric_type(metric_and_labels, metric_types)?;
        (strip_suffix(metric_and_labels), HashMap::new(), metric_type)
    };

    Ok(ParsedSample {
        metric_name,
        labels,
        value,
        metric_type,
    })
}

fn parse_metric_with_labels(
    metric_and_labels: &str,
    metric_types: &HashMap<String, MetricType>,
) -> Result<(String, HashMap<String, String>, MetricType), anyhow::Error> {
    // Format: metric_name{label1="value1",label2="value2"}
    let brace_start = metric_and_labels
        .find('{')
        .ok_or_else(|| anyhow::anyhow!("Malformed label section"))?;
    let brace_end = metric_and_labels
        .rfind('}')
        .ok_or_else(|| anyhow::anyhow!("Malformed label section"))?;

    let metric_name_with_suffix = &metric_and_labels[..brace_start];
    let labels_str = &metric_and_labels[brace_start + 1..brace_end];

    let mut labels = HashMap::new();
    if !labels_str.is_empty() {
        for label_pair in labels_str.split(',') {
            let parts: Vec<&str> = label_pair.splitn(2, '=').collect();
            if parts.len() == 2 {
                let key = parts[0].trim();
                let value = parts[1].trim().trim_matches('"');
                labels.insert(key.to_string(), value.to_string());
            }
        }
    }

    // Determine metric type - check if it's a summary quantile
    let metric_type = if labels.contains_key("quantile") {
        let base = strip_suffix(metric_name_with_suffix);
        if metric_types.get(&base) == Some(&MetricType::Summary) {
            MetricType::SummaryQuantile
        } else {
            determine_metric_type(metric_name_with_suffix, metric_types)?
        }
    } else {
        determine_metric_type(metric_name_with_suffix, metric_types)?
    };

    let base_metric_name = strip_suffix(metric_name_with_suffix);

    Ok((base_metric_name, labels, metric_type))
}

fn determine_metric_type(
    metric_name: &str,
    metric_types: &HashMap<String, MetricType>,
) -> Result<MetricType, anyhow::Error> {
    // Check for suffixes first
    if metric_name.ends_with("_bucket") {
        // Check if base metric is a histogram
        let base = metric_name.strip_suffix("_bucket").unwrap();
        if metric_types.get(base) == Some(&MetricType::Histogram) {
            return Ok(MetricType::HistogramBucket);
        }
    } else if metric_name.ends_with("_count") {
        let base = metric_name.strip_suffix("_count").unwrap();
        if metric_types.get(base) == Some(&MetricType::Histogram) {
            return Ok(MetricType::HistogramCount);
        } else if metric_types.get(base) == Some(&MetricType::Summary) {
            return Ok(MetricType::SummaryCount);
        }
    } else if metric_name.ends_with("_sum") {
        let base = metric_name.strip_suffix("_sum").unwrap();
        if metric_types.get(base) == Some(&MetricType::Histogram) {
            return Ok(MetricType::HistogramSum);
        } else if metric_types.get(base) == Some(&MetricType::Summary) {
            return Ok(MetricType::SummarySum);
        }
    }

    // Check base metric type
    if let Some(metric_type) = metric_types.get(metric_name) {
        match metric_type {
            MetricType::Counter => Ok(MetricType::Counter),
            MetricType::Gauge => Ok(MetricType::Gauge),
            MetricType::Histogram => Ok(MetricType::Histogram),
            MetricType::Summary => Ok(MetricType::Summary),
            MetricType::Untyped => Ok(MetricType::Untyped),
            _ => Ok(MetricType::Untyped), // Fallback
        }
    } else {
        Ok(MetricType::Untyped)
    }
}

fn strip_suffix(metric_name: &str) -> String {
    // Remove _bucket, _count, _sum suffixes to get base name
    if metric_name.ends_with("_bucket") {
        metric_name.strip_suffix("_bucket").unwrap().to_string()
    } else if metric_name.ends_with("_count") {
        metric_name.strip_suffix("_count").unwrap().to_string()
    } else if metric_name.ends_with("_sum") {
        metric_name.strip_suffix("_sum").unwrap().to_string()
    } else {
        metric_name.to_string()
    }
}

/// Create a unique key for a histogram based on metric name and labels (excluding le)
fn make_histogram_key(metric_name: &str, labels: &HashMap<String, String>) -> String {
    let mut label_pairs: Vec<(String, String)> = labels
        .iter()
        .filter(|(k, _)| k.as_str() != "le" && k.as_str() != "quantile")
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    label_pairs.sort();

    if label_pairs.is_empty() {
        metric_name.to_string()
    } else {
        format!("{}:{:?}", metric_name, label_pairs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_gauge() {
        let text = r#"# TYPE test_gauge gauge
test_gauge 42.5"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].metric_name, "test_gauge");
        assert_eq!(samples[0].value, 42.5);
        assert_eq!(samples[0].metric_type, MetricType::Gauge);
        assert!(samples[0].labels.is_empty());
    }

    #[test]
    fn test_parse_gauge_with_labels() {
        let text = r#"# TYPE test_metric gauge
test_metric{label1="value1",label2="value2"} 123.45"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].metric_name, "test_metric");
        assert_eq!(samples[0].value, 123.45);
        assert_eq!(samples[0].labels.get("label1"), Some(&"value1".to_string()));
        assert_eq!(samples[0].labels.get("label2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_parse_labels_with_spaces() {
        let text = r#"# TYPE test_metric gauge
test_metric{name="NVIDIA GB200",location="rack 1"} 100"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(
            samples[0].labels.get("name"),
            Some(&"NVIDIA GB200".to_string())
        );
        assert_eq!(
            samples[0].labels.get("location"),
            Some(&"rack 1".to_string())
        );
    }

    #[test]
    fn test_parse_histogram_with_count_and_sum() {
        let text = r#"# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.005"} 24054
http_request_duration_seconds_bucket{le="0.01"} 33444
http_request_duration_seconds_bucket{le="+Inf"} 179887
http_request_duration_seconds_count 179887
http_request_duration_seconds_sum 53423"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 5, "Should have 3 buckets + count + sum");

        // Check buckets
        let buckets: Vec<_> = samples
            .iter()
            .filter(|s| s.metric_type == MetricType::HistogramBucket)
            .collect();
        assert_eq!(buckets.len(), 3);

        // Check _count
        let count = samples
            .iter()
            .find(|s| s.metric_type == MetricType::HistogramCount);
        assert!(count.is_some(), "Should have _count metric");
        assert_eq!(count.unwrap().value, 179887.0);
        assert_eq!(count.unwrap().metric_name, "http_request_duration_seconds");

        // Check _sum
        let sum = samples
            .iter()
            .find(|s| s.metric_type == MetricType::HistogramSum);
        assert!(sum.is_some(), "Should have _sum metric");
        assert_eq!(sum.unwrap().value, 53423.0);
        assert_eq!(sum.unwrap().metric_name, "http_request_duration_seconds");
    }

    #[test]
    fn test_parse_histogram_with_labels() {
        let text = r#"# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{method="GET",le="0.1"} 100
http_request_duration_seconds_bucket{method="GET",le="+Inf"} 200
http_request_duration_seconds_count{method="GET"} 200
http_request_duration_seconds_sum{method="GET"} 10.5"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 4);

        // Verify count has labels
        let count = samples
            .iter()
            .find(|s| s.metric_type == MetricType::HistogramCount);
        assert!(count.is_some());
        assert_eq!(
            count.unwrap().labels.get("method"),
            Some(&"GET".to_string())
        );

        // Verify sum has labels
        let sum = samples
            .iter()
            .find(|s| s.metric_type == MetricType::HistogramSum);
        assert!(sum.is_some());
        assert_eq!(sum.unwrap().labels.get("method"), Some(&"GET".to_string()));
    }

    #[test]
    fn test_parse_summary_with_count_and_sum() {
        let text = r#"# TYPE go_gc_duration_seconds summary
go_gc_duration_seconds{quantile="0"} 0
go_gc_duration_seconds{quantile="0.25"} 0.5
go_gc_duration_seconds{quantile="0.5"} 1.0
go_gc_duration_seconds{quantile="0.75"} 1.5
go_gc_duration_seconds{quantile="1"} 2.0
go_gc_duration_seconds_sum 12.5
go_gc_duration_seconds_count 100"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 7, "Should have 5 quantiles + count + sum");

        // Check quantiles
        let quantiles: Vec<_> = samples
            .iter()
            .filter(|s| s.metric_type == MetricType::SummaryQuantile)
            .collect();
        assert_eq!(quantiles.len(), 5);

        // Check _count
        let count = samples
            .iter()
            .find(|s| s.metric_type == MetricType::SummaryCount);
        assert!(count.is_some(), "Should have summary _count");
        assert_eq!(count.unwrap().value, 100.0);

        // Check _sum
        let sum = samples
            .iter()
            .find(|s| s.metric_type == MetricType::SummarySum);
        assert!(sum.is_some(), "Should have summary _sum");
        assert_eq!(sum.unwrap().value, 12.5);
    }

    #[test]
    fn test_samples_to_rows_preserves_count_sum() {
        let text = r#"# TYPE test_histogram histogram
test_histogram_bucket{le="1.0"} 50
test_histogram_bucket{le="+Inf"} 100
test_histogram_count 100
test_histogram_sum 45.5"#;

        let samples = parse_prometheus_samples(text).unwrap();
        let rows = samples_to_rows(samples, "test_endpoint");

        // Should have only 2 rows (the buckets), not 4 (buckets + count + sum)
        assert_eq!(rows.len(), 2, "Should have 2 histogram bucket rows");

        // All buckets should have sum and count attached
        for row in &rows {
            assert_eq!(
                row.histogram_sum,
                Some(45.5),
                "All buckets should have histogram_sum"
            );
            assert_eq!(
                row.histogram_count,
                Some(100.0),
                "All buckets should have histogram_count"
            );
        }
    }

    #[test]
    fn test_counter_metric() {
        let text = r#"# TYPE node_cpu_seconds_total counter
node_cpu_seconds_total{cpu="0",mode="idle"} 12345.67"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].metric_type, MetricType::Counter);
        assert_eq!(samples[0].metric_name, "node_cpu_seconds_total");
        assert_eq!(samples[0].value, 12345.67);
    }

    #[test]
    fn test_histogram_bucket_bounds() {
        let text = r#"# TYPE test_hist histogram
test_hist_bucket{le="0.1"} 10
test_hist_bucket{le="1.0"} 50
test_hist_bucket{le="10.0"} 90
test_hist_bucket{le="+Inf"} 100
test_hist_count 100
test_hist_sum 123.4"#;

        let samples = parse_prometheus_samples(text).unwrap();
        let rows = samples_to_rows(samples, "test");

        // Check bucket bounds are set correctly (4 buckets only, no separate count/sum rows)
        let buckets: Vec<_> = rows
            .iter()
            .filter(|r| {
                r.histogram_bucket_upper.is_some()
                    || r.histogram_bucket_upper.is_none() && r.metric_name.contains("_bucket")
            })
            .collect();
        assert_eq!(
            buckets.len(),
            4,
            "Should have 4 histogram buckets, got {}",
            buckets.len()
        );

        // First bucket: 0 to 0.1
        assert_eq!(buckets[0].histogram_bucket_lower, Some(0.0));
        assert_eq!(buckets[0].histogram_bucket_upper, Some(0.1));
        // All buckets should have sum and count
        assert_eq!(buckets[0].histogram_sum, Some(123.4));
        assert_eq!(buckets[0].histogram_count, Some(100.0));

        // Second bucket: 0.1 to 1.0
        assert_eq!(buckets[1].histogram_bucket_lower, Some(0.1));
        assert_eq!(buckets[1].histogram_bucket_upper, Some(1.0));
        // All buckets should have sum and count
        assert_eq!(buckets[1].histogram_sum, Some(123.4));
        assert_eq!(buckets[1].histogram_count, Some(100.0));

        // Last bucket: 10.0 to +Inf
        assert_eq!(buckets[3].histogram_bucket_lower, Some(10.0));
        assert_eq!(buckets[3].histogram_bucket_upper, None); // +Inf
                                                             // All buckets should have sum and count
        assert_eq!(buckets[3].histogram_sum, Some(123.4));
        assert_eq!(buckets[3].histogram_count, Some(100.0));
    }

    #[test]
    fn test_mixed_metric_types() {
        let text = r#"# TYPE counter_metric counter
counter_metric 100
# TYPE gauge_metric gauge
gauge_metric{label="value"} 50.5
# TYPE hist_metric histogram
hist_metric_bucket{le="+Inf"} 10
hist_metric_count 10
hist_metric_sum 5.5"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 5);

        let counters: Vec<_> = samples
            .iter()
            .filter(|s| s.metric_type == MetricType::Counter)
            .collect();
        assert_eq!(counters.len(), 1);

        let gauges: Vec<_> = samples
            .iter()
            .filter(|s| s.metric_type == MetricType::Gauge)
            .collect();
        assert_eq!(gauges.len(), 1);

        let hist_count: Vec<_> = samples
            .iter()
            .filter(|s| s.metric_type == MetricType::HistogramCount)
            .collect();
        assert_eq!(hist_count.len(), 1);

        let hist_sum: Vec<_> = samples
            .iter()
            .filter(|s| s.metric_type == MetricType::HistogramSum)
            .collect();
        assert_eq!(hist_sum.len(), 1);
    }

    #[test]
    fn test_empty_input() {
        let text = "";
        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 0);
    }

    #[test]
    fn test_comments_only() {
        let text = r#"# HELP test_metric This is a test metric
# TYPE test_metric gauge"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 0, "Should have no samples, only metadata");
    }

    #[test]
    fn test_metric_without_type_declaration() {
        let text = r#"untyped_metric 123"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].metric_type, MetricType::Untyped);
        assert_eq!(samples[0].value, 123.0);
    }
}
