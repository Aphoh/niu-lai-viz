//! The shipped example configs parse and do what they claim on real exporter output.
use niu_lai_viz::{parse::parse_prometheus_samples, parse::samples_to_rows_with_rules, Config};
use std::path::Path;

fn rows_for(config: &str, fixture: &str) -> Vec<niu_lai_writer::Row> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let cfg = Config::from_toml(&std::fs::read_to_string(root.join(config)).unwrap()).unwrap();
    let t = &cfg.targets[0];
    let rules = t.rules.merged(&cfg.defaults).compile().unwrap();
    let text = std::fs::read_to_string(root.join("fixtures").join(fixture)).unwrap();
    samples_to_rows_with_rules(parse_prometheus_samples(&text).unwrap(), &t.name, &rules)
}

#[test]
fn dcgm_example() {
    let rows = rows_for("dcgm.toml", "dcgm.txt");
    assert!(!rows.is_empty());
    assert!(rows
        .iter()
        .all(|r| r.metric_name.starts_with("gpu_") && !r.metric_name.contains("UUID")));
    assert!(rows.iter().any(|r| r.metric_name == "gpu_GPU_UTIL"));
    let r = rows
        .iter()
        .find(|r| r.extras.contains(&("gpu".into(), "0".into())))
        .unwrap();
    assert!(r.extras.contains(&("role".into(), "prefill".into())));
}

#[test]
fn node_exporter_example() {
    let rows = rows_for("node-exporter.toml", "node-exporter.txt");
    assert!(!rows.is_empty());
    assert!(rows
        .iter()
        .all(|r| !r.metric_name.starts_with("node_") && !r.metric_name.starts_with("cpu")));
}

#[test]
fn demo_config_parses() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let cfg = Config::from_toml(&std::fs::read_to_string(root.join("demo.toml")).unwrap()).unwrap();
    assert!(cfg.targets.is_empty());
    assert_eq!(cfg.defaults.label_columns, vec!["gpu"]);
}
