#!/usr/bin/env python3
"""SQL over a niu-lai-viz runs folder (prototype for agent-written queries).

Every run's Parquet is exposed as one lazy table `samples` with columns:
  run, <run labels from run.json, e.g. lr, optimizer>, scraper_endpoint,
  metric_name (with {labels}), metric_name_clean, metric_value,
  time_since_start, timestamp_ns, histogram_*, plus any label columns.

  pip install polars
  python3 examples/query.py --root runs "SELECT run, lr, MIN(metric_value) best
      FROM samples WHERE metric_name_clean = 'train_loss' GROUP BY run, lr ORDER BY best"
  python3 examples/query.py --root runs --metrics   # metric catalogue
"""

import argparse
import glob
import json
import os

import polars as pl

CATALOGUE = """
SELECT metric_name_clean AS metric, COUNT(DISTINCT metric_name) AS series,
       COUNT(DISTINCT run) AS runs, COUNT(*) AS samples,
       MIN(metric_value) AS min, MAX(metric_value) AS max
FROM samples GROUP BY metric ORDER BY metric
"""


def samples(root: str) -> pl.LazyFrame:
    frames = []
    for meta_path in sorted(glob.glob(os.path.join(root, "*", "run.json"))):
        run_dir = os.path.dirname(meta_path)
        final = os.path.join(run_dir, "final.parquet")
        files = [final] if os.path.exists(final) else sorted(glob.glob(os.path.join(run_dir, "segments", "*.parquet")))
        if not files:
            continue
        meta = json.load(open(meta_path))
        lit = [pl.lit(os.path.basename(run_dir)).alias("run")]
        lit += [pl.lit(v).alias(k) for k, v in meta.get("labels", {}).items()]
        frames.append(pl.scan_parquet(files, missing_columns="insert", extra_columns="ignore").with_columns(lit))
    lf = pl.concat(frames, how="diagonal_relaxed")
    bare = pl.col("metric_name").str.replace(r"\{.*", "")
    return lf.with_columns(pl.coalesce(pl.col("metric_name_clean"), bare).alias("metric_name_clean"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="runs")
    ap.add_argument("--metrics", action="store_true", help="print the metric catalogue")
    ap.add_argument("sql", nargs="?")
    a = ap.parse_args()
    sql = CATALOGUE if a.metrics or not a.sql else a.sql
    pl.Config.set_tbl_rows(200).set_fmt_str_lengths(80).set_tbl_width_chars(200)
    print(pl.SQLContext(samples=samples(a.root)).execute(sql).collect())


if __name__ == "__main__":
    main()
