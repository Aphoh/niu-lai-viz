<p align="center"><img src="assets/niu-lai.svg" width="240" alt="niu-lai"></p>

# niu-lai-viz

grafana for experiments · weights & biases for the poor · 100% /metrics

Scrapes Prometheus `/metrics` endpoints into Parquet runs and serves a live UI to compare them.

## Demo

Needs Rust ([rustup](https://rustup.rs)) and Python 3.

```bash
git clone https://github.com/Aphoh/niu-lai-viz && cd niu-lai-viz
cargo build --release
python3 examples/demo.py --bin target/release/niu-lai-viz --root runs
```

Then open the printed link (http://127.0.0.1:8737). Ctrl+C stops it.
