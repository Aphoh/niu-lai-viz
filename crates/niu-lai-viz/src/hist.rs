//! Prometheus histograms: `_bucket{le=..}` series grouped per metric and cut
//! into per-bucket observation counts over time windows (bucket series are
//! cumulative both in `le` and in time, so both are differenced).
use crate::server::Series;

/// `ep/name_bucket{a="x",le="0.5"} [c=v]` -> (`ep/name{a="x"} [c=v]`, 0.5).
pub fn bucket_of(key: &str) -> Option<(String, f64)> {
    let i = key.find("_bucket{")?;
    let open = i + "_bucket".len();
    let close = open + key[open..].find('}')?;
    let (mut le, mut rest) = (None, Vec::new());
    for part in split_labels(&key[open + 1..close]) {
        match part.strip_prefix("le=") {
            Some(v) => le = Some(v.trim_matches('"')),
            None => rest.push(part),
        }
    }
    let le = match le? {
        "+Inf" | "Inf" => f64::INFINITY,
        v => v.parse().ok()?,
    };
    let mut base = key[..i].to_string();
    if !rest.is_empty() {
        base.push('{');
        base.push_str(&rest.join(","));
        base.push('}');
    }
    base.push_str(&key[close + 1..]);
    Some((base, le))
}

fn split_labels(body: &str) -> Vec<&str> {
    let (mut out, mut start, mut quoted) = (Vec::new(), 0, false);
    for (i, c) in body.char_indices() {
        match c {
            '"' => quoted = !quoted,
            ',' if !quoted => {
                out.push(&body[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < body.len() {
        out.push(&body[start..]);
    }
    out
}

/// Cumulative value at `t`: the latest sample at or before it, else the first.
fn at(s: &Series, t: f64) -> Option<f64> {
    if s.t.is_empty() {
        return None;
    }
    let i = s.t.partition_point(|&x| (x as f64) <= t);
    Some(s.v[i.saturating_sub(1)] as f64)
}

/// Observations per bucket in each window `[edges[k], edges[k+1]]` (aligned
/// time; raw time = aligned + `offset`), for buckets sorted by `le`. `None`
/// marks a window outside the data or one spanning a counter reset.
pub fn slices(buckets: &[(f64, &Series)], offset: f64, edges: &[f64]) -> Vec<Option<Vec<f32>>> {
    let cum =
        |t: f64| -> Option<Vec<f64>> { buckets.iter().map(|(_, s)| at(s, t + offset)).collect() };
    let (mut first, mut last) = (f64::INFINITY, f64::NEG_INFINITY);
    for (_, s) in buckets {
        if let (Some(&a), Some(&b)) = (s.t.first(), s.t.last()) {
            first = first.min(a as f64 - offset);
            last = last.max(b as f64 - offset);
        }
    }
    let mut prev = edges.first().and_then(|&e| cum(e));
    edges
        .windows(2)
        .map(|w| {
            let next = cum(w[1]);
            let out = (|| {
                if w[1] <= first || w[0] >= last {
                    return None;
                }
                // No scrape inside the window (e.g. scraper down): unknown, not zero.
                let (_, inf) = buckets.last()?;
                let i = inf.t.partition_point(|&x| (x as f64) <= w[0] + offset);
                if inf.t.get(i).is_none_or(|&x| x as f64 > w[1] + offset) {
                    return None;
                }
                let (a, b) = (prev.as_ref()?, next.as_ref()?);
                let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| y - x).collect();
                if *d.last()? < 0.0 {
                    return None;
                }
                let mut below = 0.0;
                Some(
                    d.into_iter()
                        .map(|v| {
                            let c = (v - below).max(0.0) as f32;
                            below = v;
                            c
                        })
                        .collect(),
                )
            })();
            prev = next;
            out
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_keys() {
        assert_eq!(
            bucket_of(r#"trainer/step_seconds_bucket{le="0.5"}"#),
            Some(("trainer/step_seconds".into(), 0.5))
        );
        assert_eq!(
            bucket_of(r#"ep/ttft_bucket{le="+Inf",model="a,b"} [gpu=0]"#),
            Some((r#"ep/ttft{model="a,b"} [gpu=0]"#.into(), f64::INFINITY))
        );
        assert_eq!(
            bucket_of(r#"ep/x_bucket{a="1",le="2",z="3"}"#),
            Some((r#"ep/x{a="1",z="3"}"#.into(), 2.0))
        );
        assert_eq!(bucket_of("ep/loss"), None);
        assert_eq!(bucket_of(r#"ep/x_bucket{a="1"}"#), None);
    }

    fn s(t: &[f32], v: &[f32]) -> Series {
        Series {
            t: t.to_vec(),
            v: v.to_vec(),
        }
    }

    #[test]
    fn windows_difference_time_and_le() {
        let t = [0.0, 10.0, 20.0, 30.0];
        let (b1, inf) = (s(&t, &[0., 5., 8., 8.]), s(&t, &[0., 6., 12., 13.]));
        let out = slices(
            &[(1.0, &b1), (f64::INFINITY, &inf)],
            0.0,
            &[0., 10., 20., 30., 40.],
        );
        assert_eq!(out[0], Some(vec![5.0, 1.0]));
        assert_eq!(out[1], Some(vec![3.0, 3.0]));
        assert_eq!(out[2], Some(vec![0.0, 1.0]));
        assert_eq!(out[3], None, "after the last sample");
        let shifted = slices(
            &[(1.0, &b1), (f64::INFINITY, &inf)],
            10.0,
            &[-20., -10., 0.],
        );
        assert_eq!(shifted, vec![None, Some(vec![5.0, 1.0])], "aligned time");
    }

    #[test]
    fn scrape_gap_is_unknown() {
        let t = [0.0, 10.0, 40.0];
        let (b1, inf) = (s(&t, &[0., 2., 8.]), s(&t, &[0., 3., 12.]));
        let out = slices(
            &[(1.0, &b1), (f64::INFINITY, &inf)],
            0.0,
            &[0., 10., 20., 30., 40.],
        );
        assert_eq!(
            out,
            vec![Some(vec![2.0, 1.0]), None, None, Some(vec![6.0, 3.0])]
        );
    }

    #[test]
    fn reset_and_noise() {
        let t = [0.0, 10.0, 20.0];
        let (b1, inf) = (s(&t, &[4., 9., 1.]), s(&t, &[5., 10., 2.]));
        let out = slices(&[(1.0, &b1), (f64::INFINITY, &inf)], 0.0, &[0., 10., 20.]);
        assert_eq!(out, vec![Some(vec![5.0, 0.0]), None]);
    }
}
