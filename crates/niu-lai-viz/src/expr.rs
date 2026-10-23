//! Tiny per-run formula language for custom plots.
//!
//! ```text
//! 'trainer/train_loss' / 'trainer/train_grad_norm'
//! mean('utilization_percent')          # across every matching series
//! rate('train_tokens_total') / 1e3     # per-second derivative
//! ```
//! A quoted string is a series key (exact, else a regex matching exactly one
//! key). Series are combined on the union of their timestamps, each holding
//! its latest value. Functions: sum/mean/min/max/count(regex), rate(x[, window_s]),
//! delta(x[, window_s]), abs(x), log10(x), sqrt(x), quantile(q, 'histogram'[, window_s]).
use crate::server::Series;
use regex::Regex;

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Num(f64),
    Key(String),
    Call(String, Vec<Expr>),
    Neg(Box<Expr>),
    Bin(char, Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Str(String),
    Ident(String),
    Op(char),
}

fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let cs: Vec<char> = src.chars().collect();
    let (mut i, mut out) = (0, Vec::new());
    while i < cs.len() {
        let c = cs[i];
        if c.is_whitespace() {
            i += 1;
        } else if c == '\'' || c == '"' {
            let mut s = String::new();
            i += 1;
            loop {
                match cs.get(i) {
                    None => return Err("unterminated string".into()),
                    Some('\\') if cs.get(i + 1) == Some(&c) => {
                        s.push(c);
                        i += 2;
                    }
                    Some(&q) if q == c => {
                        i += 1;
                        break;
                    }
                    Some(&ch) => {
                        s.push(ch);
                        i += 1;
                    }
                }
            }
            out.push(Tok::Str(s));
        } else if c.is_ascii_digit() || c == '.' {
            let st = i;
            while i < cs.len()
                && (cs[i].is_ascii_alphanumeric()
                    || cs[i] == '.'
                    || ((cs[i] == '-' || cs[i] == '+') && matches!(cs[i - 1], 'e' | 'E')))
            {
                i += 1;
            }
            let t: String = cs[st..i].iter().collect();
            out.push(Tok::Num(t.parse().map_err(|_| format!("bad number {t}"))?));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let st = i;
            while i < cs.len() && (cs[i].is_ascii_alphanumeric() || cs[i] == '_') {
                i += 1;
            }
            out.push(Tok::Ident(cs[st..i].iter().collect()));
        } else if "+-*/(),".contains(c) {
            out.push(Tok::Op(c));
            i += 1;
        } else {
            return Err(format!("unexpected '{c}'"));
        }
    }
    Ok(out)
}

struct Parser {
    t: Vec<Tok>,
    i: usize,
}

impl Parser {
    fn peek_op(&self, ops: &str) -> Option<char> {
        match self.t.get(self.i) {
            Some(Tok::Op(c)) if ops.contains(*c) => Some(*c),
            _ => None,
        }
    }
    fn expect(&mut self, c: char) -> Result<(), String> {
        if self.peek_op(&c.to_string()).is_some() {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{c}'"))
        }
    }
    fn expr(&mut self) -> Result<Expr, String> {
        let mut l = self.term()?;
        while let Some(op) = self.peek_op("+-") {
            self.i += 1;
            l = Expr::Bin(op, Box::new(l), Box::new(self.term()?));
        }
        Ok(l)
    }
    fn term(&mut self) -> Result<Expr, String> {
        let mut l = self.unary()?;
        while let Some(op) = self.peek_op("*/") {
            self.i += 1;
            l = Expr::Bin(op, Box::new(l), Box::new(self.unary()?));
        }
        Ok(l)
    }
    fn unary(&mut self) -> Result<Expr, String> {
        if self.peek_op("-").is_some() {
            self.i += 1;
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        self.primary()
    }
    fn primary(&mut self) -> Result<Expr, String> {
        let tok = self.t.get(self.i).cloned().ok_or("unexpected end")?;
        self.i += 1;
        match tok {
            Tok::Num(n) => Ok(Expr::Num(n)),
            Tok::Str(s) => Ok(Expr::Key(s)),
            Tok::Op('(') => {
                let e = self.expr()?;
                self.expect(')')?;
                Ok(e)
            }
            Tok::Ident(f) => {
                self.expect('(')?;
                let mut args = Vec::new();
                if self.peek_op(")").is_none() {
                    loop {
                        args.push(self.expr()?);
                        if self.peek_op(",").is_none() {
                            break;
                        }
                        self.i += 1;
                    }
                }
                self.expect(')')?;
                Ok(Expr::Call(f, args))
            }
            t => Err(format!("unexpected {t:?}")),
        }
    }
}

pub fn parse(src: &str) -> Result<Expr, String> {
    let mut p = Parser { t: lex(src)?, i: 0 };
    let e = p.expr()?;
    if p.i != p.t.len() {
        return Err(format!("trailing input at token {}", p.i + 1));
    }
    Ok(e)
}

enum Val {
    Num(f64),
    Ser(Vec<f64>, Vec<f64>),
}

/// Series in the run that an expression can see.
pub struct Scope<'a> {
    pub keys: &'a [String],
    pub series: &'a [Series],
}

/// Missing series are `Ok(None)` (the run simply doesn't have them).
type R<T> = Result<Option<T>, String>;

fn to_f64(s: &Series) -> (Vec<f64>, Vec<f64>) {
    (
        s.t.iter().map(|&x| x as f64).collect(),
        s.v.iter().map(|&x| x as f64).collect(),
    )
}

/// Union grid with step-hold: out[k][i] = latest value of series k at grid[i].
fn align(ss: &[(&[f64], &[f64])]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut grid: Vec<f64> = ss.iter().flat_map(|(t, _)| t.iter().copied()).collect();
    grid.sort_by(|a, b| a.total_cmp(b));
    grid.dedup();
    let cols = ss
        .iter()
        .map(|(t, v)| {
            let mut j = 0;
            grid.iter()
                .map(|&g| {
                    while j < t.len() && t[j] <= g {
                        j += 1;
                    }
                    if j == 0 {
                        f64::NAN
                    } else {
                        v[j - 1]
                    }
                })
                .collect()
        })
        .collect();
    (grid, cols)
}

impl Scope<'_> {
    fn key(&self, k: &str) -> Result<Option<&Series>, String> {
        if let Some(i) = self.keys.iter().position(|x| x == k) {
            return Ok(Some(&self.series[i]));
        }
        let re = Regex::new(k).map_err(|e| e.to_string())?;
        let hits: Vec<usize> = (0..self.keys.len())
            .filter(|&i| re.is_match(&self.keys[i]))
            .collect();
        match hits.len() {
            0 => Ok(None),
            1 => Ok(Some(&self.series[hits[0]])),
            n => Err(format!(
                "'{k}' matches {n} series (e.g. {}); use an exact key or an aggregate like mean('{k}')",
                self.keys[hits[0]]
            )),
        }
    }

    fn eval(&self, e: &Expr) -> R<Val> {
        Ok(Some(match e {
            Expr::Num(n) => Val::Num(*n),
            Expr::Key(k) => match self.key(k)? {
                Some(s) => {
                    let (t, v) = to_f64(s);
                    Val::Ser(t, v)
                }
                None => return Ok(None),
            },
            Expr::Neg(x) => match self.eval(x)? {
                Some(v) => map(v, |x| -x),
                None => return Ok(None),
            },
            Expr::Bin(op, a, b) => {
                let (Some(a), Some(b)) = (self.eval(a)?, self.eval(b)?) else {
                    return Ok(None);
                };
                let f = |x: f64, y: f64| match op {
                    '+' => x + y,
                    '-' => x - y,
                    '*' => x * y,
                    _ => x / y,
                };
                match (a, b) {
                    (Val::Num(x), Val::Num(y)) => Val::Num(f(x, y)),
                    (Val::Ser(t, v), Val::Num(y)) => {
                        Val::Ser(t, v.into_iter().map(|x| f(x, y)).collect())
                    }
                    (Val::Num(x), Val::Ser(t, v)) => {
                        Val::Ser(t, v.into_iter().map(|y| f(x, y)).collect())
                    }
                    (Val::Ser(ta, va), Val::Ser(tb, vb)) => {
                        let (g, c) = align(&[(&ta, &va), (&tb, &vb)]);
                        let v = c[0].iter().zip(&c[1]).map(|(&x, &y)| f(x, y)).collect();
                        Val::Ser(g, v)
                    }
                }
            }
            Expr::Call(name, args) => return self.call(name, args),
        }))
    }

    fn call(&self, name: &str, args: &[Expr]) -> R<Val> {
        let one = || -> Result<&Expr, String> {
            match args {
                [a] => Ok(a),
                _ => Err(format!("{name}() takes 1 argument")),
            }
        };
        match name {
            "sum" | "mean" | "min" | "max" | "count" => {
                let Expr::Key(pat) = one()? else {
                    return Err(format!("{name}() takes a quoted regex"));
                };
                let re = Regex::new(pat).map_err(|e| e.to_string())?;
                let owned: Vec<(Vec<f64>, Vec<f64>)> = self
                    .keys
                    .iter()
                    .zip(self.series)
                    .filter(|(k, _)| re.is_match(k))
                    .map(|(_, s)| to_f64(s))
                    .collect();
                if owned.is_empty() {
                    return Ok(None);
                }
                let refs: Vec<(&[f64], &[f64])> =
                    owned.iter().map(|(t, v)| (&t[..], &v[..])).collect();
                let (g, cols) = align(&refs);
                let v = (0..g.len())
                    .map(|i| {
                        let xs = cols.iter().map(|c| c[i]).filter(|x| x.is_finite());
                        let n = xs.clone().count();
                        match (name, n) {
                            ("count", _) => n as f64,
                            (_, 0) => f64::NAN,
                            ("sum", _) => xs.sum(),
                            ("mean", _) => xs.sum::<f64>() / n as f64,
                            ("min", _) => xs.fold(f64::INFINITY, f64::min),
                            _ => xs.fold(f64::NEG_INFINITY, f64::max),
                        }
                    })
                    .collect();
                Ok(Some(Val::Ser(g, v)))
            }
            "rate" | "delta" => {
                let (x, w) = match args {
                    [x] => (x, 0.0),
                    [x, Expr::Num(w)] => (x, *w),
                    _ => return Err(format!("{name}(x) or {name}(x, window_seconds)")),
                };
                match self.eval(x)? {
                    Some(Val::Ser(t, v)) => {
                        // Over the trailing window (PromQL-style), else between adjacent samples.
                        let mut out = vec![f64::NAN; v.len()];
                        let mut j = 0;
                        for i in 1..v.len() {
                            while j + 1 < i && t[j + 1] <= t[i] - w {
                                j += 1;
                            }
                            let j = if w > 0.0 { j } else { i - 1 };
                            let d = v[i] - v[j];
                            out[i] = if name == "delta" {
                                d
                            } else if d < 0.0 || t[i] <= t[j] {
                                f64::NAN // counter reset or duplicate timestamp
                            } else {
                                d / (t[i] - t[j])
                            };
                        }
                        Ok(Some(Val::Ser(t, out)))
                    }
                    Some(Val::Num(_)) => Err(format!("{name}() needs a series")),
                    None => Ok(None),
                }
            }
            "quantile" => {
                let (q, pat, w) = match args {
                    [Expr::Num(q), Expr::Key(p)] => (*q, p, 30.0),
                    [Expr::Num(q), Expr::Key(p), Expr::Num(w)] => (*q, p, *w),
                    _ => return Err("quantile(q, 'histogram'[, window_s]), e.g. quantile(0.99, 'ttft', 30)".into()),
                };
                if !(0.0..=1.0).contains(&q) {
                    return Err(format!("quantile: q={q} must be in [0, 1]"));
                }
                self.quantile(q, pat, w)
            }
            "abs" | "log10" | "sqrt" => {
                let f: fn(f64) -> f64 = match name {
                    "abs" => f64::abs,
                    "log10" => f64::log10,
                    _ => f64::sqrt,
                };
                Ok(self.eval(one()?)?.map(|v| map(v, f)))
            }
            _ => Err(format!(
                "unknown function {name}(); have sum mean min max count rate delta abs log10 sqrt quantile"
            )),
        }
    }
}

impl Scope<'_> {
    /// Quantile of a Prometheus histogram over a trailing window, like
    /// `histogram_quantile(q, rate(x_bucket[w]))`. `pat` is a histogram base key
    /// (`/api/keys` `hists`, optional `#`), exact or a regex matching one.
    fn quantile(&self, q: f64, pat: &str, w: f64) -> R<Val> {
        let pat = pat.strip_prefix('#').unwrap_or(pat);
        let re = Regex::new(pat).map_err(|e| e.to_string())?;
        let mut fams: std::collections::BTreeMap<String, Vec<(f64, usize)>> = Default::default();
        for (i, k) in self.keys.iter().enumerate() {
            if let Some((base, le)) = crate::hist::bucket_of(k) {
                fams.entry(base).or_default().push((le, i));
            }
        }
        let base = match fams.keys().find(|b| *b == pat) {
            Some(b) => b.clone(),
            None => {
                let hits: Vec<&String> = fams.keys().filter(|b| re.is_match(b)).collect();
                match hits.len() {
                    0 => return Ok(None),
                    1 => hits[0].clone(),
                    n => {
                        return Err(format!(
                            "'{pat}' matches {n} histograms (e.g. {}); use an exact one",
                            hits[0]
                        ))
                    }
                }
            }
        };
        let mut b = fams.remove(&base).unwrap_or_default();
        b.sort_by(|x, y| x.0.total_cmp(&y.0));
        b.dedup_by(|x, y| x.0 == y.0);
        if !b.last().is_some_and(|x| x.0.is_infinite()) {
            return Err(format!("histogram {base} has no +Inf bucket"));
        }
        let owned: Vec<(Vec<f64>, Vec<f64>)> =
            b.iter().map(|&(_, i)| to_f64(&self.series[i])).collect();
        let refs: Vec<(&[f64], &[f64])> = owned.iter().map(|(t, v)| (&t[..], &v[..])).collect();
        let (t, cols) = align(&refs);
        let le: Vec<f64> = b.iter().map(|x| x.0).collect();
        let mut out = vec![f64::NAN; t.len()];
        let mut j = 0;
        for i in 1..t.len() {
            while j + 1 < i && t[j + 1] <= t[i] - w {
                j += 1;
            }
            let j = if w > 0.0 { j } else { i - 1 };
            let d: Vec<f64> = cols.iter().map(|c| c[i] - c[j]).collect();
            // NaN: a bucket not scraped yet; negative +Inf: counter reset.
            if d.iter().all(|x| x.is_finite()) && d[d.len() - 1] >= 0.0 {
                out[i] = crate::hist::quantile(&le, &d, q).unwrap_or(f64::NAN);
            }
        }
        Ok(Some(Val::Ser(t, out)))
    }
}

fn map(v: Val, f: impl Fn(f64) -> f64) -> Val {
    match v {
        Val::Num(x) => Val::Num(f(x)),
        Val::Ser(t, v) => Val::Ser(t, v.into_iter().map(f).collect()),
    }
}

/// Evaluate `src` over one run. A constant result is not a plot, so it errors.
pub fn eval(src: &str, scope: &Scope) -> Result<Option<Series>, String> {
    let e = parse(src)?;
    match scope.eval(&e)? {
        None => Ok(None),
        Some(Val::Num(_)) => Err("expression has no series in it".into()),
        Some(Val::Ser(t, v)) => Ok(Some(Series {
            t: t.into_iter().map(|x| x as f32).collect(),
            v: v.into_iter().map(|x| x as f32).collect(),
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(t: &[f32], v: &[f32]) -> Series {
        Series {
            t: t.to_vec(),
            v: v.to_vec(),
        }
    }

    fn scope() -> (Vec<String>, Vec<Series>) {
        (
            vec![
                "a/loss".into(),
                "a/util [gpu=0]".into(),
                "a/util [gpu=1]".into(),
                "a/tokens_total".into(),
            ],
            vec![
                s(&[0., 1., 2.], &[4., 2., 1.]),
                s(&[0., 2.], &[10., 30.]),
                s(&[1., 2.], &[20., 50.]),
                s(&[0., 1., 2., 3.], &[0., 100., 300., 50.]),
            ],
        )
    }

    #[test]
    fn parses_precedence_and_strings() {
        let e = parse("-'a' + 2 * 'b{x=\"y\"}' / 1e-3").unwrap();
        assert!(matches!(e, Expr::Bin('+', _, _)));
        assert!(parse("mean('x'").is_err());
        assert!(parse("1 2").is_err());
    }

    #[test]
    fn evaluates_aggregates_rate_and_binops() {
        let (k, v) = scope();
        let sc = Scope {
            keys: &k,
            series: &v,
        };
        let m = eval("mean('util')", &sc).unwrap().unwrap();
        assert_eq!(m.t, vec![0., 1., 2.]);
        assert_eq!(m.v, vec![10., 15., 40.]);
        let r = eval("rate('tokens_total')", &sc).unwrap().unwrap();
        assert!(r.v[0].is_nan() && r.v[3].is_nan());
        assert_eq!(&r.v[1..3], &[100., 200.]);
        let w = eval("rate('tokens_total', 2)", &sc).unwrap().unwrap();
        assert_eq!(&w.v[1..3], &[100., 150.]);
        let d = eval("'a/loss' * 2 - 1", &sc).unwrap().unwrap();
        assert_eq!(d.v, vec![7., 3., 1.]);
        assert!(eval("'util'", &sc).err().unwrap().contains("matches 2"));
        assert!(eval("'nope'", &sc).unwrap().is_none());
        assert!(eval("1 + 2", &sc).is_err());
    }

    #[test]
    fn histogram_quantile_over_window() {
        let t = [0., 10., 20., 30.];
        let k: Vec<String> = [
            r#"ep/lat_bucket{le="1"}"#,
            r#"ep/lat_bucket{le="2"}"#,
            r#"ep/lat_bucket{le="+Inf"}"#,
            "ep/other_bucket{le=\"+Inf\"}",
        ]
        .map(String::from)
        .to_vec();
        // 10 obs/10s, first all <=1, then all in (1, 2].
        let v = vec![
            s(&t, &[0., 10., 10., 10.]),
            s(&t, &[0., 10., 20., 30.]),
            s(&t, &[0., 10., 20., 30.]),
            s(&t, &[0., 1., 2., 3.]),
        ];
        let sc = Scope {
            keys: &k,
            series: &v,
        };
        let p = eval("quantile(0.5, 'ep/lat', 10)", &sc).unwrap().unwrap();
        assert!(p.v[0].is_nan());
        assert_eq!(&p.v[1..], &[0.5, 1.5, 1.5]);
        let wide = eval("quantile(0.5, '#ep/lat', 30)", &sc).unwrap().unwrap();
        assert_eq!(wide.v[3], 1.25, "30 obs: 10 <=1, 20 in (1,2]");
        assert!(eval("quantile(0.5, 'ep/')", &sc)
            .err()
            .unwrap()
            .contains("matches 2"));
        assert!(eval("quantile(2, 'lat')", &sc).is_err());
        assert!(eval("quantile(0.5, 'nope')", &sc).unwrap().is_none());
    }
}
