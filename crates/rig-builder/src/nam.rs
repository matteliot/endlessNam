//! `.nam` loading: A2 shape validation and weight permutation into the
//! row layout `nam-engine` reads (see `crates/nam-engine/src/a2.rs`).

use std::path::Path;

use serde_json::Value;

use crate::{Error, Result, SAMPLE_RATE, Size, bail};

const KERNELS: [i64; 23] = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6];
const DILATIONS: [i64; 23] = [1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239];
const HEAD_K: usize = 16;

pub struct A2Model {
    pub name: String,
    pub channels: usize,
    /// Engine layout.
    pub weights: Vec<f32>,
    pub loudness: Option<f64>,
}

fn weight_count(c: usize) -> usize {
    let mut n = c;
    for &k in &KERNELS {
        n += k as usize * c * c + c + c + c * c + c;
    }
    n + HEAD_K * c + 2
}

/// Load an A2 WaveNet (plain or inside a SlimmableContainer) at the given size.
pub fn load_a2(path: &Path, size: Size) -> Result<A2Model> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error(format!("{}: {e} (download it into models/, see models/README.md)", path.display())))?;
    let d: Value = serde_json::from_str(&text).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let p = path.display();
    let want = size.channels();
    let mut loudness = d.pointer("/metadata/loudness").and_then(Value::as_f64);

    let model = match d.get("architecture").and_then(Value::as_str) {
        Some("SlimmableContainer") => {
            let subs = d.pointer("/config/submodels").and_then(Value::as_array).cloned().unwrap_or_default();
            let mut found = None;
            let mut have = Vec::new();
            for sm in &subs {
                let m = &sm["model"];
                let ch = m.pointer("/config/layers/0/channels").and_then(Value::as_i64);
                have.push(ch);
                if m.get("architecture").and_then(Value::as_str) == Some("WaveNet") && ch == Some(want as i64) {
                    found = Some(m.clone());
                    break;
                }
            }
            let Some(m) = found else {
                bail!("{p}: no {size:?} ({want}-channel) submodel; submodel channels {have:?}");
            };
            if let Some(l) = m.pointer("/metadata/loudness").and_then(Value::as_f64) {
                loudness = Some(l);
            }
            m
        }
        Some("WaveNet") => d.clone(),
        other => bail!(
            "{p}: architecture {other:?} not supported. The Endless engine runs NAM A2 models \
             (download the A2 version on TONE3000); A1 WaveNet/LSTM models are too heavy for its CPU."
        ),
    };

    if let Some(sr) = model.get("sample_rate").or(d.get("sample_rate")).and_then(Value::as_f64) {
        if sr as u32 != SAMPLE_RATE {
            bail!("{p}: model sample rate {sr} Hz, the Endless runs at {SAMPLE_RATE} Hz");
        }
    }
    let c = check_a2_shape(&model["config"]).map_err(|e| Error(format!("{p}: not an A2 model the engine supports ({e})")))?;
    if c != want {
        bail!("{p}: plain WaveNet has {c} channels but size {size:?} was requested");
    }
    let raw: Vec<f32> = model["weights"]
        .as_array()
        .ok_or_else(|| Error(format!("{p}: missing weights")))?
        .iter()
        .map(|v| v.as_f64().map(|x| x as f32).ok_or_else(|| Error(format!("{p}: non-numeric weight"))))
        .collect::<Result<_>>()?;
    let weights = permute(&raw, c).map_err(|e| Error(format!("{p}: {e}")))?;
    let name = d
        .pointer("/metadata/name")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| path.file_stem().map_or(String::new(), |s| s.to_string_lossy().into_owned()));
    Ok(A2Model { name, channels: c, weights, loudness })
}

fn check_a2_shape(cfg: &Value) -> std::result::Result<usize, String> {
    let layers = cfg.get("layers").and_then(Value::as_array).ok_or("no layers")?;
    if layers.len() != 1 {
        return Err("expected exactly one layer array".into());
    }
    if !cfg.get("head").is_none_or(Value::is_null) {
        return Err("post-stack head".into());
    }
    if !cfg.get("condition_dsp").is_none_or(Value::is_null) {
        return Err("condition_dsp".into());
    }
    if cfg.get("in_channels").and_then(Value::as_i64).unwrap_or(1) != 1 {
        return Err("in_channels != 1".into());
    }
    let la = &layers[0];
    let int = |k: &str| la.get(k).and_then(Value::as_i64);
    let c = int("channels").ok_or("channels")?;
    if !(c == 3 || c == 8) || int("bottleneck") != Some(c) {
        return Err(format!("channels={c} bottleneck={:?}", int("bottleneck")));
    }
    if int("input_size") != Some(1) || int("condition_size") != Some(1) {
        return Err("input/condition size".into());
    }
    let ints = |k: &str| la.get(k).and_then(Value::as_array).map(|a| a.iter().map(|v| v.as_i64()).collect::<Vec<_>>());
    if ints("kernel_sizes") != Some(KERNELS.iter().map(|&v| Some(v)).collect()) {
        return Err("kernel sizes".into());
    }
    if ints("dilations") != Some(DILATIONS.iter().map(|&v| Some(v)).collect()) {
        return Err("dilations".into());
    }
    let acts = la.get("activation").and_then(Value::as_array).ok_or("activation")?;
    if acts.len() != 23
        || !acts.iter().all(|a| {
            a.get("type").and_then(Value::as_str) == Some("LeakyReLU")
                && (a.get("negative_slope").and_then(Value::as_f64).unwrap_or(0.0) - 0.01).abs() < 1e-7
        })
    {
        return Err("activations must all be LeakyReLU(0.01)".into());
    }
    if let Some(gm) = la.get("gating_mode").and_then(Value::as_array) {
        if gm.iter().any(|g| g.as_str() != Some("none")) {
            return Err("gating".into());
        }
    }
    if la.get("gated").and_then(Value::as_bool) == Some(true) {
        return Err("gated".into());
    }
    if let Some(sa) = la.get("secondary_activation").and_then(Value::as_array) {
        if sa.iter().any(|v| !v.is_null()) {
            return Err("secondary activation".into());
        }
    }
    if la.pointer("/head1x1/active").and_then(Value::as_bool) == Some(true) {
        return Err("head1x1 active".into());
    }
    if la.pointer("/layer1x1/active").and_then(Value::as_bool) != Some(true)
        || la.pointer("/layer1x1/groups").and_then(Value::as_i64).unwrap_or(1) != 1
    {
        return Err("layer1x1".into());
    }
    let h = la.get("head").ok_or("layer head")?;
    if h.get("out_channels").and_then(Value::as_i64) != Some(1)
        || h.get("kernel_size").and_then(Value::as_i64) != Some(HEAD_K as i64)
        || h.get("bias").and_then(Value::as_bool) != Some(true)
        || h.get("head_dilation").and_then(Value::as_i64).unwrap_or(1) != 1
    {
        return Err(format!("layer head {h}"));
    }
    if let Some(obj) = la.as_object() {
        for (k, v) in obj {
            if k.contains("film") && (v.get("active").and_then(Value::as_bool) == Some(true) || v.as_bool() == Some(true)) {
                return Err(format!("FiLM {k}"));
            }
        }
    }
    if int("groups_input").unwrap_or(1) != 1 || int("groups_input_mixin").unwrap_or(1) != 1 {
        return Err("grouped conv".into());
    }
    if !la.get("slimmable").is_none_or(Value::is_null) {
        return Err("slimmable layer".into());
    }
    Ok(c as usize)
}

/// NAM weight-stream order → engine row layout.
fn permute(w: &[f32], c: usize) -> std::result::Result<Vec<f32>, String> {
    if w.len() != weight_count(c) {
        return Err(format!("weight count {} != expected {} for {c} channels", w.len(), weight_count(c)));
    }
    let mut it = w.iter().copied();
    let mut take = || it.next().unwrap();
    let mut out: Vec<f32> = (0..c).map(|_| take()).collect(); // rechannel
    for &k in &KERNELS {
        let k = k as usize;
        // NAM order: for out i, for in j, for tap k. Engine: [k][j][i].
        let mut conv = vec![0.0; k * c * c];
        for i in 0..c {
            for j in 0..c {
                for t in 0..k {
                    conv[(t * c + j) * c + i] = take();
                }
            }
        }
        out.extend(conv);
        out.extend((0..c).map(|_| take())); // conv bias
        out.extend((0..c).map(|_| take())); // mixin
        let mut l1 = vec![0.0; c * c]; // NAM: for out i, for in j. Engine: [j][i].
        for i in 0..c {
            for j in 0..c {
                l1[j * c + i] = take();
            }
        }
        out.extend(l1);
        out.extend((0..c).map(|_| take())); // 1x1 bias
    }
    let mut head = vec![0.0; HEAD_K * c]; // NAM: for in j, for tap k. Engine: [k][j].
    for j in 0..c {
        for t in 0..HEAD_K {
            head[t * c + j] = take();
        }
    }
    out.extend(head);
    out.push(take()); // head bias
    out.push(take()); // head scale
    Ok(out)
}
