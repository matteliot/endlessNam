//! Rig files → model data.
//!
//! The Endless can't read files from a patch and has no heap, so models are
//! baked into the binary. This crate loads a rig description, validates and
//! converts the `.nam` / `.wav` files it names, and either hands the data to
//! the host tools ([`BuiltRig`]) or writes Rust source for the patch
//! ([`BuiltRig::to_rust_source`]).
//!
//! Rig file (JSON, paths relative to the rig file):
//!
//! ```json
//! {
//!   "name": "Soldano SLO + RIP cab",
//!   "amps": [
//!     {"file": "../models/soldano/high_gain.nam", "name": "High Gain"},
//!     {"glob": "../models/soldano/*.nam", "gain_db": -1.5}
//!   ],
//!   "cab": {"file": "../models/rip_cab.nam"},
//!   "normalize": true,
//!   "input_gain_db": 0.0
//! }
//! ```
//!
//! * amps: `file` or `glob` (`*`/`?` in the file name), optional `name`,
//!   `gain_db`, `size` (`"lite"` default / `"full"`). All amps in a rig use the
//!   same size.
//! * cab: `.nam` (`size`, `gain_db`), `.wav` (`max_taps` default 1024,
//!   `normalize` default true = unit energy, `gain_db`), or `null`.
//! * normalize: level-match amps to -18 dB like the NAM plugin (default true).

mod nam;
mod wav;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde_json::Value;

pub use nam::{A2Model, load_a2};
pub use wav::{read_wav, write_wav_f32};

pub const SAMPLE_RATE: u32 = 48000;
const TARGET_LOUDNESS_DB: f64 = -18.0;

#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

macro_rules! bail {
    ($($t:tt)*) => { return Err($crate::Error(format!($($t)*))) };
}
pub(crate) use bail;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Size {
    Lite,
    Full,
}

impl Size {
    pub fn channels(self) -> usize {
        match self {
            Size::Lite => 3,
            Size::Full => 8,
        }
    }

    pub fn parse(s: &str) -> Result<Size> {
        match s {
            "lite" => Ok(Size::Lite),
            "full" => Ok(Size::Full),
            _ => bail!("size must be \"lite\" or \"full\", got \"{s}\""),
        }
    }
}

pub struct Amp {
    pub name: String,
    pub path: PathBuf,
    pub channels: usize,
    pub weights: Vec<f32>,
    pub loudness: Option<f64>,
    pub out_trim: f32,
}

pub enum Cab {
    None,
    Nam { path: PathBuf, channels: usize, weights: Vec<f32> },
    Ir { path: PathBuf, taps: Vec<f32>, original_len: usize, source_rate: u32 },
}

pub struct BuiltRig {
    pub name: String,
    pub input_trim: f32,
    pub amps: Vec<Amp>,
    pub amp_channels: usize,
    pub cab: Cab,
    pub cab_trim: f32,
    /// Every input file (for build-script rerun tracking).
    pub inputs: Vec<PathBuf>,
}

/// Options for building a rig without a rig file.
pub struct QuickRig {
    pub amps: Vec<PathBuf>,
    pub cab: Option<PathBuf>,
    pub size: Size,
    pub normalize: bool,
}

fn db_to_lin(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

/// Load a rig JSON file.
pub fn load_rig(path: &Path) -> Result<BuiltRig> {
    let text = std::fs::read_to_string(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let json: Value = serde_json::from_str(&text).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let base = path.parent().unwrap_or(Path::new("."));
    let mut rig = build(&json, base)?;
    rig.inputs.push(path.to_path_buf());
    Ok(rig)
}

/// Build a rig from files given directly (paths relative to the current dir).
pub fn quick_rig(q: &QuickRig) -> Result<BuiltRig> {
    let size = match q.size {
        Size::Lite => "lite",
        Size::Full => "full",
    };
    let name: Vec<String> = q
        .amps
        .iter()
        .map(|p| p.file_stem().map_or(String::new(), |s| s.to_string_lossy().into_owned()))
        .collect();
    let json = serde_json::json!({
        "name": name.join(" + "),
        "amps": q.amps.iter().map(|p| serde_json::json!({"file": p, "size": size})).collect::<Vec<_>>(),
        "cab": q.cab.as_ref().map(|p| serde_json::json!({"file": p, "size": size})),
        "normalize": q.normalize,
    });
    build(&json, Path::new("."))
}

fn get_f64(v: &Value, key: &str, default: f64) -> Result<f64> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(x) => x.as_f64().ok_or_else(|| Error(format!("\"{key}\" must be a number"))),
    }
}

fn get_bool(v: &Value, key: &str, default: bool) -> Result<bool> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(x) => x.as_bool().ok_or_else(|| Error(format!("\"{key}\" must be true/false"))),
    }
}

fn get_size(v: &Value) -> Result<Size> {
    match v.get("size") {
        None | Some(Value::Null) => Ok(Size::Lite),
        Some(Value::String(s)) => Size::parse(s),
        Some(_) => bail!("\"size\" must be a string"),
    }
}

fn get_path(v: &Value, key: &str, base: &Path) -> Result<Option<PathBuf>> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(base.join(s))),
        Some(_) => bail!("\"{key}\" must be a path string"),
    }
}

fn build(json: &Value, base: &Path) -> Result<BuiltRig> {
    let normalize = get_bool(json, "normalize", true)?;
    let mut inputs = Vec::new();

    // Expand amp entries (file or glob).
    let mut entries: Vec<(PathBuf, &Value)> = Vec::new();
    for a in json.get("amps").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]) {
        if let Some(pattern) = get_path(a, "glob", base)? {
            let matches = glob_files(&pattern)?;
            if matches.is_empty() {
                bail!("no files match {} (download the models into models/, see models/README.md)", pattern.display());
            }
            if let Some(dir) = pattern.parent() {
                inputs.push(dir.to_path_buf()); // rerun when files are added
            }
            entries.extend(matches.into_iter().map(|p| (p, a)));
        } else if let Some(p) = get_path(a, "file", base)? {
            entries.push((p, a));
        } else {
            bail!("amp entry needs \"file\" or \"glob\"");
        }
    }
    if entries.is_empty() {
        bail!("rig has no amps");
    }

    let mut amps = Vec::new();
    for (path, a) in entries {
        let size = get_size(a)?;
        let m = load_a2(&path, size)?;
        let mut trim = db_to_lin(get_f64(a, "gain_db", 0.0)?);
        if normalize {
            if let Some(l) = m.loudness {
                trim *= db_to_lin(TARGET_LOUDNESS_DB - l);
            }
        }
        // An explicit name only applies to a single-file entry.
        let name = match a.get("name").and_then(Value::as_str) {
            Some(n) if a.get("glob").is_none() => n.to_string(),
            _ => m.name.clone(),
        };
        inputs.push(path.clone());
        amps.push(Amp {
            name,
            path,
            channels: m.channels,
            weights: m.weights,
            loudness: m.loudness,
            out_trim: trim as f32,
        });
    }
    let amp_channels = amps[0].channels;
    if amps.iter().any(|a| a.channels != amp_channels) {
        bail!("all amps in a rig must use the same size (lite/full)");
    }

    let (cab, cab_trim) = match json.get("cab") {
        None | Some(Value::Null) => (Cab::None, 1.0),
        Some(c) => {
            let path = get_path(c, "file", base)?.ok_or_else(|| Error("cab needs \"file\"".into()))?;
            let trim = db_to_lin(get_f64(c, "gain_db", 0.0)?) as f32;
            inputs.push(path.clone());
            let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase());
            match ext.as_deref() {
                Some("nam") => {
                    let m = load_a2(&path, get_size(c)?)?;
                    (Cab::Nam { path, channels: m.channels, weights: m.weights }, trim)
                }
                Some("wav") => {
                    let max_taps = get_f64(c, "max_taps", 1024.0)? as usize;
                    let ir = wav::load_ir(&path, max_taps, get_bool(c, "normalize", true)?)?;
                    (
                        Cab::Ir { path, taps: ir.taps, original_len: ir.original_len, source_rate: ir.source_rate },
                        trim,
                    )
                }
                _ => bail!("{}: cab must be .nam or .wav", path.display()),
            }
        }
    };

    Ok(BuiltRig {
        name: json.get("name").and_then(Value::as_str).unwrap_or("").to_string(),
        input_trim: db_to_lin(get_f64(json, "input_gain_db", 0.0)?) as f32,
        amps,
        amp_channels,
        cab,
        cab_trim,
        inputs,
    })
}

/// `*` / `?` wildcards in the last path component only.
fn glob_files(pattern: &Path) -> Result<Vec<PathBuf>> {
    let dir = pattern.parent().unwrap_or(Path::new("."));
    let pat = pattern.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let rd = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return Ok(Vec::new()),
    };
    let mut out: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file() && wildcard(pat.as_bytes(), e.file_name().to_string_lossy().as_bytes()))
        .map(|e| e.path())
        .collect();
    out.sort();
    Ok(out)
}

fn wildcard(p: &[u8], s: &[u8]) -> bool {
    match (p.first(), s.first()) {
        (None, None) => true,
        (Some(b'*'), _) => wildcard(&p[1..], s) || (!s.is_empty() && wildcard(p, &s[1..])),
        (Some(b'?'), Some(_)) => wildcard(&p[1..], &s[1..]),
        (Some(a), Some(b)) if a == b => wildcard(&p[1..], &s[1..]),
        _ => false,
    }
}

fn rust_f32(v: f32) -> Result<String> {
    if !v.is_finite() {
        bail!("non-finite value {v} in model data");
    }
    // Debug formatting is the shortest string that round-trips the f32.
    Ok(format!("{v:?}"))
}

fn write_array(out: &mut String, name: &str, vals: &[f32]) -> Result<()> {
    writeln!(out, "static {name}: [f32; {}] = [", vals.len()).unwrap();
    for chunk in vals.chunks(8) {
        out.push_str("    ");
        for v in chunk {
            out.push_str(&rust_f32(*v)?);
            out.push_str(", ");
        }
        out.push('\n');
    }
    out.push_str("];\n\n");
    Ok(())
}

impl BuiltRig {
    /// Rust source defining `type Rig` and `static RIG_DATA` for the patch.
    pub fn to_rust_source(&self) -> Result<String> {
        let mut s = String::new();
        writeln!(s, "// GENERATED by rig-builder -- do not edit. Rig: {}", self.name).unwrap();
        let cab_type = match &self.cab {
            Cab::None => "nam_engine::NoCab".to_string(),
            Cab::Nam { channels, .. } => format!("nam_engine::A2<{channels}>"),
            Cab::Ir { taps, .. } => format!("nam_engine::Fir<{}>", taps.len().max(1)),
        };
        writeln!(s, "pub type Rig = nam_engine::Rig<{}, {cab_type}>;\n", self.amp_channels).unwrap();
        writeln!(s, "pub static RIG_DATA: nam_engine::RigData = nam_engine::RigData {{").unwrap();
        writeln!(s, "    name: {:?},", self.name).unwrap();
        writeln!(s, "    input_trim: {},", rust_f32(self.input_trim)?).unwrap();
        writeln!(s, "    amps: &AMPS,").unwrap();
        writeln!(s, "    cab: &CAB,").unwrap();
        writeln!(s, "    cab_trim: {},", rust_f32(self.cab_trim)?).unwrap();
        writeln!(s, "}};\n").unwrap();
        writeln!(s, "static AMPS: [nam_engine::AmpSlot; {}] = [", self.amps.len()).unwrap();
        for (i, a) in self.amps.iter().enumerate() {
            writeln!(
                s,
                "    nam_engine::AmpSlot {{ name: {:?}, weights: &AMP{i}, out_trim: {} }},",
                a.name,
                rust_f32(a.out_trim)?
            )
            .unwrap();
        }
        writeln!(s, "];\n").unwrap();
        for (i, a) in self.amps.iter().enumerate() {
            writeln!(s, "// {} <- {} (loudness {:?})", a.name, a.path.display(), a.loudness).unwrap();
            write_array(&mut s, &format!("AMP{i}"), &a.weights)?;
        }
        match &self.cab {
            Cab::None => write_array(&mut s, "CAB", &[])?,
            Cab::Nam { path, weights, .. } => {
                writeln!(s, "// cab <- {}", path.display()).unwrap();
                write_array(&mut s, "CAB", weights)?;
            }
            Cab::Ir { path, taps, original_len, source_rate } => {
                writeln!(s, "// cab IR <- {} ({original_len} taps @48k from {source_rate} Hz)", path.display())
                    .unwrap();
                write_array(&mut s, "CAB", taps)?;
            }
        }
        Ok(s)
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let size = if self.amp_channels == 3 { "A2-Lite" } else { "A2-Full" };
        let mut s = format!("rig '{}': {} amp(s), {size}\n", self.name, self.amps.len());
        for a in &self.amps {
            let db = 20.0 * (a.out_trim as f64).log10();
            writeln!(s, "  amp  {:<28} trim {db:+.1} dB  ({})", a.name, a.path.display()).unwrap();
        }
        match &self.cab {
            Cab::None => s.push_str("  cab  none\n"),
            Cab::Nam { path, channels, .. } => writeln!(s, "  cab  NAM {channels}-ch  ({})", path.display()).unwrap(),
            Cab::Ir { path, taps, original_len, .. } => {
                writeln!(s, "  cab  IR {} taps (of {original_len})  ({})", taps.len(), path.display()).unwrap()
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::wildcard;

    #[test]
    fn wildcards() {
        assert!(wildcard(b"*.nam", b"a b.nam"));
        assert!(!wildcard(b"*.nam", b"a.wav"));
        assert!(wildcard(b"x?z*", b"xyz.nam"));
    }
}
