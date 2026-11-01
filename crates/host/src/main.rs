//! Host tools for endlessNam.
//!
//!   endless-host render [rig options] IN.wav OUT.wav [--gain v] [--tone v] [--level v]
//!                       [--select N] [--no-cab] [--block N]
//!       rig options: --rig rigs/x.json | --amp a.nam [--amp b.nam ...] [--cab c.nam|c.wav]
//!                    [--size lite|full] [--no-normalize]
//!   endless-host info [rig options]
//!   endless-host gen-input OUT.wav
//!   endless-host compare A.wav B.wav [--max-db -90]
//!
//! `render` runs the same nam-engine `Rig` that runs on the pedal.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use nam_engine::{A2, CabStage, Fir, NoCab, Param, Rig, RigData};
use rig_builder::{BuiltRig, Cab, QuickRig, Size};

/// IR capacity on the host (the pedal build sizes the convolver exactly).
const HOST_MAX_IR: usize = 16384;

type Res<T> = Result<T, String>;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("render") => render(&args[1..]),
        Some("info") => parse_rig(&args[1..]).map(|(rig, _)| print!("{}", rig.summary())),
        Some("gen-input") => gen_input(&args[1..]),
        Some("compare") => compare(&args[1..]),
        _ => Err("usage: endless-host render|info|gen-input|compare ... (see crates/host/src/main.rs)".into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("endless-host: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Splits rig options from the rest; returns the built rig and remaining args.
fn parse_rig(args: &[String]) -> Res<(BuiltRig, Vec<String>)> {
    let mut rig_file = None;
    let mut q = QuickRig { amps: vec![], cab: None, size: Size::Lite, normalize: true };
    let mut rest = vec![];
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--rig" => rig_file = Some(PathBuf::from(val()?)),
            "--amp" => q.amps.push(PathBuf::from(val()?)),
            "--cab" => q.cab = Some(PathBuf::from(val()?)),
            "--size" => q.size = Size::parse(&val()?).map_err(|e| e.0)?,
            "--no-normalize" => q.normalize = false,
            _ => rest.push(a.clone()),
        }
    }
    let rig = match rig_file {
        Some(p) => rig_builder::load_rig(&p),
        None if !q.amps.is_empty() => rig_builder::quick_rig(&q),
        None => return Err("give --rig FILE or --amp FILE".into()),
    }
    .map_err(|e| e.0)?;
    Ok((rig, rest))
}

struct Knobs {
    gain: f32,
    tone: f32,
    level: f32,
    select: usize,
    no_cab: bool,
    block: usize,
}

fn render(args: &[String]) -> Res<()> {
    let (rig, rest) = parse_rig(args)?;
    let mut k = Knobs {
        gain: Param::Gain.default_value(),
        tone: Param::Tone.default_value(),
        level: Param::Level.default_value(),
        select: 0,
        no_cab: false,
        block: 32,
    };
    let mut files = vec![];
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        let mut num = || -> Res<f64> {
            it.next().ok_or(format!("{a} needs a value"))?.parse().map_err(|e| format!("{a}: {e}"))
        };
        match a.as_str() {
            "--gain" => k.gain = num()? as f32,
            "--tone" => k.tone = num()? as f32,
            "--level" => k.level = num()? as f32,
            "--select" => k.select = num()? as usize,
            "--block" => k.block = (num()? as usize).max(1),
            "--no-cab" => k.no_cab = true,
            f if !f.starts_with("--") => files.push(PathBuf::from(f)),
            f => return Err(format!("unknown option {f}")),
        }
    }
    let [input, output] = files.as_slice() else { return Err("render needs IN.wav OUT.wav".into()) };
    let (mut x, sr) = rig_builder::read_wav(input).map_err(|e| e.0)?;
    if sr != rig_builder::SAMPLE_RATE {
        eprintln!("warning: input is {sr} Hz, the Endless runs at 48000 Hz");
    }

    let data = leak_rig_data(&rig);
    let t0 = Instant::now();
    let amp_name = match (rig.amp_channels, &rig.cab) {
        (3, Cab::None) => run::<3, NoCab>(data, &k, &mut x),
        (3, Cab::Nam { channels: 3, .. }) => run::<3, A2<3>>(data, &k, &mut x),
        (3, Cab::Nam { channels: 8, .. }) => run::<3, A2<8>>(data, &k, &mut x),
        (3, Cab::Ir { .. }) => run::<3, Fir<HOST_MAX_IR>>(data, &k, &mut x),
        (8, Cab::None) => run::<8, NoCab>(data, &k, &mut x),
        (8, Cab::Nam { channels: 3, .. }) => run::<8, A2<3>>(data, &k, &mut x),
        (8, Cab::Nam { channels: 8, .. }) => run::<8, A2<8>>(data, &k, &mut x),
        (8, Cab::Ir { .. }) => run::<8, Fir<HOST_MAX_IR>>(data, &k, &mut x),
        _ => return Err("unsupported channel combination".into()),
    };
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "{}: amp '{amp_name}', cab {}, {:.1}x realtime on this computer",
        rig.name,
        if k.no_cab || matches!(rig.cab, Cab::None) { "off" } else { "on" },
        x.len() as f64 / 48000.0 / secs
    );
    rig_builder::write_wav_f32(output, &x, 48000).map_err(|e| e.0)
}

fn run<const AC: usize, C: CabStage>(data: &'static RigData, k: &Knobs, x: &mut [f32]) -> &'static str {
    // ~100-600 KB of state: keep it off the stack.
    let mut rig: Box<Rig<AC, C>> = Box::new(Rig::NEW);
    rig.init(data);
    rig.set_param(Param::Gain, k.gain);
    rig.set_param(Param::Tone, k.tone);
    rig.set_param(Param::Level, k.level);
    rig.snap_params();
    rig.select_amp(k.select);
    if k.no_cab {
        rig.toggle_cab();
    }
    for chunk in x.chunks_mut(k.block) {
        rig.process(chunk);
    }
    rig.amp_name()
}

fn leak_rig_data(rig: &BuiltRig) -> &'static RigData {
    let leak_f32 = |v: &[f32]| -> &'static [f32] { Box::leak(v.to_vec().into_boxed_slice()) };
    let amps: Vec<nam_engine::AmpSlot> = rig
        .amps
        .iter()
        .map(|a| nam_engine::AmpSlot {
            name: Box::leak(a.name.clone().into_boxed_str()),
            weights: leak_f32(&a.weights),
            out_trim: a.out_trim,
        })
        .collect();
    let cab: &'static [f32] = match &rig.cab {
        Cab::None => &[],
        Cab::Nam { weights, .. } => leak_f32(weights),
        Cab::Ir { taps, .. } => leak_f32(taps),
    };
    Box::leak(Box::new(RigData {
        name: Box::leak(rig.name.clone().into_boxed_str()),
        input_trim: rig.input_trim,
        amps: Box::leak(amps.into_boxed_slice()),
        cab,
        cab_trim: rig.cab_trim,
    }))
}

/// Deterministic 48 kHz test signal: log sweep, plucked notes, noise, silence.
fn gen_input(args: &[String]) -> Res<()> {
    let [out] = args else { return Err("gen-input needs OUT.wav".into()) };
    let sr = 48000usize;
    let mut x = Vec::new();
    let (f0, f1) = (40.0f64, 12000.0f64);
    let k = (f1 / f0).ln();
    for i in 0..sr {
        let t = i as f64 / sr as f64;
        let ph = 2.0 * std::f64::consts::PI * f0 * ((t * k).exp() - 1.0) / k;
        x.push((0.3 * ph.sin()) as f32);
    }
    for f in [82.4, 110.0, 146.8, 196.0, 329.6] {
        for i in 0..sr / 3 {
            let t = i as f64 / sr as f64;
            let v: f64 = (1..8).map(|h| (2.0 * std::f64::consts::PI * f * h as f64 * t).sin() / h as f64).sum();
            x.push((0.5 * (-5.0 * t).exp() * v / 2.6) as f32);
        }
    }
    let mut seed = 1u32; // xorshift noise
    for _ in 0..sr / 4 {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        x.push(0.2 * (seed as f32 / u32::MAX as f32 * 2.0 - 1.0));
    }
    x.extend(std::iter::repeat_n(0.0, sr / 4));
    rig_builder::write_wav_f32(Path::new(out), &x, 48000).map_err(|e| e.0)
}

/// Error of A relative to B (the reference), in dB re B's RMS.
fn compare(args: &[String]) -> Res<()> {
    let mut max_db = -90.0;
    let mut files = vec![];
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--max-db" {
            max_db = it.next().ok_or("--max-db needs a value")?.parse::<f64>().map_err(|e| e.to_string())?;
        } else {
            files.push(PathBuf::from(a));
        }
    }
    let [a, b] = files.as_slice() else { return Err("compare needs A.wav B.wav".into()) };
    let (a, _) = rig_builder::read_wav(a).map_err(|e| e.0)?;
    let (b, _) = rig_builder::read_wav(b).map_err(|e| e.0)?;
    let n = a.len().min(b.len());
    let mut max_diff = 0.0f64;
    let (mut err2, mut ref2) = (0.0f64, 0.0f64);
    for i in 0..n {
        let d = (a[i] - b[i]) as f64;
        max_diff = max_diff.max(d.abs());
        err2 += d * d;
        ref2 += (b[i] as f64).powi(2);
    }
    let db = if err2 == 0.0 { f64::NEG_INFINITY } else { 10.0 * (err2 / ref2).log10() };
    println!(
        "samples {n}  max|diff| {max_diff:.3e}  ref rms {:.3e}  error {db:.1} dB re signal",
        (ref2 / n as f64).sqrt()
    );
    if db < max_db { Ok(()) } else { Err(format!("error {db:.1} dB is above {max_db} dB")) }
}
