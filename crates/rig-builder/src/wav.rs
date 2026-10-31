//! Minimal WAV I/O and IR preparation (resample to 48 kHz, truncate, fade).

use std::f64::consts::PI;
use std::path::Path;

use crate::{Error, Result, SAMPLE_RATE, bail};

/// Read the first channel of a PCM 16/24/32-bit or float 32/64-bit WAV.
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let p = path.display();
    let d = std::fs::read(path).map_err(|e| Error(format!("{p}: {e}")))?;
    if d.len() < 12 || &d[0..4] != b"RIFF" || &d[8..12] != b"WAVE" {
        bail!("{p}: not a RIFF/WAVE file");
    }
    let u16_at = |o: usize| u16::from_le_bytes([d[o], d[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]);
    let mut pos = 12;
    let mut fmt = None;
    while pos + 8 <= d.len() {
        let size = u32_at(pos + 4) as usize;
        let body = pos + 8;
        let end = (body + size).min(d.len());
        match &d[pos..pos + 4] {
            b"fmt " if size >= 16 => {
                let mut tag = u16_at(body);
                if tag == 0xFFFE && size >= 26 {
                    tag = u16_at(body + 24);
                }
                fmt = Some((tag, u16_at(body + 2) as usize, u32_at(body + 4), u16_at(body + 14)));
            }
            b"data" => {
                let Some((tag, ch, sr, bits)) = fmt else { bail!("{p}: data before fmt chunk") };
                let bps = bits as usize / 8;
                if bps == 0 || ch == 0 {
                    bail!("{p}: bad format");
                }
                let frames = (end - body) / (bps * ch);
                let mut out = Vec::with_capacity(frames);
                for i in 0..frames {
                    let s = &d[body + i * bps * ch..body + i * bps * ch + bps];
                    out.push(match (tag, bits) {
                        (3, 32) => f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
                        (3, 64) => f64::from_le_bytes(s.try_into().unwrap()) as f32,
                        (1, 16) => i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0,
                        (1, 24) => (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) as f32 / 8388608.0,
                        (1, 32) => i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / 2147483648.0,
                        _ => bail!("{p}: unsupported WAV format (tag {tag}, {bits} bits)"),
                    });
                }
                return Ok((out, sr));
            }
            _ => {}
        }
        pos = body + size + (size & 1);
    }
    bail!("{p}: no data chunk")
}

/// Write mono 32-bit float WAV.
pub fn write_wav_f32(path: &Path, x: &[f32], sr: u32) -> Result<()> {
    let mut b = Vec::with_capacity(44 + 4 * x.len());
    let bytes = (x.len() * 4) as u32;
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + bytes).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&3u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&sr.to_le_bytes());
    b.extend_from_slice(&(sr * 4).to_le_bytes());
    b.extend_from_slice(&4u16.to_le_bytes());
    b.extend_from_slice(&32u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&bytes.to_le_bytes());
    for v in x {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, b).map_err(|e| Error(format!("{}: {e}", path.display())))
}

pub(crate) struct Ir {
    pub taps: Vec<f32>,
    pub original_len: usize,
    pub source_rate: u32,
}

pub(crate) fn load_ir(path: &Path, max_taps: usize, normalize: bool) -> Result<Ir> {
    let (x, sr) = read_wav(path)?;
    let mut x = resample(&x, sr, SAMPLE_RATE);
    let original_len = x.len();
    if max_taps == 0 {
        bail!("{}: max_taps must be > 0", path.display());
    }
    if x.len() > max_taps {
        x.truncate(max_taps);
        let fade = (max_taps / 8).max(1); // raised-cosine fade over the last 1/8
        for i in 0..fade {
            x[max_taps - fade + i] *= (0.5 + 0.5 * (PI * (i + 1) as f64 / fade as f64).cos()) as f32;
        }
    }
    if normalize {
        let e = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        if e > 0.0 {
            x.iter_mut().for_each(|v| *v = (*v as f64 / e) as f32);
        }
    }
    Ok(Ir { taps: x, original_len, source_rate: sr })
}

/// Windowed-sinc resampler (plenty for short IRs).
fn resample(x: &[f32], sr_in: u32, sr_out: u32) -> Vec<f32> {
    if sr_in == sr_out {
        return x.to_vec();
    }
    let ratio = sr_out as f64 / sr_in as f64;
    let cutoff = ratio.min(1.0);
    let half = 32i64;
    let n_out = (x.len() as f64 * ratio).ceil() as usize;
    (0..n_out)
        .map(|i| {
            let t = i as f64 / ratio;
            let c = t.floor() as i64;
            let mut acc = 0.0;
            for k in (c - half + 1)..=(c + half) {
                if k < 0 || k as usize >= x.len() {
                    continue;
                }
                let u = t - k as f64;
                let w = 0.5 + 0.5 * (PI * u / half as f64).cos();
                let s = if u == 0.0 { cutoff } else { (PI * cutoff * u).sin() / (PI * u) };
                acc += x[k as usize] as f64 * s * w;
            }
            acc as f32
        })
        .collect()
}
