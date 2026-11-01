//! NAM "A2" WaveNet inference.
//!
//! The math mirrors NeuralAmpModelerCore's A2 fast path
//! (`third_party/NeuralAmpModelerCore/NAM/wavenet/a2_fast.cpp`):
//!
//! ```text
//! x[c]       = rechannel[c] * in                               (layer 0 input)
//! per layer: z = Σ_k W_k · hist[t - (K-1-k)·D] + b + mixin · in
//!            a = LeakyReLU(z, 0.01)
//!            head_sum += a
//!            x_next = x + L1x1 · a + l1b                        (residual)
//! out = head_scale · (head_b + Σ_k H_k · head_sum[t - (15-k)])
//! ```
//!
//! Weights come from `rig-builder`, already permuted into rows of `C` floats:
//!
//! ```text
//! rechannel                         1 row
//! per layer: conv  [k][in j] → [out]  K·C rows
//!            conv bias                1 row
//!            mixin                    1 row
//!            1x1   [in j] → [out]     C rows
//!            1x1 bias                 1 row
//! head       [k] → [in j]             16 rows
//! head bias, head scale               2 scalars
//! ```

pub(crate) const NUM_LAYERS: usize = 23;
const HEAD_K: usize = 16;
const LEAK: f32 = 0.01;
pub(crate) const KERNELS: [usize; NUM_LAYERS] = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6];
pub(crate) const DILATIONS: [usize; NUM_LAYERS] = [
    1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239,
];

/// Largest block processed in one pass; longer buffers are split.
pub const MAX_BLOCK: usize = 32;

const fn lookback(i: usize) -> usize {
    (KERNELS[i] - 1) * DILATIONS[i]
}

/// Ring columns for one layer: lookback plus one block of new frames.
const fn ring_cols(i: usize) -> usize {
    lookback(i) + MAX_BLOCK
}

/// Start of each layer's ring in the shared history array. Each ring is
/// followed by a `MAX_BLOCK`-column mirror of its first columns, so every tap
/// reads one contiguous run of frames.
const HIST_OFFSETS: [usize; NUM_LAYERS + 1] = {
    let mut o = [0; NUM_LAYERS + 1];
    let mut i = 0;
    while i < NUM_LAYERS {
        o[i + 1] = o[i] + ring_cols(i) + MAX_BLOCK;
        i += 1;
    }
    o
};
const TOTAL_COLS: usize = HIST_OFFSETS[NUM_LAYERS];

/// Rows of C floats per layer: conv K·C, conv bias, mixin, 1x1 C, 1x1 bias.
const fn layer_rows(i: usize, c: usize) -> usize {
    KERNELS[i] * c + c + 3
}

/// Number of floats in an A2 model with `c` channels.
pub const fn weight_count(c: usize) -> usize {
    let mut rows = 1 + HEAD_K;
    let mut i = 0;
    while i < NUM_LAYERS {
        rows += layer_rows(i, c);
        i += 1;
    }
    rows * c + 2
}

const _: () = assert!(weight_count(3) == 1871);
const _: () = assert!(weight_count(8) == 12146);

const HEAD_HIST: usize = 64;
const _: () = assert!(HEAD_HIST >= HEAD_K - 1 + MAX_BLOCK);

/// Runtime state for one A2 model with `C` channels. Weights are borrowed.
/// One instance can run any weight set of the same channel count; [`A2::load`]
/// switches (and resets to the silence steady state, so no warm-up compute).
pub struct A2<const C: usize> {
    // Option so the unloaded value is all-zero bytes (keeps statics in .bss).
    rows: Option<&'static [[f32; C]]>,
    head_b: f32,
    head_scale: f32,
    write_pos: [usize; NUM_LAYERS],
    head_wp: usize,
    hist: [[f32; C]; TOTAL_COLS],
    head_hist: [[f32; C]; HEAD_HIST],
    x: [[f32; C]; MAX_BLOCK],
    z: [[f32; C]; MAX_BLOCK],
    head_sum: [[f32; C]; MAX_BLOCK],
}

impl<const C: usize> A2<C> {
    pub const WEIGHT_COUNT: usize = weight_count(C);

    /// Zeroed, unloaded model (processing before [`A2::load`] outputs silence).
    pub const NEW: Self = Self {
        rows: None,
        head_b: 0.0,
        head_scale: 0.0,
        write_pos: [0; NUM_LAYERS],
        head_wp: 0,
        hist: [[0.0; C]; TOTAL_COLS],
        head_hist: [[0.0; C]; HEAD_HIST],
        x: [[0.0; C]; MAX_BLOCK],
        z: [[0.0; C]; MAX_BLOCK],
        head_sum: [[0.0; C]; MAX_BLOCK],
    };

    /// Use a weight set (length must be [`weight_count`]`(C)`; shorter or
    /// longer slices leave the model silent).
    pub fn load(&mut self, weights: &'static [f32]) {
        if weights.len() != Self::WEIGHT_COUNT {
            self.rows = None;
            return;
        }
        let (rows, rest) = weights.split_at(weights.len() - 2);
        self.rows = Some(rows.as_chunks::<C>().0);
        self.head_b = rest[0];
        self.head_scale = rest[1];
        self.reset_to_silence();
    }

    /// Put every history buffer in the state it reaches after an infinite
    /// stretch of silence (what NAM's `prewarm()` converges to). With zero
    /// input each layer sees a constant column, so one pass computes it.
    pub fn reset_to_silence(&mut self) {
        let Some(rows) = self.rows else { return };
        let mut x = [0.0f32; C]; // rechannel * 0
        let mut hsum = [0.0f32; C];
        let mut r = 1;
        for i in 0..NUM_LAYERS {
            let k_size = KERNELS[i];
            self.hist[HIST_OFFSETS[i]..HIST_OFFSETS[i + 1]].fill(x);
            self.write_pos[i] = lookback(i);

            let conv = &rows[r..r + k_size * C];
            let conv_b = &rows[r + k_size * C];
            let l1 = &rows[r + k_size * C + 2..r + k_size * C + 2 + C];
            let l1_b = &rows[r + k_size * C + 2 + C];
            let mut a = *conv_b;
            for w in conv.chunks_exact(C) {
                for j in 0..C {
                    for o in 0..C {
                        a[o] += w[j][o] * x[j];
                    }
                }
            }
            for o in 0..C {
                a[o] = leaky(a[o]);
                hsum[o] += a[o];
            }
            for o in 0..C {
                let mut v = x[o] + l1_b[o];
                for j in 0..C {
                    v += l1[j][o] * a[j];
                }
                x[o] = v;
            }
            r += layer_rows(i, C);
        }
        self.head_hist.fill(hsum);
        self.head_wp = 0;
    }

    /// Process any number of samples in place.
    pub fn process(&mut self, buf: &mut [f32]) {
        let Some(rows) = self.rows else {
            buf.fill(0.0);
            return;
        };
        for chunk in buf.chunks_mut(MAX_BLOCK) {
            self.process_block(rows, chunk);
        }
    }

    fn process_block(&mut self, rows: &'static [[f32; C]], buf: &mut [f32]) {
        let n = buf.len();
        let rechannel = &rows[0];
        let mut cond = [0.0f32; MAX_BLOCK];
        cond[..n].copy_from_slice(buf);
        for f in 0..n {
            for c in 0..C {
                self.x[f][c] = rechannel[c] * cond[f];
            }
            self.head_sum[f] = [0.0; C];
        }

        let mut r = 1;
        for i in 0..NUM_LAYERS {
            self.layer(i, &rows[r..r + layer_rows(i, C)], &cond[..n]);
            r += layer_rows(i, C);
        }

        // Head: k=16 conv over head_sum, Bottleneck -> 1.
        let head = &rows[r..r + HEAD_K];
        let mut wp = self.head_wp;
        for (f, y_out) in buf.iter_mut().enumerate() {
            self.head_hist[wp] = self.head_sum[f];
            let mut y = self.head_b;
            for (k, hw) in head.iter().enumerate() {
                let back = HEAD_K - 1 - k;
                let col = if wp >= back { wp - back } else { wp + HEAD_HIST - back };
                let s = &self.head_hist[col];
                for j in 0..C {
                    y += hw[j] * s[j];
                }
            }
            *y_out = y * self.head_scale;
            wp += 1;
            if wp == HEAD_HIST {
                wp = 0;
            }
        }
        self.head_wp = wp;
    }

    fn layer(&mut self, i: usize, w: &[[f32; C]], cond: &[f32]) {
        let n = cond.len();
        let k_size = KERNELS[i];
        let d = DILATIONS[i];
        let ring = ring_cols(i);
        let hist = &mut self.hist[HIST_OFFSETS[i]..HIST_OFFSETS[i + 1]];

        // Write this block's inputs into the ring, then refresh the mirror.
        let wp = self.write_pos[i];
        let first = n.min(ring - wp);
        hist[wp..wp + first].copy_from_slice(&self.x[..first]);
        hist[..n - first].copy_from_slice(&self.x[first..n]);
        hist.copy_within(0..MAX_BLOCK, ring);
        let start = wp; // ring column of this block's first frame
        self.write_pos[i] = if wp + n >= ring { wp + n - ring } else { wp + n };

        let (conv, rest) = w.split_at(k_size * C);
        let conv_b = &rest[0];
        let mixin = &rest[1];
        let l1 = &rest[2..2 + C];
        let l1_b = &rest[2 + C];

        let z = &mut self.z[..n];
        for (zf, &cf) in z.iter_mut().zip(cond) {
            for o in 0..C {
                zf[o] = conv_b[o] + mixin[o] * cf;
            }
        }
        for (k, wk) in conv.chunks_exact(C).enumerate() {
            let back = (k_size - 1 - k) * d;
            let base = if start >= back { start - back } else { start + ring - back };
            let src = &hist[base..base + n];
            for (zf, s) in z.iter_mut().zip(src) {
                for j in 0..C {
                    let sj = s[j];
                    let wj = &wk[j];
                    for o in 0..C {
                        zf[o] += wj[o] * sj;
                    }
                }
            }
        }

        for ((zf, hs), xf) in z.iter().zip(&mut self.head_sum[..n]).zip(&mut self.x[..n]) {
            let mut a = [0.0f32; C];
            for o in 0..C {
                a[o] = leaky(zf[o]);
                hs[o] += a[o];
            }
            for o in 0..C {
                let mut v = l1_b[o];
                for j in 0..C {
                    v += l1[j][o] * a[j];
                }
                xf[o] += v;
            }
        }
    }
}

#[inline(always)]
fn leaky(v: f32) -> f32 {
    if v >= 0.0 { v } else { v * LEAK }
}
