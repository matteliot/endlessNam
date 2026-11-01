//! Direct-form FIR for cabinet impulse responses (`.wav` IRs).
//!
//! Cost is `taps` multiply-adds per sample, so keep IRs short on the pedal
//! (rig-builder truncates to `max_taps` with a fade-out). `N` is the capacity;
//! the loaded IR may be shorter.

pub struct Fir<const N: usize> {
    len: usize,
    pos: usize,
    rev: [f32; N],
    // Doubled history: each input is written at pos and pos + len, so the
    // window of the last `len` inputs is always contiguous.
    hist: [[f32; N]; 2],
}

impl<const N: usize> Fir<N> {
    pub const NEW: Self = Self { len: 0, pos: 0, rev: [0.0; N], hist: [[0.0; N]; 2] };

    /// Load taps (truncated to `N`). An empty IR passes audio through.
    pub fn load(&mut self, taps: &[f32]) {
        let len = taps.len().min(N);
        self.len = len;
        // Stored reversed so the inner loop walks both arrays forwards.
        for (r, t) in self.rev[..len].iter_mut().zip(taps[..len].iter().rev()) {
            *r = *t;
        }
        self.reset();
    }

    pub fn reset(&mut self) {
        self.hist = [[0.0; N]; 2];
        self.pos = 0;
    }

    pub fn process(&mut self, buf: &mut [f32]) {
        let len = self.len;
        if len == 0 {
            return;
        }
        let rev = &self.rev[..len];
        let hist = self.hist.as_flattened_mut();
        for s in buf.iter_mut() {
            hist[self.pos] = *s;
            hist[self.pos + len] = *s;
            self.pos += 1;
            if self.pos == len {
                self.pos = 0;
            }
            let window = &hist[self.pos..self.pos + len];
            let mut acc = 0.0f32;
            for (w, h) in rev.iter().zip(window) {
                acc += w * h;
            }
            *s = acc;
        }
    }
}
