#pragma once

// Direct-form FIR for cabinet impulse responses (.wav IRs).
// Cost is kMaxTaps multiply-adds per sample, so keep IRs short on the pedal
// (tools/build_rig.py truncates to `max_taps` with a fade-out).

#include <cstring>

namespace fir
{

template <int kMaxTaps>
class Convolver
{
  public:
    void load(const float* taps, int numTaps)
    {
        // Store reversed so the inner loop walks both arrays forwards.
        n_ = numTaps < kMaxTaps ? numTaps : kMaxTaps;
        for (int i = 0; i < n_; i++)
            rev_[i] = taps[n_ - 1 - i];
        reset();
    }

    void reset()
    {
        std::memset(hist_, 0, sizeof(hist_));
        pos_ = 0;
    }

    void process(const float* in, float* out, int n)
    {
        if (n_ == 0)
        {
            if (out != in)
                std::memcpy(out, in, sizeof(float) * n);
            return;
        }
        for (int f = 0; f < n; f++)
        {
            // Doubled history: sample written at pos and pos + n_, so the
            // window of the last n_ inputs is always contiguous.
            hist_[pos_] = in[f];
            hist_[pos_ + n_] = in[f];
            if (++pos_ == n_)
                pos_ = 0;
            const float* h = hist_ + pos_;
            float acc = 0.0f;
            for (int k = 0; k < n_; k++)
                acc += rev_[k] * h[k];
            out[f] = acc;
        }
    }

  private:
    int n_ = 0;
    int pos_ = 0;
    float rev_[kMaxTaps] = {};
    float hist_[2 * kMaxTaps] = {};
};

} // namespace fir
