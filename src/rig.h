#pragma once

// Amp + cab signal chain. Independent of the Endless SDK so the exact same
// code runs in the host test harness (host/render.cpp) and on the pedal.
//
//   in -> input gain -> NAM amp (one of kNumAmps) -> cab (NAM model, IR or none)
//      -> tilt EQ -> output level -> out
//
// Everything model-specific comes from the generated header rig_data.h
// (produced by tools/build_rig.py from a rigs/*.json file).

#include <cmath>
#include <cstdint>

#include "fir.h"
#include "nam_a2.h"
#include "rig_data.h"

class Rig
{
  public:
    enum Param
    {
        kGain,  // left knob
        kTone,  // mid knob
        kLevel, // right knob
        kNumParams
    };

    static constexpr float kSampleRate = 48000.0f;

    static constexpr float paramDefault(int p) { return p == kLevel ? 0.75f : 0.5f; }

    void init()
    {
        for (int p = 0; p < kNumParams; p++)
            setParam(p, paramDefault(p));
        inGain_ = inGainTarget_;
        outGain_ = outGainTarget_;
        ampIndex_ = 0;
        pendingAmp_ = -1;
        fade_ = kFadeSamples;
        cabOn_ = true;
        toneLp_ = 0.0f;
        amp_.load(rigdata::kAmps[0].weights);
#if RIG_CAB_TYPE == RIG_CAB_NAM
        cab_.load(rigdata::kCabWeights);
#elif RIG_CAB_TYPE == RIG_CAB_IR
        cab_.load(rigdata::kIrTaps, rigdata::kIrNumTaps);
#endif
        // tone filter coefficient: one-pole low-pass at ~700 Hz
        toneA_ = 1.0f - std::exp(-2.0f * 3.14159265f * 700.0f / kSampleRate);
    }

    /// value in [0, 1]
    void setParam(int p, float v)
    {
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        switch (p)
        {
            case kGain: inGainTarget_ = dbToLin((v - 0.5f) * 36.0f); break;        // -18..+18 dB
            case kTone: tilt_ = (v - 0.5f) * 2.0f; break;                           // -1..+1
            case kLevel: outGainTarget_ = dbToLin((v - 0.75f) * 48.0f); break;     // -36..+12 dB
            default: break;
        }
    }

    void nextAmp()
    {
        if (rigdata::kNumAmps > 1)
            pendingAmp_ = (ampIndex_ + 1) % rigdata::kNumAmps;
    }
    void toggleCab() { cabOn_ = !cabOn_; }

    /// Immediate switch without fades (for setup, not while playing).
    void selectAmp(int i)
    {
        ampIndex_ = ((i % rigdata::kNumAmps) + rigdata::kNumAmps) % rigdata::kNumAmps;
        amp_.load(rigdata::kAmps[ampIndex_].weights);
    }

    int ampIndex() const { return ampIndex_; }
    bool cabOn() const { return cabOn_; }

    /// Mono in, mono out; in and out may alias. Any n.
    void process(const float* in, float* out, int n)
    {
        while (n > 0)
        {
            const int m = n < kChunk ? n : kChunk;
            processChunk(in, out, m);
            in += m;
            out += m;
            n -= m;
        }
    }

  private:
    static constexpr int kChunk = nam_a2::kMaxBlock;
    static constexpr int kFadeSamples = 480; // 10 ms fade-in after switching amps

    static float dbToLin(float db) { return std::pow(10.0f, db / 20.0f); }

    void processChunk(const float* in, float* out, int n)
    {
        float buf[kChunk];
        const float smooth = 0.002f; // ~10 ms parameter smoothing
        for (int f = 0; f < n; f++)
        {
            inGain_ += (inGainTarget_ - inGain_) * smooth;
            buf[f] = in[f] * inGain_ * rigdata::kInputTrim;
        }

        amp_.process(buf, buf, n);

        const float ampTrim = rigdata::kAmps[ampIndex_].outTrim;
        for (int f = 0; f < n; f++)
            buf[f] *= ampTrim;

#if RIG_CAB_TYPE != RIG_CAB_NONE
        if (cabOn_)
        {
            cab_.process(buf, buf, n);
            for (int f = 0; f < n; f++)
                buf[f] *= rigdata::kCabTrim;
        }
#endif

        if (tilt_ > 0.001f || tilt_ < -0.001f)
        {
            const float gh = dbToLin(tilt_ * 6.0f);
            const float gl = 1.0f / gh;
            for (int f = 0; f < n; f++)
            {
                toneLp_ += (buf[f] - toneLp_) * toneA_;
                buf[f] = toneLp_ * gl + (buf[f] - toneLp_) * gh;
            }
        }
        else
        {
            for (int f = 0; f < n; f++)
                toneLp_ += (buf[f] - toneLp_) * toneA_;
        }

        for (int f = 0; f < n; f++)
        {
            outGain_ += (outGainTarget_ - outGain_) * smooth;
            float g = outGain_;
            if (fade_ < kFadeSamples)
            {
                g *= static_cast<float>(fade_) / kFadeSamples;
                fade_++;
            }
            if (pendingAmp_ >= 0)
                g *= static_cast<float>(n - 1 - f) / n; // fade out over this chunk
            float y = buf[f] * g;
            // Safety clamp: the SDK wants output inside (-1, 1).
            y = y > 0.999f ? 0.999f : (y < -0.999f ? -0.999f : y);
            out[f] = y;
        }

        if (pendingAmp_ >= 0)
        {
            ampIndex_ = pendingAmp_;
            pendingAmp_ = -1;
            amp_.load(rigdata::kAmps[ampIndex_].weights);
            fade_ = 0;
        }
    }

    nam_a2::Model<rigdata::kAmpChannels> amp_;
#if RIG_CAB_TYPE == RIG_CAB_NAM
    nam_a2::Model<rigdata::kCabChannels> cab_;
#elif RIG_CAB_TYPE == RIG_CAB_IR
    fir::Convolver<rigdata::kIrNumTaps> cab_;
#endif

    // Everything zero-initialised (keeps a static Rig in .bss); init() sets real values.
    int ampIndex_ = 0;
    int pendingAmp_ = 0;
    int fade_ = 0;
    bool cabOn_ = false;

    float inGain_ = 0.0f, inGainTarget_ = 0.0f;
    float outGain_ = 0.0f, outGainTarget_ = 0.0f;
    float tilt_ = 0.0f;
    float toneA_ = 0.0f;
    float toneLp_ = 0.0f;
};
