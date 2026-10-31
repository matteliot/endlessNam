#pragma once

// Allocation-free inference for NAM "A2" WaveNet models (A2-Lite: 3 channels,
// A2-Full: 8 channels), written for the Polyend Endless (Cortex-M7, no heap).
//
// The math mirrors NeuralAmpModelerCore's A2 fast path
// (third_party/NeuralAmpModelerCore/NAM/wavenet/a2_fast.cpp):
//
//   x[c]      = rechannel[c] * in                              (layer 0 input)
//   per layer: z = sum_k Wk * hist[t - (K-1-k)*D] + b + mixin * in
//              a = LeakyReLU(z, 0.01)
//              head_sum += a
//              x_next = x + L1x1 * a + l1b                      (residual)
//   out = head_scale * (head_b + sum_k Hk . head_sum[t - (15-k)])
//
// Weights are emitted by tools/build_rig.py already permuted into the layout
// below, so the engine reads them straight from the const arrays.

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace nam_a2
{

constexpr int kNumLayers = 23;
constexpr int kHeadKernel = 16;
constexpr float kLeakySlope = 0.01f;
constexpr int kKernelSizes[kNumLayers] = {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6};
constexpr int kDilations[kNumLayers] = {1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239,
                                        1, 13, 1, 3, 7, 17, 41, 101, 239};

// Largest block processed in one pass; bigger host buffers are chunked.
constexpr int kMaxBlock = 32;

constexpr int lookback(int layer)
{
    return (kKernelSizes[layer] - 1) * kDilations[layer];
}

// Ring buffer columns for one layer: lookback + one block of new frames, plus a
// block-sized mirror of the first columns so every tap read is contiguous.
constexpr int ringCols(int layer)
{
    return lookback(layer) + kMaxBlock;
}

constexpr int totalHistoryCols()
{
    int n = 0;
    for (int i = 0; i < kNumLayers; i++)
        n += ringCols(i) + kMaxBlock;
    return n;
}

// Number of floats in a model's weight array for a given channel count.
constexpr int weightCount(int C)
{
    int n = C; // rechannel
    for (int i = 0; i < kNumLayers; i++)
        n += kKernelSizes[i] * C * C + C + C + C * C + C; // conv w, conv b, mixin, 1x1 w, 1x1 b
    n += kHeadKernel * C + 1 + 1;                         // head w, head b, head scale
    return n;
}

static_assert(weightCount(3) == 1871, "A2-Lite weight count");
static_assert(weightCount(8) == 12146, "A2-Full weight count");

/// Runtime state for one A2 model with C channels. Weights are borrowed (const).
/// All members are zero-initialised so a static instance lands in .bss (the
/// state arrays would otherwise be stored in the patch image); call load().
/// One instance can run different weight sets of the same channel count; call
/// load() to switch (resets state to the silence steady state, no clicks from
/// garbage history, no warm-up compute).
template <int C>
class Model
{
  public:
    static constexpr int kChannels = C;
    static constexpr int kWeightCount = weightCount(C);

    void load(const float* weights)
    {
        w_ = weights;
        const float* p = w_;
        rechannel_ = p;
        p += C;
        for (int i = 0; i < kNumLayers; i++)
        {
            LayerW& L = layers_[i];
            const int K = kKernelSizes[i];
            L.conv = p;
            p += K * C * C;
            L.convB = p;
            p += C;
            L.mixin = p;
            p += C;
            L.l1x1 = p;
            p += C * C;
            L.l1x1B = p;
            p += C;
        }
        headW_ = p;
        p += kHeadKernel * C;
        headB_ = *p++;
        headScale_ = *p++;
        resetToSilence();
    }

    /// Put every history buffer in the exact state it reaches after an infinite
    /// stretch of silent input (what NAM's prewarm() converges to). With zero
    /// input every layer sees a constant column, so one pass suffices.
    void resetToSilence()
    {
        float x[C];
        float hsum[C];
        for (int c = 0; c < C; c++)
        {
            x[c] = 0.0f; // rechannel * 0
            hsum[c] = 0.0f;
        }

        float* hist = history_;
        for (int i = 0; i < kNumLayers; i++)
        {
            const LayerW& L = layers_[i];
            const int K = kKernelSizes[i];
            const int cols = ringCols(i) + kMaxBlock;
            for (int col = 0; col < cols; col++)
                std::memcpy(hist + col * C, x, sizeof(x));
            hist += cols * C;
            writePos_[i] = lookback(i);

            float a[C];
            for (int o = 0; o < C; o++)
            {
                float z = L.convB[o];
                for (int k = 0; k < K; k++)
                    for (int j = 0; j < C; j++)
                        z += L.conv[(k * C + j) * C + o] * x[j];
                a[o] = z >= 0.0f ? z : z * kLeakySlope;
                hsum[o] += a[o];
            }
            float nx[C];
            for (int o = 0; o < C; o++)
            {
                float v = x[o] + L.l1x1B[o];
                for (int j = 0; j < C; j++)
                    v += L.l1x1[j * C + o] * a[j];
                nx[o] = v;
            }
            std::memcpy(x, nx, sizeof(x));
        }

        for (int col = 0; col < kHeadHistCols; col++)
            std::memcpy(headHist_ + col * C, hsum, sizeof(hsum));
        headWritePos_ = 0;
    }

    /// Process n samples (any n). in and out may alias.
    void process(const float* in, float* out, int n)
    {
        while (n > 0)
        {
            const int m = n < kMaxBlock ? n : kMaxBlock;
            processBlock(in, out, m);
            in += m;
            out += m;
            n -= m;
        }
    }

  private:
    struct LayerW
    {
        const float* conv;  // [k][in j][out o]
        const float* convB; // [C]
        const float* mixin; // [C]
        const float* l1x1;  // [in j][out o]
        const float* l1x1B; // [C]
    };

    static constexpr int kHeadHistCols = 64; // ring for the k=16 head; must be >= 15 + kMaxBlock
    static_assert(kHeadHistCols >= kHeadKernel - 1 + kMaxBlock, "head ring too small");

    void ringWrite(int i, float* hist, int n)
    {
        const int R = ringCols(i);
        int wp = writePos_[i];
        const int first = (R - wp) < n ? (R - wp) : n;
        std::memcpy(hist + wp * C, x_, sizeof(float) * C * first);
        if (first < n)
            std::memcpy(hist, x_ + first * C, sizeof(float) * C * (n - first));
        // Mirror columns [0, kMaxBlock) to [R, R + kMaxBlock).
        std::memcpy(hist + R * C, hist, sizeof(float) * C * kMaxBlock);
        wp += n;
        if (wp >= R)
            wp -= R;
        writePos_[i] = wp;
    }

    template <int K>
    void layerK(int i, const float* hist, const float* cond, int n)
    {
        const LayerW& L = layers_[i];
        const int D = kDilations[i];
        const int R = ringCols(i);
        // Column of this block's first frame, `back` taps into the past.
        int start = writePos_[i] - n; // writePos_ already advanced
        if (start < 0)
            start += R;

        float* z = z_;
        for (int f = 0; f < n; f++)
            for (int o = 0; o < C; o++)
                z[f * C + o] = L.convB[o] + L.mixin[o] * cond[f];

        for (int k = 0; k < K; k++)
        {
            int base = start - (K - 1 - k) * D;
            if (base < 0)
                base += R;
            const float* src = hist + base * C;
            const float* W = L.conv + k * C * C;
            for (int f = 0; f < n; f++)
            {
                const float* s = src + f * C;
                float* zf = z + f * C;
                for (int j = 0; j < C; j++)
                {
                    const float sj = s[j];
                    const float* wj = W + j * C;
                    for (int o = 0; o < C; o++)
                        zf[o] += wj[o] * sj;
                }
            }
        }

        for (int f = 0; f < n; f++)
        {
            float a[C];
            float* zf = z + f * C;
            float* hs = headSum_ + f * C;
            float* xf = x_ + f * C;
            for (int o = 0; o < C; o++)
            {
                const float v = zf[o];
                a[o] = v >= 0.0f ? v : v * kLeakySlope;
                hs[o] += a[o];
            }
            for (int o = 0; o < C; o++)
            {
                float v = L.l1x1B[o];
                for (int j = 0; j < C; j++)
                    v += L.l1x1[j * C + o] * a[j];
                xf[o] += v;
            }
        }
    }

    void processBlock(const float* in, float* out, int n)
    {
        float cond[kMaxBlock];
        for (int f = 0; f < n; f++)
        {
            cond[f] = in[f];
            for (int c = 0; c < C; c++)
            {
                x_[f * C + c] = rechannel_[c] * in[f];
                headSum_[f * C + c] = 0.0f;
            }
        }

        float* hist = history_;
        for (int i = 0; i < kNumLayers; i++)
        {
            ringWrite(i, hist, n);
            if (kKernelSizes[i] == 6)
                layerK<6>(i, hist, cond, n);
            else
                layerK<15>(i, hist, cond, n);
            hist += (ringCols(i) + kMaxBlock) * C;
        }

        // Head: write head_sum into its ring, then K=16 conv to one channel.
        int wp = headWritePos_;
        for (int f = 0; f < n; f++)
        {
            std::memcpy(headHist_ + wp * C, headSum_ + f * C, sizeof(float) * C);
            float y = headB_;
            for (int k = 0; k < kHeadKernel; k++)
            {
                int col = wp - (kHeadKernel - 1 - k);
                if (col < 0)
                    col += kHeadHistCols;
                const float* s = headHist_ + col * C;
                const float* hw = headW_ + k * C;
                for (int j = 0; j < C; j++)
                    y += hw[j] * s[j];
            }
            out[f] = y * headScale_;
            if (++wp == kHeadHistCols)
                wp = 0;
        }
        headWritePos_ = wp;
    }

    const float* w_ = nullptr;
    const float* rechannel_ = nullptr;
    LayerW layers_[kNumLayers] = {};
    const float* headW_ = nullptr;
    float headB_ = 0.0f;
    float headScale_ = 0.0f;

    int writePos_[kNumLayers] = {};
    int headWritePos_ = 0;
    float history_[totalHistoryCols() * C] = {};
    float headHist_[kHeadHistCols * C] = {};
    float x_[kMaxBlock * C] = {};
    float z_[kMaxBlock * C] = {};
    float headSum_[kMaxBlock * C] = {};
};

} // namespace nam_a2
