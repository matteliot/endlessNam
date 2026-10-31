// Endless patch: NAM amp + cab rig.
//
// Controls
//   Left knob   Gain   (input drive, -18..+18 dB, noon = 0 dB)
//   Mid knob    Tone   (tilt EQ around 700 Hz, +-6 dB, noon = flat)
//   Right knob  Level  (output, -36..+12 dB, 3 o'clock = 0 dB)
//   Footswitch press  next amp capture in the rig
//   Footswitch hold   cab on/off
//   LED: colour = amp capture; bright = cab on, dim = cab off; red = CPU overload
//
// The models are compiled in from build/generated/rig_data.h; see README.md for
// how to swap them.

#include "Patch.h"

#include "rig.h"

#ifndef ENDLESS_CPU_METER
#define ENDLESS_CPU_METER 1
#endif

namespace
{

// (cab on, cab off) colour per amp slot.
constexpr Patch::Color kAmpColors[][2] = {
    {Patch::Color::kBlue, Patch::Color::kDimBlue},
    {Patch::Color::kLightGreen, Patch::Color::kDimGreen},
    {Patch::Color::kLightYellow, Patch::Color::kDimYellow},
    {Patch::Color::kLightBlueColor, Patch::Color::kDimCyan},
    {Patch::Color::kMagenta, Patch::Color::kDarkCobalt},
    {Patch::Color::kBeige, Patch::Color::kDimWhite},
    {Patch::Color::kPastelGreen, Patch::Color::kDarkLime},
};
constexpr int kNumAmpColors = sizeof(kAmpColors) / sizeof(kAmpColors[0]);

#if ENDLESS_CPU_METER
// Cortex-M7 DWT cycle counter. Read-only use: if the firmware has not enabled
// it the value never changes and the meter simply never trips.
inline uint32_t cycles()
{
    return *reinterpret_cast<volatile uint32_t*>(0xE0001004u);
}
constexpr uint32_t kCpuHz = 720000000u;
constexpr uint32_t kCyclesPerSample = kCpuHz / Patch::kSampleRate;
#endif

// Kept outside PatchImpl: a polymorphic object (non-zero vtable pointer) gets
// constant-initialised into .data, which would store the ~100 KB+ of model
// state in the .endl file. A plain all-zero object goes to .bss.
Rig rig_;

} // namespace

class PatchImpl : public Patch
{
  public:
    void init() override { rig_.init(); }

    void setWorkingBuffer(std::span<float, kWorkingBufferSize> /* buffer */) override {}

    void processAudio(std::span<float> left, std::span<float> right) override
    {
#if ENDLESS_CPU_METER
        const uint32_t t0 = cycles();
#endif
        const int n = static_cast<int>(left.size());
        rig_.process(left.data(), left.data(), n);
        for (int i = 0; i < n; i++)
            right[i] = left[i];
#if ENDLESS_CPU_METER
        const uint32_t used = cycles() - t0;
        // Trip at 90% of the per-block budget; hold the red LED ~1 s.
        if (used != 0 && static_cast<uint64_t>(used) * 10 > static_cast<uint64_t>(kCyclesPerSample) * 9 * n)
            overloadHold_ = kSampleRate;
        else if (overloadHold_ > 0)
            overloadHold_ -= n;
#endif
    }

    ParameterMetadata getParameterMetadata(int idx) override
    {
        return ParameterMetadata{0.0f, 1.0f, Rig::paramDefault(idx)};
    }

    void setParamValue(int idx, float value) override { rig_.setParam(idx, value); }

    void handleAction(int idx) override
    {
        if (idx == static_cast<int>(endless::ActionId::kLeftFootSwitchPress))
            rig_.nextAmp();
        else if (idx == static_cast<int>(endless::ActionId::kLeftFootSwitchHold))
            rig_.toggleCab();
    }

    Color getStateLedColor() override
    {
        if (overloadHold_ > 0)
            return Color::kRed;
        return kAmpColors[rig_.ampIndex() % kNumAmpColors][rig_.cabOn() ? 0 : 1];
    }

  private:
    int overloadHold_ = 0;
};

static PatchImpl patch;

Patch* Patch::getInstance()
{
    return &patch;
}
