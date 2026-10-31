// Host-side harness: runs the exact patch signal chain (src/rig.h + the
// generated rig_data.h) over a WAV file so rigs can be auditioned and checked
// against NeuralAmpModelerCore before flashing the pedal.
//
//   build/host/render in.wav out.wav [--gain 0..1] [--tone 0..1] [--level 0..1]
//                                    [--amp N] [--no-cab] [--block N]

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "rig.h"

static bool readWav(const char* path, std::vector<float>& out, uint32_t& sr)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    std::vector<char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (d.size() < 12 || std::memcmp(d.data(), "RIFF", 4) || std::memcmp(d.data() + 8, "WAVE", 4))
        return false;
    uint16_t tag = 0, ch = 0, bits = 0;
    size_t pos = 12;
    while (pos + 8 <= d.size())
    {
        uint32_t size;
        std::memcpy(&size, d.data() + pos + 4, 4);
        const char* body = d.data() + pos + 8;
        if (!std::memcmp(d.data() + pos, "fmt ", 4))
        {
            std::memcpy(&tag, body, 2);
            std::memcpy(&ch, body + 2, 2);
            std::memcpy(&sr, body + 4, 4);
            std::memcpy(&bits, body + 14, 2);
            if (tag == 0xFFFE)
                std::memcpy(&tag, body + 24, 2);
        }
        else if (!std::memcmp(d.data() + pos, "data", 4))
        {
            const int bps = bits / 8;
            const size_t n = size / (bps * ch);
            out.resize(n);
            for (size_t i = 0; i < n; i++)
            {
                const char* s = body + i * bps * ch;
                if (tag == 3 && bits == 32)
                    std::memcpy(&out[i], s, 4);
                else if (tag == 1 && bits == 16)
                {
                    int16_t v;
                    std::memcpy(&v, s, 2);
                    out[i] = v / 32768.0f;
                }
                else if (tag == 1 && bits == 24)
                {
                    int32_t v = (uint8_t)s[0] | ((uint8_t)s[1] << 8) | ((int8_t)s[2] << 16);
                    out[i] = v / 8388608.0f;
                }
                else
                {
                    std::fprintf(stderr, "unsupported wav format %d/%d\n", tag, bits);
                    return false;
                }
            }
            return true;
        }
        pos += 8 + size + (size & 1);
    }
    return false;
}

static bool writeWav(const char* path, const std::vector<float>& x, uint32_t sr)
{
    std::ofstream f(path, std::ios::binary);
    auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    const uint32_t bytes = static_cast<uint32_t>(x.size() * 4);
    f.write("RIFF", 4);
    u32(36 + bytes);
    f.write("WAVEfmt ", 8);
    u32(16);
    u16(3);
    u16(1);
    u32(sr);
    u32(sr * 4);
    u16(4);
    u16(32);
    f.write("data", 4);
    u32(bytes);
    f.write(reinterpret_cast<const char*>(x.data()), bytes);
    return f.good();
}

static Rig rig; // big (model histories) -> static, same as on the pedal

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr,
                     "usage: %s in.wav out.wav [--gain v] [--tone v] [--level v] [--amp N] [--no-cab] [--block N]\n",
                     argv[0]);
        return 1;
    }
    rig.init();
    int block = 32;
    for (int i = 3; i < argc; i++)
    {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::atof(argv[++i]) : 0.0; };
        if (a == "--gain")
            rig.setParam(Rig::kGain, (float)next());
        else if (a == "--tone")
            rig.setParam(Rig::kTone, (float)next());
        else if (a == "--level")
            rig.setParam(Rig::kLevel, (float)next());
        else if (a == "--amp")
            rig.selectAmp((int)next());
        else if (a == "--no-cab")
            rig.toggleCab();
        else if (a == "--block")
            block = (int)next();
        else
        {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 1;
        }
    }

    std::vector<float> x;
    uint32_t sr = 0;
    if (!readWav(argv[1], x, sr))
    {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    if (sr != 48000)
        std::fprintf(stderr, "warning: input is %u Hz, the Endless runs at 48000 Hz\n", sr);

    std::vector<float> y(x.size());
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t pos = 0; pos < x.size(); pos += block)
    {
        const int n = (int)std::min<size_t>(block, x.size() - pos);
        rig.process(x.data() + pos, y.data() + pos, n);
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "%s: amp '%s', cab %s, %.2fx realtime on this host\n", rigdata::kRigName,
                 rigdata::kAmps[rig.ampIndex()].name, rig.cabOn() ? "on" : "off",
                 (x.size() / 48000.0) / secs);
    return writeWav(argv[2], y, 48000) ? 0 : 1;
}
