#!/usr/bin/env python3
"""Write a deterministic 48 kHz test signal (sweep + plucked notes + noise) as float WAV."""
import math, random, struct, sys

sr, out = 48000, []
random.seed(1)
# 1 s log sweep 40 Hz -> 12 kHz
for i in range(sr):
    t = i / sr
    f0, f1 = 40.0, 12000.0
    ph = 2 * math.pi * f0 * (math.exp(t * math.log(f1 / f0)) - 1) / math.log(f1 / f0)
    out.append(0.3 * math.sin(ph))
# plucked notes: decaying harmonics
for f in (82.4, 110.0, 146.8, 196.0, 329.6):
    for i in range(sr // 3):
        t = i / sr
        v = sum(math.sin(2 * math.pi * f * h * t) / h for h in range(1, 8))
        out.append(0.5 * math.exp(-5 * t) * v / 2.6)
# noise burst + silence
out += [0.2 * random.uniform(-1, 1) for _ in range(sr // 4)] + [0.0] * (sr // 4)
with open(sys.argv[1], "wb") as f:
    b = struct.pack(f"<{len(out)}f", *out)
    f.write(b"RIFF" + struct.pack("<I", 36 + len(b)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, 1, sr, sr * 4, 4, 32)
            + b"data" + struct.pack("<I", len(b)) + b)
