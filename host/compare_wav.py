#!/usr/bin/env python3
"""Compare two float WAVs: max abs error and error relative to signal RMS."""
import math, struct, sys

def read(p):
    d = open(p, "rb").read()
    i = d.index(b"data")
    n = struct.unpack("<I", d[i + 4:i + 8])[0] // 4
    return struct.unpack(f"<{n}f", d[i + 8:i + 8 + 4 * n])

a, b = read(sys.argv[1]), read(sys.argv[2])
n = min(len(a), len(b))
err = max(abs(a[i] - b[i]) for i in range(n))
rms = math.sqrt(sum(x * x for x in b[:n]) / n)
erms = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)) / n)
db = 20 * math.log10(erms / rms) if erms > 0 else float("-inf")
print(f"samples {n}  max|diff| {err:.3g}  ref rms {rms:.3g}  error {db:.1f} dB re signal")
tol = float(sys.argv[3]) if len(sys.argv) > 3 else -80.0
sys.exit(0 if db < tol else 1)
