#!/bin/sh
# Checks the Endless engine against NeuralAmpModelerCore for every A2 model in
# models/ (Lite and Full submodels), and the IR convolver against a known IR.
# Usage: host/run_tests.sh path/to/nam-ref/render
set -e
REF=$1
T=build/test
IN=host/testdata/test_in.wav
CXX=${HOSTCXX:-clang++}
mkdir -p $T
[ -f $IN ] || python3 host/make_test_input.py $IN
fail=0

build_render() { # $1 = dir containing rig_data.h
  $CXX -std=c++17 -O2 -Wall -Wextra -Isrc -I"$1" host/render.cpp -o "$1/render"
}

for m in models/*.nam models/*/*.nam; do
  [ -f "$m" ] || continue
  name=$(basename "$m" .nam)
  for size in lite full; do
    d="$T/$name-$size"
    mkdir -p "$d"
    if ! python3 tools/build_rig.py --amp "$m" --size $size --no-normalize -o "$d/rig_data.h" > "$d/build.log" 2>&1; then
      echo "skip  $name ($size): $(tail -1 "$d/build.log")"
      continue
    fi
    build_render "$d"
    "$d/render" $IN "$d/mine.wav" 2>/dev/null
    slim=0.4; [ $size = full ] && slim=1.0
    "$REF" --slim $slim "$m" $IN "$d/ref.wav" > /dev/null 2>&1
    printf "%-5s %-40s " $size "$name"
    python3 host/compare_wav.py "$d/mine.wav" "$d/ref.wav" -90 || { echo "FAIL"; fail=1; }
  done
done

# IR convolver: an IR of 0.5 at tap 3 must give the amp output delayed 3, halved.
m=$(ls models/*.nam models/*/*.nam 2>/dev/null | head -1)
if [ -n "$m" ]; then
  d="$T/ir"
  mkdir -p "$d"
  python3 - "$d" "$m" <<'EOF'
import json, os, struct, sys
d, m = sys.argv[1], os.path.abspath(sys.argv[2])
taps = [0.0, 0.0, 0.0, 0.5]
b = struct.pack("<4f", *taps)
open(f"{d}/ir.wav", "wb").write(b"RIFF" + struct.pack("<I", 36 + len(b)) + b"WAVEfmt " +
    struct.pack("<IHHIIHH", 16, 3, 1, 48000, 192000, 4, 32) + b"data" + struct.pack("<I", len(b)) + b)
json.dump({"name": "ir-test", "normalize": False, "amps": [{"file": m}],
           "cab": {"file": os.path.abspath(f"{d}/ir.wav"), "normalize": False}}, open(f"{d}/rig.json", "w"))
EOF
  python3 tools/build_rig.py "$d/rig.json" -o "$d/rig_data.h" > /dev/null
  build_render "$d"
  "$d/render" $IN "$d/mine.wav" 2>/dev/null
  "$REF" --slim 0.4 "$m" $IN "$d/ref.wav" > /dev/null 2>&1
  python3 - "$d" <<'EOF'
import struct, sys
d = sys.argv[1]
def read(p):
    x = open(p, "rb").read(); i = x.index(b"data"); n = struct.unpack("<I", x[i+4:i+8])[0] // 4
    return list(struct.unpack(f"<{n}f", x[i+8:i+8+4*n]))
r = read(f"{d}/ref.wav")
exp = [0.0] * 3 + [0.5 * v for v in r[:-3]]
b = struct.pack(f"<{len(exp)}f", *exp)
open(f"{d}/expected.wav", "wb").write(b"RIFF" + struct.pack("<I", 36 + len(b)) + b"WAVEfmt " +
    struct.pack("<IHHIIHH", 16, 3, 1, 48000, 192000, 4, 32) + b"data" + struct.pack("<I", len(b)) + b)
EOF
  printf "%-5s %-40s " ir "delta-at-3 x0.5"
  python3 host/compare_wav.py "$d/mine.wav" "$d/expected.wav" -90 || { echo "FAIL"; fail=1; }
fi

[ $fail = 0 ] && echo "all tests passed" || { echo "TESTS FAILED"; exit 1; }
