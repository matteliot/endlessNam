#!/usr/bin/env python3
"""Turn a rig description (amp .nam files + optional cab .nam / .wav IR) into
build/generated/rig_data.h for the Endless patch and the host test harness.

The Endless has no file system access from a patch and no heap, so models are
baked into the binary: swapping amps/cabs = edit a rig file, rebuild, copy the
new .endl to the pedal.

Rig file (JSON, paths relative to the rig file):

    {
      "name": "Soldano SLO + RIP cab",
      "amps": [
        {"file": "../models/soldano/high_gain.nam", "name": "High Gain"},
        {"file": "../models/soldano/death.nam", "size": "lite", "gain_db": -1.5}
      ],
      "cab": {"file": "../models/rip_cab.nam"},
      "normalize": true,
      "input_gain_db": 0.0
    }

"amps" entries may use {"glob": "../models/soldano/*.nam"} to take every
matching file (sorted; names from the model metadata). "cab" may be a .nam,
a .wav IR, or null. "normalize" level-matches amps to -18 dB like the NAM
plugin does.

Amp entries: "size" picks the submodel of an A2 SlimmableContainer ("lite"
= 3-channel, "full" = 8-channel; default "lite" because the Endless CPU cannot
run A2-Full in real time). All amps in one rig must use the same size.

Cab entries: .nam (A2, "size" as above, "gain_db") or .wav
("max_taps", default 1024 = 21 ms; "gain_db"; "normalize": true scales the IR
to unit energy).

Usage:
    tools/build_rig.py rigs/soldano_rip.json [-o build/generated/rig_data.h]
    tools/build_rig.py --amp a.nam --amp b.nam --cab cab.wav [-o ...]
Stdlib only (no numpy).
"""

import argparse
import glob
import json
import math
import os
import struct
import sys

A2_KERNELS = [6] * 14 + [15, 15] + [6] * 7
A2_DILATIONS = [1, 3, 7, 17, 41, 101, 239] * 2 + [1, 13] + [1, 3, 7, 17, 41, 101, 239]
HEAD_K = 16
SAMPLE_RATE = 48000
TARGET_LOUDNESS_DB = -18.0
SIZE_TO_CHANNELS = {"lite": 3, "full": 8}


class RigError(Exception):
    pass


# --------------------------------------------------------------------------- NAM


def weight_count(C):
    n = C
    for K in A2_KERNELS:
        n += K * C * C + C + C + C * C + C
    return n + HEAD_K * C + 2


def check_a2_shape(cfg, where):
    def fail(msg):
        raise RigError(f"{where}: not an A2 WaveNet the Endless engine supports ({msg})")

    layers = cfg.get("layers")
    if not isinstance(layers, list) or len(layers) != 1:
        fail("expected exactly one layer array")
    if cfg.get("head") is not None:
        fail("post-stack head")
    if cfg.get("condition_dsp") is not None:
        fail("condition_dsp")
    if cfg.get("in_channels", 1) != 1:
        fail("in_channels != 1")
    la = layers[0]
    C = la.get("channels")
    if C not in (3, 8) or la.get("bottleneck") != C:
        fail(f"channels={C} bottleneck={la.get('bottleneck')}")
    if la.get("input_size") != 1 or la.get("condition_size") != 1:
        fail("input/condition size")
    if la.get("kernel_sizes") != A2_KERNELS:
        fail("kernel sizes")
    if la.get("dilations") != A2_DILATIONS:
        fail("dilations")
    for a in la.get("activation", []):
        if not (isinstance(a, dict) and a.get("type") == "LeakyReLU" and abs(a.get("negative_slope", 0) - 0.01) < 1e-7):
            fail(f"activation {a}")
    if len(la.get("activation", [])) != len(A2_KERNELS):
        fail("activation count")
    gm = la.get("gating_mode")
    if gm is not None and any(g != "none" for g in gm):
        fail("gating")
    if la.get("gated") is True:
        fail("gated")
    sa = la.get("secondary_activation")
    if sa is not None and any(x is not None for x in sa):
        fail("secondary activation")
    if (la.get("head1x1") or {}).get("active"):
        fail("head1x1 active")
    l1 = la.get("layer1x1") or {}
    if not l1.get("active") or l1.get("groups", 1) != 1:
        fail("layer1x1")
    h = la.get("head") or {}
    if h.get("out_channels") != 1 or h.get("kernel_size") != HEAD_K or not h.get("bias") or h.get("head_dilation", 1) != 1:
        fail(f"layer head {h}")
    for k, v in la.items():
        if "film" in k and isinstance(v, dict) and v.get("active"):
            fail(f"FiLM {k}")
        if "film" in k and v is True:
            fail(f"FiLM {k}")
    if la.get("groups_input", 1) != 1 or la.get("groups_input_mixin", 1) != 1:
        fail("grouped conv")
    if la.get("slimmable") is not None:
        fail("slimmable layer")
    return C


def permute_a2(weights, C):
    """NAM weight-stream order -> engine layout (see src/nam_a2.h)."""
    if len(weights) != weight_count(C):
        raise RigError(f"weight count {len(weights)} != expected {weight_count(C)} for {C} channels")
    it = iter(weights)
    take = lambda: next(it)
    out = [take() for _ in range(C)]  # rechannel
    for K in A2_KERNELS:
        conv = [0.0] * (K * C * C)
        for i in range(C):  # out
            for j in range(C):  # in
                for k in range(K):
                    conv[(k * C + j) * C + i] = take()
        out += conv
        out += [take() for _ in range(C)]  # conv bias
        out += [take() for _ in range(C)]  # mixin
        l1 = [0.0] * (C * C)
        for i in range(C):
            for j in range(C):
                l1[j * C + i] = take()
        out += l1
        out += [take() for _ in range(C)]  # 1x1 bias
    head = [0.0] * (HEAD_K * C)
    for j in range(C):
        for k in range(HEAD_K):
            head[k * C + j] = take()
    out += head
    out.append(take())  # head bias
    out.append(take())  # head scale
    return out


def check_exists(path):
    if not os.path.isfile(path):
        raise RigError(f"{path}: file not found (download it into models/, see models/README.md)")


def load_nam(path, size):
    check_exists(path)
    with open(path) as f:
        d = json.load(f)
    want_c = SIZE_TO_CHANNELS.get(size)
    if want_c is None:
        raise RigError(f"{path}: size must be one of {list(SIZE_TO_CHANNELS)}")
    arch = d.get("architecture")
    loudness = (d.get("metadata") or {}).get("loudness")
    if arch == "SlimmableContainer":
        found = None
        for sm in d["config"]["submodels"]:
            m = sm["model"]
            if m.get("architecture") != "WaveNet":
                continue
            if m["config"]["layers"][0].get("channels") == want_c:
                found = m
                break
        if found is None:
            sizes = [sm["model"]["config"]["layers"][0].get("channels") for sm in d["config"]["submodels"]]
            raise RigError(f"{path}: no {size} ({want_c}-channel) submodel; has channels {sizes}")
        model = found
        loudness = (model.get("metadata") or {}).get("loudness", loudness)
    elif arch == "WaveNet":
        model = d
    else:
        raise RigError(
            f"{path}: architecture '{arch}' not supported. The Endless engine runs NAM A2 models "
            "(download the A2 version on TONE3000). A1 Standard WaveNet/LSTM are too heavy for its CPU.")
    sr = model.get("sample_rate", d.get("sample_rate"))
    if sr is not None and int(sr) != SAMPLE_RATE:
        raise RigError(f"{path}: model sample rate {sr} Hz, Endless runs at {SAMPLE_RATE} Hz")
    C = check_a2_shape(model["config"], path)
    if C != want_c:
        raise RigError(f"{path}: plain WaveNet has {C} channels but size '{size}' was requested")
    name = (d.get("metadata") or {}).get("name") or os.path.splitext(os.path.basename(path))[0]
    return {"channels": C, "weights": permute_a2(model["weights"], C), "loudness": loudness, "name": name}


# --------------------------------------------------------------------------- WAV


def read_wav(path):
    check_exists(path)
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise RigError(f"{path}: not a RIFF/WAVE file")
    pos, fmt, frames = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, sr, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == 0xFFFE and len(body) >= 26:
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, ch, sr, bits)
        elif cid == b"data":
            frames = body
        pos += 8 + size + (size & 1)
    if fmt is None or frames is None:
        raise RigError(f"{path}: missing fmt/data chunk")
    tag, ch, sr, bits = fmt
    bps = bits // 8
    n = len(frames) // (bps * ch)
    out = []
    for i in range(n):
        o = i * bps * ch  # first channel only
        b = frames[o:o + bps]
        if tag == 3 and bits == 32:
            v = struct.unpack("<f", b)[0]
        elif tag == 3 and bits == 64:
            v = struct.unpack("<d", b)[0]
        elif tag == 1 and bits == 16:
            v = struct.unpack("<h", b)[0] / 32768.0
        elif tag == 1 and bits == 24:
            v = int.from_bytes(b, "little", signed=True) / 8388608.0
        elif tag == 1 and bits == 32:
            v = struct.unpack("<i", b)[0] / 2147483648.0
        else:
            raise RigError(f"{path}: unsupported WAV format tag={tag} bits={bits}")
        out.append(v)
    return out, sr


def resample(x, sr_in, sr_out):
    """Windowed-sinc resampler (fine for short IRs)."""
    if sr_in == sr_out:
        return x
    ratio = sr_out / sr_in
    cutoff = min(1.0, ratio)
    half = 32
    n_out = int(math.ceil(len(x) * ratio))
    y = []
    for i in range(n_out):
        t = i / ratio
        c = int(math.floor(t))
        acc = 0.0
        for k in range(c - half + 1, c + half + 1):
            if 0 <= k < len(x):
                u = t - k
                w = 0.5 + 0.5 * math.cos(math.pi * u / half)  # Hann
                s = cutoff * (1.0 if u == 0 else math.sin(math.pi * cutoff * u) / (math.pi * cutoff * u))
                acc += x[k] * s * w
        y.append(acc)
    return y


def load_ir(path, max_taps, normalize):
    x, sr = read_wav(path)
    x = resample(x, sr, SAMPLE_RATE)
    full = len(x)
    if len(x) > max_taps:
        x = x[:max_taps]
        fade = max(1, max_taps // 8)  # raised-cosine fade over the last 1/8
        for i in range(fade):
            x[max_taps - fade + i] *= 0.5 + 0.5 * math.cos(math.pi * (i + 1) / fade)
    if normalize:
        e = math.sqrt(sum(v * v for v in x))
        if e > 0:
            x = [v / e for v in x]
    return {"taps": x, "orig_len": full, "sr": sr}


# --------------------------------------------------------------------------- output


def cf(v):
    """C++ float literal (9 significant digits round-trips float32)."""
    v = float(v)
    if not math.isfinite(v):
        raise RigError(f"non-finite value {v} in model data")
    s = f"{v:.9g}"
    if "." not in s and "e" not in s:
        s += ".0"
    return s + "f"


def fmt_floats(vals, per_line=8):
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append("    " + ", ".join(cf(v) for v in vals[i:i + per_line]) + ",")
    return "\n".join(lines)


def cstr(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def db_to_lin(db):
    return 10.0 ** (db / 20.0)


def build(rig, base_dir):
    amps_cfg = []
    for a in rig.get("amps") or []:
        if "glob" in a:  # {"glob": "../models/x/*.nam", ...} -> every match, sorted
            matches = sorted(glob.glob(os.path.join(base_dir, a["glob"])))
            if not matches:
                raise RigError(f"no files match {a['glob']} (relative to {base_dir})")
            for mpath in matches:
                e = {k: v for k, v in a.items() if k not in ("glob", "name")}
                e["file"] = mpath
                amps_cfg.append(e)
        else:
            amps_cfg.append(a)
    if not amps_cfg:
        raise RigError("rig has no amps")
    normalize = rig.get("normalize", True)
    amps = []
    for a in amps_cfg:
        path = os.path.normpath(os.path.join(base_dir, a["file"]))
        m = load_nam(path, a.get("size", "lite"))
        trim = db_to_lin(a.get("gain_db", 0.0))
        if normalize and m["loudness"] is not None:
            trim *= db_to_lin(TARGET_LOUDNESS_DB - m["loudness"])
        m["trim"] = trim
        m["label"] = a.get("name", m["name"])
        m["path"] = path
        amps.append(m)
    chans = {m["channels"] for m in amps}
    if len(chans) != 1:
        raise RigError("all amps in a rig must use the same size (lite/full)")
    amp_c = chans.pop()

    cab = rig.get("cab")
    cab_type, cab_info = "NONE", None
    if cab:
        path = os.path.normpath(os.path.join(base_dir, cab["file"]))
        trim = db_to_lin(cab.get("gain_db", 0.0))
        if path.lower().endswith(".nam"):
            cab_info = load_nam(path, cab.get("size", "lite"))
            cab_type = "NAM"
        elif path.lower().endswith(".wav"):
            cab_info = load_ir(path, int(cab.get("max_taps", 1024)), cab.get("normalize", True))
            cab_type = "IR"
        else:
            raise RigError(f"{path}: cab must be .nam or .wav")
        cab_info["trim"] = trim
        cab_info["path"] = path

    out = []
    w = out.append
    w("// GENERATED by tools/build_rig.py -- do not edit. Rebuild from the rig file instead.")
    w(f"// Rig: {rig.get('name', '(unnamed)')}")
    w("#pragma once\n")
    w("#define RIG_CAB_NONE 0\n#define RIG_CAB_NAM 1\n#define RIG_CAB_IR 2")
    w(f"#define RIG_CAB_TYPE RIG_CAB_{cab_type}\n")
    w("namespace rigdata\n{\n")
    w(f"constexpr const char* kRigName = {cstr(rig.get('name', ''))};")
    w(f"constexpr float kInputTrim = {cf(db_to_lin(rig.get('input_gain_db', 0.0)))};")
    w(f"constexpr int kAmpChannels = {amp_c};")
    w(f"constexpr int kNumAmps = {len(amps)};\n")
    for i, m in enumerate(amps):
        w(f"// {m['label']}  <- {os.path.relpath(m['path'], base_dir)}  (loudness {m['loudness']})")
        w(f"alignas(8) static const float kAmp{i}Weights[{len(m['weights'])}] = {{")
        w(fmt_floats(m["weights"]))
        w("};\n")
    w("struct Amp\n{\n    const char* name;\n    const float* weights;\n    float outTrim;\n};\n")
    w("static const Amp kAmps[kNumAmps] = {")
    for i, m in enumerate(amps):
        w(f"    {{{cstr(m['label'])}, kAmp{i}Weights, {cf(m['trim'])}}},")
    w("};\n")
    if cab_type == "NAM":
        w(f"// cab <- {os.path.relpath(cab_info['path'], base_dir)}")
        w(f"constexpr int kCabChannels = {cab_info['channels']};")
        w(f"constexpr float kCabTrim = {cf(cab_info['trim'])};")
        w(f"alignas(8) static const float kCabWeights[{len(cab_info['weights'])}] = {{")
        w(fmt_floats(cab_info["weights"]))
        w("};\n")
    elif cab_type == "IR":
        w(f"// cab IR <- {os.path.relpath(cab_info['path'], base_dir)} "
          f"({cab_info['orig_len']} taps @48k, source {cab_info['sr']} Hz)")
        w(f"constexpr float kCabTrim = {cf(cab_info['trim'])};")
        w(f"constexpr int kIrNumTaps = {len(cab_info['taps'])};")
        w(f"alignas(8) static const float kIrTaps[kIrNumTaps] = {{")
        w(fmt_floats(cab_info["taps"]))
        w("};\n")
    else:
        w("constexpr float kCabTrim = 1.0f;\n")
    w("} // namespace rigdata")

    summary = [f"rig '{rig.get('name', '')}': {len(amps)} amp(s), A2-{'Lite' if amp_c == 3 else 'Full'}"]
    for m in amps:
        summary.append(f"  amp  {m['label']:<24} trim {20 * math.log10(m['trim']):+.1f} dB  ({m['path']})")
    if cab_type == "NAM":
        summary.append(f"  cab  NAM {cab_info['channels']}-ch  ({cab_info['path']})")
    elif cab_type == "IR":
        summary.append(f"  cab  IR {len(cab_info['taps'])} taps (of {cab_info['orig_len']})  ({cab_info['path']})")
    else:
        summary.append("  cab  none")
    return "\n".join(out) + "\n", "\n".join(summary)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rig", nargs="?", help="rig JSON file")
    ap.add_argument("--amp", action="append", default=[], help="amp .nam (repeatable; instead of a rig file)")
    ap.add_argument("--cab", help="cab .nam or .wav (with --amp)")
    ap.add_argument("--size", default="lite", choices=list(SIZE_TO_CHANNELS))
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("-o", "--output", default="build/generated/rig_data.h")
    args = ap.parse_args()

    if args.rig:
        with open(args.rig) as f:
            rig = json.load(f)
        base = os.path.dirname(os.path.abspath(args.rig))
    elif args.amp:
        base = os.getcwd()
        rig = {
            "name": " + ".join(os.path.splitext(os.path.basename(a))[0] for a in args.amp),
            "amps": [{"file": a, "size": args.size} for a in args.amp],
            "cab": {"file": args.cab, "size": args.size} if args.cab else None,
            "normalize": not args.no_normalize,
        }
    else:
        ap.error("give a rig file or --amp")

    try:
        header, summary = build(rig, base)
    except RigError as e:
        sys.exit(f"build_rig: error: {e}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # Only rewrite when changed so make doesn't rebuild needlessly.
    old = None
    if os.path.exists(args.output):
        with open(args.output) as f:
            old = f.read()
    if old != header:
        with open(args.output, "w") as f:
            f.write(header)
    print(summary)


if __name__ == "__main__":
    main()
