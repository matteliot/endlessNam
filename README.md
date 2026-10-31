# endlessNam

Runs [Neural Amp Modeler](https://www.neuralampmodeler.com/the-code) amp and cab
captures on the [Polyend Endless](https://polyend.com/endless/) pedal. It's
written in Rust (`no_std`) and targets the patch ABI of Polyend's
[FxPatchSDK](https://github.com/polyend/FxPatchSDK).

Signal chain: `input gain → NAM amp → cab (NAM model or WAV IR) → tilt EQ → level`

| Control | Function |
|---|---|
| Left knob | **Gain**: input drive, -18 to +18 dB (noon = 0 dB) |
| Mid knob | **Tone**: tilt EQ around 700 Hz, ±6 dB (noon = flat) |
| Right knob | **Level**: output, -36 to +12 dB (3 o'clock = 0 dB) |
| Footswitch press | Next amp capture in the rig |
| Footswitch hold | Cab on/off |
| LED | Colour shows which amp capture is active. Bright = cab on, dim = cab off, red = CPU overload |

## Setup

```sh
git clone --recursive <this repo>      # or: git submodule update --init --recursive
rustup target add thumbv7em-none-eabihf
rustup component add llvm-tools
```

You don't need a C toolchain to build the patch. `cargo xtask test` uses a C++
compiler (`clang++`) once, to build NAM's reference renderer.

Download models into `models/` (see [models/README.md](models/README.md)), then:

```sh
cargo xtask build              # -> target/endless/soldano_rip.endl
cargo xtask deploy             # build + copy to /Volumes/ENDLESS (--volume DIR to override)
```

## Swapping amps and cabs

A **rig** is a small JSON file in `rigs/`. To use different models, copy one and
change the paths:

```json
{
  "name": "My rig",
  "amps": [
    {"file": "../models/plexi_crunch.nam", "name": "Crunch"},
    {"file": "../models/plexi_lead.nam",   "name": "Lead", "gain_db": -1.0},
    {"glob": "../models/soldano_slo100/*.nam"}
  ],
  "cab": {"file": "../models/v30_4x12.wav", "max_taps": 1024},
  "normalize": true
}
```

```sh
cargo xtask deploy --rig rigs/my_rig.json
cargo xtask info --rig rigs/my_rig.json      # what's in it, and whether every file works
```

- **amps**: one or more NAM A2 captures. The footswitch cycles through them,
  and `normalize` level-matches them to -18 dB like the NAM plugin does. Each
  entry takes `file` or `glob` (with `*`/`?` in the file name), and optionally
  `name`, `gain_db` and `size`.
- **cab**: a NAM cab model (`.nam`), an impulse response (`.wav`, any sample
  rate or bit depth; it gets resampled to 48 kHz and truncated to `max_taps`
  with a fade), or `null`. Optional keys: `gain_db`, plus `normalize` for IRs.
- The models are compiled into the `.endl`, because patches can't read files on
  the pedal. `crates/endless-patch/build.rs` converts the rig named by `$RIG`
  into Rust statics. Each rig produces its own `.endl`, and you can keep several
  on the pedal at once.

### What models work

NAM **A2** models, which TONE3000 now trains by default ("A2" toggle on a tone
page). A2 files contain a Lite (3-channel) and a Full (8-channel) network. The
pedal runs **Lite** (`"size": "lite"`, the default). Full needs roughly 7× the
compute of Lite and won't keep up on the 720 MHz Cortex-M7.

A1 models (classic Standard/Lite/Feather/Nano WaveNet, LSTM) aren't supported.
The rig builder rejects anything the engine can't run and tells you why.

IR cost is `max_taps` multiply-adds per sample. The default of 1024 taps
(21 ms) covers the important part of most cab IRs.

## Auditioning on your computer

```sh
cargo xtask render my_di.wav --gain 0.6 --select 2      # -> target/render/<rig>.wav
cargo xtask test     # checks the engine against NeuralAmpModelerCore on every model in models/
```

The host renderer runs the same `nam-engine` code as the pedal.

## Layout

| Path | What |
|---|---|
| `crates/nam-engine` | `no_std` engine: NAM A2 WaveNet (`a2.rs`, matches NeuralAmpModelerCore to ~-126 dB), IR convolver (`fir.rs`), signal chain (`rig.rs`) |
| `crates/rig-builder` | Rig JSON + `.nam`/`.wav` loading and validation, Rust code generation |
| `crates/endless-patch` | Pedal binary: the SDK's `PatchHeader` ABI in Rust (`abi.rs`), knobs/footswitch/LED (`main.rs`), linker script |
| `crates/host` | `endless-host` CLI: render, info, test-signal generation, comparison |
| `xtask` | `cargo xtask build/deploy/render/info/test` |
| `third_party/` | FxPatchSDK (ABI and layout reference) and NeuralAmpModelerCore (test reference), as git submodules |

## Notes and limits

- **Not yet measured on hardware.** On-device CPU load for amp plus NAM cab is
  an estimate. If the LED goes red or the audio glitches, try an IR cab or
  `"cab": null`. The red-LED meter reads the Cortex-M7 cycle counter. If the
  firmware leaves that counter disabled, the meter never trips.
- Mono: the left input is processed, and the result goes to both outputs.
- Switching amps mutes for one block and then fades in over 10 ms. There's no
  warm-up delay, because the network state is set directly to its silence
  steady state.
- Memory: each A2-Lite model takes about 93 KB of state. Patches get 512 KB for
  code and data combined, and `cargo xtask build` checks the total.
