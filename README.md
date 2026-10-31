# endlessNam

Runs [Neural Amp Modeler](https://www.neuralampmodeler.com/the-code) amp and cab
captures on the [Polyend Endless](https://polyend.com/endless/) pedal, using
Polyend's [FxPatchSDK](https://github.com/polyend/FxPatchSDK).

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
brew install --cask gcc-arm-embedded   # Arm GNU toolchain (the Makefile finds it)
```

The Makefile also finds an unpacked toolchain in `~/opt/arm-gnu-toolchain-*/`.
That's the fallback if the Homebrew cask can't run its installer, which happens
on Intel Macs with an old Xcode.

Download models into `models/` (see [models/README.md](models/README.md)), then:

```sh
make                 # -> build/soldano_rip/soldano_rip.endl
make deploy          # copies it to /Volumes/ENDLESS (override with ENDLESS_VOLUME=...)
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
make RIG=rigs/my_rig.json deploy
```

- **amps**: one or more NAM A2 captures. The footswitch cycles through them,
  and `normalize` level-matches them to -18 dB like the NAM plugin does. Each
  entry takes `file` or `glob`, and optionally `name`, `gain_db` and `size`.
- **cab**: a NAM cab model (`.nam`), an impulse response (`.wav`, any sample
  rate or bit depth; it gets resampled to 48 kHz and truncated to `max_taps`
  with a fade), or `null`. Optional keys: `gain_db`, plus `normalize` for IRs.
- The models are compiled into the `.endl`, because patches can't read files on
  the pedal. Each rig produces its own `.endl`, and you can keep several on the
  pedal at once.

For a quick one-off build without a rig file:

```sh
python3 tools/build_rig.py --amp models/x.nam --cab models/y.wav -o build/quick/generated/rig_data.h
```

### What models work

NAM **A2** models, which TONE3000 now trains by default ("A2" toggle on a tone
page). A2 files contain a Lite (3-channel) and a Full (8-channel) network. The
pedal runs **Lite** (`"size": "lite"`, the default). Full needs roughly 7× the
compute of Lite and won't keep up on the 720 MHz Cortex-M7.

A1 models (classic Standard/Lite/Feather/Nano WaveNet, LSTM) aren't supported.
`build_rig.py` rejects anything the engine can't run and tells you why.

IR cost is `max_taps` multiply-adds per sample. The default of 1024 taps
(21 ms) covers the important part of most cab IRs. Raise it only if the LED
stays out of the red.

## Auditioning on your computer

```sh
make render IN=my_di_track.wav ARGS="--gain 0.6 --amp 2"   # -> build/<rig>/render_out.wav
make test    # checks the engine against NeuralAmpModelerCore on every model in models/
```

`host/render.cpp` runs the same `src/rig.h` code that runs on the pedal, so what
you hear on the computer is what the pedal plays (minus the pedal's own analog
input/output stages).

## Layout

| Path | What |
|---|---|
| `src/nam_a2.h` | Allocation-free NAM A2 WaveNet inference (matches NeuralAmpModelerCore's A2 path to ~-125 dB) |
| `src/fir.h` | IR convolver |
| `src/rig.h` | Signal chain: gain, amp, cab, tone, level, amp switching |
| `src/PatchImpl.cpp` | Endless SDK glue: knobs, footswitch, LED |
| `tools/build_rig.py` | Rig JSON + `.nam`/`.wav` → `build/<rig>/generated/rig_data.h` (stdlib-only Python) |
| `host/` | Native renderer and test scripts |
| `third_party/` | FxPatchSDK and NeuralAmpModelerCore (git submodules) |

## Notes and limits

- **Not yet measured on hardware.** On-device CPU load for amp plus NAM cab is
  an estimate. If the LED goes red or the audio glitches, try an IR cab or
  `"cab": null`. The red-LED meter reads the Cortex-M7 cycle counter. If the
  firmware leaves that counter disabled, the meter never trips.
- Mono: the left input is processed, and the result goes to both outputs.
- Switching amps mutes for one block and then fades in over 10 ms. There's no
  warm-up delay, because the network state is set directly to its silence
  steady state.
- Memory: each A2-Lite model takes about 94 KB of state. Patches get 512 KB for
  code and data combined.
