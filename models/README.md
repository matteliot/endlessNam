# models/

Put `.nam` captures and `.wav` impulse responses here. They're git-ignored on
purpose: most captures (including TONE3000's T3K license) let you use the files
but not redistribute them.

The default rig (`rigs/soldano_rip.json`) expects:

| File | Source |
|---|---|
| `soldano_slo100/*.nam` | https://www.tone3000.com/tones/soldano-slo-100-106781. Pick **A2** and use "Download all A2". Unzip every `.nam` into `models/soldano_slo100/`. Any file names work. |
| `megadeth_rust_in_peace_cab.nam` | https://www.tone3000.com/tones/megadeth_rust_in_peace_ir-99238 (A2). This "IR" is actually a NAM cab model. |

Only **NAM A2** models work: A2-Lite runs on the pedal, and A2-Full is too heavy
for real time. A1 models (Standard/Lite/Feather/Nano WaveNet, LSTM) aren't
supported. `cargo xtask info --rig <rig>` (or any build) tells you if a file won't work.
