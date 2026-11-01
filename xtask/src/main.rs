//! Project tasks: `cargo xtask <command>`.
//!
//!   build  [--rig rigs/x.json]                 build the pedal patch -> target/endless/<rig>.endl
//!   deploy [--rig rigs/x.json] [--volume DIR]  build and copy to the pedal (default /Volumes/ENDLESS)
//!   render [--rig rigs/x.json] IN.wav [OUT.wav] [knob options]   audition a rig on this computer
//!   info   [--rig rigs/x.json]                 show what a rig contains
//!   test                                       check the engine against NeuralAmpModelerCore

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const DEFAULT_RIG: &str = "rigs/soldano_rip.json";
const TARGET: &str = "thumbv7em-none-eabihf";
/// Patch image + bss must fit the firmware's 512 KB patch region.
const PATCH_RAM: u64 = 512 * 1024;

type Res<T> = Result<T, String>;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").canonicalize().unwrap();
    std::env::set_current_dir(&root).unwrap();
    let r = match args.first().map(String::as_str) {
        Some("build") => build(&args[1..]).map(|_| ()),
        Some("deploy") => deploy(&args[1..]),
        Some("render") => render(&args[1..]),
        Some("info") => host_tool(&prefix_rig(&args[1..], "info")),
        Some("test") => test(),
        _ => Err("usage: cargo xtask build|deploy|render|info|test (see xtask/src/main.rs)".into()),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("xtask: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(cmd: &mut Command) -> Res<()> {
    let st = cmd.status().map_err(|e| format!("{cmd:?}: {e}"))?;
    if st.success() { Ok(()) } else { Err(format!("command failed: {cmd:?}")) }
}

fn output(cmd: &mut Command) -> Res<String> {
    let o = cmd.output().map_err(|e| format!("{cmd:?}: {e}"))?;
    if !o.status.success() {
        return Err(format!("command failed: {cmd:?}\n{}", String::from_utf8_lossy(&o.stderr)));
    }
    Ok(String::from_utf8_lossy(&o.stdout).into_owned())
}

/// Pull `--rig X` out of args (default rig if absent).
fn take_rig(args: &[String]) -> (PathBuf, Vec<String>) {
    let mut rig = PathBuf::from(DEFAULT_RIG);
    let mut rest = vec![];
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--rig" {
            if let Some(v) = it.next() {
                rig = PathBuf::from(v);
            }
        } else {
            rest.push(a.clone());
        }
    }
    (std::path::absolute(&rig).unwrap_or(rig), rest)
}

fn prefix_rig(args: &[String], cmd: &str) -> Vec<String> {
    let (rig, rest) = take_rig(args);
    let mut v = vec![cmd.to_string(), "--rig".into(), rig.display().to_string()];
    v.extend(rest);
    v
}

fn llvm_tool(name: &str) -> Res<PathBuf> {
    let sysroot = output(Command::new("rustc").arg("--print").arg("sysroot"))?;
    let host = output(Command::new("rustc").arg("-vV"))?
        .lines()
        .find_map(|l| l.strip_prefix("host: ").map(str::to_string))
        .ok_or("cannot determine host triple")?;
    let p = Path::new(sysroot.trim()).join("lib/rustlib").join(host).join("bin").join(name);
    if p.exists() { Ok(p) } else { Err(format!("{name} not found; run: rustup component add llvm-tools")) }
}

fn build(args: &[String]) -> Res<PathBuf> {
    let (rig, _) = take_rig(args);
    let name = rig.file_stem().ok_or("bad rig path")?.to_string_lossy().into_owned();
    run(Command::new("cargo")
        .args(["build", "-q", "-p", "endless-patch", "--release", "--target", TARGET])
        .env("RIG", &rig))?;

    let elf = PathBuf::from(format!("target/{TARGET}/release/endless-patch"));
    let out_dir = PathBuf::from("target/endless");
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    let endl = out_dir.join(format!("{name}.endl"));
    std::fs::copy(&elf, out_dir.join(format!("{name}.elf"))).map_err(|e| e.to_string())?;
    run(Command::new(llvm_tool("llvm-objcopy")?).args(["-O", "binary"]).arg(&elf).arg(&endl))?;

    // Memory check: header + image + bss within the 512 KB patch region.
    let sizes = output(Command::new(llvm_tool("llvm-size")?).arg("-A").arg(&elf))?;
    let sec = |n: &str| -> u64 {
        sizes.lines().find(|l| l.starts_with(n)).and_then(|l| l.split_whitespace().nth(1)?.parse().ok()).unwrap_or(0)
    };
    let (hdr, img, bss) = (sec(".patch_header"), sec(".patch_image"), sec(".patch_bss"));
    let total = hdr + img + bss;
    println!("{}", output(Command::new("target/release/endless-host").args(["info", "--rig"]).arg(&rig)).unwrap_or_default().trim_end());
    println!(
        "built {} ({} bytes): image {} KB + state {} KB = {} KB of {} KB",
        endl.display(),
        std::fs::metadata(&endl).map_err(|e| e.to_string())?.len(),
        (hdr + img) / 1024,
        bss / 1024,
        total / 1024,
        PATCH_RAM / 1024
    );
    if total > PATCH_RAM {
        return Err("patch does not fit in 512 KB: use fewer amps, A2-Lite, or a shorter IR".into());
    }
    Ok(endl)
}

fn deploy(args: &[String]) -> Res<()> {
    let mut volume = PathBuf::from("/Volumes/ENDLESS");
    let mut rest = vec![];
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--volume" {
            volume = PathBuf::from(it.next().ok_or("--volume needs a path")?);
        } else {
            rest.push(a.clone());
        }
    }
    ensure_host_tool()?;
    let endl = build(&rest)?;
    if !volume.is_dir() {
        return Err(format!("Endless not mounted at {} (use --volume DIR)", volume.display()));
    }
    let dest = volume.join(endl.file_name().unwrap());
    std::fs::copy(&endl, &dest).map_err(|e| format!("copy to {}: {e}", dest.display()))?;
    println!("copied to {}", dest.display());
    Ok(())
}

fn ensure_host_tool() -> Res<()> {
    run(Command::new("cargo").args(["build", "-q", "--release", "-p", "host"]))
}

fn host_tool(args: &[String]) -> Res<()> {
    ensure_host_tool()?;
    run(Command::new("target/release/endless-host").args(args))
}

fn render(args: &[String]) -> Res<()> {
    let (rig, rest) = take_rig(args);
    let mut files: Vec<String> = vec![];
    let mut opts: Vec<String> = vec![];
    let mut it = rest.into_iter();
    while let Some(a) = it.next() {
        if a.starts_with("--") {
            let takes_value = a != "--no-cab";
            opts.push(a);
            if takes_value {
                opts.extend(it.next());
            }
        } else {
            files.push(a);
        }
    }
    let input = files.first().ok_or("render needs IN.wav")?.clone();
    let name = rig.file_stem().unwrap().to_string_lossy().into_owned();
    let out = files.get(1).cloned().unwrap_or(format!("target/render/{name}.wav"));
    if let Some(dir) = Path::new(&out).parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let mut v = vec!["render".to_string(), "--rig".into(), rig.display().to_string(), input, out.clone()];
    v.extend(opts);
    host_tool(&v)?;
    println!("wrote {out}");
    Ok(())
}

// ---------------------------------------------------------------- test

const NAMCORE: &str = "third_party/NeuralAmpModelerCore";

fn nam_reference() -> Res<PathBuf> {
    let exe = PathBuf::from("target/nam-ref/render");
    if exe.exists() {
        return Ok(exe);
    }
    std::fs::create_dir_all("target/nam-ref").map_err(|e| e.to_string())?;
    println!("building NeuralAmpModelerCore reference renderer (one-time, ~1 min)...");
    let srcs = |dir: &str| -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(format!("{NAMCORE}/{dir}"))
            .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "cpp")).collect())
            .unwrap_or_default();
        v.sort();
        v
    };
    let cxx = std::env::var("CXX").unwrap_or("clang++".into());
    run(Command::new(cxx)
        .args(["-std=c++2a", "-O2", "-DNAM_ENABLE_A2_FAST", "-DJSON_HAS_FILESYSTEM=0", "-DJSON_HAS_EXPERIMENTAL_FILESYSTEM=0"])
        .arg(format!("-I{NAMCORE}"))
        .arg(format!("-I{NAMCORE}/Dependencies/AudioDSPTools/dsp"))
        .arg(format!("-I{NAMCORE}/Dependencies/eigen"))
        .arg(format!("-I{NAMCORE}/Dependencies/nlohmann"))
        .args(srcs("NAM"))
        .args(srcs("NAM/wavenet"))
        .arg(format!("{NAMCORE}/Dependencies/AudioDSPTools/dsp/wav.cpp"))
        .arg(format!("{NAMCORE}/tools/render.cpp"))
        .arg("-o")
        .arg(&exe))?;
    Ok(exe)
}

fn find_models(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            find_models(&p, out);
        } else if p.extension().is_some_and(|e| e == "nam") {
            out.push(p);
        }
    }
}

fn test() -> Res<()> {
    run(Command::new("cargo").args(["test", "-q", "--workspace"]))?;
    ensure_host_tool()?;
    let host = "target/release/endless-host";
    let reference = nam_reference()?;
    let t = PathBuf::from("target/test");
    std::fs::create_dir_all(&t).map_err(|e| e.to_string())?;
    let input = t.join("in.wav");
    run(Command::new(host).arg("gen-input").arg(&input))?;

    let mut models = vec![];
    find_models(Path::new("models"), &mut models);
    if models.is_empty() {
        return Err("no .nam files in models/ to test against".into());
    }
    let mut failed = 0;
    for m in &models {
        for (size, slim) in [("lite", "0.4"), ("full", "1.0")] {
            let stem = m.file_stem().unwrap().to_string_lossy();
            let mine = t.join(format!("{stem}-{size}.wav"));
            let refw = t.join(format!("{stem}-{size}-ref.wav"));
            let r = Command::new(host)
                .args(["render", "--amp"])
                .arg(m)
                .args(["--size", size, "--no-normalize"])
                .arg(&input)
                .arg(&mine)
                .output()
                .map_err(|e| e.to_string())?;
            if !r.status.success() {
                println!("skip  {size:<5} {stem}: {}", String::from_utf8_lossy(&r.stderr).trim());
                continue;
            }
            output(Command::new(&reference).args(["--slim", slim]).arg(m).arg(&input).arg(&refw))?;
            print!("{size:<5} {stem:<44} ");
            flush();
            if run(Command::new(host).arg("compare").arg(&mine).arg(&refw)).is_err() {
                failed += 1;
            }
        }
    }

    // IR convolver: an IR of 0.5 at tap 3 must give the amp output delayed 3, halved.
    let m = &models[0];
    let ir = t.join("ir.wav");
    let mut taps = vec![0.0f32; 4];
    taps[3] = 0.5;
    write_wav(&ir, &taps)?;
    let rig = t.join("ir-rig.json");
    let json = format!(
        "{{\"name\": \"ir-test\", \"normalize\": false, \"amps\": [{{\"file\": {:?}}}], \
         \"cab\": {{\"file\": {:?}, \"normalize\": false}}}}",
        std::path::absolute(m).unwrap(),
        std::path::absolute(&ir).unwrap()
    );
    std::fs::write(&rig, json).map_err(|e| e.to_string())?;
    let mine = t.join("ir-mine.wav");
    output(Command::new(host).arg("render").arg("--rig").arg(&rig).arg(&input).arg(&mine))?;
    let amp_only = t.join(format!("{}-lite.wav", m.file_stem().unwrap().to_string_lossy()));
    let (x, _) = read_wav_f32(&amp_only)?;
    let mut expected = vec![0.0f32; 3];
    expected.extend(x[..x.len() - 3].iter().map(|v| v * 0.5));
    let exp = t.join("ir-expected.wav");
    write_wav(&exp, &expected)?;
    print!("ir    {:<44} ", "delta at tap 3, x0.5");
    flush();
    if run(Command::new(host).arg("compare").arg(&mine).arg(&exp)).is_err() {
        failed += 1;
    }

    if failed == 0 {
        println!("all tests passed");
        Ok(())
    } else {
        Err(format!("{failed} test(s) failed"))
    }
}

fn flush() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
}

fn write_wav(path: &Path, x: &[f32]) -> Res<()> {
    let bytes = (x.len() * 4) as u32;
    let mut b = Vec::new();
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + bytes).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    for v in [16u32.to_le_bytes(), [3, 0, 1, 0], 48000u32.to_le_bytes(), (48000u32 * 4).to_le_bytes(), [4, 0, 32, 0]] {
        b.extend_from_slice(&v);
    }
    b.extend_from_slice(b"data");
    b.extend_from_slice(&bytes.to_le_bytes());
    x.iter().for_each(|v| b.extend_from_slice(&v.to_le_bytes()));
    std::fs::write(path, b).map_err(|e| e.to_string())
}

/// Reads float WAVs written by this project (44-byte header).
fn read_wav_f32(path: &Path) -> Res<(Vec<f32>, u32)> {
    let d = std::fs::read(path).map_err(|e| e.to_string())?;
    let i = d.windows(4).position(|w| w == b"data").ok_or("no data chunk")?;
    let n = u32::from_le_bytes(d[i + 4..i + 8].try_into().unwrap()) as usize / 4;
    let x = (0..n).map(|k| f32::from_le_bytes(d[i + 8 + 4 * k..i + 12 + 4 * k].try_into().unwrap())).collect();
    Ok((x, 48000))
}
