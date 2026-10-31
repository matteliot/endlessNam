//! Bakes the rig named by $RIG (default rigs/soldano_rip.json) into the patch
//! and hands the linker the patch memory layout.

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.join("../..").canonicalize().unwrap();
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-env-changed=RIG");
    let rig_path = match std::env::var("RIG") {
        Ok(p) if !p.is_empty() => {
            let p = PathBuf::from(p);
            if p.is_absolute() { p } else { root.join(p) }
        }
        _ => root.join("rigs/soldano_rip.json"),
    };

    let rig = match rig_builder::load_rig(&rig_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("\nrig-builder: {e}\n");
            std::process::exit(1);
        }
    };
    for input in &rig.inputs {
        println!("cargo:rerun-if-changed={}", input.display());
    }
    let src = rig.to_rust_source().unwrap_or_else(|e| {
        eprintln!("\nrig-builder: {e}\n");
        std::process::exit(1);
    });
    std::fs::write(out.join("rig_data.rs"), src).unwrap();
    std::fs::write(out.join("rig_summary.txt"), rig.summary()).unwrap();

    if std::env::var("TARGET").unwrap().starts_with("thumb") {
        std::fs::copy(manifest.join("patch.ld"), out.join("patch.ld")).unwrap();
        println!("cargo:rustc-link-search={}", out.display());
        println!("cargo:rustc-link-arg=-Tpatch.ld");
        println!("cargo:rerun-if-changed=patch.ld");
    }
}
