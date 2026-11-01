//! Endless patch: NAM amp + cab rig.
//!
//! Controls
//!   Left knob   Gain   (input drive, -18..+18 dB, noon = 0 dB)
//!   Mid knob    Tone   (tilt EQ around 700 Hz, ±6 dB, noon = flat)
//!   Right knob  Level  (output, -36..+12 dB, 3 o'clock = 0 dB)
//!   Footswitch press  next amp capture in the rig
//!   Footswitch hold   cab on/off
//!   LED: colour = amp capture; bright = cab on, dim = cab off; red = CPU overload
//!
//! The models are compiled in from the rig file ($RIG, see build.rs).

#![cfg_attr(target_os = "none", no_std)]
#![cfg_attr(target_os = "none", no_main)]

mod abi;

#[allow(dead_code)]
mod rig_data {
    include!(concat!(env!("OUT_DIR"), "/rig_data.rs"));
}

use core::cell::UnsafeCell;

use abi::*;
use nam_engine::Param;

const CPU_HZ: u32 = 720_000_000;
const SAMPLE_RATE: u32 = 48_000;
const CYCLES_PER_SAMPLE: u32 = CPU_HZ / SAMPLE_RATE;

/// (cab on, cab off) LED colour per amp slot.
const AMP_COLORS: [(Color, Color); 7] = [
    (Color::Blue, Color::DimBlue),
    (Color::LightGreen, Color::DimGreen),
    (Color::LightYellow, Color::DimYellow),
    (Color::LightBlue, Color::DimCyan),
    (Color::Magenta, Color::DarkCobalt),
    (Color::Beige, Color::DimWhite),
    (Color::PastelGreen, Color::DarkLime),
];

struct State {
    rig: rig_data::Rig,
    overload_hold: i32,
    error: bool,
}

/// All patch state. The firmware calls every entry point from the audio
/// thread (SDK: "no need to synchronize"), so plain interior mutability is
/// enough. Zero-initialised, so it lives in .bss.
struct Global(UnsafeCell<State>);
unsafe impl Sync for Global {}

static STATE: Global = Global(UnsafeCell::new(State { rig: rig_data::Rig::NEW, overload_hold: 0, error: false }));

#[allow(clippy::mut_from_ref)]
fn state() -> &'static mut State {
    // SAFETY: single-threaded access from the firmware's audio thread.
    unsafe { &mut *STATE.0.get() }
}

/// Cortex-M7 DWT cycle counter. Read-only: if the firmware hasn't enabled it
/// the value never changes and the overload meter never trips.
#[inline(always)]
fn cycles() -> u32 {
    #[cfg(target_os = "none")]
    {
        unsafe { core::ptr::read_volatile(0xE000_1004 as *const u32) }
    }
    #[cfg(not(target_os = "none"))]
    {
        0
    }
}

fn param(idx: i32) -> Option<Param> {
    usize::try_from(idx).ok().and_then(Param::from_index)
}

// ---------------------------------------------------------------- entry points

unsafe extern "C" fn patch_init(env: *const PatchEnv) {
    if env.is_null() || unsafe { (*env).abi_version } != PATCH_ABI_VERSION as u32 {
        return;
    }
    state().rig.init(&rig_data::RIG_DATA);
}

unsafe extern "C" fn update_buffers(_: *const PatchEnv, left: *mut f32, right: *mut f32, samples: i32) -> f32 {
    let Ok(n) = usize::try_from(samples) else { return 0.0 };
    if left.is_null() || right.is_null() || n == 0 {
        return 0.0;
    }
    // SAFETY: the firmware passes two buffers of `samples` floats.
    let (l, r) = unsafe { (core::slice::from_raw_parts_mut(left, n), core::slice::from_raw_parts_mut(right, n)) };
    let s = state();
    let t0 = cycles();
    s.rig.process(l); // mono: left input
    r.copy_from_slice(l);
    let used = cycles().wrapping_sub(t0);
    // Trip at 90% of the block's cycle budget; hold the red LED ~1 s.
    if used != 0 && used as u64 * 10 > CYCLES_PER_SAMPLE as u64 * 9 * n as u64 {
        s.overload_hold = SAMPLE_RATE as i32;
    } else if s.overload_hold > 0 {
        s.overload_hold -= samples;
    }
    0.0
}

unsafe extern "C" fn set_buffer(_: *const PatchEnv, buffer: *mut f32) {
    // The rig keeps its state in .bss and doesn't need the working buffer.
    if buffer.is_null() {
        state().error = true;
    }
}

unsafe extern "C" fn get_buffer_size(_: *const PatchEnv) -> i32 {
    WORKING_BUFFER_SIZE
}

unsafe extern "C" fn get_param_min(_: *const PatchEnv, _: i32) -> f32 {
    0.0
}

unsafe extern "C" fn get_param_max(_: *const PatchEnv, _: i32) -> f32 {
    1.0
}

unsafe extern "C" fn get_param_default(_: *const PatchEnv, idx: i32) -> f32 {
    param(idx).map_or(0.5, Param::default_value)
}

unsafe extern "C" fn is_param_enabled(_: *const PatchEnv, idx: i32, source_id: i32) -> u8 {
    // Knobs only (source 0), like the SDK wrapper.
    (source_id == 0 && (0..PARAM_COUNT).contains(&idx)) as u8
}

unsafe fn write_c_str(s: &str, out: *mut u8, len: usize) {
    if out.is_null() || len == 0 {
        return;
    }
    let n = s.len().min(len - 1);
    // SAFETY: the firmware provides `len` writable bytes at `out`.
    unsafe {
        core::ptr::copy_nonoverlapping(s.as_ptr(), out, n);
        *out.add(n) = 0;
    }
}

unsafe extern "C" fn get_param_name(_: *const PatchEnv, idx: i32, out: *mut u8, len: usize) {
    let name = match param(idx) {
        Some(Param::Gain) => "Gain",
        Some(Param::Tone) => "Tone",
        Some(Param::Level) => "Level",
        None => "",
    };
    unsafe { write_c_str(name, out, len) }
}

unsafe extern "C" fn get_param_unit(_: *const PatchEnv, _: i32, out: *mut u8, len: usize) {
    unsafe { write_c_str("", out, len) }
}

unsafe extern "C" fn set_param(_: *const PatchEnv, idx: i32, value: f32) {
    if let Some(p) = param(idx) {
        state().rig.set_param(p, value);
    }
}

unsafe extern "C" fn special_action(_: *const PatchEnv, idx: i32) {
    let rig = &mut state().rig;
    match idx {
        ACTION_FOOTSWITCH_PRESS => rig.next_amp(),
        ACTION_FOOTSWITCH_HOLD => rig.toggle_cab(),
        _ => {}
    }
}

unsafe extern "C" fn get_state_idx(_: *const PatchEnv) -> i32 {
    let s = state();
    let color = if s.error || s.overload_hold > 0 {
        Color::Red
    } else {
        let (on, off) = AMP_COLORS[s.rig.amp_index() % AMP_COLORS.len()];
        if s.rig.cab_on() { on } else { off }
    };
    color as i32
}

// ---------------------------------------------------------------- header

#[cfg(target_os = "none")]
unsafe extern "C" {
    static __patch_bss_start: u8;
    static patch_image_size: u8;
    static patch_bss_size: u8;
}

#[cfg(target_os = "none")]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".patch_header")]
#[used]
pub static patch_header: PatchHeader = PatchHeader {
    magic: PATCH_MAGIC,
    abi_version: PATCH_ABI_VERSION,
    flags: PATCH_FLAG_NONE,
    init: patch_init,
    agent_update_buffers: update_buffers,
    agent_set_buffer: set_buffer,
    agent_get_buffer_size: get_buffer_size,
    agent_get_param_min: get_param_min,
    agent_get_param_max: get_param_max,
    agent_get_param_default: get_param_default,
    agent_is_param_enabled: is_param_enabled,
    agent_get_param_name: get_param_name,
    agent_get_param_unit: get_param_unit,
    agent_set_param: set_param,
    agent_special_action: special_action,
    agent_get_state_idx: get_state_idx,
    image_size: &raw const patch_image_size,
    bss_begin: &raw const __patch_bss_start,
    bss_size: &raw const patch_bss_size,
    reserved: [0; 16],
};

#[cfg(target_os = "none")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    // No way to report from here; the audio thread stops. The code paths are
    // written to avoid panics (see `cargo xtask build` panic check).
    loop {
        core::hint::spin_loop();
    }
}

/// Host builds (e.g. `cargo check` without a target) get a stub main.
#[cfg(not(target_os = "none"))]
fn main() {
    let env = PatchEnv { abi_version: PATCH_ABI_VERSION as u32, user_ctx: core::ptr::null_mut() };
    unsafe { patch_init(&env) };
    let mut l = [0.0f32; 64];
    let mut r = [0.0f32; 64];
    unsafe { update_buffers(&env, l.as_mut_ptr(), r.as_mut_ptr(), 64) };
    let _ = (get_buffer_size, get_param_min, get_param_max, get_param_default, is_param_enabled);
    let _ = (get_param_name, get_param_unit, set_param, special_action, get_state_idx, set_buffer);
    println!("{}", include_str!(concat!(env!("OUT_DIR"), "/rig_summary.txt")));
}
