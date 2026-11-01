//! The Endless patch ABI: a Rust mirror of the SDK's
//! `third_party/FxPatchSDK/internal/PatchABI.h` (ABI version 0x000B).
//!
//! The firmware loads the raw image at 0x8000_0000 and reads `PatchHeader`
//! from its first bytes. Field order and sizes must match the C struct
//! exactly (32-bit target: pointers, `uintptr_t` and `uint32_t` are 4 bytes).

use core::ffi::c_void;

pub const PATCH_MAGIC: u32 = 0x4843_5450; // 'PTCH'
pub const PATCH_ABI_VERSION: u16 = 0x000B;
pub const PATCH_FLAG_NONE: u16 = 0;

#[repr(C)]
pub struct PatchEnv {
    pub abi_version: u32,
    pub user_ctx: *mut c_void,
}

pub type InitFn = unsafe extern "C" fn(env: *const PatchEnv);
pub type UpdateBuffersFn = unsafe extern "C" fn(env: *const PatchEnv, left: *mut f32, right: *mut f32, samples: i32) -> f32;
pub type SetBufferFn = unsafe extern "C" fn(env: *const PatchEnv, buffer: *mut f32);
pub type GetBufferSizeFn = unsafe extern "C" fn(env: *const PatchEnv) -> i32;
pub type GetParamScalarFn = unsafe extern "C" fn(env: *const PatchEnv, idx: i32) -> f32;
pub type IsParamEnabledFn = unsafe extern "C" fn(env: *const PatchEnv, idx: i32, source_id: i32) -> u8;
pub type GetParamStringFn = unsafe extern "C" fn(env: *const PatchEnv, idx: i32, out: *mut u8, out_len: usize);
pub type SetParamFn = unsafe extern "C" fn(env: *const PatchEnv, idx: i32, value: f32);
pub type SpecialActionFn = unsafe extern "C" fn(env: *const PatchEnv, idx: i32);
pub type GetStateIdxFn = unsafe extern "C" fn(env: *const PatchEnv) -> i32;

#[repr(C)]
pub struct PatchHeader {
    pub magic: u32,
    pub abi_version: u16,
    pub flags: u16,
    pub init: InitFn,
    pub agent_update_buffers: UpdateBuffersFn,
    pub agent_set_buffer: SetBufferFn,
    pub agent_get_buffer_size: GetBufferSizeFn,
    pub agent_get_param_min: GetParamScalarFn,
    pub agent_get_param_max: GetParamScalarFn,
    pub agent_get_param_default: GetParamScalarFn,
    pub agent_is_param_enabled: IsParamEnabledFn,
    pub agent_get_param_name: GetParamStringFn,
    pub agent_get_param_unit: GetParamStringFn,
    pub agent_set_param: SetParamFn,
    pub agent_special_action: SpecialActionFn,
    pub agent_get_state_idx: GetStateIdxFn,
    /// `uint32_t image_size`: address of the linker's `patch_image_size` symbol.
    pub image_size: *const u8,
    /// `uintptr_t bss_begin`
    pub bss_begin: *const u8,
    /// `uint32_t bss_size`: address of the linker's `patch_bss_size` symbol.
    pub bss_size: *const u8,
    pub reserved: [u32; 16],
}

// The header is immutable data read by the firmware.
unsafe impl Sync for PatchHeader {}

#[cfg(target_pointer_width = "32")]
const _: () = assert!(core::mem::size_of::<PatchHeader>() == 4 + 2 + 2 + 13 * 4 + 3 * 4 + 16 * 4);

/// Hardware IDs (SDK `source/Patch.h`).
pub const PARAM_COUNT: i32 = 3;
pub const ACTION_FOOTSWITCH_PRESS: i32 = 0;
pub const ACTION_FOOTSWITCH_HOLD: i32 = 1;
/// The SDK reports this working-buffer size (floats) to the firmware.
pub const WORKING_BUFFER_SIZE: i32 = 2_400_000;

/// LED colours, in the SDK's `Patch::Color` order.
#[allow(dead_code)]
#[derive(Clone, Copy)]
#[repr(i32)]
pub enum Color {
    DimWhite,
    DarkRed,
    DarkLime,
    DarkCobalt,
    LightYellow,
    DimBlue,
    Beige,
    DimCyan,
    Magenta,
    LightBlue,
    PastelGreen,
    DimYellow,
    Blue,
    LightGreen,
    Red,
    DimGreen,
}
