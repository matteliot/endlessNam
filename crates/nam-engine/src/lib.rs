//! Allocation-free amp/cab processing for the Polyend Endless.
//!
//! - [`A2`]: NAM "A2" WaveNet inference (A2-Lite = 3 channels, A2-Full = 8)
//! - [`Fir`]: impulse-response convolver for `.wav` cabs
//! - [`Rig`]: the signal chain (gain → amp → cab → tone → level)
//!
//! Everything is `no_std`, does no allocation, and every type has a `const`
//! zero value so it can live in a `static` (and lands in `.bss`).

#![no_std]

mod a2;
mod fir;
mod rig;

pub use a2::{A2, MAX_BLOCK, weight_count};
pub use fir::Fir;
pub use rig::{AmpSlot, CabStage, NoCab, Param, Rig, RigData};
