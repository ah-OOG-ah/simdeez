//! A library that abstracts over SIMD instruction sets, including ones with differing widths.
//! SIMDeez is designed to allow you to write a function one time and produce scalar, SSE2, SSE41, AVX2 and Neon versions of the function.
//! You can either have the version you want selected automatically at runtime, at compiletime, or
//! select yourself by hand.
//!
//! SIMDeez is currently in Beta, if there are intrinsics you need that are not currently implemented, create an issue
//! and I'll add them. PRs to add more intrinsics are welcome. Currently things are well fleshed out for i32, i64, f32, and f64 types.
//!
//! As Rust stabilizes support for AVX-512 I plan to add those as well.
//!
//! Refer to the excellent [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#) for documentation on these functions.
//!
//! # Features
//!
//! * SSE2, SSE41, AVX2, Neon and scalar fallback
//! * Can be used with compile time or run time selection
//! * No runtime overhead
//! * Uses familiar intel intrinsic naming conventions, easy to port.
//!   * `_mm_add_ps(a,b)` becomes `add_ps(a,b)`
//! * Fills in missing intrinsics in older APIs with fast SIMD workarounds.
//!   * ceil, floor, round,blend, etc
//! * Can be used by `#[no_std]` projects
//! * Operator overloading: `let sum = va + vb` or `s *= s`
//! * Extract or set a single lane with the index operator: `let v1 = v[1];`
//!
//! # Trig Functions via Sleef-sys
//! A number of trigonometric and other common math functions are provided
//! in vectorized form via the Sleef-sys crate. This is an optional feature `sleef` that you can enable.
//! Doing so currently requires nightly, as well as having CMake and Clang installed.
//!
//! # Compared to stdsimd
//!
//! * SIMDeez can abstract over differing simd widths. stdsimd does not
//! * SIMDeez builds on stable rust now, stdsimd does not
//!
//! # Compared to Faster
//!
//! * SIMDeez can be used with runtime selection, Faster cannot.
//! * SIMDeez has faster fallbacks for some functions
//! * SIMDeez does not currently work with iterators, Faster does.
//! * SIMDeez uses more idiomatic intrinsic syntax while Faster uses more idomatic Rust syntax
//! * SIMDeez can be used by `#[no_std]` projects
//! * SIMDeez builds on stable rust now, Faster does not.
//!
//! All of the above could change! Faster seems to generally have the same
//! performance as long as you don't run into some of the slower fallback functions.
//!
//!
//! # Example
//!
//! ```rust
//!use simdeez::{prelude::*, simd_runtime_generate};
//!
//! // If you want your SIMD function to use use runtime feature detection to call
//!// the fastest available version, use the simd_runtime_generate macro:
//!simd_runtime_generate!(
//!    fn distance(x1: &[f32], y1: &[f32], x2: &[f32], y2: &[f32]) -> Vec<f32> {
//!        let mut result: Vec<f32> = Vec::with_capacity(x1.len());
//!        result.set_len(x1.len()); // for efficiency
//!
//!        // Set each slice to the same length for iteration efficiency
//!        let mut x1 = &x1[..x1.len()];
//!        let mut y1 = &y1[..x1.len()];
//!        let mut x2 = &x2[..x1.len()];
//!        let mut y2 = &y2[..x1.len()];
//!        let mut res = &mut result[..x1.len()];
//!
//!        // Operations have to be done in terms of the vector width
//!        // so that it will work with any size vector.
//!        // the width of a vector type is provided as a constant
//!        // so the compiler is free to optimize it more.
//!        // Vf32::WIDTH is a constant, 4 when using SSE, 8 when using AVX2, etc
//!        while x1.len() >= S::Vf32::WIDTH {
//!            //load data from your vec into an SIMD value
//!            let xv1 = S::Vf32::load_from_slice(&x1);
//!            let yv1 = S::Vf32::load_from_slice(&y1);
//!            let xv2 = S::Vf32::load_from_slice(&x2);
//!            let yv2 = S::Vf32::load_from_slice(&y2);
//!
//!            // Use the usual intrinsic syntax if you prefer
//!            let mut xdiff = xv1 - xv2;
//!            // Or use operater overloading if you like
//!            let mut ydiff = yv1 - yv2;
//!            xdiff *= xdiff;
//!            ydiff *= ydiff;
//!            let distance = (xdiff + ydiff).sqrt();
//!            // Store the SIMD value into the result vec
//!            distance.copy_to_slice(res);
//!
//!            // Move each slice to the next position
//!            x1 = &x1[S::Vf32::WIDTH..];
//!            y1 = &y1[S::Vf32::WIDTH..];
//!            x2 = &x2[S::Vf32::WIDTH..];
//!            y2 = &y2[S::Vf32::WIDTH..];
//!            res = &mut res[S::Vf32::WIDTH..];
//!        }
//!
//!        // (Optional) Compute the remaining elements. Not necessary if you are sure the length
//!        // of your data is always a multiple of the maximum S::Vf32_WIDTH you compile for (4 for SSE, 8 for AVX2, etc).
//!        // This can be asserted by putting `assert_eq!(x1.len(), 0);` here
//!        for i in 0..x1.len() {
//!            let mut xdiff = x1[i] - x2[i];
//!            let mut ydiff = y1[i] - y2[i];
//!            xdiff *= xdiff;
//!            ydiff *= ydiff;
//!            let distance = (xdiff + ydiff).sqrt();
//!            res[i] = distance;
//!        }
//!
//!        result
//!    }
//!);
//!
//!const SIZE: usize = 200;
//!
//!fn main() {
//!    let raw = (0..4)
//!        .map(|i| (0..SIZE).map(|j| (i*j) as f32).collect::<Vec<f32>>())
//!        .collect::<Vec<Vec<f32>>>();
//!
//!    let distances = distance(
//!        raw[0].as_slice(),
//!        raw[1].as_slice(),
//!        raw[2].as_slice(),
//!        raw[3].as_slice(),
//!    );
//!    assert_eq!(distances.len(), SIZE);
//!    dbg!(distances);
//!}
//! ```
//!
//! This will generate 5 functions for you:
//! * `distance<S:Simd>` the generic version of your function
//! * `distance_scalar`  a scalar fallback
//! * `distance_sse2`    SSE2 version
//! * `distance_sse41`   SSE41 version
//! * `distance_avx2`    AVX2 version
//! * `distance_neon`    Neon version
//! * `distance_runtime_select`  picks the fastest of the above at runtime
//!
//! You can use any of these you wish, though typically you would use the runtime_select version
//! unless you want to force an older instruction set to avoid throttling or for other arcane
//! reasons.
//!
//! Optionally you can use the `simd_compiletime_generate!` macro in the same way.  This will
//! produce 2 active functions via the `cfg` attribute feature:
//!
//! * `distance<S:Simd>`      the generic version of your function
//! * `distance_compiletime`  the fastest instruction set availble for the given compile time
//!   feature set
//!
//! You may also forgo the macros if you know what you are doing, just keep in mind there are lots
//! of arcane subtleties with inlining and target_features that must be managed. See how the macros
//! expand for more detail.
#![allow(clippy::missing_safety_doc)] // TODO: Work on the safety of functions
#![cfg_attr(all(feature = "no_std", not(test)), no_std)]
#[macro_use]
#[cfg(test)]
extern crate std;
pub extern crate paste;

#[cfg(test)]
mod tests;

mod ops;

pub mod prelude;

use core::ops::*;

mod invoking;

#[macro_use]
mod overloads;

mod base;
pub use base::*;

mod libm_ext;

pub mod engines;

pub use engines::scalar;

/// The abstract SIMD trait which is implemented by Avx2, Sse41, etc
pub trait Simd: 'static + Sync + Send {
    /// Vector of i8s.  Corresponds to __m128i when used
    /// with the Sse impl, __m256i when used with Avx2, or a single i8
    /// when used with Scalar.
    type Vi8: SimdInt8<Scalar = i8> + SimdBaseIo;

    /// Vector of i16s.  Corresponds to __m128i when used
    /// with the Sse impl, __m256i when used with Avx2, or a single i16
    /// when used with Scalar.
    type Vi16: SimdInt16<Scalar = i16> + SimdBaseIo;

    /// Vector of i32s.  Corresponds to __m128i when used
    /// with the Sse impl, __m256i when used with Avx2, or a single i32
    /// when used with Scalar.
    type Vi32: SimdInt32<Engine = Self, Scalar = i32> + SimdBaseIo;

    /// Vector of i64s.  Corresponds to __m128i when used
    /// with the Sse impl, __m256i when used with Avx2, or a single i64
    /// when used with Scalar.
    type Vi64: SimdInt64<Engine = Self, Scalar = i64> + SimdBaseIo;

    /// Vector of f32s.  Corresponds to __m128 when used
    /// with the Sse impl, __m256 when used with Avx2, or a single f32
    /// when used with Scalar.
    type Vf32: SimdFloat32<Engine = Self, Scalar = f32> + SimdBaseIo;

    /// Vector of f64s.  Corresponds to __m128d when used
    /// with the Sse impl, __m256d when used with Avx2, or a single f64
    /// when used with Scalar.
    type Vf64: SimdFloat64<Engine = Self, Scalar = f64> + SimdBaseIo;

    fn invoke<R>(f: impl FnOnce() -> R) -> R;

    cfg_if::cfg_if! {
        if #[cfg(feature = "sleef")] {
            unsafe fn sin_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn fast_sin_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn cos_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn fast_cos_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn asin_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn fast_asin_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn acos_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn fast_acos_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn tan_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn fast_tan_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn atan_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn fast_atan_ps(a: Self::Vf32) -> Self::Vf32;

            //hyperbolic
            unsafe fn sinh_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn fast_sinh_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn cosh_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn fast_cosh_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn asinh_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn acosh_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn tanh_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn fast_tanh_ps(a: Self::Vf32) -> Self::Vf32;
            unsafe fn atanh_ps(a: Self::Vf32) -> Self::Vf32;

            unsafe fn atan2_ps(a: Self::Vf32,b: Self::Vf32) -> Self::Vf32;
            unsafe fn fast_atan2_ps(a: Self::Vf32,b: Self::Vf32) -> Self::Vf32;
            unsafe fn ln_ps(a:Self::Vf32) -> Self::Vf32;
            unsafe fn fast_ln_ps(a:Self::Vf32) -> Self::Vf32;
            unsafe fn log2_ps(a:Self::Vf32) -> Self::Vf32;
            unsafe fn log10_ps(a:Self::Vf32) -> Self::Vf32;
            unsafe fn hypot_ps(a:Self::Vf32,b:Self::Vf32) -> Self::Vf32;
            unsafe fn fast_hypot_ps(a:Self::Vf32,b:Self::Vf32) -> Self::Vf32;

            unsafe fn fmod_ps(a:Self::Vf32,b:Self::Vf32) -> Self::Vf32;
        }
    }
}
