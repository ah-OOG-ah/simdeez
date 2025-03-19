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
//! * AVX2 and scalar fallback
//! * No runtime overhead
//! * Uses familiar intel intrinsic naming conventions, easy to port.
//!   * `_mm_add_ps(a,b)` becomes `add_ps(a,b)`
//! * Fills in missing intrinsics in older APIs with fast SIMD workarounds.
//!   * ceil, floor, round,blend, etc
//! * Can be used by `#[no_std]` projects
//! * Operator overloading: `let sum = va + vb` or `s *= s`
//! * Extract or set a single lane with the index operator: `let v1 = v[1];`
//!
//! # Compared to Faster
//!
//! * SIMDeez has faster fallbacks for some functions
//! * SIMDeez does not currently work with iterators, Faster does.
//! * SIMDeez uses more idiomatic intrinsic syntax while Faster uses more idomatic Rust syntax
//! * SIMDeez can be used by `#[no_std]` projects
//!
//! All of the above could change! Faster seems to generally have the same
//! performance as long as you don't run into some of the slower fallback functions.
//!
//! You may also forgo the macros if you know what you are doing, just keep in mind there are lots
//! of arcane subtleties with inlining and target_features that must be managed. See how the macros
//! expand for more detail.
#![allow(clippy::missing_safety_doc, unsafe_op_in_unsafe_fn)] // TODO: Work on the safety of functions
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
}
