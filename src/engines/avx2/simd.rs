#![allow(deprecated)]

use super::*;
use crate::Simd;

pub struct Avx2;
impl Simd for Avx2 {
    type Vi8 = I8x32;
    type Vi16 = I16x16;
    type Vi32 = I32x8;
    type Vf32 = F32x8;
    type Vf64 = F64x4;
    type Vi64 = I64x4;

    #[inline]
    fn invoke<R>(f: impl FnOnce() -> R) -> R {
        #[inline]
        #[target_feature(enable = "avx2", enable = "fma")]
        unsafe fn inner<R>(f: impl FnOnce() -> R) -> R {
            f()
        }

        unsafe { inner(f) }
    }
}
