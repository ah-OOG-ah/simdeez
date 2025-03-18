use super::*;
use crate::Simd;

pub struct Scalar;
impl Simd for Scalar {
    type Vi8 = I8x1;
    type Vi16 = I16x1;
    type Vi32 = I32x1;
    type Vf32 = F32x1;
    type Vf64 = F64x1;
    type Vi64 = I64x1;

    #[inline]
    fn invoke<R>(f: impl FnOnce() -> R) -> R {
        #[inline]
        unsafe fn inner<R>(f: impl FnOnce() -> R) -> R {
            f()
        }

        unsafe { inner(f) }
    }
}
