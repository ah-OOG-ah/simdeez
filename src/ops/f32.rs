use super::*;

impl_op! {
    fn add<f32> {
        for Avx2(a: __m256, b: __m256) -> __m256 {
            _mm256_add_ps(a, b)
        }
        for Scalar(a: f32, b: f32) -> f32 {
            a + b
        }
    }
}

impl_op! {
    fn sub<f32> {
        for Avx2(a: __m256, b: __m256) -> __m256 {
            _mm256_sub_ps(a, b)
        }
        for Scalar(a: f32, b: f32) -> f32 {
            a - b
        }
    }
}

impl_op! {
    fn mul<f32> {
        for Avx2(a: __m256, b: __m256) -> __m256 {
            _mm256_mul_ps(a, b)
        }
        for Scalar(a: f32, b: f32) -> f32 {
            a * b
        }
    }
}

impl_op! {
    fn div<f32> {
        for Avx2(a: __m256, b: __m256) -> __m256 {
            _mm256_div_ps(a, b)
        }
        for Scalar(a: f32, b: f32) -> f32 {
            a / b
        }
    }
}

impl_op! {
    fn mul_add<f32> {
        for Avx2(a: __m256, b: __m256, c: __m256) -> __m256 {
            _mm256_fmadd_ps(a, b, c)
        }
        for Scalar(a: f32, b: f32, c: f32) -> f32 {
            a * b + c
        }
    }
}

impl_op! {
    fn mul_sub<f32> {
        for Avx2(a: __m256, b: __m256, c: __m256) -> __m256 {
            _mm256_fmsub_ps(a, b, c)
        }
        for Scalar(a: f32, b: f32, c: f32) -> f32 {
            a * b - c
        }
    }
}

impl_op! {
    fn neg_mul_add<f32> {
        for Avx2(a: __m256, b: __m256, c: __m256) -> __m256 {
            _mm256_fnmadd_ps(a, b, c)
        }
        for Scalar(a: f32, b: f32, c: f32) -> f32 {
            c - a * b
        }
    }
}

impl_op! {
    fn neg_mul_sub<f32> {
        for Avx2(a: __m256, b: __m256, c: __m256) -> __m256 {
            _mm256_fnmsub_ps(a, b, c)
        }
        for Scalar(a: f32, b: f32, c: f32) -> f32 {
            -a * b - c
        }
    }
}

impl_op! {
    fn sqrt<f32> {
        for Avx2(a: __m256) -> __m256 {
            _mm256_sqrt_ps(a)
        }
        for Scalar(a: f32) -> f32 {
            a.m_sqrt()
        }
    }
}

impl_op! {
    fn recip<f32> {
        for Avx2(a: __m256) -> __m256 {
            _mm256_rcp_ps(a)
        }
        for Scalar(a: f32) -> f32 {
            1.0 / a
        }
    }
}

impl_op! {
    fn rsqrt<f32> {
        for Avx2(a: __m256) -> __m256 {
            _mm256_rsqrt_ps(a)
        }
        for Scalar(a: f32) -> f32 {
            1.0 / a.m_sqrt()
        }
    }
}

impl_op! {
    fn min<f32> {
        for Avx2(a: __m256, b: __m256) -> __m256 {
            _mm256_min_ps(a, b)
        }
        for Scalar(a: f32, b: f32) -> f32 {
            a.min(b)
        }
    }
}

impl_op! {
    fn max<f32> {
        for Avx2(a: __m256, b: __m256) -> __m256 {
            _mm256_max_ps(a, b)
        }
        for Scalar(a: f32, b: f32) -> f32 {
            a.max(b)
        }
    }
}

impl_op! {
    fn abs<f32> {
        for Avx2(a: __m256) -> __m256 {
            _mm256_andnot_ps(_mm256_set1_ps(-0.0), a)
        }
        for Scalar(a: f32) -> f32 {
            a.m_abs()
        }
    }
}

impl_op! {
    fn round<f32> {
        for Avx2(a: __m256) -> __m256 {
            _mm256_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
        }
        for Scalar(a: f32) -> f32 {
            a.m_round()
        }
    }
}

impl_op! {
    fn floor<f32> {
        for Avx2(a: __m256) -> __m256 {
            _mm256_round_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)
        }
        for Scalar(a: f32) -> f32 {
            a.m_floor()
        }
    }
}

impl_op! {
    fn ceil<f32> {
        for Avx2(a: __m256) -> __m256 {
            _mm256_round_ps(a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)
        }
        for Scalar(a: f32) -> f32 {
            a.m_ceil()
        }
    }
}

impl_op! {
    fn fast_round<f32> {
        for Avx2(a: __m256) -> __m256 {
            Self::round(a)
        }
        for Scalar(a: f32) -> f32 {
            Self::round(a)
        }
    }
}

impl_op! {
    fn fast_floor<f32> {
        for Avx2(a: __m256) -> __m256 {
            Self::floor(a)
        }
        for Scalar(a: f32) -> f32 {
            Self::floor(a)
        }
    }
}

impl_op! {
    fn fast_ceil<f32> {
        for Avx2(a: __m256) -> __m256 {
            Self::ceil(a)
        }
        for Scalar(a: f32) -> f32 {
            Self::ceil(a)
        }
    }
}

impl_op! {
    fn eq<f32> {
        for Avx2(a: __m256, b: __m256) -> __m256 {
            _mm256_cmp_ps(a, b, _CMP_EQ_OQ)
        }
        for Scalar(a: f32, b: f32) -> f32 {
            if a == b {
                f32::from_bits(u32::MAX)
            } else {
                0.0
            }
        }
    }
}

impl_op! {
    fn neq<f32> {
        for Avx2(a: __m256, b: __m256) -> __m256 {
            _mm256_cmp_ps(a, b, _CMP_NEQ_OQ)
        }
        for Scalar(a: f32, b: f32) -> f32 {
            if a != b {
                f32::from_bits(u32::MAX)
            } else {
                0.0
            }
        }
    }
}

impl_op! {
    fn lt<f32> {
        for Avx2(a: __m256, b: __m256) -> __m256 {
            _mm256_cmp_ps(a, b, _CMP_LT_OQ)
        }
        for Scalar(a: f32, b: f32) -> f32 {
            if a < b {
                f32::from_bits(u32::MAX)
            } else {
                0.0
            }
        }
    }
}

impl_op! {
    fn lte<f32> {
        for Avx2(a: __m256, b: __m256) -> __m256 {
            _mm256_cmp_ps(a, b, _CMP_LE_OQ)
        }
        for Scalar(a: f32, b: f32) -> f32 {
            if a <= b {
                f32::from_bits(u32::MAX)
            } else {
                0.0
            }
        }
    }
}

impl_op! {
    fn gt<f32> {
        for Avx2(a: __m256, b: __m256) -> __m256 {
            _mm256_cmp_ps(a, b, _CMP_GT_OQ)
        }
        for Scalar(a: f32, b: f32) -> f32 {
            if a > b {
                f32::from_bits(u32::MAX)
            } else {
                0.0
            }
        }
    }
}

impl_op! {
    fn gte<f32> {
        for Avx2(a: __m256, b: __m256) -> __m256 {
            _mm256_cmp_ps(a, b, _CMP_GE_OQ)
        }
        for Scalar(a: f32, b: f32) -> f32 {
            if a >= b {
                f32::from_bits(u32::MAX)
            } else {
                0.0
            }
        }
    }
}

impl_op! {
    fn blendv<f32> {
        for Avx2(a: __m256, b: __m256, mask: __m256) -> __m256 {
            _mm256_blendv_ps(a, b, mask)
        }
        for Scalar(a: f32, b: f32, mask: f32) -> f32 {
            if mask.to_bits() == 0 {
                a
            } else {
                b
            }
        }
    }
}

impl_op! {
    fn horizontal_add<f32> {
        for Avx2(a: __m256) -> f32 {
            let a = _mm256_hadd_ps(a, a);
            let b = _mm256_hadd_ps(a, a);

            let first = _mm_cvtss_f32(_mm256_extractf128_ps(b, 0));
            let second = _mm_cvtss_f32(_mm256_extractf128_ps(b, 1));

            first + second
        }
        for Scalar(a: f32) -> f32 {
            a
        }
    }
}

impl_op! {
    fn cast_i32<f32> {
        for Avx2(a: __m256) -> __m256i {
            _mm256_cvtps_epi32(a)
        }
        for Scalar(a: f32) -> i32 {
            a.m_round() as i32
        }
    }
}

impl_op! {
    fn bitcast_i32<f32> {
        for Avx2(a: __m256) -> __m256i {
            _mm256_castps_si256(a)
        }
        for Scalar(a: f32) -> i32 {
            a.to_bits() as i32
        }
    }
}

impl_op! {
    fn zeroes<f32> {
        for Avx2() -> __m256 {
            _mm256_setzero_ps()
        }
        for Scalar() -> f32 {
            0.0
        }
    }
}

impl_op! {
    fn set1<f32> {
        for Avx2(val: f32) -> __m256 {
            _mm256_set1_ps(val)
        }
        for Scalar(val: f32) -> f32 {
            val
        }
    }
}

impl_op! {
    fn load_unaligned<f32> {
        for Avx2(ptr: *const f32) -> __m256 {
            _mm256_loadu_ps(ptr)
        }
        for Scalar(ptr: *const f32) -> f32 {
            unsafe { *ptr }
        }
    }
}

impl_op! {
    fn load_aligned<f32> {
        for Avx2(ptr: *const f32) -> __m256 {
            _mm256_load_ps(ptr)
        }
        for Scalar(ptr: *const f32) -> f32 {
            unsafe { *ptr }
        }
    }
}

impl_op! {
    fn store_unaligned<f32> {
        for Avx2(ptr: *mut f32, a: __m256) {
            _mm256_storeu_ps(ptr, a)
        }
        for Scalar(ptr: *mut f32, a: f32) {
            unsafe { *ptr = a }
        }
    }
}

impl_op! {
    fn store_aligned<f32> {
        for Avx2(ptr: *mut f32, a: __m256) {
            _mm256_store_ps(ptr, a)
        }
        for Scalar(ptr: *mut f32, a: f32) {
            unsafe { *ptr = a }
        }
    }
}
