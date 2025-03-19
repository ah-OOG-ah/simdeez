use super::*;

impl_op! {
    fn add<i32> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            _mm256_add_epi32(a, b)
        }
        for Scalar(a: i32, b: i32) -> i32 {
            a.wrapping_add(b)
        }
    }
}

impl_op! {
    fn sub<i32> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            _mm256_sub_epi32(a, b)
        }
        for Scalar(a: i32, b: i32) -> i32 {
            a.wrapping_sub(b)
        }
    }
}

impl_op! {
    fn mul<i32> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            _mm256_mullo_epi32(a, b)
        }
        for Scalar(a: i32, b: i32) -> i32 {
            a.wrapping_mul(b)
        }
    }
}

impl_op! {
    fn min<i32> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            _mm256_min_epi32(a, b)
        }
        for Scalar(a: i32, b: i32) -> i32 {
            a.min(b)
        }
    }
}

impl_op! {
    fn max<i32> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            _mm256_max_epi32(a, b)
        }
        for Scalar(a: i32, b: i32) -> i32 {
            a.max(b)
        }
    }
}

impl_op! {
    fn abs<i32> {
        for Avx2(a: __m256i) -> __m256i {
            _mm256_abs_epi32(a)
        }
        for Scalar(a: i32) -> i32 {
            a.abs()
        }
    }
}

impl_op! {
    fn eq<i32> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpeq_epi32(a, b)
        }
        for Scalar(a: i32, b: i32) -> i32 {
            if a == b {
                u32::MAX as i32
            } else {
                0
            }
        }
    }
}

impl_op! {
    fn neq<i32> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            let eq = _mm256_cmpeq_epi32(a, b);
            _mm256_xor_si256(eq, _mm256_set1_epi32(u32::MAX as i32))
        }
        for Scalar(a: i32, b: i32) -> i32 {
            if a != b {
                u32::MAX as i32
            } else {
                0
            }
        }
    }
}

impl_op! {
    fn lt<i32> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            let gt = _mm256_cmpgt_epi32(a, b);
            let eq = _mm256_cmpeq_epi32(a, b);
            _mm256_andnot_si256(_mm256_or_si256(gt, eq), _mm256_set1_epi32(u32::MAX as i32))
        }
        for Scalar(a: i32, b: i32) -> i32 {
            if a < b {
                u32::MAX as i32
            } else {
                0
            }
        }
    }
}

impl_op! {
    fn lte<i32> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            let gt = _mm256_cmpgt_epi32(a, b);
            _mm256_xor_si256(gt, _mm256_set1_epi32(u32::MAX as i32))
        }
        for Scalar(a: i32, b: i32) -> i32 {
            if a <= b {
                u32::MAX as i32
            } else {
                0
            }
        }
    }
}

impl_op! {
    fn gt<i32> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi32(a, b)
        }
        for Scalar(a: i32, b: i32) -> i32 {
            if a > b {
                u32::MAX as i32
            } else {
                0
            }
        }
    }
}

impl_op! {
    fn gte<i32> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            let gt = _mm256_cmpgt_epi32(a, b);
            let eq = _mm256_cmpeq_epi32(a, b);
            _mm256_or_si256(gt, eq)
        }
        for Scalar(a: i32, b: i32) -> i32 {
            if a >= b {
                u32::MAX as i32
            } else {
                0
            }
        }
    }
}

impl_op! {
    fn blendv<i32> {
        for Avx2(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
            _mm256_blendv_epi8(a, b, mask)
        }
        for Scalar(a: i32, b: i32, mask: i32) -> i32 {
            if mask == 0 {
                a
            } else {
                b
            }
        }
    }
}

impl_op! {
    fn shl<i32> {
        for Avx2(a: __m256i, rhs: i32) -> __m256i {
            _mm256_sll_epi32(a, _mm_cvtsi32_si128(rhs))
        }
        for Scalar(a: i32, rhs: i32) -> i32 {
            a << rhs
        }
    }
}

impl_op! {
    fn shr<i32> {
        for Avx2(a: __m256i, rhs: i32) -> __m256i {
            _mm256_srl_epi32(a, _mm_cvtsi32_si128(rhs))
        }
        for Scalar(a: i32, rhs: i32) -> i32 {
            ((a as u32) >> rhs) as i32
        }
    }
}

impl_imm8_op! {
    fn shl_const<i32, const BY: i32> {
        for Avx2(a: __m256i) -> __m256i {
            _mm256_slli_epi32(a, BY)
        }
        for Scalar(a: i32) -> i32 {
            a << BY
        }
    }
}

impl_imm8_op! {
    fn shr_const<i32, const BY: i32> {
        for Avx2(a: __m256i) -> __m256i {
            _mm256_srli_epi32(a, BY)
        }
        for Scalar(a: i32) -> i32 {
            ((a as u32) >> BY) as i32
        }
    }
}

impl_op! {
    fn cast_f32<i32> {
        for Avx2(a: __m256i) -> __m256 {
            _mm256_cvtepi32_ps(a)
        }
        for Scalar(a: i32) -> f32 {
            a as f32
        }
    }
}

impl_op! {
    fn bitcast_f32<i32> {
        for Avx2(a: __m256i) -> __m256 {
            _mm256_castsi256_ps(a)
        }
        for Scalar(a: i32) -> f32 {
            f32::from_bits(a as u32)
        }
    }
}

impl_op! {
    fn extend_i64<i32> {
        for Avx2(val: __m256i) -> (__m256i, __m256i) {
            let a = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(val, 0));
            let b = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(val, 1));
            (a, b)
        }
        for Scalar(val: i32) -> (i64, i64) {
            (val as i64, 0)
        }
    }
}

impl_op! {
    fn unsigned_extend_i64<i32> {
        for Avx2(val: __m256i) -> (__m256i, __m256i) {
            let a = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(val, 0));
            let b = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(val, 1));
            (a, b)
        }
        for Scalar(val: i32) -> (i64, i64) {
            (val as u32 as u64 as i64, 0)
        }
    }
}

impl_op! {
    fn zeroes<i32> {
        for Avx2() -> __m256i {
            _mm256_setzero_si256()
        }
        for Scalar() -> i32 {
            0
        }
    }
}

impl_op! {
    fn set1<i32> {
        for Avx2(val: i32) -> __m256i {
            _mm256_set1_epi32(val)
        }
        for Scalar(val: i32) -> i32 {
            val
        }
    }
}

impl_op! {
    fn load_unaligned<i32> {
        for Avx2(ptr: *const i32) -> __m256i {
            _mm256_loadu_si256(ptr as *const __m256i)
        }
        for Scalar(ptr: *const i32) -> i32 {
            unsafe { *ptr }
        }
    }
}

impl_op! {
    fn load_aligned<i32> {
        for Avx2(ptr: *const i32) -> __m256i {
            _mm256_load_si256(ptr as *const __m256i)
        }
        for Scalar(ptr: *const i32) -> i32 {
            unsafe { *ptr }
        }
    }
}

impl_op! {
    fn store_unaligned<i32> {
        for Avx2(ptr: *mut i32, a: __m256i) {
            _mm256_storeu_si256(ptr as *mut __m256i, a)
        }
        for Scalar(ptr: *mut i32, a: i32) {
            unsafe { *ptr = a }
        }
    }
}

impl_op! {
    fn store_aligned<i32> {
        for Avx2(ptr: *mut i32, a: __m256i) {
            _mm256_store_si256(ptr as *mut __m256i, a)
        }
        for Scalar(ptr: *mut i32, a: i32) {
            unsafe { *ptr = a }
        }
    }
}
