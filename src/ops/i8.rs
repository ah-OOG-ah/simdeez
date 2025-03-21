use super::*;

impl_op! {
    fn add<i8> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            _mm256_add_epi8(a, b)
        }
        for Scalar(a: i8, b: i8) -> i8 {
            a.wrapping_add(b)
        }
    }
}

impl_op! {
    fn sub<i8> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            _mm256_sub_epi8(a, b)
        }
        for Scalar(a: i8, b: i8) -> i8 {
            a.wrapping_sub(b)
        }
    }
}

impl_op! {
    fn mul<i8> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            let mut arr1 = core::mem::transmute::<__m256i, [i8; 32]>(a);
            let arr2 = core::mem::transmute::<__m256i, [i8; 32]>(b);
            for i in 0..32 {
                arr1[i] = arr1[i].wrapping_mul(arr2[i]);
            }
            core::mem::transmute::<_, _>(arr1)
        }
        for Scalar(a: i8, b: i8) -> i8 {
            a.wrapping_mul(b)
        }
    }
}

impl_op! {
    fn min<i8> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            _mm256_min_epi8(a, b)
        }
        for Scalar(a: i8, b: i8) -> i8 {
            a.min(b)
        }
    }
}

impl_op! {
    fn max<i8> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            _mm256_max_epi8(a, b)
        }
        for Scalar(a: i8, b: i8) -> i8 {
            a.max(b)
        }
    }
}

impl_op! {
    fn abs<i8> {
        for Avx2(a: __m256i) -> __m256i {
            _mm256_abs_epi8(a)
        }
        for Scalar(a: i8) -> i8 {
            a.abs()
        }
    }
}

impl_op! {
    fn eq<i8> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpeq_epi8(a, b)
        }
        for Scalar(a: i8, b: i8) -> i8 {
            if a == b {
                u32::MAX as i8
            } else {
                0
            }
        }
    }
}

impl_op! {
    fn neq<i8> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            let eq = _mm256_cmpeq_epi8(a, b);
            _mm256_xor_si256(eq, _mm256_set1_epi8(u32::MAX as i8))
        }
        for Scalar(a: i8, b: i8) -> i8 {
            if a != b {
                u32::MAX as i8
            } else {
                0
            }
        }
    }
}

impl_op! {
    fn lt<i8> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            let gt = _mm256_cmpgt_epi8(a, b);
            let eq = _mm256_cmpeq_epi8(a, b);
            _mm256_andnot_si256(_mm256_or_si256(gt, eq), _mm256_set1_epi8(u32::MAX as i8))
        }
        for Scalar(a: i8, b: i8) -> i8 {
            if a < b {
                u32::MAX as i8
            } else {
                0
            }
        }
    }
}

impl_op! {
    fn lte<i8> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            let gt = _mm256_cmpgt_epi8(a, b);
            _mm256_xor_si256(gt, _mm256_set1_epi8(u32::MAX as i8))
        }
        for Scalar(a: i8, b: i8) -> i8 {
            if a <= b {
                u32::MAX as i8
            } else {
                0
            }
        }
    }
}

impl_op! {
    fn gt<i8> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi8(a, b)
        }
        for Scalar(a: i8, b: i8) -> i8 {
            if a > b {
                u32::MAX as i8
            } else {
                0
            }
        }
    }
}

impl_op! {
    fn gte<i8> {
        for Avx2(a: __m256i, b: __m256i) -> __m256i {
            let gt = _mm256_cmpgt_epi8(a, b);
            let eq = _mm256_cmpeq_epi8(a, b);
            _mm256_or_si256(gt, eq)
        }
        for Scalar(a: i8, b: i8) -> i8 {
            if a >= b {
                u32::MAX as i8
            } else {
                0
            }
        }
    }
}

impl_op! {
    fn blendv<i8> {
        for Avx2(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
            _mm256_blendv_epi8(a, b, mask)
        }
        for Scalar(a: i8, b: i8, mask: i8) -> i8 {
            if mask == 0 {
                a
            } else {
                b
            }
        }
    }
}

impl_op! {
    fn shl<i8> {
        for Avx2(a: __m256i, rhs: i32) -> __m256i {
            // Do 16 bit shifts, then mask out the bits that are shifted in.
            let rhs2 = _mm_cvtsi32_si128(rhs);
            let shifted_i16 = _mm256_sll_epi16(a, rhs2);

            let mask = 0x00FFu16 >> (8 - rhs) << 8;
            let mask = _mm256_set1_epi16(mask as i16);
            _mm256_andnot_si256(mask, shifted_i16)
        }
        for Scalar(a: i8, rhs: i32) -> i8 {
            a << rhs
        }
    }
}

impl_op! {
    fn shr<i8> {
        for Avx2(a: __m256i, rhs: i32) -> __m256i {
            // Do 16 bit shifts, then mask out the bits that are shifted in.
            let rhs2 = _mm_cvtsi32_si128(rhs);
            let shifted_i16 = _mm256_srl_epi16(a, rhs2);

            let mask = 0xFF00u16 << (8 - rhs) >> 8;
            let mask = _mm256_set1_epi16(mask as i16);
            _mm256_andnot_si256(mask, shifted_i16)
        }
        for Scalar(a: i8, rhs: i32) -> i8 {
            ((a as u8) >> rhs) as i8
        }
    }
}

impl_imm8_op! {
    fn shl_const<i8, const BY: i32> {
        for Avx2(a: __m256i) -> __m256i {
            Self::shl(a, BY)
        }
        for Scalar(a: i8) -> i8 {
            a << BY
        }
    }
}

impl_imm8_op! {
    fn shr_const<i8, const BY: i32> {
        for Avx2(a: __m256i) -> __m256i {
            Self::shr(a, BY)
        }
        for Scalar(a: i8) -> i8 {
            ((a as u8) >> BY) as i8
        }
    }
}

impl_op! {
    fn extend_i16<i8> {
        for Avx2(val: __m256i) -> (__m256i, __m256i) {
            let a = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(val, 0));
            let b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(val, 1));
            (a, b)
        }
        for Scalar(val: i8) -> (i16, i16) {
            (val as i16, 0)
        }
    }
}

impl_op! {
    fn unsigned_extend_i16<i8> {
        for Avx2(val: __m256i) -> (__m256i, __m256i) {
            let a = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(val, 0));
            let b = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(val, 1));
            (a, b)
        }
        for Scalar(val: i8) -> (i16, i16) {
            (val as u8 as u16 as i16, 0)
        }
    }
}

impl_op! {
    fn get_mask<i8> {
        for Avx2(val: __m256i) -> u32 {
            _mm256_movemask_epi8(val) as u32
        }
        for Scalar(val: i8) -> u32 {
            ((val as u8) & 0x1) as u32
        }
    }
}

impl_op! {
    fn zeroes<i8> {
        for Avx2() -> __m256i {
            _mm256_setzero_si256()
        }
        for Scalar() -> i8 {
            0
        }
    }
}

impl_op! {
    fn set1<i8> {
        for Avx2(val: i8) -> __m256i {
            _mm256_set1_epi8(val)
        }
        for Scalar(val: i8) -> i8 {
            val
        }
    }
}

impl_op! {
    fn load_unaligned<i8> {
        for Avx2(ptr: *const i8) -> __m256i {
            _mm256_loadu_si256(ptr as *const __m256i)
        }
        for Scalar(ptr: *const i8) -> i8 {
            unsafe { *ptr }
        }
    }
}

impl_op! {
    fn load_aligned<i8> {
        for Avx2(ptr: *const i8) -> __m256i {
            _mm256_load_si256(ptr as *const __m256i)
        }
        for Scalar(ptr: *const i8) -> i8 {
            unsafe { *ptr }
        }
    }
}

impl_op! {
    fn store_unaligned<i8> {
        for Avx2(ptr: *mut i8, a: __m256i) {
            _mm256_storeu_si256(ptr as *mut __m256i, a)
        }
        for Scalar(ptr: *mut i8, a: i8) {
            unsafe { *ptr = a }
        }
    }
}

impl_op! {
    fn store_aligned<i8> {
        for Avx2(ptr: *mut i8, a: __m256i) {
            _mm256_store_si256(ptr as *mut __m256i, a)
        }
        for Scalar(ptr: *mut i8, a: i8) {
            unsafe { *ptr = a }
        }
    }
}

impl_op! {
    fn is_truthy<i8> {
        for Avx2(a: __m256i) -> bool {
            let cmp = _mm256_cmpeq_epi8(a, _mm256_setzero_si256());
            _mm256_testz_si256(cmp, cmp) == 0
        }
        for Scalar(a: i8) -> bool {
            a != 0
        }
    }
}
