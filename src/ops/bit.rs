use super::*;

impl_op! {
    fn bit_and<binary> {
        for Avx2(a: __m256, b: __m256) -> __m256 {
            _mm256_and_ps(a, b)
        }
        for Scalar(a: u64, b: u64) -> u64 {
            a & b
        }
    }
}

impl_op! {
    fn bit_or<binary> {
        for Avx2(a: __m256, b: __m256) -> __m256 {
            _mm256_or_ps(a, b)
        }
        for Scalar(a: u64, b: u64) -> u64 {
            a | b
        }
    }
}

impl_op! {
    fn bit_xor<binary> {
        for Avx2(a: __m256, b: __m256) -> __m256 {
            _mm256_xor_ps(a, b)
        }
        for Scalar(a: u64, b: u64) -> u64 {
            a ^ b
        }
    }
}

impl_op! {
    fn bit_not<binary> {
        for Avx2(a: __m256) -> __m256 {
            let all1 = _mm256_set1_epi32(-1);
            _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(a), all1))
        }
        for Scalar(a: u64) -> u64 {
            !a
        }
    }
}

impl_op! {
    fn bit_andnot<binary> {
        for Avx2(a: __m256, b: __m256) -> __m256 {
            _mm256_andnot_ps(a, b)
        }
        for Scalar(a: u64, b: u64) -> u64 {
            !a & b
        }
    }
}
