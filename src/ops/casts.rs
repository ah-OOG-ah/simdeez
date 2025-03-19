use super::*;

impl_op! {
    fn bitcast_binary<f32> {
        for Avx2(a: __m256) -> __m256 {
            a
        }
        for Scalar(a: f32) -> u64 {
            a.to_bits() as u64
        }
    }
}

impl_op! {
    fn bitcast_f32<binary> {
        for Avx2(a: __m256) -> __m256 {
            a
        }
        for Scalar(a: u64) -> f32 {
            f32::from_bits(a as u32)
        }
    }
}

impl_op! {
    fn bitcast_binary<f64> {
        for Avx2(a: __m256d) -> __m256 {
            _mm256_castpd_ps(a)
        }
        for Scalar(a: f64) -> u64 {
            a.to_bits()
        }
    }
}

impl_op! {
    fn bitcast_f64<binary> {
        for Avx2(a: __m256) -> __m256d {
            _mm256_castps_pd(a)
        }
        for Scalar(a: u64) -> f64 {
            f64::from_bits(a)
        }
    }
}

impl_op! {
    fn bitcast_binary<i8> {
        for Avx2(a: __m256i) -> __m256 {
            _mm256_castsi256_ps(a)
        }
        for Scalar(a: i8) -> u64 {
            a as u64
        }
    }
}

impl_op! {
    fn bitcast_i8<binary> {
        for Avx2(a: __m256) -> __m256i {
            _mm256_castps_si256(a)
        }
        for Scalar(a: u64) -> i8 {
            a as i8
        }
    }
}

impl_op! {
    fn bitcast_binary<i16> {
        for Avx2(a: __m256i) -> __m256 {
            _mm256_castsi256_ps(a)
        }
        for Scalar(a: i16) -> u64 {
            a as u64
        }
    }
}

impl_op! {
    fn bitcast_i16<binary> {
        for Avx2(a: __m256) -> __m256i {
            _mm256_castps_si256(a)
        }
        for Scalar(a: u64) -> i16 {
            a as i16
        }
    }
}

impl_op! {
    fn bitcast_binary<i32> {
        for Avx2(a: __m256i) -> __m256 {
            _mm256_castsi256_ps(a)
        }
        for Scalar(a: i32) -> u64 {
            a as u64
        }
    }
}

impl_op! {
    fn bitcast_i32<binary> {
        for Avx2(a: __m256) -> __m256i {
            _mm256_castps_si256(a)
        }
        for Scalar(a: u64) -> i32 {
            a as i32
        }
    }
}

impl_op! {
    fn bitcast_binary<i64> {
        for Avx2(a: __m256i) -> __m256 {
            _mm256_castsi256_ps(a)
        }
        for Scalar(a: i64) -> u64 {
            a as u64
        }
    }
}

impl_op! {
    fn bitcast_i64<binary> {
        for Avx2(a: __m256) -> __m256i {
            _mm256_castps_si256(a)
        }
        for Scalar(a: u64) -> i64 {
            a as i64
        }
    }
}
