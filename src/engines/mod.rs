pub mod scalar;

#[cfg(any(target_arch = "x86_64", target_feature = "avx2"))]
pub mod avx2;
