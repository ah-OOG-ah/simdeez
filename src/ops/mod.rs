#![allow(dead_code)]

use crate::engines::scalar::Scalar;
use {crate::engines::avx2::Avx2, core::arch::x86_64::*};

use crate::libm_ext::FloatExt;
use core::marker::PhantomData;

mod i8;
mod i16;
mod i32;
mod i64;
mod f32;
mod f64;
mod bit;
mod casts;
#[allow(non_camel_case_types)]
pub struct binary;

pub struct Ops<T, T2>(PhantomData<(T, T2)>);

macro_rules! with_feature_flag {
    (Avx2, $($r:tt)+) => {
        $($r)+
    };
    (Scalar, $($r:tt)+) => {
        $($r)+
    };
}
use with_feature_flag;

macro_rules! with_cfg_flag {
    (Avx2, $($r:tt)+) => {
                $($r)+
    };
    (Scalar, $($r:tt)+) => {
        $($r)+
    };
}
use with_cfg_flag;

macro_rules! impl_op {
    (fn $name:ident < $scalar:ty > {
        $(
            for $engine:ident ($( $arg:ident : $arg_ty:ty ),*) $( -> $ret_ty:ty )? {
                $( $body:tt )*
            }
        )*
    }) => {
        $(
            with_cfg_flag!(
                $engine,
                impl Ops<$engine, $scalar> {
                    with_feature_flag!(
                        $engine,
                        #[inline]
                        pub unsafe fn $name($($arg: $arg_ty),*) $( -> $ret_ty )? {
                            $( $body )*
                        }
                    );
                }
            );
        )*
    }
}
use impl_op;

macro_rules! impl_imm8_op {
    (fn $name:ident < $scalar:ty, const $imm8:ident: $imm8ty:ty > {
        $(
            for $engine:ident ($( $arg:ident : $arg_ty:ty ),*) $( -> $ret_ty:ty )? {
                $( $body:tt )*
            }
        )*
    }) => {
        $(
            with_cfg_flag!(
                $engine,
                impl Ops<$engine, $scalar> {
                    with_feature_flag!(
                        $engine,
                        #[inline]
                        pub unsafe fn $name < const $imm8: $imm8ty > ($($arg: $arg_ty),*) $( -> $ret_ty )? {
                            $( $body )*
                        }
                    );
                }
            );
        )*
    }
}
use impl_imm8_op;
