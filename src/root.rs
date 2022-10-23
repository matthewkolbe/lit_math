use std::arch::x86_64::*;
use super::*;


/// Computes y=sqrt(x) on x.len() values.
#[inline]
pub fn sqrt(x: &[f64], y: &mut [f64])
{
    unsafe{
        sqrtu(x, y);
    }
}


#[inline]
pub fn sqrtv(x: &Vec<f64>, y: &mut Vec<f64>)
{
    unsafe{
        sqrtvu(x, y);
    }
}


unroll_fn!(sqrtu, sqrtvu, sqrt_parvu,sqrt_intr, 8, f64);


#[target_feature(enable ="avx512f")]
pub unsafe fn sqrt_intr(x: &__m512d, y: &mut __m512d)
{
    *y = mm512_sqrt_pd(*x);
}