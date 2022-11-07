use std::arch::x86_64::*;
use super::*;


unroll_fn!(sqrt, sqrt_intr, _mm512_loadu_pd, _mm512_storeu_pd, __m512d, f64);


#[target_feature(enable ="avx512f")]
pub unsafe fn sqrt_intr(x: &__m512d, y: &mut __m512d)
{
    *y = _mm512_sqrt_pd(*x);
}