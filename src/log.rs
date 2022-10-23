use std::arch::x86_64::*;
use super::*;


#[inline]
pub fn ln(x: &[f64], y: &mut [f64])
{
    unsafe{
        lnu(x, y);
    }
}

#[inline]
pub fn log2(x: &[f64], y: &mut [f64])
{
    unsafe{
        log2u(x, y);
    }
}

#[inline]
pub fn lnv(x: &Vec<f64>, y: &mut Vec<f64>)
{
    unsafe{
        lnvu(x, y);
    }
}

#[inline]
pub fn log2v(x: &Vec<f64>, y: &mut Vec<f64>)
{
    unsafe{
        log2vu(x, y);
    }
}


unroll_fn!(lnu, lnvu, ln_parvu, ln_intr, 8, f64);
unroll_fn!(log2u, log2vu, log2_parvu, log2_intr, 8, f64);

#[target_feature(enable ="avx512f")]
pub unsafe fn ln_intr(x: &__m512d, y: &mut __m512d)
{
    log2_intr(&x, y);
    *y = _mm512_mul_pd(D512_LN2, *y);
}

#[target_feature(enable ="avx512f")]
pub unsafe fn _mm512_ln_pd(x: __m512d) -> __m512d
{
    let mut y = log::D512_ZERO;
    log2_intr(&x, &mut y);
    _mm512_mul_pd(D512_LN2, y)
}

#[target_feature(enable ="avx512f")]
pub unsafe fn _mm512_log2_pd(x: __m512d) -> __m512d
{
    let mut y = D512_ZERO;
    log2_intr(&x, &mut y);
    y
}

#[target_feature(enable ="avx512f")]
pub unsafe fn log2_intr(x: &__m512d, y: &mut __m512d)
{
    // This algorithm uses the properties of floating point number to transform x into d*2^m, so log(x)
    // becomes log(d)+m, where d is in [1, 2]. Then it uses a series approximation of log to approximate 
    // the value in [1, 2]

    let xl = _mm512_getexp_pd(*x);
    let mantissa = _mm512_getmant_pd(*x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC);

    log2_in_1_2(&mantissa, y);

    *y = _mm512_add_pd(*y, xl);
    *y = _mm512_mask_blend_pd(_mm512_cmplt_pd_mask(*x, D512_ZERO), *y, D512_NAN);
    *y = _mm512_mask_blend_pd(_mm512_cmpeq_pd_mask(*x, D512_ZERO), *y, D512_NEGATIVE_INFINITY);
    //*y = _mm512_mask_blend_pd(_mm512_cmpeq_pd_mask(*x, log::D512_POSITIVE_INFINITY), log::D512_POSITIVE_INFINITY, *y);
    //*y = _mm512_mask_blend_pd(_mm512_cmpeq_pd_mask(*x, *x), *y, log::D512_NAN);
}


/// AVX-512 implementation of log base 2 in the interval of [1,2]
#[target_feature(enable ="avx512f")]
unsafe fn log2_in_1_2(x: &__m512d, y: &mut __m512d)
{
    *y = _mm512_mul_pd(*x, D512_TWO_THIRDS);
    *y = _mm512_div_pd(_mm512_sub_pd(*y, D512_ONE), _mm512_add_pd(*y, D512_ONE));
    let ysq = _mm512_mul_pd(*y, *y);

    let mut rx = _mm512_fmadd_pd(ysq, D512_T13, D512_T11);
    rx = _mm512_fmadd_pd(ysq, rx, D512_T9);
    rx = _mm512_fmadd_pd(ysq, rx, D512_T7);
    rx = _mm512_fmadd_pd(ysq, rx, D512_T5);
    rx = _mm512_fmadd_pd(ysq, rx, D512_T3);
    rx = _mm512_fmadd_pd(ysq, rx, D512_T1);

    rx = _mm512_mul_pd(*y, rx);
    *y = _mm512_add_pd(rx, D512_T0)
}


const D512_TWO_THIRDS: __m512d = m64x8_constant!(0.6666666666666666666);
const D512_ONE: __m512d = m64x8_constant!(1.0);
const D512_ZERO: __m512d = m64x8_constant!(0.0);
const D512_NEGATIVE_INFINITY: __m512d = m64x8_constant!(f64::NEG_INFINITY);
const D512_LN2: __m512d  = m64x8_constant!(0.6931471805599453094172321214581766);
const D512_NAN: __m512d = m64x8_constant!(f64::NAN);
const D512_T0: __m512d = m64x8_constant!(0.5849625007211562024634018319  );
const D512_T1: __m512d = m64x8_constant!(2.88539008177795423263363741  );
const D512_T3: __m512d = m64x8_constant!(0.96179669389977077508752 );
const D512_T5: __m512d = m64x8_constant!(0.577078023612080068567 );
const D512_T7: __m512d = m64x8_constant!(0.4121976972049074185   );
const D512_T9: __m512d = m64x8_constant!(0.32065422990573868  );
const D512_T11: __m512d = m64x8_constant!(0.2604711365240256 );
const D512_T13: __m512d = m64x8_constant!(0.252528834803695 );