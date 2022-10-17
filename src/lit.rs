use std::arch::x86_64::*;
use super::constants::*;
use super::*;


#[inline]
pub fn exp(x: &[f64], y: &mut [f64])
{
    unsafe{
        expu(x, y);
    }
}

#[inline]
pub fn erf(x: &[f64], y: &mut [f64])
{
    unsafe{
        erfu(x, y);
    }
}

#[inline]
pub fn ln(x: &[f64], y: &mut [f64])
{
    unsafe{
        lnu(x, y);
    }
}

unroll_fn!(expu, exp_with_offset, 8, f64);
unroll_fn!(erfu, erf_with_offset, 8, f64);
unroll_fn!(lnu, ln_with_offset, 8, f64);

#[target_feature(enable ="avx512f")]
unsafe fn exp_with_offset(x: &[f64], y: &mut [f64], offset: usize)
{
    let xx = _mm512_loadu_pd(&x[offset] as *const f64);
    let mut yy = _mm512_loadu_pd(&y[offset] as *const f64);
    expintr(&xx, &mut yy);
    _mm512_storeu_pd(&mut y[offset] as *mut f64, yy);
}

#[target_feature(enable ="avx512f")]
pub unsafe fn expintr(x: &__m512d, y: &mut __m512d)
{
    let xx = _mm512_mul_pd(*x, exp::D512_LOG2EF);
    twointr(&xx, y);
}

#[target_feature(enable ="avx512f")]
pub unsafe fn _mm512_powe_pd(x: __m512d) -> __m512d
{
    let xx = _mm512_mul_pd(x, exp::D512_LOG2EF);
    let mut y = exp::D512_ZERO;
    twointr(&xx, &mut y);
    y
}

#[target_feature(enable ="avx512f")]
pub unsafe fn _mm512_pow2_pd(x: __m512d) -> __m512d
{
    let mut y = exp::D512_ZERO;
    twointr(&x, &mut y);
    y
}


#[target_feature(enable ="avx512f")]
pub unsafe fn twointr(x: &__m512d, y: &mut __m512d)
{
    // Checks if x is greater than the highest acceptable argument. Stores the information for later to
    // modify the result. If, for example, only x[1] > EXP_HIGH, then end[1] will be infinity, and the rest
    // zero. We add this to the result at the end, which will force y[1] to be infinity.
    let inf_mask = _mm512_cmple_pd_mask(*x, exp::D512_THIGH);

    // Bound x by the maximum and minimum values this algorithm will handle.
    let mut xx = _mm512_max_pd(_mm512_min_pd(*x, exp::D512_THIGH), exp::D512_TLOW);

    // Avx.CompareNotEqual(x, x) is a hack to determine which values of x are NaN, since NaN is the only
    // value that doesn't equal itself. If any are NaN, we make the corresponding element of 'end' NaN, and
    // it acts like the infinity adjustment.
    let nan_mask = _mm512_cmpeq_pd_mask(*x, *x);

    let mut fx = _mm512_roundscale_pd(xx, _MM_FROUND_NEARBYINT);


    // This section gets a series approximation for exp(g) in (-0.5, 0.5) since that is g's range.
    xx = _mm512_sub_pd(xx, fx);
    *y = _mm512_fmadd_pd(exp::D512_T11, xx, exp::D512_T10);
    *y = _mm512_fmadd_pd(*y, xx, exp::D512_T9);
    *y = _mm512_fmadd_pd(*y, xx, exp::D512_T8);
    *y = _mm512_fmadd_pd(*y, xx, exp::D512_T7);
    *y = _mm512_fmadd_pd(*y, xx, exp::D512_T6);
    *y = _mm512_fmadd_pd(*y, xx, exp::D512_T5);
    *y = _mm512_fmadd_pd(*y, xx, exp::D512_T4);
    *y = _mm512_fmadd_pd(*y, xx, exp::D512_T3);
    *y = _mm512_fmadd_pd(*y, xx, exp::D512_T2);
    *y = _mm512_fmadd_pd(*y, xx, exp::D512_T1);
    *y = _mm512_fmadd_pd(*y, xx, exp::D512_T0);

    // Converts n to 2^n. There is no Avx2.ConvertToVector256Int64(fx) intrinsic, so we convert to int32's,
    // since the exponent of a double will never be more than a max int32, then from int to long.
    fx = _mm512_add_pd(fx, exp::D512_MAGIC_LONG_DOUBLE_ADD);
    fx = _mm512_castsi512_pd(_mm512_slli_epi64(_mm512_add_epi64(_mm512_castpd_si512(fx), exp::I512_ONE_THOUSAND_TWENTY_THREE), 52));


    // Combines the two exponentials and the end adjustments into the result.
    *y = _mm512_mul_pd(*y, fx);

    *y = _mm512_mask_blend_pd(inf_mask, exp::D512_POSITIVE_INFINITY, *y);
    *y = _mm512_mask_blend_pd(nan_mask, exp::D512_NAN, *y);
}

#[target_feature(enable ="avx512f")]
unsafe fn erf_with_offset(x: &[f64], y: &mut [f64], offset: usize)
{
    let xx = _mm512_loadu_pd(&x[offset] as *const f64);
    let mut yy = _mm512_loadu_pd(&y[offset] as *const f64);
    erfintr(&xx, &mut yy);
    _mm512_storeu_pd(&mut y[offset] as *mut f64, yy);
}

#[target_feature(enable ="avx512f")]
pub unsafe fn _mm512_erf_pd(x: __m512d) -> __m512d
{
    let mut y = exp::D512_ZERO;
    erfintr(&x, &mut y);
    y
}

/// AVX-512 implementation of the ERF function.
#[target_feature(enable ="avx512f")]
unsafe fn erfintr(x: &__m512d, y: &mut __m512d)
{

    let le_mask = _mm512_cmple_pd_mask(*x, normdist::D512NEGATIVE_ZERO);
    let xx = _mm512_abs_pd(*x);

    let mut t = _mm512_fmadd_pd(normdist::D512ONE_OVER_PI, xx, normdist::D512ONE);
    t = _mm512_div_pd(normdist::D512ONE, t);

    let mut yy = _mm512_fmadd_pd(normdist::D512E12, t, normdist::D512E11);
    yy = _mm512_fmadd_pd(yy, t, normdist::D512E10);
    yy = _mm512_fmadd_pd(yy, t, normdist::D512E9);
    yy = _mm512_fmadd_pd(yy, t, normdist::D512E8);
    yy = _mm512_fmadd_pd(yy, t, normdist::D512E7);
    yy = _mm512_fmadd_pd(yy, t, normdist::D512E6);
    yy = _mm512_fmadd_pd(yy, t, normdist::D512E5);
    yy = _mm512_fmadd_pd(yy, t, normdist::D512E4);
    yy = _mm512_fmadd_pd(yy, t, normdist::D512E3);
    yy = _mm512_fmadd_pd(yy, t, normdist::D512E2);
    yy = _mm512_fmadd_pd(yy, t, normdist::D512E1);
    yy = _mm512_mul_pd(yy, t);

    let exsq = _mm512_mul_pd(_mm512_mul_pd(xx, normdist::D512NEGONE), xx);

    expintr(&exsq, &mut t);

    yy = _mm512_mul_pd(yy, t);
    yy = _mm512_add_pd(normdist::D512ONE, yy);

    *y = _mm512_mask_blend_pd(le_mask, yy, _mm512_mul_pd(yy, normdist::D512NEGONE));
    
}


#[target_feature(enable ="avx512f")]
unsafe fn ln_with_offset(x: &[f64], y: &mut [f64], offset: usize)
{
    let xx = _mm512_loadu_pd(&x[offset] as *const f64);
    let mut yy = _mm512_loadu_pd(&y[offset] as *const f64);
    lnintr(&xx, &mut yy);
    _mm512_storeu_pd(&mut y[offset] as *mut f64, yy);
}

#[target_feature(enable ="avx512f")]
pub unsafe fn lnintr(x: &__m512d, y: &mut __m512d)
{
    log2intr(&x, y);
    *y = _mm512_mul_pd(log::D512_LN2, *y);
}

#[target_feature(enable ="avx512f")]
pub unsafe fn _mm512_ln_pd(x: __m512d) -> __m512d
{
    let mut y = exp::D512_ZERO;
    log2intr(&x, &mut y);
    _mm512_mul_pd(log::D512_LN2, y)
}

#[target_feature(enable ="avx512f")]
pub unsafe fn _mm512_log2_pd(x: __m512d) -> __m512d
{
    let mut y = exp::D512_ZERO;
    log2intr(&x, &mut y);
    y
}

#[target_feature(enable ="avx512f")]
pub unsafe fn log2intr(x: &__m512d, y: &mut __m512d)
{
    // This algorithm uses the properties of floating point number to transform x into d*2^m, so log(x)
    // becomes log(d)+m, where d is in [1, 2]. Then it uses a series approximation of log to approximate 
    // the value in [1, 2]

    let xl = _mm512_getexp_pd(*x);
    let mantissa = _mm512_getmant_pd(*x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC);

    log2_in_1_2(&mantissa, y);

    *y = _mm512_add_pd(*y, xl);
    *y = _mm512_mask_blend_pd(_mm512_cmplt_pd_mask(*x, log::D512_ZERO), *y, log::D512_NAN);
    *y = _mm512_mask_blend_pd(_mm512_cmpeq_pd_mask(*x, log::D512_ZERO), *y, log::D512_NEGATIVE_INFINITY);
    //*y = _mm512_mask_blend_pd(_mm512_cmpeq_pd_mask(*x, log::D512_POSITIVE_INFINITY), log::D512_POSITIVE_INFINITY, *y);
    //*y = _mm512_mask_blend_pd(_mm512_cmpeq_pd_mask(*x, *x), *y, log::D512_NAN);
}


/// AVX-512 implementation of log base 2 in the interval of [1,2]
#[target_feature(enable ="avx512f")]
unsafe fn log2_in_1_2(x: &__m512d, y: &mut __m512d)
{
    *y = _mm512_mul_pd(*x, log::D512_TWO_THIRDS);
    *y = _mm512_div_pd(_mm512_sub_pd(*y, log::D512_ONE), _mm512_add_pd(*y, log::D512_ONE));
    let ysq = _mm512_mul_pd(*y, *y);

    let mut rx = _mm512_fmadd_pd(ysq, log::D512_T13, log::D512_T11);
    rx = _mm512_fmadd_pd(ysq, rx, log::D512_T9);
    rx = _mm512_fmadd_pd(ysq, rx, log::D512_T7);
    rx = _mm512_fmadd_pd(ysq, rx, log::D512_T5);
    rx = _mm512_fmadd_pd(ysq, rx, log::D512_T3);
    rx = _mm512_fmadd_pd(ysq, rx, log::D512_T1);

    rx = _mm512_mul_pd(*y, rx);
    *y = _mm512_add_pd(rx, log::D512_T0)
}