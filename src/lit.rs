use std::arch::x86_64::*;
use super::constants::*;

macro_rules! unroll_fn {
    ($name:ident, $fun:expr, $vsz:literal, $numsz:ty) => {
        #[inline]
        unsafe fn $name(x: &[$numsz], y: &mut [$numsz]) {
            let n = x.len() as i32;

            if n < $vsz as i32
            {
                let mut xx = [0.0; $vsz];
                let mut yy = [0.0; $vsz];
                for i in 0..x.len() {
                    xx[i] = x[i];
                }
                
                $fun(&xx, &mut yy, 0);
                for i in 0..y.len() {
                    y[i] = yy[i];
                }
                return;
            }

            let mut i: usize = 0;
            while (i as i32) < (n - 4*$vsz - 1)
            {
                $fun(x, y, i);
                i += $vsz;
                $fun(x, y, i);
                i += $vsz;
                $fun(x, y, i);
                i += $vsz;
                $fun(x, y, i);
                i += $vsz;
            }

            while (i as i32) < (n - $vsz + 1)
            {
                $fun(x, y, i);
                i += $vsz;
            }

            if (i as i32) != n
            {
                i = (n as usize) - $vsz;
                $fun(x, y, i);
            }
        }
    };
}

#[inline]
pub fn exp512(x: &[f64], y: &mut [f64])
{
    unsafe{
        exp512u(x, y);
    }
}

#[inline]
pub fn exp256(x: &[f64], y: &mut [f64])
{
    unsafe{
        exp256u(x, y);
    }
}

#[inline]
pub fn erf512(x: &[f64], y: &mut [f64])
{
    unsafe{
        erf512u(x, y);
    }
}

#[inline]
pub fn ln512(x: &[f64], y: &mut [f64])
{
    unsafe{
        ln512u(x, y);
    }
}

unroll_fn!(exp512u, expwo512, 8, f64);
unroll_fn!(exp256u, expwo256, 4, f64);
unroll_fn!(erf512u, erfwo512, 8, f64);
unroll_fn!(ln512u, lnwo512, 8, f64);

#[target_feature(enable ="avx512f")]
unsafe fn expwo512(x: &[f64], y: &mut [f64], offset: usize)
{
    let xx = _mm512_loadu_pd(&x[offset] as *const f64);
    let mut yy = _mm512_loadu_pd(&y[offset] as *const f64);
    expo512(&xx, &mut yy);
    _mm512_storeu_pd(&mut y[offset] as *mut f64, yy);
}


#[target_feature(enable ="avx512f")]
unsafe fn expwo256(x: &[f64], y: &mut [f64], offset: usize)
{
    let xx = _mm256_loadu_pd(&x[offset] as *const f64);
    let mut yy = _mm256_loadu_pd(&y[offset] as *const f64);
    expo256(&xx, &mut yy);
    _mm256_storeu_pd(&mut y[offset] as *mut f64, yy);
}

#[target_feature(enable ="avx512f")]
pub unsafe fn expo512(x: &__m512d, y: &mut __m512d)
{
    let xx = _mm512_mul_pd(*x, exp::D512_LOG2EF);
    two512(&xx, y);
}

#[target_feature(enable ="avx512f")]
pub unsafe fn expo256(x: &__m256d, y: &mut __m256d)
{
    let xx = _mm256_mul_pd(*x, exp::D256_LOG2EF);
    two256(&xx, y);
}

#[target_feature(enable ="avx512f")]
pub unsafe fn two512(x: &__m512d, y: &mut __m512d)
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


#[target_feature(enable ="avx")]
#[target_feature(enable ="avx512f")]
#[target_feature(enable ="fma")]
pub unsafe fn two256(x: &__m256d, y: &mut __m256d)
{
    // Checks if x is greater than the highest acceptable argument. Stores the information for later to
    // modify the result. If, for example, only x[1] > EXP_HIGH, then end[1] will be infinity, and the rest
    // zero. We add this to the result at the end, which will force y[1] to be infinity.
    let inf_mask = _mm256_cmp_pd_mask(*x, exp::D256_THIGH, _CMP_GT_OS);

    // Bound x by the maximum and minimum values this algorithm will handle.
    let mut xx = _mm256_max_pd(_mm256_min_pd(*x, exp::D256_THIGH), exp::D256_TLOW);

    // Avx.CompareNotEqual(x, x) is a hack to determine which values of x are NaN, since NaN is the only
    // value that doesn't equal itself. If any are NaN, we make the corresponding element of 'end' NaN, and
    // it acts like the infinity adjustment.
    let nan_mask = _mm256_cmp_pd_mask(*x, *x, _CMP_EQ_OS);

    let mut fx = _mm256_roundscale_pd(xx, _MM_FROUND_NEARBYINT);


    // This section gets a series approximation for exp(g) in (-0.5, 0.5) since that is g's range.
    xx = _mm256_sub_pd(xx, fx);
    *y = _mm256_fmadd_pd(exp::D256_T11, xx, exp::D256_T10);
    *y = _mm256_fmadd_pd(*y, xx, exp::D256_T9);
    *y = _mm256_fmadd_pd(*y, xx, exp::D256_T8);
    *y = _mm256_fmadd_pd(*y, xx, exp::D256_T7);
    *y = _mm256_fmadd_pd(*y, xx, exp::D256_T6);
    *y = _mm256_fmadd_pd(*y, xx, exp::D256_T5);
    *y = _mm256_fmadd_pd(*y, xx, exp::D256_T4);
    *y = _mm256_fmadd_pd(*y, xx, exp::D256_T3);
    *y = _mm256_fmadd_pd(*y, xx, exp::D256_T2);
    *y = _mm256_fmadd_pd(*y, xx, exp::D256_T1);
    *y = _mm256_fmadd_pd(*y, xx, exp::D256_T0);

    // Converts n to 2^n. There is no Avx2.ConvertToVector256Int64(fx) intrinsic, so we convert to int32's,
    // since the exponent of a double will never be more than a max int32, then from int to long.
    fx = _mm256_add_pd(fx, exp::D256_MAGIC_LONG_DOUBLE_ADD);
    fx = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(_mm256_castpd_si256(fx), exp::I256_ONE_THOUSAND_TWENTY_THREE), 52));


    // Combines the two exponentials and the end adjustments into the result.
    *y = _mm256_mul_pd(*y, fx);

    *y = _mm256_mask_add_pd(exp::D256_POSITIVE_INFINITY, inf_mask, exp::D256_ZERO, *y);
    *y = _mm256_mask_add_pd(exp::D256_NAN, nan_mask, *y, exp::D256_ZERO);
}


#[target_feature(enable ="avx512f")]
unsafe fn erfwo512(x: &[f64], y: &mut [f64], offset: usize)
{
    let xx = _mm512_loadu_pd(&x[offset] as *const f64);
    let mut yy = _mm512_loadu_pd(&y[offset] as *const f64);
    erfo512(&xx, &mut yy);
    _mm512_storeu_pd(&mut y[offset] as *mut f64, yy);
}

/// AVX-512 implementation of the ERF function.
#[target_feature(enable ="avx512f")]
pub unsafe fn erfo512(x: &__m512d, y: &mut __m512d)
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

    expo512(&exsq, &mut t);

    yy = _mm512_mul_pd(yy, t);
    yy = _mm512_add_pd(normdist::D512ONE, yy);

    *y = _mm512_mask_blend_pd(le_mask, yy, _mm512_mul_pd(yy, normdist::D512NEGONE));
    
}


#[target_feature(enable ="avx512f")]
unsafe fn lnwo512(x: &[f64], y: &mut [f64], offset: usize)
{
    let xx = _mm512_loadu_pd(&x[offset] as *const f64);
    let mut yy = _mm512_loadu_pd(&y[offset] as *const f64);
    logo512(&xx, &mut yy);
    _mm512_storeu_pd(&mut y[offset] as *mut f64, yy);
}

#[target_feature(enable ="avx512f")]
pub unsafe fn logo512(x: &__m512d, y: &mut __m512d)
{
    logo2(&x, y);
    *y = _mm512_mul_pd(log::D512_LN2, *y);
}

#[target_feature(enable ="avx512f")]
pub unsafe fn logo2(x: &__m512d, y: &mut __m512d)
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