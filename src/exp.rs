use std::arch::x86_64::*;
use super::*;

#[inline]
pub fn exp(x: &[f64], y: &mut [f64])
{
    unsafe{
        expu(x, y);
    }
}

#[inline]
pub fn exp2(x: &[f64], y: &mut [f64])
{
    unsafe{
        exp2u(x, y);
    }
}

unroll_fn!(expu, exp_with_offset, 8, f64);
unroll_fn!(exp2u, exp2_with_offset, 8, f64);

#[target_feature(enable ="avx512f")]
unsafe fn exp_with_offset(x: &[f64], y: &mut [f64], offset: usize)
{
    let xx = _mm512_loadu_pd(&x[offset] as *const f64);
    let mut yy = _mm512_loadu_pd(&y[offset] as *const f64);
    exp_intr(&xx, &mut yy);
    _mm512_storeu_pd(&mut y[offset] as *mut f64, yy);
}

#[target_feature(enable ="avx512f")]
unsafe fn exp2_with_offset(x: &[f64], y: &mut [f64], offset: usize)
{
    let xx = _mm512_loadu_pd(&x[offset] as *const f64);
    let mut yy = _mm512_loadu_pd(&y[offset] as *const f64);
    exp2_intr(&xx, &mut yy);
    _mm512_storeu_pd(&mut y[offset] as *mut f64, yy);
}

#[target_feature(enable ="avx512f")]
pub unsafe fn exp_intr(x: &__m512d, y: &mut __m512d)
{
    let xx = _mm512_mul_pd(*x, D512_LOG2EF);
    exp2_intr(&xx, y);
}

#[target_feature(enable ="avx512f")]
pub unsafe fn _mm512_powe_pd(x: __m512d) -> __m512d
{
    let xx = _mm512_mul_pd(x, D512_LOG2EF);
    let mut y = D512_ZERO;
    exp2_intr(&xx, &mut y);
    y
}

#[target_feature(enable ="avx512f")]
pub unsafe fn _mm512_pow2_pd(x: __m512d) -> __m512d
{
    let mut y = D512_ZERO;
    exp2_intr(&x, &mut y);
    y
}

#[target_feature(enable ="avx512f")]
pub unsafe fn exp2_intr(x: &__m512d, y: &mut __m512d)
{
    // Checks if x is greater than the highest acceptable argument. Stores the information for later to
    // modify the result. If, for example, only x[1] > EXP_HIGH, then end[1] will be infinity, and the rest
    // zero. We add this to the result at the end, which will force y[1] to be infinity.
    let inf_mask = _mm512_cmple_pd_mask(*x, D512_THIGH);

    // Bound x by the maximum and minimum values this algorithm will handle.
    let mut xx = _mm512_max_pd(_mm512_min_pd(*x, D512_THIGH), D512_TLOW);

    // Avx.CompareNotEqual(x, x) is a hack to determine which values of x are NaN, since NaN is the only
    // value that doesn't equal itself. If any are NaN, we make the corresponding element of 'end' NaN, and
    // it acts like the infinity adjustment.
    let nan_mask = _mm512_cmpeq_pd_mask(*x, *x);

    let mut fx = _mm512_roundscale_pd(xx, _MM_FROUND_NEARBYINT);


    // This section gets a series approximation for exp(g) in (-0.5, 0.5) since that is g's range.
    xx = _mm512_sub_pd(xx, fx);
    *y = _mm512_fmadd_pd(D512_T11, xx, D512_T10);
    *y = _mm512_fmadd_pd(*y, xx, D512_T9);
    *y = _mm512_fmadd_pd(*y, xx, D512_T8);
    *y = _mm512_fmadd_pd(*y, xx, D512_T7);
    *y = _mm512_fmadd_pd(*y, xx, D512_T6);
    *y = _mm512_fmadd_pd(*y, xx, D512_T5);
    *y = _mm512_fmadd_pd(*y, xx, D512_T4);
    *y = _mm512_fmadd_pd(*y, xx, D512_T3);
    *y = _mm512_fmadd_pd(*y, xx, D512_T2);
    *y = _mm512_fmadd_pd(*y, xx, D512_T1);
    *y = _mm512_fmadd_pd(*y, xx, D512_T0);

    // Converts n to 2^n. There is no Avx2.ConvertToVector256Int64(fx) intrinsic, so we convert to int32's,
    // since the exponent of a double will never be more than a max int32, then from int to long.
    fx = _mm512_add_pd(fx, D512_MAGIC_LONG_DOUBLE_ADD);
    fx = _mm512_castsi512_pd(_mm512_slli_epi64(_mm512_add_epi64(_mm512_castpd_si512(fx), I512_ONE_THOUSAND_TWENTY_THREE), 52));


    // Combines the two exponentials and the end adjustments into the result.
    *y = _mm512_mul_pd(*y, fx);

    *y = _mm512_mask_blend_pd(inf_mask, D512_POSITIVE_INFINITY, *y);
    *y = _mm512_mask_blend_pd(nan_mask, D512_NAN, *y);
}



const D512_T0: __m512d = m64x8_constant!(1.0);
const D512_T1: __m512d = m64x8_constant!(0.6931471805599453087156032);
const D512_T2: __m512d = m64x8_constant!(0.240226506959101195979507231);
const D512_T3: __m512d = m64x8_constant!(0.05550410866482166557484);
const D512_T4: __m512d = m64x8_constant!(0.00961812910759946061829085);
const D512_T5: __m512d = m64x8_constant!(0.0013333558146398846396);
const D512_T6: __m512d = m64x8_constant!(0.0001540353044975008196326);
const D512_T7: __m512d = m64x8_constant!(0.000015252733847608224);
const D512_T8: __m512d = m64x8_constant!(0.000001321543919937730177);
const D512_T9: __m512d = m64x8_constant!(0.00000010178055034703);
const D512_T10: __m512d = m64x8_constant!(0.000000007073075504998510);
const D512_T11: __m512d = m64x8_constant!(0.00000000044560630323);
const D512_POSITIVE_INFINITY: __m512d = m64x8_constant!(f64::INFINITY);
const D512_NAN: __m512d = m64x8_constant!(f64::NAN);
const D512_LOG2EF: __m512d = m64x8_constant!(1.4426950408889634);
const D512_MAGIC_LONG_DOUBLE_ADD: __m512d = m64x8_constant!(6755399441055744.0);
const D512_THIGH: __m512d = m64x8_constant!(709.0 * 1.4426950408889634);
const D512_TLOW: __m512d = m64x8_constant!(-709.0 * 1.4426950408889634);
const D512_ZERO: __m512d = m64x8_constant!(0.0);
const I512_ONE_THOUSAND_TWENTY_THREE: __m512i = m64x8_constant!(1023i64);