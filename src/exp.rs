use std::arch::x86_64::*;
use lazy_static::lazy_static;

lazy_static!{
    static ref T0: __m512d = unsafe { _mm512_set1_pd(1.0) };
    static ref T1: __m512d = unsafe {_mm512_set1_pd(0.6931471805599453087156032)};
    static ref T2: __m512d = unsafe {_mm512_set1_pd(0.240226506959101195979507231)};
    static ref T3: __m512d = unsafe {_mm512_set1_pd(0.05550410866482166557484)};
    static ref T4: __m512d = unsafe {_mm512_set1_pd(0.00961812910759946061829085)};
    static ref T5: __m512d = unsafe {_mm512_set1_pd(0.0013333558146398846396)};
    static ref T6: __m512d = unsafe {_mm512_set1_pd(0.0001540353044975008196326)};
    static ref T7: __m512d = unsafe {_mm512_set1_pd(0.000015252733847608224)};
    static ref T8: __m512d = unsafe {_mm512_set1_pd(0.000001321543919937730177)};
    static ref T9: __m512d = unsafe {_mm512_set1_pd(0.00000010178055034703)};
    static ref T10: __m512d = unsafe {_mm512_set1_pd(0.000000007073075504998510)};
    static ref T11: __m512d = unsafe {_mm512_set1_pd(0.00000000044560630323)};
    static ref HIGH: __m512d = unsafe {_mm512_set1_pd(709.0)};
    static ref POSITIVE_INFINITY: __m512d = unsafe {_mm512_set1_pd(f64::INFINITY)};
    static ref NAN: __m512d = unsafe {_mm512_set1_pd(f64::NAN)};
    static ref LOW: __m512d = unsafe {_mm512_set1_pd(-709.0)};
    static ref LOG2EF: __m512d = unsafe {_mm512_set1_pd(1.4426950408889634)};
    static ref INVERSE_LOG2EF: __m512d = unsafe {_mm512_set1_pd(0.693147180559945)};
    static ref ONE: __m512d = unsafe {_mm512_set1_pd(1.0)};
    static ref MAGIC_LONG_DOUBLE_ADD: __m512d = unsafe {_mm512_set1_pd(6755399441055744.0)};
    static ref INVE: __m512d = unsafe {_mm512_set1_pd(0.367879441171442321595523770161)};
    static ref THIGH: __m512d = unsafe {_mm512_set1_pd(709.0 * 1.4426950408889634)};
    static ref TLOW: __m512d = unsafe {_mm512_set1_pd(-709.0 * 1.4426950408889634)};
    static ref ZERO: __m512d = unsafe {_mm512_set1_pd(0.0)};
    static ref ONE_THOUSAND_TWENTY_THREE: __m512i = unsafe {_mm512_set1_epi64(1023)};
}


#[inline]
pub unsafe fn exp(x: &[f64], y: &mut [f64])
{
    const VSZ: usize = 8;
    let n = x.len() as i32;

    // if n < 8, then we handle the special case by creating a 8 element array to work with
    if n < VSZ as i32
    {
        // nothing for now
        return;
    }

    let mut i: usize = 0;

    // Calculates values in an unrolled manner if the number of values is large enough
    while (i as i32) < (n - 31)
    {
        expwo(x, y, i);
        i += VSZ;
        expwo(x, y, i);
        i += VSZ;
        expwo(x, y, i);
        i += VSZ;
        expwo(x, y, i);
        i += VSZ;
    }

    // Calculates the remaining sets of 8 values in a standard loop
    while (i as i32) < (n - 7)
    {
        expwo(x, y, i);
        i += VSZ;
    }

    // Cleans up any excess individual values (if n%4 != 0)
    if (i as i32) != n
    {
        i = (n as usize) - VSZ;
        expwo(x, y, i);
    }
}


pub unsafe fn expwo(x: &[f64], y: &mut [f64], offset: usize)
{
    let xx = _mm512_loadu_pd(((x.as_ptr() as usize) + offset * 8 ) as *const f64);
    let mut yy = _mm512_loadu_pd(((y.as_ptr() as usize) + offset * 8) as *const f64);
    exp512(&xx, &mut yy);
    _mm512_storeu_pd(((y.as_ptr() as usize) + offset* 8) as *mut f64, yy);
}


#[inline]
pub unsafe fn exp512(x: &__m512d, y: &mut __m512d)
{
    let xx = _mm512_mul_pd(*x, *LOG2EF);
    two(&xx, y);
}

#[inline]
pub unsafe fn two(x: &__m512d, y: &mut __m512d)
{
    // Checks if x is greater than the highest acceptable argument. Stores the information for later to
    // modify the result. If, for example, only x[1] > EXP_HIGH, then end[1] will be infinity, and the rest
    // zero. We add this to the result at the end, which will force y[1] to be infinity.
    let inf_mask = _mm512_cmple_pd_mask(*x, *THIGH);

    // Bound x by the maximum and minimum values this algorithm will handle.
    let mut xx = _mm512_max_pd(_mm512_min_pd(*x, *THIGH), *TLOW);

    // Avx.CompareNotEqual(x, x) is a hack to determine which values of x are NaN, since NaN is the only
    // value that doesn't equal itself. If any are NaN, we make the corresponding element of 'end' NaN, and
    // it acts like the infinity adjustment.
    let nan_mask = _mm512_cmpeq_pd_mask(*x, *x);

    let mut fx = _mm512_roundscale_pd(xx, _MM_FROUND_NEARBYINT);


    // This section gets a series approximation for exp(g) in (-0.5, 0.5) since that is g's range.
    xx = _mm512_sub_pd(xx, fx);
    *y = _mm512_fmadd_pd(*T11, xx, *T10);
    *y = _mm512_fmadd_pd(*y, xx, *T9);
    *y = _mm512_fmadd_pd(*y, xx, *T8);
    *y = _mm512_fmadd_pd(*y, xx, *T7);
    *y = _mm512_fmadd_pd(*y, xx, *T6);
    *y = _mm512_fmadd_pd(*y, xx, *T5);
    *y = _mm512_fmadd_pd(*y, xx, *T4);
    *y = _mm512_fmadd_pd(*y, xx, *T3);
    *y = _mm512_fmadd_pd(*y, xx, *T2);
    *y = _mm512_fmadd_pd(*y, xx, *T1);
    *y = _mm512_fmadd_pd(*y, xx, *T0);

    // Converts n to 2^n. There is no Avx2.ConvertToVector256Int64(fx) intrinsic, so we convert to int32's,
    // since the exponent of a double will never be more than a max int32, then from int to long.
    fx = _mm512_add_pd(fx, *MAGIC_LONG_DOUBLE_ADD);
    fx = _mm512_castsi512_pd(_mm512_slli_epi64(_mm512_add_epi64(_mm512_castpd_si512(fx), *ONE_THOUSAND_TWENTY_THREE), 52));


    // Combines the two exponentials and the end adjustments into the result.
    *y = _mm512_mul_pd(*y, fx);

    *y = _mm512_mask_add_pd(*POSITIVE_INFINITY, inf_mask, *ZERO, *y);
    *y = _mm512_mask_add_pd(*NAN, nan_mask, *y, *ZERO);
}
