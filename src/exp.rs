use std::arch::x86_64::*;
use lazy_static::lazy_static;

lazy_static!{
    static ref D512_T0: __m512d = unsafe { _mm512_set1_pd(1.0) };
    static ref D512_T1: __m512d = unsafe {_mm512_set1_pd(0.6931471805599453087156032)};
    static ref D512_T2: __m512d = unsafe {_mm512_set1_pd(0.240226506959101195979507231)};
    static ref D512_T3: __m512d = unsafe {_mm512_set1_pd(0.05550410866482166557484)};
    static ref D512_T4: __m512d = unsafe {_mm512_set1_pd(0.00961812910759946061829085)};
    static ref D512_T5: __m512d = unsafe {_mm512_set1_pd(0.0013333558146398846396)};
    static ref D512_T6: __m512d = unsafe {_mm512_set1_pd(0.0001540353044975008196326)};
    static ref D512_T7: __m512d = unsafe {_mm512_set1_pd(0.000015252733847608224)};
    static ref D512_T8: __m512d = unsafe {_mm512_set1_pd(0.000001321543919937730177)};
    static ref D512_T9: __m512d = unsafe {_mm512_set1_pd(0.00000010178055034703)};
    static ref D512_T10: __m512d = unsafe {_mm512_set1_pd(0.000000007073075504998510)};
    static ref D512_T11: __m512d = unsafe {_mm512_set1_pd(0.00000000044560630323)};
    static ref D512_HIGH: __m512d = unsafe {_mm512_set1_pd(709.0)};
    static ref D512_POSITIVE_INFINITY: __m512d = unsafe {_mm512_set1_pd(f64::INFINITY)};
    static ref D512_NAN: __m512d = unsafe {_mm512_set1_pd(f64::NAN)};
    static ref D512_LOW: __m512d = unsafe {_mm512_set1_pd(-709.0)};
    static ref D512_LOG2EF: __m512d = unsafe {_mm512_set1_pd(1.4426950408889634)};
    static ref D512_INVERSE_LOG2EF: __m512d = unsafe {_mm512_set1_pd(0.693147180559945)};
    static ref D512_ONE: __m512d = unsafe {_mm512_set1_pd(1.0)};
    static ref D512_MAGIC_LONG_DOUBLE_ADD: __m512d = unsafe {_mm512_set1_pd(6755399441055744.0)};
    static ref D512_INVE: __m512d = unsafe {_mm512_set1_pd(0.367879441171442321595523770161)};
    static ref D512_THIGH: __m512d = unsafe {_mm512_set1_pd(709.0 * 1.4426950408889634)};
    static ref D512_TLOW: __m512d = unsafe {_mm512_set1_pd(-709.0 * 1.4426950408889634)};
    static ref D512_ZERO: __m512d = unsafe {_mm512_set1_pd(0.0)};
    static ref D512_ONE_THOUSAND_TWENTY_THREE: __m512i = unsafe {_mm512_set1_epi64(1023)};

    static ref D256_T0: __m256d = unsafe { _mm256_set1_pd(1.0) };
    static ref D256_T1: __m256d = unsafe {_mm256_set1_pd(0.6931471805599453087156032)};
    static ref D256_T2: __m256d = unsafe {_mm256_set1_pd(0.240226506959101195979507231)};
    static ref D256_T3: __m256d = unsafe {_mm256_set1_pd(0.05550410866482166557484)};
    static ref D256_T4: __m256d = unsafe {_mm256_set1_pd(0.00961812910759946061829085)};
    static ref D256_T5: __m256d = unsafe {_mm256_set1_pd(0.0013333558146398846396)};
    static ref D256_T6: __m256d = unsafe {_mm256_set1_pd(0.0001540353044975008196326)};
    static ref D256_T7: __m256d = unsafe {_mm256_set1_pd(0.000015252733847608224)};
    static ref D256_T8: __m256d = unsafe {_mm256_set1_pd(0.000001321543919937730177)};
    static ref D256_T9: __m256d = unsafe {_mm256_set1_pd(0.00000010178055034703)};
    static ref D256_T10: __m256d = unsafe {_mm256_set1_pd(0.000000007073075504998510)};
    static ref D256_T11: __m256d = unsafe {_mm256_set1_pd(0.00000000044560630323)};
    static ref D256_HIGH: __m256d = unsafe {_mm256_set1_pd(709.0)};
    static ref D256_POSITIVE_INFINITY: __m256d = unsafe {_mm256_set1_pd(f64::INFINITY)};
    static ref D256_NAN: __m256d = unsafe {_mm256_set1_pd(f64::NAN)};
    static ref D256_LOW: __m256d = unsafe {_mm256_set1_pd(-709.0)};
    static ref D256_LOG2EF: __m256d = unsafe {_mm256_set1_pd(1.4426950408889634)};
    static ref D256_INVERSE_LOG2EF: __m256d = unsafe {_mm256_set1_pd(0.693147180559945)};
    static ref D256_ONE: __m256d = unsafe {_mm256_set1_pd(1.0)};
    static ref D256_MAGIC_LONG_DOUBLE_ADD: __m256d = unsafe {_mm256_set1_pd(6755399441055744.0)};
    static ref D256_INVE: __m256d = unsafe {_mm256_set1_pd(0.367879441171442321595523770161)};
    static ref D256_THIGH: __m256d = unsafe {_mm256_set1_pd(709.0 * 1.4426950408889634)};
    static ref D256_TLOW: __m256d = unsafe {_mm256_set1_pd(-709.0 * 1.4426950408889634)};
    static ref D256_ZERO: __m256d = unsafe {_mm256_set1_pd(0.0)};
    static ref D256_ONE_THOUSAND_TWENTY_THREE: __m256i = unsafe {_mm256_set1_epi64x(1023)};
}


#[inline]
pub unsafe fn exp512(x: &[f64], y: &mut [f64])
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
        expwo512(x, y, i);
        i += VSZ;
        expwo512(x, y, i);
        i += VSZ;
        expwo512(x, y, i);
        i += VSZ;
        expwo512(x, y, i);
        i += VSZ;
    }

    // Calculates the remaining sets of 8 values in a standard loop
    while (i as i32) < (n - 7)
    {
        expwo512(x, y, i);
        i += VSZ;
    }

    // Cleans up any excess individual values (if n%4 != 0)
    if (i as i32) != n
    {
        i = (n as usize) - VSZ;
        expwo512(x, y, i);
    }
}

#[inline]
pub unsafe fn exp256(x: &[f64], y: &mut [f64])
{
    const VSZ: isize = 4;
    let n = x.len() as isize;
    let xx = x.as_ptr();
    let yy = y.as_mut_ptr();

    // if n < 8, then we handle the special case by creating a 8 element array to work with
    if n < VSZ
    {
        // nothing for now
        return;
    }

    let mut i: isize = 0;

    // Calculates values in an unrolled manner if the number of values is large enough
    if n > 31
    {
        while i < (n - 31)
        {
            expwo256(xx, yy, i);
            i += VSZ;
            expwo256(xx, yy, i);
            i += VSZ;
            expwo256(xx, yy, i);
            i += VSZ;
            expwo256(xx, yy, i);
            i += VSZ;
        }
    }

    // Calculates the remaining sets of 8 values in a standard loop
    if n > 7
    {
        while i < (n - 7)
        {
            expwo256(xx, yy, i);
            i += VSZ;
        }
    }

    // Cleans up any excess individual values (if n%4 != 0)
    if i != n
    {
        i = n - VSZ;
        expwo256(xx, yy, i);
    }
}


#[inline]
pub unsafe fn expwo512(x: &[f64], y: &mut [f64], offset: usize)
{
    let xx = _mm512_loadu_pd(&x[offset] as *const f64);
    let mut yy = _mm512_loadu_pd(&y[offset] as *const f64);
    expo512(&xx, &mut yy);
    _mm512_storeu_pd(&mut y[offset] as *mut f64, yy);
}

#[inline]
pub unsafe fn expwo256(x: *const f64, y: *mut f64, offset: isize)
{
    let xx = _mm256_loadu_pd(x.offset(offset));
    let mut yy = _mm256_loadu_pd(y.offset(offset));
    expo256(&xx, &mut yy);
    _mm256_storeu_pd(y.offset(offset), yy);
}

#[inline]
pub unsafe fn expo512(x: &__m512d, y: &mut __m512d)
{
    let xx = _mm512_mul_pd(*x, *D512_LOG2EF);
    two512(&xx, y);
}

#[inline]
pub unsafe fn expo256(x: &__m256d, y: &mut __m256d)
{
    let xx = _mm256_mul_pd(*x, *D256_LOG2EF);
    two256(&xx, y);
}

#[inline]
pub unsafe fn two512(x: &__m512d, y: &mut __m512d)
{
    // Checks if x is greater than the highest acceptable argument. Stores the information for later to
    // modify the result. If, for example, only x[1] > EXP_HIGH, then end[1] will be infinity, and the rest
    // zero. We add this to the result at the end, which will force y[1] to be infinity.
    let inf_mask = _mm512_cmple_pd_mask(*x, *D512_THIGH);

    // Bound x by the maximum and minimum values this algorithm will handle.
    let mut xx = _mm512_max_pd(_mm512_min_pd(*x, *D512_THIGH), *D512_TLOW);

    // Avx.CompareNotEqual(x, x) is a hack to determine which values of x are NaN, since NaN is the only
    // value that doesn't equal itself. If any are NaN, we make the corresponding element of 'end' NaN, and
    // it acts like the infinity adjustment.
    let nan_mask = _mm512_cmpeq_pd_mask(*x, *x);

    let mut fx = _mm512_roundscale_pd(xx, _MM_FROUND_NEARBYINT);


    // This section gets a series approximation for exp(g) in (-0.5, 0.5) since that is g's range.
    xx = _mm512_sub_pd(xx, fx);
    *y = _mm512_fmadd_pd(*D512_T11, xx, *D512_T10);
    *y = _mm512_fmadd_pd(*y, xx, *D512_T9);
    *y = _mm512_fmadd_pd(*y, xx, *D512_T8);
    *y = _mm512_fmadd_pd(*y, xx, *D512_T7);
    *y = _mm512_fmadd_pd(*y, xx, *D512_T6);
    *y = _mm512_fmadd_pd(*y, xx, *D512_T5);
    *y = _mm512_fmadd_pd(*y, xx, *D512_T4);
    *y = _mm512_fmadd_pd(*y, xx, *D512_T3);
    *y = _mm512_fmadd_pd(*y, xx, *D512_T2);
    *y = _mm512_fmadd_pd(*y, xx, *D512_T1);
    *y = _mm512_fmadd_pd(*y, xx, *D512_T0);

    // Converts n to 2^n. There is no Avx2.ConvertToVector256Int64(fx) intrinsic, so we convert to int32's,
    // since the exponent of a double will never be more than a max int32, then from int to long.
    fx = _mm512_add_pd(fx, *D512_MAGIC_LONG_DOUBLE_ADD);
    fx = _mm512_castsi512_pd(_mm512_slli_epi64(_mm512_add_epi64(_mm512_castpd_si512(fx), *D512_ONE_THOUSAND_TWENTY_THREE), 52));


    // Combines the two exponentials and the end adjustments into the result.
    *y = _mm512_mul_pd(*y, fx);

    *y = _mm512_mask_add_pd(*D512_POSITIVE_INFINITY, inf_mask, *D512_ZERO, *y);
    *y = _mm512_mask_add_pd(*D512_NAN, nan_mask, *y, *D512_ZERO);
}

#[inline]
pub unsafe fn two256(x: &__m256d, y: &mut __m256d)
{
    // Checks if x is greater than the highest acceptable argument. Stores the information for later to
    // modify the result. If, for example, only x[1] > EXP_HIGH, then end[1] will be infinity, and the rest
    // zero. We add this to the result at the end, which will force y[1] to be infinity.
    let inf_mask = _mm256_cmp_pd_mask(*x, *D256_THIGH, 1);

    // Bound x by the maximum and minimum values this algorithm will handle.
    let mut xx = _mm256_max_pd(_mm256_min_pd(*x, *D256_THIGH), *D256_TLOW);

    // Avx.CompareNotEqual(x, x) is a hack to determine which values of x are NaN, since NaN is the only
    // value that doesn't equal itself. If any are NaN, we make the corresponding element of 'end' NaN, and
    // it acts like the infinity adjustment.
    let nan_mask = _mm256_cmp_pd_mask(*x, *x, 0);

    let mut fx = _mm256_roundscale_pd(xx, _MM_FROUND_NEARBYINT);


    // This section gets a series approximation for exp(g) in (-0.5, 0.5) since that is g's range.
    xx = _mm256_sub_pd(xx, fx);
    *y = _mm256_fmadd_pd(*D256_T11, xx, *D256_T10);
    *y = _mm256_fmadd_pd(*y, xx, *D256_T9);
    *y = _mm256_fmadd_pd(*y, xx, *D256_T8);
    *y = _mm256_fmadd_pd(*y, xx, *D256_T7);
    *y = _mm256_fmadd_pd(*y, xx, *D256_T6);
    *y = _mm256_fmadd_pd(*y, xx, *D256_T5);
    *y = _mm256_fmadd_pd(*y, xx, *D256_T4);
    *y = _mm256_fmadd_pd(*y, xx, *D256_T3);
    *y = _mm256_fmadd_pd(*y, xx, *D256_T2);
    *y = _mm256_fmadd_pd(*y, xx, *D256_T1);
    *y = _mm256_fmadd_pd(*y, xx, *D256_T0);

    // Converts n to 2^n. There is no Avx2.ConvertToVector256Int64(fx) intrinsic, so we convert to int32's,
    // since the exponent of a double will never be more than a max int32, then from int to long.
    fx = _mm256_add_pd(fx, *D256_MAGIC_LONG_DOUBLE_ADD);
    fx = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(_mm256_castpd_si256(fx), *D256_ONE_THOUSAND_TWENTY_THREE), 52));


    // Combines the two exponentials and the end adjustments into the result.
    *y = _mm256_mul_pd(*y, fx);

    *y = _mm256_mask_add_pd(*D256_POSITIVE_INFINITY, inf_mask, *D256_ZERO, *y);
    *y = _mm256_mask_add_pd(*D256_NAN, nan_mask, *y, *D256_ZERO);
}