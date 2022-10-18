use std::arch::x86_64::*;
use super::*;

#[inline]
pub fn sin(x: &[f64], y: &mut [f64])
{
    unsafe{
        sinu(x, y);
    }
}

#[inline]
pub fn tan(x: &[f64], y: &mut [f64])
{
    unsafe{
        tanu(x, y);
    }
}

#[inline]
pub fn atan(x: &[f64], y: &mut [f64])
{
    unsafe{
        atanu(x, y);
    }
}

unroll_fn!(sinu, sin_with_offset, 8, f64);
unroll_fn!(tanu, tan_with_offset, 8, f64);
unroll_fn!(atanu, atan_with_offset, 8, f64);

#[target_feature(enable ="avx512f")]
unsafe fn sin_with_offset(x: &[f64], y: &mut [f64], offset: usize)
{
    let xx = _mm512_loadu_pd(&x[offset] as *const f64);
    let mut yy = _mm512_loadu_pd(&y[offset] as *const f64);
    sin_intr(&xx, &mut yy);
    _mm512_storeu_pd(&mut y[offset] as *mut f64, yy);
}

#[target_feature(enable ="avx512f")]
pub unsafe fn sin_intr(x: &__m512d, y: &mut __m512d)
{
    // Since sin() is periodic around 2pi, this converts x into the range of [0, 2pi]
    let mut xt = _mm512_sub_pd(*x, _mm512_mul_pd(D512_TWOPI, _mm512_roundscale_pd(_mm512_mul_pd(*x, D512_ONE_OVER_TWOPI), _MM_FROUND_TO_NEG_INF)));

    // Since sin() in [0, 2pi] is an odd function around pi, this converts the range to [0, pi], then stores whether
    // or not the result needs to be negated in negend.
    let negend_mask = _mm512_cmp_pd_mask(xt, D512_PI, _CMP_GT_OS);
    xt = _mm512_mask_sub_pd(xt, negend_mask, xt, D512_PI);

    // Since sin() on [0, pi] is an even function around pi/2, this "folds" the range into [0, pi/2]. I.e. 3pi/5 becomes 2pi/5.
    xt = _mm512_sub_pd(D512_HALFPI, _mm512_abs_pd(_mm512_sub_pd(xt, D512_HALFPI)));

    let xsq = _mm512_mul_pd(xt, xt);

    // This is an odd-only Taylor series approximation of sin() on [0, pi/2]. 
    let mut yy = _mm512_fmadd_pd(D512_P15, xsq, D512_P13);
    yy = _mm512_fmadd_pd(yy, xsq, D512_P11);
    yy = _mm512_fmadd_pd(yy, xsq, D512_P9);
    yy = _mm512_fmadd_pd(yy, xsq, D512_P7);
    yy = _mm512_fmadd_pd(yy, xsq, D512_P5);
    yy = _mm512_fmadd_pd(yy, xsq, D512_P3);
    yy = _mm512_fmadd_pd(yy, xsq, D512_ONE);
    yy = _mm512_mul_pd(yy, xt);
    
    yy = _mm512_mask_blend_pd(_mm512_cmpeq_pd_mask(*x, *x), D512_NAN, yy);
    *y = _mm512_mask_mul_pd(yy, negend_mask, yy, D512_NEGONE);
}

#[target_feature(enable ="avx512f")]
unsafe fn sin_in_zero_to_quarter_pi(x: &__m512d, y: &mut __m512d)
{
    let xsq = _mm512_mul_pd(*x, *x);

    // This is an odd-only Taylor series approximation of sin() on [0, pi/4]. 
    *y = _mm512_fmadd_pd(D512_SQP13, xsq, D512_SQP11);
    *y = _mm512_fmadd_pd(*y, xsq, D512_SQP9);
    *y = _mm512_fmadd_pd(*y, xsq, D512_SQP7);
    *y = _mm512_fmadd_pd(*y, xsq, D512_SQP5);
    *y = _mm512_fmadd_pd(*y, xsq, D512_SQP3);
    *y = _mm512_fmadd_pd(*y, xsq, D512_ONE);
    *y = _mm512_mul_pd(*y, *x);
}

#[target_feature(enable ="avx512f")]
unsafe fn tan_with_offset(x: &[f64], y: &mut [f64], offset: usize)
{
    let xx = _mm512_loadu_pd(&x[offset] as *const f64);
    let mut yy = _mm512_loadu_pd(&y[offset] as *const f64);
    tan_intr(&xx, &mut yy);
    _mm512_storeu_pd(&mut y[offset] as *mut f64, yy);
}


#[target_feature(enable ="avx512f")]
pub unsafe fn tan_intr(x: &__m512d, y: &mut __m512d)
{
    // Calculation:
    //     Move to range [0, Pi] with no adjustments
    //     Use oddness around Pi/2 to make range [0, Pi/2] 
    //     do_inverse_mask = avx.gt(Pi/4)
    //     do_not_inverse_mask = avx.lte(Pi / 4)
    //     mirror around Pi/4
    //     calculate tan(x) = sin(x) / sqrt(1-sin(x)^2)
    //     y = and(do_inverse, 1/y) + and(no_inverse, y)

    // Since tan() is periodic around pi, this converts x into the range of [0, pi]
    let mut xt = _mm512_sub_pd(*x, _mm512_mul_pd(D512_PI, _mm512_roundscale_pd(_mm512_mul_pd(*x, D512_ONE_OVER_PI), _MM_FROUND_TO_NEG_INF)));

    // Since tan() in [0, pi] is an odd function around pi/2, this converts the range to [0, pi/2], then stores whether
    // or not the result needs to be negated in negend.
    let negend_mask = _mm512_cmp_pd_mask(xt, D512_HALFPI, _CMP_GT_OS);
    xt = _mm512_mask_add_pd(xt, negend_mask, _mm512_mul_pd(D512_NEGATIVE_TWO, _mm512_sub_pd(xt, D512_HALFPI)), xt);

    // Since tan() on [0, pi/2] is an inversed function around pi/4, this "folds" the range into [0, pi/4]. I.e. 3pi/10 becomes 2pi/10.
    let do_inv_mask = _mm512_cmp_pd_mask(xt, D512_QUARTERPI, _CMP_GT_OS);
    xt = _mm512_sub_pd(D512_QUARTERPI, _mm512_abs_pd(_mm512_sub_pd(xt, D512_QUARTERPI)));

    // tan(x) = sin(x) / sqrt(1-sin(x)^2)
    let mut xx = D512_CT11;
    sin_in_zero_to_quarter_pi(&xt, &mut xx);

    let xsq = _mm512_mul_pd(xt, xt);

    // This is an odd-only Taylor series approximation of tan() on [0, 0.07]. 
    *y = _mm512_fmadd_pd(D512_CT11, xsq, D512_CT9);
    *y = _mm512_fmadd_pd(*y, xsq, D512_CT7);
    *y = _mm512_fmadd_pd(*y, xsq, D512_CT5);
    *y = _mm512_fmadd_pd(*y, xsq, D512_CT3);
    *y = _mm512_fmadd_pd(*y, xsq, D512_CT1);
    *y = _mm512_mul_pd(*y, xt);

    xt = _mm512_sqrt_pd(_mm512_sub_pd(D512_ONE, _mm512_mul_pd(xx, xx)));

    xx = _mm512_mask_blend_pd(do_inv_mask, _mm512_div_pd(xx, xt), _mm512_div_pd(xt, xx));
    *y = _mm512_mask_blend_pd(do_inv_mask, *y, _mm512_div_pd(D512_ONE, *y));
    *y = _mm512_mask_blend_pd(_mm512_cmple_pd_mask(xt, D512_SMALLCONDITION), xx, *y);

    *y = _mm512_mask_mul_pd(*y, negend_mask, D512_NEGONE, *y);
}

#[target_feature(enable ="avx512f")]
unsafe fn atan_with_offset(x: &[f64], y: &mut [f64], offset: usize)
{
    let xx = _mm512_loadu_pd(&x[offset] as *const f64);
    let mut yy = _mm512_loadu_pd(&y[offset] as *const f64);
    atan_intr(&xx, &mut yy);
    _mm512_storeu_pd(&mut y[offset] as *mut f64, yy);
}

#[target_feature(enable ="avx512f")]
pub unsafe fn atan_intr(x: &__m512d, y: &mut __m512d)
{
    // Idea taken from https://github.com/avrdudes/avr-libc/blob/main/libm/fplib/atan.S

    //  Algorithm:
    //  if (x < 0)
    //      return -atan(-x)
    //  else if (x > 1)
    //      return Pi/2 - atanf(1/x)
    //  else
    //      return x * (1 - C1 * x**2 + ... + CN * x**2N)

    let lt_zero_mask = _mm512_cmple_pd_mask(*x, D512_ZERO);
    let mut xx = _mm512_mask_mul_pd(*x, lt_zero_mask, D512_NEGONE, *x);
    let gt_one_mask = !_mm512_cmple_pd_mask(xx, D512_ONE);
    xx = _mm512_mask_div_pd(xx, gt_one_mask, D512_ONE, xx);
    xx = _mm512_min_pd(xx, D512_AT_BIG);
    xx = _mm512_sub_pd(xx, D512_HALF);

    // This is an odd-only Taylor series approximation of atan() on [0, 1]. 
    let mut yy = _mm512_fmadd_pd(D512_AT21, xx, D512_AT20);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT19);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT18);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT17);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT16);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT15);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT14);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT13);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT12);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT11);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT10);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT09);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT08);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT07);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT06);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT05);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT04);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT03);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT02);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT01);
    yy = _mm512_fmadd_pd(yy, xx, D512_AT00);

    // unwind the adjustments
    yy = _mm512_mask_sub_pd(yy, gt_one_mask, D512_HALFPI, yy);
    yy = _mm512_mask_mul_pd(yy, lt_zero_mask, D512_NEGONE, yy);
    *y = _mm512_mask_blend_pd(_mm512_cmpeq_pd_mask(*x, *x), D512_NAN, yy);

}


const D512_TWOPI: __m512d = m64x8_constant!(2.0 * std::f64::consts::PI);
const D512_ONE_OVER_TWOPI: __m512d = m64x8_constant!(0.5 / std::f64::consts::PI);
const D512_ONE_OVER_PI: __m512d = m64x8_constant!(1.0 / std::f64::consts::PI);
const D512_PI: __m512d = m64x8_constant!(std::f64::consts::PI);
const D512_HALFPI: __m512d = m64x8_constant!(0.5 * std::f64::consts::PI);
const D512_NEGHALFPI: __m512d = m64x8_constant!(-std::f64::consts::PI * 0.5);
const D512_QUARTERPI: __m512d = m64x8_constant!(0.25 * std::f64::consts::PI);
const D512_THIRDPI: __m512d = m64x8_constant!(std::f64::consts::PI / 3.0);
const D512_SIN_OF_QUARTERPI: __m512d = m64x8_constant!(0.7071067811865475244008443621);
const D512_P3: __m512d = m64x8_constant!(-0.166666666666663509013977  );
const D512_P5: __m512d = m64x8_constant!(0.008333333333299304989001   );
const D512_P7: __m512d = m64x8_constant!(-0.00019841269828860068271   );
const D512_P9: __m512d = m64x8_constant!(0.00000275573170815073144    );
const D512_P11: __m512d = m64x8_constant!(-0.00000002505191090496049   );
const D512_P13: __m512d = m64x8_constant!(0.000000000160490521296459   );
const D512_P15: __m512d = m64x8_constant!(-0.0000000000007384998082865);
const D512_SQP3: __m512d = m64x8_constant!(-0.1666666666666663969165095 );
const D512_SQP5: __m512d = m64x8_constant!(0.008333333333324419158220   );
const D512_SQP7: __m512d = m64x8_constant!(-0.00019841269831470328245   );
const D512_SQP9: __m512d = m64x8_constant!(0.0000027557314284120030     );
const D512_SQP11: __m512d = m64x8_constant!(-0.0000000250508528135474    );
const D512_SQP13: __m512d = m64x8_constant!(0.0000000001590238118466);
const D512_AT00: __m512d = m64x8_constant!(0.46364760900080612191885619);
const D512_AT01: __m512d = m64x8_constant!(0.8000000000000026556883);
const D512_AT02: __m512d = m64x8_constant!(-0.32000000000002407003032);
const D512_AT03: __m512d = m64x8_constant!(-0.04266666666770185722);
const D512_AT04: __m512d = m64x8_constant!(0.15360000000380523102);
const D512_AT05: __m512d = m64x8_constant!(-0.077823999897320728);
const D512_AT06: __m512d = m64x8_constant!(-0.0300373335672330673);
const D512_AT07: __m512d = m64x8_constant!(0.0650678809721744);
const D512_AT08: __m512d = m64x8_constant!(-0.02752511237599674);
const D512_AT09: __m512d = m64x8_constant!(-0.020913143751996);
const D512_AT10: __m512d = m64x8_constant!(0.0326734767157460);
const D512_AT11: __m512d = m64x8_constant!(-0.01007587713435);
const D512_AT12: __m512d = m64x8_constant!(-0.014392869846603);
const D512_AT13: __m512d = m64x8_constant!(0.0174687992811);
const D512_AT14: __m512d = m64x8_constant!(-0.00310943305922);
const D512_AT15: __m512d = m64x8_constant!(-0.009919186806);
const D512_AT16: __m512d = m64x8_constant!(0.00961864834622);
const D512_AT17: __m512d = m64x8_constant!(0.000313483966);
const D512_AT18: __m512d = m64x8_constant!(-0.0070646973307);
const D512_AT19: __m512d = m64x8_constant!(0.00363977136);
const D512_AT20: __m512d = m64x8_constant!(0.0022675623613);
const D512_AT21: __m512d = m64x8_constant!(-0.00207949497);
const D512_T23: __m512d = m64x8_constant!(-0.00015721067265618978);
const D512_T25: __m512d = m64x8_constant!(9.864578277638557E-05);
const D512_CT1: __m512d = m64x8_constant!(1.0);
const D512_CT3: __m512d = m64x8_constant!(0.3333333333333346619643685131);
const D512_CT5: __m512d = m64x8_constant!(0.1333333333236799972803215674);
const D512_CT7: __m512d = m64x8_constant!(0.0539682703825024279957999835);
const D512_CT9: __m512d = m64x8_constant!(0.0218602603709103339870063369);
const D512_CT11: __m512d = m64x8_constant!(0.0104473875384802020842874186);
const D512_SMALLCONDITION: __m512d = m64x8_constant!(0.07);
const D512_ONE: __m512d = m64x8_constant!(1.0);
const D512_NEGONE: __m512d = m64x8_constant!(-1.0);
const D512_NEGATIVE_TWO: __m512d = m64x8_constant!(-2.0);
const D512_NEGHALF: __m512d = m64x8_constant!(-0.5);
const D512_HALF: __m512d = m64x8_constant!(0.5);
const D512_ZERO: __m512d = m64x8_constant!(0.0);
const D512_NAN: __m512d = m64x8_constant!(f64::NAN);
const D512_AT_BIG: __m512d = m64x8_constant!(1e10);