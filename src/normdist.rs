use std::arch::x86_64::*;
use super::*;


#[inline]
pub fn erf(x: &[f64], y: &mut [f64])
{
    unsafe{
        erfu(x, y);
    }
}

unroll_fn!(erfu, erf_with_offset, 8, f64);


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
    let mut y = D512ZERO;
    erfintr(&x, &mut y);
    y
}

/// AVX-512 implementation of the ERF function.
#[target_feature(enable ="avx512f")]
unsafe fn erfintr(x: &__m512d, y: &mut __m512d)
{

    let le_mask = _mm512_cmple_pd_mask(*x, D512NEGATIVE_ZERO);
    let xx = _mm512_abs_pd(*x);

    let mut t = _mm512_fmadd_pd(D512ONE_OVER_PI, xx, D512ONE);
    t = _mm512_div_pd(D512ONE, t);

    let mut yy = _mm512_fmadd_pd(D512E12, t, D512E11);
    yy = _mm512_fmadd_pd(yy, t, D512E10);
    yy = _mm512_fmadd_pd(yy, t, D512E9);
    yy = _mm512_fmadd_pd(yy, t, D512E8);
    yy = _mm512_fmadd_pd(yy, t, D512E7);
    yy = _mm512_fmadd_pd(yy, t, D512E6);
    yy = _mm512_fmadd_pd(yy, t, D512E5);
    yy = _mm512_fmadd_pd(yy, t, D512E4);
    yy = _mm512_fmadd_pd(yy, t, D512E3);
    yy = _mm512_fmadd_pd(yy, t, D512E2);
    yy = _mm512_fmadd_pd(yy, t, D512E1);
    yy = _mm512_mul_pd(yy, t);

    let exsq = _mm512_mul_pd(_mm512_mul_pd(xx, D512NEGONE), xx);

    super::exp::expintr(&exsq, &mut t);

    yy = _mm512_mul_pd(yy, t);
    yy = _mm512_add_pd(D512ONE, yy);

    *y = _mm512_mask_blend_pd(le_mask, yy, _mm512_mul_pd(yy, D512NEGONE));
    
}


const D512ONE: __m512d = m64x8_constant!(1.0);
const D512NEGONE: __m512d = m64x8_constant!(-1.0);
const D512SQRT2: __m512d = m64x8_constant!(1.4142135623730950488);
const D512HALF: __m512d = m64x8_constant!(0.5);
const D512NEGATIVE_ZERO: __m512d = m64x8_constant!(-0.0);
const D512ZERO: __m512d = m64x8_constant!(0.0);
const D512ONE_OVER_PI: __m512d = m64x8_constant!(1.0/ std::f64::consts::PI);
const D512E1: __m512d = m64x8_constant!(-0.17916959767319535  );
const D512E2: __m512d = m64x8_constant!(-0.18542742267595866  );
const D512E3: __m512d = m64x8_constant!(-0.13452915843880847  );
const D512E4: __m512d = m64x8_constant!(-0.2784782860163457   );
const D512E5: __m512d = m64x8_constant!(0.14246708134992647   );
const D512E6: __m512d = m64x8_constant!(-0.41925118422030655  );
const D512E7: __m512d = m64x8_constant!(0.03746722734143839   );
const D512E8: __m512d = m64x8_constant!(0.3009176755909412    );
const D512E9: __m512d = m64x8_constant!(-0.6169463046791893   );
const D512E10: __m512d = m64x8_constant!(0.4759112697935371   );
const D512E11: __m512d = m64x8_constant!(-0.1651167117117661  );
const D512E12: __m512d = m64x8_constant!(0.022155411339686473 );

