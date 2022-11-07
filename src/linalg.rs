use std::arch::x86_64::*;
use super::*;

#[inline]
pub fn dot(x: &[f64], y: &mut [f64]) -> f64
{
    unsafe{
        
        let nn = x.len();
        const VSZ: usize = 8;
        let n = nn as i32; 
        assert_eq!(nn, y.len());

        if n < VSZ as i32
        {
            let mut r = 0.0;

            for ii in 0..nn {
                r += x[ii] * y[ii]; 
            }

            return r;
        }

        let mut xx: __m512d;
        let mut yy: __m512d;
        let mut rr = D512_ZERO;
        let mut i: usize = 0;
        if n > 31
        {
            let mut xx1: __m512d;
            let mut yy1: __m512d;
            let mut rr1: __m512d = D512_ZERO;
            let mut xx2: __m512d;
            let mut yy2: __m512d;
            let mut rr2: __m512d = D512_ZERO;
            let mut xx3: __m512d;
            let mut yy3: __m512d;
            let mut rr3: __m512d = D512_ZERO;

            while (i as i32) < (n - 31)
            {
                xx = _mm512_loadu_pd(&x[i] as *const f64);
                yy = _mm512_loadu_pd(&y[i] as *const f64);
                i += VSZ;
                xx1 = _mm512_loadu_pd(&x[i] as *const f64);
                yy1 = _mm512_loadu_pd(&y[i] as *const f64);
                i += VSZ;
                xx2 = _mm512_loadu_pd(&x[i] as *const f64);
                yy2 = _mm512_loadu_pd(&y[i] as *const f64);
                i += VSZ;
                xx3 = _mm512_loadu_pd(&x[i] as *const f64);
                yy3 = _mm512_loadu_pd(&y[i] as *const f64);
                i += VSZ;

                rr = _mm512_fmadd_pd(xx, yy, rr);
                rr1 = _mm512_fmadd_pd(xx1, yy1, rr1);
                rr2 = _mm512_fmadd_pd(xx2, yy2, rr2);
                rr3 = _mm512_fmadd_pd(xx3, yy3, rr3);
            }

            rr2 = _mm512_add_pd(rr2, rr3);
            rr = _mm512_add_pd(rr, rr1);
            rr = _mm512_add_pd(rr, rr2);
        }

        while (i as i32) < (n - 7)
        {
            xx = _mm512_loadu_pd(&x[i] as *const f64);
            yy = _mm512_loadu_pd(&y[i] as *const f64);
            i += VSZ;
            rr = _mm512_fmadd_pd(xx, yy, rr);
        }

        let mut r = _mm512_reduce_add_pd(rr);

        for ii in i..nn {
            r += x[ii] * y[ii]; 
        }

        r
    }
}



const D512_ZERO: __m512d = m64x8_constant!(0.0);