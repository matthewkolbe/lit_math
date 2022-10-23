
#[macro_export]
macro_rules! unroll_fn {
    ($name:ident, $vecname:ident, $vecparname:ident,$fun:expr, $vsz:literal, $numsz:ty) => {

        #[target_feature(enable ="avx512f")]
        unsafe fn $name(x: &[$numsz], y: &mut [$numsz]) {
            let nn = x.len();
            let n = nn as i32; 
            assert_eq!(nn, y.len());

            if n < $vsz as i32
            {
                let mut xa = [0.0; $vsz];
                let mut ya = [0.0; $vsz];
                for i in 0..nn {
                    xa[i] = x[i];
                }
                
                let xx = _mm512_loadu_pd(&xa[0] as *const $numsz);
                let mut yy = _mm512_loadu_pd(&ya[0] as *const $numsz);

                $fun(&xx, &mut yy);
                _mm512_storeu_pd(&mut ya[0] as *mut $numsz, yy);

                for i in 0..nn {
                    y[i] = ya[i];
                }
                return;
            }

            let mut xx: __m512d;
            let mut yy: __m512d;
            let mut i: usize = 0;
            if(n > 31)
            {
                let mut xx1: __m512d;
                let mut yy1: __m512d;
                let mut xx2: __m512d;
                let mut yy2: __m512d;
                let mut xx3: __m512d;
                let mut yy3: __m512d;

                while (i as i32) < (n - 4*$vsz - 1)
                {
                    xx = _mm512_loadu_pd(&x[i] as *const $numsz);
                    yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                    xx1 = _mm512_loadu_pd(&x[i+$vsz] as *const $numsz);
                    yy1 = _mm512_loadu_pd(&y[i+$vsz] as *const $numsz);
                    xx2 = _mm512_loadu_pd(&x[i+2*$vsz] as *const $numsz);
                    yy2 = _mm512_loadu_pd(&y[i+2*$vsz] as *const $numsz);
                    xx3 = _mm512_loadu_pd(&x[i+3*$vsz] as *const $numsz);
                    yy3 = _mm512_loadu_pd(&y[i+3*$vsz] as *const $numsz);
                    $fun(&xx, &mut yy);
                    $fun(&xx1, &mut yy1);
                    $fun(&xx2, &mut yy2);
                    $fun(&xx3, &mut yy3);

                    _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                    i += $vsz;
                    _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy1);
                    i += $vsz;
                    _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy2);
                    i += $vsz;
                    _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy3);
                    i += $vsz;
                }
            }

            while (i as i32) < (n - $vsz + 1)
            {
                xx = _mm512_loadu_pd(&x[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;
            }

            if (i as i32) != n
            {
                i = (n as usize) - $vsz;
                xx = _mm512_loadu_pd(&x[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
            }
        }

        #[target_feature(enable ="avx512f")]
        unsafe fn $vecname(x: &Vec<$numsz>, y: &mut Vec<$numsz>) {
            let nn = x.len();
            let n = nn as i32; 
            assert_eq!(nn, y.len());

            if n < $vsz as i32
            {
                let mut xa = [0.0; $vsz];
                let mut ya = [0.0; $vsz];
                for i in 0..nn {
                    xa[i] = x[i];
                }
                
                let xx = _mm512_loadu_pd(&xa[0] as *const $numsz);
                let mut yy = _mm512_loadu_pd(&ya[0] as *const $numsz);

                $fun(&xx, &mut yy);
                _mm512_storeu_pd(&mut ya[0] as *mut $numsz, yy);

                for i in 0..nn {
                    y[i] = ya[i];
                }
                return;
            }

            let mut xx: __m512d;
            let mut yy: __m512d;
            let mut i: usize = 0;

            if(n > 4*$vsz)
            {
                let mut xx1: __m512d;
                let mut yy1: __m512d;
                let mut xx2: __m512d;
                let mut yy2: __m512d;
                let mut xx3: __m512d;
                let mut yy3: __m512d;

                while (i as i32) < (n - 4*$vsz - 1)
                {
                    xx = _mm512_loadu_pd(&x[i] as *const $numsz);
                    xx1 = _mm512_loadu_pd(&x[i+$vsz] as *const $numsz);
                    xx2 = _mm512_loadu_pd(&x[i+2*$vsz] as *const $numsz);
                    xx3 = _mm512_loadu_pd(&x[i+3*$vsz] as *const $numsz);
                    yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                    yy1 = _mm512_loadu_pd(&y[i+$vsz] as *const $numsz);
                    yy2 = _mm512_loadu_pd(&y[i+2*$vsz] as *const $numsz);
                    yy3 = _mm512_loadu_pd(&y[i+3*$vsz] as *const $numsz);
                    $fun(&xx, &mut yy);
                    $fun(&xx1, &mut yy1);
                    $fun(&xx2, &mut yy2);
                    $fun(&xx3, &mut yy3);

                    _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                    i += $vsz;
                    _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy1);
                    i += $vsz;
                    _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy2);
                    i += $vsz;
                    _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy3);
                    i += $vsz;
                }
            }

            while (i as i32) < (n - $vsz + 1)
            {
                xx = _mm512_loadu_pd(&x[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;
            }

            if (i as i32) != n
            {
                i = (n as usize) - $vsz;
                xx = _mm512_loadu_pd(&x[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
            }
        }

        #[inline]
        #[target_feature(enable ="avx512f")]
        unsafe fn $vecparname(x: &Vec<$numsz>, y: &mut Vec<$numsz>) {
            use rayon::prelude::*;
            let chunk: usize = x.len() / 32;

            y.par_chunks_mut(chunk).enumerate().for_each(|(index, slice)|  exp(&x[(index*chunk)..(index*chunk+slice.len())], slice) );
            
        }
    }
}

#[macro_export]
macro_rules! unroll_fn_2 {
    ($name:ident, $vecname:ident, $fun:expr, $vsz:literal, $numsz:ty) => {
        #[target_feature(enable ="avx512f")]
        unsafe fn $name(x0: &[$numsz], x1: &[$numsz], y: &mut [$numsz]) {
            let nn = x0.len();
            let n = nn as i32; 
            assert_eq!(nn, y.len());
            assert_eq!(nn, x1.len());

            if n < $vsz as i32
            {
                let mut xa0 = [0.0; $vsz];
                let mut xa1 = [0.0; $vsz];
                let mut ya = [0.0; $vsz];
                for i in 0..nn {
                    xa0[i] = x0[i];
                    xa1[i] = x1[i];
                }
                
                let xx0 = _mm512_loadu_pd(&xa0[0] as *const $numsz);
                let xx1 = _mm512_loadu_pd(&xa1[0] as *const $numsz);
                let mut yy = _mm512_loadu_pd(&ya[0] as *const $numsz);

                $fun(&xx0, &xx1, &mut yy);
                _mm512_storeu_pd(&mut ya[0] as *mut $numsz, yy);

                for i in 0..nn {
                    y[i] = ya[i];
                }
                return;
            }

            let mut xx0: __m512d;
            let mut xx1: __m512d;
            let mut yy: __m512d;

            let mut i: usize = 0;
            while (i as i32) < (n - 4*$vsz - 1)
            {
                xx0 = _mm512_loadu_pd(&x0[i] as *const $numsz);
                xx1 = _mm512_loadu_pd(&x1[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx0, &xx1, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx0 = _mm512_loadu_pd(&x0[i] as *const $numsz);
                xx1 = _mm512_loadu_pd(&x1[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx0, &xx1, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx0 = _mm512_loadu_pd(&x0[i] as *const $numsz);
                xx1 = _mm512_loadu_pd(&x1[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx0, &xx1, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx0 = _mm512_loadu_pd(&x0[i] as *const $numsz);
                xx1 = _mm512_loadu_pd(&x1[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx0, &xx1, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;
            }

            while (i as i32) < (n - $vsz + 1)
            {
                xx0 = _mm512_loadu_pd(&x0[i] as *const $numsz);
                xx1 = _mm512_loadu_pd(&x1[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx0, &xx1, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;
            }

            if (i as i32) != n
            {
                i = (n as usize) - $vsz;
                xx0 = _mm512_loadu_pd(&x0[i] as *const $numsz);
                xx1 = _mm512_loadu_pd(&x1[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx0, &xx1, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
            }
        }

        #[target_feature(enable ="avx512f")]
        unsafe fn $vecname(x0: &Vec<$numsz>, x1: &Vec<$numsz>, y: &mut Vec<$numsz>) {
            let nn = x0.len();
            let n = nn as i32; 
            assert_eq!(nn, y.len());
            assert_eq!(nn, x1.len());

            if n < $vsz as i32
            {
                let mut xa0 = [0.0; $vsz];
                let mut xa1 = [0.0; $vsz];
                let mut ya = [0.0; $vsz];
                for i in 0..nn {
                    xa0[i] = x0[i];
                    xa1[i] = x1[i];
                }
                
                let xx0 = _mm512_loadu_pd(&xa0[0] as *const $numsz);
                let xx1 = _mm512_loadu_pd(&xa1[0] as *const $numsz);
                let mut yy = _mm512_loadu_pd(&ya[0] as *const $numsz);

                $fun(&xx0, &xx1, &mut yy);
                _mm512_storeu_pd(&mut ya[0] as *mut $numsz, yy);

                for i in 0..nn {
                    y[i] = ya[i];
                }
                return;
            }

            let mut xx0: __m512d;
            let mut xx1: __m512d;
            let mut yy: __m512d;

            let mut i: usize = 0;
            while (i as i32) < (n - 4*$vsz - 1)
            {
                xx0 = _mm512_loadu_pd(&x0[i] as *const $numsz);
                xx1 = _mm512_loadu_pd(&x1[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx0, &xx1, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx0 = _mm512_loadu_pd(&x0[i] as *const $numsz);
                xx1 = _mm512_loadu_pd(&x1[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx0, &xx1, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx0 = _mm512_loadu_pd(&x0[i] as *const $numsz);
                xx1 = _mm512_loadu_pd(&x1[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx0, &xx1, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx0 = _mm512_loadu_pd(&x0[i] as *const $numsz);
                xx1 = _mm512_loadu_pd(&x1[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx0, &xx1, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;
            }

            while (i as i32) < (n - $vsz + 1)
            {
                xx0 = _mm512_loadu_pd(&x0[i] as *const $numsz);
                xx1 = _mm512_loadu_pd(&x1[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx0, &xx1, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;
            }

            if (i as i32) != n
            {
                i = (n as usize) - $vsz;
                xx0 = _mm512_loadu_pd(&x0[i] as *const $numsz);
                xx1 = _mm512_loadu_pd(&x1[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx0, &xx1, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
            }
        }
    }
}

#[macro_export]
macro_rules! unroll_fn_256 {
    ($name:ident, $vecname:ident, $fun:expr, $vsz:literal, $numsz:ty) => {
        
        #[inline]
        #[target_feature(enable ="avx2")]
        #[target_feature(enable ="avx")]
        #[target_feature(enable ="fma")]
        unsafe fn $name(x: &[$numsz], y: &mut [$numsz]) {
            let nn = x.len();
            let n = nn as i32; 
            assert_eq!(nn, y.len());

            if n < $vsz as i32
            {
                let mut xa = [0.0; $vsz];
                let mut ya = [0.0; $vsz];
                for i in 0..nn {
                    xa[i] = x[i];
                }
                
                let xx = _mm256_loadu_pd(&xa[0] as *const $numsz);
                let mut yy = _mm256_loadu_pd(&ya[0] as *const $numsz);

                $fun(&xx, &mut yy);
                _mm256_storeu_pd(&mut ya[0] as *mut $numsz, yy);

                for i in 0..nn {
                    y[i] = ya[i];
                }
                return;
            }

            let mut xx: __m256d;
            let mut yy: __m256d;

            let mut i: usize = 0;
            while (i as i32) < (n - 4*$vsz - 1)
            {
                xx = _mm256_loadu_pd(&x[i] as *const $numsz);
                yy = _mm256_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm256_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx = _mm256_loadu_pd(&x[i] as *const $numsz);
                yy = _mm256_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm256_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx = _mm256_loadu_pd(&x[i] as *const $numsz);
                yy = _mm256_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm256_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx = _mm256_loadu_pd(&x[i] as *const $numsz);
                yy = _mm256_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm256_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;
            }

            while (i as i32) < (n - $vsz + 1)
            {
                xx = _mm256_loadu_pd(&x[i] as *const $numsz);
                yy = _mm256_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm256_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;
            }

            if (i as i32) != n
            {
                i = (n as usize) - $vsz;
                xx = _mm256_loadu_pd(&x[i] as *const $numsz);
                yy = _mm256_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm256_storeu_pd(&mut y[i] as *mut $numsz, yy);
            }
        }

        #[inline]
        #[target_feature(enable ="avx2")]
        #[target_feature(enable ="avx")]
        #[target_feature(enable ="fma")]
        unsafe fn $vecname(x: &Vec<$numsz>, y: &mut Vec<$numsz>) {
            let nn = x.len();
            let n = nn as i32; 
            assert_eq!(nn, y.len());

            if n < $vsz as i32
            {
                let mut xa = [0.0; $vsz];
                let mut ya = [0.0; $vsz];
                for i in 0..nn {
                    xa[i] = x[i];
                }
                
                let xx = _mm256_loadu_pd(&xa[0] as *const $numsz);
                let mut yy = _mm256_loadu_pd(&ya[0] as *const $numsz);

                $fun(&xx, &mut yy);
                _mm256_storeu_pd(&mut ya[0] as *mut $numsz, yy);

                for i in 0..nn {
                    y[i] = ya[i];
                }
                return;
            }

            let mut xx: __m256d;
            let mut yy: __m256d;

            let mut i: usize = 0;
            while (i as i32) < (n - 4*$vsz - 1)
            {
                xx = _mm256_loadu_pd(&x[i] as *const $numsz);
                yy = _mm256_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm256_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx = _mm256_loadu_pd(&x[i] as *const $numsz);
                yy = _mm256_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm256_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx = _mm256_loadu_pd(&x[i] as *const $numsz);
                yy = _mm256_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm256_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx = _mm256_loadu_pd(&x[i] as *const $numsz);
                yy = _mm256_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm256_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;
            }

            while (i as i32) < (n - $vsz + 1)
            {
                xx = _mm256_loadu_pd(&x[i] as *const $numsz);
                yy = _mm256_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm256_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;
            }

            if (i as i32) != n
            {
                i = (n as usize) - $vsz;
                xx = _mm256_loadu_pd(&x[i] as *const $numsz);
                yy = _mm256_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm256_storeu_pd(&mut y[i] as *mut $numsz, yy);
            }
        }
    }
}
