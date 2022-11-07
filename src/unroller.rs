
// $name   -- a canonoical identifier of the function being created (e.g. exp for exponential)
// $fun    -- the avx implementation with an avx512 signature (where x is type __m512, and y (the output) is a &mut __m512)
// $load   -- the conversion function from a pointer to a $simdty
// $store  -- the conversion function from a $simdty to a pointer
// $simdty -- the SIMD type we're working on (e.g. __m512d). Must match $numty
// $numty  -- the funamental data type (e.g. f64, f32, u16)
//
// TODO: $numty is implied by $simdty. There has to be a way to get this input given $simdty.
#[macro_export]
macro_rules! unroll_fn {
    ($name:ident, $fun:expr, $load:expr, $store:expr, $simdty:ty, $numty:ty) => {
        
        paste::paste! {

            #[inline]
            pub fn $name(x: &[$numty], y: &mut [$numty])
            {
                unsafe{
                    [<$name u>](x, y);
                }
            }

            #[inline]
            pub fn [<$name v>](x: &Vec<$numty>, y: &mut Vec<$numty>)
            {
                unsafe{
                    [<$name vu>](x, y);
                }
            }

            #[inline]
            pub fn [<$name _parv>](x: &Vec<$numty>, y: &mut Vec<$numty>)
            {
                unsafe{
                    [<$name _parvu>](x, y);
                }
            }

            #[target_feature(enable ="avx512f")]
            unsafe fn [<$name u>](x: &[$numty], y: &mut [$numty]) {
                let nn = x.len();
                let n = nn as i32; 
                assert_eq!(nn, y.len());
                const VSZ: i32 = lane_size!($simdty);
                const VSZU: usize = lane_size!($simdty);

                if n < VSZ as i32
                {
                    let mut xa = [0.0; VSZU];
                    let mut ya = [0.0; VSZU];
                    for i in 0..nn {
                        xa[i] = x[i];
                    }
                    
                    let xx = $load(&xa[0] as *const $numty);
                    let mut yy = $load(&ya[0] as *const $numty);

                    $fun(&xx, &mut yy);
                    _mm512_storeu_pd(&mut ya[0] as *mut $numty, yy);

                    for i in 0..nn {
                        y[i] = ya[i];
                    }
                    return;
                }

                let mut xx: $simdty;
                let mut yy: $simdty;
                let mut i: usize = 0;
                if n >= 4*VSZ
                {
                    let mut xx1: $simdty;
                    let mut yy1: $simdty;
                    let mut xx2: $simdty;
                    let mut yy2: $simdty;
                    let mut xx3: $simdty;
                    let mut yy3: $simdty;

                    while (i as i32) <= (n - 4*VSZ)
                    {
                        xx = $load(&x[i] as *const $numty);
                        yy = $load(&y[i] as *const $numty);
                        xx1 = $load(&x[i+VSZU] as *const $numty);
                        yy1 = $load(&y[i+VSZU] as *const $numty);
                        xx2 = $load(&x[i+2*VSZU] as *const $numty);
                        yy2 = $load(&y[i+2*VSZU] as *const $numty);
                        xx3 = $load(&x[i+3*VSZU] as *const $numty);
                        yy3 = $load(&y[i+3*VSZU] as *const $numty);
                        $fun(&xx, &mut yy);
                        $fun(&xx1, &mut yy1);
                        $fun(&xx2, &mut yy2);
                        $fun(&xx3, &mut yy3);

                        $store(&mut y[i] as *mut $numty, yy);
                        i += VSZU;
                        $store(&mut y[i] as *mut $numty, yy1);
                        i += VSZU;
                        $store(&mut y[i] as *mut $numty, yy2);
                        i += VSZU;
                        $store(&mut y[i] as *mut $numty, yy3);
                        i += VSZU;
                    }
                }

                while (i as i32) <= (n - VSZ)
                {
                    xx = $load(&x[i] as *const $numty);
                    yy = $load(&y[i] as *const $numty);
                    $fun(&xx, &mut yy);
                    $store(&mut y[i] as *mut $numty, yy);
                    i += VSZU;
                }

                if i != nn
                {
                    i = nn - VSZU;
                    xx = $load(&x[i] as *const $numty);
                    yy = $load(&y[i] as *const $numty);
                    $fun(&xx, &mut yy);
                    $store(&mut y[i] as *mut $numty, yy);
                }
            }

            #[target_feature(enable ="avx512f")]
            #[target_feature(enable ="avx512dq")]
            unsafe fn [<$name vu>](x: &Vec<$numty>, y: &mut Vec<$numty>) {
                let nn = x.len();
                let n = nn as i32; 
                assert_eq!(nn, y.len());
                const VSZ: i32 = lane_size!($simdty);
                const VSZU: usize = lane_size!($simdty);

                if n < VSZ as i32
                {
                    let mut xa = [0.0; VSZU];
                    let mut ya = [0.0; VSZU];
                    for i in 0..nn {
                        xa[i] = x[i];
                    }
                    
                    let xx = $load(&xa[0] as *const $numty);
                    let mut yy = $load(&ya[0] as *const $numty);

                    $fun(&xx, &mut yy);
                    $store(&mut ya[0] as *mut $numty, yy);

                    for i in 0..nn {
                        y[i] = ya[i];
                    }
                    return;
                }

                let mut xx: $simdty;
                let mut yy: $simdty;
                let mut i: usize = 0;

                if n >= 4*VSZ
                {
                    let mut xx1: $simdty;
                    let mut yy1: $simdty;
                    let mut xx2: $simdty;
                    let mut yy2: $simdty;
                    let mut xx3: $simdty;
                    let mut yy3: $simdty;

                    while (i as i32) <= (n - 4*VSZ)
                    {
                        xx = $load(&x[i] as *const $numty);
                        xx1 = $load(&x[i+VSZU] as *const $numty);
                        xx2 = $load(&x[i+2*VSZU] as *const $numty);
                        xx3 = $load(&x[i+3*VSZU] as *const $numty);
                        yy = $load(&y[i] as *const $numty);
                        yy1 = $load(&y[i+VSZU] as *const $numty);
                        yy2 = $load(&y[i+2*VSZU] as *const $numty);
                        yy3 = $load(&y[i+3*VSZU] as *const $numty);
                        $fun(&xx, &mut yy);
                        $fun(&xx1, &mut yy1);
                        $fun(&xx2, &mut yy2);
                        $fun(&xx3, &mut yy3);

                        $store(&mut y[i] as *mut $numty, yy);
                        i += VSZU;
                        $store(&mut y[i] as *mut $numty, yy1);
                        i += VSZU;
                        $store(&mut y[i] as *mut $numty, yy2);
                        i += VSZU;
                        $store(&mut y[i] as *mut $numty, yy3);
                        i += VSZU;
                    }
                }

                while (i as i32) <= (n - VSZ)
                {
                    xx = $load(&x[i] as *const $numty);
                    yy = $load(&y[i] as *const $numty);
                    $fun(&xx, &mut yy);
                    $store(&mut y[i] as *mut $numty, yy);
                    i += VSZU;
                }

                if i != nn
                {
                    i = nn - VSZU;
                    xx = $load(&x[i] as *const $numty);
                    yy = $load(&y[i] as *const $numty);
                    $fun(&xx, &mut yy);
                    $store(&mut y[i] as *mut $numty, yy);
                }
            }

            #[inline]
            #[target_feature(enable ="avx512f")]
            unsafe fn [<$name  _parvu>](x: &Vec<$numty>, y: &mut Vec<$numty>) {
                use rayon::prelude::*;
                let chunk: usize = x.len() / 32;

                y.par_chunks_mut(chunk).enumerate().for_each(|(index, slice)|  $name(&x[(index*chunk)..(index*chunk+slice.len())], slice) );
            }
        }
    }
}


#[macro_export]
macro_rules! lane_size {
    (__m256d) => { 4} ;
    (__m512d) => { 8 };
    (__m256s) => { 8} ;
    (__m512s) => { 16 };
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
macro_rules! unroll_fn_5 {
    ($name:ident, $vecname:ident, $fun:expr, $vsz:literal, $numsz:ty) => {
        #[target_feature(enable ="avx512f")]
        unsafe fn $name(x0: &[$numsz], x1: &[$numsz], x2: &[$numsz], x3: &[$numsz], x4: &[$numsz],  y: &mut [$numsz]) {
            let nn = x0.len();
            let n = nn as i32; 
            assert_eq!(nn, y.len());
            assert_eq!(nn, x1.len());

            if n < $vsz as i32
            {
                let mut xa0 = [0.0; $vsz];
                let mut xa1 = [0.0; $vsz];
                let mut xa2 = [0.0; $vsz];
                let mut xa3 = [0.0; $vsz];
                let mut xa4 = [0.0; $vsz];
                let mut ya = [0.0; $vsz];
                for i in 0..nn {
                    xa0[i] = x0[i];
                    xa1[i] = x1[i];
                    xa2[i] = x2[i];
                    xa3[i] = x3[i];
                    xa4[i] = x4[i];
                }
                
                let xx0 = _mm512_loadu_pd(&xa0[0] as *const $numsz);
                let xx1 = _mm512_loadu_pd(&xa1[0] as *const $numsz);
                let xx2 = _mm512_loadu_pd(&xa2[0] as *const $numsz);
                let xx3 = _mm512_loadu_pd(&xa3[0] as *const $numsz);
                let xx4 = _mm512_loadu_pd(&xa4[0] as *const $numsz);
                let mut yy = _mm512_loadu_pd(&ya[0] as *const $numsz);

                $fun(&xx0, &xx1, &xx2, &xx3, &xx4, &mut yy);
                _mm512_storeu_pd(&mut ya[0] as *mut $numsz, yy);

                for i in 0..nn {
                    y[i] = ya[i];
                }
                return;
            }

            let mut xx0: __m512d;
            let mut xx1: __m512d;
            let mut xx2: __m512d;
            let mut xx3: __m512d;
            let mut xx4: __m512d;
            let mut yy: __m512d;

            let mut i: usize = 0;
            while (i as i32) < (n - 4*$vsz - 1)
            {
                xx0 = _mm512_loadu_pd(&x0[i] as *const $numsz);
                xx1 = _mm512_loadu_pd(&x1[i] as *const $numsz);
                xx2 = _mm512_loadu_pd(&x2[i] as *const $numsz);
                xx3 = _mm512_loadu_pd(&x3[i] as *const $numsz);
                xx4 = _mm512_loadu_pd(&x4[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx0, &xx1, &xx2, &xx3, &xx4, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx0 = _mm512_loadu_pd(&x0[i] as *const $numsz);
                xx1 = _mm512_loadu_pd(&x1[i] as *const $numsz);
                xx2 = _mm512_loadu_pd(&x2[i] as *const $numsz);
                xx3 = _mm512_loadu_pd(&x3[i] as *const $numsz);
                xx4 = _mm512_loadu_pd(&x4[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx0, &xx1, &xx2, &xx3, &xx4, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx0 = _mm512_loadu_pd(&x0[i] as *const $numsz);
                xx1 = _mm512_loadu_pd(&x1[i] as *const $numsz);
                xx2 = _mm512_loadu_pd(&x2[i] as *const $numsz);
                xx3 = _mm512_loadu_pd(&x3[i] as *const $numsz);
                xx4 = _mm512_loadu_pd(&x4[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx0, &xx1, &xx2, &xx3, &xx4, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx0 = _mm512_loadu_pd(&x0[i] as *const $numsz);
                xx1 = _mm512_loadu_pd(&x1[i] as *const $numsz);
                xx2 = _mm512_loadu_pd(&x2[i] as *const $numsz);
                xx3 = _mm512_loadu_pd(&x3[i] as *const $numsz);
                xx4 = _mm512_loadu_pd(&x4[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx0, &xx1, &xx2, &xx3, &xx4, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;
            }

            while (i as i32) < (n - $vsz + 1)
            {
                xx0 = _mm512_loadu_pd(&x0[i] as *const $numsz);
                xx1 = _mm512_loadu_pd(&x1[i] as *const $numsz);
                xx2 = _mm512_loadu_pd(&x2[i] as *const $numsz);
                xx3 = _mm512_loadu_pd(&x3[i] as *const $numsz);
                xx4 = _mm512_loadu_pd(&x4[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx0, &xx1, &xx2, &xx3, &xx4, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;
            }

            if (i as i32) != n
            {
                i = (n as usize) - $vsz;
                xx0 = _mm512_loadu_pd(&x0[i] as *const $numsz);
                xx1 = _mm512_loadu_pd(&x1[i] as *const $numsz);
                xx2 = _mm512_loadu_pd(&x2[i] as *const $numsz);
                xx3 = _mm512_loadu_pd(&x3[i] as *const $numsz);
                xx4 = _mm512_loadu_pd(&x4[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx0, &xx1, &xx2, &xx3, &xx4, &mut yy);
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
