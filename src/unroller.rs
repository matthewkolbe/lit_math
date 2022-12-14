
// $name   -- a canonoical identifier of the function being created (e.g. exp for exponential)
// $fun    -- the avx implementation with an avx512 signature (where x is type __m512, and y (the output) is a &mut __m512)
// $load   -- the conversion function from a pointer to a $simdty
// $store  -- the conversion function from a $simdty to a pointer
// $simdty -- the SIMD type we're working on (e.g. __m512d). Must match $numty
// $numty  -- the funamental data type (e.g. f64, f32, u16)
//
// TODO: $numty is implied by $simdty. There has to be a way to get this input given $simdty.
//
// BIG TODO: this creates x -> y functions. Expand the macro to do (x0, x1) -> y functions, and 
// so on for higher dimensions. 
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
            pub fn [<$name _par>](x: &[$numty], y: &mut [$numty])
            {
                unsafe{
                    [<$name _paru>](x, y);
                }
            }

            attr_helper!($simdty, unsafe fn [<$name u>](x: &[$numty], y: &mut [$numty]) {
                let nn = x.len();
                let n = nn as i32; 
                assert_eq!(nn, y.len());
                const VSZ: i32 = lane_size!($simdty);
                const VSZU: usize = lane_size!($simdty);
                const VSZU2: usize = 2 * VSZU;
                const VSZU3: usize = 3 * VSZU;
                const VSZ4: i32 = 4 * VSZ;

                if n < VSZ as i32
                {
                    let mut xa = [0.0; VSZU];
                    let mut ya = [0.0; VSZU];
                    for i in 0..nn {
                        xa[i] = x[i];
                    }
                    
                    let xx = $load(xa.as_ptr());
                    let mut yy = $load(ya.as_mut_ptr());

                    $fun(&xx, &mut yy);
                    $store(ya.as_mut_ptr(), yy);

                    for i in 0..nn {
                        y[i] = ya[i];
                    }
                    return;
                }

                let mut xx: $simdty;
                let mut yy: $simdty;
                let mut i: usize = 0;
                let xptr = x.as_ptr();
                let yptr = y.as_mut_ptr();

                if n >= VSZ4
                {
                    let mut xx1: $simdty;
                    let mut yy1: $simdty;
                    let mut xx2: $simdty;
                    let mut yy2: $simdty;
                    let mut xx3: $simdty;
                    let mut yy3: $simdty;

                    while (i as i32) <= (n - VSZ4)
                    {
                        xx = $load(xptr.add(i));
                        yy = $load(yptr.add(i));
                        xx1 = $load(xptr.add(i+VSZU));
                        yy1 = $load(yptr.add(i+VSZU));
                        xx2 = $load(xptr.add(i+VSZU2));
                        yy2 = $load(yptr.add(i+VSZU2));
                        xx3 = $load(xptr.add(i+VSZU3));
                        yy3 = $load(yptr.add(i+VSZU3));

                        $fun(&xx, &mut yy);
                        $fun(&xx1, &mut yy1);
                        $fun(&xx2, &mut yy2);
                        $fun(&xx3, &mut yy3);

                        $store(yptr.add(i), yy);
                        i += VSZU;
                        $store(yptr.add(i), yy1);
                        i += VSZU;
                        $store(yptr.add(i), yy2);
                        i += VSZU;
                        $store(yptr.add(i), yy3);
                        i += VSZU;
                    }
                }

                while (i as i32) <= (n - VSZ)
                {
                    xx = $load(xptr.add(i));
                    yy = $load(yptr.add(i));
                    $fun(&xx, &mut yy);
                    $store(yptr.add(i), yy);
                    i += VSZU;
                }

                if i != nn
                {
                    i = nn - VSZU;
                    xx = $load(xptr.add(i));
                    yy = $load(yptr.add(i));
                    $fun(&xx, &mut yy);
                    $store(yptr.add(i), yy);
                }
            });

            #[inline]
            unsafe fn [<$name  _paru>](x: &[$numty], y: &mut [$numty]) {
                use rayon::prelude::*;
                let chunk: usize = x.len() / 32;

                y.par_chunks_mut(chunk).enumerate().for_each(|(index, slice)|  $name(&x[(index*chunk)..(index*chunk+slice.len())], slice) );
            }
        }
    }
}

#[macro_export]
macro_rules! lane_size {
    (__m256d) => { 4 } ;
    (__m512d) => { 8 };
    (__m256s) => { 8 } ;
    (__m512s) => { 16 };
}

#[macro_export]
macro_rules! attr_helper {
    (__m512d, $function:item) => {
        #[inline]
        #[target_feature(enable ="avx512f,avx512dq,avx512vl,avx512vpopcntdq,avx512vpclmulqdq,avx512cd,avx512bw")]
        $function
    };
    (__m256d, $function:item) => {
        #[inline]
        #[target_feature(enable ="avx2")]
        #[target_feature(enable ="avx")]
        #[target_feature(enable ="fma")]
        $function
    };
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
