
#[macro_export]
macro_rules! unroll_fn {
    ($name:ident, $vecname:ident, $fun:expr, $vsz:literal, $numsz:ty) => {
        #[target_feature(enable ="avx512f")]
        unsafe fn $name(x: &[$numsz], y: &mut [$numsz]) {
            let n = x.len() as i32;


            if n < $vsz as i32
            {
                let mut xa = [0.0; $vsz];
                let mut ya = [0.0; $vsz];
                for i in 0..x.len() {
                    xa[i] = x[i];
                }
                
                let xx = _mm512_loadu_pd(&xa[0] as *const $numsz);
                let mut yy = _mm512_loadu_pd(&ya[0] as *const $numsz);

                $fun(&xx, &mut yy);
                _mm512_storeu_pd(&mut ya[0] as *mut $numsz, yy);

                for i in 0..y.len() {
                    y[i] = ya[i];
                }
                return;
            }

            let mut xx: __m512d;
            let mut yy: __m512d;

            let mut i: usize = 0;
            while (i as i32) < (n - 4*$vsz - 1)
            {
                xx = _mm512_loadu_pd(&x[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx = _mm512_loadu_pd(&x[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx = _mm512_loadu_pd(&x[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx = _mm512_loadu_pd(&x[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;
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
            let n = x.len() as i32;


            if n < $vsz as i32
            {
                let mut xa = [0.0; $vsz];
                let mut ya = [0.0; $vsz];
                for i in 0..x.len() {
                    xa[i] = x[i];
                }
                
                let xx = _mm512_loadu_pd(&xa[0] as *const $numsz);
                let mut yy = _mm512_loadu_pd(&ya[0] as *const $numsz);

                $fun(&xx, &mut yy);
                _mm512_storeu_pd(&mut ya[0] as *mut $numsz, yy);

                for i in 0..y.len() {
                    y[i] = ya[i];
                }
                return;
            }

            let mut xx: __m512d;
            let mut yy: __m512d;

            let mut i: usize = 0;
            while (i as i32) < (n - 4*$vsz - 1)
            {
                xx = _mm512_loadu_pd(&x[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx = _mm512_loadu_pd(&x[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx = _mm512_loadu_pd(&x[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;

                xx = _mm512_loadu_pd(&x[i] as *const $numsz);
                yy = _mm512_loadu_pd(&y[i] as *const $numsz);
                $fun(&xx, &mut yy);
                _mm512_storeu_pd(&mut y[i] as *mut $numsz, yy);
                i += $vsz;
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
    }
}