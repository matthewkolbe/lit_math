
#[macro_export]
macro_rules! unroll_fn {
    ($name:ident, $fun:expr, $vsz:literal, $numsz:ty) => {
        #[inline]
        unsafe fn $name(x: &[$numsz], y: &mut [$numsz]) {
            let n = x.len() as i32;

            if n < $vsz as i32
            {
                let mut xx = [0.0; $vsz];
                let mut yy = [0.0; $vsz];
                for i in 0..x.len() {
                    xx[i] = x[i];
                }
                
                $fun(&xx, &mut yy, 0);
                for i in 0..y.len() {
                    y[i] = yy[i];
                }
                return;
            }

            let mut i: usize = 0;
            while (i as i32) < (n - 4*$vsz - 1)
            {
                $fun(x, y, i);
                i += $vsz;
                $fun(x, y, i);
                i += $vsz;
                $fun(x, y, i);
                i += $vsz;
                $fun(x, y, i);
                i += $vsz;
            }

            while (i as i32) < (n - $vsz + 1)
            {
                $fun(x, y, i);
                i += $vsz;
            }

            if (i as i32) != n
            {
                i = (n as usize) - $vsz;
                $fun(x, y, i);
            }
        }
    };
}