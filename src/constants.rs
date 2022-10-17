
// constant storage was inspired from Maximilian Hofmann in 
// https://github.com/almost-split/bogograd/blob/11602b513245cfc57ba0bfaa08a14705c6c2c151/src/linalg/blocks.rs

#[macro_export]
macro_rules! m64x8_constant {
    ( $x:expr ) => {
        unsafe { std::mem::transmute::<_, _>(($x, $x, $x, $x, $x, $x, $x, $x)) }
    };
}