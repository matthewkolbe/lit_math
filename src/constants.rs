
pub mod expc
{
    //#![cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    use core::arch::x86_64::*;

    #[allow(unused_macros)]
    macro_rules! m64x4_constant {
        ( $x:expr ) => {
            unsafe { std::mem::transmute::<_, _>(($x, $x, $x, $x)) }
        };
    }

    #[allow(unused_macros)]
    macro_rules! m64x8_constant {
        ( $x:expr ) => {
            unsafe { std::mem::transmute::<_, _>(($x, $x, $x, $x, $x, $x, $x, $x)) }
        };
    }

    pub const D256_T0: __m256d = m64x4_constant!(1.0);
    pub const D256_T1: __m256d = m64x4_constant!(0.6931471805599453087156032);
    pub const D256_T2: __m256d = m64x4_constant!(0.240226506959101195979507231);
    pub const D256_T3: __m256d = m64x4_constant!(0.05550410866482166557484);
    pub const D256_T4: __m256d = m64x4_constant!(0.00961812910759946061829085);
    pub const D256_T5: __m256d = m64x4_constant!(0.0013333558146398846396);
    pub const D256_T6: __m256d = m64x4_constant!(0.0001540353044975008196326);
    pub const D256_T7: __m256d = m64x4_constant!(0.000015252733847608224);
    pub const D256_T8: __m256d = m64x4_constant!(0.000001321543919937730177);
    pub const D256_T9: __m256d = m64x4_constant!(0.00000010178055034703);
    pub const D256_T10: __m256d = m64x4_constant!(0.000000007073075504998510);
    pub const D256_T11: __m256d = m64x4_constant!(0.00000000044560630323);
    pub const D256_POSITIVE_INFINITY: __m256d = m64x4_constant!(f64::INFINITY);
    pub const D256_NAN: __m256d = m64x4_constant!(f64::NAN);
    pub const D256_LOG2EF: __m256d = m64x4_constant!(1.4426950408889634);
    pub const D256_MAGIC_LONG_DOUBLE_ADD: __m256d = m64x4_constant!(6755399441055744.0);
    pub const D256_THIGH: __m256d = m64x4_constant!(709.0 * 1.4426950408889634);
    pub const D256_TLOW: __m256d = m64x4_constant!(-709.0 * 1.4426950408889634);
    pub const D256_ZERO: __m256d = m64x4_constant!(0.0);
    pub const D256_ONE_THOUSAND_TWENTY_THREE: __m256i = m64x4_constant!(1023i64);

    pub const D512_T0: __m512d = m64x8_constant!(1.0);
    pub const D512_T1: __m512d = m64x8_constant!(0.6931471805599453087156032);
    pub const D512_T2: __m512d = m64x8_constant!(0.240226506959101195979507231);
    pub const D512_T3: __m512d = m64x8_constant!(0.05550410866482166557484);
    pub const D512_T4: __m512d = m64x8_constant!(0.00961812910759946061829085);
    pub const D512_T5: __m512d = m64x8_constant!(0.0013333558146398846396);
    pub const D512_T6: __m512d = m64x8_constant!(0.0001540353044975008196326);
    pub const D512_T7: __m512d = m64x8_constant!(0.000015252733847608224);
    pub const D512_T8: __m512d = m64x8_constant!(0.000001321543919937730177);
    pub const D512_T9: __m512d = m64x8_constant!(0.00000010178055034703);
    pub const D512_T10: __m512d = m64x8_constant!(0.000000007073075504998510);
    pub const D512_T11: __m512d = m64x8_constant!(0.00000000044560630323);
    pub const D512_POSITIVE_INFINITY: __m512d = m64x8_constant!(f64::INFINITY);
    pub const D512_NAN: __m512d = m64x8_constant!(f64::NAN);
    pub const D512_LOG2EF: __m512d = m64x8_constant!(1.4426950408889634);
    pub const D512_MAGIC_LONG_DOUBLE_ADD: __m512d = m64x8_constant!(6755399441055744.0);
    pub const D512_THIGH: __m512d = m64x8_constant!(709.0 * 1.4426950408889634);
    pub const D512_TLOW: __m512d = m64x8_constant!(-709.0 * 1.4426950408889634);
    pub const D512_ZERO: __m512d = m64x8_constant!(0.0);
    pub const D512_ONE_THOUSAND_TWENTY_THREE: __m512i = m64x8_constant!(1023i64);
}
