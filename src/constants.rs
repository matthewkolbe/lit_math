
// constant storage was inspired from Maximilian Hofmann in 
// https://github.com/almost-split/bogograd/blob/11602b513245cfc57ba0bfaa08a14705c6c2c151/src/linalg/blocks.rs
use core::arch::x86_64::*;
    
macro_rules! m64x8_constant {
    ( $x:expr ) => {
        unsafe { std::mem::transmute::<_, _>(($x, $x, $x, $x, $x, $x, $x, $x)) }
    };
}


pub mod exp
{
    use super::*;

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
    pub const I512_ONE_THOUSAND_TWENTY_THREE: __m512i = m64x8_constant!(1023i64);
}

pub mod normdist
{
    use super::*;

    pub const D512ONE: __m512d = m64x8_constant!(1.0);
    pub const D512NEGONE: __m512d = m64x8_constant!(-1.0);
    pub const D512SQRT2: __m512d = m64x8_constant!(1.4142135623730950488);
    pub const D512HALF: __m512d = m64x8_constant!(0.5);
    pub const D512NEGATIVE_ZERO: __m512d = m64x8_constant!(-0.0);
    pub const D512ZERO: __m512d = m64x8_constant!(0.0);
    pub const D512ONE_OVER_PI: __m512d = m64x8_constant!(1.0/ std::f64::consts::PI);
    pub const D512E1: __m512d = m64x8_constant!(-0.17916959767319535  );
    pub const D512E2: __m512d = m64x8_constant!(-0.18542742267595866  );
    pub const D512E3: __m512d = m64x8_constant!(-0.13452915843880847  );
    pub const D512E4: __m512d = m64x8_constant!(-0.2784782860163457   );
    pub const D512E5: __m512d = m64x8_constant!(0.14246708134992647   );
    pub const D512E6: __m512d = m64x8_constant!(-0.41925118422030655  );
    pub const D512E7: __m512d = m64x8_constant!(0.03746722734143839   );
    pub const D512E8: __m512d = m64x8_constant!(0.3009176755909412    );
    pub const D512E9: __m512d = m64x8_constant!(-0.6169463046791893   );
    pub const D512E10: __m512d = m64x8_constant!(0.4759112697935371   );
    pub const D512E11: __m512d = m64x8_constant!(-0.1651167117117661  );
    pub const D512E12: __m512d = m64x8_constant!(0.022155411339686473);
}


pub mod log {
    use super::*;

    pub const D512_TWO_THIRDS: __m512d = m64x8_constant!(0.6666666666666666666);
    pub const D512_ONE: __m512d = m64x8_constant!(1.0);
    pub const D512_ZERO: __m512d = m64x8_constant!(0.0);
    pub const D512_NEGATIVE_INFINITY: __m512d = m64x8_constant!(f64::NEG_INFINITY);
    pub const D512_LN2: __m512d  = m64x8_constant!(0.6931471805599453094172321214581766);
    pub const D512_NAN: __m512d = m64x8_constant!(f64::NAN);

    pub const D512_T0: __m512d = m64x8_constant!(0.5849625007211562024634018319  );
    pub const D512_T1: __m512d = m64x8_constant!(2.88539008177795423263363741  );
    pub const D512_T3: __m512d = m64x8_constant!(0.96179669389977077508752 );
    pub const D512_T5: __m512d = m64x8_constant!(0.577078023612080068567 );
    pub const D512_T7: __m512d = m64x8_constant!(0.4121976972049074185   );
    pub const D512_T9: __m512d = m64x8_constant!(0.32065422990573868  );
    pub const D512_T11: __m512d = m64x8_constant!(0.2604711365240256 );
    pub const D512_T13: __m512d = m64x8_constant!(0.252528834803695 );

}