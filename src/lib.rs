#![feature(stdsimd)]
#![feature(avx512_target_feature)]

mod constants;
mod unroller;
mod exp;
mod log;
mod normdist;
mod trig;

pub use exp::*;
pub use log::*;
pub use normdist::*;
pub use trig::*;
pub use unroller::*;