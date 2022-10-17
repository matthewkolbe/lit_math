#![feature(stdsimd)]
#![feature(avx512_target_feature)]

mod constants;
mod unroller;
mod exp;
mod log;
mod normdist;

pub use exp::*;
pub use log::*;
pub use normdist::*;