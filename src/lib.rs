#![feature(stdsimd)]
#![feature(avx512_target_feature)]
#![feature(target_feature_11)]

mod constants;
mod unroller;
mod exp;
mod log;
mod normdist;
mod trig;
mod linalg;
mod root;

pub use exp::*;
pub use log::*;
pub use normdist::*;
pub use trig::*;
pub use unroller::*;
pub use linalg::*;
pub use root::*;