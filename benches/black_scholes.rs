#![feature(stdsimd)]
#![feature(new_uninit)]
#![feature(avx512_target_feature)]

use std::arch::x86_64::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lit_math::*;
use rand;
use statrs::function::erf::erf;

#[inline]
fn norm(x: f64) -> f64
{
    0.5 * (1.0 + erf(x))
}

fn bs(x: f64, sigma: f64) -> f64
{
    // i actually have no clue whether or not this produces the right values. it's simply a test to
    // compare for benchmarks, and it takes roughly the same number of calculations. 
    
    let root_t = f64::sqrt(TTE);
    let d1 = (f64::ln(UL / x) +(RATE + 0.5 * sigma * sigma)*TTE) / (sigma * root_t);
    let d2 = d1 - sigma * root_t;
    norm(d1) * UL - norm(d2)*x * f64::exp(-RATE * TTE)
}

#[target_feature(enable ="avx512f")]
unsafe fn bs_intr(x: &__m512d, sigma: &__m512d, y: &mut __m512d)
{
    // i actually have no clue whether or not this produces the right values. it's simply a test to
    // compare for benchmarks, and it takes roughly the same number of calculations. 

    let sigma_root_t = _mm512_mul_pd(*sigma, _mm512_sqrt_pd(D512_TTE));
    let mut d1 = _mm512_ln_pd(_mm512_div_pd(*x, D512_UL));
    let mut d2 = _mm512_mul_pd(D512_TTE, _mm512_add_pd(D512_RATE, _mm512_mul_pd(D512_HALF, _mm512_mul_pd(*sigma, *sigma))));
    d1 = _mm512_add_pd(d1, d2);
    d1 = _mm512_div_pd(d1, sigma_root_t);
    d2 = _mm512_sub_pd(d1, sigma_root_t);

    *y = _mm512_mul_pd(_mm512_std_norm_cdf_pd(d1), D512_UL);
    *y = _mm512_sub_pd(*y, _mm512_mul_pd(_mm512_mul_pd(_mm512_powe_pd(_mm512_mul_pd(D512_NEGRATE, D512_TTE)), *x), _mm512_std_norm_cdf_pd(d2)));
}

#[inline]
pub fn blksv(x: &Vec<f64>, sigma: &Vec<f64>, y: &mut Vec<f64>)
{
    unsafe{
        blksvu(x, sigma, y);
    }
}

unroll_fn_2!(blksu, blksvu, bs_intr, 8, f64);


const TTE: f64  = 0.1;
const UL: f64  = 100.0;
const RATE: f64  = 0.04;

const D512_TTE: __m512d = m64x8_constant!(TTE);
const D512_UL: __m512d = m64x8_constant!(UL);
const D512_RATE: __m512d = m64x8_constant!(RATE);
const D512_HALF: __m512d = m64x8_constant!(0.5);
const D512_NEGRATE: __m512d = m64x8_constant!(-RATE);


const N: usize = 32;

pub fn bs_naive(c: &mut Criterion) {
    c.bench_function("bs_naive", |b| {   
        let mut x = Vec::new();
        let mut sigma = Vec::new();
        let mut y = Vec::new();
        for i in 0..N{
            x.push(rand::random());
            x[i] *= 200.0;
            sigma.push(rand::random());
            sigma[i] += 0.05;
            y.push(0.0);
        }

        b.iter(|| {
            for i in 0..x.len() {
                black_box(y[i] = bs(x[i], sigma[i])); }
        });
    });
}

pub fn bs512(c: &mut Criterion) {
    c.bench_function("bs_512", |b| {   
        let mut x = Vec::new();
        let mut sigma = Vec::new();
        let mut y = Vec::new();
        for i in 0..N{
            x.push(rand::random());
            x[i] *= 200.0;
            sigma.push(rand::random());
            sigma[i] += 0.05;
            y.push(0.0);
        }

        b.iter(|| {
            black_box(blksv(&x, &sigma, &mut y)); 
        });
    });
}
