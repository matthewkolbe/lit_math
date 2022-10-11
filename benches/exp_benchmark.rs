#![feature(stdsimd)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lit_math::*;
use rand;
use core::arch::x86_64::*;

#[inline(always)]
pub fn naive(x: &[f64], y: &mut [f64]) {

    for i in 0..x.len() {
        y[i] = x[i] + y[i]; }
}


fn add_naive(c: &mut Criterion) {
    c.bench_function("add_naive", |b| {    
        let mut x= [0.0; 10000];
        let mut y = [0.0; 10000];
        for i in 0..x.len(){
            x[i] = rand::random();
            y[i] = rand::random();
        }

        b.iter(|| {
            naive(&x, &mut y);
            black_box(y[5]);
        })
    });
}

#[inline(always)]
pub unsafe fn simd(x: &[f64], y: &mut [f64]) {

    let mut ii: usize = 0;
    let nn = x.len();
    let yy = y.as_mut_ptr();
    let xx = x.as_ptr();

    while ii < nn
    {
        let mut yoff = _mm256_loadu_pd(yy.add(ii));
        let mut xoff = _mm256_loadu_pd(xx.add(ii));
        yoff = _mm256_add_pd(xoff, yoff);
        _mm256_storeu_pd(yy.add(ii),  yoff);
        ii+=4;

        yoff = _mm256_loadu_pd(yy.add(ii));
        xoff = _mm256_loadu_pd(xx.add(ii));
        yoff = _mm256_add_pd(xoff, yoff);
        _mm256_storeu_pd(yy.add(ii),  yoff);
        ii+=4;

        yoff = _mm256_loadu_pd(yy.add(ii));
        xoff = _mm256_loadu_pd(xx.add(ii));
        yoff = _mm256_add_pd(xoff, yoff);
        _mm256_storeu_pd(yy.add(ii),  yoff);
        ii+=4;

        yoff = _mm256_loadu_pd(yy.add(ii));
        xoff = _mm256_loadu_pd(xx.add(ii));
        yoff = _mm256_add_pd(xoff, yoff);
        _mm256_storeu_pd(yy.add(ii),  yoff);
        ii+=4;
    }

}

fn add_simd(c: &mut Criterion) {
    unsafe{
        c.bench_function("add_simd", |b| {    
            let mut x= [0.0; 10000];
            let mut y = [0.0; 10000];
            for i in 0..x.len(){
                x[i] = rand::random();
                y[i] = rand::random();
            }


            b.iter(|| {
                simd(&x, &mut y);
                black_box(y[5]);
            });
        });
    }
}

fn exps_naive(c: &mut Criterion) {
    c.bench_function("exps_naive", |b| {    
        let mut x= [0.0; 2048];
        let mut y = [0.0; 2048];
        for i in 0..x.len(){
            x[i] = rand::random();
        }

        b.iter(|| {
            for i in 0..x.len() {
                black_box(y[i] = f64::exp(x[i])); }
        });
    });
}

fn exps256(c: &mut Criterion) {
    unsafe{
        c.bench_function("exps256", |b| {    
            let x= [0.0; 2048];
            let mut y = [0.0; 2048];
            b.iter(|| {
                exp256(&x, &mut y);
                black_box(y[0]);
            });
        });
    }
}

fn exps512(c: &mut Criterion) {
    unsafe{
        c.bench_function("exps512", |b| {    
            let x = [0.0; 2048];
            let mut y = [0.0; 2048];
            b.iter(|| {
                exp512(&x, &mut y);
                black_box(y[0]);
            });
        });
    }
}


criterion_group!(benches, add_simd, add_naive, exps_naive, exps256, exps512);
criterion_main!(benches);