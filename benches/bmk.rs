#![feature(stdsimd)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lit_math::*;
use rand;


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

fn lns_naive(c: &mut Criterion) {
    c.bench_function("lns_naive", |b| {    
        let mut x= [0.0; 2048];
        let mut y = [0.0; 2048];
        for i in 0..x.len(){
            x[i] = rand::random();
            x[i] += 1.0;
        }

        b.iter(|| {
            for i in 0..x.len() {
                black_box(y[i] = f64::ln(x[i])); }
        });
    });
}

fn log2_naive(c: &mut Criterion) {
    c.bench_function("log2_naive", |b| {    
        let mut x= [0.0; 2048];
        let mut y = [0.0; 2048];
        for i in 0..x.len(){
            x[i] = rand::random();
            x[i] += 1.0;
        }

        b.iter(|| {
            for i in 0..x.len() {
                black_box(y[i] = f64::log2(x[i])); }
        });
    });
}

fn erfs_naive(c: &mut Criterion) {
    use statrs::function::erf::erf;

    c.bench_function("erfs_naive", |b| {    
        let mut x= [0.0; 2048];
        let mut y = [0.0; 2048];
        for i in 0..x.len(){
            x[i] = rand::random();
        }

        b.iter(|| {
            for i in 0..x.len() {
                black_box(y[i] = erf(x[i])); }
        });
    });
}

fn atans_naive(c: &mut Criterion) {
    c.bench_function("atans_naive", |b| {    
        let mut x= [0.0; 2048];
        let mut y = [0.0; 2048];
        for i in 0..x.len(){
            x[i] = rand::random();
            x[i] -= 0.5;
            x[i] *= 100.0;
        }

        b.iter(|| {
            for i in 0..x.len() {
                black_box(y[i] = f64::atan(x[i])); }
        });
    });
}

fn sins_naive(c: &mut Criterion) {
    c.bench_function("sins_naive", |b| {    
        let mut x= [0.0; 2048];
        let mut y = [0.0; 2048];
        for i in 0..x.len(){
            x[i] = rand::random();
            x[i] -= 0.5;
            x[i] *= 100.0;
        }

        b.iter(|| {
            for i in 0..x.len() {
                black_box(y[i] = f64::atan(x[i])); }
        });
    });
}


fn exps512(c: &mut Criterion) {
    c.bench_function("exps512", |b| {    
        let mut x = [0.0; 2048];
        let mut y = [0.0; 2048];
        for i in 0..x.len(){
            x[i] = rand::random();
        }

        b.iter(|| {
            exp(&x, &mut y);
            black_box(y[0]);
        });
    });
}

fn erfs512(c: &mut Criterion) {
    c.bench_function("erfs512", |b| {    
        let mut x = [0.0; 2048];
        let mut y = [0.0; 2048];
        for i in 0..x.len(){
            x[i] = rand::random();
        }

        b.iter(|| {
            erf(&x, &mut y);
            black_box(y[0]);
        });
    });
}

fn lns512(c: &mut Criterion) {
    c.bench_function("lns512", |b| {    
        let mut x = [0.0; 2048];
        let mut y = [0.0; 2048];
        for i in 0..x.len(){
            x[i] = rand::random();
            x[i] += 1.0;
        }

        b.iter(|| {
            ln(&x, &mut y);
            black_box(y[0]);
        });
    });
}

fn log2s512(c: &mut Criterion) {
    c.bench_function("log2s512", |b| {    
        let mut x = [0.0; 2048];
        let mut y = [0.0; 2048];
        for i in 0..x.len(){
            x[i] = rand::random();
            x[i] += 1.0;
        }

        b.iter(|| {
            log2(&x, &mut y);
            black_box(y[0]);
        });
    });
}

fn atans512(c: &mut Criterion) {
    c.bench_function("atans512", |b| {    
        let mut x = [0.0; 2048];
        let mut y = [0.0; 2048];
        for i in 0..x.len(){
            x[i] = rand::random();
            x[i] -= 0.5;
            x[i] *= 100.0;
        }

        b.iter(|| {
            atan(&x, &mut y);
            black_box(y[0]);
        });
    });
}

fn sins512(c: &mut Criterion) {
    c.bench_function("sins512", |b| {    
        let mut x = [0.0; 2048];
        let mut y = [0.0; 2048];
        for i in 0..x.len(){
            x[i] = rand::random();
            x[i] -= 0.5;
            x[i] *= 10.0;
        }

        b.iter(|| {
            sin(&x, &mut y);
            black_box(y[0]);
        });
    });
}


criterion_group!(benches, 
    exps_naive, 
    lns_naive, 
    log2_naive, 
    erfs_naive, 
    atans_naive,
    sins_naive,
    exps512, 
    erfs512, 
    lns512,
    log2s512,
    atans512,
    sins512
);
criterion_main!(benches);