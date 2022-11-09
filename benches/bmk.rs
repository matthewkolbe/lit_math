#![feature(stdsimd)]
#![feature(new_uninit)]
#![feature(avx512_target_feature)]
#![feature(target_feature_11)]

mod black_scholes;
use black_scholes::*;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lit_math::*;
use rand;

const N: usize = 10_000_000;

fn exps_naive(c: &mut Criterion) {
    c.bench_function("exps_naive", |b| {   
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..N{
            x.push(rand::random());
            x[i] -= 0.5;
            x[i] *= 100.0;
            y.push(0.0);
        }

        b.iter(|| {

            for i in 0..x.len() {
                black_box(y[i] = f64::exp(x[i])); }
        });
    });
}

fn exps_naive_par(c: &mut Criterion) {
    use rayon::prelude::*;

    c.bench_function("exps_naive_par", |b| {   
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..N{
            x.push(rand::random());
            x[i] -= 0.5;
            x[i] *= 100.0;
            y.push(0.0);
        }

        b.iter(|| {

            const CHUNK: usize = N / 32;

            y.par_chunks_mut(CHUNK).enumerate().for_each(|(index, slice)| {
                for i in slice.iter_mut(){
                    *i = f64::exp(x[index]);
                }
            });
        });
    });
}

fn lns_naive(c: &mut Criterion) {
    c.bench_function("lns_naive", |b| {    
        let mut x= [0.0; N];
        let mut y = [0.0; N];
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
        let mut x= [0.0; N];
        let mut y = [0.0; N];
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
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..N{
            x.push(rand::random());
            x[i] -= 0.5;
            x[i] *= 100.0;
            y.push(0.0);
        }

        b.iter(|| {
            for i in 0..x.len() {
                black_box(y[i] = erf(x[i])); }
        });
    });
}

fn atans_naive(c: &mut Criterion) {
    c.bench_function("atans_naive", |b| {    
        let mut x= [0.0; N];
        let mut y = [0.0; N];
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
        let mut x= [0.0; N];
        let mut y = [0.0; N];
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


fn exps512_par(c: &mut Criterion) {

    c.bench_function("exps512_par", |b| {    
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..N{
            x.push(rand::random());
            x[i] -= 0.5;
            x[i] *= 100.0;
            y.push(0.0);
        }

        b.iter(|| {
            exp_par(&x, &mut y);
            black_box(y[0]);
        });
    });
}

fn expvs512(c: &mut Criterion) {
    c.bench_function("expvs512", |b| {    
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..N{
            x.push(rand::random());
            x[i] -= 0.5;
            x[i] *= 100.0;
            y.push(0.0);
        }

        b.iter(|| {
            exp(&x, &mut y);
            black_box(y[0]);
        });
    });
}

fn expvs256(c: &mut Criterion) {
    c.bench_function("expvs256", |b| {    
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..N{
            x.push(rand::random());
            x[i] -= 0.5;
            x[i] *= 100.0;
            y.push(0.0);
        }

        b.iter(|| {
            exp256v(&x, &mut y);
            black_box(y[0]);
        });
    });
}

fn exps256_par(c: &mut Criterion) {
    use rayon::prelude::*;

    c.bench_function("exps256_par", |b| {    
        let mut x: Vec<f64> = Vec::new();
        let mut y: Vec<f64> = Vec::new();
        for i in 0..N{
            x.push(rand::random());
            x[i] -= 0.5;
            x[i] *= 100.0;
            y.push(0.0);
        }
        const CORES: usize = 32;

        b.iter(|| {

            const CHUNK: usize = N / CORES;

            y.par_chunks_mut(CHUNK).enumerate().for_each(|(index, slice)| {
                exp256(&x[index..(index+CHUNK)], slice);
            });
            
            black_box(y[0]);
        });
    });
}



fn erfs512(c: &mut Criterion) {
    c.bench_function("erfs512", |b| {    
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..N{
            x.push(rand::random());
            x[i] -= 0.5;
            x[i] *= 100.0;
            y.push(0.0);
        }
        b.iter(|| {
            erf(&x, &mut y);
            black_box(y[0]);
        });
    });
}

fn lns512(c: &mut Criterion) {
    c.bench_function("lns512", |b| {    
        let mut x = [0.0; N];
        let mut y = [0.0; N];
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
        let mut x = [0.0; N];
        let mut y = [0.0; N];
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
        let mut x = [0.0; N];
        let mut y = [0.0; N];
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
        let mut x = [0.0; N];
        let mut y = [0.0; N];
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
    // exps_naive, 
    // exps_naive_par,
    // lns_naive, 
    // log2_naive, 
    // erfs_naive, 
    // atans_naive,
    // sins_naive,
    exps512_par,
    expvs512,
    // expvs256,
    // exps256_par,
    // lns512,
    // log2s512,
    // erfs512,
    // atans512,
    // sins512,
    // bs_naive,
    // bs512
);
criterion_main!(benches);