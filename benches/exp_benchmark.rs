use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lit_math::*;


fn array_of_exps(c: &mut Criterion) {
    unsafe{
        c.bench_function("array_of_exps", |b| {    
            let x = [0.0; 1000];
            let mut y = [0.0; 1000];
            b.iter(|| {
                exp(&x, &mut y);
                black_box(y[0]);
            })
        });
    }
}


criterion_group!(benches, array_of_exps);
criterion_main!(benches);