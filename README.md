## lit_math
A collection of AVX-512 accelerated mathematical functions for Rust.

This is very similar to a Rust implementation of the [Intel MKL VM Mathematical Functions](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/vector-mathematical-functions/vm-mathematical-functions.html)

### Usage

There are three public interfaces for each function `func`:
1. `func(in: &[f64], out: &mut [f64]) -> ()`
2. `unsafe func_intr(in: &__m512d, out: &mut __m512d) -> ()`
3. `unsafe _m512_func_pd(in: __m512d) -> __m512d`

The first is used to compute an array of inputs as quickly as possible for the given function. The benchmarks section features plenty of examples of how this is done. Say you want to calculate $e^x$ on n $x$ values: you allocate what `x` inputs you want to calc on, prealloate a `y` return array, and call `lit_math::exp(&x, &mut y)`. 

The other two can be used as building blocks to make more complicated functions.

### Speedup

On my Ryzen 7950x, the speedup is roughly 6x with no loss of precision for `ln`, `log2`, `exp` and `pow2` when calculating on 2048 sized arrays.