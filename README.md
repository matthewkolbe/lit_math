## lit_math
A collection of AVX-512 accelerated mathematical functions for Rust.

This is very similar to a Rust implementation of the [Intel MKL VM Mathematical Functions](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/vector-mathematical-functions/vm-mathematical-functions.html)

### Usage

There are four public interfaces for each function `func`:
1. `func(in: &[f64], out: &mut [f64]) -> ()`
2. `func_par(in: &[f64], out: &mut [f64]) -> ()`
3. `unsafe func_intr(in: &__m512d, out: &mut __m512d) -> ()`
4. `unsafe _m512_func_pd(in: __m512d) -> __m512d`

The first one is used to compute an array of inputs as quickly as possible for the given function. The benchmarks section features plenty of examples of how this is done. Say you want to calculate $e^x$ on n $x$ values: you allocate what `x` inputs you want to calc on, prealloate a `y` return array, and call `lit_math::exp(&x, &mut y)`. The second will automatically use a Rayon parallelization scheme to go faster (or slower), depending on your inputs and CPU.

The other two can be used as building blocks to make more complicated functions.

### More Advanced Usage

The `unroll_fn` macro can be used to turn any function with the signature `unsafe func_intr(in: &__m512d, out: &mut __m512d) -> ()` into optimized unrolled and parallel functions like those mentioned in 1 and 2 above. This macro is the one used to generate 1 and 2, so examples for how to use it are in the source for `exp`.

TODO: expand the interface to handle an arbitrarily large number of input values.  

### Speedup

On my Ryzen 7950x, the speedup is roughly 8x with no loss of precision for `ln`, `log2`, `exp` and `pow2` when calculating on 2048 sized arrays.