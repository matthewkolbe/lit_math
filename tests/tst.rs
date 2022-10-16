#[cfg(test)]

use lit_math::lit::*;
use approx::*;

#[test]
fn exp_test()
{
    let x = [0.0, 1.0, f64::INFINITY, -900.0, 900.0, f64::NAN, f64::NEG_INFINITY, -3.14159, 3.14159];
    let mut y = [0.0; 9];

    unsafe{
        exp512(&x, &mut y);}

    let mut r = relative_eq!(y[0], f64::exp(x[0]), epsilon = 1e-16);
    assert!(r);
    r = relative_eq!(y[1], f64::exp(x[1]), epsilon = 1e-16);
    assert!(r);
    assert_eq!(y[2], f64::INFINITY);
    assert_eq!(y[3], 0.0);
    assert_eq!(y[4], f64::INFINITY);
    assert!(f64::is_nan(y[5]));
    assert_eq!(y[6], 0.0);
    r = relative_eq!(y[7], f64::exp(x[7]), epsilon = 1e-16);
    assert!(r);
    r = relative_eq!(y[8], f64::exp(x[8]), epsilon = 1e-14);
    assert!(r);
}

#[test]
fn erf_test()
{
    use statrs::function::erf::erf;

    let x = [0.0, 1.0, f64::INFINITY, -100000.0, 100000.0, f64::NAN, f64::NEG_INFINITY, -3.14159, 3.14159];
    let mut y = [0.0; 9];
    let eps = 1e-11;

    unsafe{
        erf512(&x, &mut y);}

    let mut r = relative_eq!(y[0], erf(x[0]), epsilon = eps);
    assert!(r);
    r = relative_eq!(y[1], erf(x[1]), epsilon = eps);
    assert!(r);
    assert_eq!(y[2], 1.0);
    assert_eq!(y[3], -1.0);
    assert_eq!(y[4], 1.0);
    assert!(f64::is_nan(y[5]));
    assert_eq!(y[6], -1.0);
    r = relative_eq!(y[7], erf(x[7]), epsilon = eps);
    assert!(r);
    r = relative_eq!(y[8], erf(x[8]), epsilon = eps);
    assert!(r);
}

#[test]
fn log_test()
{
    let x =  [1.0, 0.51, f64::INFINITY, f64::NAN, f64::NEG_INFINITY, 0.0, 3.14159, -0.5];
    let mut y= [0.0; 8];
    let eps = 5e-16;

    unsafe{
        ln512(&x, &mut y);}

    println!("{:?}", y);

    let mut r = relative_eq!(y[0], f64::ln(x[0]), epsilon = eps);
    assert!(r);
    r = relative_eq!(y[1], f64::ln(x[1]), epsilon = eps);
    assert!(r);
    assert!(f64::is_infinite(y[2]));
    assert!(f64::is_nan(y[3]));
    assert!(f64::is_nan(y[4]));
    assert!(f64::is_infinite(y[5]));
    assert!(f64::is_sign_negative(y[5]));
    r = relative_eq!(y[6], f64::ln(x[6]), epsilon = eps);
    assert!(r);
    assert!(f64::is_nan(y[7]));

}