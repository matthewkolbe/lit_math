#[cfg(test)]

use lit_math::*;
use approx::*;

#[test]
fn exp_test()
{
    let eps = 1e-17;

    let mut x = [0.0; 1000];
    x[0] = 0.0;
    x[1] = 1.0;
    x[2] = f64::INFINITY;
    x[3] = -900.0;
    x[4] = 900.0;
    x[5] = f64::NAN;
    x[6] = f64::NEG_INFINITY;

    for i in 7..1000
    {
        x[i] = -0.5 + (i as f64) / 1000.0;
    }

    let mut y = [0.0; 1000];

    exp(&x, &mut y);

    let mut r = relative_eq!(y[0], f64::exp(x[0]), epsilon = eps);
    assert!(r);
    r = relative_eq!(y[1], f64::exp(x[1]), epsilon = eps);
    assert!(r);
    assert_eq!(y[2], f64::INFINITY);
    assert_eq!(y[3], 0.0);
    assert_eq!(y[4], f64::INFINITY);
    assert!(f64::is_nan(y[5]));
    assert_eq!(y[6], 0.0);

    for i in 7..1000
    {
        r = relative_eq!(y[i], f64::exp(x[i]), epsilon = eps);
        assert!(r);
    }

}

#[test]
fn erf_test()
{
    use statrs::function::erf::erf;

    let x = [0.0, 1.0, f64::INFINITY, -100000.0, 100000.0, f64::NAN, f64::NEG_INFINITY, -3.14159, 3.14159];
    let mut y = [0.0; 9];
    let eps = 1e-11;

    lit_math::erf(&x, &mut y);

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

    ln(&x, &mut y);

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

#[test]
fn sin_test()
{
    let eps = 8e-16;

    let mut x = [0.0; 1000];
    x[0] = 0.0;
    x[1] = 1.0;
    x[2] = f64::INFINITY;
    x[3] = -900.0;
    x[4] = 900.0;
    x[5] = f64::NAN;
    x[6] = f64::NEG_INFINITY;

    for i in 7..1000 {
        x[i] = -std::f64::consts::PI + 2.0 * std::f64::consts::PI * (i as f64) / 1000.0;
    }

    let mut y = [0.0; 1000];

    sin(&x, &mut y);

    //println!("{:?}", y);

    let mut r = relative_eq!(y[0], f64::sin(x[0]), epsilon = eps);
    assert!(r);
    r = relative_eq!(y[1], f64::sin(x[1]), epsilon = eps);
    assert!(r);
    assert!(f64::is_nan(y[2]));
    // assert_eq!(y[3], 0.0);
    // assert_eq!(y[4], f64::INFINITY);
    assert!(f64::is_nan(y[5]));
    assert!(f64::is_nan(y[6]));

    for i in 7..1000
    {
        r = relative_eq!(y[i], f64::sin(x[i]), epsilon = eps);
        assert!(r);
    }

}

#[test]
fn tan_test()
{
    let eps = 8e-16;

    let mut x = [0.0; 1000];
    let mut y = [0.0; 1000];
    x[0] = 0.0;
    x[1] = 1.0;
    x[2] = f64::INFINITY;
    x[3] = -900.0;
    x[4] = 900.0;
    x[5] = f64::NAN;
    x[6] = f64::NEG_INFINITY;

    for i in 7..x.len() {
        x[i] = -0.28 * std::f64::consts::PI + 0.56 * std::f64::consts::PI * (i as f64) / (x.len() as f64);
    }
    tan(&x, &mut y);
    //println!("{:?}", y);

    let mut r = ulps_eq!(y[0], f64::tan(x[0]), epsilon = eps);
    assert!(r);
    r = ulps_eq!(y[1], f64::tan(x[1]), epsilon = eps);
    assert!(r);
    assert!(f64::is_nan(y[2]));
    // assert_eq!(y[3], 0.0);
    // assert_eq!(y[4], f64::INFINITY);
    assert!(f64::is_nan(y[5]));
    assert!(f64::is_nan(y[6]));

    for i in 7..x.len()
    {
        r = ulps_eq!(y[i], f64::tan(x[i]), epsilon = eps);
        assert!(r);
    }

}

#[test]
fn atan_test()
{
    let eps = 5e-16;

    let mut x = [0.0; 1000];
    let mut y = [0.0; 1000];
    x[0] = f64::NAN;
    x[1] = 1.0;
    x[2] = f64::INFINITY;
    x[3] = -900.0;
    x[4] = 900.0;
    x[5] = 0.0;
    x[6] = f64::NEG_INFINITY;

    for i in 7..x.len() {
        x[i] = -10.0 * std::f64::consts::PI + 20.0 * std::f64::consts::PI * (i as f64) / (x.len() as f64);
    }
    atan(&x, &mut y);
    //println!("{:?}", y);

    assert!(f64::is_nan(y[0]));

    for i in 1..x.len()
    {
        let r = relative_eq!(y[i], f64::atan(x[i]), epsilon = eps);
        assert!(r);
    }

}