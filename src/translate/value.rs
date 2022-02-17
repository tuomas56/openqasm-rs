use std::ops::Neg;

use num::{integer::Roots, Signed, Zero};

use super::EvalError;

/// A value of type a + b * pi, with a, b Gaussian rationals.
///
/// This is the main type used for the evaluation of parameters
/// in translation. It is formally defined as the set Q(i) + pi Q(i).
/// This format is useful because it accurately represents the most
/// common values for parameters in quantum programs accurately.
///
/// Since this set is only closed under addition, not multiplication
/// or other operations, approximations are taken when necessary.
/// In general, if every intermediate subexpression of a calculation
/// is representable in this form, then the implementation
/// makes an effort to find the exact value of the calculation.
/// However, this is not guaranteed, and you should only rely
/// on addition, subtraction and scalar multiplication to be exact.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Value {
    /// The rational part of the value.
    pub a: num::Complex<num::Rational64>,
    /// The pi-rational part of the value.
    pub b: num::Complex<num::Rational64>,
}

impl Value {
    pub const PI: Value = Value {
        a: Value::RAW_ZERO,
        b: num::Complex::new(
            num::Rational64::new_raw(1, 1),
            num::Rational64::new_raw(0, 1),
        ),
    };

    // The value pi/2.
    pub const PI_2: Value = Value {
        a: Value::RAW_ZERO,
        b: num::Complex::new(
            num::Rational64::new_raw(1, 2),
            num::Rational64::new_raw(0, 1),
        ),
    };

    pub const ZERO: Value = Value {
        a: Value::RAW_ZERO,
        b: Value::RAW_ZERO,
    };

    pub const I: Value = Value {
        a: num::Complex::new(
            num::Rational64::new_raw(0, 1),
            num::Rational64::new_raw(1, 1),
        ),
        b: Value::RAW_ZERO,
    };

    const RAW_ZERO: num::Complex<num::Rational64> = num::Complex::new(
        num::Rational64::new_raw(0, 1),
        num::Rational64::new_raw(0, 1),
    );

    /// Create a `Value` from the given integer.
    pub const fn int(val: i64) -> Value {
        Value {
            a: num::Complex::new(
                num::Rational64::new_raw(val, 1),
                num::Rational64::new_raw(0, 1),
            ),
            b: Value::RAW_ZERO,
        }
    }

    pub fn into_float(self) -> num::Complex<f32> {
        let a = *self.a.re.numer() as f32 / *self.a.re.denom() as f32;
        let b = *self.b.re.numer() as f32 / *self.b.re.denom() as f32;
        let re = a + 3.14159265 * b;
        let a = *self.a.im.numer() as f32 / *self.a.im.denom() as f32;
        let b = *self.b.im.numer() as f32 / *self.b.im.denom() as f32;
        let im = a + 3.14159265 * b;
        num::Complex::new(re, im)
    }

    /// Attempt to convert a floating point number into
    /// a `Value`. This will fail if the number is out
    /// of range of a 64-bit rational. No attempt is made
    /// to recognize fractions of pi.
    pub fn from_float(val: num::Complex<f32>) -> Option<Value> {
        Some(Value {
            a: num::Complex::new(
                num::Rational64::approximate_float(val.re)?,
                num::Rational64::approximate_float(val.im)?,
            ),
            b: num::Complex::new(0.into(), 0.into()),
        })
    }

    fn to_rat(self) -> Option<num::Complex<num::Rational64>> {
        if self.b == Value::RAW_ZERO {
            Some(self.a)
        } else {
            None
        }
    }

    fn to_pi_rat(self) -> Option<num::Complex<num::Rational64>> {
        if self.a == Value::RAW_ZERO {
            Some(self.b)
        } else {
            None
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} + {}π", self.a, self.b)
    }
}

impl std::ops::Add<Value> for Value {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        Value {
            a: self.a + other.a,
            b: self.b + other.b,
        }
    }
}

impl std::ops::Sub<Value> for Value {
    type Output = Value;

    fn sub(self, other: Value) -> Value {
        Value {
            a: self.a - other.a,
            b: self.b - other.b,
        }
    }
}

impl std::ops::Mul<Value> for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Value {
        let mut c = self.b * other.b;
        // If there is a non-zero pi^2 term, approximate it:
        if c != Value::RAW_ZERO {
            c *= num::Complex::new(
                num::Rational64::approximate_float(3.14159265 * 3.14159265).unwrap(),
                0.into(),
            );
        }

        Value {
            a: self.a * other.a + c,
            b: self.b * other.a + self.a * other.b,
        }
    }
}

impl Value {
    pub(super) fn div_internal(self, other: Value) -> Result<Value, EvalError> {
        if other == Value::ZERO {
            Err(EvalError::DivideByZero)
        } else if let Some(rat) = other.to_rat() {
            // value / rational is value:
            Ok(Value {
                a: self.a / rat,
                b: self.b / rat,
            })
        } else if let Some(orat) = other.to_pi_rat() {
            // pi-rational / pi-rational is rational:
            if let Some(srat) = self.to_pi_rat() {
                Ok(Value {
                    a: srat / orat,
                    b: Value::RAW_ZERO,
                })
            } else {
                let f = self.into_float() / other.into_float();
                Value::from_float(f).ok_or(EvalError::ApproximateFail(f))
            }
        } else {
            // The only other rational case is when the numerator
            // and denominator are a rational multiple of each other.
            // (and this must be the case as pi is transcendental).
            let aa = self.a / other.a;
            let bb = self.b / other.b;
            if aa == bb {
                Ok(Value {
                    a: aa,
                    b: Value::RAW_ZERO,
                })
            } else {
                let f = self.into_float() / other.into_float();
                Value::from_float(f).ok_or(EvalError::ApproximateFail(f))
            }
        }
    }

    /// Attempt to divide this value by another,
    /// failing if the divisor is zero, or the result
    /// is out of range.
    pub fn checked_div(self, other: Value) -> Option<Value> {
        self.div_internal(other).ok()
    }
}

impl std::ops::Div<Value> for Value {
    type Output = Value;

    fn div(self, other: Value) -> Value {
        self.div_internal(other).unwrap()
    }
}

impl Value {
    fn int_root(val: i64, root: u32) -> Option<num::Complex<num::Rational64>> {
        let (mul, val): (num::Complex<num::Rational64>, u64) = if val < 0 {
            if root & 1 == 1 {
                // Odd root of -1 is just -1
                (num::Complex::new((-1).into(), 0.into()), (-val) as u64)
            } else if root & 3 == 2 {
                // Even root of -1 with n = 2 (mod 4) is just i
                (num::Complex::new(0.into(), 1.into()), (-val) as u64)
            } else {
                // Even root of -1 with n = 0 (mod 4) is irrational
                return None;
            }
        } else {
            (num::Complex::new(1.into(), 0.into()), val as u64)
        };

        let ans = val.nth_root(root);
        if ans.checked_pow(root) == Some(val) {
            Some(mul * num::Complex::new((ans as i64).into(), 0.into()))
        } else {
            None
        }
    }

    fn rat_root(val: num::Rational64, root: u32) -> Option<num::Complex<num::Rational64>> {
        // A square root is rational iff its numerator and denominator
        // are both squares.
        let a = Value::int_root(*val.numer(), root)?;
        let b = Value::int_root(*val.denom(), root)?;
        Some(a / b)
    }

    fn extract_root(
        val: num::Complex<num::Rational64>,
        root: u32,
    ) -> Result<num::Complex<num::Rational64>, EvalError> {
        if root == 1 {
            return Ok(val);
        } else if val.im.is_zero() && val.re == 1.into() {
            return Ok(val);
        }

        if val.im.is_zero() {
            // Roots of reals are either exact or not:
            if let Some(root) = Value::rat_root(val.re, root) {
                Ok(root)
            } else {
                let f = Value {
                    a: val,
                    b: Value::RAW_ZERO,
                }
                .into_float();
                let fp = f.powf(1.0 / (root as f32));
                Value::from_float(fp)
                    .map(|v| v.a)
                    .ok_or(EvalError::ApproximateFail(fp))
            }
        } else {
            // Roots of complex numbers are neve rational:
            let f = Value {
                a: val,
                b: Value::RAW_ZERO,
            }
            .into_float();
            let fp = f.powf(1.0 / (root as f32));
            Value::from_float(fp)
                .map(|v| v.a)
                .ok_or(EvalError::ApproximateFail(fp))
        }
    }

    pub(super) fn pow_internal(self, other: Value) -> Result<Value, EvalError> {
        match (self.to_rat(), other.to_rat()) {
            // The only rational case of exponentiation is rational
            // to a rational power, and even then not always.
            (Some(srat), Some(orat)) => {
                // k^{a/b + ic/d} = k^(a/b)(k^(c/d))^i
                // and the only rational value of x^i is for x = 1.
                let ap = srat.powi(*orat.re.numer() as i32);
                let aa = Value::extract_root(ap, *orat.re.denom() as u32)?;
                let bp = srat.powi(*orat.im.numer() as i32);
                let bb = Value::extract_root(bp, *orat.im.denom() as u32)?;
                if bb.im.is_zero() && bb.re == 1.into() {
                    Ok(Value {
                        a: aa,
                        b: Value::RAW_ZERO,
                    })
                } else {
                    let f = self.into_float().powc(other.into_float());
                    Value::from_float(f).ok_or(EvalError::ApproximateFail(f))
                }
            }
            _ => {
                let f = self.into_float().powc(other.into_float());
                Value::from_float(f).ok_or(EvalError::ApproximateFail(f))
            }
        }
    }

    /// Attempt to find a power of this value, failing if the result is out of range.
    pub fn checked_pow(self, other: Value) -> Option<Value> {
        self.pow_internal(other).ok()
    }
}

impl Value {
    pub(super) fn sqrt_internal(self) -> Result<Value, EvalError> {
        if let Some(srat) = self.to_rat() {
            if srat.im.is_zero() {
                // Square root of real number is just the root:
                if let Some(root) = Value::rat_root(srat.re, 2) {
                    return Ok(Value {
                        a: root,
                        b: Value::RAW_ZERO,
                    });
                }
            } else {
                // For complex numbers, the principal square root is given
                // by sqrt(z) = sqrt(|z|) * (z + r) / |z + r|, and hence is
                // rational exactly when |z|^2 is a fourth power and |z + r|^2
                // is a square.
                let mag2 = srat.norm_sqr();
                if let Some(r) = Value::rat_root(mag2, 2) {
                    let r = r.re;
                    let rmag2 = (srat + r).norm_sqr();
                    if let Some(fac) = Value::rat_root(r / rmag2, 2) {
                        return Ok(Value {
                            a: fac * (srat + r),
                            b: Value::RAW_ZERO,
                        });
                    }
                }
            }
        }

        let f = self.into_float().sqrt();
        Value::from_float(f).ok_or(EvalError::ApproximateFail(f))
    }

    pub fn checked_sqrt(self) -> Option<Value> {
        self.sqrt_internal().ok()
    }

    /// Negate this value.
    pub fn neg(self) -> Value {
        Value {
            a: self.a.neg(),
            b: self.b.neg(),
        }
    }

    pub(super) fn sin_internal(self) -> Result<Value, EvalError> {
        if let Some(srat) = self.to_pi_rat() {
            if srat.im.is_zero() {
                let mut f = (srat.re / 2).fract() * 2;
                if f.is_negative() {
                    f += 2;
                }

                // By Niven's theorem, the only rational values of sin for
                // 0 <= theta <= pi/2 are sin(0) = 0, sin(pi/6) = 1/2 and sin(pi/2) = 1.
                if f == num::Rational64::new(0, 1) {
                    return Ok(Value {
                        a: Value::RAW_ZERO,
                        b: Value::RAW_ZERO,
                    });
                } else if f == num::Rational64::new(1, 6) || f == num::Rational64::new(5, 6) {
                    return Ok(Value {
                        a: num::Complex::new(num::Rational64::new(1, 2), 0.into()),
                        b: Value::RAW_ZERO,
                    });
                } else if f == num::Rational64::new(1, 2) {
                    return Ok(Value {
                        a: num::Complex::new(num::Rational64::new(1, 1), 0.into()),
                        b: Value::RAW_ZERO,
                    });
                } else if f == num::Rational64::new(7, 6) || f == num::Rational64::new(11, 6) {
                    return Ok(Value {
                        a: num::Complex::new(num::Rational64::new(-1, 2), 0.into()),
                        b: Value::RAW_ZERO,
                    });
                } else if f == num::Rational64::new(3, 2) {
                    return Ok(Value {
                        a: num::Complex::new(num::Rational64::new(-1, 1), 0.into()),
                        b: Value::RAW_ZERO,
                    });
                }
            }
        }

        let f = self.into_float().sin();
        Value::from_float(f).ok_or(EvalError::ApproximateFail(f))
    }

    pub fn checked_sin(self) -> Option<Value> {
        self.sin_internal().ok()
    }

    pub(super) fn cos_internal(self) -> Result<Value, EvalError> {
        (self + Value::PI_2).sin_internal()
    }

    pub fn checked_cos(self) -> Option<Value> {
        self.cos_internal().ok()
    }

    pub(super) fn tan_internal(self) -> Result<Value, EvalError> {
        self.sin_internal()?.div_internal(self.cos_internal()?)
    }

    pub fn checked_tan(self) -> Option<Value> {
        self.tan_internal().ok()
    }

    pub(super) fn exp_internal(self) -> Result<Value, EvalError> {
        // The only time the exponential is rational is when
        // you have e^{ialpha} for alpha some fractions of pi.
        if let Some(srat) = self.to_pi_rat() {
            if srat.re.is_zero() {
                let arg = Value {
                    a: Value::RAW_ZERO,
                    b: num::Complex::new(srat.im, 0.into()),
                };

                let a = arg.cos_internal()?;
                let b = arg.sin_internal()?;

                return Ok(a + Value::I * b);
            }
        }

        let f = self.into_float().exp();
        Value::from_float(f).ok_or(EvalError::ApproximateFail(f))
    }

    pub fn checked_exp(self) -> Option<Value> {
        self.exp_internal().ok()
    }

    pub(super) fn ln_internal(self) -> Result<Value, EvalError> {
        if let Some(srat) = self.to_rat() {
            match (
                *srat.re.numer(),
                *srat.re.denom(),
                *srat.im.numer(),
                *srat.im.denom(),
            ) {
                (1, 1, 0, _) => return Ok(Value::ZERO),
                (-1, 1, 0, _) => return Ok(Value::I * Value::PI),
                (0, _, 1, 1) => return Ok(Value::I * Value::PI_2),
                (0, _, -1, 1) => return Ok(Value::int(3) * Value::I * Value::PI_2),
                _ => (),
            }
        }

        let f = self.into_float().ln();
        Value::from_float(f).ok_or(EvalError::ApproximateFail(f))
    }

    pub fn checked_ln(self) -> Option<Value> {
        self.ln_internal().ok()
    }
}

#[test]
fn int_root() {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    for _ in 0..10000 {
        for k in 2..10 {
            let a: u32 = rng.gen();
            let ap = if let Some(ap) = (a as i64 + 1).checked_pow(k) {
                ap
            } else {
                continue;
            };
            let val1 = Value::int_root(ap, k);
            let val2 = Value::int_root(ap + 1, k);

            assert_eq!(val1, Some(Value::int(a as i64 + 1).a));
            assert_eq!(val2, None);
        }
    }
}
