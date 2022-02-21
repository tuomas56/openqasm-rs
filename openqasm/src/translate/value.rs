use num::{
    integer::Roots,
    traits::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub},
    Signed,
};

use super::EvalError;

/// A value of type a + b * pi, with a, b rationals.
///
/// This is the main type used for the evaluation of parameters
/// in translation. This format is useful because it accurately
/// represents the most common values for parameters in quantum
/// programs accurately.
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
    pub a: num::Rational64,
    /// The pi-rational part of the value.
    pub b: num::Rational64,
}

impl Value {
    pub const PI: Value = Value {
        a: Value::RAW_ZERO,
        b: num::Rational64::new_raw(1, 1),
    };

    // The value pi/2.
    pub const PI_2: Value = Value {
        a: Value::RAW_ZERO,
        b: num::Rational64::new_raw(1, 2),
    };

    pub const ZERO: Value = Value {
        a: Value::RAW_ZERO,
        b: Value::RAW_ZERO,
    };

    const RAW_ZERO: num::Rational64 = num::Rational64::new_raw(0, 1);
    const PI_SQUARED: num::Rational64 =
        num::Rational64::new_raw(6499908331188584841, 658578405682719844);

    /// Create a `Value` from the given integer.
    pub const fn int(val: i64) -> Value {
        Value {
            a: num::Rational64::new_raw(val, 1),
            b: Value::RAW_ZERO,
        }
    }

    pub fn into_float(self) -> f32 {
        let a = *self.a.numer() as f32 / *self.a.denom() as f32;
        let b = *self.b.numer() as f32 / *self.b.denom() as f32;
        a + 3.14159265 * b
    }

    /// Attempt to convert a floating point number into
    /// a `Value`. This will fail if the number is out
    /// of range of a 64-bit rational. No attempt is made
    /// to recognize fractions of pi.
    pub fn from_float(val: f32) -> Option<Value> {
        Some(Value {
            a: num::Rational64::approximate_float(val)?,
            b: Value::RAW_ZERO,
        })
    }

    fn to_rat(self) -> Option<num::Rational64> {
        if self.b == Value::RAW_ZERO {
            Some(self.a)
        } else {
            None
        }
    }

    fn to_pi_rat(self) -> Option<num::Rational64> {
        if self.a == Value::RAW_ZERO {
            Some(self.b)
        } else {
            None
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.b == Value::RAW_ZERO {
            write!(f, "{}", self.a)
        } else if self.a == Value::RAW_ZERO {
            if *self.b.numer() == 1 {
                write!(f, "π/{}", self.b.denom())
            } else {
                write!(f, "{}π", self.b)
            }
        } else {
            if *self.b.numer() == 1 {
                write!(f, "{}+π/{}", self.a, self.b.denom())
            } else {
                write!(f, "{}+{}π", self.a, self.b)
            }
        }
    }
}

impl Value {
    pub(super) fn add_internal(self, other: Value) -> Result<Value, EvalError> {
        Ok(Value {
            a: self
                .a
                .checked_add(&other.a)
                .ok_or(EvalError::OverflowError)?,
            b: self
                .b
                .checked_add(&other.b)
                .ok_or(EvalError::OverflowError)?,
        })
    }

    pub(super) fn sub_internal(self, other: Value) -> Result<Value, EvalError> {
        Ok(Value {
            a: self
                .a
                .checked_sub(&other.a)
                .ok_or(EvalError::OverflowError)?,
            b: self
                .b
                .checked_sub(&other.b)
                .ok_or(EvalError::OverflowError)?,
        })
    }

    pub(super) fn mul_internal(self, other: Value) -> Result<Value, EvalError> {
        let c = self
            .b
            .checked_mul(&other.b)
            .ok_or(EvalError::OverflowError)?
            .checked_mul(&Value::PI_SQUARED)
            .ok_or(EvalError::OverflowError)?;

        Ok(Value {
            a: self
                .a
                .checked_mul(&other.a)
                .ok_or(EvalError::OverflowError)?
                .checked_add(&c)
                .ok_or(EvalError::OverflowError)?,
            b: self
                .b
                .checked_mul(&other.a)
                .ok_or(EvalError::OverflowError)?
                .checked_add(
                    &self
                        .a
                        .checked_mul(&other.b)
                        .ok_or(EvalError::OverflowError)?,
                )
                .ok_or(EvalError::OverflowError)?,
        })
    }

    pub(super) fn div_internal(self, other: Value) -> Result<Value, EvalError> {
        if other == Value::ZERO {
            Err(EvalError::DivideByZero)
        } else if let Some(rat) = other.to_rat() {
            // value / rational is value:
            Ok(Value {
                a: self.a.checked_div(&rat).ok_or(EvalError::OverflowError)?,
                b: self.b.checked_div(&rat).ok_or(EvalError::OverflowError)?,
            })
        } else if let Some(orat) = other.to_pi_rat() {
            // pi-rational / pi-rational is rational:
            if let Some(srat) = self.to_pi_rat() {
                Ok(Value {
                    a: srat.checked_div(&orat).ok_or(EvalError::OverflowError)?,
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
            let aa = self
                .a
                .checked_div(&other.a)
                .ok_or(EvalError::OverflowError)?;
            let bb = self
                .b
                .checked_div(&other.b)
                .ok_or(EvalError::OverflowError)?;
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
    fn int_root(val: u64, root: u64) -> Option<u64> {
        if let Ok(root) = u32::try_from(root) {
            let ans = val.nth_root(root);
            if ans.checked_pow(root) == Some(val) {
                Some(ans)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn rat_root(val: num::Rational64, root: u64) -> Option<num::Rational64> {
        // A square root is rational iff its numerator and denominator
        // are both squares.
        if val.is_negative() {
            return None;
        }

        let a = Value::int_root(*val.numer() as u64, root)?;
        let b = Value::int_root(*val.denom() as u64, root)?;
        Some(num::Rational64::new(a as i64, b as i64))
    }

    fn extract_root(val: num::Rational64, root: u64) -> Result<num::Rational64, EvalError> {
        if root == 1 {
            return Ok(val);
        } else if val == 1.into() {
            return Ok(val);
        }

        if let Some(root) = Value::rat_root(val, root) {
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
    }

    pub(super) fn pow_internal(self, other: Value) -> Result<Value, EvalError> {
        match (self.to_rat(), other.to_rat()) {
            // The only rational case of exponentiation is rational
            // to a rational power, and even then not always.
            (Some(srat), Some(orat)) => {
                let exp = orat.numer().abs() as usize;
                let recip = orat.numer().is_negative();
                let ap = num::traits::checked_pow(srat, exp).ok_or(EvalError::OverflowError)?;
                let aa = Value::extract_root(ap, *orat.denom() as u64)?;

                if recip {
                    if aa == Value::RAW_ZERO {
                        return Err(EvalError::DivideByZero);
                    } else {
                        return Ok(Value {
                            a: aa.recip(),
                            b: Value::RAW_ZERO,
                        });
                    }
                } else {
                    return Ok(Value {
                        a: aa,
                        b: Value::RAW_ZERO,
                    });
                }
            }
            _ => (),
        }

        let f = self.into_float().powf(other.into_float());
        Value::from_float(f).ok_or(EvalError::ApproximateFail(f))
    }

    /// Attempt to find a power of this value, failing if the result is out of range.
    pub fn checked_pow(self, other: Value) -> Option<Value> {
        self.pow_internal(other).ok()
    }
}

impl Value {
    pub(super) fn sqrt_internal(self) -> Result<Value, EvalError> {
        if let Some(srat) = self.to_rat() {
            if let Some(root) = Value::rat_root(srat, 2) {
                return Ok(Value {
                    a: root,
                    b: Value::RAW_ZERO,
                });
            }
        }

        let f = self.into_float().sqrt();
        Value::from_float(f).ok_or(EvalError::ApproximateFail(f))
    }

    pub fn checked_sqrt(self) -> Option<Value> {
        self.sqrt_internal().ok()
    }

    pub(super) fn neg_internal(self) -> Result<Value, EvalError> {
        self.mul_internal(Value::int(-1))
    }

    /// Negate this value.
    pub fn checked_neg(self) -> Option<Value> {
        self.neg_internal().ok()
    }

    pub(super) fn sin_internal(self) -> Result<Value, EvalError> {
        if let Some(srat) = self.to_pi_rat() {
            let mut f = srat
                .checked_div(&2.into())
                .ok_or(EvalError::OverflowError)?
                .fract()
                * 2;
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
                    a: num::Rational64::new(1, 2),
                    b: Value::RAW_ZERO,
                });
            } else if f == num::Rational64::new(1, 2) {
                return Ok(Value {
                    a: num::Rational64::new(1, 1),
                    b: Value::RAW_ZERO,
                });
            } else if f == num::Rational64::new(7, 6) || f == num::Rational64::new(11, 6) {
                return Ok(Value {
                    a: num::Rational64::new(-1, 2),
                    b: Value::RAW_ZERO,
                });
            } else if f == num::Rational64::new(3, 2) {
                return Ok(Value {
                    a: num::Rational64::new(-1, 1),
                    b: Value::RAW_ZERO,
                });
            }
        }

        let f = self.into_float().sin();
        Value::from_float(f).ok_or(EvalError::ApproximateFail(f))
    }

    pub fn checked_sin(self) -> Option<Value> {
        self.sin_internal().ok()
    }

    pub(super) fn cos_internal(self) -> Result<Value, EvalError> {
        self.add_internal(Value::PI_2)?.sin_internal()
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
        let f = self.into_float().exp();
        Value::from_float(f).ok_or(EvalError::ApproximateFail(f))
    }

    pub fn checked_exp(self) -> Option<Value> {
        self.exp_internal().ok()
    }

    pub(super) fn ln_internal(self) -> Result<Value, EvalError> {
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
        for k in 2u64..10 {
            let a: u32 = rng.gen();
            let ap = if let Some(ap) = (a as u64 + 1).checked_pow(k as u32) {
                ap
            } else {
                continue;
            };
            let val1 = Value::int_root(ap, k);
            let val2 = Value::int_root(ap + 1, k);

            assert_eq!(val1, Some(a as u64 + 1));
            assert_eq!(val2, None);
        }
    }
}
