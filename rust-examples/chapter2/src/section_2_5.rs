//! Section 2.5: Systems with Generic Operations
//!
//! This module demonstrates Rust's approach to generic operations through:
//! - Trait-based polymorphism (replacing Scheme's apply-generic)
//! - Type coercion via From/Into traits
//! - Generic polynomial arithmetic with arbitrary coefficient types

use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

// ============================================================================
// 2.5.1: Generic Arithmetic Operations
// ============================================================================

/// A Number type that can represent different kinds of numbers
#[derive(Debug, Clone, PartialEq)]
pub enum Number {
    SchemeNumber(i64),
    Rational(Rational),
    Complex(Complex),
}

/// Rational number representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rational {
    numer: i64,
    denom: i64,
}

impl Rational {
    pub fn new(n: i64, d: i64) -> Self {
        let g = gcd(n.abs(), d.abs());
        Rational {
            numer: n / g,
            denom: d / g,
        }
    }

    pub fn numer(&self) -> i64 {
        self.numer
    }

    pub fn denom(&self) -> i64 {
        self.denom
    }
}

/// Complex number representation (using rectangular form internally)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    real: f64,
    imag: f64,
}

impl Complex {
    pub fn from_real_imag(real: f64, imag: f64) -> Self {
        Complex { real, imag }
    }

    pub fn from_mag_ang(magnitude: f64, angle: f64) -> Self {
        Complex {
            real: magnitude * angle.cos(),
            imag: magnitude * angle.sin(),
        }
    }

    pub fn real_part(&self) -> f64 {
        self.real
    }

    pub fn imag_part(&self) -> f64 {
        self.imag
    }

    pub fn magnitude(&self) -> f64 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    pub fn angle(&self) -> f64 {
        self.imag.atan2(self.real)
    }
}

// Generic arithmetic operations on Number type
// In Scheme: (define (add x y) (apply-generic 'add x y))
// In Rust: Use trait implementations with pattern matching

impl Add for Number {
    type Output = Number;

    fn add(self, other: Number) -> Number {
        use Number::*;
        match (self, other) {
            (SchemeNumber(a), SchemeNumber(b)) => SchemeNumber(a + b),
            (Rational(a), Rational(b)) => Rational(a + b),
            (Complex(a), Complex(b)) => Complex(a + b),
            // Mixed types - use coercion (see section 2.5.2)
            (a, b) => {
                let (a_c, b_c) = coerce_to_common(a, b);
                a_c + b_c
            }
        }
    }
}

impl Sub for Number {
    type Output = Number;

    fn sub(self, other: Number) -> Number {
        use Number::*;
        match (self, other) {
            (SchemeNumber(a), SchemeNumber(b)) => SchemeNumber(a - b),
            (Rational(a), Rational(b)) => Rational(a - b),
            (Complex(a), Complex(b)) => Complex(a - b),
            (a, b) => {
                let (a_c, b_c) = coerce_to_common(a, b);
                a_c - b_c
            }
        }
    }
}

impl Mul for Number {
    type Output = Number;

    fn mul(self, other: Number) -> Number {
        use Number::*;
        match (self, other) {
            (SchemeNumber(a), SchemeNumber(b)) => SchemeNumber(a * b),
            (Rational(a), Rational(b)) => Rational(a * b),
            (Complex(a), Complex(b)) => Complex(a * b),
            (a, b) => {
                let (a_c, b_c) = coerce_to_common(a, b);
                a_c * b_c
            }
        }
    }
}

impl Div for Number {
    type Output = Number;

    fn div(self, other: Number) -> Number {
        use Number::*;
        match (self, other) {
            (SchemeNumber(a), SchemeNumber(b)) => SchemeNumber(a / b),
            (Rational(a), Rational(b)) => Rational(a / b),
            (Complex(a), Complex(b)) => Complex(a / b),
            (a, b) => {
                let (a_c, b_c) = coerce_to_common(a, b);
                a_c / b_c
            }
        }
    }
}

// Rational arithmetic
impl Add for Rational {
    type Output = Rational;

    fn add(self, other: Rational) -> Rational {
        Rational::new(
            self.numer * other.denom + other.numer * self.denom,
            self.denom * other.denom,
        )
    }
}

impl Sub for Rational {
    type Output = Rational;

    fn sub(self, other: Rational) -> Rational {
        Rational::new(
            self.numer * other.denom - other.numer * self.denom,
            self.denom * other.denom,
        )
    }
}

impl Mul for Rational {
    type Output = Rational;

    fn mul(self, other: Rational) -> Rational {
        Rational::new(self.numer * other.numer, self.denom * other.denom)
    }
}

impl Div for Rational {
    type Output = Rational;

    fn div(self, other: Rational) -> Rational {
        Rational::new(self.numer * other.denom, self.denom * other.numer)
    }
}

// Complex arithmetic
impl Add for Complex {
    type Output = Complex;

    fn add(self, other: Complex) -> Complex {
        Complex::from_real_imag(self.real + other.real, self.imag + other.imag)
    }
}

impl Sub for Complex {
    type Output = Complex;

    fn sub(self, other: Complex) -> Complex {
        Complex::from_real_imag(self.real - other.real, self.imag - other.imag)
    }
}

impl Mul for Complex {
    type Output = Complex;

    fn mul(self, other: Complex) -> Complex {
        Complex::from_mag_ang(
            self.magnitude() * other.magnitude(),
            self.angle() + other.angle(),
        )
    }
}

impl Div for Complex {
    type Output = Complex;

    fn div(self, other: Complex) -> Complex {
        Complex::from_mag_ang(
            self.magnitude() / other.magnitude(),
            self.angle() - other.angle(),
        )
    }
}

// Utility function
fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

// Exercise 2.79: Generic equality predicate
pub trait GenericEq {
    fn equ(&self, other: &Self) -> bool;
}

impl GenericEq for Number {
    fn equ(&self, other: &Number) -> bool {
        use Number::*;
        match (self, other) {
            (SchemeNumber(a), SchemeNumber(b)) => a == b,
            (Rational(a), Rational(b)) => a == b,
            (Complex(a), Complex(b)) => {
                (a.real - b.real).abs() < 1e-10 && (a.imag - b.imag).abs() < 1e-10
            }
            _ => false,
        }
    }
}

// Exercise 2.80: Generic zero predicate
pub trait GenericZero {
    fn is_zero(&self) -> bool;
}

impl GenericZero for Number {
    fn is_zero(&self) -> bool {
        match self {
            Number::SchemeNumber(n) => *n == 0,
            Number::Rational(r) => r.numer == 0,
            Number::Complex(c) => c.real.abs() < 1e-10 && c.imag.abs() < 1e-10,
        }
    }
}

// ============================================================================
// 2.5.2: Combining Data of Different Types (Coercion)
// ============================================================================

// Type tower: Integer -> Rational -> Real -> Complex
//
// In Scheme, coercion is done via a coercion table:
//   (put-coercion 'scheme-number 'complex scheme-number->complex)
//
// In Rust, we use From/Into traits to establish type conversions

impl From<i64> for Rational {
    fn from(n: i64) -> Rational {
        Rational::new(n, 1)
    }
}

impl From<i64> for Complex {
    fn from(n: i64) -> Complex {
        Complex::from_real_imag(n as f64, 0.0)
    }
}

impl From<Rational> for Complex {
    fn from(r: Rational) -> Complex {
        Complex::from_real_imag(r.numer as f64 / r.denom as f64, 0.0)
    }
}

// Coercion strategy: raise lower type to higher type in tower
fn coerce_to_common(a: Number, b: Number) -> (Number, Number) {
    use Number::*;
    match (&a, &b) {
        // Same types - no coercion needed
        (SchemeNumber(_), SchemeNumber(_))
        | (Rational(_), Rational(_))
        | (Complex(_), Complex(_)) => (a, b),

        // Scheme number + Rational -> both to Rational
        (SchemeNumber(n), Rational(_)) => (Number::Rational(self::Rational::from(*n)), b),
        (Rational(_), SchemeNumber(n)) => (a, Number::Rational(self::Rational::from(*n))),

        // Scheme number + Complex -> both to Complex
        (SchemeNumber(n), Complex(_)) => (Number::Complex(self::Complex::from(*n)), b),
        (Complex(_), SchemeNumber(n)) => (a, Number::Complex(self::Complex::from(*n))),

        // Rational + Complex -> both to Complex
        (Rational(r), Complex(_)) => (Number::Complex(self::Complex::from(*r)), b),
        (Complex(_), Rational(r)) => (a, Number::Complex(self::Complex::from(*r))),
    }
}

// Exercise 2.83: Raise operation
pub trait Raise {
    type Output;
    fn raise(self) -> Self::Output;
}

impl Raise for i64 {
    type Output = Rational;
    fn raise(self) -> Rational {
        Rational::from(self)
    }
}

impl Raise for Rational {
    type Output = Complex;
    fn raise(self) -> Complex {
        Complex::from(self)
    }
}

// Exercise 2.84: Type level in tower
fn type_level(n: &Number) -> u8 {
    match n {
        Number::SchemeNumber(_) => 1,
        Number::Rational(_) => 2,
        Number::Complex(_) => 3,
    }
}

// Exercise 2.85: Drop operation (simplify to lowest representation)
pub trait Drop {
    fn drop_if_possible(self) -> Self;
}

impl Drop for Number {
    fn drop_if_possible(self) -> Number {
        match self {
            Number::Complex(c) => {
                if c.imag.abs() < 1e-10 {
                    let real_val = c.real.round() as i64;
                    if (c.real - real_val as f64).abs() < 1e-10 {
                        Number::SchemeNumber(real_val)
                    } else {
                        Number::Complex(c)
                    }
                } else {
                    Number::Complex(c)
                }
            }
            Number::Rational(r) => {
                if r.denom == 1 {
                    Number::SchemeNumber(r.numer)
                } else {
                    Number::Rational(r)
                }
            }
            n => n,
        }
    }
}

// ============================================================================
// 2.5.3: Example: Symbolic Algebra (Polynomials)
// ============================================================================

/// Generic numeric trait for polynomial coefficients
pub trait Numeric:
    Clone
    + PartialEq
    + fmt::Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    fn zero() -> Self;
    fn is_zero(&self) -> bool;
}

impl Numeric for i64 {
    fn zero() -> Self {
        0
    }
    fn is_zero(&self) -> bool {
        *self == 0
    }
}

impl Numeric for f64 {
    fn zero() -> Self {
        0.0
    }
    fn is_zero(&self) -> bool {
        self.abs() < 1e-10
    }
}

impl Numeric for Rational {
    fn zero() -> Self {
        Rational::new(0, 1)
    }
    fn is_zero(&self) -> bool {
        self.numer == 0
    }
}

impl Numeric for Complex {
    fn zero() -> Self {
        Complex::from_real_imag(0.0, 0.0)
    }
    fn is_zero(&self) -> bool {
        self.real.abs() < 1e-10 && self.imag.abs() < 1e-10
    }
}

/// A term in a polynomial: coefficient * variable^order
#[derive(Debug, Clone, PartialEq)]
pub struct Term<T: Numeric> {
    order: usize,
    coeff: T,
}

impl<T: Numeric> Term<T> {
    pub fn new(order: usize, coeff: T) -> Self {
        Term { order, coeff }
    }

    pub fn order(&self) -> usize {
        self.order
    }

    pub fn coeff(&self) -> &T {
        &self.coeff
    }
}

/// A polynomial with a single variable
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial<T: Numeric> {
    variable: char,
    terms: Vec<Term<T>>,
}

impl<T: Numeric> Polynomial<T> {
    /// Create a new polynomial with given variable
    pub fn new(variable: char, mut terms: Vec<Term<T>>) -> Self {
        // Remove zero coefficient terms
        terms.retain(|t| !t.coeff.is_zero());
        // Sort by order (highest first)
        terms.sort_by(|a, b| b.order.cmp(&a.order));
        Polynomial { variable, terms }
    }

    pub fn variable(&self) -> char {
        self.variable
    }

    pub fn terms(&self) -> &[Term<T>] {
        &self.terms
    }

    fn same_variable(&self, other: &Polynomial<T>) -> bool {
        self.variable == other.variable
    }

    /// Add two term lists
    fn add_terms(t1: &[Term<T>], t2: &[Term<T>]) -> Vec<Term<T>> {
        match (t1.first(), t2.first()) {
            (None, _) => t2.to_vec(),
            (_, None) => t1.to_vec(),
            (Some(term1), Some(term2)) => {
                if term1.order > term2.order {
                    let mut result = vec![term1.clone()];
                    result.extend(Self::add_terms(&t1[1..], t2));
                    result
                } else if term1.order < term2.order {
                    let mut result = vec![term2.clone()];
                    result.extend(Self::add_terms(t1, &t2[1..]));
                    result
                } else {
                    // Same order - add coefficients
                    let new_coeff = term1.coeff.clone() + term2.coeff.clone();
                    let mut result = if !new_coeff.is_zero() {
                        vec![Term::new(term1.order, new_coeff)]
                    } else {
                        vec![]
                    };
                    result.extend(Self::add_terms(&t1[1..], &t2[1..]));
                    result
                }
            }
        }
    }

    /// Multiply a term by all terms in a list
    fn mul_term_by_all_terms(t: &Term<T>, terms: &[Term<T>]) -> Vec<Term<T>> {
        terms
            .iter()
            .map(|term| Term::new(t.order + term.order, t.coeff.clone() * term.coeff.clone()))
            .collect()
    }

    /// Multiply two term lists
    fn mul_terms(t1: &[Term<T>], t2: &[Term<T>]) -> Vec<Term<T>> {
        if t1.is_empty() {
            vec![]
        } else {
            let first_products = Self::mul_term_by_all_terms(&t1[0], t2);
            let rest_products = Self::mul_terms(&t1[1..], t2);
            Self::add_terms(&first_products, &rest_products)
        }
    }
}

// Polynomial arithmetic
impl<T: Numeric> Add for Polynomial<T> {
    type Output = Polynomial<T>;

    fn add(self, other: Polynomial<T>) -> Polynomial<T> {
        assert!(
            self.same_variable(&other),
            "Polynomials must have the same variable"
        );
        let new_terms = Self::add_terms(&self.terms, &other.terms);
        Polynomial::new(self.variable, new_terms)
    }
}

impl<T: Numeric> Mul for Polynomial<T> {
    type Output = Polynomial<T>;

    fn mul(self, other: Polynomial<T>) -> Polynomial<T> {
        assert!(
            self.same_variable(&other),
            "Polynomials must have the same variable"
        );
        let new_terms = Self::mul_terms(&self.terms, &other.terms);
        Polynomial::new(self.variable, new_terms)
    }
}

// Display implementation for polynomials
impl<T: Numeric + fmt::Display> fmt::Display for Polynomial<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }

        for (i, term) in self.terms.iter().enumerate() {
            if i > 0 && !term.coeff.is_zero() {
                write!(f, " + ")?;
            }

            match term.order {
                0 => write!(f, "{}", term.coeff)?,
                1 => write!(f, "{}{}", term.coeff, self.variable)?,
                n => write!(f, "{}{}^{}", term.coeff, self.variable, n)?,
            }
        }
        Ok(())
    }
}

// Exercise 2.87: Zero test for polynomials
impl<T: Numeric> GenericZero for Polynomial<T> {
    fn is_zero(&self) -> bool {
        self.terms.is_empty() || self.terms.iter().all(|t| t.coeff.is_zero())
    }
}

// Exercise 2.88: Polynomial negation
impl<T: Numeric> std::ops::Neg for Polynomial<T>
where
    T: std::ops::Neg<Output = T>,
{
    type Output = Polynomial<T>;

    fn neg(self) -> Polynomial<T> {
        let negated_terms: Vec<Term<T>> = self
            .terms
            .into_iter()
            .map(|t| Term::new(t.order, -t.coeff))
            .collect();
        Polynomial::new(self.variable, negated_terms)
    }
}

// Exercise 2.88: Polynomial subtraction
impl<T: Numeric> Sub for Polynomial<T>
where
    T: std::ops::Neg<Output = T>,
{
    type Output = Polynomial<T>;

    fn sub(self, other: Polynomial<T>) -> Polynomial<T> {
        self + (-other)
    }
}

// Exercise 2.91: Polynomial division
impl<T: Numeric> Polynomial<T>
where
    T: std::ops::Neg<Output = T>,
{
    /// Divide two polynomials, returning (quotient, remainder)
    pub fn div_poly(self, divisor: Polynomial<T>) -> (Polynomial<T>, Polynomial<T>) {
        assert!(
            self.same_variable(&divisor),
            "Polynomials must have the same variable"
        );
        let (q_terms, r_terms) = Self::div_terms(&self.terms, &divisor.terms);
        (
            Polynomial::new(self.variable, q_terms),
            Polynomial::new(self.variable, r_terms),
        )
    }

    fn div_terms(dividend: &[Term<T>], divisor: &[Term<T>]) -> (Vec<Term<T>>, Vec<Term<T>>) {
        if dividend.is_empty() {
            return (vec![], vec![]);
        }

        let t1 = &dividend[0];
        let t2 = &divisor[0];

        if t2.order > t1.order {
            (vec![], dividend.to_vec())
        } else {
            let new_coeff = t1.coeff.clone() / t2.coeff.clone();
            let new_order = t1.order - t2.order;
            let new_term = Term::new(new_order, new_coeff);

            // Multiply divisor by new term
            let product = Self::mul_term_by_all_terms(&new_term, divisor);
            // Subtract from dividend
            let rest_dividend = Self::subtract_terms(dividend, &product);
            // Recursively divide
            let (rest_quot, rest_rem) = Self::div_terms(&rest_dividend, divisor);

            let mut quotient = vec![new_term];
            quotient.extend(rest_quot);
            (quotient, rest_rem)
        }
    }

    fn subtract_terms(t1: &[Term<T>], t2: &[Term<T>]) -> Vec<Term<T>>
    where
        T: std::ops::Neg<Output = T>,
    {
        let negated: Vec<Term<T>> = t2
            .iter()
            .map(|t| Term::new(t.order, -t.coeff.clone()))
            .collect();
        Self::add_terms(t1, &negated)
    }
}

// Negation for basic numeric types (needed for subtraction)
impl std::ops::Neg for Rational {
    type Output = Rational;
    fn neg(self) -> Rational {
        Rational::new(-self.numer, self.denom)
    }
}

impl std::ops::Neg for Complex {
    type Output = Complex;
    fn neg(self) -> Complex {
        Complex::from_real_imag(-self.real, -self.imag)
    }
}

// Display implementations
impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denom == 1 {
            write!(f, "{}", self.numer)
        } else {
            write!(f, "{}/{}", self.numer, self.denom)
        }
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.imag >= 0.0 {
            write!(f, "{}+{}i", self.real, self.imag)
        } else {
            write!(f, "{}{}i", self.real, self.imag)
        }
    }
}

// ============================================================================
// Tests and Examples
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational_arithmetic() {
        let r1 = Rational::new(1, 2);
        let r2 = Rational::new(1, 3);
        let sum = r1 + r2;
        assert_eq!(sum, Rational::new(5, 6));
    }

    #[test]
    fn test_complex_arithmetic() {
        let c1 = Complex::from_real_imag(3.0, 4.0);
        let c2 = Complex::from_real_imag(1.0, 2.0);
        let sum = c1 + c2;
        assert!((sum.real - 4.0).abs() < 1e-10);
        assert!((sum.imag - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_generic_number_operations() {
        let n1 = Number::SchemeNumber(5);
        let n2 = Number::SchemeNumber(3);
        let sum = n1 + n2;
        assert_eq!(sum, Number::SchemeNumber(8));
    }

    #[test]
    fn test_coercion() {
        let n = Number::SchemeNumber(5);
        let c = Number::Complex(Complex::from_real_imag(3.0, 4.0));
        let sum = n + c;
        if let Number::Complex(result) = sum {
            assert!((result.real - 8.0).abs() < 1e-10);
            assert!((result.imag - 4.0).abs() < 1e-10);
        } else {
            panic!("Expected Complex result");
        }
    }

    #[test]
    fn test_polynomial_addition() {
        // 3x^2 + 2x + 1
        let p1 = Polynomial::new('x', vec![Term::new(2, 3), Term::new(1, 2), Term::new(0, 1)]);

        // 2x^2 + x + 5
        let p2 = Polynomial::new('x', vec![Term::new(2, 2), Term::new(1, 1), Term::new(0, 5)]);

        // Result: 5x^2 + 3x + 6
        let sum = p1 + p2;
        assert_eq!(sum.terms.len(), 3);
        assert_eq!(sum.terms[0], Term::new(2, 5));
        assert_eq!(sum.terms[1], Term::new(1, 3));
        assert_eq!(sum.terms[2], Term::new(0, 6));
    }

    #[test]
    fn test_polynomial_multiplication() {
        // (x + 1)
        let p1 = Polynomial::new('x', vec![Term::new(1, 1), Term::new(0, 1)]);

        // (x + 2)
        let p2 = Polynomial::new('x', vec![Term::new(1, 1), Term::new(0, 2)]);

        // Result: x^2 + 3x + 2
        let product = p1 * p2;
        assert_eq!(product.terms.len(), 3);
        assert_eq!(product.terms[0], Term::new(2, 1));
        assert_eq!(product.terms[1], Term::new(1, 3));
        assert_eq!(product.terms[2], Term::new(0, 2));
    }

    #[test]
    fn test_polynomial_with_rational_coefficients() {
        // (1/2)x + 1/3
        let p1: Polynomial<Rational> = Polynomial::new(
            'x',
            vec![
                Term::new(1, Rational::new(1, 2)),
                Term::new(0, Rational::new(1, 3)),
            ],
        );

        // (1/4)x + 1/6
        let p2: Polynomial<Rational> = Polynomial::new(
            'x',
            vec![
                Term::new(1, Rational::new(1, 4)),
                Term::new(0, Rational::new(1, 6)),
            ],
        );

        let sum = p1 + p2;
        // (3/4)x + 1/2
        assert_eq!(sum.terms[0].coeff(), &Rational::new(3, 4));
        assert_eq!(sum.terms[1].coeff(), &Rational::new(1, 2));
    }
}

// Example usage in main
pub fn run_examples() {
    println!("=== Section 2.5: Generic Operations ===\n");

    // Generic arithmetic
    println!("Generic Arithmetic:");
    let n1 = Number::SchemeNumber(10);
    let n2 = Number::SchemeNumber(5);
    println!("10 + 5 = {:?}", n1.clone() + n2.clone());
    println!("10 * 5 = {:?}", n1 * n2);

    // Rational numbers
    println!("\nRational Numbers:");
    let r1 = Rational::new(1, 2);
    let r2 = Rational::new(1, 3);
    println!("1/2 + 1/3 = {}", r1 + r2);

    // Complex numbers
    println!("\nComplex Numbers:");
    let c1 = Complex::from_real_imag(3.0, 4.0);
    let c2 = Complex::from_real_imag(1.0, 2.0);
    println!("(3+4i) + (1+2i) = {}", c1 + c2);

    // Coercion
    println!("\nType Coercion:");
    let n = Number::SchemeNumber(5);
    let c = Number::Complex(Complex::from_real_imag(3.0, 4.0));
    println!("5 + (3+4i) = {:?}", n + c);

    // Polynomials
    println!("\nPolynomial Arithmetic:");
    let p1 = Polynomial::new('x', vec![Term::new(2, 1), Term::new(0, -1)]);
    println!("p1 = {}", p1);

    let p2 = Polynomial::new('x', vec![Term::new(1, 1), Term::new(0, 1)]);
    println!("p2 = {}", p2);

    let sum = p1.clone() + p2.clone();
    println!("p1 + p2 = {}", sum);

    let product = p1 * p2;
    println!("p1 * p2 = {}", product);

    // Polynomial with rational coefficients
    println!("\nPolynomial with Rational Coefficients:");
    let pr: Polynomial<Rational> = Polynomial::new(
        'y',
        vec![
            Term::new(2, Rational::new(1, 2)),
            Term::new(1, Rational::new(2, 3)),
            Term::new(0, Rational::new(1, 4)),
        ],
    );
    println!("p(y) = {}", pr);
}
