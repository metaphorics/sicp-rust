//! Section 2.4: Multiple Representations for Abstract Data
//!
//! This section demonstrates:
//! - Multiple representations for complex numbers (rectangular and polar)
//! - Tagged data using Rust enums
//! - Data-directed programming using operation tables
//! - Message passing style programming

use std::collections::HashMap;

// ============================================================================
// Section 2.4.1: Representations for Complex Numbers
// ============================================================================

/// Trait defining operations on complex numbers
pub trait ComplexOps {
    fn real_part(&self) -> f64;
    fn imag_part(&self) -> f64;
    fn magnitude(&self) -> f64;
    fn angle(&self) -> f64;
}

// Ben's rectangular representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rectangular {
    x: f64,
    y: f64,
}

impl Rectangular {
    pub fn new(x: f64, y: f64) -> Self {
        Rectangular { x, y }
    }

    pub fn from_mag_ang(r: f64, a: f64) -> Self {
        Rectangular {
            x: r * a.cos(),
            y: r * a.sin(),
        }
    }
}

impl ComplexOps for Rectangular {
    fn real_part(&self) -> f64 {
        self.x
    }

    fn imag_part(&self) -> f64 {
        self.y
    }

    fn magnitude(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }

    fn angle(&self) -> f64 {
        self.y.atan2(self.x)
    }
}

// Alyssa's polar representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Polar {
    r: f64,
    angle: f64,
}

impl Polar {
    pub fn new(r: f64, angle: f64) -> Self {
        Polar { r, angle }
    }

    pub fn from_real_imag(x: f64, y: f64) -> Self {
        Polar {
            r: (x.powi(2) + y.powi(2)).sqrt(),
            angle: y.atan2(x),
        }
    }
}

impl ComplexOps for Polar {
    fn real_part(&self) -> f64 {
        self.r * self.angle.cos()
    }

    fn imag_part(&self) -> f64 {
        self.r * self.angle.sin()
    }

    fn magnitude(&self) -> f64 {
        self.r
    }

    fn angle(&self) -> f64 {
        self.angle
    }
}

// Generic arithmetic operations
pub fn add_complex<T: ComplexOps, U: ComplexOps>(z1: &T, z2: &U) -> Rectangular {
    Rectangular::new(
        z1.real_part() + z2.real_part(),
        z1.imag_part() + z2.imag_part(),
    )
}

pub fn sub_complex<T: ComplexOps, U: ComplexOps>(z1: &T, z2: &U) -> Rectangular {
    Rectangular::new(
        z1.real_part() - z2.real_part(),
        z1.imag_part() - z2.imag_part(),
    )
}

pub fn mul_complex<T: ComplexOps, U: ComplexOps>(z1: &T, z2: &U) -> Polar {
    Polar::new(z1.magnitude() * z2.magnitude(), z1.angle() + z2.angle())
}

pub fn div_complex<T: ComplexOps, U: ComplexOps>(z1: &T, z2: &U) -> Polar {
    Polar::new(z1.magnitude() / z2.magnitude(), z1.angle() - z2.angle())
}

// ============================================================================
// Section 2.4.2: Tagged Data
// ============================================================================

/// Complex number with tagged representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Complex {
    Rectangular(f64, f64), // (x, y)
    Polar(f64, f64),       // (r, angle)
}

impl Complex {
    /// Get the type tag as a string
    pub fn type_tag(&self) -> &str {
        match self {
            Complex::Rectangular(_, _) => "rectangular",
            Complex::Polar(_, _) => "polar",
        }
    }

    /// Extract the contents as a tuple
    pub fn contents(&self) -> (f64, f64) {
        match self {
            Complex::Rectangular(x, y) => (*x, *y),
            Complex::Polar(r, a) => (*r, *a),
        }
    }

    /// Check if this is a rectangular representation
    pub fn is_rectangular(&self) -> bool {
        matches!(self, Complex::Rectangular(_, _))
    }

    /// Check if this is a polar representation
    pub fn is_polar(&self) -> bool {
        matches!(self, Complex::Polar(_, _))
    }
}

impl ComplexOps for Complex {
    fn real_part(&self) -> f64 {
        match self {
            Complex::Rectangular(x, _) => *x,
            Complex::Polar(r, a) => r * a.cos(),
        }
    }

    fn imag_part(&self) -> f64 {
        match self {
            Complex::Rectangular(_, y) => *y,
            Complex::Polar(r, a) => r * a.sin(),
        }
    }

    fn magnitude(&self) -> f64 {
        match self {
            Complex::Rectangular(x, y) => (x.powi(2) + y.powi(2)).sqrt(),
            Complex::Polar(r, _) => *r,
        }
    }

    fn angle(&self) -> f64 {
        match self {
            Complex::Rectangular(x, y) => y.atan2(*x),
            Complex::Polar(_, a) => *a,
        }
    }
}

/// Constructor functions that choose the appropriate representation
pub fn make_from_real_imag(x: f64, y: f64) -> Complex {
    Complex::Rectangular(x, y)
}

pub fn make_from_mag_ang(r: f64, a: f64) -> Complex {
    Complex::Polar(r, a)
}

// Arithmetic operations on tagged complex numbers
pub fn add_complex_tagged(z1: &Complex, z2: &Complex) -> Complex {
    make_from_real_imag(
        z1.real_part() + z2.real_part(),
        z1.imag_part() + z2.imag_part(),
    )
}

pub fn sub_complex_tagged(z1: &Complex, z2: &Complex) -> Complex {
    make_from_real_imag(
        z1.real_part() - z2.real_part(),
        z1.imag_part() - z2.imag_part(),
    )
}

pub fn mul_complex_tagged(z1: &Complex, z2: &Complex) -> Complex {
    make_from_mag_ang(z1.magnitude() * z2.magnitude(), z1.angle() + z2.angle())
}

pub fn div_complex_tagged(z1: &Complex, z2: &Complex) -> Complex {
    make_from_mag_ang(z1.magnitude() / z2.magnitude(), z1.angle() - z2.angle())
}

// ============================================================================
// Section 2.4.3: Data-Directed Programming and Additivity
// ============================================================================

/// Function pointer types for operation table (zero-cost, no heap allocation)
pub type OperationFn = fn(f64, f64) -> f64;
pub type ConstructorFn = fn(f64, f64) -> Complex;

/// Operation table for data-directed programming
pub struct OperationTable {
    operations: HashMap<(String, String), OperationFn>,
    constructors: HashMap<(String, String), ConstructorFn>,
}

impl OperationTable {
    pub fn new() -> Self {
        OperationTable {
            operations: HashMap::new(),
            constructors: HashMap::new(),
        }
    }

    pub fn put(&mut self, op: &str, typ: &str, proc: OperationFn) {
        self.operations
            .insert((op.to_string(), typ.to_string()), proc);
    }

    pub fn get(&self, op: &str, typ: &str) -> Option<OperationFn> {
        self.operations
            .get(&(op.to_string(), typ.to_string()))
            .copied()
    }

    pub fn put_constructor(&mut self, op: &str, typ: &str, proc: ConstructorFn) {
        self.constructors
            .insert((op.to_string(), typ.to_string()), proc);
    }

    pub fn get_constructor(&self, op: &str, typ: &str) -> Option<ConstructorFn> {
        self.constructors
            .get(&(op.to_string(), typ.to_string()))
            .copied()
    }
}

impl Default for OperationTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Install Ben's rectangular package
pub fn install_rectangular_package(table: &mut OperationTable) {
    // Internal procedures
    fn real_part(x: f64, _y: f64) -> f64 {
        x
    }

    fn imag_part(_x: f64, y: f64) -> f64 {
        y
    }

    fn magnitude(x: f64, y: f64) -> f64 {
        (x.powi(2) + y.powi(2)).sqrt()
    }

    fn angle(x: f64, y: f64) -> f64 {
        y.atan2(x)
    }

    fn make_rect_from_mag_ang(r: f64, a: f64) -> Complex {
        Complex::Rectangular(r * a.cos(), r * a.sin())
    }

    // Interface to the rest of the system
    table.put("real-part", "rectangular", real_part);
    table.put("imag-part", "rectangular", imag_part);
    table.put("magnitude", "rectangular", magnitude);
    table.put("angle", "rectangular", angle);

    table.put_constructor("make-from-real-imag", "rectangular", Complex::Rectangular);
    table.put_constructor("make-from-mag-ang", "rectangular", make_rect_from_mag_ang);
}

/// Install Alyssa's polar package
pub fn install_polar_package(table: &mut OperationTable) {
    // Internal procedures
    fn magnitude(r: f64, _angle: f64) -> f64 {
        r
    }

    fn angle(_r: f64, angle: f64) -> f64 {
        angle
    }

    fn real_part(r: f64, angle: f64) -> f64 {
        r * angle.cos()
    }

    fn imag_part(r: f64, angle: f64) -> f64 {
        r * angle.sin()
    }

    fn make_polar_from_real_imag(x: f64, y: f64) -> Complex {
        Complex::Polar((x.powi(2) + y.powi(2)).sqrt(), y.atan2(x))
    }

    // Interface to the rest of the system
    table.put("real-part", "polar", real_part);
    table.put("imag-part", "polar", imag_part);
    table.put("magnitude", "polar", magnitude);
    table.put("angle", "polar", angle);

    table.put_constructor("make-from-real-imag", "polar", make_polar_from_real_imag);
    table.put_constructor("make-from-mag-ang", "polar", Complex::Polar);
}

/// Apply a generic operation to arguments
pub fn apply_generic(op: &str, z: &Complex, table: &OperationTable) -> f64 {
    let type_tag = z.type_tag();
    let contents = z.contents();

    if let Some(proc) = table.get(op, type_tag) {
        proc(contents.0, contents.1)
    } else {
        panic!("No method for type: {} ({})", op, type_tag)
    }
}

/// Generic selectors using the operation table
pub fn real_part_generic(z: &Complex, table: &OperationTable) -> f64 {
    apply_generic("real-part", z, table)
}

pub fn imag_part_generic(z: &Complex, table: &OperationTable) -> f64 {
    apply_generic("imag-part", z, table)
}

pub fn magnitude_generic(z: &Complex, table: &OperationTable) -> f64 {
    apply_generic("magnitude", z, table)
}

pub fn angle_generic(z: &Complex, table: &OperationTable) -> f64 {
    apply_generic("angle", z, table)
}

/// Constructors using the operation table
pub fn make_from_real_imag_generic(x: f64, y: f64, table: &OperationTable) -> Complex {
    table
        .get_constructor("make-from-real-imag", "rectangular")
        .expect("Constructor not found")(x, y)
}

pub fn make_from_mag_ang_generic(r: f64, a: f64, table: &OperationTable) -> Complex {
    table
        .get_constructor("make-from-mag-ang", "polar")
        .expect("Constructor not found")(r, a)
}

// ============================================================================
// Message Passing
// ============================================================================

/// Message-passing style complex number
pub type MessagePassingComplex = Box<dyn Fn(&str) -> f64>;

/// Create a complex number from real and imaginary parts using message passing
pub fn make_from_real_imag_message(x: f64, y: f64) -> MessagePassingComplex {
    Box::new(move |op: &str| match op {
        "real-part" => x,
        "imag-part" => y,
        "magnitude" => (x.powi(2) + y.powi(2)).sqrt(),
        "angle" => y.atan2(x),
        _ => panic!("Unknown op: MAKE-FROM-REAL-IMAG {}", op),
    })
}

/// Create a complex number from magnitude and angle using message passing
pub fn make_from_mag_ang_message(r: f64, a: f64) -> MessagePassingComplex {
    Box::new(move |op: &str| match op {
        "real-part" => r * a.cos(),
        "imag-part" => r * a.sin(),
        "magnitude" => r,
        "angle" => a,
        _ => panic!("Unknown op: MAKE-FROM-MAG-ANG {}", op),
    })
}

/// Apply a generic operation to a message-passing complex number
pub fn apply_generic_message(op: &str, arg: &MessagePassingComplex) -> f64 {
    arg(op)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_rectangular_representation() {
        let z = Rectangular::new(3.0, 4.0);
        assert_eq!(z.real_part(), 3.0);
        assert_eq!(z.imag_part(), 4.0);
        assert_eq!(z.magnitude(), 5.0);
        assert!(approx_eq(z.angle(), 0.9272952180016122));
    }

    #[test]
    fn test_polar_representation() {
        let z = Polar::new(5.0, PI / 4.0);
        assert!(approx_eq(z.real_part(), 3.5355339059327373));
        assert!(approx_eq(z.imag_part(), 3.5355339059327373));
        assert_eq!(z.magnitude(), 5.0);
        assert_eq!(z.angle(), PI / 4.0);
    }

    #[test]
    fn test_complex_arithmetic() {
        let z1 = Rectangular::new(3.0, 4.0);
        let z2 = Rectangular::new(1.0, 2.0);

        let sum = add_complex(&z1, &z2);
        assert_eq!(sum.real_part(), 4.0);
        assert_eq!(sum.imag_part(), 6.0);

        let diff = sub_complex(&z1, &z2);
        assert_eq!(diff.real_part(), 2.0);
        assert_eq!(diff.imag_part(), 2.0);

        let prod = mul_complex(&z1, &z2);
        assert!(approx_eq(prod.magnitude(), 5.0 * 2.23606797749979));
        assert!(approx_eq(
            prod.angle(),
            0.9272952180016122 + 1.1071487177940904
        ));

        let quot = div_complex(&z1, &z2);
        assert!(approx_eq(quot.magnitude(), 5.0 / 2.23606797749979));
        assert!(approx_eq(
            quot.angle(),
            0.9272952180016122 - 1.1071487177940904
        ));
    }

    #[test]
    fn test_tagged_complex() {
        let z1 = Complex::Rectangular(3.0, 4.0);
        let z2 = Complex::Polar(5.0, PI / 4.0);

        assert_eq!(z1.type_tag(), "rectangular");
        assert_eq!(z2.type_tag(), "polar");

        assert!(z1.is_rectangular());
        assert!(!z1.is_polar());
        assert!(!z2.is_rectangular());
        assert!(z2.is_polar());

        assert_eq!(z1.real_part(), 3.0);
        assert_eq!(z1.imag_part(), 4.0);
        assert_eq!(z1.magnitude(), 5.0);

        assert!(approx_eq(z2.real_part(), 3.5355339059327373));
        assert!(approx_eq(z2.imag_part(), 3.5355339059327373));
        assert_eq!(z2.magnitude(), 5.0);
        assert_eq!(z2.angle(), PI / 4.0);
    }

    #[test]
    fn test_tagged_arithmetic() {
        let z1 = make_from_real_imag(3.0, 4.0);
        let z2 = make_from_real_imag(1.0, 2.0);

        let sum = add_complex_tagged(&z1, &z2);
        assert!(approx_eq(sum.real_part(), 4.0));
        assert!(approx_eq(sum.imag_part(), 6.0));

        let prod = mul_complex_tagged(&z1, &z2);
        assert!(approx_eq(prod.magnitude(), 5.0 * 2.23606797749979));
    }

    #[test]
    fn test_data_directed_programming() {
        let mut table = OperationTable::new();
        install_rectangular_package(&mut table);
        install_polar_package(&mut table);

        let z1 = Complex::Rectangular(3.0, 4.0);
        let z2 = Complex::Polar(5.0, PI / 4.0);

        assert_eq!(real_part_generic(&z1, &table), 3.0);
        assert_eq!(imag_part_generic(&z1, &table), 4.0);
        assert_eq!(magnitude_generic(&z1, &table), 5.0);

        assert!(approx_eq(
            real_part_generic(&z2, &table),
            3.5355339059327373
        ));
        assert!(approx_eq(
            imag_part_generic(&z2, &table),
            3.5355339059327373
        ));
        assert_eq!(magnitude_generic(&z2, &table), 5.0);
        assert_eq!(angle_generic(&z2, &table), PI / 4.0);
    }

    #[test]
    fn test_generic_constructors() {
        let mut table = OperationTable::new();
        install_rectangular_package(&mut table);
        install_polar_package(&mut table);

        let z1 = make_from_real_imag_generic(3.0, 4.0, &table);
        assert!(z1.is_rectangular());
        assert_eq!(z1.real_part(), 3.0);
        assert_eq!(z1.imag_part(), 4.0);

        let z2 = make_from_mag_ang_generic(5.0, PI / 4.0, &table);
        assert!(z2.is_polar());
        assert_eq!(z2.magnitude(), 5.0);
        assert_eq!(z2.angle(), PI / 4.0);
    }

    #[test]
    fn test_message_passing() {
        let z = make_from_real_imag_message(3.0, 4.0);

        assert_eq!(apply_generic_message("real-part", &z), 3.0);
        assert_eq!(apply_generic_message("imag-part", &z), 4.0);
        assert_eq!(apply_generic_message("magnitude", &z), 5.0);
        assert!(approx_eq(
            apply_generic_message("angle", &z),
            0.9272952180016122
        ));
    }

    #[test]
    fn test_message_passing_polar() {
        let z = make_from_mag_ang_message(5.0, PI / 4.0);

        assert!(approx_eq(
            apply_generic_message("real-part", &z),
            3.5355339059327373
        ));
        assert!(approx_eq(
            apply_generic_message("imag-part", &z),
            3.5355339059327373
        ));
        assert_eq!(apply_generic_message("magnitude", &z), 5.0);
        assert_eq!(apply_generic_message("angle", &z), PI / 4.0);
    }

    #[test]
    #[should_panic(expected = "Unknown op")]
    fn test_message_passing_invalid_op() {
        let z = make_from_real_imag_message(3.0, 4.0);
        apply_generic_message("invalid-op", &z);
    }
}
