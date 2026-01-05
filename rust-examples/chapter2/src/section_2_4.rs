//! 2.4절: 추상 데이터의 다중 표현
//! (Section 2.4: Multiple Representations for Abstract Data)
//!
//! 이 섹션에서는 다음을 보여줍니다:
//! - 복소수의 다중 표현 (직교 좌표계와 극 좌표계)
//! - Rust 열거형(enum)을 사용한 태그된 데이터(tagged data)
//! - 연산 테이블을 사용한 데이터 주도 프로그래밍 (Data-directed programming)
//! - 메시지 전달 스타일 프로그래밍 (Message passing style programming)

use std::collections::HashMap;

// ============================================================================
// 2.4.1절: 복소수 표현 (Representations for Complex Numbers)
// ============================================================================

/// 복소수 연산을 정의하는 트레이트
pub trait ComplexOps {
    fn real_part(&self) -> f64;
    fn imag_part(&self) -> f64;
    fn magnitude(&self) -> f64;
    fn angle(&self) -> f64;
}

// Ben의 직교 좌표계 표현 (Ben's rectangular representation)
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

// Alyssa의 극 좌표계 표현 (Alyssa's polar representation)
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

// 제네릭 산술 연산
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
// 2.4.2절: 태그된 데이터 (Tagged Data)
// ============================================================================

/// 태그된 표현을 가진 복소수
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Complex {
    Rectangular(f64, f64), // (x, y)
    Polar(f64, f64),       // (r, angle)
}

impl Complex {
    /// 타입 태그를 문자열로 반환
    pub fn type_tag(&self) -> &str {
        match self {
            Complex::Rectangular(_, _) => "rectangular",
            Complex::Polar(_, _) => "polar",
        }
    }

    /// 내용을 튜플로 추출
    pub fn contents(&self) -> (f64, f64) {
        match self {
            Complex::Rectangular(x, y) => (*x, *y),
            Complex::Polar(r, a) => (*r, *a),
        }
    }

    /// 직교 좌표계 표현인지 확인
    pub fn is_rectangular(&self) -> bool {
        matches!(self, Complex::Rectangular(_, _))
    }

    /// 극 좌표계 표현인지 확인
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

/// 적절한 표현을 선택하는 생성자 함수들
pub fn make_from_real_imag(x: f64, y: f64) -> Complex {
    Complex::Rectangular(x, y)
}

pub fn make_from_mag_ang(r: f64, a: f64) -> Complex {
    Complex::Polar(r, a)
}

// 태그된 복소수에 대한 산술 연산
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
// 2.4.3절: 데이터 주도 프로그래밍과 가법성
// (Data-Directed Programming and Additivity)
// ============================================================================

/// 연산 테이블을 위한 함수 포인터 타입 (제로 비용, 힙 할당 없음)
pub type OperationFn = fn(f64, f64) -> f64;
pub type ConstructorFn = fn(f64, f64) -> Complex;

/// 데이터 주도 프로그래밍을 위한 연산 테이블
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

/// Ben의 직교 좌표계 패키지 설치 (Install Ben's rectangular package)
pub fn install_rectangular_package(table: &mut OperationTable) {
    // 내부 프로시저
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

    // 시스템의 나머지 부분에 대한 인터페이스
    table.put("real-part", "rectangular", real_part);
    table.put("imag-part", "rectangular", imag_part);
    table.put("magnitude", "rectangular", magnitude);
    table.put("angle", "rectangular", angle);

    table.put_constructor("make-from-real-imag", "rectangular", Complex::Rectangular);
    table.put_constructor("make-from-mag-ang", "rectangular", make_rect_from_mag_ang);
}

/// Alyssa의 극 좌표계 패키지 설치 (Install Alyssa's polar package)
pub fn install_polar_package(table: &mut OperationTable) {
    // 내부 프로시저
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

    // 시스템의 나머지 부분에 대한 인터페이스
    table.put("real-part", "polar", real_part);
    table.put("imag-part", "polar", imag_part);
    table.put("magnitude", "polar", magnitude);
    table.put("angle", "polar", angle);

    table.put_constructor("make-from-real-imag", "polar", make_polar_from_real_imag);
    table.put_constructor("make-from-mag-ang", "polar", Complex::Polar);
}

/// 제네릭 연산 적용
pub fn apply_generic(op: &str, z: &Complex, table: &OperationTable) -> f64 {
    let type_tag = z.type_tag();
    let contents = z.contents();

    if let Some(proc) = table.get(op, type_tag) {
        proc(contents.0, contents.1)
    } else {
        panic!("No method for type: {} ({})", op, type_tag)
    }
}

/// 연산 테이블을 사용한 제네릭 선택자
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

/// 연산 테이블을 사용한 생성자
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
// 메시지 전달 (Message Passing)
// ============================================================================

/// 메시지 전달 스타일 복소수
pub type MessagePassingComplex = Box<dyn Fn(&str) -> f64>;

/// 메시지 전달을 사용하여 실수부와 허수부로 복소수 생성
pub fn make_from_real_imag_message(x: f64, y: f64) -> MessagePassingComplex {
    Box::new(move |op: &str| match op {
        "real-part" => x,
        "imag-part" => y,
        "magnitude" => (x.powi(2) + y.powi(2)).sqrt(),
        "angle" => y.atan2(x),
        _ => panic!("Unknown op: MAKE-FROM-REAL-IMAG {}", op),
    })
}

/// 메시지 전달을 사용하여 크기와 각도로 복소수 생성
pub fn make_from_mag_ang_message(r: f64, a: f64) -> MessagePassingComplex {
    Box::new(move |op: &str| match op {
        "real-part" => r * a.cos(),
        "imag-part" => r * a.sin(),
        "magnitude" => r,
        "angle" => a,
        _ => panic!("Unknown op: MAKE-FROM-MAG-ANG {}", op),
    })
}

/// 메시지 전달 복소수에 제네릭 연산 적용
pub fn apply_generic_message(op: &str, arg: &MessagePassingComplex) -> f64 {
    arg(op)
}

// ============================================================================
// 테스트 (Tests)
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
