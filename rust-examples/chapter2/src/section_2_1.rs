//! 2.1절: 데이터 추상화 소개 (Section 2.1: Introduction to Data Abstraction)
//!
//! 이 섹션에서는 유리수와 구간 산술을 통해 데이터 추상화를 설명합니다.

use std::fmt;

// =============================================================================
// 2.1.1: 유리수 산술 연산 (Arithmetic Operations for Rational Numbers)
// =============================================================================

/// 분자와 분모로 표현된 유리수
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rational {
    num: i64,
    denom: i64,
}

impl Rational {
    /// 새로운 유리수를 생성하고, 기약분수로 약분합니다.
    pub fn new(n: i64, d: i64) -> Self {
        assert!(d != 0, "분모는 0이 될 수 없습니다");
        let g = gcd(n, d);
        // 부호 정규화: 분모는 항상 양수
        let (num, denom) = if d < 0 {
            (-n / g, -d / g)
        } else {
            (n / g, d / g)
        };
        Rational { num, denom }
    }

    /// 분자를 반환합니다.
    pub fn num(&self) -> i64 {
        self.num
    }

    /// 분모를 반환합니다.
    pub fn denom(&self) -> i64 {
        self.denom
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}/{}", self.num, self.denom)
    }
}

/// 유클리드 호제법을 사용한 최대공약수 (GCD)
pub fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

/// 두 유리수의 덧셈
pub fn add_rat(x: &Rational, y: &Rational) -> Rational {
    Rational::new(
        x.num() * y.denom() + y.num() * x.denom(),
        x.denom() * y.denom(),
    )
}

/// 두 유리수의 뺄셈
pub fn sub_rat(x: &Rational, y: &Rational) -> Rational {
    Rational::new(
        x.num() * y.denom() - y.num() * x.denom(),
        x.denom() * y.denom(),
    )
}

/// 두 유리수의 곱셈
pub fn mul_rat(x: &Rational, y: &Rational) -> Rational {
    Rational::new(x.num() * y.num(), x.denom() * y.denom())
}

/// 두 유리수의 나눗셈
pub fn div_rat(x: &Rational, y: &Rational) -> Rational {
    Rational::new(x.num() * y.denom(), x.denom() * y.num())
}

/// 두 유리수의 등가성 검사
pub fn equal_rat(x: &Rational, y: &Rational) -> bool {
    x.num() * y.denom() == y.num() * x.denom()
}

// =============================================================================
// 2.1.2: 추상화 장벽 - 점과 선분 (Abstraction Barriers - Points and Segments)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    start: Point,
    end: Point,
}

impl Segment {
    pub fn new(start: Point, end: Point) -> Self {
        Segment { start, end }
    }

    pub fn start(&self) -> Point {
        self.start
    }

    pub fn end(&self) -> Point {
        self.end
    }
}

pub fn midpoint_segment(seg: &Segment) -> Point {
    Point::new(
        (seg.start().x + seg.end().x) / 2.0,
        (seg.start().y + seg.end().y) / 2.0,
    )
}

// =============================================================================
// 2.1.4: 구간 산술 (Interval Arithmetic)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    lower: f64,
    upper: f64,
}

impl Interval {
    /// 하한과 상한으로 새로운 구간을 생성합니다.
    pub fn new(a: f64, b: f64) -> Self {
        Interval {
            lower: a.min(b),
            upper: a.max(b),
        }
    }

    /// 중심과 너비(width)로 구간을 생성한다
    /// (Creates an interval from center and width).
    pub fn from_center_width(c: f64, w: f64) -> Self {
        Interval::new(c - w, c + w)
    }

    /// 중심과 백분율 공차(tolerance)로 구간을 생성한다
    /// (Creates an interval from center and percent tolerance).
    pub fn from_center_percent(c: f64, p: f64) -> Self {
        let w = c.abs() * p / 100.0;
        Interval::from_center_width(c, w)
    }

    pub fn lower_bound(&self) -> f64 {
        self.lower
    }

    pub fn upper_bound(&self) -> f64 {
        self.upper
    }

    pub fn center(&self) -> f64 {
        (self.lower + self.upper) / 2.0
    }

    pub fn width(&self) -> f64 {
        (self.upper - self.lower) / 2.0
    }

    pub fn percent(&self) -> f64 {
        100.0 * self.width() / self.center().abs()
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {}]", self.lower, self.upper)
    }
}

pub fn add_interval(x: &Interval, y: &Interval) -> Interval {
    Interval::new(
        x.lower_bound() + y.lower_bound(),
        x.upper_bound() + y.upper_bound(),
    )
}

pub fn sub_interval(x: &Interval, y: &Interval) -> Interval {
    Interval::new(
        x.lower_bound() - y.upper_bound(),
        x.upper_bound() - y.lower_bound(),
    )
}

pub fn mul_interval(x: &Interval, y: &Interval) -> Interval {
    let p1 = x.lower_bound() * y.lower_bound();
    let p2 = x.lower_bound() * y.upper_bound();
    let p3 = x.upper_bound() * y.lower_bound();
    let p4 = x.upper_bound() * y.upper_bound();
    Interval::new(p1.min(p2).min(p3).min(p4), p1.max(p2).max(p3).max(p4))
}

pub fn div_interval(x: &Interval, y: &Interval) -> Result<Interval, &'static str> {
    // 0을 포함하는 구간으로 나누는지 확인
    if y.lower_bound() <= 0.0 && y.upper_bound() >= 0.0 {
        return Err("Division by interval spanning zero");
    }
    Ok(mul_interval(
        x,
        &Interval::new(1.0 / y.upper_bound(), 1.0 / y.lower_bound()),
    ))
}

// 병렬 저항 공식
pub fn par1(r1: &Interval, r2: &Interval) -> Result<Interval, &'static str> {
    div_interval(&mul_interval(r1, r2), &add_interval(r1, r2))
}

pub fn par2(r1: &Interval, r2: &Interval) -> Result<Interval, &'static str> {
    let one = Interval::new(1.0, 1.0);
    div_interval(
        &one,
        &add_interval(&div_interval(&one, r1)?, &div_interval(&one, r2)?),
    )
}

// =============================================================================
// 테스트 (Tests)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational_arithmetic() {
        let one_half = Rational::new(1, 2);
        let one_third = Rational::new(1, 3);

        assert_eq!(format!("{}", one_half), "1/2");
        assert_eq!(format!("{}", one_third), "1/3");

        let sum = add_rat(&one_half, &one_third);
        assert_eq!(format!("{}", sum), "5/6");

        let product = mul_rat(&one_half, &one_third);
        assert_eq!(format!("{}", product), "1/6");

        let double_third = add_rat(&one_third, &one_third);
        assert_eq!(format!("{}", double_third), "2/3");
    }

    #[test]
    fn test_rational_normalization() {
        // 부호 정규화 테스트 (Test sign normalization)
        let r1 = Rational::new(-1, 2);
        assert_eq!(r1.num(), -1);
        assert_eq!(r1.denom(), 2);

        let r2 = Rational::new(1, -2);
        assert_eq!(r2.num(), -1);
        assert_eq!(r2.denom(), 2);

        let r3 = Rational::new(-1, -2);
        assert_eq!(r3.num(), 1);
        assert_eq!(r3.denom(), 2);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(48, 18), 6);
        assert_eq!(gcd(18, 48), 6);
        assert_eq!(gcd(-48, 18), 6);
        assert_eq!(gcd(48, -18), 6);
    }

    #[test]
    fn test_point_and_segment() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(4.0, 4.0);
        let seg = Segment::new(p1, p2);

        let mid = midpoint_segment(&seg);
        assert_eq!(mid.x, 2.0);
        assert_eq!(mid.y, 2.0);
    }

    #[test]
    fn test_interval_arithmetic() {
        let i1 = Interval::new(2.0, 4.0);
        let i2 = Interval::new(1.0, 3.0);

        let sum = add_interval(&i1, &i2);
        assert_eq!(sum.lower_bound(), 3.0);
        assert_eq!(sum.upper_bound(), 7.0);

        let diff = sub_interval(&i1, &i2);
        assert_eq!(diff.lower_bound(), -1.0);
        assert_eq!(diff.upper_bound(), 3.0);

        let prod = mul_interval(&i1, &i2);
        assert_eq!(prod.lower_bound(), 2.0);
        assert_eq!(prod.upper_bound(), 12.0);
    }

    #[test]
    fn test_interval_division() {
        let i1 = Interval::new(2.0, 4.0);
        let i2 = Interval::new(1.0, 2.0);

        let quot = div_interval(&i1, &i2).unwrap();
        assert_eq!(quot.lower_bound(), 1.0);
        assert_eq!(quot.upper_bound(), 4.0);

        // 0을 포함하는 구간으로 나눗셈 테스트 (Test division by interval spanning zero)
        let i3 = Interval::new(-1.0, 1.0);
        assert!(div_interval(&i1, &i3).is_err());
    }

    #[test]
    fn test_interval_center_width() {
        let i = Interval::from_center_width(3.5, 0.15);
        assert_eq!(i.lower_bound(), 3.35);
        assert_eq!(i.upper_bound(), 3.65);
        assert_eq!(i.center(), 3.5);
        assert!((i.width() - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_interval_percent() {
        let i = Interval::from_center_percent(6.8, 10.0);
        assert!((i.lower_bound() - 6.12).abs() < 0.01);
        assert!((i.upper_bound() - 7.48).abs() < 0.01);
        assert!((i.percent() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_parallel_resistors() {
        // 10% 공차의 6.8옴 (6.8 ohm with 10% tolerance)
        let r1 = Interval::from_center_percent(6.8, 10.0);
        // 5% 공차의 4.7옴 (4.7 ohm with 5% tolerance)
        let r2 = Interval::from_center_percent(4.7, 5.0);

        let result1 = par1(&r1, &r2).unwrap();
        let result2 = par2(&r1, &r2).unwrap();

        // 둘 다 유효한 구간이어야 하지만, 구간 산술의 의존성 문제로
        // 값이 다를 수 있다 (Both should give valid intervals, though they may differ
        // due to the dependency problem in interval arithmetic)
        println!("par1: {}", result1);
        println!("par2: {}", result2);

        // par2가 더 좁은 경계를 제공해야 한다 (par2 should give tighter bounds)
        assert!(result2.width() <= result1.width());
    }
}
