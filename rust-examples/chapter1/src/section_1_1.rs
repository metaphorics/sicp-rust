//! 1.1절: 프로그래밍의 요소 (Section 1.1: Elements of Programming)
//!
//! 모든 강력한 프로그래밍 언어는 세 가지 메커니즘을 가지고 있습니다:
//! - **원시 표현식 (Primitive expressions)**: 가장 단순한 개체들
//! - **조합 수단 (Means of combination)**: 더 단순한 요소들로부터 복합 요소를 만드는 방법
//! - **추상화 수단 (Means of abstraction)**: 복합 요소들에 이름을 붙이고 조작할 수 있는 방법

/// 숫자의 제곱을 계산합니다 (Calculates the square of a number).
///
/// # 예시 (Examples)
/// ```
/// use sicp_chapter1::square;
/// assert_eq!(square(5), 25);
/// assert_eq!(square(3), 9);
/// ```
pub fn square(x: i64) -> i64 {
    x * x
}

/// 두 숫자의 제곱의 합을 계산합니다 (Calculates the sum of squares of two numbers).
///
/// # 예시 (Examples)
/// ```
/// use sicp_chapter1::sum_of_squares;
/// assert_eq!(sum_of_squares(3, 4), 25);
/// ```
pub fn sum_of_squares(x: i64, y: i64) -> i64 {
    square(x) + square(y)
}

/// 숫자의 절댓값을 계산합니다 (Calculates the absolute value of a number).
///
/// # 예시 (Examples)
/// ```
/// use sicp_chapter1::abs;
/// assert_eq!(abs(-5), 5);
/// assert_eq!(abs(5), 5);
/// ```
pub fn abs(x: i64) -> i64 {
    if x < 0 { -x } else { x }
}

/// 두 숫자의 평균을 계산합니다 (Calculates the average of two numbers).
pub fn average(x: f64, y: f64) -> f64 {
    (x + y) / 2.0
}

/// 제곱근을 계산하기 위한 뉴턴의 방법 (Newton's method for calculating square roots).
///
/// # 예시 (Examples)
/// ```
/// use sicp_chapter1::sqrt;
/// let result = sqrt(9.0);
/// assert!((result - 3.0).abs() < 0.0001);
/// ```
pub fn sqrt(x: f64) -> f64 {
    fn good_enough(guess: f64, x: f64) -> bool {
        (square_f64(guess) - x).abs() < 0.001
    }

    fn improve(guess: f64, x: f64) -> f64 {
        average(guess, x / guess)
    }

    fn sqrt_iter(guess: f64, x: f64) -> f64 {
        if good_enough(guess, x) {
            guess
        } else {
            sqrt_iter(improve(guess, x), x)
        }
    }

    sqrt_iter(1.0, x)
}

fn square_f64(x: f64) -> f64 {
    x * x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square() {
        assert_eq!(square(5), 25);
        assert_eq!(square(0), 0);
        assert_eq!(square(-3), 9);
    }

    #[test]
    fn test_sum_of_squares() {
        assert_eq!(sum_of_squares(3, 4), 25);
        assert_eq!(sum_of_squares(0, 0), 0);
    }

    #[test]
    fn test_sqrt() {
        let result = sqrt(2.0);
        assert!((result * result - 2.0).abs() < 0.001);
    }
}
