//! Section 1.1: The Elements of Programming
//!
//! Every powerful programming language has three mechanisms:
//! - **Primitive expressions**: the simplest entities
//! - **Means of combination**: compound elements from simpler ones
//! - **Means of abstraction**: compound elements can be named and manipulated

/// Computes the square of a number.
///
/// # Examples
/// ```
/// use sicp_chapter1::square;
/// assert_eq!(square(5), 25);
/// assert_eq!(square(3), 9);
/// ```
pub fn square(x: i64) -> i64 {
    x * x
}

/// Computes the sum of squares of two numbers.
///
/// # Examples
/// ```
/// use sicp_chapter1::sum_of_squares;
/// assert_eq!(sum_of_squares(3, 4), 25);
/// ```
pub fn sum_of_squares(x: i64, y: i64) -> i64 {
    square(x) + square(y)
}

/// Computes the absolute value of a number.
///
/// # Examples
/// ```
/// use sicp_chapter1::abs;
/// assert_eq!(abs(-5), 5);
/// assert_eq!(abs(5), 5);
/// ```
pub fn abs(x: i64) -> i64 {
    if x < 0 { -x } else { x }
}

/// Computes the average of two numbers.
pub fn average(x: f64, y: f64) -> f64 {
    (x + y) / 2.0
}

/// Newton's method for computing square roots.
///
/// # Examples
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
