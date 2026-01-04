//! Section 1.2: Procedures and the Processes They Generate
//!
//! A procedure is a pattern for the local evolution of a computational process.
//! This section explores:
//! - Linear recursion and iteration
//! - Tree recursion
//! - Orders of growth

/// Factorial using linear recursive process.
/// The process builds up a chain of deferred operations.
///
/// # Examples
/// ```
/// use sicp_chapter1::section_1_2::factorial_recursive;
/// assert_eq!(factorial_recursive(5), 120);
/// ```
pub fn factorial_recursive(n: u64) -> u64 {
    if n == 0 {
        1
    } else {
        n * factorial_recursive(n - 1)
    }
}

/// Factorial using linear iterative process.
/// Rust does not guarantee tail-call optimization, but this pattern
/// maintains constant space through explicit accumulator.
///
/// # Examples
/// ```
/// use sicp_chapter1::section_1_2::factorial_iterative;
/// assert_eq!(factorial_iterative(5), 120);
/// ```
pub fn factorial_iterative(n: u64) -> u64 {
    fn fact_iter(product: u64, counter: u64, max_count: u64) -> u64 {
        if counter > max_count {
            product
        } else {
            fact_iter(product * counter, counter + 1, max_count)
        }
    }
    fact_iter(1, 1, n)
}

/// Idiomatic Rust factorial using iterators.
/// This is the preferred approach in Rust.
///
/// # Examples
/// ```
/// use sicp_chapter1::section_1_2::factorial;
/// assert_eq!(factorial(5), 120);
/// ```
pub fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

/// Fibonacci using tree recursion.
/// This has exponential time complexity O(phi^n).
///
/// # Examples
/// ```
/// use sicp_chapter1::section_1_2::fib_tree;
/// assert_eq!(fib_tree(10), 55);
/// ```
pub fn fib_tree(n: u64) -> u64 {
    if n == 0 {
        0
    } else if n == 1 {
        1
    } else {
        fib_tree(n - 1) + fib_tree(n - 2)
    }
}

/// Fibonacci using iterative process.
/// Linear time O(n), constant space O(1).
///
/// # Examples
/// ```
/// use sicp_chapter1::section_1_2::fib;
/// assert_eq!(fib(10), 55);
/// ```
pub fn fib(n: u64) -> u64 {
    fn fib_iter(a: u64, b: u64, count: u64) -> u64 {
        if count == 0 {
            b
        } else {
            fib_iter(a + b, a, count - 1)
        }
    }
    fib_iter(1, 0, n)
}

/// Greatest Common Divisor using Euclid's algorithm.
/// O(log(min(a,b))) time complexity.
///
/// # Examples
/// ```
/// use sicp_chapter1::section_1_2::gcd;
/// assert_eq!(gcd(206, 40), 2);
/// ```
pub fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Test if n is prime using trial division.
/// O(sqrt(n)) time complexity.
///
/// # Examples
/// ```
/// use sicp_chapter1::section_1_2::is_prime;
/// assert!(is_prime(17));
/// assert!(!is_prime(15));
/// ```
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    fn find_divisor(n: u64, test_divisor: u64) -> u64 {
        if test_divisor * test_divisor > n {
            n
        } else if n.is_multiple_of(test_divisor) {
            test_divisor
        } else {
            find_divisor(n, test_divisor + 1)
        }
    }
    find_divisor(n, 2) == n
}

/// Fast exponentiation using successive squaring.
/// O(log n) time complexity.
///
/// # Examples
/// ```
/// use sicp_chapter1::section_1_2::fast_expt;
/// assert_eq!(fast_expt(2, 10), 1024);
/// ```
pub fn fast_expt(base: u64, exp: u64) -> u64 {
    if exp == 0 {
        1
    } else if exp.is_multiple_of(2) {
        let half = fast_expt(base, exp / 2);
        half * half
    } else {
        base * fast_expt(base, exp - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial_variants() {
        for n in 0..10 {
            assert_eq!(factorial_recursive(n), factorial_iterative(n));
            assert_eq!(factorial_recursive(n), factorial(n));
        }
    }

    #[test]
    fn test_fibonacci_variants() {
        for n in 0..15 {
            assert_eq!(fib_tree(n), fib(n));
        }
    }

    #[test]
    fn test_primes() {
        let primes: Vec<u64> = (2..50).filter(|&n| is_prime(n)).collect();
        assert_eq!(
            primes,
            vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        );
    }
}
