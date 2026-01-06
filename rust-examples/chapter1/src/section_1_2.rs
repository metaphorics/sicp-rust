//! 1.2절: 프로시저와 그들이 생성하는 프로세스 (Section 1.2: Procedures and the Processes They Generate)
//!
//! 프로시저는 계산 프로세스의 국소적 진화를 위한 패턴입니다.
//! 이 섹션에서는 다음을 다룹니다:
//! - 선형 재귀와 반복 (Linear recursion and iteration)
//! - 트리 재귀 (Tree recursion)
//! - 증가 차수 (Orders of growth)

/// 선형 재귀 프로세스를 사용한 팩토리얼 (Factorial using a linear recursive process).
/// 이 프로세스는 지연된 연산들의 체인을 구축합니다 (This process builds up a chain of deferred operations).
///
/// # 예시 (Examples)
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

/// 선형 반복 프로세스를 사용한 팩토리얼 (Factorial using a linear iterative process).
/// Rust는 꼬리 호출 최적화(TCO)를 보장하지 않지만, 이 패턴은
/// 명시적 누산기(accumulator)를 통해 상수 공간을 유지합니다
/// (Rust does not guarantee TCO, but this pattern maintains constant space via an explicit accumulator).
///
/// # 예시 (Examples)
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

/// 이터레이터를 사용한 관용적인 Rust 팩토리얼 (Idiomatic Rust factorial using iterators).
/// 이것이 Rust에서 선호되는 접근 방식이다 (This is the preferred approach in Rust).
///
/// # 예시 (Examples)
/// ```
/// use sicp_chapter1::section_1_2::factorial;
/// assert_eq!(factorial(5), 120);
/// ```
pub fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

/// 트리 재귀를 사용한 피보나치 (Fibonacci using tree recursion).
/// 이것은 지수 시간 복잡도 O(phi^n)를 가진다 (This has exponential time complexity O(phi^n)).
///
/// # 예시 (Examples)
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

/// 반복 프로세스를 사용한 피보나치 (Fibonacci using an iterative process).
/// 선형 시간 O(n), 상수 공간 O(1) (Linear time O(n), constant space O(1)).
///
/// # 예시 (Examples)
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

/// 유클리드 호제법을 사용한 최대공약수(GCD)
/// (Greatest common divisor using Euclidean algorithm).
/// O(log(min(a,b))) 시간 복잡도 (Time complexity O(log(min(a,b)))).
///
/// # 예시 (Examples)
/// ```
/// use sicp_chapter1::section_1_2::gcd;
/// assert_eq!(gcd(206, 40), 2);
/// ```
pub fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// 시범 나눗셈(trial division)을 사용한 소수 판별
/// (Primality test using trial division).
/// O(sqrt(n)) 시간 복잡도 (Time complexity O(sqrt(n))).
///
/// # 예시 (Examples)
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
        } else if n % test_divisor == 0 {
            // Rust에서는 is_multiple_of 대신 % 연산자가 더 일반적입니다 (In Rust, the % operator is more common than is_multiple_of).
            test_divisor
        } else {
            find_divisor(n, test_divisor + 1)
        }
    }
    find_divisor(n, 2) == n
}

/// 연속 제곱을 사용한 빠른 거듭제곱 (Fast exponentiation using successive squaring).
/// O(log n) 시간 복잡도 (Time complexity O(log n)).
///
/// # 예시 (Examples)
/// ```
/// use sicp_chapter1::section_1_2::fast_expt;
/// assert_eq!(fast_expt(2, 10), 1024);
/// ```
pub fn fast_expt(base: u64, exp: u64) -> u64 {
    if exp == 0 {
        1
    } else if exp % 2 == 0 {
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
