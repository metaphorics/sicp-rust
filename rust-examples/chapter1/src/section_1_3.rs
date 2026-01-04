//! Section 1.3: Formulating Abstractions with Higher-Order Procedures
//!
//! Procedures that manipulate procedures are called higher-order procedures.
//! This section demonstrates Rust's closure system (Fn, FnMut, FnOnce).

/// Generic sum using higher-order function.
/// Demonstrates Rust's generic bounds with Fn traits.
///
/// Note: Recursive higher-order functions in Rust are complex due to
/// ownership. The idiomatic approach is to use iteration (see `sum_iter`).
///
/// # Examples
/// ```
/// use sicp_chapter1::section_1_3::sum_iter;
/// let sum_cubes = sum_iter(|x| x * x * x, 1, |x| x + 1, 10);
/// assert_eq!(sum_cubes, 3025);
/// ```
pub fn sum<Term, Next>(term: &Term, a: i64, next: &Next, b: i64) -> i64
where
    Term: Fn(i64) -> i64,
    Next: Fn(i64) -> i64,
{
    if a > b {
        0
    } else {
        term(a) + sum(term, next(a), next, b)
    }
}

/// Iterative sum using Rust idioms.
/// More efficient as it avoids recursion overhead.
pub fn sum_iter<Term, Next>(term: Term, mut a: i64, next: Next, b: i64) -> i64
where
    Term: Fn(i64) -> i64,
    Next: Fn(i64) -> i64,
{
    let mut acc = 0;
    while a <= b {
        acc += term(a);
        a = next(a);
    }
    acc
}

/// Sum of integers from a to b.
pub fn sum_integers(a: i64, b: i64) -> i64 {
    sum_iter(|x| x, a, |x| x + 1, b)
}

/// Sum of cubes from a to b.
pub fn sum_cubes(a: i64, b: i64) -> i64 {
    sum_iter(|x| x * x * x, a, |x| x + 1, b)
}

/// Pi approximation using sum of series.
/// pi/8 = 1/(1*3) + 1/(5*7) + 1/(9*11) + ...
pub fn pi_sum(a: i64, b: i64) -> f64 {
    fn sum_f64<Term, Next>(term: &Term, a: i64, next: &Next, b: i64) -> f64
    where
        Term: Fn(i64) -> f64,
        Next: Fn(i64) -> i64,
    {
        if a > b {
            0.0
        } else {
            term(a) + sum_f64(term, next(a), next, b)
        }
    }
    sum_f64(&|x| 1.0 / ((x * (x + 2)) as f64), a, &|x| x + 4, b)
}

/// Generic product using higher-order function.
pub fn product<Term, Next>(term: &Term, a: i64, next: &Next, b: i64) -> i64
where
    Term: Fn(i64) -> i64,
    Next: Fn(i64) -> i64,
{
    if a > b {
        1
    } else {
        term(a) * product(term, next(a), next, b)
    }
}

/// Generic accumulate - generalizes sum and product.
/// Uses iterative approach to avoid ownership issues with recursive closures.
pub fn accumulate<T, Combiner, Term, Next>(
    combiner: Combiner,
    null_value: T,
    term: Term,
    mut a: i64,
    next: Next,
    b: i64,
) -> T
where
    T: Copy,
    Combiner: Fn(T, T) -> T,
    Term: Fn(i64) -> T,
    Next: Fn(i64) -> i64,
{
    let mut result = null_value;
    while a <= b {
        result = combiner(term(a), result);
        a = next(a);
    }
    result
}

/// Fixed-point search.
/// Finds x where f(x) = x within tolerance.
pub fn fixed_point<F>(f: F, first_guess: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    const TOLERANCE: f64 = 0.00001;

    fn close_enough(v1: f64, v2: f64) -> bool {
        (v1 - v2).abs() < TOLERANCE
    }

    let mut guess = first_guess;
    loop {
        let next = f(guess);
        if close_enough(guess, next) {
            return next;
        }
        guess = next;
    }
}

/// Average damping - technique to help convergence.
pub fn average_damp<F>(f: F) -> impl Fn(f64) -> f64
where
    F: Fn(f64) -> f64,
{
    move |x| (x + f(x)) / 2.0
}

/// Square root using fixed-point with average damping.
pub fn sqrt_fixed_point(x: f64) -> f64 {
    fixed_point(average_damp(move |y| x / y), 1.0)
}

/// Newton's method as fixed-point of transformed function.
pub fn newtons_method<G>(g: G, guess: f64) -> f64
where
    G: Fn(f64) -> f64 + Copy,
{
    const DX: f64 = 0.00001;

    // Newton transform: x - g(x)/g'(x)
    // where g'(x) is approximated by (g(x+dx) - g(x))/dx
    let newton_transform = move |x: f64| {
        let deriv = (g(x + DX) - g(x)) / DX;
        x - g(x) / deriv
    };

    fixed_point(newton_transform, guess)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        assert_eq!(sum_integers(1, 10), 55);
        assert_eq!(sum_cubes(1, 10), 3025);
    }

    #[test]
    fn test_product() {
        // Factorial as product
        let factorial = |n| product(&|x| x, 1, &|x| x + 1, n);
        assert_eq!(factorial(5), 120);
    }

    #[test]
    fn test_accumulate() {
        // Sum using accumulate
        let sum_acc = accumulate(|a, b| a + b, 0, |x| x, 1, |x| x + 1, 10);
        assert_eq!(sum_acc, 55);

        // Product using accumulate
        let prod_acc = accumulate(|a, b| a * b, 1, |x| x, 1, |x| x + 1, 5);
        assert_eq!(prod_acc, 120);
    }

    #[test]
    fn test_sqrt_fixed_point() {
        let result = sqrt_fixed_point(2.0);
        assert!((result * result - 2.0).abs() < 0.0001);
    }

    #[test]
    fn test_newtons_method() {
        // Find sqrt(2) by finding zero of x^2 - 2
        let result = newtons_method(|x| x * x - 2.0, 1.0);
        assert!((result * result - 2.0).abs() < 0.0001);
    }
}
