//! 1.3절: 고차 프로시저를 이용한 추상화의 정식화
//! (Section 1.3: Formulating Abstractions with Higher-Order Procedures)
//!
//! 프로시저를 조작하는 프로시저를 고차 프로시저(higher-order procedures)라고 한다
//! (Procedures that manipulate procedures are called higher-order procedures).
//! 이 절에서는 Rust의 클로저 시스템(Fn, FnMut, FnOnce)을 보여준다
//! (This section demonstrates Rust's closure system (Fn, FnMut, FnOnce)).

/// 고차 함수를 사용한 제네릭 합계(sum)
/// (Generic sum using higher-order functions).
/// Rust의 Fn 트레이트를 사용한 제네릭 바운드를 보여준다
/// (Demonstrates generic bounds using Rust's Fn trait).
///
/// 참고: Rust에서 재귀적 고차 함수는 소유권 문제로 인해 복잡하다
/// (Note: In Rust, recursive higher-order functions are complex due to ownership).
/// 관용적인 접근 방식은 반복(iteration)을 사용하는 것이다 (`sum_iter` 참조)
/// (The idiomatic approach uses iteration; see `sum_iter`).
///
/// # 예시 (Examples)
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

/// Rust 관용구를 사용한 반복적 합계 (Iterative sum using Rust idioms).
/// 재귀 오버헤드를 피하므로 더 효율적입니다.
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

/// a부터 b까지 정수의 합 (Sum of integers from a to b).
pub fn sum_integers(a: i64, b: i64) -> i64 {
    sum_iter(|x| x, a, |x| x + 1, b)
}

/// a부터 b까지 세제곱의 합 (Sum of cubes from a to b).
pub fn sum_cubes(a: i64, b: i64) -> i64 {
    sum_iter(|x| x * x * x, a, |x| x + 1, b)
}

/// 급수의 합을 이용한 파이(Pi) 근사 (Pi approximation using series sums).
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

/// 고차 함수를 사용한 제네릭 곱(product) (Generic product using higher-order functions).
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

/// 제네릭 누산(accumulate) - 합계와 곱을 일반화
/// (Generic accumulate - generalizes sum and product).
/// 재귀적 클로저의 소유권 문제를 피하기 위해 반복적 접근 방식을 사용합니다.
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

/// 고정점(Fixed-point) 탐색 (Fixed-point search).
/// 허용 오차 내에서 f(x) = x인 x를 찾는다
/// (Finds x such that f(x) = x within a tolerance).
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

/// 평균 감쇠(Average damping) - 수렴을 돕는 기법
/// (Average damping - a technique to help convergence).
pub fn average_damp<F>(f: F) -> impl Fn(f64) -> f64
where
    F: Fn(f64) -> f64,
{
    move |x| (x + f(x)) / 2.0
}

/// 평균 감쇠를 사용한 고정점 방식의 제곱근 구하기.
pub fn sqrt_fixed_point(x: f64) -> f64 {
    fixed_point(average_damp(move |y| x / y), 1.0)
}

/// 변환된 함수의 고정점으로서의 뉴턴의 방법.
pub fn newtons_method<G>(g: G, guess: f64) -> f64
where
    G: Fn(f64) -> f64 + Copy,
{
    const DX: f64 = 0.00001;

    // 뉴턴 변환: x - g(x)/g'(x) (Newton transform: x - g(x)/g'(x))
    // g'(x)는 (g(x+dx) - g(x))/dx로 근사한다
    // (where g'(x) is approximated by (g(x+dx) - g(x))/dx)
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
        // 곱으로 표현한 팩토리얼 (Factorial as product)
        let factorial = |n| product(&|x| x, 1, &|x| x + 1, n);
        assert_eq!(factorial(5), 120);
    }

    #[test]
    fn test_accumulate() {
        // 누산으로 합 계산 (Sum using accumulate)
        let sum_acc = accumulate(|a, b| a + b, 0, |x| x, 1, |x| x + 1, 10);
        assert_eq!(sum_acc, 55);

        // 누산으로 곱 계산 (Product using accumulate)
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
        // x^2 - 2의 영점을 찾아 sqrt(2)를 구한다
        // (Find sqrt(2) by finding zero of x^2 - 2)
        let result = newtons_method(|x| x * x - 2.0, 1.0);
        assert!((result * result - 2.0).abs() < 0.0001);
    }
}
