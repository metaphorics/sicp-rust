//! SICP 3.5절: 스트림 (Streams)
//!
//! 이 절은 러스트의 Iterator 트레이트를 사용해 지연 평가와 무한 수열을 보여준다
//! (This section demonstrates lazy evaluation and infinite sequences using Rust's Iterator trait).
//! Scheme의 스트림은 지연 평가를 위해 `delay`와 `force`를 사용한다.
//! 러스트에서는 Iterator 트레이트가 기본적으로 지연 평가를 제공하며,
//! `next()`가 호출될 때 요소를 필요 시점에 계산한다
//! (In Scheme, streams use `delay` and `force` for lazy evaluation. In Rust, the Iterator trait
//! provides lazy evaluation by default - elements are computed on-demand when `next()` is called).
//!
//! 핵심 매핑 (Key mappings):
//! - `delay/force` → Iterator 트레이트 (기본 지연) (Iterator trait (lazy by default))
//! - `cons-stream` → Iterator 생성자 (Iterator constructors)
//! - `stream-car` → `next()` 메서드 (method)
//! - `stream-cdr` → Iterator 자체 (The iterator itself)
//! - `stream-map` → `.map()` 컴비네이터 (combinator)
//! - `stream-filter` → `.filter()` 컴비네이터 (combinator)

use std::rc::Rc;

// =============================================================================
// 3.5.1절: 스트림은 지연된 리스트 (Streams Are Delayed Lists)
// =============================================================================

/// 즉시 평가와 지연 평가의 효율 차이를 보여준다
/// (Demonstrates the efficiency difference between eager and lazy evaluation).
///
/// Scheme에서는 즉시 리스트가 모든 요소를 즉시 구성하고,
/// 스트림은 필요할 때 요소를 만든다. 러스트 이터레이터는 기본적으로 지연 평가다
/// (In Scheme, eager lists construct all elements immediately, while streams
/// construct elements on-demand. Rust iterators are lazy by default).
pub mod delayed_lists {
    /// 구간에서 두 번째 소수를 찾는다 (즉시 버전 - 비교용)
    /// (Finds the second prime in a range (eager version - for comparison)).
    pub fn second_prime_eager(low: u64, high: u64) -> Option<u64> {
        let primes: Vec<u64> = (low..=high).filter(|&n| is_prime(n)).collect(); // 모든 소수를 즉시 계산 (Eagerly evaluates ALL primes)

        primes.get(1).copied()
    }

    /// 구간에서 두 번째 소수를 찾는다 (지연 버전 - 이터레이터)
    /// (Finds the second prime in a range (lazy version - iterator)).
    pub fn second_prime_lazy(low: u64, high: u64) -> Option<u64> {
        (low..=high).filter(|&n| is_prime(n)).nth(1) // 두 번째 소수를 찾을 때까지만 계산 (Only computes until we find the 2nd prime)
    }

    /// 간단한 소수 판정 (Simple primality test).
    fn is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n.is_multiple_of(2) {
            return false;
        }

        let limit = (n as f64).sqrt() as u64;
        (3..=limit).step_by(2).all(|i| !n.is_multiple_of(i))
    }

    /// 이터레이터 컴비네이터 시연 (map, filter, take)
    /// (Demonstrates iterator combinators (map, filter, take)).
    pub fn sum_of_squares_of_odd_numbers(n: usize) -> u64 {
        (0..)
            .filter(|x| x % 2 == 1) // 홀수 필터 (Filter odd numbers)
            .map(|x| x * x) // 제곱 (Square them)
            .take(n) // 처음 n개 (Take first n)
            .sum() // 합산 (Sum them up)
    }
}

// =============================================================================
// 3.5.2절: 무한 스트림 (Infinite Streams)
// =============================================================================

/// n에서 시작하는 정수 무한 스트림
/// (Infinite stream of integers starting from n).
pub struct IntegersFrom {
    current: i64,
}

impl IntegersFrom {
    pub fn new(start: i64) -> Self {
        IntegersFrom { current: start }
    }
}

impl Iterator for IntegersFrom {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.current;
        self.current += 1;
        Some(value)
    }
}

/// 피보나치 수열 무한 스트림 (Infinite stream of Fibonacci numbers).
pub struct Fibonacci {
    current: u64,
    next: u64,
}

impl Fibonacci {
    pub fn new() -> Self {
        Fibonacci {
            current: 0,
            next: 1,
        }
    }
}

impl Iterator for Fibonacci {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.current;
        let new_next = self.current + self.next;
        self.current = self.next;
        self.next = new_next;
        Some(result)
    }
}

impl Default for Fibonacci {
    fn default() -> Self {
        Self::new()
    }
}

/// 에라토스테네스의 체 - 소수 무한 스트림
/// (Sieve of Eratosthenes - infinite stream of prime numbers).
///
/// 재귀적으로 필터링하는 더 복잡한 무한 스트림을 보여준다
/// (This demonstrates a more complex infinite stream that filters recursively).
/// Scheme에서는 중첩 스트림 필터링을 사용한다
/// (In Scheme, this uses nested stream filtering).
pub struct Sieve {
    candidates: Box<dyn Iterator<Item = u64>>,
}

impl Sieve {
    pub fn new() -> Self {
        Sieve {
            candidates: Box::new(2..),
        }
    }
}

impl Iterator for Sieve {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let prime = self.candidates.next()?;

        // 이 소수의 배수를 걸러내는 새 이터레이터 생성
        // (Create new iterator that filters out multiples of this prime)
        let old_candidates = std::mem::replace(&mut self.candidates, Box::new(std::iter::empty()));

        self.candidates = Box::new(old_candidates.filter(move |&n| n % prime != 0));

        Some(prime)
    }
}

impl Default for Sieve {
    fn default() -> Self {
        Self::new()
    }
}

/// 더 효율적인 알고리즘을 사용하는 대안 소수 체
/// (Alternative prime sieve using a more efficient algorithm).
/// 이미 찾은 소수로 나눠떨어짐을 확인한다
/// (Checks divisibility by previously found primes).
pub struct PrimesOptimized {
    primes_so_far: Vec<u64>,
    candidate: u64,
}

impl PrimesOptimized {
    pub fn new() -> Self {
        PrimesOptimized {
            primes_so_far: vec![],
            candidate: 2,
        }
    }

    fn is_prime(&self, n: u64) -> bool {
        let limit = (n as f64).sqrt() as u64;
        self.primes_so_far
            .iter()
            .take_while(|&&p| p <= limit)
            .all(|&p| !n.is_multiple_of(p))
    }
}

impl Iterator for PrimesOptimized {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.is_prime(self.candidate) {
                let prime = self.candidate;
                self.primes_so_far.push(prime);
                self.candidate += if self.candidate == 2 { 1 } else { 2 };
                return Some(prime);
            }
            self.candidate += if self.candidate == 2 { 1 } else { 2 };
        }
    }
}

impl Default for PrimesOptimized {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 암시적 스트림 정의 (Implicit Stream Definitions)
// =============================================================================

/// 암시적/자기참조 스트림 정의를 시연한다
/// (Demonstrates implicit/self-referential stream definitions).
/// 스킴 (Scheme): (define ones (cons-stream 1 ones))
///
/// Rust에서는 `std::iter::repeat`를 사용하거나 커스텀 이터레이터를 구현한다
/// (In Rust, we use `std::iter::repeat` or implement custom iterators).
pub fn ones() -> impl Iterator<Item = i64> {
    std::iter::repeat(1)
}

/// 이전 요소를 두 배로 하는 스트림: 1, 2, 4, 8, 16, ...
/// (Stream that doubles each previous element: 1, 2, 4, 8, 16, ...)
/// 스킴 (Scheme): (define double (cons-stream 1 (scale-stream double 2)))
pub struct DoublingStream {
    current: u64,
}

impl DoublingStream {
    pub fn new() -> Self {
        DoublingStream { current: 1 }
    }
}

impl Iterator for DoublingStream {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.current;
        self.current *= 2;
        Some(value)
    }
}

impl Default for DoublingStream {
    fn default() -> Self {
        Self::new()
    }
}

/// 두 이터레이터를 원소별로 더하는 헬퍼 함수
/// (Helper function to add two iterators element-wise).
pub fn add_streams<I1, I2>(s1: I1, s2: I2) -> impl Iterator<Item = i64>
where
    I1: Iterator<Item = i64>,
    I2: Iterator<Item = i64>,
{
    s1.zip(s2).map(|(a, b)| a + b)
}

/// 두 이터레이터를 원소별로 곱하는 헬퍼 함수
/// (Helper function to multiply two iterators element-wise).
pub fn mul_streams<I1, I2>(s1: I1, s2: I2) -> impl Iterator<Item = i64>
where
    I1: Iterator<Item = i64>,
    I2: Iterator<Item = i64>,
{
    s1.zip(s2).map(|(a, b)| a * b)
}

/// 스트림을 상수로 스케일하는 헬퍼 함수
/// (Helper function to scale a stream by a constant).
pub fn scale_stream<I>(stream: I, factor: i64) -> impl Iterator<Item = i64>
where
    I: Iterator<Item = i64>,
{
    stream.map(move |x| x * factor)
}

// =============================================================================
// 3.5.3절: 스트림 패러다임 활용 (Exploiting the Stream Paradigm)
// =============================================================================

/// 스트림을 이용한 반복적 개선 - 제곱근 근사
/// (Iterative improvement using streams - square root approximation).
///
/// Scheme에서는 스트림이 상태 변수를 대체할 수 있음을 보여준다
/// (In Scheme, this demonstrates how streams can replace state variables).
pub struct SqrtStream {
    x: f64,
    guess: f64,
}

impl SqrtStream {
    pub fn new(x: f64) -> Self {
        SqrtStream { x, guess: 1.0 }
    }

    fn improve_guess(guess: f64, x: f64) -> f64 {
        (guess + x / guess) / 2.0
    }
}

impl Iterator for SqrtStream {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.guess;
        self.guess = Self::improve_guess(self.guess, self.x);
        Some(current)
    }
}

/// 부분 합 스트림 (Stream of partial sums).
pub struct PartialSums<I>
where
    I: Iterator<Item = f64>,
{
    stream: I,
    sum: f64,
}

impl<I> PartialSums<I>
where
    I: Iterator<Item = f64>,
{
    pub fn new(stream: I) -> Self {
        PartialSums { stream, sum: 0.0 }
    }
}

impl<I> Iterator for PartialSums<I>
where
    I: Iterator<Item = f64>,
{
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.stream.next()?;
        self.sum += value;
        Some(self.sum)
    }
}

/// 교대 급수로 파이 근사 (Pi approximation using alternating series).
pub struct PiSummands {
    n: f64,
}

impl PiSummands {
    pub fn new() -> Self {
        PiSummands { n: 1.0 }
    }
}

impl Iterator for PiSummands {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let term = 1.0 / self.n;
        self.n += 2.0;
        Some(term)
    }
}

impl Default for PiSummands {
    fn default() -> Self {
        Self::new()
    }
}

/// 수열 가속을 위한 오일러 변환 (Euler transform for sequence acceleration).
pub struct EulerTransform<I>
where
    I: Iterator<Item = f64>,
{
    stream: I,
    s0: Option<f64>,
    s1: Option<f64>,
}

impl<I> EulerTransform<I>
where
    I: Iterator<Item = f64>,
{
    pub fn new(mut stream: I) -> Self {
        let s0 = stream.next();
        let s1 = stream.next();
        EulerTransform { stream, s0, s1 }
    }
}

impl<I> Iterator for EulerTransform<I>
where
    I: Iterator<Item = f64>,
{
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let s0 = self.s0?;
        let s1 = self.s1?;
        let s2 = self.stream.next()?;

        let result = s2 - (s2 - s1).powi(2) / (s0 - 2.0 * s1 + s2);

        self.s0 = Some(s1);
        self.s1 = Some(s2);

        Some(result)
    }
}

/// 스트림 제한 - 연속 요소가 허용 오차 안에 들어올 때를 찾는다
/// (Stream limit - finds when consecutive elements are within tolerance).
pub fn stream_limit<I>(mut stream: I, tolerance: f64) -> Option<f64>
where
    I: Iterator<Item = f64>,
{
    let mut prev = stream.next()?;

    for current in stream {
        if (current - prev).abs() < tolerance {
            return Some(current);
        }
        prev = current;
    }

    None
}

// =============================================================================
// 쌍의 무한 스트림 (Infinite Streams of Pairs)
// =============================================================================

/// i <= j를 만족하는 (i, j) 쌍을 생성하는 이터레이터
/// (Iterator that generates pairs (i, j) where i <= j).
pub struct Pairs {
    s_index: usize,
    t_index: usize,
    in_first_row: bool,
}

impl Pairs {
    pub fn new() -> Self {
        Pairs {
            s_index: 0,
            t_index: 0,
            in_first_row: true,
        }
    }
}

impl Iterator for Pairs {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.in_first_row {
            let pair = (self.s_index, self.t_index);
            self.t_index += 1;

            // 첫 번째 행에서 충분히 생성했으면 전환
            // (After generating enough pairs from first row, switch)
            if self.t_index > self.s_index + 10 {
                self.in_first_row = false;
                self.s_index = 1;
                self.t_index = 1;
            }

            Some(pair)
        } else {
            // 단순화된 버전이다 - 완전한 구현은 대각선 순회를 올바르게 교차해야 한다
            // (This is a simplified version - a complete implementation
            // would properly interleave the diagonal traversal)
            let pair = (self.s_index, self.t_index);

            if self.t_index < self.s_index + 5 {
                self.t_index += 1;
            } else {
                self.s_index += 1;
                self.t_index = self.s_index;
            }

            Some(pair)
        }
    }
}

impl Default for Pairs {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 3.5.4절: 스트림과 지연 평가 (Streams and Delayed Evaluation)
// =============================================================================

/// 신호 적분기 - 시간에 따라 합을 누적 (Signal integrator - accumulates sum over time).
///
/// 이터레이터로 신호 처리를 하는 방법을 보여준다
/// (This demonstrates using iterators for signal processing).
/// Scheme에서는 피드백 루프를 위해 명시적 지연을 사용한다
/// (In Scheme, this uses explicit delay for feedback loops).
pub struct Integrator<I>
where
    I: Iterator<Item = f64>,
{
    integrand: I,
    dt: f64,
    accumulator: f64,
}

impl<I> Integrator<I>
where
    I: Iterator<Item = f64>,
{
    pub fn new(integrand: I, initial_value: f64, dt: f64) -> Self {
        Integrator {
            integrand,
            dt,
            accumulator: initial_value,
        }
    }
}

impl<I> Iterator for Integrator<I>
where
    I: Iterator<Item = f64>,
{
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.accumulator;

        if let Some(value) = self.integrand.next() {
            self.accumulator += value * self.dt;
        }

        Some(current)
    }
}

/// 미분방정식 dy/dt = f(y)를 초기 조건 y(0) = y0로 푼다
/// (Solve differential equation dy/dt = f(y) with initial condition y(0) = y0).
///
/// Scheme의 delay/force가 만드는 피드백 루프를 Rc<RefCell<>>로 구성한다
/// (This uses Rc<RefCell<>> to create the feedback loop that Scheme achieves
/// with delay/force).
pub fn solve<F>(f: F, y0: f64, dt: f64) -> impl Iterator<Item = f64>
where
    F: Fn(f64) -> f64 + 'static,
{
    // 러스트에 더 관용적인 간단한 반복 접근을 사용한다
    // (We'll use a simpler iterative approach that's more idiomatic in Rust)
    SolveIterator {
        f: Rc::new(f),
        y: y0,
        dt,
    }
}

struct SolveIterator<F>
where
    F: Fn(f64) -> f64,
{
    f: Rc<F>,
    y: f64,
    dt: f64,
}

impl<F> Iterator for SolveIterator<F>
where
    F: Fn(f64) -> f64,
{
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.y;
        let dy = (self.f)(self.y);
        self.y += dy * self.dt;
        Some(current)
    }
}

// =============================================================================
// 3.5.5절: 함수형 프로그램의 모듈성 (Modularity of Functional Programs)
// =============================================================================

/// 스트림을 사용하는 몬테카를로 추정 (Monte Carlo estimation using streams).
pub mod monte_carlo {

    /// 선형 합동 난수 생성기 (Linear congruential random number generator).
    pub struct RandomNumbers {
        current: u64,
        a: u64,
        c: u64,
        m: u64,
    }

    impl RandomNumbers {
        pub fn new(seed: u64) -> Self {
            RandomNumbers {
                current: seed,
                a: 1664525,
                c: 1013904223,
                m: 2u64.pow(32),
            }
        }

        /// [0, 1) 구간의 난수를 생성한다 (Generate random number in [0, 1)).
        pub fn next_f64(&mut self) -> f64 {
            self.current = (self.a * self.current + self.c) % self.m;
            self.current as f64 / self.m as f64
        }
    }

    impl Iterator for RandomNumbers {
        type Item = u64;

        fn next(&mut self) -> Option<Self::Item> {
            self.current = (self.a * self.current + self.c) % self.m;
            Some(self.current)
        }
    }

    /// 실험 결과 스트림 (성공/실패)
    /// (Stream of experiment results (pass/fail)).
    pub fn cesaro_stream(seed: u64) -> impl Iterator<Item = bool> {
        let mut rng = RandomNumbers::new(seed);

        std::iter::from_fn(move || {
            let x = rng.next().unwrap();
            let y = rng.next().unwrap();
            Some(gcd(x, y) == 1)
        })
    }

    /// 유클리드 알고리즘으로 최대공약수 계산
    /// (GCD using Euclid's algorithm).
    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }

    /// 몬테카를로 방법으로 파이를 추정한다
    /// (Estimate pi using Monte Carlo method).
    pub fn estimate_pi(trials: usize, seed: u64) -> f64 {
        let successes = cesaro_stream(seed).take(trials).filter(|&x| x).count();

        let probability = successes as f64 / trials as f64;
        (6.0 / probability).sqrt()
    }
}

// =============================================================================
// 테스트 (Tests)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_vs_eager_evaluation() {
        // 두 방법 모두 같은 결과를 찾아야 한다 (Both should find the same result)
        let eager = delayed_lists::second_prime_eager(10_000, 11_000);
        let lazy = delayed_lists::second_prime_lazy(10_000, 11_000);
        assert_eq!(eager, lazy);
        assert_eq!(lazy, Some(10_009));
    }

    #[test]
    fn test_sum_of_squares_odd() {
        // 처음 5개 홀수의 제곱 합: 1² + 3² + 5² + 7² + 9² = 165
        // (Sum of squares of first 5 odd numbers: 1² + 3² + 5² + 7² + 9² = 165)
        let result = delayed_lists::sum_of_squares_of_odd_numbers(5);
        assert_eq!(result, 165);
    }

    #[test]
    fn test_integers_from() {
        let integers: Vec<i64> = IntegersFrom::new(1).take(10).collect();
        assert_eq!(integers, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_fibonacci() {
        let fibs: Vec<u64> = Fibonacci::new().take(10).collect();
        assert_eq!(fibs, vec![0, 1, 1, 2, 3, 5, 8, 13, 21, 34]);
    }

    #[test]
    fn test_primes_sieve() {
        let primes: Vec<u64> = Sieve::new().take(10).collect();
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_primes_optimized() {
        let primes: Vec<u64> = PrimesOptimized::new().take(10).collect();
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);

        // 50번째 소수 찾기 테스트 (SICP와 동일)
        // (Test finding 50th prime (as in SICP))
        let prime_50 = PrimesOptimized::new().nth(49);
        assert_eq!(prime_50, Some(229)); // 실제는 229, 233이 아님 (SICP는 0부터 n번째) (Actually 229, not 233 (SICP has nth starting from 0))
    }

    #[test]
    fn test_ones_stream() {
        let first_five: Vec<i64> = ones().take(5).collect();
        assert_eq!(first_five, vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_doubling_stream() {
        let powers_of_2: Vec<u64> = DoublingStream::new().take(8).collect();
        assert_eq!(powers_of_2, vec![1, 2, 4, 8, 16, 32, 64, 128]);
    }

    #[test]
    fn test_add_streams() {
        let s1 = vec![1, 2, 3, 4, 5];
        let s2 = vec![10, 20, 30, 40, 50];
        let result: Vec<i64> = add_streams(s1.into_iter(), s2.into_iter()).collect();
        assert_eq!(result, vec![11, 22, 33, 44, 55]);
    }

    #[test]
    fn test_sqrt_stream() {
        let mut sqrt2 = SqrtStream::new(2.0);

        // 초기 근사치가 sqrt(2) ≈ 1.414213562로 수렴
        // (First few approximations converge to sqrt(2) ≈ 1.414213562)
        let _approx1 = sqrt2.next().unwrap(); // 1.0
        let _approx2 = sqrt2.next().unwrap(); // 1.5
        let _approx3 = sqrt2.next().unwrap(); // 1.41666...
        let approx4 = sqrt2.next().unwrap(); // 1.41421...

        assert!((approx4 - 1.414213562).abs() < 0.001);
    }

    #[test]
    fn test_partial_sums() {
        let integers = vec![1, 2, 3, 4, 5];
        let sums: Vec<f64> = PartialSums::new(integers.into_iter().map(|x| x as f64)).collect();
        assert_eq!(sums, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
    }

    #[test]
    fn test_pi_approximation() {
        // pi/4를 위한 교대 급수 생성 (Generate alternating series for pi/4)
        let pi_stream = PiSummands::new()
            .enumerate()
            .map(|(i, term)| if i % 2 == 0 { term } else { -term });

        let partial_sums = PartialSums::new(pi_stream);
        let pi_approx: Vec<f64> = partial_sums.map(|x| x * 4.0).take(8).collect();

        // 파이에 대한 수렴 확인 (Verify convergence toward pi)
        let last = pi_approx.last().unwrap();
        assert!((last - std::f64::consts::PI).abs() < 0.5);
    }

    #[test]
    fn test_euler_transform() {
        // pi 급수를 만들고 오일러 변환 적용
        // (Create pi series and apply Euler transformation)
        let pi_summands = PiSummands::new()
            .enumerate()
            .map(|(i, term)| if i % 2 == 0 { term } else { -term });

        let partial_sums = PartialSums::new(pi_summands);
        let pi_values: Vec<f64> = partial_sums.map(|x| x * 4.0).take(10).collect();

        // 오일러 변환 적용 (Apply Euler transform)
        let transformed: Vec<f64> = EulerTransform::new(pi_values.into_iter()).take(5).collect();

        // 변환된 급수는 더 빨리 수렴해야 함 (Transformed series should converge faster)
        if let Some(&last) = transformed.last() {
            assert!((last - std::f64::consts::PI).abs() < 0.1);
        }
    }

    #[test]
    fn test_stream_limit() {
        let sqrt2_stream = SqrtStream::new(2.0);
        let result = stream_limit(sqrt2_stream, 0.0001);

        assert!(result.is_some());
        let value = result.unwrap();
        assert!((value - 2.0_f64.sqrt()).abs() < 0.001);
    }

    #[test]
    fn test_pairs() {
        let pairs: Vec<(usize, usize)> = Pairs::new().take(15).collect();

        // 첫 번째 쌍이 (0, 0)인지 확인 (Verify first pair is (0, 0))
        assert_eq!(pairs[0], (0, 0));

        // 모든 쌍이 i <= j를 만족하는지 확인 (Verify all pairs satisfy i <= j)
        for (i, j) in &pairs {
            assert!(
                i <= j,
                "쌍 ({}, {})이 i <= j를 위반 (Pair ({}, {}) violates i <= j)",
                i,
                j,
                i,
                j
            );
        }
    }

    #[test]
    fn test_integrator() {
        // 상수 스트림 적분 (선형 증가여야 함)
        // (Integrate constant stream (should give linear growth))
        let ones_iter = std::iter::repeat(1.0);
        let integral: Vec<f64> = Integrator::new(ones_iter, 0.0, 0.1).take(11).collect();

        // 기대값: 0.0, 0.1, 0.2, 0.3, ..., 1.0
        // (Should be: 0.0, 0.1, 0.2, 0.3, ..., 1.0)
        for (i, &value) in integral.iter().enumerate() {
            let expected = i as f64 * 0.1;
            assert!(
                (value - expected).abs() < 0.0001,
                "예상 {}, 실제 {} (Expected {}, got {})",
                expected,
                value,
                expected,
                value
            );
        }
    }

    #[test]
    fn test_solve_differential_equation() {
        // dy/dt = y, y(0) = 1을 풀이
        // (Solve dy/dt = y with y(0) = 1)
        // 해석해: y = e^t (Analytical solution: y = e^t)
        let solution: Vec<f64> = solve(|y| y, 1.0, 0.001).take(1001).collect();

        // y(1) ≈ e ≈ 2.71828 확인 (Check y(1) ≈ e ≈ 2.71828)
        let y_at_1 = solution[1000];
        assert!((y_at_1 - std::f64::consts::E).abs() < 0.01);
    }

    #[test]
    fn test_monte_carlo_pi() {
        let estimate = monte_carlo::estimate_pi(10000, 42);

        // 파이에 가까워야 함 (10% 이내) (Should be close to pi (within 10%))
        assert!(
            (estimate - std::f64::consts::PI).abs() < 0.5,
            "파이 추정치 {}가 실제 파이에서 너무 멀다 (Pi estimate {} too far from actual pi)",
            estimate,
            estimate
        );
    }

    #[test]
    fn test_scale_stream() {
        let numbers = vec![1, 2, 3, 4, 5];
        let scaled: Vec<i64> = scale_stream(numbers.into_iter(), 10).collect();
        assert_eq!(scaled, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_mul_streams() {
        let s1 = vec![1, 2, 3, 4, 5];
        let s2 = vec![10, 10, 10, 10, 10];
        let result: Vec<i64> = mul_streams(s1.into_iter(), s2.into_iter()).collect();
        assert_eq!(result, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_infinite_stream_interleaving() {
        // 여러 무한 스트림을 함께 다룰 수 있는지 테스트
        // (Test that we can work with multiple infinite streams)
        let integers = IntegersFrom::new(1);
        let fibs = Fibonacci::new();

        // 각 스트림에서 하나씩 가져와 교차 (Interleave them by taking one from each)
        let interleaved: Vec<i64> = integers
            .take(5)
            .zip(fibs.take(5))
            .flat_map(|(a, b)| vec![a, b as i64])
            .collect();

        // 기대값: 1, 0, 2, 1, 3, 1, 4, 2, 5, 3
        // (Should be: 1, 0, 2, 1, 3, 1, 4, 2, 5, 3)
        assert_eq!(interleaved, vec![1, 0, 2, 1, 3, 1, 4, 2, 5, 3]);
    }
}
