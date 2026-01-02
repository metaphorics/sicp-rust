//! SICP Section 3.5: Streams
//!
//! This section demonstrates lazy evaluation and infinite sequences using Rust's Iterator trait.
//! In Scheme, streams use `delay` and `force` for lazy evaluation. In Rust, the Iterator trait
//! provides lazy evaluation by default - elements are computed on-demand when `next()` is called.
//!
//! Key mappings:
//! - `delay/force` → Iterator trait (lazy by default)
//! - `cons-stream` → Iterator constructors
//! - `stream-car` → `next()` method
//! - `stream-cdr` → The iterator itself
//! - `stream-map` → `.map()` combinator
//! - `stream-filter` → `.filter()` combinator

use std::rc::Rc;

// =============================================================================
// Section 3.5.1: Streams Are Delayed Lists
// =============================================================================

/// Demonstrates the efficiency difference between eager and lazy evaluation.
///
/// In Scheme, eager lists construct all elements immediately, while streams
/// construct elements on-demand. Rust iterators are lazy by default.
pub mod delayed_lists {
    /// Find the second prime in a range (eager version - for comparison)
    pub fn second_prime_eager(low: u64, high: u64) -> Option<u64> {
        let primes: Vec<u64> = (low..=high).filter(|&n| is_prime(n)).collect(); // Eagerly evaluates ALL primes

        primes.get(1).copied()
    }

    /// Find the second prime in a range (lazy version - iterator)
    pub fn second_prime_lazy(low: u64, high: u64) -> Option<u64> {
        (low..=high).filter(|&n| is_prime(n)).nth(1) // Only computes until we find the 2nd prime
    }

    /// Simple primality test
    fn is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }

        let limit = (n as f64).sqrt() as u64;
        (3..=limit).step_by(2).all(|i| n % i != 0)
    }

    /// Demonstrates iterator combinators (map, filter, take)
    pub fn sum_of_squares_of_odd_numbers(n: usize) -> u64 {
        (0..)
            .filter(|x| x % 2 == 1) // Filter odd numbers
            .map(|x| x * x) // Square them
            .take(n) // Take first n
            .sum() // Sum them up
    }
}

// =============================================================================
// Section 3.5.2: Infinite Streams
// =============================================================================

/// Infinite stream of integers starting from n
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

/// Infinite stream of Fibonacci numbers
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

/// Sieve of Eratosthenes - infinite stream of prime numbers
///
/// This demonstrates a more complex infinite stream that filters
/// recursively. In Scheme, this uses nested stream filtering.
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

        // Create new iterator that filters out multiples of this prime
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

/// Alternative prime sieve using a more efficient algorithm
/// that checks divisibility by previously found primes
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
            .all(|&p| n % p != 0)
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
// Implicit Stream Definitions
// =============================================================================

/// Demonstrates implicit/self-referential stream definitions.
/// In Scheme: (define ones (cons-stream 1 ones))
///
/// In Rust, we use `std::iter::repeat` or implement custom iterators.
pub fn ones() -> impl Iterator<Item = i64> {
    std::iter::repeat(1)
}

/// Stream that doubles each previous element: 1, 2, 4, 8, 16, ...
/// In Scheme: (define double (cons-stream 1 (scale-stream double 2)))
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

/// Helper function to add two iterators element-wise
pub fn add_streams<I1, I2>(s1: I1, s2: I2) -> impl Iterator<Item = i64>
where
    I1: Iterator<Item = i64>,
    I2: Iterator<Item = i64>,
{
    s1.zip(s2).map(|(a, b)| a + b)
}

/// Helper function to multiply two iterators element-wise
pub fn mul_streams<I1, I2>(s1: I1, s2: I2) -> impl Iterator<Item = i64>
where
    I1: Iterator<Item = i64>,
    I2: Iterator<Item = i64>,
{
    s1.zip(s2).map(|(a, b)| a * b)
}

/// Helper function to scale a stream by a constant
pub fn scale_stream<I>(stream: I, factor: i64) -> impl Iterator<Item = i64>
where
    I: Iterator<Item = i64>,
{
    stream.map(move |x| x * factor)
}

// =============================================================================
// Section 3.5.3: Exploiting the Stream Paradigm
// =============================================================================

/// Iterative improvement using streams - square root approximation
///
/// In Scheme, this demonstrates how streams can replace state variables.
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

/// Stream of partial sums
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

/// Pi approximation using alternating series
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

/// Euler transform for sequence acceleration
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

/// Stream limit - finds when consecutive elements are within tolerance
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
// Infinite Streams of Pairs
// =============================================================================

/// Iterator that generates pairs (i, j) where i <= j
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

            // After generating enough pairs from first row, switch
            if self.t_index > self.s_index + 10 {
                self.in_first_row = false;
                self.s_index = 1;
                self.t_index = 1;
            }

            Some(pair)
        } else {
            // This is a simplified version - a complete implementation
            // would properly interleave the diagonal traversal
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
// Section 3.5.4: Streams and Delayed Evaluation
// =============================================================================

/// Signal integrator - accumulates sum over time
///
/// This demonstrates using iterators for signal processing.
/// In Scheme, this uses explicit delay for feedback loops.
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

/// Solve differential equation dy/dt = f(y) with initial condition y(0) = y0
///
/// This uses Rc<RefCell<>> to create the feedback loop that Scheme achieves
/// with delay/force.
pub fn solve<F>(f: F, y0: f64, dt: f64) -> impl Iterator<Item = f64>
where
    F: Fn(f64) -> f64 + 'static,
{
    // We'll use a simpler iterative approach that's more idiomatic in Rust
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
// Section 3.5.5: Modularity of Functional Programs
// =============================================================================

/// Monte Carlo estimation using streams
pub mod monte_carlo {

    /// Linear congruential random number generator
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

        /// Generate random number in [0, 1)
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

    /// Stream of experiment results (pass/fail)
    pub fn cesaro_stream(seed: u64) -> impl Iterator<Item = bool> {
        let mut rng = RandomNumbers::new(seed);

        std::iter::from_fn(move || {
            let x = rng.next().unwrap();
            let y = rng.next().unwrap();
            Some(gcd(x, y) == 1)
        })
    }

    /// GCD using Euclid's algorithm
    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 { a } else { gcd(b, a % b) }
    }

    /// Estimate pi using Monte Carlo method
    pub fn estimate_pi(trials: usize, seed: u64) -> f64 {
        let successes = cesaro_stream(seed).take(trials).filter(|&x| x).count();

        let probability = successes as f64 / trials as f64;
        (6.0 / probability).sqrt()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_vs_eager_evaluation() {
        // Both should find the same result
        let eager = delayed_lists::second_prime_eager(10_000, 11_000);
        let lazy = delayed_lists::second_prime_lazy(10_000, 11_000);
        assert_eq!(eager, lazy);
        assert_eq!(lazy, Some(10_009));
    }

    #[test]
    fn test_sum_of_squares_odd() {
        // Sum of squares of first 5 odd numbers: 1² + 3² + 5² + 7² + 9² = 165
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

        // Test finding 50th prime (as in SICP)
        let prime_50 = PrimesOptimized::new().nth(49);
        assert_eq!(prime_50, Some(229)); // Actually 229, not 233 (SICP has nth starting from 0)
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

        // First few approximations converge to sqrt(2) ≈ 1.414213562
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
        // Generate alternating series for pi/4
        let pi_stream = PiSummands::new()
            .enumerate()
            .map(|(i, term)| if i % 2 == 0 { term } else { -term });

        let partial_sums = PartialSums::new(pi_stream);
        let pi_approx: Vec<f64> = partial_sums.map(|x| x * 4.0).take(8).collect();

        // Verify convergence toward pi
        let last = pi_approx.last().unwrap();
        assert!((last - std::f64::consts::PI).abs() < 0.5);
    }

    #[test]
    fn test_euler_transform() {
        // Create pi series and apply Euler transformation
        let pi_summands = PiSummands::new()
            .enumerate()
            .map(|(i, term)| if i % 2 == 0 { term } else { -term });

        let partial_sums = PartialSums::new(pi_summands);
        let pi_values: Vec<f64> = partial_sums.map(|x| x * 4.0).take(10).collect();

        // Apply Euler transform
        let transformed: Vec<f64> = EulerTransform::new(pi_values.into_iter()).take(5).collect();

        // Transformed series should converge faster
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

        // Verify first pair is (0, 0)
        assert_eq!(pairs[0], (0, 0));

        // Verify all pairs satisfy i <= j
        for (i, j) in &pairs {
            assert!(i <= j, "Pair ({}, {}) violates i <= j", i, j);
        }
    }

    #[test]
    fn test_integrator() {
        // Integrate constant stream (should give linear growth)
        let ones_iter = std::iter::repeat(1.0);
        let integral: Vec<f64> = Integrator::new(ones_iter, 0.0, 0.1).take(11).collect();

        // Should be: 0.0, 0.1, 0.2, 0.3, ..., 1.0
        for (i, &value) in integral.iter().enumerate() {
            let expected = i as f64 * 0.1;
            assert!(
                (value - expected).abs() < 0.0001,
                "Expected {}, got {}",
                expected,
                value
            );
        }
    }

    #[test]
    fn test_solve_differential_equation() {
        // Solve dy/dt = y with y(0) = 1
        // Analytical solution: y = e^t
        let solution: Vec<f64> = solve(|y| y, 1.0, 0.001).take(1001).collect();

        // Check y(1) ≈ e ≈ 2.71828
        let y_at_1 = solution[1000];
        assert!((y_at_1 - std::f64::consts::E).abs() < 0.01);
    }

    #[test]
    fn test_monte_carlo_pi() {
        let estimate = monte_carlo::estimate_pi(10000, 42);

        // Should be close to pi (within 10%)
        assert!(
            (estimate - std::f64::consts::PI).abs() < 0.5,
            "Pi estimate {} too far from actual pi",
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
        // Test that we can work with multiple infinite streams
        let integers = IntegersFrom::new(1);
        let fibs = Fibonacci::new();

        // Interleave them by taking one from each
        let interleaved: Vec<i64> = integers
            .take(5)
            .zip(fibs.take(5))
            .flat_map(|(a, b)| vec![a, b as i64])
            .collect();

        // Should be: 1, 0, 2, 1, 3, 1, 4, 2, 5, 3
        assert_eq!(interleaved, vec![1, 0, 2, 1, 3, 1, 4, 2, 5, 3]);
    }
}
