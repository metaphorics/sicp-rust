# SICP Chapter 3.5: Streams in Rust

## Overview

This guide explains how SICP's stream concepts map to Rust's Iterator trait. In Scheme, streams use `delay` and `force` for lazy evaluation. In Rust, the Iterator trait provides lazy evaluation by default.

## Concept Mappings

| Scheme Concept | Rust Equivalent | Example |
|----------------|-----------------|---------|
| `delay` | Iterator (lazy by default) | `(0..).filter(...)` |
| `force` | `.next()` or `.collect()` | `iter.next()` |
| `cons-stream` | Iterator constructors | `std::iter::once(x).chain(rest)` |
| `stream-car` | `.next().unwrap()` | `stream.next().unwrap()` |
| `stream-cdr` | The iterator itself | `stream.skip(1)` |
| `stream-map` | `.map()` | `stream.map(\|x\| x * 2)` |
| `stream-filter` | `.filter()` | `stream.filter(\|&x\| x > 0)` |
| `stream-ref` | `.nth()` | `stream.nth(5)` |
| `the-empty-stream` | `std::iter::empty()` | `std::iter::empty::<i32>()` |
| Infinite stream | Custom Iterator impl | `struct IntegersFrom { ... }` |

## Key Features

### 1. Lazy Evaluation

**Scheme:**
```scheme
(define (integers-starting-from n)
  (cons-stream n (integers-starting-from (+ n 1))))
```

**Rust:**
```rust
struct IntegersFrom {
    current: i64,
}

impl Iterator for IntegersFrom {
    type Item = i64;
    fn next(&mut self) -> Option<Self::Item> {
        let value = self.current;
        self.current += 1;
        Some(value)
    }
}
```

### 2. Infinite Sequences

**Fibonacci (Scheme):**
```scheme
(define (fibgen a b)
  (cons-stream a (fibgen b (+ a b))))
(define fibs (fibgen 0 1))
```

**Fibonacci (Rust):**
```rust
struct Fibonacci {
    current: u64,
    next: u64,
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
```

### 3. Stream Operations

**Adding Streams (Scheme):**
```scheme
(define (add-streams s1 s2)
  (stream-map + s1 s2))
```

**Adding Streams (Rust):**
```rust
fn add_streams<I1, I2>(s1: I1, s2: I2) -> impl Iterator<Item = i64>
where
    I1: Iterator<Item = i64>,
    I2: Iterator<Item = i64>,
{
    s1.zip(s2).map(|(a, b)| a + b)
}
```

### 4. Sieve of Eratosthenes

**Scheme:**
```scheme
(define (sieve stream)
  (cons-stream
   (stream-car stream)
   (sieve (stream-filter
           (lambda (x)
             (not (divisible? x (stream-car stream))))
           (stream-cdr stream)))))

(define primes (sieve (integers-starting-from 2)))
```

**Rust:**
```rust
struct Sieve {
    candidates: Box<dyn Iterator<Item = u64>>,
}

impl Iterator for Sieve {
    type Item = u64;
    fn next(&mut self) -> Option<Self::Item> {
        let prime = self.candidates.next()?;
        let old_candidates = std::mem::replace(
            &mut self.candidates,
            Box::new(std::iter::empty()),
        );
        self.candidates = Box::new(
            old_candidates.filter(move |&n| n % prime != 0)
        );
        Some(prime)
    }
}
```

### 5. Implicit Stream Definitions

**Ones (Scheme):**
```scheme
(define ones (cons-stream 1 ones))
```

**Ones (Rust):**
```rust
fn ones() -> impl Iterator<Item = i64> {
    std::iter::repeat(1)
}
```

**Integers (Scheme - implicit):**
```scheme
(define integers
  (cons-stream 1 (add-streams ones integers)))
```

**Integers (Rust - using scan):**
```rust
fn integers() -> impl Iterator<Item = i64> {
    std::iter::repeat(1)
        .scan(0, |state, x| {
            *state += x;
            Some(*state)
        })
}
```

### 6. Signal Processing

**Integration (Scheme):**
```scheme
(define (integral integrand initial-value dt)
  (define int
    (cons-stream
     initial-value
     (add-streams (scale-stream integrand dt) int)))
  int)
```

**Integration (Rust):**
```rust
struct Integrator<I>
where
    I: Iterator<Item = f64>,
{
    integrand: I,
    dt: f64,
    accumulator: f64,
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
```

## Running Examples

### Run all tests:
```bash
cargo test --lib section_3_5
```

### Run the demonstration:
```bash
cargo run --example streams_demo
```

### Expected test output:
```
running 21 tests
test section_3_5::tests::test_add_streams ... ok
test section_3_5::tests::test_doubling_stream ... ok
test section_3_5::tests::test_euler_transform ... ok
test section_3_5::tests::test_fibonacci ... ok
test section_3_5::tests::test_infinite_stream_interleaving ... ok
test section_3_5::tests::test_integers_from ... ok
test section_3_5::tests::test_integrator ... ok
test section_3_5::tests::test_lazy_vs_eager_evaluation ... ok
test section_3_5::tests::test_monte_carlo_pi ... ok
test section_3_5::tests::test_mul_streams ... ok
test section_3_5::tests::test_ones_stream ... ok
test section_3_5::tests::test_pairs ... ok
test section_3_5::tests::test_partial_sums ... ok
test section_3_5::tests::test_pi_approximation ... ok
test section_3_5::tests::test_primes_optimized ... ok
test section_3_5::tests::test_primes_sieve ... ok
test section_3_5::tests::test_scale_stream ... ok
test section_3_5::tests::test_solve_differential_equation ... ok
test section_3_5::tests::test_sqrt_stream ... ok
test section_3_5::tests::test_stream_limit ... ok
test section_3_5::tests::test_sum_of_squares_odd ... ok

test result: ok. 21 passed; 0 failed
```

## Key Insights

1. **Lazy by Default**: Rust iterators are lazy by default - no explicit delay/force needed
2. **Zero-Cost Abstraction**: Iterator combinators compile to efficient machine code
3. **Type Safety**: The type system ensures correct composition of streams
4. **Ownership**: Iterator ownership prevents issues with shared mutable state
5. **Composability**: Iterator combinators (`map`, `filter`, `take`, etc.) compose naturally
6. **Infinite Sequences**: Just as natural in Rust as in Scheme

## Advanced Patterns

### Sequence Acceleration (Euler Transform)

The Euler transform accelerates convergence of alternating series:

```rust
struct EulerTransform<I>
where
    I: Iterator<Item = f64>,
{
    stream: I,
    s0: Option<f64>,
    s1: Option<f64>,
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
```

### Differential Equation Solver

```rust
fn solve<F>(f: F, y0: f64, dt: f64) -> impl Iterator<Item = f64>
where
    F: Fn(f64) -> f64 + 'static,
{
    SolveIterator {
        f: Rc::new(f),
        y: y0,
        dt,
    }
}

// Solves: dy/dt = f(y), y(0) = y0
```

## Differences from Scheme

1. **Explicit Types**: Rust requires type annotations for custom iterators
2. **Ownership**: Must think about ownership when chaining iterators
3. **No Memoization**: Rust iterators don't automatically cache values (use `Peekable` or custom caching)
4. **Finite by Convention**: `Option<T>` signals end, not a special empty value
5. **No Implicit Recursion**: Must explicitly design recursive stream structures

## Benefits of Rust Approach

1. **Memory Safety**: No dangling references or use-after-free
2. **Thread Safety**: Send/Sync traits ensure safe concurrency
3. **Performance**: Zero-cost abstractions with inline optimizations
4. **Compile-Time Guarantees**: Type system catches many bugs early
5. **Clear Ownership**: Explicit ownership prevents subtle bugs

## Exercises

Try implementing these SICP exercises in Rust:

- Exercise 3.53: Describe the stream `s = 1 + (s + s)`
- Exercise 3.54: Define `mul-streams` and use it for factorials
- Exercise 3.55: Implement `partial-sums`
- Exercise 3.56: Hamming numbers (2, 3, 5 only as prime factors)
- Exercise 3.59: Power series operations
- Exercise 3.64: Stream limit with tolerance
- Exercise 3.65: ln(2) approximation with acceleration

All examples are implemented in `section_3_5.rs` with comprehensive tests!
