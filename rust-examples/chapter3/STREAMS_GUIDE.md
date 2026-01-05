# SICP 3.5장: Rust에서의 스트림 (Streams in Rust)

## 개요 (Overview)

이 가이드는 SICP의 스트림 개념이 Rust의 `Iterator` 트레이트와 어떻게 매핑되는지 설명합니다.
Scheme에서 스트림은 지연 평가(lazy evaluation)를 위해 `delay`와 `force`를 사용합니다.
Rust에서는 `Iterator` 트레이트가 기본적으로 지연 평가를 제공합니다.

## 개념 매핑 (Concept Mappings)

| Scheme 개념        | Rust 대응                    | 예시                             |
| ------------------ | ---------------------------- | -------------------------------- |
| `delay`            | 이터레이터 (기본적으로 지연) | `(0..).filter(...)`              |
| `force`            | `.next()` 또는 `.collect()`  | `iter.next()`                    |
| `cons-stream`      | 이터레이터 생성자            | `std::iter::once(x).chain(rest)` |
| `stream-car`       | `.next().unwrap()`           | `stream.next().unwrap()`         |
| `stream-cdr`       | 이터레이터 자체              | `stream.skip(1)`                 |
| `stream-map`       | `.map()`                     | `stream.map(\|x\| x * 2)`        |
| `stream-filter`    | `.filter()`                  | `stream.filter(\|&x\| x > 0)`    |
| `stream-ref`       | `.nth()`                     | `stream.nth(5)`                  |
| `the-empty-stream` | `std::iter::empty()`         | `std::iter::empty::<i32>()`      |
| 무한 스트림        | 사용자 정의 이터레이터 구현  | `struct IntegersFrom { ... }`    |

## 핵심 기능 (Key Features)

### 1. 지연 평가 (Lazy Evaluation)

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

### 2. 무한 시퀀스 (Infinite Sequences)

**피보나치 (Scheme):**

```scheme
(define (fibgen a b)
  (cons-stream a (fibgen b (+ a b))))
(define fibs (fibgen 0 1))
```

**피보나치 (Rust):**

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

### 3. 스트림 연산 (Stream Operations)

**스트림 더하기 (Scheme):**

```scheme
(define (add-streams s1 s2)
  (stream-map + s1 s2))
```

**스트림 더하기 (Rust):**

```rust
fn add_streams<I1, I2>(s1: I1, s2: I2) -> impl Iterator<Item = i64>
where
    I1: Iterator<Item = i64>,
    I2: Iterator<Item = i64>,
{
    s1.zip(s2).map(|(a, b)| a + b)
}
```

### 4. 에라토스테네스의 체 (Sieve of Eratosthenes)

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

### 5. 암시적 스트림 정의 (Implicit Stream Definitions)

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

### 6. 신호 처리 (Signal Processing)

**적분 (Scheme):**

```scheme
(define (integral integrand initial-value dt)
  (define int
    (cons-stream
     initial-value
     (add-streams (scale-stream integrand dt) int)))
  int)
```

**적분 (Rust):**

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

## 예제 실행 (Running Examples)

### 모든 테스트 실행:

```bash
cargo test --lib section_3_5
```

### 데모 실행:

```bash
cargo run --example streams_demo
```

### 예상 테스트 출력:

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

## 핵심 통찰 (Key Insights)

1. **기본적인 지연 평가 (Lazy by Default)**: Rust 이터레이터는 기본적으로 지연 평가되므로 명시적인 delay/force가 필요 없습니다.
2. **비용 없는 추상화 (Zero-Cost Abstraction)**: 이터레이터 콤비네이터는 효율적인 기계어로 컴파일됩니다.
3. **타입 안전성 (Type Safety)**: 타입 시스템이 스트림의 올바른 구성을 보장합니다.
4. **소유권 (Ownership)**: 이터레이터 소유권은 공유된 가변 상태 문제를 방지합니다.
5. **합성 가능성 (Composability)**: 이터레이터 콤비네이터들(`map`, `filter`, `take` 등)은 자연스럽게 합성됩니다.
6. **무한 시퀀스 (Infinite Sequences)**: Rust에서도 Scheme만큼 자연스럽게 무한 시퀀스를 다룰 수 있습니다.

## 고급 패턴 (Advanced Patterns)

### 수열 가속 (오일러 변환)

오일러 변환은 교대 급수의 수렴을 가속화합니다:

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

### 미분 방정식 풀이 (Differential Equation Solver)

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

## Scheme과의 차이점 (Differences from Scheme)

1. **명시적 타입 (Explicit Types)**: Rust는 사용자 정의 이터레이터에 타입 어노테이션이 필요합니다.
2. **소유권 (Ownership)**: 이터레이터를 체이닝할 때 소유권에 대해 생각해야 합니다.
3. **메모이제이션 없음 (No Memoization)**: Rust 이터레이터는 값을 자동으로 캐시하지 않습니다 (`Peekable`이나 사용자 정의 캐싱 사용).
4. **관습적인 유한성 (Finite by Convention)**: `Option<T>`가 종료를 알립니다(특별한 empty 값 대신).
5. **암시적 재귀 없음 (No Implicit Recursion)**: 재귀적 스트림 구조를 명시적으로 설계해야 합니다.

## Rust 접근 방식의 이점 (Benefits of Rust Approach)

1. **메모리 안전성 (Memory Safety)**: 댕글링 포인터(dangling pointer)나 해제 후 사용(use-after-free) 오류가 없습니다.
2. **스레드 안전성 (Thread Safety)**: Send/Sync 트레이트가 안전한 동시성을 보장합니다.
3. **성능 (Performance)**: 인라인 최적화를 통한 비용 없는 추상화.
4. **컴파일 타임 보장 (Compile-Time Guarantees)**: 타입 시스템이 많은 버그를 조기에 잡아냅니다.
5. **명확한 소유권 (Clear Ownership)**: 명시적 소유권은 미묘한 버그를 방지합니다.

## 연습 문제 (Exercises)

다음 SICP 연습 문제들을 Rust로 구현해 보세요:

- Exercise 3.53: 스트림 `s = 1 + (s + s)` 기술하기
- Exercise 3.54: `mul-streams`를 정의하고 팩토리얼에 사용하기
- Exercise 3.55: `partial-sums` 구현하기
- Exercise 3.56: 해밍 수 (소인수가 2, 3, 5뿐인 수)
- Exercise 3.59: 거듭제곱 급수 연산
- Exercise 3.64: 허용 오차를 가진 스트림 극한(limit)
- Exercise 3.65: 가속을 이용한 ln(2) 근사

모든 예제는 `section_3_5.rs`에 포괄적인 테스트와 함께 구현되어 있습니다!
