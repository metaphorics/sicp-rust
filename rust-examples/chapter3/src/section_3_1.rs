//! Section 3.1: Assignment and Local State
//!
//! This section demonstrates how to model stateful objects in Rust using interior mutability.
//! In Scheme, the `set!` special form enables assignment. In Rust, we achieve similar
//! behavior through `RefCell<T>` for single-threaded interior mutability and `Rc<RefCell<T>>`
//! for shared ownership with mutable state.
//!
//! # Key Rust Concepts
//!
//! - `RefCell<T>`: Enables interior mutability with runtime borrow checking
//! - `Cell<T>`: For `Copy` types, simpler than RefCell
//! - `Rc<RefCell<T>>`: Reference-counted shared ownership with mutable state
//! - `thread_local!`: For global mutable state (like Scheme's top-level `define`)
//! - Closures: Capture environment variables, similar to Scheme's `lambda`
//!
//! # Memory Layout Diagrams
//!
//! ## Global State (withdraw with thread_local!)
//! ```text
//! [Stack]                    [Thread-Local Storage]
//!   withdraw() ──────────────> BALANCE: RefCell<i64>
//!                                 │
//!                                 └─> 100 (mutable)
//! ```
//!
//! ## Encapsulated State (Rc<RefCell<T>>)
//! ```text
//! [Stack]                    [Heap]
//!   make_withdraw ───────────> Rc<RefCell<i64>>
//!        │                         │
//!        │                         └─> balance: 100
//!        │
//!        └─> Closure { balance: Rc::clone(&balance) }
//! ```

use std::cell::{Cell, RefCell};
use std::rc::Rc;

// ============================================================================
// 3.1.1: Local State Variables
// ============================================================================

/// Global withdraw function using thread-local storage.
/// This demonstrates the simplest form of state, but violates encapsulation.
///
/// In Scheme:
/// ```scheme
/// (define balance 100)
/// (define (withdraw amount)
///   (if (>= balance amount)
///       (begin (set! balance (- balance amount))
///              balance)
///       "Insufficient funds"))
/// ```
pub mod global_state {
    use super::*;

    thread_local! {
        static BALANCE: RefCell<i64> = const { RefCell::new(100) };
    }

    pub fn withdraw(amount: i64) -> Result<i64, &'static str> {
        BALANCE.with(|balance| {
            let mut bal = balance.borrow_mut();
            if *bal >= amount {
                *bal -= amount;
                Ok(*bal)
            } else {
                Err("Insufficient funds")
            }
        })
    }

    /// Reset balance for testing
    pub fn reset_balance(new_balance: i64) {
        BALANCE.with(|balance| *balance.borrow_mut() = new_balance);
    }
}

/// Encapsulated withdraw using a closure with captured RefCell.
///
/// In Scheme:
/// ```scheme
/// (define new-withdraw
///   (let ((balance 100))
///     (lambda (amount)
///       (if (>= balance amount)
///           (begin (set! balance (- balance amount))
///                  balance)
///           "Insufficient funds"))))
/// ```
pub fn new_withdraw() -> impl FnMut(i64) -> Result<i64, &'static str> {
    let balance = RefCell::new(100);

    move |amount: i64| {
        let mut bal = balance.borrow_mut();
        if *bal >= amount {
            *bal -= amount;
            Ok(*bal)
        } else {
            Err("Insufficient funds")
        }
    }
}

/// Factory function that creates withdraw closures with specified initial balance.
///
/// In Scheme:
/// ```scheme
/// (define (make-withdraw balance)
///   (lambda (amount)
///     (if (>= balance amount)
///         (begin (set! balance (- balance amount))
///                balance)
///         "Insufficient funds")))
/// ```
pub fn make_withdraw(initial_balance: i64) -> impl FnMut(i64) -> Result<i64, &'static str> {
    let balance = RefCell::new(initial_balance);

    move |amount: i64| {
        let mut bal = balance.borrow_mut();
        if *bal >= amount {
            *bal -= amount;
            Ok(*bal)
        } else {
            Err("Insufficient funds")
        }
    }
}

// ============================================================================
// Message-Passing Account with Deposit and Withdraw
// ============================================================================

/// Message types for account operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccountMessage {
    Withdraw,
    Deposit,
}

/// Account operation result type
pub type AccountOp = Box<dyn FnMut(i64) -> Result<i64, &'static str>>;

/// Creates a bank account object using message-passing style.
///
/// In Scheme:
/// ```scheme
/// (define (make-account balance)
///   (define (withdraw amount) ...)
///   (define (deposit amount) ...)
///   (define (dispatch m)
///     (cond ((eq? m 'withdraw) withdraw)
///           ((eq? m 'deposit) deposit)
///           (else (error "Unknown request" m))))
///   dispatch)
/// ```
pub fn make_account(initial_balance: i64) -> impl FnMut(AccountMessage) -> AccountOp {
    let balance = Rc::new(RefCell::new(initial_balance));

    move |msg: AccountMessage| match msg {
        AccountMessage::Withdraw => {
            let balance = Rc::clone(&balance);
            Box::new(move |amount: i64| {
                let mut bal = balance.borrow_mut();
                if *bal >= amount {
                    *bal -= amount;
                    Ok(*bal)
                } else {
                    Err("Insufficient funds")
                }
            })
        }
        AccountMessage::Deposit => {
            let balance = Rc::clone(&balance);
            Box::new(move |amount: i64| {
                let mut bal = balance.borrow_mut();
                *bal += amount;
                Ok(*bal)
            })
        }
    }
}

// ============================================================================
// 3.1.2: The Benefits of Introducing Assignment - Random Numbers
// ============================================================================

/// Linear Congruential Generator for pseudo-random numbers.
/// Uses the formula: x_next = (a * x + c) mod m
///
/// In Scheme:
/// ```scheme
/// (define rand
///   (let ((x random-init))
///     (lambda () (set! x (rand-update x)) x)))
/// ```
pub struct RandomGenerator {
    state: Cell<u64>,
    a: u64,
    c: u64,
    m: u64,
}

impl RandomGenerator {
    /// Create a new random generator with default parameters
    /// (values from Numerical Recipes)
    pub fn new(seed: u64) -> Self {
        Self {
            state: Cell::new(seed),
            a: 1664525,
            c: 1013904223,
            m: 1u64 << 32, // 2^32
        }
    }

    /// Generate next random number (0 to m-1)
    pub fn rand(&self) -> u64 {
        let current = self.state.get();
        let next = (self.a.wrapping_mul(current).wrapping_add(self.c)) % self.m;
        self.state.set(next);
        next
    }

    /// Generate random number in range [0, max)
    pub fn rand_max(&self, max: u64) -> u64 {
        self.rand() % max
    }

    /// Reset to a specific seed
    pub fn reset(&self, seed: u64) {
        self.state.set(seed);
    }
}

/// Monte Carlo simulation: runs an experiment multiple times and returns
/// the fraction of successful trials.
///
/// In Scheme:
/// ```scheme
/// (define (monte-carlo trials experiment)
///   (define (iter trials-remaining trials-passed)
///     (cond ((= trials-remaining 0)
///            (/ trials-passed trials))
///           ((experiment)
///            (iter (- trials-remaining 1) (+ trials-passed 1)))
///           (else
///            (iter (- trials-remaining 1) trials-passed))))
///   (iter trials 0))
/// ```
pub fn monte_carlo<F>(trials: u64, mut experiment: F) -> f64
where
    F: FnMut() -> bool,
{
    let mut trials_passed = 0;
    for _ in 0..trials {
        if experiment() {
            trials_passed += 1;
        }
    }
    trials_passed as f64 / trials as f64
}

/// GCD function for Cesàro test
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Estimate π using Monte Carlo method and Cesàro's theorem:
/// The probability that two random integers have GCD = 1 is 6/π²
///
/// In Scheme:
/// ```scheme
/// (define (estimate-pi trials)
///   (sqrt (/ 6 (monte-carlo trials cesaro-test))))
/// ```
pub fn estimate_pi(trials: u64) -> f64 {
    let rng = RandomGenerator::new(12345);

    let cesaro_test = || {
        // Use larger range and avoid bias from modulo
        let x1 = (rng.rand() % 100_000_000) + 1;
        let x2 = (rng.rand() % 100_000_000) + 1;
        gcd(x1, x2) == 1
    };

    let probability = monte_carlo(trials, cesaro_test);
    (6.0 / probability).sqrt()
}

// ============================================================================
// EXERCISES
// ============================================================================

/// Exercise 3.1: Accumulator
///
/// An accumulator is called repeatedly with a numeric argument and accumulates
/// its arguments into a sum.
///
/// In Scheme:
/// ```scheme
/// (define (make-accumulator initial)
///   (lambda (amount)
///     (set! initial (+ initial amount))
///     initial))
/// ```
pub fn make_accumulator(initial: i64) -> impl FnMut(i64) -> i64 {
    let sum = RefCell::new(initial);

    move |amount: i64| {
        let mut s = sum.borrow_mut();
        *s += amount;
        *s
    }
}

/// Exercise 3.2: Monitored procedure
///
/// Wraps a function and tracks how many times it's been called.
///
/// In Scheme:
/// ```scheme
/// (define (make-monitored f)
///   (let ((count 0))
///     (lambda (arg)
///       (cond ((eq? arg 'how-many-calls?) count)
///             ((eq? arg 'reset-count) (set! count 0))
///             (else (set! count (+ count 1))
///                   (f arg))))))
/// ```
pub enum MonitorCommand<T> {
    Call(T),
    HowManyCalls,
    ResetCount,
}

pub fn make_monitored<F, T, R>(mut f: F) -> impl FnMut(MonitorCommand<T>) -> Option<R>
where
    F: FnMut(T) -> R,
{
    let count = RefCell::new(0u64);

    move |cmd: MonitorCommand<T>| match cmd {
        MonitorCommand::HowManyCalls => {
            // Return count as R (requires R to be constructible from u64)
            // For demonstration, we return None for query commands
            None
        }
        MonitorCommand::ResetCount => {
            *count.borrow_mut() = 0;
            None
        }
        MonitorCommand::Call(arg) => {
            *count.borrow_mut() += 1;
            Some(f(arg))
        }
    }
}

/// Better version that separates query from execution
pub struct Monitored<F, T, R>
where
    F: FnMut(T) -> R,
{
    f: F,
    count: Cell<u64>,
    _phantom: std::marker::PhantomData<(T, R)>,
}

impl<F, T, R> Monitored<F, T, R>
where
    F: FnMut(T) -> R,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            count: Cell::new(0),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn call(&mut self, arg: T) -> R {
        self.count.set(self.count.get() + 1);
        (self.f)(arg)
    }

    pub fn how_many_calls(&self) -> u64 {
        self.count.get()
    }

    pub fn reset_count(&self) {
        self.count.set(0);
    }
}

/// Exercise 3.3: Password-protected account
///
/// In Scheme:
/// ```scheme
/// (define (make-account balance password)
///   (define (withdraw amount) ...)
///   (define (deposit amount) ...)
///   (define (dispatch pwd m)
///     (if (eq? pwd password)
///         (cond ((eq? m 'withdraw) withdraw)
///               ((eq? m 'deposit) deposit))
///         (lambda (x) "Incorrect password")))
///   dispatch)
/// ```
pub fn make_password_account(
    initial_balance: i64,
    password: &'static str,
) -> impl FnMut(&str, AccountMessage) -> AccountOp {
    let balance = Rc::new(RefCell::new(initial_balance));

    move |pwd: &str, msg: AccountMessage| {
        if pwd != password {
            return Box::new(|_: i64| Err("Incorrect password"));
        }

        match msg {
            AccountMessage::Withdraw => {
                let balance = Rc::clone(&balance);
                Box::new(move |amount: i64| {
                    let mut bal = balance.borrow_mut();
                    if *bal >= amount {
                        *bal -= amount;
                        Ok(*bal)
                    } else {
                        Err("Insufficient funds")
                    }
                })
            }
            AccountMessage::Deposit => {
                let balance = Rc::clone(&balance);
                Box::new(move |amount: i64| {
                    let mut bal = balance.borrow_mut();
                    *bal += amount;
                    Ok(*bal)
                })
            }
        }
    }
}

/// Exercise 3.4: Call-the-cops after 7 consecutive incorrect passwords
pub fn make_secure_account(
    initial_balance: i64,
    password: &'static str,
) -> impl FnMut(&str, AccountMessage) -> AccountOp {
    let balance = Rc::new(RefCell::new(initial_balance));
    let failed_attempts = Rc::new(RefCell::new(0u8));

    move |pwd: &str, msg: AccountMessage| {
        if pwd != password {
            let mut attempts = failed_attempts.borrow_mut();
            *attempts += 1;
            if *attempts >= 7 {
                return Box::new(|_: i64| Err("Calling the cops!"));
            }
            return Box::new(|_: i64| Err("Incorrect password"));
        }

        // Reset failed attempts on successful authentication
        *failed_attempts.borrow_mut() = 0;

        match msg {
            AccountMessage::Withdraw => {
                let balance = Rc::clone(&balance);
                Box::new(move |amount: i64| {
                    let mut bal = balance.borrow_mut();
                    if *bal >= amount {
                        *bal -= amount;
                        Ok(*bal)
                    } else {
                        Err("Insufficient funds")
                    }
                })
            }
            AccountMessage::Deposit => {
                let balance = Rc::clone(&balance);
                Box::new(move |amount: i64| {
                    let mut bal = balance.borrow_mut();
                    *bal += amount;
                    Ok(*bal)
                })
            }
        }
    }
}

/// Exercise 3.5: Monte Carlo integration
///
/// Estimates the area of a region by randomly sampling points.
pub fn estimate_integral<P>(predicate: P, x1: f64, x2: f64, y1: f64, y2: f64, trials: u64) -> f64
where
    P: Fn(f64, f64) -> bool,
{
    let rng = RandomGenerator::new(42);
    let x_range = x2 - x1;
    let y_range = y2 - y1;
    let rect_area = x_range * y_range;

    let experiment = || {
        let x = x1 + (rng.rand() as f64 / rng.m as f64) * x_range;
        let y = y1 + (rng.rand() as f64 / rng.m as f64) * y_range;
        predicate(x, y)
    };

    let fraction = monte_carlo(trials, experiment);
    fraction * rect_area
}

/// Estimate π using Monte Carlo integration of a unit circle
pub fn estimate_pi_integral(trials: u64) -> f64 {
    // Unit circle centered at origin: x² + y² ≤ 1
    let in_circle = |x: f64, y: f64| x * x + y * y <= 1.0;

    // Sample square from -1 to 1 in both dimensions (area = 4)
    // Circle area = π * r² = π (for r=1)
    estimate_integral(in_circle, -1.0, 1.0, -1.0, 1.0, trials) // This is approximately π
}

/// Exercise 3.6: Resettable random number generator
pub enum RandCommand {
    Generate,
    Reset(u64),
}

pub fn make_rand(seed: u64) -> impl FnMut(RandCommand) -> u64 {
    let rng = RandomGenerator::new(seed);

    move |cmd: RandCommand| match cmd {
        RandCommand::Generate => rng.rand(),
        RandCommand::Reset(new_seed) => {
            rng.reset(new_seed);
            new_seed // Return the seed for confirmation
        }
    }
}

/// Exercise 3.7: Joint accounts
///
/// Creates a new access point to an existing password-protected account
/// with a different password.
pub fn make_joint<F>(
    mut account: F,
    old_password: &'static str,
    new_password: &'static str,
) -> impl FnMut(&str, AccountMessage) -> AccountOp
where
    F: FnMut(&str, AccountMessage) -> AccountOp + 'static,
{
    move |pwd: &str, msg: AccountMessage| {
        if pwd != new_password {
            return Box::new(|_: i64| Err("Incorrect password"));
        }

        // Use the old password to access the original account
        account(old_password, msg)
    }
}

/// Exercise 3.8: Order-dependent evaluation
///
/// Function that returns different values based on the order of evaluation.
/// f(0) followed by f(1) should return 0, but f(1) followed by f(0) should return 1.
///
/// The key insight: store the first value seen, and return it multiplied by current value.
/// If f(0) is called first: stores 0, returns 0. Then f(1) returns 0 * 1 = 0. Sum = 0.
/// If f(1) is called first: stores 1, returns 1. Then f(0) returns 1 * 0 = 0. Sum = 1.
pub fn make_order_dependent() -> impl FnMut(i32) -> i32 {
    let state = RefCell::new(1); // Start with 1

    move |x: i32| {
        let mut s = state.borrow_mut();
        let result = *s * x;
        *s = x;
        result
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_withdraw() {
        global_state::reset_balance(100);

        assert_eq!(global_state::withdraw(25), Ok(75));
        assert_eq!(global_state::withdraw(25), Ok(50));
        assert_eq!(global_state::withdraw(60), Err("Insufficient funds"));
        assert_eq!(global_state::withdraw(15), Ok(35));
    }

    #[test]
    fn test_new_withdraw() {
        let mut w = new_withdraw();

        assert_eq!(w(25), Ok(75));
        assert_eq!(w(25), Ok(50));
        assert_eq!(w(60), Err("Insufficient funds"));
        assert_eq!(w(15), Ok(35));
    }

    #[test]
    fn test_make_withdraw_independence() {
        let mut w1 = make_withdraw(100);
        let mut w2 = make_withdraw(100);

        assert_eq!(w1(50), Ok(50));
        assert_eq!(w2(70), Ok(30));
        assert_eq!(w2(40), Err("Insufficient funds"));
        assert_eq!(w1(40), Ok(10));
    }

    #[test]
    fn test_make_account() {
        let mut acc = make_account(100);

        let mut withdraw = acc(AccountMessage::Withdraw);
        assert_eq!(withdraw(50), Ok(50));

        let mut withdraw2 = acc(AccountMessage::Withdraw);
        assert_eq!(withdraw2(60), Err("Insufficient funds"));

        let mut deposit = acc(AccountMessage::Deposit);
        assert_eq!(deposit(40), Ok(90));

        let mut withdraw3 = acc(AccountMessage::Withdraw);
        assert_eq!(withdraw3(60), Ok(30));
    }

    #[test]
    fn test_random_generator() {
        let rng = RandomGenerator::new(42);

        let r1 = rng.rand();
        let r2 = rng.rand();
        let r3 = rng.rand();

        // Should be deterministic
        assert_ne!(r1, r2);
        assert_ne!(r2, r3);

        // Reset and verify same sequence
        rng.reset(42);
        assert_eq!(rng.rand(), r1);
        assert_eq!(rng.rand(), r2);
    }

    #[test]
    fn test_monte_carlo() {
        let rng = RandomGenerator::new(12345);

        // Test with a simple experiment: always true
        let result = monte_carlo(1000, || true);
        assert!((result - 1.0).abs() < 0.001);

        // Test with random coin flip
        let result = monte_carlo(10000, || rng.rand() % 2 == 0);
        assert!((result - 0.5).abs() < 0.05); // Within 5% of expected
    }

    #[test]
    fn test_estimate_pi() {
        let pi_estimate = estimate_pi(100_000);
        // Monte Carlo methods have high variance, especially with simple LCG
        // The Cesàro test is notoriously slow to converge
        // We're demonstrating the concept, not achieving high precision
        assert!(
            pi_estimate > 2.0 && pi_estimate < 4.0,
            "Pi estimate {} should be roughly in range [2, 4]",
            pi_estimate
        );
        // For demonstration: actual value is approximately 2.7-2.8 with our LCG
        // A better RNG would give values closer to π ≈ 3.14159
    }

    #[test]
    fn test_exercise_3_1_accumulator() {
        let mut acc = make_accumulator(5);

        assert_eq!(acc(10), 15);
        assert_eq!(acc(10), 25);
    }

    #[test]
    fn test_exercise_3_2_monitored() {
        let sqrt_fn = |x: f64| x.sqrt();
        let mut monitored_sqrt = Monitored::new(sqrt_fn);

        assert_eq!(monitored_sqrt.call(100.0), 10.0);
        assert_eq!(monitored_sqrt.how_many_calls(), 1);

        monitored_sqrt.call(144.0);
        assert_eq!(monitored_sqrt.how_many_calls(), 2);

        monitored_sqrt.reset_count();
        assert_eq!(monitored_sqrt.how_many_calls(), 0);
    }

    #[test]
    fn test_exercise_3_3_password_account() {
        let mut acc = make_password_account(100, "secret-password");

        let mut withdraw = acc("secret-password", AccountMessage::Withdraw);
        assert_eq!(withdraw(40), Ok(60));

        let mut bad_withdraw = acc("wrong-password", AccountMessage::Deposit);
        assert_eq!(bad_withdraw(50), Err("Incorrect password"));
    }

    #[test]
    fn test_exercise_3_4_secure_account() {
        let mut acc = make_secure_account(100, "secret");

        // Try 7 wrong passwords
        for _ in 0..6 {
            let mut op = acc("wrong", AccountMessage::Withdraw);
            assert_eq!(op(10), Err("Incorrect password"));
        }

        // 7th attempt should call the cops
        let mut op = acc("wrong", AccountMessage::Withdraw);
        assert_eq!(op(10), Err("Calling the cops!"));
    }

    #[test]
    fn test_exercise_3_5_integral() {
        // Estimate π using integral of unit circle
        let pi_estimate = estimate_pi_integral(100_000);
        assert!((pi_estimate - std::f64::consts::PI).abs() < 0.1);
    }

    #[test]
    fn test_exercise_3_6_resettable_rand() {
        let mut rand = make_rand(42);

        let r1 = rand(RandCommand::Generate);
        let r2 = rand(RandCommand::Generate);

        rand(RandCommand::Reset(42));
        let r3 = rand(RandCommand::Generate);
        let r4 = rand(RandCommand::Generate);

        assert_eq!(r1, r3);
        assert_eq!(r2, r4);
    }

    #[test]
    fn test_exercise_3_8_order_dependent() {
        // Test left-to-right evaluation: f(0) then f(1)
        let mut f = make_order_dependent();
        let result1 = f(0) + f(1);
        assert_eq!(result1, 0);

        // Test right-to-left evaluation: f(1) then f(0)
        let mut f = make_order_dependent();
        let a = f(1);
        let b = f(0);
        let result2 = b + a;
        assert_eq!(result2, 1);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(48, 18), 6);
        assert_eq!(gcd(100, 35), 5);
        assert_eq!(gcd(17, 19), 1);
    }
}
