//! Section 3.1: Assignment and Local State
//!
//! This section demonstrates idiomatic Rust approaches to modeling stateful objects.
//! Instead of using `Rc<RefCell<T>>` (a code smell), we use:
//!
//! - **Pure functional patterns**: Operations return new state instead of mutating
//! - **`Cell<T>`**: For simple `Copy` types when interior mutability is truly needed
//! - **Struct-based APIs**: Replace closures-with-state with explicit structs
//!
//! # Key Rust Concepts
//!
//! - `Cell<T>`: Simple interior mutability for `Copy` types (no borrow API needed)
//! - Functional updates: `fn withdraw(&self, amount) -> (Self, Result<...>)`
//! - Struct methods: Clearer ownership than closure captures
//!
//! # Comparison: Scheme vs Idiomatic Rust
//!
//! | Scheme | Anti-Pattern Rust | Idiomatic Rust |
//! |--------|-------------------|----------------|
//! | `set!` | `RefCell::borrow_mut()` | Return new value |
//! | Closure with state | `Rc<RefCell<T>>` in closure | Struct with methods |
//! | Message passing | `match msg { ... borrow_mut ... }` | Enum + struct methods |

use std::cell::Cell;

// ============================================================================
// 3.1.1: Local State Variables - Functional Approach
// ============================================================================

/// A bank account with balance tracking.
///
/// Instead of mutable closures with `RefCell`, we use a simple struct.
/// Operations return a new `Account` (functional update pattern).
///
/// # Example
///
/// ```
/// use sicp_chapter3::section_3_1::Account;
///
/// let acc = Account::new(100);
/// let (acc, result) = acc.withdraw(25);
/// assert_eq!(result, Ok(75));
///
/// let (acc, result) = acc.withdraw(25);
/// assert_eq!(result, Ok(50));
///
/// let (acc, result) = acc.withdraw(60);
/// assert_eq!(result, Err("Insufficient funds"));
/// assert_eq!(acc.balance(), 50); // Balance unchanged on failure
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Account {
    balance: i64,
}

impl Account {
    /// Creates a new account with the given initial balance.
    #[must_use]
    pub fn new(initial_balance: i64) -> Self {
        Self {
            balance: initial_balance,
        }
    }

    /// Returns the current balance.
    #[must_use]
    pub fn balance(&self) -> i64 {
        self.balance
    }

    /// Withdraws the specified amount, returning (new_account, result).
    ///
    /// On success, returns the new balance. On failure, returns an error
    /// and the account is unchanged.
    pub fn withdraw(&self, amount: i64) -> (Self, Result<i64, &'static str>) {
        if self.balance >= amount {
            let new_balance = self.balance - amount;
            (
                Self {
                    balance: new_balance,
                },
                Ok(new_balance),
            )
        } else {
            (*self, Err("Insufficient funds"))
        }
    }

    /// Deposits the specified amount, returning (new_account, new_balance).
    #[must_use]
    pub fn deposit(&self, amount: i64) -> (Self, i64) {
        let new_balance = self.balance + amount;
        (
            Self {
                balance: new_balance,
            },
            new_balance,
        )
    }
}

/// Factory function that creates a withdraw closure (for compatibility).
///
/// Note: This still uses `Cell<T>` for interior mutability, but `Cell` is
/// much simpler than `RefCell` for `Copy` types - no borrow API needed.
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
    let balance = Cell::new(initial_balance);

    move |amount: i64| {
        let current = balance.get();
        if current >= amount {
            let new_balance = current - amount;
            balance.set(new_balance);
            Ok(new_balance)
        } else {
            Err("Insufficient funds")
        }
    }
}

/// Global withdraw using thread-local storage (for demonstration).
///
/// Note: Global mutable state is generally discouraged in Rust.
/// This is shown for pedagogical comparison with Scheme.
pub mod global_state {
    use std::cell::Cell;

    thread_local! {
        static BALANCE: Cell<i64> = const { Cell::new(100) };
    }

    pub fn withdraw(amount: i64) -> Result<i64, &'static str> {
        BALANCE.with(|balance| {
            let current = balance.get();
            if current >= amount {
                let new_balance = current - amount;
                balance.set(new_balance);
                Ok(new_balance)
            } else {
                Err("Insufficient funds")
            }
        })
    }

    pub fn reset_balance(new_balance: i64) {
        BALANCE.with(|balance| balance.set(new_balance));
    }
}

// ============================================================================
// Password-Protected Account (Functional)
// ============================================================================

/// A password-protected bank account.
///
/// Instead of `Rc<RefCell<T>>` with message-passing closures, we use
/// a struct with methods that return new state.
#[derive(Debug, Clone)]
pub struct PasswordAccount {
    balance: i64,
    password: String,
}

impl PasswordAccount {
    /// Creates a new password-protected account.
    #[must_use]
    pub fn new(initial_balance: i64, password: impl Into<String>) -> Self {
        Self {
            balance: initial_balance,
            password: password.into(),
        }
    }

    /// Returns the current balance (no password needed for checking).
    #[must_use]
    pub fn balance(&self) -> i64 {
        self.balance
    }

    /// Attempts to withdraw, returning (new_account, result).
    pub fn withdraw(&self, pwd: &str, amount: i64) -> (Self, Result<i64, &'static str>) {
        if pwd != self.password {
            return (self.clone(), Err("Incorrect password"));
        }
        if self.balance >= amount {
            let new_balance = self.balance - amount;
            (
                Self {
                    balance: new_balance,
                    password: self.password.clone(),
                },
                Ok(new_balance),
            )
        } else {
            (self.clone(), Err("Insufficient funds"))
        }
    }

    /// Attempts to deposit, returning (new_account, result).
    pub fn deposit(&self, pwd: &str, amount: i64) -> (Self, Result<i64, &'static str>) {
        if pwd != self.password {
            return (self.clone(), Err("Incorrect password"));
        }
        let new_balance = self.balance + amount;
        (
            Self {
                balance: new_balance,
                password: self.password.clone(),
            },
            Ok(new_balance),
        )
    }
}

/// A secure account that tracks failed password attempts.
#[derive(Debug, Clone)]
pub struct SecureAccount {
    balance: i64,
    password: String,
    failed_attempts: u8,
}

impl SecureAccount {
    /// Creates a new secure account.
    #[must_use]
    pub fn new(initial_balance: i64, password: impl Into<String>) -> Self {
        Self {
            balance: initial_balance,
            password: password.into(),
            failed_attempts: 0,
        }
    }

    /// Returns the current balance.
    #[must_use]
    pub fn balance(&self) -> i64 {
        self.balance
    }

    /// Returns the number of failed password attempts.
    #[must_use]
    pub fn failed_attempts(&self) -> u8 {
        self.failed_attempts
    }

    /// Attempts to withdraw, returning (new_account, result).
    /// After 7 consecutive wrong passwords, returns "Calling the cops!".
    pub fn withdraw(&self, pwd: &str, amount: i64) -> (Self, Result<i64, &'static str>) {
        if pwd != self.password {
            let new_attempts = self.failed_attempts + 1;
            let error = if new_attempts >= 7 {
                "Calling the cops!"
            } else {
                "Incorrect password"
            };
            return (
                Self {
                    balance: self.balance,
                    password: self.password.clone(),
                    failed_attempts: new_attempts,
                },
                Err(error),
            );
        }

        // Reset failed attempts on successful auth
        if self.balance >= amount {
            let new_balance = self.balance - amount;
            (
                Self {
                    balance: new_balance,
                    password: self.password.clone(),
                    failed_attempts: 0,
                },
                Ok(new_balance),
            )
        } else {
            (
                Self {
                    balance: self.balance,
                    password: self.password.clone(),
                    failed_attempts: 0,
                },
                Err("Insufficient funds"),
            )
        }
    }

    /// Attempts to deposit, returning (new_account, result).
    pub fn deposit(&self, pwd: &str, amount: i64) -> (Self, Result<i64, &'static str>) {
        if pwd != self.password {
            let new_attempts = self.failed_attempts + 1;
            let error = if new_attempts >= 7 {
                "Calling the cops!"
            } else {
                "Incorrect password"
            };
            return (
                Self {
                    balance: self.balance,
                    password: self.password.clone(),
                    failed_attempts: new_attempts,
                },
                Err(error),
            );
        }

        let new_balance = self.balance + amount;
        (
            Self {
                balance: new_balance,
                password: self.password.clone(),
                failed_attempts: 0,
            },
            Ok(new_balance),
        )
    }
}

// ============================================================================
// 3.1.2: Random Numbers
// ============================================================================

/// Linear Congruential Generator for pseudo-random numbers.
///
/// Uses `Cell<u64>` for state since `u64` is `Copy` - simpler than `RefCell`.
pub struct RandomGenerator {
    state: Cell<u64>,
    a: u64,
    c: u64,
    m: u64,
}

impl RandomGenerator {
    /// Creates a new random generator with default parameters.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: Cell::new(seed),
            a: 1664525,
            c: 1013904223,
            m: 1u64 << 32,
        }
    }

    /// Generates the next random number.
    pub fn rand(&self) -> u64 {
        let current = self.state.get();
        let next = (self.a.wrapping_mul(current).wrapping_add(self.c)) % self.m;
        self.state.set(next);
        next
    }

    /// Generates a random number in range [0, max).
    pub fn rand_max(&self, max: u64) -> u64 {
        self.rand() % max
    }

    /// Resets to a specific seed.
    pub fn reset(&self, seed: u64) {
        self.state.set(seed);
    }
}

/// Monte Carlo simulation.
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

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Estimates pi using Monte Carlo and Cesaro's theorem.
pub fn estimate_pi(trials: u64) -> f64 {
    let rng = RandomGenerator::new(12345);

    let cesaro_test = || {
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

/// Exercise 3.1: Accumulator (functional version)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Accumulator {
    sum: i64,
}

impl Accumulator {
    #[must_use]
    pub fn new(initial: i64) -> Self {
        Self { sum: initial }
    }

    #[must_use]
    pub fn add(&self, amount: i64) -> (Self, i64) {
        let new_sum = self.sum + amount;
        (Self { sum: new_sum }, new_sum)
    }

    #[must_use]
    pub fn value(&self) -> i64 {
        self.sum
    }
}

/// Closure-based accumulator using Cell (for compatibility).
pub fn make_accumulator(initial: i64) -> impl FnMut(i64) -> i64 {
    let sum = Cell::new(initial);

    move |amount: i64| {
        let new_sum = sum.get() + amount;
        sum.set(new_sum);
        new_sum
    }
}

/// Exercise 3.2: Monitored procedure
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

/// Exercise 3.5: Monte Carlo integration
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

/// Estimates pi using Monte Carlo integration.
pub fn estimate_pi_integral(trials: u64) -> f64 {
    let in_circle = |x: f64, y: f64| x * x + y * y <= 1.0;
    estimate_integral(in_circle, -1.0, 1.0, -1.0, 1.0, trials)
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
            new_seed
        }
    }
}

/// Exercise 3.7: Joint accounts (functional)
///
/// Creates a view of an account with a different password.
#[derive(Debug, Clone)]
pub struct JointAccount {
    inner: PasswordAccount,
    joint_password: String,
}

impl JointAccount {
    pub fn new(
        account: PasswordAccount,
        original_pwd: &str,
        new_pwd: impl Into<String>,
    ) -> Option<Self> {
        // Verify the original password works
        let (_, result) = account.withdraw(original_pwd, 0);
        if result.is_err() && result != Err("Insufficient funds") {
            return None;
        }
        Some(Self {
            inner: account,
            joint_password: new_pwd.into(),
        })
    }

    pub fn withdraw(&self, pwd: &str, amount: i64) -> (Self, Result<i64, &'static str>) {
        if pwd != self.joint_password {
            return (self.clone(), Err("Incorrect password"));
        }
        // We need the original password - but in functional style we'd need to store it
        // For simplicity, this shows the concept
        let (new_inner, result) = self.inner.withdraw(&self.inner.password, amount);
        (
            Self {
                inner: new_inner,
                joint_password: self.joint_password.clone(),
            },
            result,
        )
    }
}

/// Exercise 3.8: Order-dependent evaluation
pub fn make_order_dependent() -> impl FnMut(i32) -> i32 {
    let state = Cell::new(1i32);

    move |x: i32| {
        let current = state.get();
        let result = current * x;
        state.set(x);
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
    fn test_account_functional() {
        let acc = Account::new(100);

        let (acc, result) = acc.withdraw(25);
        assert_eq!(result, Ok(75));

        let (acc, result) = acc.withdraw(25);
        assert_eq!(result, Ok(50));

        let (acc, result) = acc.withdraw(60);
        assert_eq!(result, Err("Insufficient funds"));
        assert_eq!(acc.balance(), 50);

        let (acc, result) = acc.withdraw(15);
        assert_eq!(result, Ok(35));
        assert_eq!(acc.balance(), 35);
    }

    #[test]
    fn test_account_deposit() {
        let acc = Account::new(100);
        let (acc, balance) = acc.deposit(50);
        assert_eq!(balance, 150);
        assert_eq!(acc.balance(), 150);
    }

    #[test]
    fn test_global_withdraw() {
        global_state::reset_balance(100);

        assert_eq!(global_state::withdraw(25), Ok(75));
        assert_eq!(global_state::withdraw(25), Ok(50));
        assert_eq!(global_state::withdraw(60), Err("Insufficient funds"));
        assert_eq!(global_state::withdraw(15), Ok(35));
    }

    #[test]
    fn test_make_withdraw() {
        let mut w1 = make_withdraw(100);
        let mut w2 = make_withdraw(100);

        assert_eq!(w1(50), Ok(50));
        assert_eq!(w2(70), Ok(30));
        assert_eq!(w2(40), Err("Insufficient funds"));
        assert_eq!(w1(40), Ok(10));
    }

    #[test]
    fn test_password_account() {
        let acc = PasswordAccount::new(100, "secret");

        let (acc, result) = acc.withdraw("secret", 40);
        assert_eq!(result, Ok(60));

        let (acc, result) = acc.withdraw("wrong", 10);
        assert_eq!(result, Err("Incorrect password"));
        assert_eq!(acc.balance(), 60);

        let (acc, result) = acc.deposit("secret", 20);
        assert_eq!(result, Ok(80));
        assert_eq!(acc.balance(), 80);
    }

    #[test]
    fn test_secure_account() {
        let acc = SecureAccount::new(100, "secret");

        // Try 6 wrong passwords
        let mut acc = acc;
        for _ in 0..6 {
            let (new_acc, result) = acc.withdraw("wrong", 10);
            assert_eq!(result, Err("Incorrect password"));
            acc = new_acc;
        }

        // 7th attempt should call the cops
        let (_, result) = acc.withdraw("wrong", 10);
        assert_eq!(result, Err("Calling the cops!"));
    }

    #[test]
    fn test_random_generator() {
        let rng = RandomGenerator::new(42);

        let r1 = rng.rand();
        let r2 = rng.rand();
        let r3 = rng.rand();

        assert_ne!(r1, r2);
        assert_ne!(r2, r3);

        rng.reset(42);
        assert_eq!(rng.rand(), r1);
        assert_eq!(rng.rand(), r2);
    }

    #[test]
    fn test_monte_carlo() {
        let rng = RandomGenerator::new(12345);

        let result = monte_carlo(1000, || true);
        assert!((result - 1.0).abs() < 0.001);

        let result = monte_carlo(10000, || rng.rand() % 2 == 0);
        assert!((result - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_estimate_pi() {
        let pi_estimate = estimate_pi(100_000);
        assert!(
            pi_estimate > 2.0 && pi_estimate < 4.0,
            "Pi estimate {} should be in range [2, 4]",
            pi_estimate
        );
    }

    #[test]
    fn test_accumulator_functional() {
        let acc = Accumulator::new(5);
        let (acc, result) = acc.add(10);
        assert_eq!(result, 15);
        let (_, result) = acc.add(10);
        assert_eq!(result, 25);
    }

    #[test]
    fn test_accumulator_closure() {
        let mut acc = make_accumulator(5);
        assert_eq!(acc(10), 15);
        assert_eq!(acc(10), 25);
    }

    #[test]
    fn test_monitored() {
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
    fn test_estimate_pi_integral() {
        let pi_estimate = estimate_pi_integral(100_000);
        assert!((pi_estimate - std::f64::consts::PI).abs() < 0.1);
    }

    #[test]
    fn test_resettable_rand() {
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
    fn test_order_dependent() {
        let mut f = make_order_dependent();
        let result1 = f(0) + f(1);
        assert_eq!(result1, 0);

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
