//! SICP Section 3.4: Concurrency - Time Is of the Essence
//!
//! This module demonstrates Rust's "fearless concurrency" approach to the problems
//! discussed in SICP Chapter 3.4. While SICP uses Scheme's serializers and mutexes,
//! Rust provides compile-time guarantees through its ownership system and type system.
//!
//! # Concurrency Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Concurrency Model                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                             │
//! │  Thread 1                 Shared State           Thread 2   │
//! │     │                          │                     │      │
//! │     ├─> Arc::clone() ──────────┼──────> Arc::clone()─┤      │
//! │     │                          │                     │      │
//! │     ├─> lock() ─────────> Mutex<T> <────── lock() ───┤      │
//! │     │                      (owner)                   │      │
//! │     ├─> MutexGuard<T>          │                     │      │
//! │     │   (exclusive access)     │                     │      │
//! │     └─> unlock (auto-drop)     │                     │      │
//! │                                │                     │      │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Memory Safety Guarantees
//!
//! ```text
//! Stack (Thread 1)              Heap                   Stack (Thread 2)
//! ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
//! │ Arc<Mutex<T>>├────────>│ Arc Counter  │<────────┤Arc<Mutex<T>> │
//! │   (clone)    │         │   count: 2   │         │  (clone)     │
//! └──────────────┘         │      ↓       │         └──────────────┘
//!                          │   Mutex<T>   │
//!                          │      ↓       │
//!                          │   Data: T    │
//!                          └──────────────┘
//!
//! Lock acquisition creates exclusive borrow:
//! MutexGuard<'lock, T> → &'lock mut T (can't alias while locked)
//! ```

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// ==============================================================================
// Section 3.4.1: The Nature of Time in Concurrent Systems
// ==============================================================================

/// A simple bank account with interior mutability using Mutex.
///
/// # Rust vs Scheme Comparison
///
/// Scheme (SICP):
/// ```scheme
/// (define (make-account balance)
///   (define (withdraw amount)
///     (if (>= balance amount)
///         (begin (set! balance (- balance amount))
///                balance)
///         "Insufficient funds"))
///   (define (deposit amount)
///     (set! balance (+ balance amount))
///     balance)
///   (define (dispatch m)
///     (cond ((eq? m 'withdraw) withdraw)
///           ((eq? m 'deposit) deposit)
///           ((eq? m 'balance) balance)))
///   dispatch)
/// ```
///
/// Rust provides compile-time safety through Mutex<T>:
/// - No data races possible (enforced by type system)
/// - Lock must be acquired to access data
/// - Automatic unlock via RAII (Drop trait)
/// - Send/Sync traits ensure thread safety
#[derive(Debug)]
pub struct Account {
    balance: i64,
}

impl Account {
    pub fn new(initial_balance: i64) -> Self {
        Account {
            balance: initial_balance,
        }
    }

    /// Withdraw amount from account (requires exclusive access via Mutex)
    pub fn withdraw(&mut self, amount: i64) -> Result<i64, String> {
        if self.balance >= amount {
            self.balance -= amount;
            Ok(self.balance)
        } else {
            Err("Insufficient funds".to_string())
        }
    }

    /// Deposit amount into account (requires exclusive access via Mutex)
    pub fn deposit(&mut self, amount: i64) -> i64 {
        self.balance += amount;
        self.balance
    }

    /// Get current balance (read-only, but still needs lock for consistency)
    pub fn balance(&self) -> i64 {
        self.balance
    }
}

/// Demonstrates the race condition from Figure 3.29 (SICP)
///
/// Without proper synchronization, two concurrent withdrawals can lead to
/// incorrect final balance. Rust prevents this at compile time - you MUST
/// use Mutex to share mutable state between threads.
///
/// # Concurrency Diagram: Race Condition
///
/// ```text
/// Time │ Thread 1 (Peter -$10)      │ Memory  │ Thread 2 (Paul -$25)
/// ─────┼────────────────────────────┼─────────┼──────────────────────
///   1  │ read balance: $100         │  $100   │
///   2  │ compute: 100 - 10 = 90     │  $100   │ read balance: $100
///   3  │                            │  $100   │ compute: 100 - 25 = 75
///   4  │ write balance: $90         │  $90    │
///   5  │                            │  $75    │ write balance: $75  ← WRONG!
///      │                            │         │ (overwrites Peter's update)
///
/// Expected: $100 - $10 - $25 = $65
/// Actual:   $75 (lost update!)
/// ```
///
/// Rust prevents this by requiring Mutex:
/// ```rust,ignore
/// let account = Arc::new(Mutex::new(Account::new(100)));
/// // Must lock to access: account.lock().unwrap().withdraw(10)
/// ```
pub fn demonstrate_race_condition_prevention() {
    // This demonstrates that Rust's type system prevents the race condition
    println!("\n=== Race Condition Prevention ===");
    println!("In Scheme, concurrent access to shared state requires manual serialization.");
    println!("In Rust, the type system enforces it at compile time!\n");

    let account = Arc::new(Mutex::new(Account::new(100)));

    let acc1 = Arc::clone(&account);
    let acc2 = Arc::clone(&account);

    let peter = thread::spawn(move || {
        // Lock automatically acquired and released (RAII)
        let mut acc = acc1.lock().unwrap();
        println!("Peter: Acquired lock, withdrawing $10");
        thread::sleep(Duration::from_millis(10)); // Simulate computation
        let result = acc.withdraw(10);
        println!("Peter: Withdrew $10, result: {:?}", result);
        // Lock automatically released when `acc` (MutexGuard) goes out of scope
    });

    let paul = thread::spawn(move || {
        thread::sleep(Duration::from_millis(5));
        println!("Paul: Attempting to acquire lock...");
        let mut acc = acc2.lock().unwrap(); // Blocks until lock available
        println!("Paul: Acquired lock, withdrawing $25");
        let result = acc.withdraw(25);
        println!("Paul: Withdrew $25, result: {:?}", result);
    });

    peter.join().unwrap();
    paul.join().unwrap();

    let final_balance = account.lock().unwrap().balance();
    println!("\nFinal balance: ${} (correct: $65)", final_balance);
    assert_eq!(final_balance, 65, "Race condition prevented by Mutex!");
}

// ==============================================================================
// Section 3.4.2: Mechanisms for Controlling Concurrency
// ==============================================================================

/// Serializer wrapper that mimics SICP's make-serializer
///
/// # Scheme (SICP):
/// ```scheme
/// (define (make-serializer)
///   (let ((mutex (make-mutex)))
///     (lambda (p)
///       (define (serialized-p . args)
///         (mutex 'acquire)
///         (let ((val (apply p args)))
///           (mutex 'release)
///           val))
///       serialized-p)))
/// ```
///
/// # Rust Equivalent:
///
/// The Mutex<T> type already provides serialization through lock guards.
/// This wrapper demonstrates the concept explicitly.
pub struct Serializer<T> {
    mutex: Arc<Mutex<T>>,
}

impl<T> Serializer<T> {
    pub fn new(value: T) -> Self {
        Serializer {
            mutex: Arc::new(Mutex::new(value)),
        }
    }

    /// Execute a function with serialized access to the protected value
    pub fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        let mut guard = self.mutex.lock().unwrap();
        f(&mut *guard)
        // Lock automatically released when guard drops
    }

    /// Clone the Arc for sharing between threads
    pub fn clone_arc(&self) -> Arc<Mutex<T>> {
        Arc::clone(&self.mutex)
    }
}

/// Demonstrates parallel execution with serialization (Exercise 3.38-3.40)
///
/// # Exercise 3.38 equivalent in Rust
///
/// Three concurrent operations on shared state:
/// - Peter: balance += 10
/// - Paul: balance -= 20
/// - Mary: balance -= balance / 2
///
/// Without serialization: many possible outcomes (race conditions)
/// With serialization: only 6 valid sequential orderings
pub fn parallel_execute_example() {
    println!("\n=== Parallel Execute (Exercise 3.38) ===");

    // Without proper synchronization (unsafe in Scheme, prevented by Rust type system)
    println!("Rust prevents unsynchronized access at compile time!");
    println!("Must use Arc<Mutex<T>> to share mutable state.\n");

    let balance = Arc::new(Mutex::new(100));

    let b1 = Arc::clone(&balance);
    let b2 = Arc::clone(&balance);
    let b3 = Arc::clone(&balance);

    // Peter: +10
    let peter = thread::spawn(move || {
        let mut b = b1.lock().unwrap();
        *b += 10;
        println!("Peter: balance += 10, now ${}", *b);
    });

    // Paul: -20
    let paul = thread::spawn(move || {
        let mut b = b2.lock().unwrap();
        *b -= 20;
        println!("Paul: balance -= 20, now ${}", *b);
    });

    // Mary: -= balance/2
    let mary = thread::spawn(move || {
        let mut b = b3.lock().unwrap();
        let half = *b / 2;
        *b -= half;
        println!("Mary: balance -= balance/2, now ${}", *b);
    });

    peter.join().unwrap();
    paul.join().unwrap();
    mary.join().unwrap();

    let final_balance = balance.lock().unwrap();
    println!("\nFinal balance: ${}", *final_balance);
    println!("Possible sequential results: $35, $40, $45, $50");
    // Note: Due to lock ordering, result is deterministic in this simple case
}

/// Thread-safe bank account with automatic serialization
///
/// Maps to SICP's protected make-account (page 312)
pub struct ThreadSafeAccount {
    balance: Arc<Mutex<i64>>,
}

impl ThreadSafeAccount {
    pub fn new(initial_balance: i64) -> Self {
        ThreadSafeAccount {
            balance: Arc::new(Mutex::new(initial_balance)),
        }
    }

    pub fn withdraw(&self, amount: i64) -> Result<i64, String> {
        let mut balance = self.balance.lock().unwrap();
        if *balance >= amount {
            *balance -= amount;
            Ok(*balance)
        } else {
            Err("Insufficient funds".to_string())
        }
    }

    pub fn deposit(&self, amount: i64) -> i64 {
        let mut balance = self.balance.lock().unwrap();
        *balance += amount;
        *balance
    }

    pub fn get_balance(&self) -> i64 {
        *self.balance.lock().unwrap()
    }

    /// Clone for sharing across threads (Arc reference counting)
    pub fn clone_handle(&self) -> Self {
        ThreadSafeAccount {
            balance: Arc::clone(&self.balance),
        }
    }
}

// ==============================================================================
// Deadlock Examples (Section 3.4.2 - Complexity of Multiple Shared Resources)
// ==============================================================================

/// Account with exposed serializer for exchange operations
///
/// Maps to SICP's make-account-and-serializer (page 316)
pub struct AccountWithSerializer {
    balance: i64,
    id: u64,
}

impl AccountWithSerializer {
    pub fn new(balance: i64, id: u64) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(AccountWithSerializer { balance, id }))
    }

    pub fn balance(&self) -> i64 {
        self.balance
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn withdraw(&mut self, amount: i64) -> Result<i64, String> {
        if self.balance >= amount {
            self.balance -= amount;
            Ok(self.balance)
        } else {
            Err("Insufficient funds".to_string())
        }
    }

    pub fn deposit(&mut self, amount: i64) -> i64 {
        self.balance += amount;
        self.balance
    }
}

/// Exchange balances between two accounts (NAIVE - can deadlock!)
///
/// # Deadlock Scenario (Figure 3.30 region):
///
/// ```text
/// Time │ Thread 1: exchange(a1, a2)  │ Thread 2: exchange(a2, a1)
/// ─────┼─────────────────────────────┼────────────────────────────
///   1  │ lock(a1) ✓                  │
///   2  │                             │ lock(a2) ✓
///   3  │ lock(a2) [BLOCKED]          │
///   4  │      ↓                      │ lock(a1) [BLOCKED]
///      │   DEADLOCK!                 │      ↓
///      │   Both threads waiting      │   DEADLOCK!
/// ```
///
/// This is allowed in Rust but will cause runtime deadlock.
/// Rust cannot prevent all deadlocks at compile time.
#[allow(dead_code)]
fn exchange_naive(
    account1: &Arc<Mutex<AccountWithSerializer>>,
    account2: &Arc<Mutex<AccountWithSerializer>>,
) {
    // DANGER: Can deadlock if another thread locks in opposite order!
    let mut acc1 = account1.lock().unwrap();
    thread::sleep(Duration::from_millis(10)); // Increase deadlock chance
    let mut acc2 = account2.lock().unwrap();

    let diff = acc1.balance() - acc2.balance();
    let _ = acc1.withdraw(diff);
    acc2.deposit(diff);
}

/// Exchange with deadlock prevention via ordered locking (Exercise 3.48)
///
/// # Deadlock Prevention Strategy:
///
/// Always acquire locks in ascending order of account ID.
///
/// ```text
/// Thread 1: exchange(a1, a2)    Thread 2: exchange(a2, a1)
///     ↓                              ↓
/// Order: [a1, a2] by ID         Order: [a1, a2] by ID (reordered!)
///     ↓                              ↓
/// lock(a1) ✓                     lock(a1) [BLOCKED]
/// lock(a2) ✓                          ↓
/// perform exchange               wait for Thread 1...
/// unlock(a2), unlock(a1)              ↓
///                                lock(a1) ✓
///                                lock(a2) ✓
///                                perform exchange
///
/// No deadlock possible!
/// ```
pub fn exchange_safe(
    account1: &Arc<Mutex<AccountWithSerializer>>,
    account2: &Arc<Mutex<AccountWithSerializer>>,
) {
    // Peek at IDs to determine lock order (requires careful design)
    // In production, use a higher-level abstraction or lock-free structures

    // Always lock lower ID first to prevent deadlock
    let id1 = account1.lock().unwrap().id();
    let id2 = account2.lock().unwrap().id();

    // Acquire locks in ascending ID order
    let (mut guard1, mut guard2) = if id1 < id2 {
        (account1.lock().unwrap(), account2.lock().unwrap())
    } else {
        (account2.lock().unwrap(), account1.lock().unwrap())
    };

    // Now determine which guard corresponds to which account
    // and perform the exchange correctly
    let (acc1, acc2) = if guard1.id() == id1 {
        (&mut *guard1, &mut *guard2)
    } else {
        (&mut *guard2, &mut *guard1)
    };

    // Equalize balances by transferring half the difference
    let diff = acc1.balance() - acc2.balance();
    let transfer = diff / 2;

    if transfer > 0 {
        let _ = acc1.withdraw(transfer);
        acc2.deposit(transfer);
    } else if transfer < 0 {
        let _ = acc2.withdraw(-transfer);
        acc1.deposit(-transfer);
    }
    // If diff is 0, balances are already equal
}

/// Demonstrate deadlock and prevention
pub fn demonstrate_deadlock_prevention() {
    println!("\n=== Deadlock Prevention (Exercise 3.48) ===");

    let account1 = AccountWithSerializer::new(100, 1);
    let account2 = AccountWithSerializer::new(50, 2);

    println!(
        "Initial: Account1=${}, Account2=${}",
        account1.lock().unwrap().balance(),
        account2.lock().unwrap().balance()
    );

    // Safe exchange using ordered locking
    let a1 = Arc::clone(&account1);
    let a2 = Arc::clone(&account2);
    let t1 = thread::spawn(move || {
        exchange_safe(&a1, &a2);
        println!("Thread 1: Exchanged a1 ↔ a2");
    });

    let a1 = Arc::clone(&account1);
    let a2 = Arc::clone(&account2);
    let t2 = thread::spawn(move || {
        exchange_safe(&a2, &a1); // Note: opposite order, but safe!
        println!("Thread 2: Exchanged a2 ↔ a1");
    });

    t1.join().unwrap();
    t2.join().unwrap();

    println!(
        "After exchange: Account1=${}, Account2={}",
        account1.lock().unwrap().balance(),
        account2.lock().unwrap().balance()
    );
    println!("No deadlock occurred due to ordered locking!");
}

// ==============================================================================
// Atomic Operations and Test-and-Set (SICP page 318-320)
// ==============================================================================

/// Simple mutex implementation using atomic test-and-set
///
/// # SICP Implementation (Scheme):
/// ```scheme
/// (define (make-mutex)
///   (let ((cell (list false)))
///     (define (the-mutex m)
///       (cond ((eq? m 'acquire)
///              (if (test-and-set! cell)
///                  (the-mutex 'acquire)))
///             ((eq? m 'release) (clear! cell))))
///     the-mutex))
///
/// (define (test-and-set! cell)
///   (if (car cell)
///       true
///       (begin (set-car! cell true)
///              false)))
/// ```
///
/// # Rust with Atomics:
///
/// Rust provides hardware-level atomic operations through `std::sync::atomic`.
/// These are truly atomic (no race conditions possible) and use CPU instructions
/// like CAS (Compare-And-Swap).
pub struct SimpleMutex {
    locked: AtomicBool,
}

impl SimpleMutex {
    pub fn new() -> Self {
        SimpleMutex {
            locked: AtomicBool::new(false),
        }
    }

    /// Acquire the mutex (busy-wait until available)
    ///
    /// Uses compare_exchange which is atomic at hardware level
    pub fn acquire(&self) {
        // Spin until we successfully set locked from false to true
        while self
            .locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            // Busy wait (in production, use std::hint::spin_loop())
            thread::yield_now();
        }
    }

    /// Release the mutex
    pub fn release(&self) {
        self.locked.store(false, Ordering::Release);
    }

    /// Test-and-set operation (atomic)
    ///
    /// Returns true if was already locked, false if we just acquired it
    pub fn test_and_set(&self) -> bool {
        self.locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
    }
}

impl Default for SimpleMutex {
    fn default() -> Self {
        Self::new()
    }
}

/// Demonstrate atomic test-and-set operations
pub fn demonstrate_atomic_operations() {
    println!("\n=== Atomic Test-and-Set (SICP page 318) ===");

    let mutex = Arc::new(SimpleMutex::new());
    let counter = Arc::new(AtomicI64::new(0));

    let mut handles = vec![];

    // Spawn 10 threads that increment counter 1000 times each
    for i in 0..10 {
        let m = Arc::clone(&mutex);
        let c = Arc::clone(&counter);

        let handle = thread::spawn(move || {
            for _ in 0..1000 {
                m.acquire();
                // Critical section
                let val = c.load(Ordering::Relaxed);
                c.store(val + 1, Ordering::Relaxed);
                m.release();
            }
            println!("Thread {} completed", i);
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let final_count = counter.load(Ordering::Relaxed);
    println!("\nFinal count: {} (expected: 10000)", final_count);
    assert_eq!(final_count, 10000, "Atomic operations ensure correctness!");
}

// ==============================================================================
// Tests (Exercises 3.38-3.47)
// ==============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_account_withdraw_deposit() {
        let account = Arc::new(Mutex::new(Account::new(100)));

        let acc = Arc::clone(&account);
        let t1 = thread::spawn(move || {
            acc.lock().unwrap().deposit(50);
        });

        let acc = Arc::clone(&account);
        let t2 = thread::spawn(move || {
            acc.lock().unwrap().withdraw(30).unwrap();
        });

        t1.join().unwrap();
        t2.join().unwrap();

        assert_eq!(account.lock().unwrap().balance(), 120);
    }

    #[test]
    fn test_concurrent_withdrawals_no_race() {
        // Exercise 3.38: Ensure proper serialization
        let account = Arc::new(Mutex::new(Account::new(100)));

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let acc = Arc::clone(&account);
                thread::spawn(move || {
                    let _ = acc.lock().unwrap().withdraw(10);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // All withdrawals should be serialized
        assert_eq!(account.lock().unwrap().balance(), 0);
    }

    #[test]
    fn test_thread_safe_account() {
        let account = ThreadSafeAccount::new(1000);

        let acc1 = account.clone_handle();
        let t1 = thread::spawn(move || {
            for _ in 0..100 {
                acc1.deposit(10);
            }
        });

        let acc2 = account.clone_handle();
        let t2 = thread::spawn(move || {
            for _ in 0..100 {
                let _ = acc2.withdraw(5);
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // 1000 + (100 * 10) - (100 * 5) = 1500
        assert_eq!(account.get_balance(), 1500);
    }

    #[test]
    fn test_serializer() {
        let serializer = Serializer::new(0);
        let s = Arc::new(serializer);

        let mut handles = vec![];

        for _ in 0..10 {
            let ser = Arc::clone(&s);
            let h = thread::spawn(move || {
                for _ in 0..1000 {
                    ser.execute(|val| *val += 1);
                }
            });
            handles.push(h);
        }

        for h in handles {
            h.join().unwrap();
        }

        let final_val = s.execute(|val| *val);
        assert_eq!(final_val, 10000);
    }

    #[test]
    fn test_exchange_safe() {
        let a1 = AccountWithSerializer::new(100, 1);
        let a2 = AccountWithSerializer::new(50, 2);

        exchange_safe(&a1, &a2);

        // After exchange, balances should be swapped
        assert_eq!(a1.lock().unwrap().balance(), 75);
        assert_eq!(a2.lock().unwrap().balance(), 75);
    }

    #[test]
    fn test_atomic_mutex() {
        let mutex = SimpleMutex::new();

        assert!(!mutex.test_and_set()); // First test-and-set acquires
        assert!(mutex.test_and_set()); // Second fails (already locked)

        mutex.release();
        assert!(!mutex.test_and_set()); // Can acquire again
    }

    /// Exercise 3.39: Serialized execution possibilities
    #[test]
    fn test_exercise_3_39() {
        // With partial serialization: x = ((s (lambda () (* x x))))
        // Possible outcomes: 100, 110, 121 (101 and 11 eliminated)

        let x = Arc::new(Mutex::new(10));

        let x1 = Arc::clone(&x);
        let t1 = thread::spawn(move || {
            let val = {
                let guard = x1.lock().unwrap();
                *guard * *guard // Serialized read
            };
            *x1.lock().unwrap() = val; // Unserialized write
        });

        let x2 = Arc::clone(&x);
        let t2 = thread::spawn(move || {
            let mut guard = x2.lock().unwrap();
            *guard += 1; // Fully serialized
        });

        t1.join().unwrap();
        t2.join().unwrap();

        let result = *x.lock().unwrap();
        println!("Exercise 3.39 result: {}", result);
        assert!(result == 100 || result == 101 || result == 110 || result == 121);
    }

    /// Exercise 3.40: Multiple concurrent operations
    #[test]
    fn test_exercise_3_40_serialized() {
        // (define x 10)
        // (parallel-execute
        //  (s (lambda () (set! x (* x x))))
        //  (s (lambda () (set! x (* x x x)))))
        //
        // With serialization: only 1,000,000 and 1,000,000,000,000 possible

        let x = Arc::new(Mutex::new(10i64));

        let x1 = Arc::clone(&x);
        let t1 = thread::spawn(move || {
            let mut guard = x1.lock().unwrap();
            *guard = *guard * *guard; // x * x
        });

        let x2 = Arc::clone(&x);
        let t2 = thread::spawn(move || {
            let mut guard = x2.lock().unwrap();
            *guard = *guard * *guard * *guard; // x * x * x
        });

        t1.join().unwrap();
        t2.join().unwrap();

        let result = *x.lock().unwrap();
        println!("Exercise 3.40 result: {}", result);
        // With serialization, only two outcomes possible
        assert!(result == 1_000_000 || result == 1_000_000_000_000);
    }
}
