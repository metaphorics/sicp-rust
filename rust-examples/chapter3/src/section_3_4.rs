//! SICP 3.4절: 동시성 (Concurrency) - 시간이 핵심이다 (Time Is of the Essence)
//!
//! 이 모듈은 SICP 3.4장에서 논의한 문제에 대해 러스트의 "두려움 없는 동시성"
//! (fearless concurrency) 접근을 보여준다. SICP가 Scheme의 serializer와 mutex를
//! 사용하는 반면, 러스트는 소유권 시스템과 타입 시스템을 통해 컴파일 타임 보장을 제공한다
//! (This module demonstrates Rust's "fearless concurrency" approach to the problems
//! discussed in SICP Chapter 3.4. While SICP uses Scheme's serializers and mutexes,
//! Rust provides compile-time guarantees through its ownership system and type system).
//!
//! # 동시성 아키텍처 (Concurrency Architecture)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                 동시성 모델 (Concurrency Model)             │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                             │
//! │  스레드 1 (Thread 1)     공유 상태 (Shared State)  스레드 2  │
//! │     │                          │                     │      │
//! │     ├─> Arc::clone() ──────────┼──────> Arc::clone()─┤      │
//! │     │                          │                     │      │
//! │     ├─> lock() ─────────> Mutex<T> <────── lock() ───┤      │
//! │     │                    (소유자 (owner))             │      │
//! │     ├─> MutexGuard<T>          │                     │      │
//! │     │ (배타적 접근 (exclusive access))                │      │
//! │     └─> 잠금 해제 (자동 드롭) (unlock (auto-drop))     │      │
//! │                                │                     │      │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # 메모리 안전 보장 (Memory Safety Guarantees)
//!
//! ```text
//! 스택 (Thread 1)               힙 (Heap)             스택 (Thread 2)
//! ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
//! │ Arc<Mutex<T>>├────────>│ Arc 카운터   │<────────┤Arc<Mutex<T>> │
//! │   (clone)    │         │ (Arc Counter)│         │  (clone)     │
//! └──────────────┘         │   count: 2   │         └──────────────┘
//!                          │      ↓       │
//!                          │   Mutex<T>   │
//!                          │      ↓       │
//!                          │ 데이터 (Data): T │
//!                          └──────────────┘
//!
//! 락 획득은 배타적 대여를 만든다 (Lock acquisition creates exclusive borrow):
//! MutexGuard<'lock, T> → &'lock mut T (잠금 중에는 별칭 불가 (can't alias while locked))
//! ```

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// ==============================================================================
// 3.4.1절: 동시 시스템에서 시간의 본질 (The Nature of Time in Concurrent Systems)
// ==============================================================================

/// Mutex로 내부 가변성을 가진 간단한 은행 계좌
/// (A simple bank account with interior mutability using Mutex).
///
/// # 러스트 vs 스킴 비교 (Rust vs Scheme Comparison)
///
/// Scheme (SICP):
/// ```scheme
/// (define (make-account balance)
///   (define (withdraw amount)
///     (if (>= balance amount)
///         (begin (set! balance (- balance amount))
///                balance)
///         "잔액 부족 (Insufficient funds)"))
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
/// 러스트는 Mutex<T>를 통해 컴파일 타임 안전성을 제공한다
/// (Rust provides compile-time safety through Mutex<T>):
/// - 데이터 레이스 불가 (타입 시스템이 강제) (No data races possible (enforced by type system))
/// - 데이터 접근 전에 락 필요 (Lock must be acquired to access data)
/// - RAII(Drop 트레이트)로 자동 해제 (Automatic unlock via RAII (Drop trait))
/// - Send/Sync 트레이트가 스레드 안전을 보장 (Send/Sync traits ensure thread safety)
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

    /// 계좌에서 금액을 출금한다 (Mutex로 배타적 접근 필요)
    /// (Withdraw amount from account (requires exclusive access via Mutex))
    pub fn withdraw(&mut self, amount: i64) -> Result<i64, String> {
        if self.balance >= amount {
            self.balance -= amount;
            Ok(self.balance)
        } else {
            Err("잔액 부족 (Insufficient funds)".to_string())
        }
    }

    /// 계좌에 금액을 입금한다 (Mutex로 배타적 접근 필요)
    /// (Deposit amount into account (requires exclusive access via Mutex))
    pub fn deposit(&mut self, amount: i64) -> i64 {
        self.balance += amount;
        self.balance
    }

    /// 현재 잔액 조회 (읽기 전용이지만 일관성을 위해 락 필요)
    /// (Get current balance (read-only, but still needs lock for consistency))
    pub fn balance(&self) -> i64 {
        self.balance
    }
}

/// 그림 3.29 (Figure 3.29)의 레이스 컨디션을 보여준다
/// (Demonstrates the race condition from Figure 3.29 (SICP)).
///
/// 적절한 동기화가 없으면 두 동시 출금이 최종 잔액을 틀리게 만든다.
/// 러스트는 컴파일 타임에 이를 막으며, 스레드 간 가변 상태 공유에는 Mutex가 필수다
/// (Without proper synchronization, two concurrent withdrawals can lead to
/// incorrect final balance. Rust prevents this at compile time - you MUST
/// use Mutex to share mutable state between threads).
///
/// # 동시성 다이어그램 (Concurrency Diagram): 레이스 컨디션 (Race Condition)
///
/// ```text
/// 시간 (Time) │ 스레드 1 (Thread 1) (Peter -$10) │ 메모리 (Memory) │ 스레드 2 (Thread 2) (Paul -$25)
/// ───────────┼──────────────────────────────────┼─────────────────┼────────────────────────────────
///   1        │ 잔액 읽기 (read balance): $100    │  $100           │
///   2        │ 계산 (compute): 100 - 10 = 90     │  $100           │ 잔액 읽기 (read balance): $100
///   3        │                                   │  $100           │ 계산 (compute): 100 - 25 = 75
///   4        │ 잔액 쓰기 (write balance): $90    │  $90            │
///   5        │                                   │  $75            │ 잔액 쓰기 (write balance): $75  ← 오류! (WRONG!)
///            │                                   │                 │ (Peter의 갱신을 덮어씀 (overwrites Peter's update))
///
/// 기대값 (Expected): $100 - $10 - $25 = $65
/// 실제값 (Actual):   $75 (갱신 손실 (lost update)!)
/// ```
///
/// 러스트는 Mutex를 강제해 이를 막는다 (Rust prevents this by requiring Mutex):
/// ```rust,ignore
/// let account = Arc::new(Mutex::new(Account::new(100)));
/// // Must lock to access: account.lock().unwrap().withdraw(10)
/// ```
pub fn demonstrate_race_condition_prevention() {
    // 러스트 타입 시스템이 레이스 컨디션을 방지함을 보여준다
    // (This demonstrates that Rust's type system prevents the race condition)
    println!("\n=== 레이스 컨디션 방지 (Race Condition Prevention) ===");
    println!(
        "Scheme에서는 공유 상태의 동시 접근에 수동 직렬화가 필요하다."
    );
    println!("Rust에서는 타입 시스템이 컴파일 타임에 이를 강제한다!\n");

    let account = Arc::new(Mutex::new(Account::new(100)));

    let acc1 = Arc::clone(&account);
    let acc2 = Arc::clone(&account);

    let peter = thread::spawn(move || {
        // 락은 자동 획득/해제된다 (RAII)
        let mut acc = acc1.lock().unwrap();
        println!("Peter: 락 획득, $10 출금 (Acquired lock, withdrawing $10)");
        thread::sleep(Duration::from_millis(10)); // 계산 시뮬레이션 (Simulate computation)
        let result = acc.withdraw(10);
        println!("Peter: $10 출금, 결과: {:?}", result);
        // `acc`(MutexGuard)가 스코프를 벗어나면 락이 자동 해제됨
        // (Lock automatically released when `acc` (MutexGuard) goes out of scope)
    });

    let paul = thread::spawn(move || {
        thread::sleep(Duration::from_millis(5));
        println!("Paul: 락 획득 시도... (Attempting to acquire lock...)");
        let mut acc = acc2.lock().unwrap(); // Blocks until lock available
        println!("Paul: 락 획득, $25 출금 (Acquired lock, withdrawing $25)");
        let result = acc.withdraw(25);
        println!("Paul: $25 출금, 결과: {:?}", result);
    });

    peter.join().unwrap();
    paul.join().unwrap();

    let final_balance = account.lock().unwrap().balance();
    println!(
        "\n최종 잔액: ${} (정상값: $65) (Final balance: ${} (correct: $65))",
        final_balance, final_balance
    );
    assert_eq!(final_balance, 65, "Mutex로 레이스 컨디션 방지! (Race condition prevented by Mutex!)");
}

// ==============================================================================
// 3.4.2절: 동시성 제어 메커니즘 (Mechanisms for Controlling Concurrency)
// ==============================================================================

/// SICP의 make-serializer를 모방한 serializer 래퍼
/// (Serializer wrapper that mimics SICP's make-serializer)
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
/// # 러스트 대응 (Rust Equivalent):
///
/// Mutex<T> 타입은 락 가드를 통해 이미 직렬화를 제공한다
/// (The Mutex<T> type already provides serialization through lock guards).
/// 이 래퍼는 개념을 명시적으로 보여준다
/// (This wrapper demonstrates the concept explicitly).
pub struct Serializer<T> {
    mutex: Arc<Mutex<T>>,
}

impl<T> Serializer<T> {
    pub fn new(value: T) -> Self {
        Serializer {
            mutex: Arc::new(Mutex::new(value)),
        }
    }

    /// 보호된 값에 대해 직렬화된 접근으로 함수를 실행
    /// (Execute a function with serialized access to the protected value)
    pub fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        let mut guard = self.mutex.lock().unwrap();
        f(&mut *guard)
        // guard가 드롭되면 락이 자동 해제된다 (Lock automatically released when guard drops)
    }

    /// 스레드 간 공유를 위한 Arc 복제 (Clone the Arc for sharing between threads)
    pub fn clone_arc(&self) -> Arc<Mutex<T>> {
        Arc::clone(&self.mutex)
    }
}

/// 직렬화된 병렬 실행을 시연한다 (연습문제 3.38-3.40)
/// (Demonstrates parallel execution with serialization (Exercise 3.38-3.40))
///
/// # 연습문제 3.38의 러스트 대응 (Exercise 3.38 equivalent in Rust)
///
/// 공유 상태에 대한 세 가지 동시 연산:
/// - Peter: balance += 10
/// - Paul: balance -= 20
/// - Mary: balance -= balance / 2
///
/// 직렬화 없음: 다양한 결과 가능 (레이스 컨디션)
/// 직렬화 있음: 유효한 순차 순서는 6개뿐
pub fn parallel_execute_example() {
    println!("\n=== 병렬 실행 (연습문제 3.38) (Parallel Execute) ===");

    // 적절한 동기화가 없으면 (Scheme에서는 위험, Rust에서는 타입 시스템이 방지)
    // (Without proper synchronization (unsafe in Scheme, prevented by Rust type system))
    println!("Rust는 동기화되지 않은 접근을 컴파일 타임에 막는다!");
    println!("가변 상태 공유에는 Arc<Mutex<T>>가 필요하다.\n");

    let balance = Arc::new(Mutex::new(100));

    let b1 = Arc::clone(&balance);
    let b2 = Arc::clone(&balance);
    let b3 = Arc::clone(&balance);

    // Peter: +10
    let peter = thread::spawn(move || {
        let mut b = b1.lock().unwrap();
        *b += 10;
        println!("Peter: balance += 10, now ${} (현재 잔액)", *b);
    });

    // Paul: -20
    let paul = thread::spawn(move || {
        let mut b = b2.lock().unwrap();
        *b -= 20;
        println!("Paul: balance -= 20, now ${} (현재 잔액)", *b);
    });

    // Mary: -= balance/2
    let mary = thread::spawn(move || {
        let mut b = b3.lock().unwrap();
        let half = *b / 2;
        *b -= half;
        println!("Mary: balance -= balance/2, now ${} (현재 잔액)", *b);
    });

    peter.join().unwrap();
    paul.join().unwrap();
    mary.join().unwrap();

    let final_balance = balance.lock().unwrap();
    println!("\n최종 잔액: ${} (Final balance)", *final_balance);
    println!("가능한 순차 결과: $35, $40, $45, $50 (Possible sequential results)");
    // 참고: 락 순서 때문에 이 단순한 경우 결과가 결정적이다
    // (Note: Due to lock ordering, result is deterministic in this simple case)
}

/// 자동 직렬화를 갖춘 스레드 안전 은행 계좌
/// (Thread-safe bank account with automatic serialization)
///
/// SICP의 보호된 make-account에 대응 (312쪽)
/// (Maps to SICP's protected make-account (page 312))
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
            Err("잔액 부족 (Insufficient funds)".to_string())
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

    /// 스레드 간 공유를 위한 복제 (Arc 참조 카운팅)
    /// (Clone for sharing across threads (Arc reference counting))
    pub fn clone_handle(&self) -> Self {
        ThreadSafeAccount {
            balance: Arc::clone(&self.balance),
        }
    }
}

// ==============================================================================
// 데드락 예시 (Deadlock examples)
// (3.4.2절 - 다중 공유 자원의 복잡성 (Section 3.4.2 - Complexity of Multiple Shared Resources))
// ==============================================================================

/// 교환 연산을 위한 serializer를 노출한 계좌
/// (Account with exposed serializer for exchange operations)
///
/// SICP의 make-account-and-serializer에 대응 (316쪽)
/// (Maps to SICP's make-account-and-serializer (page 316))
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
            Err("잔액 부족 (Insufficient funds)".to_string())
        }
    }

    pub fn deposit(&mut self, amount: i64) -> i64 {
        self.balance += amount;
        self.balance
    }
}

/// 두 계좌 사이의 잔액을 교환 (순진한 구현 - 데드락 가능)
/// (Exchange balances between two accounts (NAIVE - can deadlock!))
///
/// # 데드락 시나리오 (Deadlock Scenario) (그림 3.30 일부 (Figure 3.30 region)):
///
/// ```text
/// 시간 (Time) │ 스레드 1 (Thread 1): exchange(a1, a2) │ 스레드 2 (Thread 2): exchange(a2, a1)
/// ───────────┼───────────────────────────────────────┼──────────────────────────────────────
///   1        │ lock(a1) ✓                            │
///   2        │                                       │ lock(a2) ✓
///   3        │ lock(a2) [차단됨 (BLOCKED)]            │
///   4        │      ↓                                │ lock(a1) [차단됨 (BLOCKED)]
///            │   데드락! (DEADLOCK!)                  │      ↓
///            │   두 스레드 모두 대기                  │   데드락! (DEADLOCK!)
/// ```
///
/// 러스트에서도 허용되지만 런타임 데드락을 일으킨다
/// (This is allowed in Rust but will cause runtime deadlock).
/// 러스트는 모든 데드락을 컴파일 타임에 막을 수는 없다
/// (Rust cannot prevent all deadlocks at compile time).
#[allow(dead_code)]
fn exchange_naive(
    account1: &Arc<Mutex<AccountWithSerializer>>,
    account2: &Arc<Mutex<AccountWithSerializer>>,
) {
    // 위험: 다른 스레드가 반대 순서로 락을 잡으면 데드락 가능!
    // (DANGER: Can deadlock if another thread locks in opposite order!)
    let mut acc1 = account1.lock().unwrap();
    thread::sleep(Duration::from_millis(10)); // 데드락 가능성 증가 (Increase deadlock chance)
    let mut acc2 = account2.lock().unwrap();

    let diff = acc1.balance() - acc2.balance();
    let _ = acc1.withdraw(diff);
    acc2.deposit(diff);
}

/// 정렬된 락 획득으로 데드락을 방지하는 교환 (연습문제 3.48)
/// (Exchange with deadlock prevention via ordered locking (Exercise 3.48))
///
/// # 데드락 방지 전략 (Deadlock Prevention Strategy):
///
/// 계좌 ID 오름차순으로 항상 락을 획득한다
/// (Always acquire locks in ascending order of account ID).
///
/// ```text
/// 스레드 1 (Thread 1): exchange(a1, a2)    스레드 2 (Thread 2): exchange(a2, a1)
///     ↓                                         ↓
/// 순서: [a1, a2] ID 기준 (Order: [a1, a2] by ID)
///                                        순서: [a1, a2] ID 기준 (재정렬됨 (reordered!))
///     ↓                                         ↓
/// lock(a1) ✓                              lock(a1) [차단됨 (BLOCKED)]
/// lock(a2) ✓                                   ↓
/// 교환 수행 (perform exchange)                  스레드 1을 대기 (wait for Thread 1)
/// unlock(a2), unlock(a1)                        ↓
///                                         lock(a1) ✓
///                                         lock(a2) ✓
///                                         교환 수행 (perform exchange)
///
/// 데드락 불가능! (No deadlock possible!)
/// ```
pub fn exchange_safe(
    account1: &Arc<Mutex<AccountWithSerializer>>,
    account2: &Arc<Mutex<AccountWithSerializer>>,
) {
    // 락 순서를 정하기 위해 ID를 미리 확인한다 (설계 주의 필요)
    // (Peek at IDs to determine lock order (requires careful design))
    // 프로덕션에서는 상위 수준의 추상화나 락-프리 구조를 사용하라
    // (In production, use a higher-level abstraction or lock-free structures)

    // 데드락 방지를 위해 작은 ID부터 락 획득
    // (Always lock lower ID first to prevent deadlock)
    let id1 = account1.lock().unwrap().id();
    let id2 = account2.lock().unwrap().id();

    // ID 오름차순으로 락 획득 (Acquire locks in ascending ID order)
    let (mut guard1, mut guard2) = if id1 < id2 {
        (account1.lock().unwrap(), account2.lock().unwrap())
    } else {
        (account2.lock().unwrap(), account1.lock().unwrap())
    };

    // 어떤 guard가 어떤 계좌인지 결정한 뒤 올바르게 교환 수행
    // (Now determine which guard corresponds to which account
    // and perform the exchange correctly)
    let (acc1, acc2) = if guard1.id() == id1 {
        (&mut *guard1, &mut *guard2)
    } else {
        (&mut *guard2, &mut *guard1)
    };

    // 차이의 절반을 이체해 잔액을 평준화
    // (Equalize balances by transferring half the difference)
    let diff = acc1.balance() - acc2.balance();
    let transfer = diff / 2;

    if transfer > 0 {
        let _ = acc1.withdraw(transfer);
        acc2.deposit(transfer);
    } else if transfer < 0 {
        let _ = acc2.withdraw(-transfer);
        acc1.deposit(-transfer);
    }
    // diff가 0이면 잔액이 이미 동일함 (If diff is 0, balances are already equal)
}

/// 데드락과 방지를 시연 (Demonstrate deadlock and prevention)
pub fn demonstrate_deadlock_prevention() {
    println!("\n=== 데드락 방지 (연습문제 3.48) (Deadlock Prevention) ===");

    let account1 = AccountWithSerializer::new(100, 1);
    let account2 = AccountWithSerializer::new(50, 2);

    println!(
        "초기: Account1=${}, Account2=${} (Initial)",
        account1.lock().unwrap().balance(),
        account2.lock().unwrap().balance()
    );

    // 정렬된 락 획득으로 안전한 교환 (Safe exchange using ordered locking)
    let a1 = Arc::clone(&account1);
    let a2 = Arc::clone(&account2);
    let t1 = thread::spawn(move || {
        exchange_safe(&a1, &a2);
        println!("스레드 1: a1 ↔ a2 교환 (Thread 1: Exchanged a1 ↔ a2)");
    });

    let a1 = Arc::clone(&account1);
    let a2 = Arc::clone(&account2);
    let t2 = thread::spawn(move || {
        exchange_safe(&a2, &a1); // 주의: 반대 순서지만 안전함 (Note: opposite order, but safe!)
        println!("스레드 2: a2 ↔ a1 교환 (Thread 2: Exchanged a2 ↔ a1)");
    });

    t1.join().unwrap();
    t2.join().unwrap();

    println!(
        "교환 후: Account1=${}, Account2={} (After exchange)",
        account1.lock().unwrap().balance(),
        account2.lock().unwrap().balance()
    );
    println!("정렬된 락 획득으로 데드락이 발생하지 않았다!");
}

// ==============================================================================
// 원자적 연산과 테스트-앤드-셋 (Atomic operations and test-and-set) (SICP 318-320쪽)
// ==============================================================================

/// 원자적 테스트-앤드-셋을 사용하는 간단한 뮤텍스 구현
/// (Simple mutex implementation using atomic test-and-set)
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
/// # 원자 연산을 사용하는 러스트 (Rust with Atomics):
///
/// 러스트는 `std::sync::atomic`으로 하드웨어 수준의 원자 연산을 제공한다
/// (Rust provides hardware-level atomic operations through `std::sync::atomic`).
/// 이는 실제 원자적이며(레이스 컨디션 불가), CAS(Compare-And-Swap) 같은 CPU
/// 명령을 사용한다
/// (These are truly atomic (no race conditions possible) and use CPU instructions
/// like CAS (Compare-And-Swap)).
pub struct SimpleMutex {
    locked: AtomicBool,
}

impl SimpleMutex {
    pub fn new() -> Self {
        SimpleMutex {
            locked: AtomicBool::new(false),
        }
    }

    /// 뮤텍스를 획득한다 (사용 가능할 때까지 바쁜 대기)
    /// (Acquire the mutex (busy-wait until available))
    ///
    /// compare_exchange는 하드웨어 수준에서 원자적이다
    /// (Uses compare_exchange which is atomic at hardware level)
    pub fn acquire(&self) {
        // locked를 false에서 true로 성공적으로 바꿀 때까지 스핀
        // (Spin until we successfully set locked from false to true)
        while self
            .locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            // 바쁜 대기 (프로덕션에서는 std::hint::spin_loop() 사용)
            // (Busy wait (in production, use std::hint::spin_loop()))
            thread::yield_now();
        }
    }

    /// 뮤텍스 해제 (Release the mutex)
    pub fn release(&self) {
        self.locked.store(false, Ordering::Release);
    }

    /// 테스트-앤드-셋 연산 (원자적)
    /// (Test-and-set operation (atomic))
    ///
    /// 이미 잠겨 있었다면 true, 이번에 획득했다면 false
    /// (Returns true if was already locked, false if we just acquired it)
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

/// 원자적 테스트-앤드-셋 연산을 시연
/// (Demonstrate atomic test-and-set operations)
pub fn demonstrate_atomic_operations() {
    println!("\n=== 원자적 테스트-앤드-셋 (SICP 318쪽) (Atomic Test-and-Set) ===");

    let mutex = Arc::new(SimpleMutex::new());
    let counter = Arc::new(AtomicI64::new(0));

    let mut handles = vec![];

    // 카운터를 각각 1000회 증가시키는 스레드 10개 생성
    // (Spawn 10 threads that increment counter 1000 times each)
    for i in 0..10 {
        let m = Arc::clone(&mutex);
        let c = Arc::clone(&counter);

        let handle = thread::spawn(move || {
            for _ in 0..1000 {
                m.acquire();
                // 임계 구역 (Critical section)
                let val = c.load(Ordering::Relaxed);
                c.store(val + 1, Ordering::Relaxed);
                m.release();
            }
            println!("스레드 {} 완료 (Thread {} completed)", i, i);
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let final_count = counter.load(Ordering::Relaxed);
    println!(
        "\n최종 카운트: {} (예상: 10000) (Final count: {} (expected: 10000))",
        final_count, final_count
    );
    assert_eq!(
        final_count, 10000,
        "원자 연산이 정확성을 보장한다! (Atomic operations ensure correctness!)"
    );
}

// ==============================================================================
// 테스트 (Tests) (연습문제 3.38-3.47 (Exercises 3.38-3.47))
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
        // 연습문제 3.38: 올바른 직렬화 보장
        // (Exercise 3.38: Ensure proper serialization)
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

        // 모든 출금은 직렬화되어야 함 (All withdrawals should be serialized)
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

        // 교환 후 잔액이 서로 바뀌어야 한다 (After exchange, balances should be swapped)
        assert_eq!(a1.lock().unwrap().balance(), 75);
        assert_eq!(a2.lock().unwrap().balance(), 75);
    }

    #[test]
    fn test_atomic_mutex() {
        let mutex = SimpleMutex::new();

        assert!(!mutex.test_and_set()); // 첫 테스트-앤드-셋은 획득 (First test-and-set acquires)
        assert!(mutex.test_and_set()); // 두 번째는 실패 (이미 잠김) (Second fails (already locked))

        mutex.release();
        assert!(!mutex.test_and_set()); // 다시 획득 가능 (Can acquire again)
    }

    /// 연습문제 3.39: 직렬화된 실행 가능성
    /// (Exercise 3.39: Serialized execution possibilities)
    #[test]
    fn test_exercise_3_39() {
        // 부분 직렬화: x = ((s (lambda () (* x x))))
        // 가능한 결과: 100, 110, 121 (101과 11은 제거됨)

        let x = Arc::new(Mutex::new(10));

        let x1 = Arc::clone(&x);
        let t1 = thread::spawn(move || {
            let val = {
                let guard = x1.lock().unwrap();
                *guard * *guard // 직렬화된 읽기 (Serialized read)
            };
            *x1.lock().unwrap() = val; // 비직렬화 쓰기 (Unserialized write)
        });

        let x2 = Arc::clone(&x);
        let t2 = thread::spawn(move || {
            let mut guard = x2.lock().unwrap();
            *guard += 1; // 완전 직렬화 (Fully serialized)
        });

        t1.join().unwrap();
        t2.join().unwrap();

        let result = *x.lock().unwrap();
        println!("연습문제 3.39 결과: {} (Exercise 3.39 result)", result);
        assert!(result == 100 || result == 101 || result == 110 || result == 121);
    }

    /// 연습문제 3.40: 다중 동시 연산
    /// (Exercise 3.40: Multiple concurrent operations)
    #[test]
    fn test_exercise_3_40_serialized() {
        // (define x 10)
        // (parallel-execute
        //  (s (lambda () (set! x (* x x))))
        //  (s (lambda () (set! x (* x x x)))))
        //
        // 직렬화가 있으면 가능한 값은 1,000,000과 1,000,000,000,000뿐

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
        println!("연습문제 3.40 결과: {} (Exercise 3.40 result)", result);
        // 직렬화가 있으면 가능한 결과는 두 가지뿐 (With serialization, only two outcomes possible)
        assert!(result == 1_000_000 || result == 1_000_000_000_000);
    }
}
