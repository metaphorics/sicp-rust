//! 3.2절: 평가의 환경 모델 (Section 3.2: The Environment Model of Evaluation)
//!
//! 이 절은 SICP의 환경 모델을 관용적인 Rust 패턴으로 번역한다
//! (This section translates SICP's environment model to Rust using idiomatic patterns).
//! `Rc<RefCell<T>>` 대신 다음을 사용한다:
//! (Instead of `Rc<RefCell<T>>`, we use:)
//!
//! - **영속적 환경 (Persistent environments)**: 구조적 공유를 갖는 불변 `Environment<V>`
//! - **함수형 갱신 (Functional updates)**: 연산이 새 상태를 반환
//! - **`Cell<T>`**: 내부 가변성이 필요할 때 단순 `Copy` 타입에 사용
//!   (For simple `Copy` types when interior mutability is needed)
//!
//! 핵심 개념 (Key concepts):
//! - 프레임의 영속 체인으로서의 환경 (Environments as persistent chains of frames)
//! - (코드, 환경) 쌍으로서의 프로시저 - 환경은 공유가 아닌 소유
//!   (Procedures as (code, environment) pairs - environment is owned, not shared)
//! - 변수 조회는 환경 체인을 따라간다(불변) (Variable lookup walks the environment chain (immutably))
//! - 클로저는 정의 환경을 캡처한다(구조적 공유로 O(1) 복제)
//!   (Closures capture their defining environment (by clone, which is O(1) due to sharing))

use sicp_common::Environment;
use std::cell::Cell;
use std::fmt;

// =============================================================================
// 파트 1 (Part 1): 런타임 표현을 위한 값 타입 (Value type for runtime representation)
// =============================================================================

/// 환경 모델을 보여주기 위한 런타임 값 타입 (Runtime value type for demonstrating the environment model).
/// 프로시저는 캡처한 환경을 소유한다(`Rc<RefCell<>>` 없음)
/// (Procedures own their captured environment (no `Rc<RefCell<>>`)).
#[derive(Clone)]
pub enum Value {
    Number(f64),
    String(String),
    /// (code, environment)로 표현된 프로시저 (A procedure represented as (code, environment)).
    /// 환경은 Rc<RefCell<>>로 공유되지 않고 소유된다
    /// (The environment is owned, not shared via Rc<RefCell<>>).
    Procedure {
        params: Vec<String>,
        body: String,
        env: Environment<Value>,
    },
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Number(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Procedure { params, .. } => {
                write!(f, "#<procedure ({})>", params.join(" "))
            }
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

// =============================================================================
// 파트 2 (Part 2): 환경 예시 (Environment examples)
// =============================================================================

/// 영속적 환경 모델을 시연한다 (Demonstrates the persistent environment model).
///
/// # 예시 (Example)
///
/// ```
/// use sicp_chapter3::section_3_2::env_demo;
///
/// let (result_x, result_y, result_z) = env_demo();
/// assert_eq!(result_x, Some(7));  // 내부에서 섀도잉 (Shadowed in inner)
/// assert_eq!(result_y, Some(5));  // 외부에서 옴 (From outer)
/// assert_eq!(result_z, Some(6));  // 내부에서 옴 (From inner)
/// ```
pub fn env_demo() -> (Option<i64>, Option<i64>, Option<i64>) {
    // 외부 환경 생성(전역 프레임과 유사) (Create outer environment (like global frame))
    let outer = Environment::<i64>::new()
        .define("x".to_string(), 3)
        .define("y".to_string(), 5);

    // 외부를 확장한 내부 환경 생성(x를 섀도잉) (Create inner environment extending outer (shadows x))
    let inner = outer.extend([("z".to_string(), 6), ("x".to_string(), 7)]);

    // 조회 (Lookups)
    let x = inner.lookup("x").copied(); // 7 (섀도잉됨 (shadowed))
    let y = inner.lookup("y").copied(); // 5 (외부에서 옴 (from outer))
    let z = inner.lookup("z").copied(); // 6 (내부에서 옴 (from inner))

    (x, y, z)
}

// =============================================================================
// 파트 3 (Part 3): 러스트 클로저 의미론 (Rust closure semantics, native implementation)
// =============================================================================

/// 러스트 클로저가 환경 모델을 자연스럽게 구현하는 방식을 보여준다
/// (Demonstrates how Rust closures naturally implement the environment model).
/// 클로저는 환경을 캡처하며, SICP의 (code, environment) 쌍과 유사하다
/// (The closure captures its environment, similar to SICP's (code, environment) pairs).
pub fn square(x: i64) -> i64 {
    x * x
}

/// 그림 3.4 (Figure 3.4): 전역 프레임의 세 프로시저 (Three procedures in the global frame).
pub fn sum_of_squares(x: i64, y: i64) -> i64 {
    square(x) + square(y)
}

pub fn f(a: i64) -> i64 {
    sum_of_squares(a + 1, a * 2)
}

// =============================================================================
// 파트 4 (Part 4): 지역 상태 - 함수형 접근 (Local state - functional approach)
// =============================================================================

/// 함수형 상태를 사용하는 출금 함수 (새 상태를 반환)
/// (A withdraw function using functional state (returns new state)).
///
/// 내부 상태를 변경하는 대신 (new_balance, result)를 반환한다
/// (Instead of mutating internal state, returns (new_balance, result)).
/// 이는 관용적인 러스트/함수형 접근이다
/// (This is the idiomatic Rust/functional approach).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wallet {
    balance: f64,
}

impl Wallet {
    pub fn new(initial_balance: f64) -> Self {
        Self {
            balance: initial_balance,
        }
    }

    pub fn balance(&self) -> f64 {
        self.balance
    }

    /// 출금은 (new_wallet, result)를 반환한다 (Withdraw returns (new_wallet, result)).
    pub fn withdraw(&self, amount: f64) -> (Self, Result<f64, &'static str>) {
        if self.balance >= amount {
            let new_balance = self.balance - amount;
            (
                Self {
                    balance: new_balance,
                },
                Ok(new_balance),
            )
        } else {
            (*self, Err("잔액 부족 (Insufficient funds)"))
        }
    }

    /// 입금은 (new_wallet, new_balance)를 반환한다 (Deposit returns (new_wallet, new_balance)).
    #[must_use]
    pub fn deposit(&self, amount: f64) -> (Self, f64) {
        let new_balance = self.balance + amount;
        (
            Self {
                balance: new_balance,
            },
            new_balance,
        )
    }
}

/// 내부 가변성을 위해 Cell<f64>를 사용해 출금 클로저를 만든다
/// (Creates a withdraw closure using Cell<f64> for interior mutability).
///
/// `Copy` 타입에는 RefCell보다 단순하다
/// (This is simpler than RefCell for `Copy` types).
pub fn make_withdraw(initial_balance: f64) -> impl FnMut(f64) -> Result<f64, &'static str> {
    let balance = Cell::new(initial_balance);

    move |amount: f64| {
        let current = balance.get();
        if current >= amount {
            let new_balance = current - amount;
            balance.set(new_balance);
            Ok(new_balance)
        } else {
            Err("잔액 부족 (Insufficient funds)")
        }
    }
}

/// Cell<f64>를 사용하는 구조체 기반 출금 처리기
/// (Struct-based withdraw processor using Cell<f64>).
pub struct WithdrawProcessor {
    balance: Cell<f64>,
}

impl WithdrawProcessor {
    pub fn new(initial_balance: f64) -> Self {
        Self {
            balance: Cell::new(initial_balance),
        }
    }

    pub fn withdraw(&self, amount: f64) -> Result<f64, &'static str> {
        let current = self.balance.get();
        if current >= amount {
            let new_balance = current - amount;
            self.balance.set(new_balance);
            Ok(new_balance)
        } else {
            Err("잔액 부족 (Insufficient funds)")
        }
    }

    pub fn balance(&self) -> f64 {
        self.balance.get()
    }
}

// =============================================================================
// 파트 5 (Part 5): 연습문제 3.9 (Exercise 3.9) - 팩토리얼 환경 (Factorial environments)
// =============================================================================

/// 재귀 팩토리얼 (SICP 연습문제 3.9 첫 번째 버전에 해당)
/// (Recursive factorial (corresponds to SICP Exercise 3.9 first version)).
pub fn factorial_recursive(n: i64) -> i64 {
    if n == 1 {
        1
    } else {
        n * factorial_recursive(n - 1)
    }
}

/// 내부 헬퍼를 쓰는 반복 팩토리얼 (연습문제 3.9 두 번째 버전)
/// (Iterative factorial using an internal helper (Exercise 3.9 second version)).
pub fn factorial_iterative(n: i64) -> i64 {
    fn fact_iter(product: i64, counter: i64, max_count: i64) -> i64 {
        if counter > max_count {
            product
        } else {
            fact_iter(counter * product, counter + 1, max_count)
        }
    }
    fact_iter(1, 1, n)
}

// =============================================================================
// 파트 6 (Part 6): 연습문제 3.11 (Exercise 3.11) - 메시지 전달 계좌 (Message-passing account)
// =============================================================================

/// 함수형 상태 패턴을 사용하는 은행 계좌
/// (A bank account using functional state pattern).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Account {
    balance: f64,
}

impl Account {
    pub fn new(initial_balance: f64) -> Self {
        Self {
            balance: initial_balance,
        }
    }

    pub fn balance(&self) -> f64 {
        self.balance
    }

    pub fn withdraw(&self, amount: f64) -> (Self, Result<f64, &'static str>) {
        if self.balance >= amount {
            let new_balance = self.balance - amount;
            (
                Self {
                    balance: new_balance,
                },
                Ok(new_balance),
            )
        } else {
            (*self, Err("잔액 부족 (Insufficient funds)"))
        }
    }

    #[must_use]
    pub fn deposit(&self, amount: f64) -> (Self, f64) {
        let new_balance = self.balance + amount;
        (
            Self {
                balance: new_balance,
            },
            new_balance,
        )
    }
}

/// 내부 가변성에 Cell을 사용하는 메시지 전달 계좌
/// (Message-passing account using Cell for interior mutability).
pub struct MutableAccount {
    balance: Cell<f64>,
}

impl MutableAccount {
    pub fn new(initial_balance: f64) -> Self {
        Self {
            balance: Cell::new(initial_balance),
        }
    }

    pub fn balance(&self) -> f64 {
        self.balance.get()
    }

    pub fn withdraw(&self, amount: f64) -> Result<f64, &'static str> {
        let current = self.balance.get();
        if current >= amount {
            let new_balance = current - amount;
            self.balance.set(new_balance);
            Ok(new_balance)
        } else {
            Err("잔액 부족 (Insufficient funds)")
        }
    }

    pub fn deposit(&self, amount: f64) -> f64 {
        let new_balance = self.balance.get() + amount;
        self.balance.set(new_balance);
        new_balance
    }
}

// =============================================================================
// 테스트 (Tests)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_demo() {
        let (x, y, z) = env_demo();
        assert_eq!(x, Some(7)); // 섀도잉됨 (Shadowed)
        assert_eq!(y, Some(5)); // 외부에서 옴 (From outer)
        assert_eq!(z, Some(6)); // 내부에서 옴 (From inner)
    }

    #[test]
    fn test_persistent_environment() {
        let env1 = Environment::<i64>::new().define("x".to_string(), 10);

        // 함수형 갱신은 새 환경을 만든다 (Functional update creates new environment)
        let env2 = env1.define("x".to_string(), 20);

        // 원본은 변하지 않음 (영속적) (Original unchanged (persistent))
        assert_eq!(env1.lookup("x"), Some(&10));
        assert_eq!(env2.lookup("x"), Some(&20));
    }

    #[test]
    fn test_environment_extend() {
        let outer = Environment::<i64>::new()
            .define("a".to_string(), 1)
            .define("b".to_string(), 2);

        let inner = outer.extend([
            ("c".to_string(), 3),
            ("a".to_string(), 100), // a를 섀도잉함 (Shadows a)
        ]);

        assert_eq!(inner.lookup("a"), Some(&100)); // 섀도잉됨 (Shadowed)
        assert_eq!(inner.lookup("b"), Some(&2)); // 부모에서 옴 (From parent)
        assert_eq!(inner.lookup("c"), Some(&3)); // 지역 (Local)
        assert_eq!(outer.lookup("a"), Some(&1)); // 변하지 않음 (Unchanged)
    }

    #[test]
    fn test_square_procedure() {
        assert_eq!(square(5), 25);
    }

    #[test]
    fn test_nested_procedure_calls() {
        // (f 5) = (sum-of-squares 6 10) = 36 + 100 = 136
        assert_eq!(f(5), 136);
    }

    #[test]
    fn test_wallet_functional() {
        let w = Wallet::new(100.0);

        let (w, result) = w.withdraw(50.0);
        assert_eq!(result, Ok(50.0));

        let (w, result) = w.withdraw(30.0);
        assert_eq!(result, Ok(20.0));

        let (_, result) = w.withdraw(25.0);
        assert_eq!(result, Err("잔액 부족 (Insufficient funds)"));
    }

    #[test]
    fn test_make_withdraw() {
        let mut w1 = make_withdraw(100.0);

        assert_eq!(w1(50.0), Ok(50.0));
        assert_eq!(w1(30.0), Ok(20.0));
        assert_eq!(w1(25.0), Err("잔액 부족 (Insufficient funds)"));
        assert_eq!(w1(10.0), Ok(10.0));
    }

    #[test]
    fn test_independent_withdraw_objects() {
        let mut w1 = make_withdraw(100.0);
        let mut w2 = make_withdraw(100.0);

        assert_eq!(w1(50.0), Ok(50.0));
        assert_eq!(w2(30.0), Ok(70.0));
        assert_eq!(w1(10.0), Ok(40.0));
        assert_eq!(w2(10.0), Ok(60.0));
    }

    #[test]
    fn test_withdraw_processor() {
        let w = WithdrawProcessor::new(100.0);

        assert_eq!(w.withdraw(50.0), Ok(50.0));
        assert_eq!(w.withdraw(30.0), Ok(20.0));
        assert_eq!(w.withdraw(25.0), Err("Insufficient funds"));
        assert_eq!(w.balance(), 20.0);
    }

    #[test]
    fn test_factorial_recursive() {
        assert_eq!(factorial_recursive(1), 1);
        assert_eq!(factorial_recursive(6), 720);
    }

    #[test]
    fn test_factorial_iterative() {
        assert_eq!(factorial_iterative(1), 1);
        assert_eq!(factorial_iterative(6), 720);
    }

    #[test]
    fn test_account_functional() {
        let acc = Account::new(50.0);

        let (acc, balance) = acc.deposit(40.0);
        assert_eq!(balance, 90.0);

        let (acc, result) = acc.withdraw(60.0);
        assert_eq!(result, Ok(30.0));

        let (_, result) = acc.withdraw(50.0);
        assert_eq!(result, Err("Insufficient funds"));
    }

    #[test]
    fn test_mutable_account() {
        let acc = MutableAccount::new(50.0);

        assert_eq!(acc.deposit(40.0), 90.0);
        assert_eq!(acc.withdraw(60.0), Ok(30.0));
        assert_eq!(acc.balance(), 30.0);
        assert_eq!(acc.withdraw(50.0), Err("Insufficient funds"));
    }

    #[test]
    fn test_independent_accounts() {
        let acc1 = MutableAccount::new(50.0);
        let acc2 = MutableAccount::new(100.0);

        acc1.deposit(40.0);
        acc2.withdraw(30.0).ok();

        assert_eq!(acc1.balance(), 90.0);
        assert_eq!(acc2.balance(), 70.0);
    }

    #[test]
    fn test_closure_captures_environment() {
        let x = 10;
        let add_x = |y| x + y;
        assert_eq!(add_x(5), 15);
    }

    #[test]
    fn test_value_procedure() {
        let env = Environment::<Value>::new().define("x".to_string(), Value::Number(42.0));

        let proc = Value::Procedure {
            params: vec!["y".to_string()],
            body: "(+ x y)".to_string(),
            env: env.clone(),
        };

        // The procedure owns its environment
        if let Value::Procedure {
            env: captured_env, ..
        } = proc
        {
            assert!(captured_env.lookup("x").is_some());
        }
    }
}
