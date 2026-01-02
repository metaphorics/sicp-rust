//! Section 3.2: The Environment Model of Evaluation
//!
//! This section translates SICP's environment model to Rust, demonstrating how
//! environments, frames, and closures work. We show both an explicit environment
//! implementation (for pedagogy) and native Rust closure semantics.
//!
//! Key concepts:
//! - Environments as chains of frames (HashMap + parent link)
//! - Procedures as (code, environment) pairs
//! - Variable lookup walks the environment chain
//! - Closures capture their defining environment
//! - Local state through closure capture

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

// =============================================================================
// PART 1: EXPLICIT ENVIRONMENT MODEL (Educational)
// =============================================================================

/// An environment is a sequence of frames, where each frame is a table of
/// bindings. Each frame has a pointer to its enclosing (parent) environment.
///
/// This corresponds to Figure 3.1 in SICP, where environments are shown as
/// linked frames pointing to their parent environments.
#[derive(Clone)]
pub struct Environment<T: Clone> {
    /// The current frame: a table of variable bindings
    frame: HashMap<String, T>,
    /// The enclosing environment (parent), if any
    parent: Option<Rc<RefCell<Environment<T>>>>,
}

impl<T: Clone> Environment<T> {
    /// Create a new empty environment with no parent (global environment)
    pub fn new() -> Self {
        Environment {
            frame: HashMap::new(),
            parent: None,
        }
    }

    /// Create a new environment extending a parent environment.
    /// This is used when applying procedures - each procedure call creates
    /// a new frame extending the procedure's environment.
    ///
    /// Corresponds to SICP's "extending an environment" when calling a procedure.
    pub fn extend(parent: Rc<RefCell<Environment<T>>>) -> Self {
        Environment {
            frame: HashMap::new(),
            parent: Some(parent),
        }
    }

    /// Define a variable in the current frame.
    /// If the variable already exists in the current frame, it is rebound.
    ///
    /// Corresponds to SICP's `define` operation.
    pub fn define(&mut self, var: String, val: T) {
        self.frame.insert(var, val);
    }

    /// Look up a variable's value in the environment.
    /// Searches the current frame first, then walks up the parent chain.
    ///
    /// Corresponds to SICP's variable lookup operation.
    pub fn lookup(&self, var: &str) -> Option<T> {
        if let Some(val) = self.frame.get(var) {
            Some(val.clone())
        } else if let Some(ref parent) = self.parent {
            parent.borrow().lookup(var)
        } else {
            None
        }
    }

    /// Set a variable's value in the environment where it's defined.
    /// Walks up the environment chain to find the first binding.
    ///
    /// Corresponds to SICP's `set!` operation.
    pub fn set(&mut self, var: &str, val: T) -> Result<(), String> {
        if self.frame.contains_key(var) {
            self.frame.insert(var.to_string(), val);
            Ok(())
        } else if let Some(ref parent) = self.parent {
            parent.borrow_mut().set(var, val)
        } else {
            Err(format!("Unbound variable: {}", var))
        }
    }

    /// Get the number of bindings in the current frame
    pub fn frame_size(&self) -> usize {
        self.frame.len()
    }

    /// Check if a variable is bound in the current frame (not including parents)
    pub fn bound_in_frame(&self, var: &str) -> bool {
        self.frame.contains_key(var)
    }
}

impl<T: Clone + fmt::Display> fmt::Debug for Environment<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Frame[")?;
        for (i, (k, v)) in self.frame.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", k, v)?;
        }
        write!(f, "]")?;
        if self.parent.is_some() {
            write!(f, " -> parent")?;
        }
        Ok(())
    }
}

impl<T: Clone> Default for Environment<T> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// PART 2: VALUE TYPE FOR RUNTIME REPRESENTATION
// =============================================================================

/// Runtime value type for demonstrating the environment model.
/// This is a simplified version of what a Scheme interpreter would use.
#[derive(Clone)]
pub enum Value {
    Number(f64),
    String(String),
    /// A procedure represented as (code, environment)
    /// In a real implementation, this would contain compiled code or AST
    Procedure {
        params: Vec<String>,
        body: String, // Simplified: in reality would be an AST
        env: Rc<RefCell<Environment<Value>>>,
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

// =============================================================================
// PART 3: RUST CLOSURE SEMANTICS (Native Implementation)
// =============================================================================

/// Demonstrates how Rust closures naturally implement the environment model.
/// Each closure captures its environment, similar to SICP's (code, environment) pairs.
///
/// This is Figure 3.2 in Rust: defining square in the global environment.
/// The closure captures no environment (it's a pure function).
pub fn square(x: i64) -> i64 {
    x * x
}

/// Figure 3.4: Three procedures in the global frame.
/// Each is a simple function defined at the top level.
pub fn sum_of_squares(x: i64, y: i64) -> i64 {
    square(x) + square(y)
}

pub fn f(a: i64) -> i64 {
    sum_of_squares(a + 1, a * 2)
}

// =============================================================================
// PART 4: FRAMES AS REPOSITORY OF LOCAL STATE
// =============================================================================

/// Figure 3.6-3.9: The `make_withdraw` procedure demonstrates local state.
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
///
/// In Rust, we use closures with captured mutable state via RefCell.
/// The closure captures `balance` in its environment (corresponds to E1 in Figure 3.7).
pub fn make_withdraw(initial_balance: f64) -> impl FnMut(f64) -> Result<f64, String> {
    // The balance is stored in the closure's environment (like E1 in SICP)
    let balance = Rc::new(RefCell::new(initial_balance));

    move |amount: f64| {
        let mut bal = balance.borrow_mut();
        if *bal >= amount {
            *bal -= amount;
            Ok(*bal)
        } else {
            Err("Insufficient funds".to_string())
        }
    }
}

/// Alternative implementation showing explicit environment capture.
/// This makes the environment structure more visible for pedagogical purposes.
pub struct WithdrawProcessor {
    // This field is the "environment E1" from Figure 3.7
    // It persists across multiple calls to the procedure
    balance: RefCell<f64>,
}

impl WithdrawProcessor {
    pub fn new(initial_balance: f64) -> Self {
        WithdrawProcessor {
            balance: RefCell::new(initial_balance),
        }
    }

    pub fn withdraw(&self, amount: f64) -> Result<f64, String> {
        let mut balance = self.balance.borrow_mut();
        if *balance >= amount {
            *balance -= amount;
            Ok(*balance)
        } else {
            Err("Insufficient funds".to_string())
        }
    }
}

// =============================================================================
// PART 5: EXERCISE 3.9 - FACTORIAL ENVIRONMENTS
// =============================================================================

/// Recursive factorial (corresponds to SICP Exercise 3.9 first version)
///
/// Each recursive call creates a new environment frame with a binding for n.
/// The frames form a chain: E1(n=6) -> E2(n=5) -> E3(n=4) -> ... -> E6(n=1)
pub fn factorial_recursive(n: i64) -> i64 {
    if n == 1 {
        1
    } else {
        n * factorial_recursive(n - 1)
    }
}

/// Iterative factorial using an internal helper (Exercise 3.9 second version)
///
/// The helper function fact_iter is tail-recursive. In the environment model,
/// each call creates a new frame, but old frames can be garbage collected
/// because there are no pending operations.
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
// PART 6: EXERCISE 3.10 - LET AS SYNTACTIC SUGAR
// =============================================================================

/// Alternative make_withdraw using explicit let binding.
/// In Scheme:
/// ```scheme
/// (define (make-withdraw initial-amount)
///   (let ((balance initial-amount))
///     (lambda (amount) ...)))
/// ```
///
/// In Rust, `let` bindings are naturally part of the lexical scope.
/// This version is functionally identical to make_withdraw but shows
/// the environment structure more explicitly.
pub fn make_withdraw_with_let(initial_amount: f64) -> impl FnMut(f64) -> Result<f64, String> {
    // The let binding creates a new scope (like an inner lambda in Scheme)
    let balance = Rc::new(RefCell::new(initial_amount));

    move |amount: f64| {
        let mut bal = balance.borrow_mut();
        if *bal >= amount {
            *bal -= amount;
            Ok(*bal)
        } else {
            Err("Insufficient funds".to_string())
        }
    }
}

// =============================================================================
// PART 7: EXERCISE 3.11 - INTERNAL DEFINITIONS
// =============================================================================

/// Message-passing account with internal definitions (Exercise 3.11)
///
/// In Scheme:
/// ```scheme
/// (define (make-account balance)
///   (define (withdraw amount) ...)
///   (define (deposit amount) ...)
///   (define (dispatch m) ...)
///   dispatch)
/// ```
///
/// In Rust, we use an enum to represent messages and closures for the methods.
pub enum AccountMessage {
    Withdraw,
    Deposit,
}

pub fn make_account(
    initial_balance: f64,
) -> impl Fn(AccountMessage) -> Box<dyn Fn(f64) -> Result<f64, String>> {
    // The balance is in the environment created by make_account (like E1)
    let balance = Rc::new(RefCell::new(initial_balance));

    move |message: AccountMessage| {
        let balance = Rc::clone(&balance);
        match message {
            AccountMessage::Withdraw => Box::new(move |amount: f64| {
                let mut bal = balance.borrow_mut();
                if *bal >= amount {
                    *bal -= amount;
                    Ok(*bal)
                } else {
                    Err("Insufficient funds".to_string())
                }
            }) as Box<dyn Fn(f64) -> Result<f64, String>>,
            AccountMessage::Deposit => Box::new(move |amount: f64| {
                let mut bal = balance.borrow_mut();
                *bal += amount;
                Ok(*bal)
            }) as Box<dyn Fn(f64) -> Result<f64, String>>,
        }
    }
}

/// Alternative account implementation using a more Rust-idiomatic approach
pub struct Account {
    balance: RefCell<f64>,
}

impl Account {
    pub fn new(initial_balance: f64) -> Self {
        Account {
            balance: RefCell::new(initial_balance),
        }
    }

    pub fn withdraw(&self, amount: f64) -> Result<f64, String> {
        let mut balance = self.balance.borrow_mut();
        if *balance >= amount {
            *balance -= amount;
            Ok(*balance)
        } else {
            Err("Insufficient funds".to_string())
        }
    }

    pub fn deposit(&self, amount: f64) -> f64 {
        let mut balance = self.balance.borrow_mut();
        *balance += amount;
        *balance
    }

    pub fn balance(&self) -> f64 {
        *self.balance.borrow()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test Environment Operations
    // -------------------------------------------------------------------------

    #[test]
    fn test_environment_define_and_lookup() {
        // Figure 3.1: Simple environment structure
        let mut global = Environment::<i64>::new();
        global.define("x".to_string(), 3);
        global.define("y".to_string(), 5);

        assert_eq!(global.lookup("x"), Some(3));
        assert_eq!(global.lookup("y"), Some(5));
        assert_eq!(global.lookup("z"), None);
    }

    #[test]
    fn test_environment_shadowing() {
        // Figure 3.1: Frame II shadows x from frame I
        let global = Rc::new(RefCell::new(Environment::<i64>::new()));
        global.borrow_mut().define("x".to_string(), 3);
        global.borrow_mut().define("y".to_string(), 5);

        let mut frame2 = Environment::extend(Rc::clone(&global));
        frame2.define("z".to_string(), 6);
        frame2.define("x".to_string(), 7); // Shadows x from global

        assert_eq!(frame2.lookup("x"), Some(7)); // Finds shadowed value
        assert_eq!(frame2.lookup("y"), Some(5)); // Finds in parent
        assert_eq!(frame2.lookup("z"), Some(6)); // Finds in current frame
        assert_eq!(global.borrow().lookup("x"), Some(3)); // Original unchanged
    }

    #[test]
    fn test_environment_set() {
        // Test set! operation
        let global = Rc::new(RefCell::new(Environment::<i64>::new()));
        global.borrow_mut().define("x".to_string(), 10);

        let mut frame = Environment::extend(Rc::clone(&global));
        frame.define("y".to_string(), 20);

        // Set x in parent environment
        assert!(frame.set("x", 100).is_ok());
        assert_eq!(global.borrow().lookup("x"), Some(100));

        // Set y in current frame
        assert!(frame.set("y", 200).is_ok());
        assert_eq!(frame.lookup("y"), Some(200));

        // Try to set undefined variable
        assert!(frame.set("z", 300).is_err());
    }

    // -------------------------------------------------------------------------
    // Test Procedure Application
    // -------------------------------------------------------------------------

    #[test]
    fn test_square_procedure() {
        // Figure 3.2-3.3: Evaluating (square 5)
        assert_eq!(square(5), 25);
    }

    #[test]
    fn test_nested_procedure_calls() {
        // Figure 3.4-3.5: Evaluating (f 5)
        // (f 5) = (sum-of-squares (+ 5 1) (* 5 2))
        //       = (sum-of-squares 6 10)
        //       = (+ (square 6) (square 10))
        //       = (+ 36 100)
        //       = 136
        assert_eq!(f(5), 136);
    }

    // -------------------------------------------------------------------------
    // Test Local State
    // -------------------------------------------------------------------------

    #[test]
    fn test_make_withdraw() {
        // Figure 3.6-3.9: make-withdraw with local state
        let mut w1 = make_withdraw(100.0);

        // First withdrawal
        assert_eq!(w1(50.0), Ok(50.0));

        // Second withdrawal
        assert_eq!(w1(30.0), Ok(20.0));

        // Insufficient funds
        assert!(w1(25.0).is_err());

        // Balance should still be 20
        assert_eq!(w1(10.0), Ok(10.0));
    }

    #[test]
    fn test_independent_withdraw_objects() {
        // Figure 3.10: Two independent withdraw objects
        let mut w1 = make_withdraw(100.0);
        let mut w2 = make_withdraw(100.0);

        assert_eq!(w1(50.0), Ok(50.0));
        assert_eq!(w2(30.0), Ok(70.0));

        // w1 and w2 maintain independent state
        assert_eq!(w1(10.0), Ok(40.0));
        assert_eq!(w2(10.0), Ok(60.0));
    }

    #[test]
    fn test_withdraw_processor() {
        let w1 = WithdrawProcessor::new(100.0);

        assert_eq!(w1.withdraw(50.0), Ok(50.0));
        assert_eq!(w1.withdraw(30.0), Ok(20.0));
        assert!(w1.withdraw(25.0).is_err());
    }

    // -------------------------------------------------------------------------
    // Test Exercise 3.9: Factorial
    // -------------------------------------------------------------------------

    #[test]
    fn test_factorial_recursive() {
        assert_eq!(factorial_recursive(1), 1);
        assert_eq!(factorial_recursive(6), 720);

        // Each recursive call creates a new environment frame
        // E1(n=6) -> E2(n=5) -> E3(n=4) -> E4(n=3) -> E5(n=2) -> E6(n=1)
    }

    #[test]
    fn test_factorial_iterative() {
        assert_eq!(factorial_iterative(1), 1);
        assert_eq!(factorial_iterative(6), 720);

        // Tail-recursive, so old frames can be reclaimed
    }

    // -------------------------------------------------------------------------
    // Test Exercise 3.10: Let as Syntactic Sugar
    // -------------------------------------------------------------------------

    #[test]
    fn test_make_withdraw_with_let() {
        let mut w1 = make_withdraw_with_let(100.0);

        assert_eq!(w1(50.0), Ok(50.0));
        assert_eq!(w1(30.0), Ok(20.0));

        // Behavior identical to make_withdraw
    }

    // -------------------------------------------------------------------------
    // Test Exercise 3.11: Internal Definitions
    // -------------------------------------------------------------------------

    #[test]
    fn test_make_account_message_passing() {
        let acc = make_account(50.0);

        // Get deposit procedure
        let deposit = acc(AccountMessage::Deposit);
        assert_eq!(deposit(40.0), Ok(90.0));

        // Get withdraw procedure
        let withdraw = acc(AccountMessage::Withdraw);
        assert_eq!(withdraw(60.0), Ok(30.0));
    }

    #[test]
    fn test_account_struct() {
        let acc = Account::new(50.0);

        assert_eq!(acc.deposit(40.0), 90.0);
        assert_eq!(acc.withdraw(60.0), Ok(30.0));
        assert_eq!(acc.balance(), 30.0);

        assert!(acc.withdraw(50.0).is_err());
    }

    #[test]
    fn test_independent_accounts() {
        let acc1 = Account::new(50.0);
        let acc2 = Account::new(100.0);

        acc1.deposit(40.0);
        acc2.withdraw(30.0).ok();

        // Independent state
        assert_eq!(acc1.balance(), 90.0);
        assert_eq!(acc2.balance(), 70.0);
    }

    // -------------------------------------------------------------------------
    // Test Environment Model Internals
    // -------------------------------------------------------------------------

    #[test]
    fn test_environment_chain_depth() {
        let global = Rc::new(RefCell::new(Environment::<i64>::new()));
        global.borrow_mut().define("a".to_string(), 1);

        let e1 = Rc::new(RefCell::new(Environment::extend(Rc::clone(&global))));
        e1.borrow_mut().define("b".to_string(), 2);

        let e2 = Rc::new(RefCell::new(Environment::extend(Rc::clone(&e1))));
        e2.borrow_mut().define("c".to_string(), 3);

        // Lookup walks the chain
        assert_eq!(e2.borrow().lookup("c"), Some(3)); // Current frame
        assert_eq!(e2.borrow().lookup("b"), Some(2)); // Parent frame
        assert_eq!(e2.borrow().lookup("a"), Some(1)); // Grandparent frame
        assert_eq!(e2.borrow().lookup("d"), None); // Not found
    }

    #[test]
    fn test_closure_captures_environment() {
        // Demonstrate that Rust closures capture their environment
        let x = 10;
        let add_x = |y| x + y;

        assert_eq!(add_x(5), 15);

        // x is captured in the closure's environment
        // This is similar to SICP's (lambda (y) (+ x y)) capturing x
    }

    #[test]
    fn test_multiple_closures_share_state() {
        // Two closures sharing the same captured state
        let balance = Rc::new(RefCell::new(100.0));

        let balance_clone = Rc::clone(&balance);
        let withdraw = move |amount: f64| {
            let mut bal = balance_clone.borrow_mut();
            *bal -= amount;
            *bal
        };

        let balance_clone = Rc::clone(&balance);
        let deposit = move |amount: f64| {
            let mut bal = balance_clone.borrow_mut();
            *bal += amount;
            *bal
        };

        assert_eq!(deposit(50.0), 150.0);
        assert_eq!(withdraw(30.0), 120.0);
    }
}
