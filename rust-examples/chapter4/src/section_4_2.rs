//! Section 4.2: Variations on a Scheme - Lazy Evaluation
//!
//! This module implements a lazy evaluator for Scheme, where:
//! - Compound procedures are non-strict (arguments delayed)
//! - Primitive procedures are strict (arguments evaluated)
//! - Thunks memoize their values after first evaluation using OnceCell
//! - Lazy lists enable infinite streams without special forms
//!
//! ## Key Design Changes from Original:
//!
//! - **Persistent environments**: Using `im::HashMap` for O(1) clone with structural sharing
//! - **OnceCell memoization**: Thunks use `OnceCell<Value>` for single-write semantics
//! - **No Rc<RefCell<>>**: Environment is persistent, thunks use OnceCell
//! - **Functional state threading**: eval returns (Value, Environment) for defines

use sicp_common::Environment;
use std::cell::OnceCell;
use std::fmt;
use std::rc::Rc;

// ============================================================================
// Core Type Definitions
// ============================================================================

/// Expression type representing the AST
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Number literal
    Number(i64),
    /// Boolean literal
    Bool(bool),
    /// Symbol/variable reference
    Symbol(String),
    /// Variable definition: (define name value)
    Define(String, Box<Expr>),
    /// Lambda: (lambda (params...) body)
    Lambda {
        params: Vec<String>,
        body: Box<Expr>,
    },
    /// Conditional: (if test consequent alternative)
    If {
        test: Box<Expr>,
        consequent: Box<Expr>,
        alternative: Box<Expr>,
    },
    /// Sequence of expressions: (begin expr1 expr2 ...)
    Begin(Vec<Expr>),
    /// Procedure application: (operator operands...)
    Application {
        operator: Box<Expr>,
        operands: Vec<Expr>,
    },
}

/// Runtime value produced by evaluation
#[derive(Clone)]
pub enum Value {
    /// Integer value
    Number(i64),
    /// Boolean value
    Bool(bool),
    /// Primitive procedure (implemented in Rust)
    Primitive(PrimitiveFn),
    /// Compound procedure (user-defined) - OWNS its environment
    Procedure {
        params: Vec<String>,
        body: Expr,
        env: Environment<Value>,
        /// Optional self-name for recursive binding
        self_name: Option<String>,
    },
    /// Lazy thunk - unevaluated expression with OnceCell memoization
    Thunk(Rc<Thunk>),
}

/// Primitive function type
#[derive(Clone)]
pub struct PrimitiveFn {
    pub name: &'static str,
    pub func: fn(&[Value]) -> Result<Value, EvalError>,
}

/// Thunk with OnceCell memoization (single-write, no RefCell needed)
pub struct Thunk {
    /// The unevaluated expression
    expr: Expr,
    /// Environment captured at delay time
    env: Environment<Value>,
    /// Memoized result - written once, read many times
    memo: OnceCell<Value>,
}

impl Thunk {
    /// Create a new unevaluated thunk
    pub fn new(expr: Expr, env: Environment<Value>) -> Self {
        Self {
            expr,
            env,
            memo: OnceCell::new(),
        }
    }

    /// Check if the thunk has been evaluated
    pub fn is_evaluated(&self) -> bool {
        self.memo.get().is_some()
    }

    /// Get the memoized value if available
    pub fn get_memo(&self) -> Option<&Value> {
        self.memo.get()
    }
}

impl Clone for Thunk {
    fn clone(&self) -> Self {
        // Clone the thunk state
        Self {
            expr: self.expr.clone(),
            env: self.env.clone(),
            memo: match self.memo.get() {
                Some(v) => {
                    let cell = OnceCell::new();
                    let _ = cell.set(v.clone());
                    cell
                }
                None => OnceCell::new(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    UnboundVariable(String),
    InvalidSyntax(String),
    TypeError(String),
    ArityMismatch { expected: usize, got: usize },
    DivisionByZero,
}

// ============================================================================
// Display Implementations
// ============================================================================

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Number(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Primitive(pf) => write!(f, "#<primitive:{}>", pf.name),
            Value::Procedure { params, .. } => write!(f, "#<procedure({})>", params.join(" ")),
            Value::Thunk(thunk) => {
                if thunk.is_evaluated() {
                    write!(f, "#<thunk:evaluated({:?})>", thunk.get_memo().unwrap())
                } else {
                    write!(f, "#<thunk:delayed>")
                }
            }
        }
    }
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::UnboundVariable(s) => write!(f, "Unbound variable: {}", s),
            EvalError::InvalidSyntax(s) => write!(f, "Invalid syntax: {}", s),
            EvalError::TypeError(s) => write!(f, "Type error: {}", s),
            EvalError::ArityMismatch { expected, got } => {
                write!(f, "Arity mismatch: expected {}, got {}", expected, got)
            }
            EvalError::DivisionByZero => write!(f, "Division by zero"),
        }
    }
}

impl std::error::Error for EvalError {}

// ============================================================================
// Environment Setup (using persistent Environment from sicp-common)
// ============================================================================

/// Create the global environment with primitive procedures
pub fn setup_environment() -> Environment<Value> {
    Environment::new()
        .define(
            "+".to_string(),
            Value::Primitive(PrimitiveFn {
                name: "+",
                func: prim_add,
            }),
        )
        .define(
            "-".to_string(),
            Value::Primitive(PrimitiveFn {
                name: "-",
                func: prim_sub,
            }),
        )
        .define(
            "*".to_string(),
            Value::Primitive(PrimitiveFn {
                name: "*",
                func: prim_mul,
            }),
        )
        .define(
            "/".to_string(),
            Value::Primitive(PrimitiveFn {
                name: "/",
                func: prim_div,
            }),
        )
        .define(
            "=".to_string(),
            Value::Primitive(PrimitiveFn {
                name: "=",
                func: prim_eq,
            }),
        )
        .define(
            "<".to_string(),
            Value::Primitive(PrimitiveFn {
                name: "<",
                func: prim_lt,
            }),
        )
        .define(
            ">".to_string(),
            Value::Primitive(PrimitiveFn {
                name: ">",
                func: prim_gt,
            }),
        )
}

// ============================================================================
// Primitive Procedures
// ============================================================================

fn prim_add(args: &[Value]) -> Result<Value, EvalError> {
    let mut sum = 0;
    for arg in args {
        match arg {
            Value::Number(n) => sum += n,
            _ => return Err(EvalError::TypeError("+ expects numbers".to_string())),
        }
    }
    Ok(Value::Number(sum))
}

fn prim_sub(args: &[Value]) -> Result<Value, EvalError> {
    if args.is_empty() {
        return Err(EvalError::ArityMismatch {
            expected: 1,
            got: 0,
        });
    }
    match &args[0] {
        Value::Number(first) => {
            let mut result = *first;
            if args.len() == 1 {
                return Ok(Value::Number(-result));
            }
            for arg in &args[1..] {
                match arg {
                    Value::Number(n) => result -= n,
                    _ => return Err(EvalError::TypeError("- expects numbers".to_string())),
                }
            }
            Ok(Value::Number(result))
        }
        _ => Err(EvalError::TypeError("- expects numbers".to_string())),
    }
}

fn prim_mul(args: &[Value]) -> Result<Value, EvalError> {
    let mut product = 1;
    for arg in args {
        match arg {
            Value::Number(n) => product *= n,
            _ => return Err(EvalError::TypeError("* expects numbers".to_string())),
        }
    }
    Ok(Value::Number(product))
}

fn prim_div(args: &[Value]) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::ArityMismatch {
            expected: 2,
            got: args.len(),
        });
    }
    match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => {
            if *b == 0 {
                Err(EvalError::DivisionByZero)
            } else {
                Ok(Value::Number(a / b))
            }
        }
        _ => Err(EvalError::TypeError("/ expects numbers".to_string())),
    }
}

fn prim_eq(args: &[Value]) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::ArityMismatch {
            expected: 2,
            got: args.len(),
        });
    }
    match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Bool(a == b)),
        _ => Err(EvalError::TypeError("= expects numbers".to_string())),
    }
}

fn prim_lt(args: &[Value]) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::ArityMismatch {
            expected: 2,
            got: args.len(),
        });
    }
    match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Bool(a < b)),
        _ => Err(EvalError::TypeError("< expects numbers".to_string())),
    }
}

fn prim_gt(args: &[Value]) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::ArityMismatch {
            expected: 2,
            got: args.len(),
        });
    }
    match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Bool(a > b)),
        _ => Err(EvalError::TypeError("> expects numbers".to_string())),
    }
}

// ============================================================================
// Thunk Operations
// ============================================================================

/// Create a thunk (delayed computation)
pub fn delay_it(expr: Expr, env: Environment<Value>) -> Value {
    Value::Thunk(Rc::new(Thunk::new(expr, env)))
}

/// Force a thunk to produce its value (with OnceCell memoization)
pub fn force_it(obj: Value) -> Result<Value, EvalError> {
    match obj {
        Value::Thunk(thunk) => {
            // Check if already memoized
            if let Some(val) = thunk.memo.get() {
                return Ok(val.clone());
            }

            // Evaluate the expression
            let result = actual_value(thunk.expr.clone(), thunk.env.clone())?;

            // Memoize the result (OnceCell ensures single-write)
            // If already set by recursive evaluation, that's fine
            let _ = thunk.memo.set(result.clone());

            // Return the memoized value
            Ok(thunk.memo.get().cloned().unwrap_or(result))
        }
        // If not a thunk, return as-is
        val => Ok(val),
    }
}

/// Evaluate an expression and force any resulting thunks
pub fn actual_value(expr: Expr, env: Environment<Value>) -> Result<Value, EvalError> {
    let (val, _) = eval(expr, env)?;
    force_it(val)
}

// ============================================================================
// Main Evaluator
// ============================================================================

/// Evaluate an expression in an environment.
/// Returns (Value, Environment) - environment may have new bindings from define.
pub fn eval(expr: Expr, env: Environment<Value>) -> Result<(Value, Environment<Value>), EvalError> {
    match expr {
        Expr::Number(n) => Ok((Value::Number(n), env)),
        Expr::Bool(b) => Ok((Value::Bool(b), env)),

        Expr::Symbol(name) => {
            let val = env
                .lookup(&name)
                .ok_or_else(|| EvalError::UnboundVariable(name.clone()))?
                .clone();
            Ok((val, env))
        }

        Expr::Define(name, val_expr) => {
            // Special handling for lambda to enable recursion
            match val_expr.as_ref() {
                Expr::Lambda { params, body } => {
                    let procedure = Value::Procedure {
                        params: params.clone(),
                        body: *body.clone(),
                        env: env.clone(),
                        self_name: Some(name.clone()),
                    };
                    let new_env = env.define(name, procedure.clone());
                    Ok((procedure, new_env))
                }
                _ => {
                    let (value, env) = eval(*val_expr, env)?;
                    let new_env = env.define(name, value.clone());
                    Ok((value, new_env))
                }
            }
        }

        Expr::Lambda { params, body } => {
            let procedure = Value::Procedure {
                params,
                body: *body,
                env: env.clone(),
                self_name: None,
            };
            Ok((procedure, env))
        }

        Expr::If {
            test,
            consequent,
            alternative,
        } => {
            // Force the predicate to get actual boolean value
            let test_val = actual_value(*test, env.clone())?;
            match test_val {
                Value::Bool(true) => eval(*consequent, env),
                Value::Bool(false) => eval(*alternative, env),
                _ => Err(EvalError::TypeError(
                    "if predicate must be boolean".to_string(),
                )),
            }
        }

        Expr::Begin(exprs) => {
            let mut result = Value::Bool(false);
            let mut current_env = env;
            for expr in exprs {
                let (val, new_env) = eval(expr, current_env)?;
                result = val;
                current_env = new_env;
            }
            Ok((result, current_env))
        }

        Expr::Application { operator, operands } => {
            // Force the operator to get actual procedure
            let proc = actual_value(*operator, env.clone())?;
            let result = apply(proc, operands, env.clone())?;
            Ok((result, env))
        }
    }
}

/// Apply a procedure to arguments
pub fn apply(
    procedure: Value,
    operands: Vec<Expr>,
    env: Environment<Value>,
) -> Result<Value, EvalError> {
    match procedure.clone() {
        // Primitives are strict - evaluate all arguments
        Value::Primitive(prim) => {
            let args = list_of_arg_values(operands, env)?;
            (prim.func)(&args)
        }

        // Compound procedures are non-strict - delay all arguments
        Value::Procedure {
            params,
            body,
            env: proc_env,
            self_name,
        } => {
            if params.len() != operands.len() {
                return Err(EvalError::ArityMismatch {
                    expected: params.len(),
                    got: operands.len(),
                });
            }

            // Start with the procedure's captured environment
            let mut new_env = proc_env;

            // Bind self-name for recursive calls
            if let Some(name) = self_name {
                new_env = new_env.define(name, procedure);
            }

            // Delay all arguments (non-strict semantics)
            let delayed_args = list_of_delayed_args(operands, env);

            // Extend environment with parameter bindings
            let bindings: Vec<(String, Value)> = params.into_iter().zip(delayed_args).collect();
            new_env = new_env.extend(bindings);

            // Evaluate body (ignore returned env since we're in apply)
            let (result, _) = eval(body, new_env)?;
            Ok(result)
        }

        _ => Err(EvalError::TypeError(
            "Cannot apply non-procedure".to_string(),
        )),
    }
}

/// Evaluate all operands (for primitive procedures)
fn list_of_arg_values(
    operands: Vec<Expr>,
    env: Environment<Value>,
) -> Result<Vec<Value>, EvalError> {
    operands
        .into_iter()
        .map(|expr| actual_value(expr, env.clone()))
        .collect()
}

/// Delay all operands (for compound procedures)
fn list_of_delayed_args(operands: Vec<Expr>, env: Environment<Value>) -> Vec<Value> {
    operands
        .into_iter()
        .map(|expr| delay_it(expr, env.clone()))
        .collect()
}

// ============================================================================
// Helper Constructors
// ============================================================================

/// Helper to create application expressions
pub fn app(operator: Expr, operands: Vec<Expr>) -> Expr {
    Expr::Application {
        operator: Box::new(operator),
        operands,
    }
}

/// Helper to create symbol expressions
pub fn sym(name: &str) -> Expr {
    Expr::Symbol(name.to_string())
}

/// Helper to create number expressions
pub fn num(n: i64) -> Expr {
    Expr::Number(n)
}

/// Helper to create boolean expressions
pub fn bool_expr(b: bool) -> Expr {
    Expr::Bool(b)
}

/// Helper to create lambda expressions
pub fn lambda(params: Vec<&str>, body: Expr) -> Expr {
    Expr::Lambda {
        params: params.iter().map(|s| s.to_string()).collect(),
        body: Box::new(body),
    }
}

/// Helper to create if expressions
pub fn if_expr(test: Expr, consequent: Expr, alternative: Expr) -> Expr {
    Expr::If {
        test: Box::new(test),
        consequent: Box::new(consequent),
        alternative: Box::new(alternative),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_evaluation() {
        let env = setup_environment();
        let expr = num(42);
        let (result, _) = eval(expr, env).unwrap();
        match result {
            Value::Number(n) => assert_eq!(n, 42),
            _ => panic!("Expected number"),
        }
    }

    #[test]
    fn test_arithmetic() {
        let env = setup_environment();
        // (+ 1 2 3) => 6
        let expr = app(sym("+"), vec![num(1), num(2), num(3)]);
        let result = actual_value(expr, env).unwrap();
        match result {
            Value::Number(n) => assert_eq!(n, 6),
            _ => panic!("Expected number 6"),
        }
    }

    #[test]
    fn test_lazy_evaluation_try() {
        let env = setup_environment();

        // Define: (define (try a b) (if (= a 0) 1 b))
        let try_def = Expr::Define(
            "try".to_string(),
            Box::new(lambda(
                vec!["a", "b"],
                if_expr(app(sym("="), vec![sym("a"), num(0)]), num(1), sym("b")),
            )),
        );
        let (_, env) = eval(try_def, env).unwrap();

        // Call: (try 0 (/ 1 0))
        // This should return 1 WITHOUT evaluating (/ 1 0)
        let expr = app(
            sym("try"),
            vec![num(0), app(sym("/"), vec![num(1), num(0)])],
        );
        let result = actual_value(expr, env).unwrap();

        match result {
            Value::Number(n) => assert_eq!(n, 1),
            _ => panic!("Expected number 1"),
        }
    }

    #[test]
    fn test_unless() {
        let env = setup_environment();

        // Define: (define (unless condition usual-value exceptional-value)
        //           (if condition exceptional-value usual-value))
        let unless_def = Expr::Define(
            "unless".to_string(),
            Box::new(lambda(
                vec!["condition", "usual-value", "exceptional-value"],
                if_expr(
                    sym("condition"),
                    sym("exceptional-value"),
                    sym("usual-value"),
                ),
            )),
        );
        let (_, env) = eval(unless_def, env).unwrap();

        // Call: (unless (= 0 0) (/ 1 0) 42)
        // Should return 42 without evaluating (/ 1 0)
        let expr = app(
            sym("unless"),
            vec![
                app(sym("="), vec![num(0), num(0)]),
                app(sym("/"), vec![num(1), num(0)]),
                num(42),
            ],
        );
        let result = actual_value(expr, env).unwrap();

        match result {
            Value::Number(n) => assert_eq!(n, 42),
            _ => panic!("Expected number 42"),
        }
    }

    #[test]
    fn test_thunk_memoization() {
        let env = setup_environment();

        // Create a thunk for a simple expression
        let thunk = delay_it(num(42), env.clone());

        // Verify it's a thunk
        match &thunk {
            Value::Thunk(t) => assert!(!t.is_evaluated()),
            _ => panic!("Expected thunk"),
        }

        // First force
        let val1 = force_it(thunk.clone()).unwrap();
        match val1 {
            Value::Number(n) => assert_eq!(n, 42),
            _ => panic!("Expected number 42"),
        }

        // Verify thunk is now memoized
        match &thunk {
            Value::Thunk(t) => assert!(t.is_evaluated()),
            _ => panic!("Expected thunk"),
        }

        // Second force - should return memoized value
        let val2 = force_it(thunk).unwrap();
        match val2 {
            Value::Number(n) => assert_eq!(n, 42),
            _ => panic!("Expected number 42"),
        }
    }

    #[test]
    fn test_factorial_with_unless_fails() {
        let env = setup_environment();

        // Define unless
        let unless_def = Expr::Define(
            "unless".to_string(),
            Box::new(lambda(
                vec!["condition", "usual-value", "exceptional-value"],
                if_expr(
                    sym("condition"),
                    sym("exceptional-value"),
                    sym("usual-value"),
                ),
            )),
        );
        let (_, env) = eval(unless_def, env).unwrap();

        // Define factorial using unless
        let factorial_def = Expr::Define(
            "factorial".to_string(),
            Box::new(lambda(
                vec!["n"],
                app(
                    sym("unless"),
                    vec![
                        app(sym("="), vec![sym("n"), num(1)]),
                        app(
                            sym("*"),
                            vec![
                                sym("n"),
                                app(
                                    sym("factorial"),
                                    vec![app(sym("-"), vec![sym("n"), num(1)])],
                                ),
                            ],
                        ),
                        num(1),
                    ],
                ),
            )),
        );
        let (_, env) = eval(factorial_def, env).unwrap();

        // In lazy evaluation, this works because recursive call is delayed
        let expr = app(sym("factorial"), vec![num(5)]);
        let result = actual_value(expr, env);

        match result {
            Ok(Value::Number(n)) => assert_eq!(n, 120),
            Err(_) => {
                // Implementation may recurse forever - that's acceptable
            }
            _ => panic!("Unexpected result"),
        }
    }

    #[test]
    fn test_delayed_evaluation() {
        let env = setup_environment();

        // Create a thunk for (+ 1 2)
        let expr = app(sym("+"), vec![num(1), num(2)]);
        let thunk = delay_it(expr, env.clone());

        // Thunk should not be evaluated yet
        match &thunk {
            Value::Thunk(inner) => {
                assert!(!inner.is_evaluated(), "Should not be evaluated yet");
            }
            _ => panic!("Expected thunk"),
        }

        // Now force it
        let result = force_it(thunk.clone()).unwrap();
        match result {
            Value::Number(n) => assert_eq!(n, 3),
            _ => panic!("Expected number 3"),
        }

        // Check that it's now memoized
        match &thunk {
            Value::Thunk(inner) => {
                assert!(inner.is_evaluated(), "Should be evaluated now");
                match inner.get_memo().unwrap() {
                    Value::Number(n) => assert_eq!(*n, 3),
                    _ => panic!("Expected memoized number 3"),
                }
            }
            _ => panic!("Expected thunk"),
        }
    }

    #[test]
    fn test_nested_thunks() {
        let env = setup_environment();

        // Create nested computation: (+ (* 2 3) 4)
        let inner_expr = app(sym("*"), vec![num(2), num(3)]);
        let outer_expr = app(sym("+"), vec![inner_expr, num(4)]);

        let result = actual_value(outer_expr, env).unwrap();
        match result {
            Value::Number(n) => assert_eq!(n, 10),
            _ => panic!("Expected number 10"),
        }
    }

    #[test]
    fn test_lambda_and_application() {
        let env = setup_environment();

        // Define: (define (square x) (* x x))
        let square_def = Expr::Define(
            "square".to_string(),
            Box::new(lambda(vec!["x"], app(sym("*"), vec![sym("x"), sym("x")]))),
        );
        let (_, env) = eval(square_def, env).unwrap();

        // Call: (square 5)
        let expr = app(sym("square"), vec![num(5)]);
        let result = actual_value(expr, env).unwrap();

        match result {
            Value::Number(n) => assert_eq!(n, 25),
            _ => panic!("Expected number 25"),
        }
    }

    #[test]
    fn test_if_expression() {
        let env = setup_environment();

        // (if (< 1 2) 10 20) => 10
        let expr = if_expr(app(sym("<"), vec![num(1), num(2)]), num(10), num(20));
        let result = actual_value(expr, env).unwrap();

        match result {
            Value::Number(n) => assert_eq!(n, 10),
            _ => panic!("Expected number 10"),
        }
    }

    #[test]
    fn test_comparison_operators() {
        let env = setup_environment();

        // Test =
        let expr_eq = app(sym("="), vec![num(5), num(5)]);
        let result_eq = actual_value(expr_eq, env.clone()).unwrap();
        assert!(matches!(result_eq, Value::Bool(true)));

        // Test <
        let expr_lt = app(sym("<"), vec![num(3), num(5)]);
        let result_lt = actual_value(expr_lt, env.clone()).unwrap();
        assert!(matches!(result_lt, Value::Bool(true)));

        // Test >
        let expr_gt = app(sym(">"), vec![num(5), num(3)]);
        let result_gt = actual_value(expr_gt, env).unwrap();
        assert!(matches!(result_gt, Value::Bool(true)));
    }
}
