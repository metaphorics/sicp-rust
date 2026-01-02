//! Section 4.1: The Metacircular Evaluator
//!
//! A Scheme interpreter written in Rust, demonstrating metalinguistic abstraction.
//!
//! ## Key Rust Mappings from Scheme:
//!
//! - `eval/apply` → `match` on `enum Expr` for exhaustive dispatch
//! - Expressions → Algebraic data types (enum Expr with variants)
//! - Environments → `HashMap<String, Value>` with `Rc<RefCell<>>` for parent links
//! - Special forms → match arms for if, lambda, define, etc.
//! - Closures → Struct capturing parameters, body, and environment
//!
//! ## Architecture:
//!
//! ```text
//! eval(expr, env) → Value
//!   ↓
//!   match expr {
//!     Number/String → self-evaluating
//!     Symbol → env.lookup(name)
//!     Quote → quoted expression
//!     If → eval predicate, then consequent or alternative
//!     Lambda → create closure capturing environment
//!     Define → bind value in environment
//!     Set → update existing binding
//!     Begin → eval sequence, return last
//!     Cond → desugar to nested if
//!     Let → desugar to lambda application
//!     Application → apply(eval(operator), eval(operands))
//!   }
//!
//! apply(procedure, args) → Value
//!   ↓
//!   match procedure {
//!     Primitive(f) → f(args)
//!     Closure{params, body, env} → eval(body, env.extend(params, args))
//!   }
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

/// Expression type - represents the abstract syntax tree
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Self-evaluating numbers
    Number(i64),
    /// Self-evaluating strings
    String(String),
    /// Variable reference
    Symbol(String),
    /// Quoted expression: (quote expr)
    Quote(Box<Expr>),
    /// Conditional: (if predicate consequent alternative)
    If {
        predicate: Box<Expr>,
        consequent: Box<Expr>,
        alternative: Box<Expr>,
    },
    /// Lambda abstraction: (lambda (params...) body...)
    Lambda {
        params: Vec<String>,
        body: Vec<Expr>,
    },
    /// Variable definition: (define var expr)
    Define { name: String, value: Box<Expr> },
    /// Assignment: (set! var expr)
    Set { name: String, value: Box<Expr> },
    /// Sequence: (begin expr...)
    Begin(Vec<Expr>),
    /// Conditional with multiple clauses: (cond (pred expr)... (else expr))
    Cond(Vec<CondClause>),
    /// Let binding: (let ((var expr)...) body...)
    Let {
        bindings: Vec<(String, Expr)>,
        body: Vec<Expr>,
    },
    /// Procedure application: (operator operand...)
    Application {
        operator: Box<Expr>,
        operands: Vec<Expr>,
    },
    /// Empty list - used in quote
    Nil,
    /// Cons pair for quoted lists
    Cons(Box<Expr>, Box<Expr>),
}

/// Cond clause: predicate and actions
#[derive(Debug, Clone, PartialEq)]
pub struct CondClause {
    pub predicate: Expr,
    pub actions: Vec<Expr>,
}

/// Runtime values produced by evaluation
#[derive(Debug, Clone)]
pub enum Value {
    Number(i64),
    Bool(bool),
    String(String),
    Symbol(String),
    /// Compound procedure (closure)
    Closure {
        params: Vec<String>,
        body: Vec<Expr>,
        env: Rc<RefCell<Environment>>,
    },
    /// Primitive procedure
    Primitive(String, PrimitiveFn),
    /// Cons pair
    Pair(Box<Value>, Box<Value>),
    /// Empty list
    Nil,
    /// Void (returned by define/set!)
    Void,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Number(a), Value::Number(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Symbol(a), Value::Symbol(b)) => a == b,
            (Value::Nil, Value::Nil) => true,
            (Value::Void, Value::Void) => true,
            (Value::Pair(a1, a2), Value::Pair(b1, b2)) => a1 == b1 && a2 == b2,
            // Procedures compared by identity (not structural equality)
            _ => false,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Number(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", if *b { "#t" } else { "#f" }),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Symbol(s) => write!(f, "{}", s),
            Value::Closure { params, .. } => {
                write!(f, "#<procedure ({})>", params.join(" "))
            }
            Value::Primitive(name, _) => write!(f, "#<primitive:{}>", name),
            Value::Nil => write!(f, "()"),
            Value::Void => write!(f, "#<void>"),
            Value::Pair(car, cdr) => {
                write!(f, "(")?;
                write!(f, "{}", car)?;
                let mut current = cdr.as_ref();
                loop {
                    match current {
                        Value::Nil => break,
                        Value::Pair(car, cdr) => {
                            write!(f, " {}", car)?;
                            current = cdr.as_ref();
                        }
                        _ => {
                            write!(f, " . {}", current)?;
                            break;
                        }
                    }
                }
                write!(f, ")")
            }
        }
    }
}

/// Primitive function type
pub type PrimitiveFn = fn(&[Value]) -> Result<Value, EvalError>;

/// Evaluation errors
#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    UnboundVariable(String),
    TypeError(String),
    ArityMismatch { expected: usize, got: usize },
    DivisionByZero,
    InvalidSyntax(String),
    RuntimeError(String),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::UnboundVariable(var) => write!(f, "Unbound variable: {}", var),
            EvalError::TypeError(msg) => write!(f, "Type error: {}", msg),
            EvalError::ArityMismatch { expected, got } => {
                write!(f, "Arity mismatch: expected {}, got {}", expected, got)
            }
            EvalError::DivisionByZero => write!(f, "Division by zero"),
            EvalError::InvalidSyntax(msg) => write!(f, "Invalid syntax: {}", msg),
            EvalError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
        }
    }
}

impl std::error::Error for EvalError {}

/// Environment: maps variable names to values
///
/// Environments are organized as a chain of frames, each represented as a HashMap.
/// The parent link enables lexical scoping.
///
/// ## Ownership Model:
/// - `Rc<RefCell<Environment>>` allows shared mutable access
/// - Closures capture their defining environment
/// - Child environments reference parent via Rc
#[derive(Debug, Clone)]
pub struct Environment {
    frame: HashMap<String, Value>,
    parent: Option<Rc<RefCell<Environment>>>,
}

impl Environment {
    /// Create a new empty environment with no parent
    pub fn new() -> Self {
        Environment {
            frame: HashMap::new(),
            parent: None,
        }
    }

    /// Create a new environment extending a parent
    pub fn extend(parent: Rc<RefCell<Environment>>) -> Self {
        Environment {
            frame: HashMap::new(),
            parent: Some(parent),
        }
    }

    /// Look up a variable's value in this environment or its parents
    pub fn lookup(&self, name: &str) -> Result<Value, EvalError> {
        if let Some(value) = self.frame.get(name) {
            Ok(value.clone())
        } else if let Some(ref parent) = self.parent {
            parent.borrow().lookup(name)
        } else {
            Err(EvalError::UnboundVariable(name.to_string()))
        }
    }

    /// Define a variable in the current frame (creates or updates)
    pub fn define(&mut self, name: String, value: Value) {
        self.frame.insert(name, value);
    }

    /// Set an existing variable's value (searches up the environment chain)
    pub fn set(&mut self, name: &str, value: Value) -> Result<(), EvalError> {
        if self.frame.contains_key(name) {
            self.frame.insert(name.to_string(), value);
            Ok(())
        } else if let Some(ref parent) = self.parent {
            parent.borrow_mut().set(name, value)
        } else {
            Err(EvalError::UnboundVariable(name.to_string()))
        }
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluate an expression in an environment
///
/// This is the core of the interpreter, implementing the eval-apply cycle.
pub fn eval(expr: &Expr, env: Rc<RefCell<Environment>>) -> Result<Value, EvalError> {
    match expr {
        // Self-evaluating expressions
        Expr::Number(n) => Ok(Value::Number(*n)),
        Expr::String(s) => Ok(Value::String(s.clone())),

        // Variable lookup
        Expr::Symbol(name) => env.borrow().lookup(name),

        // Quote: return the expression unevaluated
        Expr::Quote(expr) => expr_to_value(expr),

        // If: evaluate predicate, then consequent or alternative
        Expr::If {
            predicate,
            consequent,
            alternative,
        } => {
            let pred_value = eval(predicate, env.clone())?;
            if is_true(&pred_value) {
                eval(consequent, env)
            } else {
                eval(alternative, env)
            }
        }

        // Lambda: create a closure capturing the current environment
        Expr::Lambda { params, body } => Ok(Value::Closure {
            params: params.clone(),
            body: body.clone(),
            env: env.clone(),
        }),

        // Define: evaluate value and bind in current environment
        Expr::Define { name, value } => {
            let val = eval(value, env.clone())?;
            env.borrow_mut().define(name.clone(), val);
            Ok(Value::Void)
        }

        // Set!: evaluate value and update existing binding
        Expr::Set { name, value } => {
            let val = eval(value, env.clone())?;
            env.borrow_mut().set(name, val)?;
            Ok(Value::Void)
        }

        // Begin: evaluate sequence, return last value
        Expr::Begin(exprs) => eval_sequence(exprs, env),

        // Cond: transform to nested if and evaluate
        Expr::Cond(clauses) => {
            let if_expr = cond_to_if(clauses)?;
            eval(&if_expr, env)
        }

        // Let: transform to lambda application
        Expr::Let { bindings, body } => {
            let let_expr = let_to_application(bindings, body);
            eval(&let_expr, env)
        }

        // Application: evaluate operator and operands, then apply
        Expr::Application { operator, operands } => {
            let proc = eval(operator, env.clone())?;
            let args = eval_list(operands, env)?;
            apply(proc, args)
        }

        // Empty list
        Expr::Nil => Ok(Value::Nil),

        // Cons (only in quoted expressions)
        Expr::Cons(car, cdr) => {
            let car_val = expr_to_value(car)?;
            let cdr_val = expr_to_value(cdr)?;
            Ok(Value::Pair(Box::new(car_val), Box::new(cdr_val)))
        }
    }
}

/// Apply a procedure to arguments
pub fn apply(procedure: Value, args: Vec<Value>) -> Result<Value, EvalError> {
    match procedure {
        // Primitive procedure: call the Rust function
        Value::Primitive(_, func) => func(&args),

        // Compound procedure: evaluate body in extended environment
        Value::Closure { params, body, env } => {
            if params.len() != args.len() {
                return Err(EvalError::ArityMismatch {
                    expected: params.len(),
                    got: args.len(),
                });
            }

            // Create new environment extending the closure's environment
            let mut new_env = Environment::extend(env);

            // Bind parameters to arguments
            for (param, arg) in params.iter().zip(args.iter()) {
                new_env.define(param.clone(), arg.clone());
            }

            let new_env_rc = Rc::new(RefCell::new(new_env));
            eval_sequence(&body, new_env_rc)
        }

        _ => Err(EvalError::TypeError(format!(
            "Cannot apply non-procedure: {}",
            procedure
        ))),
    }
}

/// Evaluate a sequence of expressions, returning the last value
fn eval_sequence(exprs: &[Expr], env: Rc<RefCell<Environment>>) -> Result<Value, EvalError> {
    if exprs.is_empty() {
        return Ok(Value::Void);
    }

    let mut result = Value::Void;
    for expr in exprs {
        result = eval(expr, env.clone())?;
    }
    Ok(result)
}

/// Evaluate a list of expressions, returning a vector of values
fn eval_list(exprs: &[Expr], env: Rc<RefCell<Environment>>) -> Result<Vec<Value>, EvalError> {
    exprs.iter().map(|e| eval(e, env.clone())).collect()
}

/// Convert an Expr to a Value (for quoted expressions)
fn expr_to_value(expr: &Expr) -> Result<Value, EvalError> {
    match expr {
        Expr::Number(n) => Ok(Value::Number(*n)),
        Expr::String(s) => Ok(Value::String(s.clone())),
        Expr::Symbol(s) => Ok(Value::Symbol(s.clone())),
        Expr::Nil => Ok(Value::Nil),
        Expr::Cons(car, cdr) => {
            let car_val = expr_to_value(car)?;
            let cdr_val = expr_to_value(cdr)?;
            Ok(Value::Pair(Box::new(car_val), Box::new(cdr_val)))
        }
        _ => Err(EvalError::InvalidSyntax(
            "Cannot quote complex expression".to_string(),
        )),
    }
}

/// Test if a value is true (everything except #f is true)
fn is_true(value: &Value) -> bool {
    !matches!(value, Value::Bool(false))
}

/// Transform cond to nested if
fn cond_to_if(clauses: &[CondClause]) -> Result<Expr, EvalError> {
    if clauses.is_empty() {
        return Ok(Expr::Quote(Box::new(Expr::Symbol("false".to_string()))));
    }

    let first = &clauses[0];
    let rest = &clauses[1..];

    // Check for else clause
    if let Expr::Symbol(s) = &first.predicate {
        if s == "else" {
            if !rest.is_empty() {
                return Err(EvalError::InvalidSyntax(
                    "else clause must be last".to_string(),
                ));
            }
            return Ok(if first.actions.len() == 1 {
                first.actions[0].clone()
            } else {
                Expr::Begin(first.actions.clone())
            });
        }
    }

    let consequent = if first.actions.len() == 1 {
        first.actions[0].clone()
    } else {
        Expr::Begin(first.actions.clone())
    };

    let alternative = cond_to_if(rest)?;

    Ok(Expr::If {
        predicate: Box::new(first.predicate.clone()),
        consequent: Box::new(consequent),
        alternative: Box::new(alternative),
    })
}

/// Transform let to lambda application
fn let_to_application(bindings: &[(String, Expr)], body: &[Expr]) -> Expr {
    let params: Vec<String> = bindings.iter().map(|(name, _)| name.clone()).collect();
    let args: Vec<Expr> = bindings.iter().map(|(_, expr)| expr.clone()).collect();

    Expr::Application {
        operator: Box::new(Expr::Lambda {
            params,
            body: body.to_vec(),
        }),
        operands: args,
    }
}

// ============================================================================
// Primitive Procedures
// ============================================================================

/// Create a global environment with primitive procedures
pub fn setup_environment() -> Rc<RefCell<Environment>> {
    let mut env = Environment::new();

    // Arithmetic
    env.define("+".to_string(), Value::Primitive("+".to_string(), prim_add));
    env.define("-".to_string(), Value::Primitive("-".to_string(), prim_sub));
    env.define("*".to_string(), Value::Primitive("*".to_string(), prim_mul));
    env.define("/".to_string(), Value::Primitive("/".to_string(), prim_div));

    // Comparison
    env.define("=".to_string(), Value::Primitive("=".to_string(), prim_eq));
    env.define("<".to_string(), Value::Primitive("<".to_string(), prim_lt));
    env.define(">".to_string(), Value::Primitive(">".to_string(), prim_gt));
    env.define(
        "<=".to_string(),
        Value::Primitive("<=".to_string(), prim_lte),
    );
    env.define(
        ">=".to_string(),
        Value::Primitive(">=".to_string(), prim_gte),
    );

    // List operations
    env.define(
        "cons".to_string(),
        Value::Primitive("cons".to_string(), prim_cons),
    );
    env.define(
        "car".to_string(),
        Value::Primitive("car".to_string(), prim_car),
    );
    env.define(
        "cdr".to_string(),
        Value::Primitive("cdr".to_string(), prim_cdr),
    );
    env.define(
        "null?".to_string(),
        Value::Primitive("null?".to_string(), prim_null),
    );
    env.define(
        "list".to_string(),
        Value::Primitive("list".to_string(), prim_list),
    );

    // Type predicates
    env.define(
        "number?".to_string(),
        Value::Primitive("number?".to_string(), prim_number_p),
    );
    env.define(
        "symbol?".to_string(),
        Value::Primitive("symbol?".to_string(), prim_symbol_p),
    );
    env.define(
        "pair?".to_string(),
        Value::Primitive("pair?".to_string(), prim_pair_p),
    );

    // Display
    env.define(
        "display".to_string(),
        Value::Primitive("display".to_string(), prim_display),
    );

    // Boolean constants
    env.define("true".to_string(), Value::Bool(true));
    env.define("false".to_string(), Value::Bool(false));

    Rc::new(RefCell::new(env))
}

// Arithmetic primitives
fn prim_add(args: &[Value]) -> Result<Value, EvalError> {
    let sum = args.iter().try_fold(0i64, |acc, v| match v {
        Value::Number(n) => Ok(acc + n),
        _ => Err(EvalError::TypeError("+ requires numbers".to_string())),
    })?;
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
            if args.len() == 1 {
                Ok(Value::Number(-first))
            } else {
                let result = args[1..].iter().try_fold(*first, |acc, v| match v {
                    Value::Number(n) => Ok(acc - n),
                    _ => Err(EvalError::TypeError("- requires numbers".to_string())),
                })?;
                Ok(Value::Number(result))
            }
        }
        _ => Err(EvalError::TypeError("- requires numbers".to_string())),
    }
}

fn prim_mul(args: &[Value]) -> Result<Value, EvalError> {
    let product = args.iter().try_fold(1i64, |acc, v| match v {
        Value::Number(n) => Ok(acc * n),
        _ => Err(EvalError::TypeError("* requires numbers".to_string())),
    })?;
    Ok(Value::Number(product))
}

fn prim_div(args: &[Value]) -> Result<Value, EvalError> {
    if args.is_empty() {
        return Err(EvalError::ArityMismatch {
            expected: 1,
            got: 0,
        });
    }
    match &args[0] {
        Value::Number(first) => {
            let result = args[1..].iter().try_fold(*first, |acc, v| match v {
                Value::Number(n) => {
                    if *n == 0 {
                        Err(EvalError::DivisionByZero)
                    } else {
                        Ok(acc / n)
                    }
                }
                _ => Err(EvalError::TypeError("/ requires numbers".to_string())),
            })?;
            Ok(Value::Number(result))
        }
        _ => Err(EvalError::TypeError("/ requires numbers".to_string())),
    }
}

// Comparison primitives
fn prim_eq(args: &[Value]) -> Result<Value, EvalError> {
    if args.len() < 2 {
        return Err(EvalError::ArityMismatch {
            expected: 2,
            got: args.len(),
        });
    }
    match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Bool(a == b)),
        _ => Err(EvalError::TypeError("= requires numbers".to_string())),
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
        _ => Err(EvalError::TypeError("< requires numbers".to_string())),
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
        _ => Err(EvalError::TypeError("> requires numbers".to_string())),
    }
}

fn prim_lte(args: &[Value]) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::ArityMismatch {
            expected: 2,
            got: args.len(),
        });
    }
    match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Bool(a <= b)),
        _ => Err(EvalError::TypeError("<= requires numbers".to_string())),
    }
}

fn prim_gte(args: &[Value]) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::ArityMismatch {
            expected: 2,
            got: args.len(),
        });
    }
    match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Bool(a >= b)),
        _ => Err(EvalError::TypeError(">= requires numbers".to_string())),
    }
}

// List primitives
fn prim_cons(args: &[Value]) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::ArityMismatch {
            expected: 2,
            got: args.len(),
        });
    }
    Ok(Value::Pair(
        Box::new(args[0].clone()),
        Box::new(args[1].clone()),
    ))
}

fn prim_car(args: &[Value]) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::ArityMismatch {
            expected: 1,
            got: args.len(),
        });
    }
    match &args[0] {
        Value::Pair(car, _) => Ok((**car).clone()),
        _ => Err(EvalError::TypeError("car requires a pair".to_string())),
    }
}

fn prim_cdr(args: &[Value]) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::ArityMismatch {
            expected: 1,
            got: args.len(),
        });
    }
    match &args[0] {
        Value::Pair(_, cdr) => Ok((**cdr).clone()),
        _ => Err(EvalError::TypeError("cdr requires a pair".to_string())),
    }
}

fn prim_null(args: &[Value]) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::ArityMismatch {
            expected: 1,
            got: args.len(),
        });
    }
    Ok(Value::Bool(matches!(args[0], Value::Nil)))
}

fn prim_list(args: &[Value]) -> Result<Value, EvalError> {
    Ok(list_from_vec(args.to_vec()))
}

// Type predicates
fn prim_number_p(args: &[Value]) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::ArityMismatch {
            expected: 1,
            got: args.len(),
        });
    }
    Ok(Value::Bool(matches!(args[0], Value::Number(_))))
}

fn prim_symbol_p(args: &[Value]) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::ArityMismatch {
            expected: 1,
            got: args.len(),
        });
    }
    Ok(Value::Bool(matches!(args[0], Value::Symbol(_))))
}

fn prim_pair_p(args: &[Value]) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::ArityMismatch {
            expected: 1,
            got: args.len(),
        });
    }
    Ok(Value::Bool(matches!(args[0], Value::Pair(_, _))))
}

fn prim_display(args: &[Value]) -> Result<Value, EvalError> {
    for arg in args {
        print!("{}", arg);
    }
    Ok(Value::Void)
}

// Helper: convert Vec<Value> to proper list
fn list_from_vec(values: Vec<Value>) -> Value {
    values
        .into_iter()
        .rev()
        .fold(Value::Nil, |acc, v| Value::Pair(Box::new(v), Box::new(acc)))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to evaluate a string expression
    fn eval_expr(expr: Expr) -> Result<Value, EvalError> {
        let env = setup_environment();
        eval(&expr, env)
    }

    #[test]
    fn test_self_evaluating() {
        assert_eq!(eval_expr(Expr::Number(42)).unwrap(), Value::Number(42));
        assert_eq!(
            eval_expr(Expr::String("hello".to_string())).unwrap(),
            Value::String("hello".to_string())
        );
    }

    #[test]
    fn test_arithmetic() {
        // (+ 1 2 3)
        let expr = Expr::Application {
            operator: Box::new(Expr::Symbol("+".to_string())),
            operands: vec![Expr::Number(1), Expr::Number(2), Expr::Number(3)],
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Number(6));

        // (* 2 3 4)
        let expr = Expr::Application {
            operator: Box::new(Expr::Symbol("*".to_string())),
            operands: vec![Expr::Number(2), Expr::Number(3), Expr::Number(4)],
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Number(24));

        // (- 10 3)
        let expr = Expr::Application {
            operator: Box::new(Expr::Symbol("-".to_string())),
            operands: vec![Expr::Number(10), Expr::Number(3)],
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Number(7));

        // (/ 20 4)
        let expr = Expr::Application {
            operator: Box::new(Expr::Symbol("/".to_string())),
            operands: vec![Expr::Number(20), Expr::Number(4)],
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Number(5));
    }

    #[test]
    fn test_comparison() {
        // (= 5 5)
        let expr = Expr::Application {
            operator: Box::new(Expr::Symbol("=".to_string())),
            operands: vec![Expr::Number(5), Expr::Number(5)],
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Bool(true));

        // (< 3 5)
        let expr = Expr::Application {
            operator: Box::new(Expr::Symbol("<".to_string())),
            operands: vec![Expr::Number(3), Expr::Number(5)],
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Bool(true));

        // (> 3 5)
        let expr = Expr::Application {
            operator: Box::new(Expr::Symbol(">".to_string())),
            operands: vec![Expr::Number(3), Expr::Number(5)],
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Bool(false));
    }

    #[test]
    fn test_quote() {
        // (quote x)
        let expr = Expr::Quote(Box::new(Expr::Symbol("x".to_string())));
        assert_eq!(eval_expr(expr).unwrap(), Value::Symbol("x".to_string()));
    }

    #[test]
    fn test_if() {
        // (if true 1 2)
        let expr = Expr::If {
            predicate: Box::new(Expr::Symbol("true".to_string())),
            consequent: Box::new(Expr::Number(1)),
            alternative: Box::new(Expr::Number(2)),
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Number(1));

        // (if false 1 2)
        let expr = Expr::If {
            predicate: Box::new(Expr::Symbol("false".to_string())),
            consequent: Box::new(Expr::Number(1)),
            alternative: Box::new(Expr::Number(2)),
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Number(2));

        // (if (< 3 5) 10 20)
        let expr = Expr::If {
            predicate: Box::new(Expr::Application {
                operator: Box::new(Expr::Symbol("<".to_string())),
                operands: vec![Expr::Number(3), Expr::Number(5)],
            }),
            consequent: Box::new(Expr::Number(10)),
            alternative: Box::new(Expr::Number(20)),
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Number(10));
    }

    #[test]
    fn test_define_and_lookup() {
        let env = setup_environment();

        // (define x 42)
        let define_expr = Expr::Define {
            name: "x".to_string(),
            value: Box::new(Expr::Number(42)),
        };
        eval(&define_expr, env.clone()).unwrap();

        // x
        let lookup_expr = Expr::Symbol("x".to_string());
        assert_eq!(eval(&lookup_expr, env).unwrap(), Value::Number(42));
    }

    #[test]
    fn test_set() {
        let env = setup_environment();

        // (define x 10)
        let define_expr = Expr::Define {
            name: "x".to_string(),
            value: Box::new(Expr::Number(10)),
        };
        eval(&define_expr, env.clone()).unwrap();

        // (set! x 20)
        let set_expr = Expr::Set {
            name: "x".to_string(),
            value: Box::new(Expr::Number(20)),
        };
        eval(&set_expr, env.clone()).unwrap();

        // x
        let lookup_expr = Expr::Symbol("x".to_string());
        assert_eq!(eval(&lookup_expr, env).unwrap(), Value::Number(20));
    }

    #[test]
    fn test_lambda_and_application() {
        let env = setup_environment();

        // (define square (lambda (x) (* x x)))
        let define_expr = Expr::Define {
            name: "square".to_string(),
            value: Box::new(Expr::Lambda {
                params: vec!["x".to_string()],
                body: vec![Expr::Application {
                    operator: Box::new(Expr::Symbol("*".to_string())),
                    operands: vec![Expr::Symbol("x".to_string()), Expr::Symbol("x".to_string())],
                }],
            }),
        };
        eval(&define_expr, env.clone()).unwrap();

        // (square 5)
        let app_expr = Expr::Application {
            operator: Box::new(Expr::Symbol("square".to_string())),
            operands: vec![Expr::Number(5)],
        };
        assert_eq!(eval(&app_expr, env).unwrap(), Value::Number(25));
    }

    #[test]
    fn test_begin() {
        let env = setup_environment();

        // (begin (define x 10) (define y 20) (+ x y))
        let expr = Expr::Begin(vec![
            Expr::Define {
                name: "x".to_string(),
                value: Box::new(Expr::Number(10)),
            },
            Expr::Define {
                name: "y".to_string(),
                value: Box::new(Expr::Number(20)),
            },
            Expr::Application {
                operator: Box::new(Expr::Symbol("+".to_string())),
                operands: vec![Expr::Symbol("x".to_string()), Expr::Symbol("y".to_string())],
            },
        ]);
        assert_eq!(eval(&expr, env).unwrap(), Value::Number(30));
    }

    #[test]
    fn test_cond() {
        // (cond ((< 5 3) 1) ((> 5 3) 2) (else 3))
        let expr = Expr::Cond(vec![
            CondClause {
                predicate: Expr::Application {
                    operator: Box::new(Expr::Symbol("<".to_string())),
                    operands: vec![Expr::Number(5), Expr::Number(3)],
                },
                actions: vec![Expr::Number(1)],
            },
            CondClause {
                predicate: Expr::Application {
                    operator: Box::new(Expr::Symbol(">".to_string())),
                    operands: vec![Expr::Number(5), Expr::Number(3)],
                },
                actions: vec![Expr::Number(2)],
            },
            CondClause {
                predicate: Expr::Symbol("else".to_string()),
                actions: vec![Expr::Number(3)],
            },
        ]);
        assert_eq!(eval_expr(expr).unwrap(), Value::Number(2));
    }

    #[test]
    fn test_let() {
        // (let ((x 10) (y 20)) (+ x y))
        let expr = Expr::Let {
            bindings: vec![
                ("x".to_string(), Expr::Number(10)),
                ("y".to_string(), Expr::Number(20)),
            ],
            body: vec![Expr::Application {
                operator: Box::new(Expr::Symbol("+".to_string())),
                operands: vec![Expr::Symbol("x".to_string()), Expr::Symbol("y".to_string())],
            }],
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Number(30));
    }

    #[test]
    fn test_factorial() {
        let env = setup_environment();

        // (define factorial
        //   (lambda (n)
        //     (if (= n 0)
        //         1
        //         (* n (factorial (- n 1))))))
        let define_expr = Expr::Define {
            name: "factorial".to_string(),
            value: Box::new(Expr::Lambda {
                params: vec!["n".to_string()],
                body: vec![Expr::If {
                    predicate: Box::new(Expr::Application {
                        operator: Box::new(Expr::Symbol("=".to_string())),
                        operands: vec![Expr::Symbol("n".to_string()), Expr::Number(0)],
                    }),
                    consequent: Box::new(Expr::Number(1)),
                    alternative: Box::new(Expr::Application {
                        operator: Box::new(Expr::Symbol("*".to_string())),
                        operands: vec![
                            Expr::Symbol("n".to_string()),
                            Expr::Application {
                                operator: Box::new(Expr::Symbol("factorial".to_string())),
                                operands: vec![Expr::Application {
                                    operator: Box::new(Expr::Symbol("-".to_string())),
                                    operands: vec![Expr::Symbol("n".to_string()), Expr::Number(1)],
                                }],
                            },
                        ],
                    }),
                }],
            }),
        };
        eval(&define_expr, env.clone()).unwrap();

        // (factorial 5)
        let app_expr = Expr::Application {
            operator: Box::new(Expr::Symbol("factorial".to_string())),
            operands: vec![Expr::Number(5)],
        };
        assert_eq!(eval(&app_expr, env).unwrap(), Value::Number(120));
    }

    #[test]
    fn test_list_operations() {
        let env = setup_environment();

        // (define lst (list 1 2 3))
        let define_expr = Expr::Define {
            name: "lst".to_string(),
            value: Box::new(Expr::Application {
                operator: Box::new(Expr::Symbol("list".to_string())),
                operands: vec![Expr::Number(1), Expr::Number(2), Expr::Number(3)],
            }),
        };
        eval(&define_expr, env.clone()).unwrap();

        // (car lst)
        let car_expr = Expr::Application {
            operator: Box::new(Expr::Symbol("car".to_string())),
            operands: vec![Expr::Symbol("lst".to_string())],
        };
        assert_eq!(eval(&car_expr, env.clone()).unwrap(), Value::Number(1));

        // (car (cdr lst))
        let cadr_expr = Expr::Application {
            operator: Box::new(Expr::Symbol("car".to_string())),
            operands: vec![Expr::Application {
                operator: Box::new(Expr::Symbol("cdr".to_string())),
                operands: vec![Expr::Symbol("lst".to_string())],
            }],
        };
        assert_eq!(eval(&cadr_expr, env).unwrap(), Value::Number(2));
    }

    #[test]
    fn test_closure_captures_environment() {
        let env = setup_environment();

        // (define make-adder
        //   (lambda (x)
        //     (lambda (y) (+ x y))))
        let define_expr = Expr::Define {
            name: "make-adder".to_string(),
            value: Box::new(Expr::Lambda {
                params: vec!["x".to_string()],
                body: vec![Expr::Lambda {
                    params: vec!["y".to_string()],
                    body: vec![Expr::Application {
                        operator: Box::new(Expr::Symbol("+".to_string())),
                        operands: vec![
                            Expr::Symbol("x".to_string()),
                            Expr::Symbol("y".to_string()),
                        ],
                    }],
                }],
            }),
        };
        eval(&define_expr, env.clone()).unwrap();

        // (define add5 (make-adder 5))
        let define_add5 = Expr::Define {
            name: "add5".to_string(),
            value: Box::new(Expr::Application {
                operator: Box::new(Expr::Symbol("make-adder".to_string())),
                operands: vec![Expr::Number(5)],
            }),
        };
        eval(&define_add5, env.clone()).unwrap();

        // (add5 10)
        let app_expr = Expr::Application {
            operator: Box::new(Expr::Symbol("add5".to_string())),
            operands: vec![Expr::Number(10)],
        };
        assert_eq!(eval(&app_expr, env).unwrap(), Value::Number(15));
    }

    #[test]
    fn test_error_unbound_variable() {
        let result = eval_expr(Expr::Symbol("undefined".to_string()));
        assert!(matches!(result, Err(EvalError::UnboundVariable(_))));
    }

    #[test]
    fn test_error_arity_mismatch() {
        let env = setup_environment();

        // (define f (lambda (x y) (+ x y)))
        let define_expr = Expr::Define {
            name: "f".to_string(),
            value: Box::new(Expr::Lambda {
                params: vec!["x".to_string(), "y".to_string()],
                body: vec![Expr::Application {
                    operator: Box::new(Expr::Symbol("+".to_string())),
                    operands: vec![Expr::Symbol("x".to_string()), Expr::Symbol("y".to_string())],
                }],
            }),
        };
        eval(&define_expr, env.clone()).unwrap();

        // (f 1) - wrong arity
        let app_expr = Expr::Application {
            operator: Box::new(Expr::Symbol("f".to_string())),
            operands: vec![Expr::Number(1)],
        };
        let result = eval(&app_expr, env);
        assert!(matches!(result, Err(EvalError::ArityMismatch { .. })));
    }

    #[test]
    fn test_error_division_by_zero() {
        let expr = Expr::Application {
            operator: Box::new(Expr::Symbol("/".to_string())),
            operands: vec![Expr::Number(10), Expr::Number(0)],
        };
        let result = eval_expr(expr);
        assert!(matches!(result, Err(EvalError::DivisionByZero)));
    }

    #[test]
    fn test_nested_let() {
        // (let ((x 10))
        //   (let ((y 20))
        //     (+ x y)))
        let expr = Expr::Let {
            bindings: vec![("x".to_string(), Expr::Number(10))],
            body: vec![Expr::Let {
                bindings: vec![("y".to_string(), Expr::Number(20))],
                body: vec![Expr::Application {
                    operator: Box::new(Expr::Symbol("+".to_string())),
                    operands: vec![Expr::Symbol("x".to_string()), Expr::Symbol("y".to_string())],
                }],
            }],
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Number(30));
    }
}
