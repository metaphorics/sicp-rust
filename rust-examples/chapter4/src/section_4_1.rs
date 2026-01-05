//! 4.1절: 메타순환 평가기 (The Metacircular Evaluator)
//!
//! 영속적 환경을 사용해 관용적인 러스트로 작성한 Scheme 인터프리터
//! (A Scheme interpreter written in idiomatic Rust using persistent environments).
//!
//! ## SICP Scheme 대비 핵심 설계 변경 (Key Design Changes from SICP Scheme):
//!
//! - **영속적 환경 (Persistent environments)**: 구조적 공유로 O(1) 복제를 위한 `im::HashMap` 사용
//! - **소유 클로저 (Owned closures)**: 클로저가 환경을 복제해 캡처 (공유 참조 아님)
//! - **함수형 상태 스레딩 (Functional state threading)**: `eval`이 `(Value, Environment)`를 반환
//! - **Rc<RefCell<>> 없음 (No Rc<RefCell<>>)**: 타입 시스템을 통한 명시적 소유권
//!
//! ## 러스트 vs 스킴 의미론 (Rust vs Scheme Semantics):
//!
//! Scheme에서는 `set!`가 모든 클로저가 보는 공유 환경을 변이한다.
//! 이 순수 함수형 구현에서는 `set!`가 새 환경을 반환하므로
//! 이전 환경을 캡처한 클로저는 변경을 보지 못한다.
//! 이는 순수 함수형 의미론에 부합하며 러스트의 소유권 모델을 보여준다
//! (In Scheme, `set!` mutates a shared environment visible to all closures.
//! In this pure functional implementation, `set!` returns a new environment,
//! so closures that captured the old environment won't see the change.
//! This matches pure functional semantics and demonstrates Rust's ownership model).
//!
//! ## 아키텍처 (Architecture):
//!
//! ```text
//! eval(expr, env) → (Value, Environment)
//!   ↓
//!   match expr {
//!     Number/String → (자기 평가 (self-evaluating), env)
//!     Symbol → (env.lookup(name), env)  // 변수 조회 (variable lookup)
//!     Lambda → (Closure { env: env.clone() }, env)  // 클로저 생성 (closure creation)
//!     Define → env' = env.define(name, value); (Void, env')  // 정의 (definition)
//!     Application → apply(procedure, args)  // 적용 (application)
//!   }
//! ```

use sicp_common::Environment;
use std::fmt;

// =============================================================================
// 식 타입 (AST) (Expression Types)
// =============================================================================

/// 식 타입 - 추상 구문 트리를 나타낸다 (Expression type - represents the abstract syntax tree).
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// 자기 평가 숫자 (Self-evaluating numbers)
    Number(i64),
    /// 자기 평가 문자열 (Self-evaluating strings)
    String(String),
    /// 변수 참조 (Variable reference)
    Symbol(String),
    /// 인용 표현식 (Quoted expression): (quote expr)
    Quote(Box<Expr>),
    /// 조건식 (Conditional): (if predicate consequent alternative)
    If {
        predicate: Box<Expr>,
        consequent: Box<Expr>,
        alternative: Box<Expr>,
    },
    /// 람다 추상화 (Lambda abstraction): (lambda (params...) body...)
    Lambda {
        params: Vec<String>,
        body: Vec<Expr>,
    },
    /// 변수 정의 (Variable definition): (define var expr)
    Define { name: String, value: Box<Expr> },
    /// 순차 실행 (Sequence): (begin expr...)
    Begin(Vec<Expr>),
    /// 다중 절 조건식 (Conditional with multiple clauses): (cond (pred expr)... (else expr))
    Cond(Vec<CondClause>),
    /// let 바인딩 (Let binding): (let ((var expr)...) body...)
    Let {
        bindings: Vec<(String, Expr)>,
        body: Vec<Expr>,
    },
    /// 프로시저 적용 (Procedure application): (operator operand...)
    Application {
        operator: Box<Expr>,
        operands: Vec<Expr>,
    },
    /// 빈 리스트 - quote에서 사용 (Empty list - used in quote)
    Nil,
    /// 인용 리스트용 cons 쌍 (Cons pair for quoted lists)
    Cons(Box<Expr>, Box<Expr>),
}

/// cond 절: 술어와 동작 (Cond clause: predicate and actions).
#[derive(Debug, Clone, PartialEq)]
pub struct CondClause {
    pub predicate: Expr,
    pub actions: Vec<Expr>,
}

// =============================================================================
// 런타임 값 (Runtime Values)
// =============================================================================

/// 평가로 생성되는 런타임 값 (Runtime values produced by evaluation).
///
/// 클로저는 환경을 소유한다 (Rc<RefCell<>> 없음)
/// (Closures own their environment (no Rc<RefCell<>>)).
#[derive(Debug, Clone)]
pub enum Value {
    Number(i64),
    Bool(bool),
    String(String),
    Symbol(String),
    /// 복합 프로시저 - 클로저가 캡처한 환경을 소유한다
    /// (Compound procedure - closure OWNS its captured environment)
    Closure {
        params: Vec<String>,
        body: Vec<Expr>,
        env: Environment<Value>,
        /// 호출 시 재귀 바인딩을 위한 선택적 self 이름
        /// (Optional self-name for recursive binding at call time)
        self_name: Option<String>,
    },
    /// 기본 프로시저 (Primitive procedure)
    Primitive(String, PrimitiveFn),
    /// cons 쌍 (Cons pair)
    Pair(Box<Value>, Box<Value>),
    /// 빈 리스트 (Empty list)
    Nil,
    /// void (define의 반환값) (Void (returned by define))
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
                write!(f, "#<프로시저 (procedure) ({})>", params.join(" "))
            }
            Value::Primitive(name, _) => write!(f, "#<기본-프로시저 (primitive):{}>", name),
            Value::Nil => write!(f, "()"),
            Value::Void => write!(f, "#<무효 (void)>"),
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

/// 기본 함수 타입 (Primitive function type).
pub type PrimitiveFn = fn(&[Value]) -> Result<Value, EvalError>;

// =============================================================================
// 오류 (Errors)
// =============================================================================

/// 평가 오류 (Evaluation errors).
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
            EvalError::UnboundVariable(var) => {
                write!(f, "바인딩되지 않은 변수 (Unbound variable): {}", var)
            }
            EvalError::TypeError(msg) => write!(f, "타입 오류 (Type error): {}", msg),
            EvalError::ArityMismatch { expected, got } => {
                write!(
                    f,
                    "인자 수 불일치 (Arity mismatch): expected {}, got {}",
                    expected, got
                )
            }
            EvalError::DivisionByZero => write!(f, "0으로 나눔 (Division by zero)"),
            EvalError::InvalidSyntax(msg) => write!(f, "잘못된 문법 (Invalid syntax): {}", msg),
            EvalError::RuntimeError(msg) => write!(f, "런타임 오류 (Runtime error): {}", msg),
        }
    }
}

impl std::error::Error for EvalError {}

// =============================================================================
// 평가기 핵심 (Evaluator Core)
// =============================================================================

/// 환경에서 식을 평가한다 (Evaluate an expression in an environment).
///
/// `(Value, Environment)`를 반환하며, 새 환경은 `define`에서 생긴 새 바인딩을 포함할 수 있다
/// (Returns `(Value, Environment)` - the new environment may have new bindings
/// from `define` expressions).
/// 이 함수형 접근은 변이 없이 상태를 평가기 내부로 스레딩한다
/// (This functional approach threads state through the evaluator without mutation).
pub fn eval(
    expr: &Expr,
    env: Environment<Value>,
) -> Result<(Value, Environment<Value>), EvalError> {
    match expr {
        // 자기 평가 식 (Self-evaluating expressions)
        Expr::Number(n) => Ok((Value::Number(*n), env)),
        Expr::String(s) => Ok((Value::String(s.clone()), env)),

        // 변수 조회 (Variable lookup)
        Expr::Symbol(name) => {
            let value = env
                .lookup(name)
                .ok_or_else(|| EvalError::UnboundVariable(name.clone()))?
                .clone();
            Ok((value, env))
        }

        // 인용: 식을 평가하지 않고 반환 (Quote: return the expression unevaluated)
        Expr::Quote(quoted) => {
            let value = expr_to_value(quoted)?;
            Ok((value, env))
        }

        // if: 술어를 평가하고, 결과에 따라 분기
        // (If: evaluate predicate, then consequent or alternative)
        Expr::If {
            predicate,
            consequent,
            alternative,
        } => {
            let (pred_value, env) = eval(predicate, env)?;
            if is_true(&pred_value) {
                eval(consequent, env)
            } else {
                eval(alternative, env)
            }
        }

        // 람다: 현재 환경을 캡처하는 클로저 생성
        // (Lambda: create a closure capturing the current environment)
        Expr::Lambda { params, body } => {
            let closure = Value::Closure {
                params: params.clone(),
                body: body.clone(),
                env: env.clone(), // 클로저는 복제본을 소유 (구조적 공유로 O(1)) (Closure OWNS a copy (O(1) due to structural sharing))
                self_name: None,  // 익명 람다 (Anonymous lambda)
            };
            Ok((closure, env))
        }

        // define: 값을 평가하고 새 환경에 바인딩
        // (Define: evaluate value and bind in a new environment)
        // 람다는 self_name으로 재귀가 가능하도록 특별 처리
        // (Special handling for lambda to enable recursion via self_name)
        Expr::Define { name, value } => {
            match value.as_ref() {
                // 람다 정의는 self_name을 설정해 재귀 바인딩 가능
                // (For lambda definitions, set self_name for recursive binding)
                Expr::Lambda { params, body } => {
                    let closure = Value::Closure {
                        params: params.clone(),
                        body: body.clone(),
                        env: env.clone(),
                        self_name: Some(name.clone()), // 재귀 자기 참조 허용 (Enable recursive self-reference)
                    };
                    let new_env = env.define(name.clone(), closure);
                    Ok((Value::Void, new_env))
                }
                // 람다가 아닌 값: 일반 평가
                // (Non-lambda values: evaluate normally)
                _ => {
                    let (val, env) = eval(value, env)?;
                    let new_env = env.define(name.clone(), val);
                    Ok((Value::Void, new_env))
                }
            }
        }

        // begin: 순차 평가 후 마지막 값을 반환
        // (Begin: evaluate sequence, return last value)
        Expr::Begin(exprs) => eval_sequence(exprs, env),

        // cond: 중첩 if로 변환 후 평가
        // (Cond: transform to nested if and evaluate)
        Expr::Cond(clauses) => {
            let if_expr = cond_to_if(clauses)?;
            eval(&if_expr, env)
        }

        // let: 람다 적용으로 변환
        // (Let: transform to lambda application)
        Expr::Let { bindings, body } => {
            let let_expr = let_to_application(bindings, body);
            eval(&let_expr, env)
        }

        // 적용: 연산자와 피연산자를 평가하고 적용
        // (Application: evaluate operator and operands, then apply)
        Expr::Application { operator, operands } => {
            let (proc, env) = eval(operator, env)?;
            let (args, env) = eval_list(operands, env)?;
            let result = apply(proc, args)?;
            Ok((result, env))
        }

        // 빈 리스트 (Empty list)
        Expr::Nil => Ok((Value::Nil, env)),

        // cons (인용된 표현식에서만) (Cons (only in quoted expressions))
        Expr::Cons(car, cdr) => {
            let car_val = expr_to_value(car)?;
            let cdr_val = expr_to_value(cdr)?;
            Ok((Value::Pair(Box::new(car_val), Box::new(cdr_val)), env))
        }
    }
}

/// 프로시저에 인자를 적용한다 (Apply a procedure to arguments).
pub fn apply(procedure: Value, args: Vec<Value>) -> Result<Value, EvalError> {
    match procedure.clone() {
        // 기본 프로시저: 러스트 함수를 호출
        // (Primitive procedure: call the Rust function)
        Value::Primitive(_, func) => func(&args),

        // 복합 프로시저: 확장된 환경에서 본문 평가
        // (Compound procedure: evaluate body in extended environment)
        Value::Closure {
            params,
            body,
            env,
            self_name,
        } => {
            if params.len() != args.len() {
                return Err(EvalError::ArityMismatch {
                    expected: params.len(),
                    got: args.len(),
                });
            }

            // 클로저가 캡처한 환경에서 시작
            // (Start with the closure's captured environment)
            let mut new_env = env;

            // 재귀 호출을 위해 self-name을 바인딩 (재귀의 핵심 수정)
            // (Bind self-name for recursive calls (the key fix for recursion!))
            if let Some(name) = self_name {
                new_env = new_env.define(name, procedure);
            }

            // 매개변수 바인딩으로 확장 (Extend with parameter bindings)
            let bindings: Vec<(String, Value)> = params.into_iter().zip(args).collect();
            new_env = new_env.extend(bindings);

            // 본문을 평가하고 값만 반환 (최종 환경은 무시)
            // (Evaluate body and return just the value (ignore final environment))
            let (result, _) = eval_sequence(&body, new_env)?;
            Ok(result)
        }

        _ => Err(EvalError::TypeError(format!(
            "프로시저가 아닌 값에 적용 불가 (Cannot apply non-procedure): {}",
            procedure
        ))),
    }
}

/// 식의 시퀀스를 평가하고 마지막 값을 반환한다
/// (Evaluate a sequence of expressions, returning the last value).
fn eval_sequence(
    exprs: &[Expr],
    mut env: Environment<Value>,
) -> Result<(Value, Environment<Value>), EvalError> {
    if exprs.is_empty() {
        return Ok((Value::Void, env));
    }

    let mut result = Value::Void;
    for expr in exprs {
        let (val, new_env) = eval(expr, env)?;
        result = val;
        env = new_env;
    }
    Ok((result, env))
}

/// 식 리스트를 평가하고 값 벡터를 반환한다
/// (Evaluate a list of expressions, returning a vector of values).
fn eval_list(
    exprs: &[Expr],
    mut env: Environment<Value>,
) -> Result<(Vec<Value>, Environment<Value>), EvalError> {
    let mut values = Vec::with_capacity(exprs.len());
    for expr in exprs {
        let (val, new_env) = eval(expr, env)?;
        values.push(val);
        env = new_env;
    }
    Ok((values, env))
}

/// Expr를 Value로 변환한다 (인용 표현식용)
/// (Convert an Expr to a Value (for quoted expressions)).
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
            "복잡한 표현식은 인용할 수 없음 (Cannot quote complex expression)".to_string(),
        )),
    }
}

/// 값이 참인지 확인 (#f만 거짓)
/// (Test if a value is true (everything except #f is true)).
fn is_true(value: &Value) -> bool {
    !matches!(value, Value::Bool(false))
}

/// cond를 중첩 if로 변환한다 (Transform cond to nested if).
fn cond_to_if(clauses: &[CondClause]) -> Result<Expr, EvalError> {
    if clauses.is_empty() {
        return Ok(Expr::Quote(Box::new(Expr::Symbol("false".to_string()))));
    }

    let first = &clauses[0];
    let rest = &clauses[1..];

    // else 절 확인 (Check for else clause)
    if let Expr::Symbol(s) = &first.predicate
        && s == "else"
    {
        if !rest.is_empty() {
            return Err(EvalError::InvalidSyntax(
                "else 절은 마지막이어야 함 (else clause must be last)".to_string(),
            ));
        }
        return Ok(if first.actions.len() == 1 {
            first.actions[0].clone()
        } else {
            Expr::Begin(first.actions.clone())
        });
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

/// let을 람다 적용으로 변환한다 (Transform let to lambda application).
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

// =============================================================================
// 기본 프로시저 (Primitive Procedures)
// =============================================================================

/// 기본 프로시저가 들어 있는 전역 환경을 만든다
/// (Create a global environment with primitive procedures).
pub fn setup_environment() -> Environment<Value> {
    let env = Environment::new();

    // 산술 (Arithmetic)
    let env = env.define("+".to_string(), Value::Primitive("+".to_string(), prim_add));
    let env = env.define("-".to_string(), Value::Primitive("-".to_string(), prim_sub));
    let env = env.define("*".to_string(), Value::Primitive("*".to_string(), prim_mul));
    let env = env.define("/".to_string(), Value::Primitive("/".to_string(), prim_div));

    // 비교 (Comparison)
    let env = env.define("=".to_string(), Value::Primitive("=".to_string(), prim_eq));
    let env = env.define("<".to_string(), Value::Primitive("<".to_string(), prim_lt));
    let env = env.define(">".to_string(), Value::Primitive(">".to_string(), prim_gt));
    let env = env.define(
        "<=".to_string(),
        Value::Primitive("<=".to_string(), prim_lte),
    );
    let env = env.define(
        ">=".to_string(),
        Value::Primitive(">=".to_string(), prim_gte),
    );

    // 리스트 연산 (List operations)
    let env = env.define(
        "cons".to_string(),
        Value::Primitive("cons".to_string(), prim_cons),
    );
    let env = env.define(
        "car".to_string(),
        Value::Primitive("car".to_string(), prim_car),
    );
    let env = env.define(
        "cdr".to_string(),
        Value::Primitive("cdr".to_string(), prim_cdr),
    );
    let env = env.define(
        "null?".to_string(),
        Value::Primitive("null?".to_string(), prim_null),
    );
    let env = env.define(
        "list".to_string(),
        Value::Primitive("list".to_string(), prim_list),
    );

    // 타입 판정 (Type predicates)
    let env = env.define(
        "number?".to_string(),
        Value::Primitive("number?".to_string(), prim_number_p),
    );
    let env = env.define(
        "symbol?".to_string(),
        Value::Primitive("symbol?".to_string(), prim_symbol_p),
    );
    let env = env.define(
        "pair?".to_string(),
        Value::Primitive("pair?".to_string(), prim_pair_p),
    );

    // 표시 (Display)
    let env = env.define(
        "display".to_string(),
        Value::Primitive("display".to_string(), prim_display),
    );

    // 불리언 상수 (Boolean constants)
    let env = env.define("true".to_string(), Value::Bool(true));

    env.define("false".to_string(), Value::Bool(false))
}

// 산술 기본 프로시저 (Arithmetic primitives)
fn prim_add(args: &[Value]) -> Result<Value, EvalError> {
    let sum = args.iter().try_fold(0i64, |acc, v| match v {
        Value::Number(n) => Ok(acc + n),
        _ => Err(EvalError::TypeError(
            "+ 는 숫자가 필요함 (+ requires numbers)".to_string(),
        )),
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
                    _ => Err(EvalError::TypeError(
                        "- 는 숫자가 필요함 (- requires numbers)".to_string(),
                    )),
                })?;
                Ok(Value::Number(result))
            }
        }
        _ => Err(EvalError::TypeError(
            "- 는 숫자가 필요함 (- requires numbers)".to_string(),
        )),
    }
}

fn prim_mul(args: &[Value]) -> Result<Value, EvalError> {
    let product = args.iter().try_fold(1i64, |acc, v| match v {
        Value::Number(n) => Ok(acc * n),
        _ => Err(EvalError::TypeError(
            "* 는 숫자가 필요함 (* requires numbers)".to_string(),
        )),
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
                _ => Err(EvalError::TypeError(
                    "/ 는 숫자가 필요함 (/ requires numbers)".to_string(),
                )),
            })?;
            Ok(Value::Number(result))
        }
        _ => Err(EvalError::TypeError(
            "/ 는 숫자가 필요함 (/ requires numbers)".to_string(),
        )),
    }
}

// 비교 기본 프로시저 (Comparison primitives)
fn prim_eq(args: &[Value]) -> Result<Value, EvalError> {
    if args.len() < 2 {
        return Err(EvalError::ArityMismatch {
            expected: 2,
            got: args.len(),
        });
    }
    match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Bool(a == b)),
        _ => Err(EvalError::TypeError(
            "= 는 숫자가 필요함 (= requires numbers)".to_string(),
        )),
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
        _ => Err(EvalError::TypeError(
            "< 는 숫자가 필요함 (< requires numbers)".to_string(),
        )),
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
        _ => Err(EvalError::TypeError(
            "> 는 숫자가 필요함 (> requires numbers)".to_string(),
        )),
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
        _ => Err(EvalError::TypeError(
            "<= 는 숫자가 필요함 (<= requires numbers)".to_string(),
        )),
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
        _ => Err(EvalError::TypeError(
            ">= 는 숫자가 필요함 (>= requires numbers)".to_string(),
        )),
    }
}

// 리스트 기본 프로시저 (List primitives)
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
        _ => Err(EvalError::TypeError(
            "car 는 쌍이 필요함 (car requires a pair)".to_string(),
        )),
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
        _ => Err(EvalError::TypeError(
            "cdr 는 쌍이 필요함 (cdr requires a pair)".to_string(),
        )),
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

// 타입 판정 (Type predicates)
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

// 헬퍼: Vec<Value>를 올바른 리스트로 변환
// (Helper: convert Vec<Value> to proper list)
fn list_from_vec(values: Vec<Value>) -> Value {
    values
        .into_iter()
        .rev()
        .fold(Value::Nil, |acc, v| Value::Pair(Box::new(v), Box::new(acc)))
}

// =============================================================================
// 테스트 (Tests)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 새 환경에서 식을 평가하는 헬퍼
    /// (Helper to evaluate an expression in a fresh environment)
    fn eval_expr(expr: Expr) -> Result<Value, EvalError> {
        let env = setup_environment();
        let (value, _) = eval(&expr, env)?;
        Ok(value)
    }

    #[test]
    fn test_self_evaluating() {
        assert_eq!(eval_expr(Expr::Number(42)).unwrap(), Value::Number(42));
        assert_eq!(
            eval_expr(Expr::String("안녕 (hello)".to_string())).unwrap(),
            Value::String("안녕 (hello)".to_string())
        );
    }

    #[test]
    fn test_arithmetic() {
        // (+ 1 2 3) - 덧셈 예시 (addition example)
        let expr = Expr::Application {
            operator: Box::new(Expr::Symbol("+".to_string())),
            operands: vec![Expr::Number(1), Expr::Number(2), Expr::Number(3)],
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Number(6));

        // (* 2 3 4) - 곱셈 예시 (multiplication example)
        let expr = Expr::Application {
            operator: Box::new(Expr::Symbol("*".to_string())),
            operands: vec![Expr::Number(2), Expr::Number(3), Expr::Number(4)],
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Number(24));

        // (- 10 3) - 뺄셈 예시 (subtraction example)
        let expr = Expr::Application {
            operator: Box::new(Expr::Symbol("-".to_string())),
            operands: vec![Expr::Number(10), Expr::Number(3)],
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Number(7));

        // (/ 20 4) - 나눗셈 예시 (division example)
        let expr = Expr::Application {
            operator: Box::new(Expr::Symbol("/".to_string())),
            operands: vec![Expr::Number(20), Expr::Number(4)],
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Number(5));
    }

    #[test]
    fn test_comparison() {
        // (= 5 5) - 동등 비교 예시 (equality example)
        let expr = Expr::Application {
            operator: Box::new(Expr::Symbol("=".to_string())),
            operands: vec![Expr::Number(5), Expr::Number(5)],
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Bool(true));

        // (< 3 5) - 크기 비교 예시 (less-than example)
        let expr = Expr::Application {
            operator: Box::new(Expr::Symbol("<".to_string())),
            operands: vec![Expr::Number(3), Expr::Number(5)],
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Bool(true));

        // (> 3 5) - 크기 비교 예시 (greater-than example)
        let expr = Expr::Application {
            operator: Box::new(Expr::Symbol(">".to_string())),
            operands: vec![Expr::Number(3), Expr::Number(5)],
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Bool(false));
    }

    #[test]
    fn test_quote() {
        // (quote x) - 인용 예시 (quote example)
        let expr = Expr::Quote(Box::new(Expr::Symbol("x".to_string())));
        assert_eq!(eval_expr(expr).unwrap(), Value::Symbol("x".to_string()));
    }

    #[test]
    fn test_if() {
        // (if true 1 2) - if 예시 (if example)
        let expr = Expr::If {
            predicate: Box::new(Expr::Symbol("true".to_string())),
            consequent: Box::new(Expr::Number(1)),
            alternative: Box::new(Expr::Number(2)),
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Number(1));

        // (if false 1 2) - if 예시 (if example)
        let expr = Expr::If {
            predicate: Box::new(Expr::Symbol("false".to_string())),
            consequent: Box::new(Expr::Number(1)),
            alternative: Box::new(Expr::Number(2)),
        };
        assert_eq!(eval_expr(expr).unwrap(), Value::Number(2));

        // (if (< 3 5) 10 20) - if 예시 (if example)
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

        // (define x 42) - x를 42로 정의 (define x as 42)
        let define_expr = Expr::Define {
            name: "x".to_string(),
            value: Box::new(Expr::Number(42)),
        };
        let (_, env) = eval(&define_expr, env).unwrap();

        // x - 변수 조회 (variable lookup)
        let lookup_expr = Expr::Symbol("x".to_string());
        let (value, _) = eval(&lookup_expr, env).unwrap();
        assert_eq!(value, Value::Number(42));
    }

    #[test]
    fn test_lambda_and_application() {
        let env = setup_environment();

        // (define square (lambda (x) (* x x))) - 제곱 함수 정의 (square definition)
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
        let (_, env) = eval(&define_expr, env).unwrap();

        // (square 5) - 함수 호출 예시 (function call example)
        let app_expr = Expr::Application {
            operator: Box::new(Expr::Symbol("square".to_string())),
            operands: vec![Expr::Number(5)],
        };
        let (value, _) = eval(&app_expr, env).unwrap();
        assert_eq!(value, Value::Number(25));
    }

    #[test]
    fn test_begin() {
        let env = setup_environment();

        // (begin (define x 10) (define y 20) (+ x y)) - begin 예시 (begin example)
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
        let (value, _) = eval(&expr, env).unwrap();
        assert_eq!(value, Value::Number(30));
    }

    #[test]
    fn test_cond() {
        // (cond ((< 5 3) 1) ((> 5 3) 2) (else 3)) - cond 예시 (cond example)
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
        // (let ((x 10) (y 20)) (+ x y)) - let 예시 (let example)
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

        // 팩토리얼 정의 예시 (factorial definition example)
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
        let (_, env) = eval(&define_expr, env).unwrap();

        // (factorial 5) - 호출 예시 (call example)
        let app_expr = Expr::Application {
            operator: Box::new(Expr::Symbol("factorial".to_string())),
            operands: vec![Expr::Number(5)],
        };
        let (value, _) = eval(&app_expr, env).unwrap();
        assert_eq!(value, Value::Number(120));
    }

    #[test]
    fn test_list_operations() {
        let env = setup_environment();

        // (define lst (list 1 2 3)) - 리스트 정의 (list definition)
        let define_expr = Expr::Define {
            name: "lst".to_string(),
            value: Box::new(Expr::Application {
                operator: Box::new(Expr::Symbol("list".to_string())),
                operands: vec![Expr::Number(1), Expr::Number(2), Expr::Number(3)],
            }),
        };
        let (_, env) = eval(&define_expr, env).unwrap();

        // (car lst) - 리스트 첫 요소 (first element)
        let car_expr = Expr::Application {
            operator: Box::new(Expr::Symbol("car".to_string())),
            operands: vec![Expr::Symbol("lst".to_string())],
        };
        let (value, env) = eval(&car_expr, env).unwrap();
        assert_eq!(value, Value::Number(1));

        // (car (cdr lst)) - 두 번째 요소 (second element)
        let cadr_expr = Expr::Application {
            operator: Box::new(Expr::Symbol("car".to_string())),
            operands: vec![Expr::Application {
                operator: Box::new(Expr::Symbol("cdr".to_string())),
                operands: vec![Expr::Symbol("lst".to_string())],
            }],
        };
        let (value, _) = eval(&cadr_expr, env).unwrap();
        assert_eq!(value, Value::Number(2));
    }

    #[test]
    fn test_closure_captures_environment() {
        let env = setup_environment();

        // 클로저 캡처 예시 (closure capture example)
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
        let (_, env) = eval(&define_expr, env).unwrap();

        // (define add5 (make-adder 5)) - 부분 적용 (partial application)
        let define_add5 = Expr::Define {
            name: "add5".to_string(),
            value: Box::new(Expr::Application {
                operator: Box::new(Expr::Symbol("make-adder".to_string())),
                operands: vec![Expr::Number(5)],
            }),
        };
        let (_, env) = eval(&define_add5, env).unwrap();

        // (add5 10) - 클로저 호출 (closure call)
        let app_expr = Expr::Application {
            operator: Box::new(Expr::Symbol("add5".to_string())),
            operands: vec![Expr::Number(10)],
        };
        let (value, _) = eval(&app_expr, env).unwrap();
        assert_eq!(value, Value::Number(15));
    }

    #[test]
    fn test_error_unbound_variable() {
        let result = eval_expr(Expr::Symbol("undefined".to_string()));
        assert!(matches!(result, Err(EvalError::UnboundVariable(_))));
    }

    #[test]
    fn test_error_arity_mismatch() {
        let env = setup_environment();

        // (define f (lambda (x y) (+ x y))) - 2인자 함수 정의 (two-arg function)
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
        let (_, env) = eval(&define_expr, env).unwrap();

        // (f 1) - 인자 수 오류 (wrong arity)
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
        // 중첩 let 예시 (nested let example)
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
