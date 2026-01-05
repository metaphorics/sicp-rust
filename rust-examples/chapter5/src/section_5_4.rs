//! SICP 5.4절: 명시적-제어 평가기 (The Explicit-Control Evaluator)
//!
//! 이 모듈은 Scheme 유사 언어를 위한 레지스터 머신 평가기를 구현한다.
//! 평가는 7개의 레지스터 (exp, env, val, continue, proc, argl, unev)와
//! 명시적 제어 흐름으로 프로그램을 실행하는 스택을 사용한다.
//! (This module implements a register machine evaluator for a Scheme-like language.
//! The evaluator uses 7 registers (exp, env, val, continue, proc, argl, unev) and
//! a stack to execute programs with explicit control flow.)
//!
//! ## 원문 대비 핵심 설계 변경 (Key Design Changes from Original):
//!
//! - **영속 환경 (Persistent environments)**: 구조적 공유로 O(1) 복제를 위해 `im::HashMap` 사용
//! - **소유 클로저 (Owned closures)**: 클로저가 환경을 clone으로 캡처 (Rc<RefCell<>> 사용 안 함)
//! - **set! 없음 (No set!)**: 순수 함수형 접근 - define만 사용, 변경 없음
//! - **쌍에 Box 사용 (Box for pairs)**: 쌍 구조의 소유권을 단순화
//!
//! 시연되는 핵심 개념 (Key concepts demonstrated):
//! - 명시적 제어를 가진 레지스터 기반 평가 (Register-based evaluation with explicit control)
//! - continue 레지스터 관리를 통한 꼬리 호출 최적화 (Tail-call optimization through careful continue register management)
//! - 머신 상태 저장/복원을 위한 스택 연산 (Stack operations for saving and restoring machine state)
//! - 명시적 타입 검사에 의한 표현식 디스패치 (Expression dispatch via explicit type checking)

use sicp_common::Environment;
use std::fmt;

/// 평가기에서의 표현식 타입 (Expression types in the evaluator)
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// 자기평가 표현식 (숫자, 불리언, 문자열)
    /// (Self-evaluating expressions (numbers, booleans, strings))
    Number(i64),
    Bool(bool),
    String(String),

    /// 변수 참조 (Variable reference)
    Symbol(String),

    /// 인용 표현식 (Quoted expression)
    Quote(Box<Expr>),

    /// 람다 표현식: (lambda (params...) body...)
    /// (Lambda expression: (lambda (params...) body...))
    Lambda {
        params: Vec<String>,
        body: Vec<Expr>,
    },

    /// if 표현식: (if predicate consequent alternative)
    /// (If expression: (if predicate consequent alternative))
    If {
        predicate: Box<Expr>,
        consequent: Box<Expr>,
        alternative: Box<Expr>,
    },

    /// 정의: (define var value)
    /// (Definition: (define var value))
    Definition {
        var: String,
        value: Box<Expr>,
    },

    /// begin 시퀀스: (begin expr...)
    /// (Begin sequence: (begin expr...))
    Begin(Vec<Expr>),

    /// 프로시저 적용: (operator operands...)
    /// (Procedure application: (operator operands...))
    Application {
        operator: Box<Expr>,
        operands: Vec<Expr>,
    },

    /// 리스트 구성 (인용 리스트용)
    /// (List construction (for quoted lists))
    Pair(Box<Expr>, Box<Expr>),

    /// 빈 리스트 (Empty list)
    Nil,
}

/// 런타임 값 (Runtime values)
#[derive(Debug, Clone)]
pub enum Value {
    Number(i64),
    Bool(bool),
    String(String),
    Symbol(String),
    Pair(Box<Value>, Box<Value>),
    Nil,

    /// 소유 환경을 가진 복합 프로시저 (영속)
    /// (Compound procedure with OWNED environment (persistent))
    Procedure {
        params: Vec<String>,
        body: Vec<Expr>,
        env: Environment<Value>,
        /// 재귀 바인딩을 위한 선택적 자기 이름
        /// (Optional self-name for recursive binding)
        self_name: Option<String>,
    },

    /// 기본 프로시저 (Primitive procedure)
    Primitive(String),

    /// 정의를 위한 특수 "ok" 값 (Special "ok" value for definitions)
    Ok,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Number(a), Value::Number(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Symbol(a), Value::Symbol(b)) => a == b,
            (Value::Nil, Value::Nil) => true,
            (Value::Ok, Value::Ok) => true,
            (Value::Pair(a1, a2), Value::Pair(b1, b2)) => a1 == b1 && a2 == b2,
            _ => false,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Number(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", if *b { "#t" } else { "#f" }),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Symbol(s) => write!(f, "{}", s),
            Value::Nil => write!(f, "()"),
            Value::Pair(_, _) => {
                write!(f, "(")?;
                self.write_list(f)?;
                write!(f, ")")
            }
            Value::Procedure { .. } => write!(f, "#<프로시저 (procedure)>"),
            Value::Primitive(name) => write!(f, "#<기본-프로시저 (primitive):{}>", name),
            Value::Ok => write!(f, "확인(ok)"),
        }
    }
}

impl Value {
    fn write_list(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Pair(car, cdr) => {
                write!(f, "{}", car)?;
                match cdr.as_ref() {
                    Value::Nil => Ok(()),
                    Value::Pair(_, _) => {
                        write!(f, " ")?;
                        cdr.write_list(f)
                    }
                    other => write!(f, " . {}", other),
                }
            }
            _ => write!(f, "{}", self),
        }
    }

    pub fn is_true(&self) -> bool {
        !matches!(self, Value::Bool(false))
    }
}

/// 레지스터 상태 저장용 스택 프레임 타입
/// (Stack frame types for saving register state)
#[derive(Debug, Clone)]
pub enum StackFrame {
    Continue(Label),
    Env(Environment<Value>),
    Unev(Vec<Expr>),
    Argl(Vec<Value>),
    Proc(Value),
    Exp(Expr),
}

/// 레지스터 머신의 제어 라벨
/// (Control labels for the register machine)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Label {
    EvalDispatch,
    EvSelfEval,
    EvVariable,
    EvQuoted,
    EvLambda,
    EvIf,
    EvIfDecide,
    EvIfConsequent,
    EvIfAlternative,
    EvDefinition,
    EvDefinition1,
    EvBegin,
    EvApplication,
    EvApplDidOperator,
    EvApplOperandLoop,
    EvApplAccumulateArg,
    EvApplLastArg,
    EvApplAccumLastArg,
    ApplyDispatch,
    PrimitiveApply,
    CompoundApply,
    EvSequence,
    EvSequenceContinue,
    EvSequenceLastExp,
    Done,
    Error,
}

/// 명시적-제어 평가기 머신
/// (The explicit-control evaluator machine)
pub struct EvaluatorMachine {
    // 7개의 레지스터 (Seven registers)
    exp: Option<Expr>,
    env: Environment<Value>,
    val: Option<Value>,
    continue_reg: Label,
    proc: Option<Value>,
    argl: Vec<Value>,
    unev: Vec<Expr>,

    // 상태 저장용 스택 (Stack for saving state)
    stack: Vec<StackFrame>,

    // 성능 메트릭 (Performance metrics)
    pub total_pushes: usize,
    pub max_depth: usize,

    // 현재 명령 라벨 (Current instruction label)
    current_label: Label,

    // 오류 메시지 (Error message)
    error: Option<String>,
}

impl EvaluatorMachine {
    pub fn new() -> Self {
        let global_env = Self::setup_environment();
        EvaluatorMachine {
            exp: None,
            env: global_env,
            val: None,
            continue_reg: Label::Done,
            proc: None,
            argl: Vec::new(),
            unev: Vec::new(),
            stack: Vec::new(),
            total_pushes: 0,
            max_depth: 0,
            current_label: Label::EvalDispatch,
            error: None,
        }
    }

    /// 기본 프로시저를 포함한 전역 환경 생성
    /// (Create global environment with primitive procedures)
    fn setup_environment() -> Environment<Value> {
        let primitives = vec![
            "+",
            "-",
            "*",
            "/",
            "=",
            "<",
            ">",
            "<=",
            ">=",
            "쌍(cons)",
            "머리(car)",
            "꼬리(cdr)",
            "빈리스트?(null?)",
            "쌍?(pair?)",
            "리스트(list)",
            "표시(display)",
            "새줄(newline)",
        ];

        let mut env = Environment::new();
        for prim in primitives {
            env = env.define(prim.to_string(), Value::Primitive(prim.to_string()));
        }

        // 특수 상수 정의 (Define special constants)
        env = env.define("참(true)".to_string(), Value::Bool(true));
        env = env.define("거짓(false)".to_string(), Value::Bool(false));

        env
    }

    /// 메트릭 추적을 포함한 스택 연산
    /// (Stack operations with metrics tracking)
    fn save(&mut self, frame: StackFrame) {
        self.stack.push(frame);
        self.total_pushes += 1;
        if self.stack.len() > self.max_depth {
            self.max_depth = self.stack.len();
        }
    }

    fn restore_continue(&mut self) -> Result<(), String> {
        match self.stack.pop() {
            Some(StackFrame::Continue(label)) => {
                self.continue_reg = label;
                Ok(())
            }
            Some(_) => Err("스택에 Continue가 있어야 함 (Expected Continue on stack)".to_string()),
            None => Err("스택 언더플로 (Stack underflow)".to_string()),
        }
    }

    fn restore_env(&mut self) -> Result<(), String> {
        match self.stack.pop() {
            Some(StackFrame::Env(env)) => {
                self.env = env;
                Ok(())
            }
            Some(_) => Err("스택에 Env가 있어야 함 (Expected Env on stack)".to_string()),
            None => Err("스택 언더플로 (Stack underflow)".to_string()),
        }
    }

    fn restore_unev(&mut self) -> Result<(), String> {
        match self.stack.pop() {
            Some(StackFrame::Unev(unev)) => {
                self.unev = unev;
                Ok(())
            }
            Some(_) => Err("스택에 Unev가 있어야 함 (Expected Unev on stack)".to_string()),
            None => Err("스택 언더플로 (Stack underflow)".to_string()),
        }
    }

    fn restore_argl(&mut self) -> Result<(), String> {
        match self.stack.pop() {
            Some(StackFrame::Argl(argl)) => {
                self.argl = argl;
                Ok(())
            }
            Some(_) => Err("스택에 Argl가 있어야 함 (Expected Argl on stack)".to_string()),
            None => Err("스택 언더플로 (Stack underflow)".to_string()),
        }
    }

    fn restore_proc(&mut self) -> Result<(), String> {
        match self.stack.pop() {
            Some(StackFrame::Proc(proc)) => {
                self.proc = Some(proc);
                Ok(())
            }
            Some(_) => Err("스택에 Proc가 있어야 함 (Expected Proc on stack)".to_string()),
            None => Err("스택 언더플로 (Stack underflow)".to_string()),
        }
    }

    fn restore_exp(&mut self) -> Result<(), String> {
        match self.stack.pop() {
            Some(StackFrame::Exp(exp)) => {
                self.exp = Some(exp);
                Ok(())
            }
            Some(_) => Err("스택에 Exp가 있어야 함 (Expected Exp on stack)".to_string()),
            None => Err("스택 언더플로 (Stack underflow)".to_string()),
        }
    }

    /// 표현식 평가 (Evaluate an expression)
    pub fn eval(&mut self, expr: Expr) -> Result<Value, String> {
        self.exp = Some(expr);
        self.continue_reg = Label::Done;
        self.current_label = Label::EvalDispatch;
        self.total_pushes = 0;
        self.max_depth = 0;

        self.run()
    }

    /// 메인 평가 루프 (Main evaluation loop)
    fn run(&mut self) -> Result<Value, String> {
        loop {
            match self.current_label {
                Label::EvalDispatch => self.eval_dispatch()?,
                Label::EvSelfEval => self.ev_self_eval()?,
                Label::EvVariable => self.ev_variable()?,
                Label::EvQuoted => self.ev_quoted()?,
                Label::EvLambda => self.ev_lambda()?,
                Label::EvIf => self.ev_if()?,
                Label::EvIfDecide => self.ev_if_decide()?,
                Label::EvIfConsequent => self.ev_if_consequent()?,
                Label::EvIfAlternative => self.ev_if_alternative()?,
                Label::EvDefinition => self.ev_definition()?,
                Label::EvDefinition1 => self.ev_definition_1()?,
                Label::EvBegin => self.ev_begin()?,
                Label::EvApplication => self.ev_application()?,
                Label::EvApplDidOperator => self.ev_appl_did_operator()?,
                Label::EvApplOperandLoop => self.ev_appl_operand_loop()?,
                Label::EvApplAccumulateArg => self.ev_appl_accumulate_arg()?,
                Label::EvApplLastArg => self.ev_appl_last_arg()?,
                Label::EvApplAccumLastArg => self.ev_appl_accum_last_arg()?,
                Label::ApplyDispatch => self.apply_dispatch()?,
                Label::PrimitiveApply => self.primitive_apply()?,
                Label::CompoundApply => self.compound_apply()?,
                Label::EvSequence => self.ev_sequence()?,
                Label::EvSequenceContinue => self.ev_sequence_continue()?,
                Label::EvSequenceLastExp => self.ev_sequence_last_exp()?,
                Label::Done => {
                    return self
                        .val
                        .clone()
                        .ok_or_else(|| "값이 생성되지 않음 (No value produced)".to_string());
                }
                Label::Error => {
                    return Err(self
                        .error
                        .clone()
                        .unwrap_or_else(|| "알 수 없는 오류 (Unknown error)".to_string()));
                }
            }
        }
    }

    /// 표현식 타입에 따른 디스패치
    /// (Dispatch based on expression type)
    fn eval_dispatch(&mut self) -> Result<(), String> {
        let exp = self
            .exp
            .clone()
            .ok_or("exp 레지스터에 표현식이 없음 (No expression in exp register)")?;

        match exp {
            Expr::Number(_) | Expr::Bool(_) | Expr::String(_) | Expr::Nil => {
                self.current_label = Label::EvSelfEval;
            }
            Expr::Symbol(_) => {
                self.current_label = Label::EvVariable;
            }
            Expr::Quote(_) => {
                self.current_label = Label::EvQuoted;
            }
            Expr::Lambda { .. } => {
                self.current_label = Label::EvLambda;
            }
            Expr::If { .. } => {
                self.current_label = Label::EvIf;
            }
            Expr::Definition { .. } => {
                self.current_label = Label::EvDefinition;
            }
            Expr::Begin(_) => {
                self.current_label = Label::EvBegin;
            }
            Expr::Application { .. } => {
                self.current_label = Label::EvApplication;
            }
            _ => {
                self.error = Some(format!(
                    "알 수 없는 표현식 타입: {:?} (Unknown expression type)",
                    exp
                ));
                self.current_label = Label::Error;
            }
        }
        Ok(())
    }

    /// 자기평가 표현식 평가 (Evaluate self-evaluating expression)
    fn ev_self_eval(&mut self) -> Result<(), String> {
        let exp = self
            .exp
            .clone()
            .ok_or("표현식이 없음 (No expression)")?;
        self.val = Some(match exp {
            Expr::Number(n) => Value::Number(n),
            Expr::Bool(b) => Value::Bool(b),
            Expr::String(s) => Value::String(s),
            Expr::Nil => Value::Nil,
            _ => return Err("자기평가 표현식이 아님 (Not self-evaluating)".to_string()),
        });
        self.current_label = self.continue_reg;
        Ok(())
    }

    /// 변수 평가 (Evaluate variable)
    fn ev_variable(&mut self) -> Result<(), String> {
        let var = match self.exp.clone() {
            Some(Expr::Symbol(s)) => s,
            _ => return Err("심볼이 필요함 (Expected symbol)".to_string()),
        };

        self.val = Some(
            self.env
                .lookup(&var)
                .cloned()
                .ok_or_else(|| format!("바인딩되지 않은 변수: {} (Unbound variable)", var))?,
        );
        self.current_label = self.continue_reg;
        Ok(())
    }

    /// 인용 표현식 평가 (Evaluate quoted expression)
    fn ev_quoted(&mut self) -> Result<(), String> {
        let quoted = match self.exp.clone() {
            Some(Expr::Quote(inner)) => *inner,
            _ => return Err("인용이 필요함 (Expected quote)".to_string()),
        };

        self.val = Some(self.expr_to_value(quoted));
        self.current_label = self.continue_reg;
        Ok(())
    }

    /// 표현식을 값으로 변환 (인용 데이터용)
    /// (Convert expression to value (for quoted data))
    fn expr_to_value(&self, expr: Expr) -> Value {
        match expr {
            Expr::Number(n) => Value::Number(n),
            Expr::Bool(b) => Value::Bool(b),
            Expr::String(s) => Value::String(s),
            Expr::Symbol(s) => Value::Symbol(s),
            Expr::Nil => Value::Nil,
            Expr::Pair(car, cdr) => Value::Pair(
                Box::new(self.expr_to_value(*car)),
                Box::new(self.expr_to_value(*cdr)),
            ),
            _ => Value::Symbol(format!("{:?}", expr)),
        }
    }

    /// 람다 표현식 평가 (Evaluate lambda expression)
    fn ev_lambda(&mut self) -> Result<(), String> {
        let (params, body) = match self.exp.clone() {
            Some(Expr::Lambda { params, body }) => (params, body),
            _ => return Err("람다가 필요함 (Expected lambda)".to_string()),
        };

        self.val = Some(Value::Procedure {
            params,
            body,
            env: self.env.clone(), // 구조적 공유로 O(1) 복제 (O(1) clone due to structural sharing)
            self_name: None,
        });
        self.current_label = self.continue_reg;
        Ok(())
    }

    /// if 표현식 평가 (Evaluate if expression)
    fn ev_if(&mut self) -> Result<(), String> {
        let (predicate, _, _) = match self.exp.clone() {
            Some(Expr::If {
                predicate,
                consequent,
                alternative,
            }) => (predicate, consequent, alternative),
            _ => return Err("if가 필요함 (Expected if)".to_string()),
        };

        self.save(StackFrame::Exp(self.exp.clone().unwrap()));
        self.save(StackFrame::Env(self.env.clone()));
        self.save(StackFrame::Continue(self.continue_reg));
        self.continue_reg = Label::EvIfDecide;
        self.exp = Some(*predicate);
        self.current_label = Label::EvalDispatch;
        Ok(())
    }

    fn ev_if_decide(&mut self) -> Result<(), String> {
        self.restore_continue()?;
        self.restore_env()?;
        self.restore_exp()?;

        let is_true = self
            .val
            .as_ref()
            .ok_or("값이 없음 (No value)")?
            .is_true();

        if is_true {
            self.current_label = Label::EvIfConsequent;
        } else {
            self.current_label = Label::EvIfAlternative;
        }
        Ok(())
    }

    fn ev_if_consequent(&mut self) -> Result<(), String> {
        let consequent = match self.exp.clone() {
            Some(Expr::If { consequent, .. }) => *consequent,
            _ => return Err("if가 필요함 (Expected if)".to_string()),
        };

        self.exp = Some(consequent);
        self.current_label = Label::EvalDispatch;
        Ok(())
    }

    fn ev_if_alternative(&mut self) -> Result<(), String> {
        let alternative = match self.exp.clone() {
            Some(Expr::If { alternative, .. }) => *alternative,
            _ => return Err("if가 필요함 (Expected if)".to_string()),
        };

        self.exp = Some(alternative);
        self.current_label = Label::EvalDispatch;
        Ok(())
    }

    /// 정의 평가 (Evaluate definition)
    fn ev_definition(&mut self) -> Result<(), String> {
        let (var, value) = match self.exp.clone() {
            Some(Expr::Definition { var, value }) => (var, value),
            _ => return Err("정의가 필요함 (Expected definition)".to_string()),
        };

        // 재귀 바인딩을 위한 람다 정의인지 확인
        // (Check if defining a lambda for recursive binding)
        let is_lambda = matches!(value.as_ref(), Expr::Lambda { .. });
        let var_name = var.clone();

        self.unev = vec![Expr::Symbol(var)];
        self.save(StackFrame::Unev(self.unev.clone()));

        // 자기 이름 바인딩을 위해 람다 정의 여부 저장
        // (Store whether this is a lambda definition for self-name binding)
        if is_lambda {
            // 람다 정의는 ev_definition_1에서 self-name 처리
            // (For lambda definitions, we handle self-name in ev_definition_1)
            self.save(StackFrame::Exp(Expr::Symbol(var_name)));
        } else {
            self.save(StackFrame::Exp(Expr::Nil));
        }

        self.exp = Some(*value);
        self.save(StackFrame::Env(self.env.clone()));
        self.save(StackFrame::Continue(self.continue_reg));
        self.continue_reg = Label::EvDefinition1;
        self.current_label = Label::EvalDispatch;
        Ok(())
    }

    fn ev_definition_1(&mut self) -> Result<(), String> {
        self.restore_continue()?;
        self.restore_env()?;
        self.restore_exp()?; // 변수 이름 또는 Nil (The var name or Nil)

        // 재귀를 위해 self_name 설정이 필요한지 확인
        // (Check if we need to set self_name for recursion)
        let self_name = match self.exp.as_ref() {
            Some(Expr::Symbol(s)) => Some(s.clone()),
            _ => None,
        };

        self.restore_unev()?;

        let var = match &self.unev[0] {
            Expr::Symbol(s) => s.clone(),
            _ => return Err("심볼이 필요함 (Expected symbol)".to_string()),
        };

        let mut val = self
            .val
            .clone()
            .ok_or("값이 없음 (No value)")?;

        // 프로시저라면 재귀 바인딩을 위해 self_name 설정
        // (If this is a procedure, set its self_name for recursive binding)
        if let Some(name) = self_name
            && let Value::Procedure {
                params, body, env, ..
            } = val
        {
            val = Value::Procedure {
                params,
                body,
                env,
                self_name: Some(name),
            };
        }

        // 현재 환경에 정의 (영속 업데이트)
        // (Define in current environment (persistent update))
        self.env = self.env.define(var, val);

        self.val = Some(Value::Ok);
        self.current_label = self.continue_reg;
        Ok(())
    }

    /// begin 표현식 평가 (Evaluate begin expression)
    fn ev_begin(&mut self) -> Result<(), String> {
        let exprs = match self.exp.clone() {
            Some(Expr::Begin(exprs)) => exprs,
            _ => return Err("begin이 필요함 (Expected begin)".to_string()),
        };

        self.unev = exprs;
        self.save(StackFrame::Continue(self.continue_reg));
        self.current_label = Label::EvSequence;
        Ok(())
    }

    /// 프로시저 적용 평가 (Evaluate procedure application)
    fn ev_application(&mut self) -> Result<(), String> {
        let (operator, operands) = match self.exp.clone() {
            Some(Expr::Application { operator, operands }) => (operator, operands),
            _ => return Err("적용이 필요함 (Expected application)".to_string()),
        };

        self.save(StackFrame::Continue(self.continue_reg));
        self.save(StackFrame::Env(self.env.clone()));
        self.unev = operands;
        self.save(StackFrame::Unev(self.unev.clone()));
        self.exp = Some(*operator);
        self.continue_reg = Label::EvApplDidOperator;
        self.current_label = Label::EvalDispatch;
        Ok(())
    }

    fn ev_appl_did_operator(&mut self) -> Result<(), String> {
        self.restore_unev()?;
        self.restore_env()?;
        self.argl = Vec::new();
        self.proc = self.val.clone();

        if self.unev.is_empty() {
            self.current_label = Label::ApplyDispatch;
        } else {
            self.save(StackFrame::Proc(self.proc.clone().unwrap()));
            self.current_label = Label::EvApplOperandLoop;
        }
        Ok(())
    }

    fn ev_appl_operand_loop(&mut self) -> Result<(), String> {
        self.save(StackFrame::Argl(self.argl.clone()));
        self.exp = Some(self.unev[0].clone());

        if self.unev.len() == 1 {
            self.current_label = Label::EvApplLastArg;
        } else {
            self.save(StackFrame::Env(self.env.clone()));
            self.save(StackFrame::Unev(self.unev.clone()));
            self.continue_reg = Label::EvApplAccumulateArg;
            self.current_label = Label::EvalDispatch;
        }
        Ok(())
    }

    fn ev_appl_accumulate_arg(&mut self) -> Result<(), String> {
        self.restore_unev()?;
        self.restore_env()?;
        self.restore_argl()?;

        self.argl
            .push(self.val.clone().ok_or("값이 없음 (No value)")?);
        self.unev = self.unev[1..].to_vec();
        self.current_label = Label::EvApplOperandLoop;
        Ok(())
    }

    fn ev_appl_last_arg(&mut self) -> Result<(), String> {
        self.continue_reg = Label::EvApplAccumLastArg;
        self.current_label = Label::EvalDispatch;
        Ok(())
    }

    fn ev_appl_accum_last_arg(&mut self) -> Result<(), String> {
        self.restore_argl()?;
        self.argl
            .push(self.val.clone().ok_or("값이 없음 (No value)")?);
        self.restore_proc()?;
        self.current_label = Label::ApplyDispatch;
        Ok(())
    }

    /// 인자를 프로시저에 적용 (Apply procedure to arguments)
    fn apply_dispatch(&mut self) -> Result<(), String> {
        let proc = self
            .proc
            .clone()
            .ok_or("프로시저가 없음 (No procedure)")?;

        match proc {
            Value::Primitive(_) => {
                self.current_label = Label::PrimitiveApply;
            }
            Value::Procedure { .. } => {
                self.current_label = Label::CompoundApply;
            }
            _ => {
                self.error = Some(format!(
                    "프로시저가 아님: {:?} (Not a procedure)",
                    proc
                ));
                self.current_label = Label::Error;
            }
        }
        Ok(())
    }

    fn primitive_apply(&mut self) -> Result<(), String> {
        let prim_name = match self.proc.clone() {
            Some(Value::Primitive(name)) => name,
            _ => return Err("기본 프로시저가 필요함 (Expected primitive)".to_string()),
        };

        self.val = Some(self.apply_primitive(&prim_name, &self.argl)?);
        self.restore_continue()?;
        self.current_label = self.continue_reg;
        Ok(())
    }

    fn compound_apply(&mut self) -> Result<(), String> {
        let (params, body, proc_env, self_name) = match self.proc.clone() {
            Some(Value::Procedure {
                params,
                body,
                env,
                self_name,
            }) => (params, body, env, self_name),
            _ => return Err("복합 프로시저가 필요함 (Expected compound procedure)".to_string()),
        };

        if params.len() != self.argl.len() {
            return Err(format!(
                "인자 개수 불일치: 예상 {}, 실제 {} (Argument count mismatch: expected {}, got {})",
                params.len(),
                self.argl.len(),
                params.len(),
                self.argl.len()
            ));
        }

        // 프로시저가 캡처한 환경에서 시작
        // (Start with the procedure's captured environment)
        let mut new_env = proc_env;

        // 재귀 호출을 위한 self-name 바인딩
        // (Bind self-name for recursive calls)
        if let Some(name) = self_name {
            new_env = new_env.define(name, self.proc.clone().unwrap());
        }

        // 매개변수 바인딩으로 확장
        // (Extend with parameter bindings)
        let bindings: Vec<(String, Value)> =
            params.into_iter().zip(self.argl.iter().cloned()).collect();
        new_env = new_env.extend(bindings);

        self.env = new_env;
        self.unev = body;
        self.current_label = Label::EvSequence;
        Ok(())
    }

    /// 꼬리 재귀 최적화를 포함한 시퀀스 평가
    /// (Evaluate sequence with tail recursion optimization)
    fn ev_sequence(&mut self) -> Result<(), String> {
        if self.unev.is_empty() {
            return Err("빈 시퀀스 (Empty sequence)".to_string());
        }

        self.exp = Some(self.unev[0].clone());

        if self.unev.len() == 1 {
            // 마지막 표현식: 꼬리 호출 최적화
            // (Last expression: tail call optimization)
            self.current_label = Label::EvSequenceLastExp;
        } else {
            self.save(StackFrame::Unev(self.unev.clone()));
            self.save(StackFrame::Env(self.env.clone()));
            self.continue_reg = Label::EvSequenceContinue;
            self.current_label = Label::EvalDispatch;
        }
        Ok(())
    }

    fn ev_sequence_continue(&mut self) -> Result<(), String> {
        // 저장된 env를 스택에서 꺼내기 전에 현재 환경을 보존
        // (정의로 업데이트되었을 수 있음)
        // (Preserve the current environment (which may have been updated by a definition)
        // before popping the saved env off the stack.)
        let current_env = self.env.clone();
        self.restore_env()?; // 저장된 env를 스택에서 꺼냄 (폐기)
        self.env = current_env; // 정의를 포함한 현재 env 복원

        self.restore_unev()?;
        self.unev = self.unev[1..].to_vec();
        self.current_label = Label::EvSequence;
        Ok(())
    }

    fn ev_sequence_last_exp(&mut self) -> Result<(), String> {
        // 꼬리 호출 최적화: eval 전에 continue 복원
        // (Tail call optimization: restore continue before eval)
        self.restore_continue()?;
        self.current_label = Label::EvalDispatch;
        Ok(())
    }

    /// 기본 프로시저 적용 (Apply primitive procedure)
    fn apply_primitive(&self, name: &str, args: &[Value]) -> Result<Value, String> {
        match name {
            "+" => {
                let sum = args.iter().try_fold(0i64, |acc, v| {
                    if let Value::Number(n) = v {
                        Ok(acc + n)
                    } else {
                        Err("숫자가 아닌 인자: + (Non-numeric argument to +)".to_string())
                    }
                })?;
                Ok(Value::Number(sum))
            }
            "-" => {
                if args.is_empty() {
                    return Err("-는 최소 한 개의 인자가 필요함 (- requires at least one argument)".to_string());
                }
                let first = match &args[0] {
                    Value::Number(n) => *n,
                    _ => return Err("숫자가 아닌 인자: - (Non-numeric argument to -)".to_string()),
                };
                if args.len() == 1 {
                    Ok(Value::Number(-first))
                } else {
                    let rest_sum = args[1..].iter().try_fold(0i64, |acc, v| {
                        if let Value::Number(n) = v {
                            Ok(acc + n)
                        } else {
                            Err("숫자가 아닌 인자: - (Non-numeric argument to -)".to_string())
                        }
                    })?;
                    Ok(Value::Number(first - rest_sum))
                }
            }
            "*" => {
                let product = args.iter().try_fold(1i64, |acc, v| {
                    if let Value::Number(n) = v {
                        Ok(acc * n)
                    } else {
                        Err("숫자가 아닌 인자: * (Non-numeric argument to *)".to_string())
                    }
                })?;
                Ok(Value::Number(product))
            }
            "/" => {
                if args.len() < 2 {
                    return Err("/는 최소 두 개의 인자가 필요함 (/ requires at least two arguments)".to_string());
                }
                let first = match &args[0] {
                    Value::Number(n) => *n,
                    _ => return Err("숫자가 아닌 인자: / (Non-numeric argument to /)".to_string()),
                };
                let rest_product = args[1..].iter().try_fold(1i64, |acc, v| {
                    if let Value::Number(n) = v {
                        Ok(acc * n)
                    } else {
                        Err("숫자가 아닌 인자: / (Non-numeric argument to /)".to_string())
                    }
                })?;
                if rest_product == 0 {
                    return Err("0으로 나눔 (Division by zero)".to_string());
                }
                Ok(Value::Number(first / rest_product))
            }
            "=" => {
                if args.len() != 2 {
                    return Err("=는 정확히 두 개의 인자가 필요함 (= requires exactly two arguments)".to_string());
                }
                Ok(Value::Bool(args[0] == args[1]))
            }
            "<" => {
                if args.len() != 2 {
                    return Err("<는 정확히 두 개의 인자가 필요함 (< requires exactly two arguments)".to_string());
                }
                match (&args[0], &args[1]) {
                    (Value::Number(a), Value::Number(b)) => Ok(Value::Bool(a < b)),
                    _ => Err("숫자가 아닌 인자들: < (Non-numeric arguments to <)".to_string()),
                }
            }
            ">" => {
                if args.len() != 2 {
                    return Err(">는 정확히 두 개의 인자가 필요함 (> requires exactly two arguments)".to_string());
                }
                match (&args[0], &args[1]) {
                    (Value::Number(a), Value::Number(b)) => Ok(Value::Bool(a > b)),
                    _ => Err("숫자가 아닌 인자들: > (Non-numeric arguments to >)".to_string()),
                }
            }
            "<=" => {
                if args.len() != 2 {
                    return Err("<=는 정확히 두 개의 인자가 필요함 (<= requires exactly two arguments)".to_string());
                }
                match (&args[0], &args[1]) {
                    (Value::Number(a), Value::Number(b)) => Ok(Value::Bool(a <= b)),
                    _ => Err("숫자가 아닌 인자들: <= (Non-numeric arguments to <=)".to_string()),
                }
            }
            ">=" => {
                if args.len() != 2 {
                    return Err(">=는 정확히 두 개의 인자가 필요함 (>= requires exactly two arguments)".to_string());
                }
                match (&args[0], &args[1]) {
                    (Value::Number(a), Value::Number(b)) => Ok(Value::Bool(a >= b)),
                    _ => Err("숫자가 아닌 인자들: >= (Non-numeric arguments to >=)".to_string()),
                }
            }
            "쌍(cons)" => {
                if args.len() != 2 {
                    return Err("쌍(cons)는 정확히 두 개의 인자가 필요함 (cons requires exactly two arguments)".to_string());
                }
                Ok(Value::Pair(
                    Box::new(args[0].clone()),
                    Box::new(args[1].clone()),
                ))
            }
            "머리(car)" => {
                if args.len() != 1 {
                    return Err("머리(car)는 정확히 한 개의 인자가 필요함 (car requires exactly one argument)".to_string());
                }
                match &args[0] {
                    Value::Pair(car, _) => Ok(*car.clone()),
                    _ => Err("머리(car)는 쌍이 필요함 (car requires a pair)".to_string()),
                }
            }
            "꼬리(cdr)" => {
                if args.len() != 1 {
                    return Err("꼬리(cdr)는 정확히 한 개의 인자가 필요함 (cdr requires exactly one argument)".to_string());
                }
                match &args[0] {
                    Value::Pair(_, cdr) => Ok(*cdr.clone()),
                    _ => Err("꼬리(cdr)는 쌍이 필요함 (cdr requires a pair)".to_string()),
                }
            }
            "빈리스트?(null?)" => {
                if args.len() != 1 {
                    return Err("빈리스트?(null?)는 정확히 한 개의 인자가 필요함 (null? requires exactly one argument)".to_string());
                }
                Ok(Value::Bool(matches!(args[0], Value::Nil)))
            }
            "쌍?(pair?)" => {
                if args.len() != 1 {
                    return Err("쌍?(pair?)는 정확히 한 개의 인자가 필요함 (pair? requires exactly one argument)".to_string());
                }
                Ok(Value::Bool(matches!(args[0], Value::Pair(_, _))))
            }
            "리스트(list)" => {
                let mut result = Value::Nil;
                for arg in args.iter().rev() {
                    result = Value::Pair(Box::new(arg.clone()), Box::new(result));
                }
                Ok(result)
            }
            "표시(display)" => {
                if args.len() != 1 {
                    return Err("표시(display)는 정확히 한 개의 인자가 필요함 (display requires exactly one argument)".to_string());
                }
                print!("{}", args[0]);
                Ok(Value::Ok)
            }
            "새줄(newline)" => {
                if !args.is_empty() {
                    return Err("새줄(newline)은 인자가 필요 없음 (newline requires no arguments)".to_string());
                }
                println!();
                Ok(Value::Ok)
            }
            _ => Err(format!(
                "알 수 없는 기본 프로시저: {} (Unknown primitive)",
                name
            )),
        }
    }

    pub fn reset_metrics(&mut self) {
        self.total_pushes = 0;
        self.max_depth = 0;
    }

    /// 전역 환경에 변수 정의 (설정용)
    /// (Define a variable in the global environment (for setup))
    pub fn define(&mut self, var: &str, val: Value) {
        self.env = self.env.define(var.to_string(), val);
    }
}

impl Default for EvaluatorMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_evaluating() {
        let mut machine = EvaluatorMachine::new();

        let result = machine.eval(Expr::Number(42)).unwrap();
        assert_eq!(result, Value::Number(42));

        let result = machine.eval(Expr::Bool(true)).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_variable_lookup() {
        let mut machine = EvaluatorMachine::new();
        machine.define("x", Value::Number(10));

        let result = machine.eval(Expr::Symbol("x".to_string())).unwrap();
        assert_eq!(result, Value::Number(10));
    }

    #[test]
    fn test_quoted_expression() {
        let mut machine = EvaluatorMachine::new();

        let result = machine
            .eval(Expr::Quote(Box::new(Expr::Symbol(
                "안녕 (hello)".to_string(),
            ))))
            .unwrap();

        assert_eq!(result, Value::Symbol("안녕 (hello)".to_string()));
    }

    #[test]
    fn test_lambda_creation() {
        let mut machine = EvaluatorMachine::new();

        let lambda = Expr::Lambda {
            params: vec!["x".to_string()],
            body: vec![Expr::Symbol("x".to_string())],
        };

        let result = machine.eval(lambda).unwrap();
        assert!(matches!(result, Value::Procedure { .. }));
    }

    #[test]
    fn test_if_expression() {
        let mut machine = EvaluatorMachine::new();

        // (if true 1 2) (참/거짓은 불리언)
        let if_expr = Expr::If {
            predicate: Box::new(Expr::Bool(true)),
            consequent: Box::new(Expr::Number(1)),
            alternative: Box::new(Expr::Number(2)),
        };

        let result = machine.eval(if_expr).unwrap();
        assert_eq!(result, Value::Number(1));

        // (if false 1 2)
        let if_expr = Expr::If {
            predicate: Box::new(Expr::Bool(false)),
            consequent: Box::new(Expr::Number(1)),
            alternative: Box::new(Expr::Number(2)),
        };

        let result = machine.eval(if_expr).unwrap();
        assert_eq!(result, Value::Number(2));
    }

    #[test]
    fn test_definition() {
        let mut machine = EvaluatorMachine::new();

        let def = Expr::Definition {
            var: "x".to_string(),
            value: Box::new(Expr::Number(42)),
        };

        machine.eval(def).unwrap();

        let result = machine.eval(Expr::Symbol("x".to_string())).unwrap();
        assert_eq!(result, Value::Number(42));
    }

    #[test]
    fn test_begin_sequence() {
        let mut machine = EvaluatorMachine::new();

        let begin = Expr::Begin(vec![Expr::Number(1), Expr::Number(2), Expr::Number(3)]);

        let result = machine.eval(begin).unwrap();
        assert_eq!(result, Value::Number(3));
    }

    #[test]
    fn test_primitive_application() {
        let mut machine = EvaluatorMachine::new();

        // (+ 1 2 3)
        let app = Expr::Application {
            operator: Box::new(Expr::Symbol("+".to_string())),
            operands: vec![Expr::Number(1), Expr::Number(2), Expr::Number(3)],
        };

        let result = machine.eval(app).unwrap();
        assert_eq!(result, Value::Number(6));
    }

    #[test]
    fn test_compound_procedure_application() {
        let mut machine = EvaluatorMachine::new();

        // (define square (lambda (x) (* x x)))
        let lambda = Expr::Lambda {
            params: vec!["x".to_string()],
            body: vec![Expr::Application {
                operator: Box::new(Expr::Symbol("*".to_string())),
                operands: vec![Expr::Symbol("x".to_string()), Expr::Symbol("x".to_string())],
            }],
        };

        machine
            .eval(Expr::Definition {
                var: "square".to_string(),
                value: Box::new(lambda),
            })
            .unwrap();

        // (square 5)
        let app = Expr::Application {
            operator: Box::new(Expr::Symbol("square".to_string())),
            operands: vec![Expr::Number(5)],
        };

        let result = machine.eval(app).unwrap();
        assert_eq!(result, Value::Number(25));
    }

    #[test]
    fn test_tail_recursion_factorial() {
        let mut machine = EvaluatorMachine::new();

        // 내부 iter 함수를 가진 반복 factorial 정의
        // (Define iterative factorial with inner iter function)
        let iter_lambda = Expr::Lambda {
            params: vec!["product".to_string(), "counter".to_string()],
            body: vec![Expr::If {
                predicate: Box::new(Expr::Application {
                    operator: Box::new(Expr::Symbol(">".to_string())),
                    operands: vec![
                        Expr::Symbol("counter".to_string()),
                        Expr::Symbol("n".to_string()),
                    ],
                }),
                consequent: Box::new(Expr::Symbol("product".to_string())),
                alternative: Box::new(Expr::Application {
                    operator: Box::new(Expr::Symbol("iter".to_string())),
                    operands: vec![
                        Expr::Application {
                            operator: Box::new(Expr::Symbol("*".to_string())),
                            operands: vec![
                                Expr::Symbol("counter".to_string()),
                                Expr::Symbol("product".to_string()),
                            ],
                        },
                        Expr::Application {
                            operator: Box::new(Expr::Symbol("+".to_string())),
                            operands: vec![Expr::Symbol("counter".to_string()), Expr::Number(1)],
                        },
                    ],
                }),
            }],
        };

        let factorial_lambda = Expr::Lambda {
            params: vec!["n".to_string()],
            body: vec![
                Expr::Definition {
                    var: "iter".to_string(),
                    value: Box::new(iter_lambda),
                },
                Expr::Application {
                    operator: Box::new(Expr::Symbol("iter".to_string())),
                    operands: vec![Expr::Number(1), Expr::Number(1)],
                },
            ],
        };

        machine
            .eval(Expr::Definition {
                var: "factorial".to_string(),
                value: Box::new(factorial_lambda),
            })
            .unwrap();

        // factorial(5) = 120 테스트 (Test factorial(5) = 120)
        machine.reset_metrics();
        let result = machine
            .eval(Expr::Application {
                operator: Box::new(Expr::Symbol("factorial".to_string())),
                operands: vec![Expr::Number(5)],
            })
            .unwrap();

        assert_eq!(result, Value::Number(120));

        // 더 큰 n 테스트 (Test with larger n)
        machine.reset_metrics();
        let result = machine
            .eval(Expr::Application {
                operator: Box::new(Expr::Symbol("factorial".to_string())),
                operands: vec![Expr::Number(10)],
            })
            .unwrap();

        assert_eq!(result, Value::Number(3628800));
    }

    #[test]
    fn test_recursive_factorial() {
        let mut machine = EvaluatorMachine::new();

        // 재귀 factorial
        // (Recursive factorial)
        let factorial_lambda = Expr::Lambda {
            params: vec!["n".to_string()],
            body: vec![Expr::If {
                predicate: Box::new(Expr::Application {
                    operator: Box::new(Expr::Symbol("=".to_string())),
                    operands: vec![Expr::Symbol("n".to_string()), Expr::Number(1)],
                }),
                consequent: Box::new(Expr::Number(1)),
                alternative: Box::new(Expr::Application {
                    operator: Box::new(Expr::Symbol("*".to_string())),
                    operands: vec![
                        Expr::Application {
                            operator: Box::new(Expr::Symbol("factorial".to_string())),
                            operands: vec![Expr::Application {
                                operator: Box::new(Expr::Symbol("-".to_string())),
                                operands: vec![Expr::Symbol("n".to_string()), Expr::Number(1)],
                            }],
                        },
                        Expr::Symbol("n".to_string()),
                    ],
                }),
            }],
        };

        machine
            .eval(Expr::Definition {
                var: "factorial".to_string(),
                value: Box::new(factorial_lambda),
            })
            .unwrap();

        machine.reset_metrics();
        let result = machine
            .eval(Expr::Application {
                operator: Box::new(Expr::Symbol("factorial".to_string())),
                operands: vec![Expr::Number(5)],
            })
            .unwrap();

        assert_eq!(result, Value::Number(120));
    }

    #[test]
    fn test_list_operations() {
        let mut machine = EvaluatorMachine::new();

        // (쌍(cons) 1 (쌍(cons) 2 (쌍(cons) 3 '())))
        let list_expr = Expr::Application {
            operator: Box::new(Expr::Symbol("쌍(cons)".to_string())),
            operands: vec![
                Expr::Number(1),
                Expr::Application {
                    operator: Box::new(Expr::Symbol("쌍(cons)".to_string())),
                    operands: vec![
                        Expr::Number(2),
                        Expr::Application {
                            operator: Box::new(Expr::Symbol("쌍(cons)".to_string())),
                            operands: vec![Expr::Number(3), Expr::Nil],
                        },
                    ],
                },
            ],
        };

        let result = machine.eval(list_expr).unwrap();

        // 구조 검증 (Verify structure)
        if let Value::Pair(car1, cdr1) = result {
            assert_eq!(*car1, Value::Number(1));
            if let Value::Pair(car2, cdr2) = *cdr1 {
                assert_eq!(*car2, Value::Number(2));
                if let Value::Pair(car3, cdr3) = *cdr2 {
                    assert_eq!(*car3, Value::Number(3));
                    assert_eq!(*cdr3, Value::Nil);
                } else {
                    panic!("쌍이 필요함 (Expected pair)");
                }
            } else {
                panic!("쌍이 필요함 (Expected pair)");
            }
        } else {
            panic!("쌍이 필요함 (Expected pair)");
        }
    }

    #[test]
    fn test_closures() {
        let mut machine = EvaluatorMachine::new();

        // (define make-adder (lambda (x) (lambda (y) (+ x y))))
        let inner_lambda = Expr::Lambda {
            params: vec!["y".to_string()],
            body: vec![Expr::Application {
                operator: Box::new(Expr::Symbol("+".to_string())),
                operands: vec![Expr::Symbol("x".to_string()), Expr::Symbol("y".to_string())],
            }],
        };

        let make_adder = Expr::Lambda {
            params: vec!["x".to_string()],
            body: vec![inner_lambda],
        };

        machine
            .eval(Expr::Definition {
                var: "make-adder".to_string(),
                value: Box::new(make_adder),
            })
            .unwrap();

        // (define add5 (make-adder 5))
        machine
            .eval(Expr::Definition {
                var: "add5".to_string(),
                value: Box::new(Expr::Application {
                    operator: Box::new(Expr::Symbol("make-adder".to_string())),
                    operands: vec![Expr::Number(5)],
                }),
            })
            .unwrap();

        // (add5 10) 결과는 15여야 함 ((add5 10) should be 15)
        let result = machine
            .eval(Expr::Application {
                operator: Box::new(Expr::Symbol("add5".to_string())),
                operands: vec![Expr::Number(10)],
            })
            .unwrap();

        assert_eq!(result, Value::Number(15));
    }
}
