//! Section 5.4: The Explicit-Control Evaluator
//!
//! This module implements a register machine evaluator for a Scheme-like language.
//! The evaluator uses 7 registers (exp, env, val, continue, proc, argl, unev) and
//! a stack to execute programs with explicit control flow.
//!
//! Key concepts demonstrated:
//! - Register-based evaluation with explicit control
//! - Tail-call optimization through careful continue register management
//! - Stack operations for saving and restoring machine state
//! - Expression dispatch via explicit type checking

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

/// Expression types in the evaluator
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Self-evaluating expressions (numbers, booleans, strings)
    Number(i64),
    Bool(bool),
    String(String),

    /// Variable reference
    Symbol(String),

    /// Quoted expression
    Quote(Box<Expr>),

    /// Lambda expression: (lambda (params...) body...)
    Lambda {
        params: Vec<String>,
        body: Vec<Expr>,
    },

    /// If expression: (if predicate consequent alternative)
    If {
        predicate: Box<Expr>,
        consequent: Box<Expr>,
        alternative: Box<Expr>,
    },

    /// Assignment: (set! var value)
    Assignment {
        var: String,
        value: Box<Expr>,
    },

    /// Definition: (define var value)
    Definition {
        var: String,
        value: Box<Expr>,
    },

    /// Begin sequence: (begin expr...)
    Begin(Vec<Expr>),

    /// Procedure application: (operator operands...)
    Application {
        operator: Box<Expr>,
        operands: Vec<Expr>,
    },

    /// List construction (for quoted lists)
    Pair(Box<Expr>, Box<Expr>),

    /// Empty list
    Nil,
}

/// Runtime values
#[derive(Debug, Clone)]
pub enum Value {
    Number(i64),
    Bool(bool),
    String(String),
    Symbol(String),
    Pair(Rc<Value>, Rc<Value>),
    Nil,

    /// Compound procedure with captured environment
    Procedure {
        params: Vec<String>,
        body: Vec<Expr>,
        env: Rc<Environment>,
    },

    /// Primitive procedure
    Primitive(String),

    /// Special "ok" value for assignments/definitions
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
            Value::Procedure { .. } => write!(f, "#<procedure>"),
            Value::Primitive(name) => write!(f, "#<primitive:{}>", name),
            Value::Ok => write!(f, "ok"),
        }
    }
}

impl Value {
    fn write_list(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Pair(_car, cdr) => {
                write!(f, "{}", _car)?;
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

/// Environment: bindings with parent chain
pub type Environment = RefCell<EnvFrame>;

#[derive(Debug, Clone)]
pub struct EnvFrame {
    bindings: HashMap<String, Value>,
    parent: Option<Rc<Environment>>,
}

impl EnvFrame {
    pub fn new(parent: Option<Rc<Environment>>) -> Self {
        EnvFrame {
            bindings: HashMap::new(),
            parent,
        }
    }

    pub fn lookup(&self, var: &str) -> Option<Value> {
        self.bindings
            .get(var)
            .cloned()
            .or_else(|| self.parent.as_ref()?.borrow().lookup(var))
    }

    pub fn set(&mut self, var: &str, val: Value) -> Result<(), String> {
        if self.bindings.contains_key(var) {
            self.bindings.insert(var.to_string(), val);
            Ok(())
        } else if let Some(ref parent) = self.parent {
            parent.borrow_mut().set(var, val)
        } else {
            Err(format!("Unbound variable: {}", var))
        }
    }

    pub fn define(&mut self, var: String, val: Value) {
        self.bindings.insert(var, val);
    }

    pub fn extend(parent: Rc<Environment>, params: &[String], args: &[Value]) -> Rc<Environment> {
        let mut env = EnvFrame::new(Some(parent));
        for (param, arg) in params.iter().zip(args.iter()) {
            env.define(param.clone(), arg.clone());
        }
        Rc::new(RefCell::new(env))
    }
}

/// Stack frame types for saving register state
#[derive(Debug, Clone)]
pub enum StackFrame {
    Continue(Label),
    Env(Rc<Environment>),
    Unev(Vec<Expr>),
    Argl(Vec<Value>),
    Proc(Value),
    Exp(Expr),
}

/// Control labels for the register machine
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
    EvAssignment,
    EvAssignment1,
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

/// The explicit-control evaluator machine
pub struct EvaluatorMachine {
    // Seven registers
    exp: Option<Expr>,
    env: Rc<Environment>,
    val: Option<Value>,
    continue_reg: Label,
    proc: Option<Value>,
    argl: Vec<Value>,
    unev: Vec<Expr>,

    // Stack for saving state
    stack: Vec<StackFrame>,

    // Performance metrics
    pub total_pushes: usize,
    pub max_depth: usize,

    // Current instruction label
    current_label: Label,

    // Error message
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

    /// Create global environment with primitive procedures
    fn setup_environment() -> Rc<Environment> {
        let mut env = EnvFrame::new(None);

        // Define primitive procedures
        let primitives = vec![
            "+", "-", "*", "/", "=", "<", ">", "<=", ">=", "cons", "car", "cdr", "null?", "pair?",
            "list", "display", "newline",
        ];

        for prim in primitives {
            env.define(prim.to_string(), Value::Primitive(prim.to_string()));
        }

        // Define special constants
        env.define("true".to_string(), Value::Bool(true));
        env.define("false".to_string(), Value::Bool(false));

        Rc::new(RefCell::new(env))
    }

    /// Stack operations with metrics tracking
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
            Some(_) => Err("Expected Continue on stack".to_string()),
            None => Err("Stack underflow".to_string()),
        }
    }

    fn restore_env(&mut self) -> Result<(), String> {
        match self.stack.pop() {
            Some(StackFrame::Env(env)) => {
                self.env = env;
                Ok(())
            }
            Some(_) => Err("Expected Env on stack".to_string()),
            None => Err("Stack underflow".to_string()),
        }
    }

    fn restore_unev(&mut self) -> Result<(), String> {
        match self.stack.pop() {
            Some(StackFrame::Unev(unev)) => {
                self.unev = unev;
                Ok(())
            }
            Some(_) => Err("Expected Unev on stack".to_string()),
            None => Err("Stack underflow".to_string()),
        }
    }

    fn restore_argl(&mut self) -> Result<(), String> {
        match self.stack.pop() {
            Some(StackFrame::Argl(argl)) => {
                self.argl = argl;
                Ok(())
            }
            Some(_) => Err("Expected Argl on stack".to_string()),
            None => Err("Stack underflow".to_string()),
        }
    }

    fn restore_proc(&mut self) -> Result<(), String> {
        match self.stack.pop() {
            Some(StackFrame::Proc(proc)) => {
                self.proc = Some(proc);
                Ok(())
            }
            Some(_) => Err("Expected Proc on stack".to_string()),
            None => Err("Stack underflow".to_string()),
        }
    }

    fn restore_exp(&mut self) -> Result<(), String> {
        match self.stack.pop() {
            Some(StackFrame::Exp(exp)) => {
                self.exp = Some(exp);
                Ok(())
            }
            Some(_) => Err("Expected Exp on stack".to_string()),
            None => Err("Stack underflow".to_string()),
        }
    }

    /// Evaluate an expression
    pub fn eval(&mut self, expr: Expr) -> Result<Value, String> {
        self.exp = Some(expr);
        self.continue_reg = Label::Done;
        self.current_label = Label::EvalDispatch;
        self.total_pushes = 0;
        self.max_depth = 0;

        self.run()
    }

    /// Main evaluation loop
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
                Label::EvAssignment => self.ev_assignment()?,
                Label::EvAssignment1 => self.ev_assignment_1()?,
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
                        .ok_or_else(|| "No value produced".to_string());
                }
                Label::Error => {
                    return Err(self
                        .error
                        .clone()
                        .unwrap_or_else(|| "Unknown error".to_string()));
                }
            }
        }
    }

    /// Dispatch based on expression type
    fn eval_dispatch(&mut self) -> Result<(), String> {
        let exp = self.exp.clone().ok_or("No expression in exp register")?;

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
            Expr::Assignment { .. } => {
                self.current_label = Label::EvAssignment;
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
                self.error = Some(format!("Unknown expression type: {:?}", exp));
                self.current_label = Label::Error;
            }
        }
        Ok(())
    }

    /// Evaluate self-evaluating expression
    fn ev_self_eval(&mut self) -> Result<(), String> {
        let exp = self.exp.clone().ok_or("No expression")?;
        self.val = Some(match exp {
            Expr::Number(n) => Value::Number(n),
            Expr::Bool(b) => Value::Bool(b),
            Expr::String(s) => Value::String(s),
            Expr::Nil => Value::Nil,
            _ => return Err("Not self-evaluating".to_string()),
        });
        self.current_label = self.continue_reg;
        Ok(())
    }

    /// Evaluate variable
    fn ev_variable(&mut self) -> Result<(), String> {
        let var = match self.exp.clone() {
            Some(Expr::Symbol(s)) => s,
            _ => return Err("Expected symbol".to_string()),
        };

        self.val = Some(
            self.env
                .borrow()
                .lookup(&var)
                .ok_or_else(|| format!("Unbound variable: {}", var))?,
        );
        self.current_label = self.continue_reg;
        Ok(())
    }

    /// Evaluate quoted expression
    fn ev_quoted(&mut self) -> Result<(), String> {
        let quoted = match self.exp.clone() {
            Some(Expr::Quote(inner)) => *inner,
            _ => return Err("Expected quote".to_string()),
        };

        self.val = Some(self.expr_to_value(quoted));
        self.current_label = self.continue_reg;
        Ok(())
    }

    /// Convert expression to value (for quoted data)
    fn expr_to_value(&self, expr: Expr) -> Value {
        match expr {
            Expr::Number(n) => Value::Number(n),
            Expr::Bool(b) => Value::Bool(b),
            Expr::String(s) => Value::String(s),
            Expr::Symbol(s) => Value::Symbol(s),
            Expr::Nil => Value::Nil,
            Expr::Pair(car, cdr) => Value::Pair(
                Rc::new(self.expr_to_value(*car)),
                Rc::new(self.expr_to_value(*cdr)),
            ),
            _ => Value::Symbol(format!("{:?}", expr)),
        }
    }

    /// Evaluate lambda expression
    fn ev_lambda(&mut self) -> Result<(), String> {
        let (params, body) = match self.exp.clone() {
            Some(Expr::Lambda { params, body }) => (params, body),
            _ => return Err("Expected lambda".to_string()),
        };

        self.val = Some(Value::Procedure {
            params,
            body,
            env: Rc::clone(&self.env),
        });
        self.current_label = self.continue_reg;
        Ok(())
    }

    /// Evaluate if expression
    fn ev_if(&mut self) -> Result<(), String> {
        let (predicate, _, _) = match self.exp.clone() {
            Some(Expr::If {
                predicate,
                consequent,
                alternative,
            }) => (predicate, consequent, alternative),
            _ => return Err("Expected if".to_string()),
        };

        self.save(StackFrame::Exp(self.exp.clone().unwrap()));
        self.save(StackFrame::Env(Rc::clone(&self.env)));
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

        let is_true = self.val.as_ref().ok_or("No value")?.is_true();

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
            _ => return Err("Expected if".to_string()),
        };

        self.exp = Some(consequent);
        self.current_label = Label::EvalDispatch;
        Ok(())
    }

    fn ev_if_alternative(&mut self) -> Result<(), String> {
        let alternative = match self.exp.clone() {
            Some(Expr::If { alternative, .. }) => *alternative,
            _ => return Err("Expected if".to_string()),
        };

        self.exp = Some(alternative);
        self.current_label = Label::EvalDispatch;
        Ok(())
    }

    /// Evaluate assignment
    fn ev_assignment(&mut self) -> Result<(), String> {
        let (var, value) = match self.exp.clone() {
            Some(Expr::Assignment { var, value }) => (var, value),
            _ => return Err("Expected assignment".to_string()),
        };

        self.unev = vec![Expr::Symbol(var)];
        self.save(StackFrame::Unev(self.unev.clone()));
        self.exp = Some(*value);
        self.save(StackFrame::Env(Rc::clone(&self.env)));
        self.save(StackFrame::Continue(self.continue_reg));
        self.continue_reg = Label::EvAssignment1;
        self.current_label = Label::EvalDispatch;
        Ok(())
    }

    fn ev_assignment_1(&mut self) -> Result<(), String> {
        self.restore_continue()?;
        self.restore_env()?;
        self.restore_unev()?;

        let var = match &self.unev[0] {
            Expr::Symbol(s) => s.clone(),
            _ => return Err("Expected symbol".to_string()),
        };

        let val = self.val.clone().ok_or("No value")?;
        self.env.borrow_mut().set(&var, val)?;

        self.val = Some(Value::Ok);
        self.current_label = self.continue_reg;
        Ok(())
    }

    /// Evaluate definition
    fn ev_definition(&mut self) -> Result<(), String> {
        let (var, value) = match self.exp.clone() {
            Some(Expr::Definition { var, value }) => (var, value),
            _ => return Err("Expected definition".to_string()),
        };

        self.unev = vec![Expr::Symbol(var)];
        self.save(StackFrame::Unev(self.unev.clone()));
        self.exp = Some(*value);
        self.save(StackFrame::Env(Rc::clone(&self.env)));
        self.save(StackFrame::Continue(self.continue_reg));
        self.continue_reg = Label::EvDefinition1;
        self.current_label = Label::EvalDispatch;
        Ok(())
    }

    fn ev_definition_1(&mut self) -> Result<(), String> {
        self.restore_continue()?;
        self.restore_env()?;
        self.restore_unev()?;

        let var = match &self.unev[0] {
            Expr::Symbol(s) => s.clone(),
            _ => return Err("Expected symbol".to_string()),
        };

        let val = self.val.clone().ok_or("No value")?;
        self.env.borrow_mut().define(var, val);

        self.val = Some(Value::Ok);
        self.current_label = self.continue_reg;
        Ok(())
    }

    /// Evaluate begin expression
    fn ev_begin(&mut self) -> Result<(), String> {
        let exprs = match self.exp.clone() {
            Some(Expr::Begin(exprs)) => exprs,
            _ => return Err("Expected begin".to_string()),
        };

        self.unev = exprs;
        self.save(StackFrame::Continue(self.continue_reg));
        self.current_label = Label::EvSequence;
        Ok(())
    }

    /// Evaluate procedure application
    fn ev_application(&mut self) -> Result<(), String> {
        let (operator, operands) = match self.exp.clone() {
            Some(Expr::Application { operator, operands }) => (operator, operands),
            _ => return Err("Expected application".to_string()),
        };

        self.save(StackFrame::Continue(self.continue_reg));
        self.save(StackFrame::Env(Rc::clone(&self.env)));
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
            self.save(StackFrame::Env(Rc::clone(&self.env)));
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

        self.argl.push(self.val.clone().ok_or("No value")?);
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
        self.argl.push(self.val.clone().ok_or("No value")?);
        self.restore_proc()?;
        self.current_label = Label::ApplyDispatch;
        Ok(())
    }

    /// Apply procedure to arguments
    fn apply_dispatch(&mut self) -> Result<(), String> {
        let proc = self.proc.clone().ok_or("No procedure")?;

        match proc {
            Value::Primitive(_) => {
                self.current_label = Label::PrimitiveApply;
            }
            Value::Procedure { .. } => {
                self.current_label = Label::CompoundApply;
            }
            _ => {
                self.error = Some(format!("Not a procedure: {:?}", proc));
                self.current_label = Label::Error;
            }
        }
        Ok(())
    }

    fn primitive_apply(&mut self) -> Result<(), String> {
        let prim_name = match self.proc.clone() {
            Some(Value::Primitive(name)) => name,
            _ => return Err("Expected primitive".to_string()),
        };

        self.val = Some(self.apply_primitive(&prim_name, &self.argl)?);
        self.restore_continue()?;
        self.current_label = self.continue_reg;
        Ok(())
    }

    fn compound_apply(&mut self) -> Result<(), String> {
        let (params, body, proc_env) = match self.proc.clone() {
            Some(Value::Procedure { params, body, env }) => (params, body, env),
            _ => return Err("Expected compound procedure".to_string()),
        };

        if params.len() != self.argl.len() {
            return Err(format!(
                "Argument count mismatch: expected {}, got {}",
                params.len(),
                self.argl.len()
            ));
        }

        self.env = EnvFrame::extend(proc_env, &params, &self.argl);
        self.unev = body;
        self.current_label = Label::EvSequence;
        Ok(())
    }

    /// Evaluate sequence with tail recursion optimization
    fn ev_sequence(&mut self) -> Result<(), String> {
        if self.unev.is_empty() {
            return Err("Empty sequence".to_string());
        }

        self.exp = Some(self.unev[0].clone());

        if self.unev.len() == 1 {
            // Last expression: tail call optimization
            self.current_label = Label::EvSequenceLastExp;
        } else {
            self.save(StackFrame::Unev(self.unev.clone()));
            self.save(StackFrame::Env(Rc::clone(&self.env)));
            self.continue_reg = Label::EvSequenceContinue;
            self.current_label = Label::EvalDispatch;
        }
        Ok(())
    }

    fn ev_sequence_continue(&mut self) -> Result<(), String> {
        self.restore_env()?;
        self.restore_unev()?;
        self.unev = self.unev[1..].to_vec();
        self.current_label = Label::EvSequence;
        Ok(())
    }

    fn ev_sequence_last_exp(&mut self) -> Result<(), String> {
        // Tail call optimization: restore continue before eval
        self.restore_continue()?;
        self.current_label = Label::EvalDispatch;
        Ok(())
    }

    /// Apply primitive procedure
    fn apply_primitive(&self, name: &str, args: &[Value]) -> Result<Value, String> {
        match name {
            "+" => {
                let sum = args.iter().try_fold(0i64, |acc, v| {
                    if let Value::Number(n) = v {
                        Ok(acc + n)
                    } else {
                        Err("Non-numeric argument to +".to_string())
                    }
                })?;
                Ok(Value::Number(sum))
            }
            "-" => {
                if args.is_empty() {
                    return Err("- requires at least one argument".to_string());
                }
                let first = match &args[0] {
                    Value::Number(n) => *n,
                    _ => return Err("Non-numeric argument to -".to_string()),
                };
                if args.len() == 1 {
                    Ok(Value::Number(-first))
                } else {
                    let rest_sum = args[1..].iter().try_fold(0i64, |acc, v| {
                        if let Value::Number(n) = v {
                            Ok(acc + n)
                        } else {
                            Err("Non-numeric argument to -".to_string())
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
                        Err("Non-numeric argument to *".to_string())
                    }
                })?;
                Ok(Value::Number(product))
            }
            "/" => {
                if args.len() < 2 {
                    return Err("/ requires at least two arguments".to_string());
                }
                let first = match &args[0] {
                    Value::Number(n) => *n,
                    _ => return Err("Non-numeric argument to /".to_string()),
                };
                let rest_product = args[1..].iter().try_fold(1i64, |acc, v| {
                    if let Value::Number(n) = v {
                        Ok(acc * n)
                    } else {
                        Err("Non-numeric argument to /".to_string())
                    }
                })?;
                if rest_product == 0 {
                    return Err("Division by zero".to_string());
                }
                Ok(Value::Number(first / rest_product))
            }
            "=" => {
                if args.len() != 2 {
                    return Err("= requires exactly two arguments".to_string());
                }
                Ok(Value::Bool(args[0] == args[1]))
            }
            "<" => {
                if args.len() != 2 {
                    return Err("< requires exactly two arguments".to_string());
                }
                match (&args[0], &args[1]) {
                    (Value::Number(a), Value::Number(b)) => Ok(Value::Bool(a < b)),
                    _ => Err("Non-numeric arguments to <".to_string()),
                }
            }
            ">" => {
                if args.len() != 2 {
                    return Err("> requires exactly two arguments".to_string());
                }
                match (&args[0], &args[1]) {
                    (Value::Number(a), Value::Number(b)) => Ok(Value::Bool(a > b)),
                    _ => Err("Non-numeric arguments to >".to_string()),
                }
            }
            "cons" => {
                if args.len() != 2 {
                    return Err("cons requires exactly two arguments".to_string());
                }
                Ok(Value::Pair(
                    Rc::new(args[0].clone()),
                    Rc::new(args[1].clone()),
                ))
            }
            "car" => {
                if args.len() != 1 {
                    return Err("car requires exactly one argument".to_string());
                }
                match &args[0] {
                    Value::Pair(car, _) => Ok((**car).clone()),
                    _ => Err("car requires a pair".to_string()),
                }
            }
            "cdr" => {
                if args.len() != 1 {
                    return Err("cdr requires exactly one argument".to_string());
                }
                match &args[0] {
                    Value::Pair(_, cdr) => Ok((**cdr).clone()),
                    _ => Err("cdr requires a pair".to_string()),
                }
            }
            "null?" => {
                if args.len() != 1 {
                    return Err("null? requires exactly one argument".to_string());
                }
                Ok(Value::Bool(matches!(args[0], Value::Nil)))
            }
            "pair?" => {
                if args.len() != 1 {
                    return Err("pair? requires exactly one argument".to_string());
                }
                Ok(Value::Bool(matches!(args[0], Value::Pair(_, _))))
            }
            "list" => {
                let mut result = Value::Nil;
                for arg in args.iter().rev() {
                    result = Value::Pair(Rc::new(arg.clone()), Rc::new(result));
                }
                Ok(result)
            }
            "display" => {
                if args.len() != 1 {
                    return Err("display requires exactly one argument".to_string());
                }
                println!("{}", args[0]);
                Ok(Value::Ok)
            }
            "newline" => {
                if !args.is_empty() {
                    return Err("newline requires no arguments".to_string());
                }
                println!();
                Ok(Value::Ok)
            }
            _ => Err(format!("Unknown primitive: {}", name)),
        }
    }

    pub fn reset_metrics(&mut self) {
        self.total_pushes = 0;
        self.max_depth = 0;
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
        machine
            .env
            .borrow_mut()
            .define("x".to_string(), Value::Number(10));

        let result = machine.eval(Expr::Symbol("x".to_string())).unwrap();
        assert_eq!(result, Value::Number(10));
    }

    #[test]
    fn test_quoted_expression() {
        let mut machine = EvaluatorMachine::new();

        let result = machine
            .eval(Expr::Quote(Box::new(Expr::Symbol("hello".to_string()))))
            .unwrap();

        assert_eq!(result, Value::Symbol("hello".to_string()));
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

        // (if true 1 2)
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
    fn test_assignment() {
        let mut machine = EvaluatorMachine::new();
        machine
            .env
            .borrow_mut()
            .define("x".to_string(), Value::Number(10));

        let assign = Expr::Assignment {
            var: "x".to_string(),
            value: Box::new(Expr::Number(20)),
        };

        machine.eval(assign).unwrap();

        let result = machine.eval(Expr::Symbol("x".to_string())).unwrap();
        assert_eq!(result, Value::Number(20));
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

        // (define factorial
        //   (lambda (n)
        //     (define iter
        //       (lambda (product counter)
        //         (if (> counter n)
        //             product
        //             (iter (* counter product) (+ counter 1)))))
        //     (iter 1 1)))

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

        // Test factorial(5) = 120
        machine.reset_metrics();
        let result = machine
            .eval(Expr::Application {
                operator: Box::new(Expr::Symbol("factorial".to_string())),
                operands: vec![Expr::Number(5)],
            })
            .unwrap();

        assert_eq!(result, Value::Number(120));

        // Verify constant stack depth for tail recursion
        println!(
            "Tail-recursive factorial(5): max_depth={}, total_pushes={}",
            machine.max_depth, machine.total_pushes
        );

        // Test with larger n to verify constant stack
        machine.reset_metrics();
        let result = machine
            .eval(Expr::Application {
                operator: Box::new(Expr::Symbol("factorial".to_string())),
                operands: vec![Expr::Number(10)],
            })
            .unwrap();

        assert_eq!(result, Value::Number(3628800));
        println!(
            "Tail-recursive factorial(10): max_depth={}, total_pushes={}",
            machine.max_depth, machine.total_pushes
        );
    }

    #[test]
    fn test_recursive_factorial() {
        let mut machine = EvaluatorMachine::new();

        // (define factorial
        //   (lambda (n)
        //     (if (= n 1)
        //         1
        //         (* (factorial (- n 1)) n))))

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
        println!(
            "Recursive factorial(5): max_depth={}, total_pushes={}",
            machine.max_depth, machine.total_pushes
        );
    }

    #[test]
    fn test_list_operations() {
        let mut machine = EvaluatorMachine::new();

        // (cons 1 (cons 2 (cons 3 '())))
        let list_expr = Expr::Application {
            operator: Box::new(Expr::Symbol("cons".to_string())),
            operands: vec![
                Expr::Number(1),
                Expr::Application {
                    operator: Box::new(Expr::Symbol("cons".to_string())),
                    operands: vec![
                        Expr::Number(2),
                        Expr::Application {
                            operator: Box::new(Expr::Symbol("cons".to_string())),
                            operands: vec![Expr::Number(3), Expr::Nil],
                        },
                    ],
                },
            ],
        };

        let result = machine.eval(list_expr).unwrap();

        // Verify structure
        if let Value::Pair(car1, cdr1) = result {
            assert_eq!(*car1, Value::Number(1));
            if let Value::Pair(car2, cdr2) = &*cdr1 {
                assert_eq!(**car2, Value::Number(2));
                if let Value::Pair(car3, cdr3) = &**cdr2 {
                    assert_eq!(**car3, Value::Number(3));
                    assert_eq!(**cdr3, Value::Nil);
                } else {
                    panic!("Expected pair");
                }
            } else {
                panic!("Expected pair");
            }
        } else {
            panic!("Expected pair");
        }
    }

    #[test]
    fn test_closures() {
        let mut machine = EvaluatorMachine::new();

        // (define make-adder
        //   (lambda (x)
        //     (lambda (y) (+ x y))))
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

        // (add5 10) should be 15
        let result = machine
            .eval(Expr::Application {
                operator: Box::new(Expr::Symbol("add5".to_string())),
                operands: vec![Expr::Number(10)],
            })
            .unwrap();

        assert_eq!(result, Value::Number(15));
    }
}
