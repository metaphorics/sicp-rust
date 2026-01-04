//! SICP Section 5.5: Compilation
//!
//! This module implements a compiler that translates Scheme-like expressions
//! into instruction sequences for a register machine. The compiler performs
//! the same analysis as the metacircular evaluator but generates machine
//! instructions instead of execution procedures.
//!
//! ## Key Concepts
//!
//! - **Instruction Sequences**: Contain needs/modifies register sets and statements
//! - **Targets**: Specify which register receives the result (usually `val`)
//! - **Linkage**: Specifies what happens after execution (next/return/label)
//! - **Register Optimization**: The compiler avoids unnecessary save/restore operations
//!
//! ## Rust Mapping
//!
//! - Instruction sequences → `InstructionSeq` struct with register tracking
//! - Linkage descriptors → `Linkage` enum (Next, Return, Label)
//! - Register machine → `Register` and `Instruction` enums
//! - Sequence combiners → functions operating on `InstructionSeq`

use std::collections::HashSet;
use std::fmt;

// ============================================================================
// Types
// ============================================================================

/// Virtual machine registers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Register {
    /// Environment register (current evaluation environment)
    Env,
    /// Procedure register (holds procedure to be applied)
    Proc,
    /// Value register (holds intermediate and final results)
    Val,
    /// Argument list register (holds arguments for procedure application)
    Argl,
    /// Continue register (holds return address)
    Continue,
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Register::Env => write!(f, "env"),
            Register::Proc => write!(f, "proc"),
            Register::Val => write!(f, "val"),
            Register::Argl => write!(f, "argl"),
            Register::Continue => write!(f, "continue"),
        }
    }
}

/// All registers used by the compiler
pub const ALL_REGS: &[Register] = &[
    Register::Env,
    Register::Proc,
    Register::Val,
    Register::Argl,
    Register::Continue,
];

/// Linkage descriptor specifying what happens after instruction execution
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Linkage {
    /// Continue to the next instruction in sequence
    Next,
    /// Return from the current procedure
    Return,
    /// Jump to a named label
    Label(String),
}

impl fmt::Display for Linkage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Linkage::Next => write!(f, "next"),
            Linkage::Return => write!(f, "return"),
            Linkage::Label(label) => write!(f, "{}", label),
        }
    }
}

/// Machine instruction (simplified representation)
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    /// Assign a value to a register: (assign reg (op operation) args...)
    Assign {
        target: Register,
        source: Box<InstructionValue>,
    },
    /// Perform an operation (side effect only): (perform (op operation) args...)
    Perform { operation: Box<InstructionValue> },
    /// Test a condition: (test (op predicate) args...)
    Test { condition: Box<InstructionValue> },
    /// Branch to a label if test is true: (branch (label name))
    Branch { label: String },
    /// Unconditional goto: (goto destination)
    Goto { destination: Box<InstructionValue> },
    /// Label marking a position in code
    Label { name: String },
    /// Save register to stack: (save reg)
    Save { register: Register },
    /// Restore register from stack: (restore reg)
    Restore { register: Register },
}

/// Value source for instructions
#[derive(Debug, Clone, PartialEq)]
pub enum InstructionValue {
    /// Constant value
    Const(Value),
    /// Register reference: (reg name)
    Reg(Register),
    /// Label reference: (label name)
    Label(String),
    /// Operation application: (op name) with arguments
    Op {
        name: String,
        args: Vec<InstructionValue>,
    },
}

/// Runtime values (simplified)
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Number(i64),
    String(String),
    Symbol(String),
    Bool(bool),
    Nil,
    /// Special marker for 'ok' result
    Ok,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Number(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Symbol(s) => write!(f, "{}", s),
            Value::Bool(true) => write!(f, "#t"),
            Value::Bool(false) => write!(f, "#f"),
            Value::Nil => write!(f, "()"),
            Value::Ok => write!(f, "ok"),
        }
    }
}

/// Expression AST (simplified from SICP chapter 4)
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Self-evaluating numbers
    Number(i64),
    /// Self-evaluating strings
    String(String),
    /// Variable reference (symbol)
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
    /// Conditional with multiple clauses: (cond (pred expr)...)
    Cond(Vec<(Expr, Expr)>),
    /// Procedure application: (operator operand...)
    Application {
        operator: Box<Expr>,
        operands: Vec<Expr>,
    },
}

/// An instruction sequence with register usage information
///
/// Contains:
/// - `needs`: Registers that must be initialized before execution
/// - `modifies`: Registers whose values are changed by execution
/// - `statements`: The actual machine instructions
#[derive(Debug, Clone)]
pub struct InstructionSeq {
    pub needs: HashSet<Register>,
    pub modifies: HashSet<Register>,
    pub statements: Vec<Instruction>,
}

impl InstructionSeq {
    /// Create a new instruction sequence
    pub fn new(
        needs: HashSet<Register>,
        modifies: HashSet<Register>,
        statements: Vec<Instruction>,
    ) -> Self {
        Self {
            needs,
            modifies,
            statements,
        }
    }

    /// Create an empty instruction sequence
    pub fn empty() -> Self {
        Self::new(HashSet::new(), HashSet::new(), Vec::new())
    }

    /// Check if this sequence needs a specific register
    pub fn needs_register(&self, reg: Register) -> bool {
        self.needs.contains(&reg)
    }

    /// Check if this sequence modifies a specific register
    pub fn modifies_register(&self, reg: Register) -> bool {
        self.modifies.contains(&reg)
    }
}

// ============================================================================
// Label Generation
// ============================================================================

use std::sync::atomic::{AtomicUsize, Ordering};

static LABEL_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Generate a unique label with the given prefix
pub fn make_label(prefix: &str) -> String {
    let counter = LABEL_COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("{}{}", prefix, counter)
}

/// Reset label counter (useful for testing)
pub fn reset_label_counter() {
    LABEL_COUNTER.store(0, Ordering::SeqCst);
}

// ============================================================================
// Set Operations (for register sets)
// ============================================================================

/// Union of two register sets
fn set_union(s1: &HashSet<Register>, s2: &HashSet<Register>) -> HashSet<Register> {
    s1.union(s2).copied().collect()
}

/// Difference of two register sets (elements in s1 not in s2)
fn set_difference(s1: &HashSet<Register>, s2: &HashSet<Register>) -> HashSet<Register> {
    s1.difference(s2).copied().collect()
}

// ============================================================================
// Sequence Combiners
// ============================================================================

/// Append two instruction sequences sequentially
///
/// The resulting sequence:
/// - Needs: registers needed by seq1 ∪ (registers needed by seq2 - registers modified by seq1)
/// - Modifies: registers modified by seq1 ∪ registers modified by seq2
/// - Statements: seq1 statements followed by seq2 statements
pub fn append_instruction_sequences(seq1: InstructionSeq, seq2: InstructionSeq) -> InstructionSeq {
    let needs = set_union(&seq1.needs, &set_difference(&seq2.needs, &seq1.modifies));
    let modifies = set_union(&seq1.modifies, &seq2.modifies);
    let mut statements = seq1.statements;
    statements.extend(seq2.statements);
    InstructionSeq::new(needs, modifies, statements)
}

/// Append multiple instruction sequences
pub fn append_sequences(seqs: Vec<InstructionSeq>) -> InstructionSeq {
    seqs.into_iter()
        .reduce(append_instruction_sequences)
        .unwrap_or_else(InstructionSeq::empty)
}

/// Preserve registers around execution of seq1 before seq2
///
/// Inserts save/restore around seq1 for any register in `regs` that:
/// - Is modified by seq1, AND
/// - Is needed by seq2
pub fn preserving(regs: &[Register], seq1: InstructionSeq, seq2: InstructionSeq) -> InstructionSeq {
    if regs.is_empty() {
        return append_instruction_sequences(seq1, seq2);
    }

    let first_reg = regs[0];
    let rest_regs = &regs[1..];

    if seq1.modifies_register(first_reg) && seq2.needs_register(first_reg) {
        // Need to save and restore this register
        let mut needs = seq1.needs.clone();
        needs.insert(first_reg);

        let modifies = set_difference(&seq1.modifies, &HashSet::from([first_reg]));

        let mut statements = vec![Instruction::Save {
            register: first_reg,
        }];
        statements.extend(seq1.statements);
        statements.push(Instruction::Restore {
            register: first_reg,
        });

        let protected_seq1 = InstructionSeq::new(needs, modifies, statements);
        preserving(rest_regs, protected_seq1, seq2)
    } else {
        // No save/restore needed for this register
        preserving(rest_regs, seq1, seq2)
    }
}

/// Append body sequence to main sequence without considering body's register use
///
/// Used for lambda compilation where the body is not "in line" -
/// it's code that will be jumped to later, not executed sequentially.
pub fn tack_on_instruction_sequence(seq: InstructionSeq, body: InstructionSeq) -> InstructionSeq {
    let mut statements = seq.statements;
    statements.extend(body.statements);
    InstructionSeq::new(seq.needs, seq.modifies, statements)
}

/// Combine two instruction sequences that execute in parallel (not sequentially)
///
/// Used for conditional branches - only one will actually execute,
/// so the second sequence still needs all its registers even if the
/// first sequence modifies them.
pub fn parallel_instruction_sequences(
    seq1: InstructionSeq,
    seq2: InstructionSeq,
) -> InstructionSeq {
    let needs = set_union(&seq1.needs, &seq2.needs);
    let modifies = set_union(&seq1.modifies, &seq2.modifies);
    let mut statements = seq1.statements;
    statements.extend(seq2.statements);
    InstructionSeq::new(needs, modifies, statements)
}

// ============================================================================
// Linkage Code Generation
// ============================================================================

/// Generate code for a linkage descriptor
pub fn compile_linkage(linkage: &Linkage) -> InstructionSeq {
    match linkage {
        Linkage::Return => {
            // Return: jump to address in continue register
            InstructionSeq::new(
                HashSet::from([Register::Continue]),
                HashSet::new(),
                vec![Instruction::Goto {
                    destination: Box::new(InstructionValue::Reg(Register::Continue)),
                }],
            )
        }
        Linkage::Next => {
            // Next: no code needed, just fall through
            InstructionSeq::empty()
        }
        Linkage::Label(label) => {
            // Jump to specific label
            InstructionSeq::new(
                HashSet::new(),
                HashSet::new(),
                vec![Instruction::Goto {
                    destination: Box::new(InstructionValue::Label(label.clone())),
                }],
            )
        }
    }
}

/// Append linkage code to an instruction sequence
///
/// Preserves the continue register if the linkage is Return,
/// since seq might modify continue but the linkage needs it.
pub fn end_with_linkage(linkage: &Linkage, seq: InstructionSeq) -> InstructionSeq {
    preserving(&[Register::Continue], seq, compile_linkage(linkage))
}

// ============================================================================
// Expression Compilation
// ============================================================================

/// Compile an expression to an instruction sequence
///
/// # Arguments
/// - `expr`: The expression to compile
/// - `target`: Register where the result should be placed
/// - `linkage`: What to do after computing the result
pub fn compile(expr: &Expr, target: Register, linkage: &Linkage) -> InstructionSeq {
    match expr {
        Expr::Number(_) | Expr::String(_) => compile_self_evaluating(expr, target, linkage),
        Expr::Quote(_) => compile_quoted(expr, target, linkage),
        Expr::Symbol(_) => compile_variable(expr, target, linkage),
        Expr::Set { .. } => compile_assignment(expr, target, linkage),
        Expr::Define { .. } => compile_definition(expr, target, linkage),
        Expr::If { .. } => compile_if(expr, target, linkage),
        Expr::Lambda { .. } => compile_lambda(expr, target, linkage),
        Expr::Begin(exprs) => compile_sequence(exprs, target, linkage),
        Expr::Cond(clauses) => {
            // Convert cond to nested if
            let if_expr = cond_to_if(clauses);
            compile(&if_expr, target, linkage)
        }
        Expr::Application { .. } => compile_application(expr, target, linkage),
    }
}

// ============================================================================
// Simple Expression Compilers
// ============================================================================

/// Compile self-evaluating expressions (numbers, strings)
fn compile_self_evaluating(expr: &Expr, target: Register, linkage: &Linkage) -> InstructionSeq {
    let value = match expr {
        Expr::Number(n) => Value::Number(*n),
        Expr::String(s) => Value::String(s.clone()),
        _ => panic!("Not a self-evaluating expression"),
    };

    let seq = InstructionSeq::new(
        HashSet::new(),
        HashSet::from([target]),
        vec![Instruction::Assign {
            target,
            source: Box::new(InstructionValue::Const(value)),
        }],
    );

    end_with_linkage(linkage, seq)
}

/// Compile quoted expressions
fn compile_quoted(expr: &Expr, target: Register, linkage: &Linkage) -> InstructionSeq {
    let quoted_value = match expr {
        Expr::Quote(e) => expr_to_value(e),
        _ => panic!("Not a quoted expression"),
    };

    let seq = InstructionSeq::new(
        HashSet::new(),
        HashSet::from([target]),
        vec![Instruction::Assign {
            target,
            source: Box::new(InstructionValue::Const(quoted_value)),
        }],
    );

    end_with_linkage(linkage, seq)
}

/// Compile variable lookup
fn compile_variable(expr: &Expr, target: Register, linkage: &Linkage) -> InstructionSeq {
    let var_name = match expr {
        Expr::Symbol(name) => name.clone(),
        _ => panic!("Not a variable"),
    };

    let seq = InstructionSeq::new(
        HashSet::from([Register::Env]),
        HashSet::from([target]),
        vec![Instruction::Assign {
            target,
            source: Box::new(InstructionValue::Op {
                name: "lookup-variable-value".to_string(),
                args: vec![
                    InstructionValue::Const(Value::Symbol(var_name)),
                    InstructionValue::Reg(Register::Env),
                ],
            }),
        }],
    );

    end_with_linkage(linkage, seq)
}

/// Compile assignment (set!)
fn compile_assignment(expr: &Expr, target: Register, linkage: &Linkage) -> InstructionSeq {
    let (var_name, value_expr) = match expr {
        Expr::Set { name, value } => (name.clone(), value.as_ref()),
        _ => panic!("Not an assignment"),
    };

    let get_value_code = compile(value_expr, Register::Val, &Linkage::Next);

    let set_code = InstructionSeq::new(
        HashSet::from([Register::Env, Register::Val]),
        HashSet::from([target]),
        vec![
            Instruction::Perform {
                operation: Box::new(InstructionValue::Op {
                    name: "set-variable-value!".to_string(),
                    args: vec![
                        InstructionValue::Const(Value::Symbol(var_name)),
                        InstructionValue::Reg(Register::Val),
                        InstructionValue::Reg(Register::Env),
                    ],
                }),
            },
            Instruction::Assign {
                target,
                source: Box::new(InstructionValue::Const(Value::Ok)),
            },
        ],
    );

    end_with_linkage(
        linkage,
        preserving(&[Register::Env], get_value_code, set_code),
    )
}

/// Compile definition
fn compile_definition(expr: &Expr, target: Register, linkage: &Linkage) -> InstructionSeq {
    let (var_name, value_expr) = match expr {
        Expr::Define { name, value } => (name.clone(), value.as_ref()),
        _ => panic!("Not a definition"),
    };

    let get_value_code = compile(value_expr, Register::Val, &Linkage::Next);

    let define_code = InstructionSeq::new(
        HashSet::from([Register::Env, Register::Val]),
        HashSet::from([target]),
        vec![
            Instruction::Perform {
                operation: Box::new(InstructionValue::Op {
                    name: "define-variable!".to_string(),
                    args: vec![
                        InstructionValue::Const(Value::Symbol(var_name)),
                        InstructionValue::Reg(Register::Val),
                        InstructionValue::Reg(Register::Env),
                    ],
                }),
            },
            Instruction::Assign {
                target,
                source: Box::new(InstructionValue::Const(Value::Ok)),
            },
        ],
    );

    end_with_linkage(
        linkage,
        preserving(&[Register::Env], get_value_code, define_code),
    )
}

// ============================================================================
// Conditional Compilation
// ============================================================================

/// Compile if expression
fn compile_if(expr: &Expr, target: Register, linkage: &Linkage) -> InstructionSeq {
    let (predicate, consequent, alternative) = match expr {
        Expr::If {
            predicate,
            consequent,
            alternative,
        } => (
            predicate.as_ref(),
            consequent.as_ref(),
            alternative.as_ref(),
        ),
        _ => panic!("Not an if expression"),
    };

    let t_branch = make_label("true-branch");
    let f_branch = make_label("false-branch");
    let after_if = make_label("after-if");

    let consequent_linkage = if linkage == &Linkage::Next {
        Linkage::Label(after_if.clone())
    } else {
        linkage.clone()
    };

    let p_code = compile(predicate, Register::Val, &Linkage::Next);
    let c_code = compile(consequent, target, &consequent_linkage);
    let a_code = compile(alternative, target, linkage);

    let test_code = InstructionSeq::new(
        HashSet::from([Register::Val]),
        HashSet::new(),
        vec![
            Instruction::Test {
                condition: Box::new(InstructionValue::Op {
                    name: "false?".to_string(),
                    args: vec![InstructionValue::Reg(Register::Val)],
                }),
            },
            Instruction::Branch {
                label: f_branch.clone(),
            },
        ],
    );

    let branches = parallel_instruction_sequences(
        append_instruction_sequences(
            InstructionSeq::new(
                HashSet::new(),
                HashSet::new(),
                vec![Instruction::Label { name: t_branch }],
            ),
            c_code,
        ),
        append_instruction_sequences(
            InstructionSeq::new(
                HashSet::new(),
                HashSet::new(),
                vec![Instruction::Label { name: f_branch }],
            ),
            a_code,
        ),
    );

    let after_if_label = InstructionSeq::new(
        HashSet::new(),
        HashSet::new(),
        vec![Instruction::Label { name: after_if }],
    );

    preserving(
        &[Register::Env, Register::Continue],
        p_code,
        append_sequences(vec![test_code, branches, after_if_label]),
    )
}

/// Convert cond to nested if expressions
fn cond_to_if(clauses: &[(Expr, Expr)]) -> Expr {
    if clauses.is_empty() {
        // No clauses - return false
        Expr::Number(0) // Using 0 as false placeholder
    } else {
        let (pred, conseq) = &clauses[0];
        let rest = &clauses[1..];
        Expr::If {
            predicate: Box::new(pred.clone()),
            consequent: Box::new(conseq.clone()),
            alternative: Box::new(cond_to_if(rest)),
        }
    }
}

// ============================================================================
// Sequence Compilation
// ============================================================================

/// Compile a sequence of expressions (for begin or procedure body)
fn compile_sequence(exprs: &[Expr], target: Register, linkage: &Linkage) -> InstructionSeq {
    if exprs.is_empty() {
        InstructionSeq::empty()
    } else if exprs.len() == 1 {
        compile(&exprs[0], target, linkage)
    } else {
        preserving(
            &[Register::Env, Register::Continue],
            compile(&exprs[0], target, &Linkage::Next),
            compile_sequence(&exprs[1..], target, linkage),
        )
    }
}

// ============================================================================
// Lambda Compilation
// ============================================================================

/// Compile lambda expression (procedure creation)
fn compile_lambda(expr: &Expr, target: Register, linkage: &Linkage) -> InstructionSeq {
    let (params, body) = match expr {
        Expr::Lambda { params, body } => (params, body),
        _ => panic!("Not a lambda expression"),
    };

    let proc_entry = make_label("entry");
    let after_lambda = make_label("after-lambda");

    let lambda_linkage = if linkage == &Linkage::Next {
        Linkage::Label(after_lambda.clone())
    } else {
        linkage.clone()
    };

    let make_proc = InstructionSeq::new(
        HashSet::from([Register::Env]),
        HashSet::from([target]),
        vec![Instruction::Assign {
            target,
            source: Box::new(InstructionValue::Op {
                name: "make-compiled-procedure".to_string(),
                args: vec![
                    InstructionValue::Label(proc_entry.clone()),
                    InstructionValue::Reg(Register::Env),
                ],
            }),
        }],
    );

    let proc_body = compile_lambda_body(params, body, &proc_entry);

    let after_lambda_label = InstructionSeq::new(
        HashSet::new(),
        HashSet::new(),
        vec![Instruction::Label { name: after_lambda }],
    );

    append_sequences(vec![
        tack_on_instruction_sequence(end_with_linkage(&lambda_linkage, make_proc), proc_body),
        after_lambda_label,
    ])
}

/// Compile lambda body
fn compile_lambda_body(params: &[String], body: &[Expr], entry_label: &str) -> InstructionSeq {
    let param_list = params
        .iter()
        .map(|p| Value::Symbol(p.clone()))
        .collect::<Vec<_>>();

    let entry = InstructionSeq::new(
        HashSet::from([Register::Env, Register::Proc, Register::Argl]),
        HashSet::from([Register::Env]),
        vec![
            Instruction::Label {
                name: entry_label.to_string(),
            },
            Instruction::Assign {
                target: Register::Env,
                source: Box::new(InstructionValue::Op {
                    name: "compiled-procedure-env".to_string(),
                    args: vec![InstructionValue::Reg(Register::Proc)],
                }),
            },
            Instruction::Assign {
                target: Register::Env,
                source: Box::new(InstructionValue::Op {
                    name: "extend-environment".to_string(),
                    args: vec![
                        InstructionValue::Const(Value::Symbol(format!("{:?}", param_list))),
                        InstructionValue::Reg(Register::Argl),
                        InstructionValue::Reg(Register::Env),
                    ],
                }),
            },
        ],
    );

    append_instruction_sequences(
        entry,
        compile_sequence(body, Register::Val, &Linkage::Return),
    )
}

// ============================================================================
// Application Compilation
// ============================================================================

/// Compile procedure application
fn compile_application(expr: &Expr, target: Register, linkage: &Linkage) -> InstructionSeq {
    let (operator, operands) = match expr {
        Expr::Application { operator, operands } => (operator.as_ref(), operands),
        _ => panic!("Not an application"),
    };

    let proc_code = compile(operator, Register::Proc, &Linkage::Next);
    let operand_codes: Vec<InstructionSeq> = operands
        .iter()
        .map(|operand| compile(operand, Register::Val, &Linkage::Next))
        .collect();

    preserving(
        &[Register::Env, Register::Continue],
        proc_code,
        preserving(
            &[Register::Proc, Register::Continue],
            construct_arglist(operand_codes),
            compile_procedure_call(target, linkage),
        ),
    )
}

/// Construct argument list from compiled operand codes
fn construct_arglist(operand_codes: Vec<InstructionSeq>) -> InstructionSeq {
    if operand_codes.is_empty() {
        InstructionSeq::new(
            HashSet::new(),
            HashSet::from([Register::Argl]),
            vec![Instruction::Assign {
                target: Register::Argl,
                source: Box::new(InstructionValue::Const(Value::Nil)),
            }],
        )
    } else {
        let mut reversed_codes = operand_codes;
        reversed_codes.reverse();

        let code_to_get_last_arg = append_instruction_sequences(
            reversed_codes[0].clone(),
            InstructionSeq::new(
                HashSet::from([Register::Val]),
                HashSet::from([Register::Argl]),
                vec![Instruction::Assign {
                    target: Register::Argl,
                    source: Box::new(InstructionValue::Op {
                        name: "list".to_string(),
                        args: vec![InstructionValue::Reg(Register::Val)],
                    }),
                }],
            ),
        );

        if reversed_codes.len() == 1 {
            code_to_get_last_arg
        } else {
            preserving(
                &[Register::Env],
                code_to_get_last_arg,
                code_to_get_rest_args(&reversed_codes[1..]),
            )
        }
    }
}

/// Build code to get remaining arguments (after the last one)
fn code_to_get_rest_args(operand_codes: &[InstructionSeq]) -> InstructionSeq {
    let code_for_next_arg = preserving(
        &[Register::Argl],
        operand_codes[0].clone(),
        InstructionSeq::new(
            HashSet::from([Register::Val, Register::Argl]),
            HashSet::from([Register::Argl]),
            vec![Instruction::Assign {
                target: Register::Argl,
                source: Box::new(InstructionValue::Op {
                    name: "cons".to_string(),
                    args: vec![
                        InstructionValue::Reg(Register::Val),
                        InstructionValue::Reg(Register::Argl),
                    ],
                }),
            }],
        ),
    );

    if operand_codes.len() == 1 {
        code_for_next_arg
    } else {
        preserving(
            &[Register::Env],
            code_for_next_arg,
            code_to_get_rest_args(&operand_codes[1..]),
        )
    }
}

/// Compile procedure call (primitive vs compiled dispatch)
fn compile_procedure_call(target: Register, linkage: &Linkage) -> InstructionSeq {
    let primitive_branch = make_label("primitive-branch");
    let compiled_branch = make_label("compiled-branch");
    let after_call = make_label("after-call");

    let compiled_linkage = if linkage == &Linkage::Next {
        Linkage::Label(after_call.clone())
    } else {
        linkage.clone()
    };

    let test_code = InstructionSeq::new(
        HashSet::from([Register::Proc]),
        HashSet::new(),
        vec![
            Instruction::Test {
                condition: Box::new(InstructionValue::Op {
                    name: "primitive-procedure?".to_string(),
                    args: vec![InstructionValue::Reg(Register::Proc)],
                }),
            },
            Instruction::Branch {
                label: primitive_branch.clone(),
            },
        ],
    );

    let compiled_case = append_instruction_sequences(
        InstructionSeq::new(
            HashSet::new(),
            HashSet::new(),
            vec![Instruction::Label {
                name: compiled_branch,
            }],
        ),
        compile_proc_appl(target, &compiled_linkage),
    );

    let primitive_case = append_instruction_sequences(
        InstructionSeq::new(
            HashSet::new(),
            HashSet::new(),
            vec![Instruction::Label {
                name: primitive_branch,
            }],
        ),
        end_with_linkage(
            linkage,
            InstructionSeq::new(
                HashSet::from([Register::Proc, Register::Argl]),
                HashSet::from([target]),
                vec![Instruction::Assign {
                    target,
                    source: Box::new(InstructionValue::Op {
                        name: "apply-primitive-procedure".to_string(),
                        args: vec![
                            InstructionValue::Reg(Register::Proc),
                            InstructionValue::Reg(Register::Argl),
                        ],
                    }),
                }],
            ),
        ),
    );

    let after_call_label = InstructionSeq::new(
        HashSet::new(),
        HashSet::new(),
        vec![Instruction::Label { name: after_call }],
    );

    append_sequences(vec![
        test_code,
        parallel_instruction_sequences(compiled_case, primitive_case),
        after_call_label,
    ])
}

/// Apply compiled procedure
fn compile_proc_appl(target: Register, linkage: &Linkage) -> InstructionSeq {
    let all_regs_set: HashSet<Register> = ALL_REGS.iter().copied().collect();

    match (target, linkage) {
        (Register::Val, Linkage::Return) => {
            // Tail call optimization: just jump to procedure
            InstructionSeq::new(
                HashSet::from([Register::Proc, Register::Continue]),
                all_regs_set,
                vec![
                    Instruction::Assign {
                        target: Register::Val,
                        source: Box::new(InstructionValue::Op {
                            name: "compiled-procedure-entry".to_string(),
                            args: vec![InstructionValue::Reg(Register::Proc)],
                        }),
                    },
                    Instruction::Goto {
                        destination: Box::new(InstructionValue::Reg(Register::Val)),
                    },
                ],
            )
        }
        (Register::Val, _) => {
            // Set continue to linkage target, then jump
            let continue_target = match linkage {
                Linkage::Label(label) => InstructionValue::Label(label.clone()),
                _ => panic!("Invalid linkage for val target"),
            };

            InstructionSeq::new(
                HashSet::from([Register::Proc]),
                all_regs_set,
                vec![
                    Instruction::Assign {
                        target: Register::Continue,
                        source: Box::new(continue_target),
                    },
                    Instruction::Assign {
                        target: Register::Val,
                        source: Box::new(InstructionValue::Op {
                            name: "compiled-procedure-entry".to_string(),
                            args: vec![InstructionValue::Reg(Register::Proc)],
                        }),
                    },
                    Instruction::Goto {
                        destination: Box::new(InstructionValue::Reg(Register::Val)),
                    },
                ],
            )
        }
        (_, Linkage::Return) => {
            panic!("Return linkage with target != val not supported");
        }
        (_, _) => {
            // Non-val target with label linkage
            let proc_return = make_label("proc-return");
            let linkage_label = match linkage {
                Linkage::Label(label) => label.clone(),
                _ => panic!("Invalid linkage"),
            };

            InstructionSeq::new(
                HashSet::from([Register::Proc]),
                all_regs_set,
                vec![
                    Instruction::Assign {
                        target: Register::Continue,
                        source: Box::new(InstructionValue::Label(proc_return.clone())),
                    },
                    Instruction::Assign {
                        target: Register::Val,
                        source: Box::new(InstructionValue::Op {
                            name: "compiled-procedure-entry".to_string(),
                            args: vec![InstructionValue::Reg(Register::Proc)],
                        }),
                    },
                    Instruction::Goto {
                        destination: Box::new(InstructionValue::Reg(Register::Val)),
                    },
                    Instruction::Label { name: proc_return },
                    Instruction::Assign {
                        target,
                        source: Box::new(InstructionValue::Reg(Register::Val)),
                    },
                    Instruction::Goto {
                        destination: Box::new(InstructionValue::Label(linkage_label)),
                    },
                ],
            )
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert expression to runtime value (for quoted expressions)
fn expr_to_value(expr: &Expr) -> Value {
    match expr {
        Expr::Number(n) => Value::Number(*n),
        Expr::String(s) => Value::String(s.clone()),
        Expr::Symbol(s) => Value::Symbol(s.clone()),
        _ => Value::Symbol("<complex-value>".to_string()),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_self_evaluating() {
        reset_label_counter();
        let expr = Expr::Number(42);
        let seq = compile(&expr, Register::Val, &Linkage::Next);

        assert_eq!(seq.statements.len(), 1);
        assert!(seq.modifies_register(Register::Val));
        assert!(!seq.needs_register(Register::Env));
    }

    #[test]
    fn test_compile_variable() {
        reset_label_counter();
        let expr = Expr::Symbol("x".to_string());
        let seq = compile(&expr, Register::Val, &Linkage::Next);

        assert!(seq.needs_register(Register::Env));
        assert!(seq.modifies_register(Register::Val));
        assert_eq!(seq.statements.len(), 1);
    }

    #[test]
    fn test_compile_if() {
        reset_label_counter();
        let expr = Expr::If {
            predicate: Box::new(Expr::Symbol("x".to_string())),
            consequent: Box::new(Expr::Number(1)),
            alternative: Box::new(Expr::Number(2)),
        };

        let seq = compile(&expr, Register::Val, &Linkage::Next);

        // Should have multiple statements including labels, test, branch
        assert!(seq.statements.len() > 5);
        assert!(seq.needs_register(Register::Env));
    }

    #[test]
    fn test_compile_lambda() {
        reset_label_counter();
        let expr = Expr::Lambda {
            params: vec!["x".to_string()],
            body: vec![Expr::Symbol("x".to_string())],
        };

        let seq = compile(&expr, Register::Val, &Linkage::Next);

        // Should create procedure and include body code
        assert!(seq.statements.len() > 3);
        assert!(seq.needs_register(Register::Env));
        assert!(seq.modifies_register(Register::Val));
    }

    #[test]
    fn test_compile_application() {
        reset_label_counter();
        let expr = Expr::Application {
            operator: Box::new(Expr::Symbol("+".to_string())),
            operands: vec![Expr::Number(1), Expr::Number(2)],
        };

        let seq = compile(&expr, Register::Val, &Linkage::Next);

        // Should compile operator, operands, construct arglist, and call
        assert!(seq.statements.len() > 5);
        assert!(seq.needs_register(Register::Env));
    }

    #[test]
    fn test_compile_definition() {
        reset_label_counter();
        let expr = Expr::Define {
            name: "x".to_string(),
            value: Box::new(Expr::Number(42)),
        };

        let seq = compile(&expr, Register::Val, &Linkage::Next);

        assert!(seq.needs_register(Register::Env));
        assert!(seq.modifies_register(Register::Val));
    }

    #[test]
    fn test_sequence_combiners() {
        let seq1 = InstructionSeq::new(
            HashSet::from([Register::Env]),
            HashSet::from([Register::Val]),
            vec![],
        );

        let seq2 = InstructionSeq::new(
            HashSet::from([Register::Val]),
            HashSet::from([Register::Proc]),
            vec![],
        );

        let combined = append_instruction_sequences(seq1.clone(), seq2.clone());

        // Should need Env (from seq1) and Val (from seq2, not modified by seq1... wait, it is!)
        // Actually: needs Env (seq1 needs) ∪ (Val (seq2 needs) - Val (seq1 modifies))
        // = Env ∪ (Val - Val) = Env ∪ ∅ = Env
        assert!(combined.needs_register(Register::Env));
        assert!(!combined.needs_register(Register::Val)); // Val is provided by seq1

        // Should modify Val (seq1) ∪ Proc (seq2)
        assert!(combined.modifies_register(Register::Val));
        assert!(combined.modifies_register(Register::Proc));
    }

    #[test]
    fn test_preserving_no_save_needed() {
        let seq1 = InstructionSeq::new(
            HashSet::from([Register::Env]),
            HashSet::from([Register::Val]),
            vec![],
        );

        let seq2 = InstructionSeq::new(
            HashSet::from([Register::Proc]), // Doesn't need Val
            HashSet::from([Register::Argl]),
            vec![],
        );

        let preserved = preserving(&[Register::Val], seq1, seq2);

        // Should NOT insert save/restore since seq2 doesn't need Val
        assert_eq!(preserved.statements.len(), 0);
    }

    #[test]
    fn test_preserving_with_save() {
        let seq1 = InstructionSeq::new(HashSet::new(), HashSet::from([Register::Val]), vec![]);

        let seq2 = InstructionSeq::new(
            HashSet::from([Register::Val]), // DOES need Val
            HashSet::new(),
            vec![],
        );

        let preserved = preserving(&[Register::Val], seq1, seq2);

        // Should insert save/restore
        assert_eq!(preserved.statements.len(), 2); // save + restore
    }

    #[test]
    fn test_factorial_compilation() {
        reset_label_counter();

        // (define (factorial n)
        //   (if (= n 1)
        //       1
        //       (* (factorial (- n 1)) n)))
        let factorial = Expr::Define {
            name: "factorial".to_string(),
            value: Box::new(Expr::Lambda {
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
            }),
        };

        let seq = compile(&factorial, Register::Val, &Linkage::Next);

        // Should produce a complex instruction sequence
        assert!(seq.statements.len() > 10);
        assert!(seq.needs_register(Register::Env));

        // Check that it contains expected instruction types
        let has_label = seq
            .statements
            .iter()
            .any(|inst| matches!(inst, Instruction::Label { .. }));
        let has_test = seq
            .statements
            .iter()
            .any(|inst| matches!(inst, Instruction::Test { .. }));
        let has_branch = seq
            .statements
            .iter()
            .any(|inst| matches!(inst, Instruction::Branch { .. }));

        assert!(has_label, "Should have labels");
        assert!(has_test, "Should have test instructions");
        assert!(has_branch, "Should have branch instructions");
    }

    #[test]
    fn test_tail_call_optimization() {
        reset_label_counter();

        // Lambda that tail-calls another procedure
        let expr = Expr::Lambda {
            params: vec!["x".to_string()],
            body: vec![Expr::Application {
                operator: Box::new(Expr::Symbol("f".to_string())),
                operands: vec![Expr::Symbol("x".to_string())],
            }],
        };

        let seq = compile(&expr, Register::Val, &Linkage::Next);

        // The body compiles with linkage=return, enabling tail call optimization
        // This should not push a new frame for the final call
        assert!(seq.statements.len() > 0);
    }
}
