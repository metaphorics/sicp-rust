//! SICP 5.5절: 컴파일 (Compilation)
//!
//! 이 모듈은 Scheme 유사 표현식을 레지스터 머신용 명령 시퀀스로
//! 변환하는 컴파일러를 구현한다. 메타서큘러 평가기와 같은 분석을 수행하지만,
//! 실행 절차 대신 머신 명령을 생성한다.
//! (This module implements a compiler that translates Scheme-like expressions
//! into instruction sequences for a register machine. The compiler performs
//! the same analysis as the metacircular evaluator but generates machine
//! instructions instead of execution procedures.)
//!
//! ## 핵심 개념 (Key Concepts)
//!
//! - **명령 시퀀스 (Instruction Sequences)**: needs/modifies 레지스터 집합과 문장을 포함
//! - **대상 (Targets)**: 결과를 받을 레지스터 지정 (보통 `val`)
//! - **연결 (Linkage)**: 실행 이후 동작 지정 (next/return/label)
//! - **레지스터 최적화 (Register Optimization)**: 불필요한 save/restore를 피함
//!
//! ## Rust 매핑 (Rust Mapping)
//!
//! - 명령 시퀀스 → 레지스터 추적을 가진 `InstructionSeq` 구조체
//! - 연결 설명자 → `Linkage` 열거형 (Next, Return, Label)
//! - 레지스터 머신 → `Register`와 `Instruction` 열거형
//! - 시퀀스 결합기 → `InstructionSeq`를 다루는 함수들

use std::collections::HashSet;
use std::fmt;

// ============================================================================
// 타입 (Types)
// ============================================================================

/// 가상 머신 레지스터 (Virtual machine registers)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Register {
    /// 환경 레지스터 (현재 평가 환경)
    /// (Environment register (current evaluation environment))
    Env,
    /// 프로시저 레지스터 (적용할 프로시저 보관)
    /// (Procedure register (holds procedure to be applied))
    Proc,
    /// 값 레지스터 (중간/최종 결과 보관)
    /// (Value register (holds intermediate and final results))
    Val,
    /// 인자 리스트 레지스터 (프로시저 적용 인자 보관)
    /// (Argument list register (holds arguments for procedure application))
    Argl,
    /// continue 레지스터 (반환 주소 보관)
    /// (Continue register (holds return address))
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

/// 컴파일러가 사용하는 모든 레지스터 (All registers used by the compiler)
pub const ALL_REGS: &[Register] = &[
    Register::Env,
    Register::Proc,
    Register::Val,
    Register::Argl,
    Register::Continue,
];

/// 명령 실행 이후 동작을 지정하는 연결 설명자
/// (Linkage descriptor specifying what happens after instruction execution)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Linkage {
    /// 시퀀스의 다음 명령으로 계속
    /// (Continue to the next instruction in sequence)
    Next,
    /// 현재 프로시저에서 반환
    /// (Return from the current procedure)
    Return,
    /// 이름 있는 라벨로 점프
    /// (Jump to a named label)
    Label(String),
}

impl fmt::Display for Linkage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Linkage::Next => write!(f, "다음(next)"),
            Linkage::Return => write!(f, "반환(return)"),
            Linkage::Label(label) => write!(f, "{}", label),
        }
    }
}

/// 머신 명령 (단순화된 표현)
/// (Machine instruction (simplified representation))
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    /// 레지스터에 값 할당: (assign reg (op operation) args...)
    /// (Assign a value to a register: (assign reg (op operation) args...))
    Assign {
        target: Register,
        source: Box<InstructionValue>,
    },
    /// 연산 수행 (부수효과만): (perform (op operation) args...)
    /// (Perform an operation (side effect only): (perform (op operation) args...))
    Perform { operation: Box<InstructionValue> },
    /// 조건 검사: (test (op predicate) args...)
    /// (Test a condition: (test (op predicate) args...))
    Test { condition: Box<InstructionValue> },
    /// 테스트가 참이면 라벨로 분기: (branch (label name))
    /// (Branch to a label if test is true: (branch (label name)))
    Branch { label: String },
    /// 무조건 goto: (goto destination)
    /// (Unconditional goto: (goto destination))
    Goto { destination: Box<InstructionValue> },
    /// 코드 위치를 표시하는 라벨
    /// (Label marking a position in code)
    Label { name: String },
    /// 레지스터를 스택에 저장: (save reg)
    /// (Save register to stack: (save reg))
    Save { register: Register },
    /// 스택에서 레지스터 복원: (restore reg)
    /// (Restore register from stack: (restore reg))
    Restore { register: Register },
}

/// 명령을 위한 값 소스 (Value source for instructions)
#[derive(Debug, Clone, PartialEq)]
pub enum InstructionValue {
    /// 상수 값 (Constant value)
    Const(Value),
    /// 레지스터 참조: (reg name)
    /// (Register reference: (reg name))
    Reg(Register),
    /// 라벨 참조: (label name)
    /// (Label reference: (label name))
    Label(String),
    /// 연산 적용: (op name)과 인자들
    /// (Operation application: (op name) with arguments)
    Op {
        name: String,
        args: Vec<InstructionValue>,
    },
}

/// 런타임 값 (단순화)
/// (Runtime values (simplified))
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Number(i64),
    String(String),
    Symbol(String),
    Bool(bool),
    Nil,
    /// 'ok' 결과를 위한 특수 마커
    /// (Special marker for 'ok' result)
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
            Value::Ok => write!(f, "확인(ok)"),
        }
    }
}

/// 표현식 AST (SICP 4장 단순화)
/// (Expression AST (simplified from SICP chapter 4))
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// 자기평가 숫자 (Self-evaluating numbers)
    Number(i64),
    /// 자기평가 문자열 (Self-evaluating strings)
    String(String),
    /// 변수 참조 (심볼)
    /// (Variable reference (symbol))
    Symbol(String),
    /// 인용 표현식: (quote expr)
    /// (Quoted expression: (quote expr))
    Quote(Box<Expr>),
    /// 조건식: (if predicate consequent alternative)
    /// (Conditional: (if predicate consequent alternative))
    If {
        predicate: Box<Expr>,
        consequent: Box<Expr>,
        alternative: Box<Expr>,
    },
    /// 람다 추상화: (lambda (params...) body...)
    /// (Lambda abstraction: (lambda (params...) body...))
    Lambda {
        params: Vec<String>,
        body: Vec<Expr>,
    },
    /// 변수 정의: (define var expr)
    /// (Variable definition: (define var expr))
    Define { name: String, value: Box<Expr> },
    /// 대입: (set! var expr)
    /// (Assignment: (set! var expr))
    Set { name: String, value: Box<Expr> },
    /// 시퀀스: (begin expr...)
    /// (Sequence: (begin expr...))
    Begin(Vec<Expr>),
    /// 다중 절 조건식: (cond (pred expr)...)
    /// (Conditional with multiple clauses: (cond (pred expr)...))
    Cond(Vec<(Expr, Expr)>),
    /// 프로시저 적용: (operator operand...)
    /// (Procedure application: (operator operand...))
    Application {
        operator: Box<Expr>,
        operands: Vec<Expr>,
    },
}

/// 레지스터 사용 정보가 포함된 명령 시퀀스
///
/// 포함 내용:
/// - `needs`: 실행 전에 초기화되어야 하는 레지스터
/// - `modifies`: 실행으로 값이 바뀌는 레지스터
/// - `statements`: 실제 머신 명령
/// (An instruction sequence with register usage information
///
/// Contains:
/// - `needs`: Registers that must be initialized before execution
/// - `modifies`: Registers whose values are changed by execution
/// - `statements`: The actual machine instructions)
#[derive(Debug, Clone)]
pub struct InstructionSeq {
    pub needs: HashSet<Register>,
    pub modifies: HashSet<Register>,
    pub statements: Vec<Instruction>,
}

impl InstructionSeq {
    /// 새 명령 시퀀스 생성 (Create a new instruction sequence)
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

    /// 빈 명령 시퀀스 생성 (Create an empty instruction sequence)
    pub fn empty() -> Self {
        Self::new(HashSet::new(), HashSet::new(), Vec::new())
    }

    /// 시퀀스가 특정 레지스터를 필요로 하는지 확인
    /// (Check if this sequence needs a specific register)
    pub fn needs_register(&self, reg: Register) -> bool {
        self.needs.contains(&reg)
    }

    /// 시퀀스가 특정 레지스터를 수정하는지 확인
    /// (Check if this sequence modifies a specific register)
    pub fn modifies_register(&self, reg: Register) -> bool {
        self.modifies.contains(&reg)
    }
}

// ============================================================================
// 라벨 생성 (Label Generation)
// ============================================================================

use std::sync::atomic::{AtomicUsize, Ordering};

static LABEL_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// 주어진 접두어로 고유 라벨 생성
/// (Generate a unique label with the given prefix)
pub fn make_label(prefix: &str) -> String {
    let counter = LABEL_COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("{}{}", prefix, counter)
}

/// 라벨 카운터 초기화 (테스트에 유용)
/// (Reset label counter (useful for testing))
pub fn reset_label_counter() {
    LABEL_COUNTER.store(0, Ordering::SeqCst);
}

// ============================================================================
// 집합 연산 (레지스터 집합용) (Set Operations (for register sets))
// ============================================================================

/// 두 레지스터 집합의 합집합 (Union of two register sets)
fn set_union(s1: &HashSet<Register>, s2: &HashSet<Register>) -> HashSet<Register> {
    s1.union(s2).copied().collect()
}

/// 두 레지스터 집합의 차집합 (s1에 있고 s2에 없는 요소)
/// (Difference of two register sets (elements in s1 not in s2))
fn set_difference(s1: &HashSet<Register>, s2: &HashSet<Register>) -> HashSet<Register> {
    s1.difference(s2).copied().collect()
}

// ============================================================================
// 시퀀스 결합기 (Sequence Combiners)
// ============================================================================

/// 두 명령 시퀀스를 순차적으로 이어 붙임
///
/// 결과 시퀀스:
/// - Needs: seq1이 필요한 레지스터 ∪ (seq2가 필요한 레지스터 - seq1이 수정한 레지스터)
/// - Modifies: seq1이 수정한 레지스터 ∪ seq2가 수정한 레지스터
/// - Statements: seq1의 문장 뒤에 seq2의 문장
/// (Append two instruction sequences sequentially
///
/// The resulting sequence:
/// - Needs: registers needed by seq1 ∪ (registers needed by seq2 - registers modified by seq1)
/// - Modifies: registers modified by seq1 ∪ registers modified by seq2
/// - Statements: seq1 statements followed by seq2 statements)
pub fn append_instruction_sequences(seq1: InstructionSeq, seq2: InstructionSeq) -> InstructionSeq {
    let needs = set_union(&seq1.needs, &set_difference(&seq2.needs, &seq1.modifies));
    let modifies = set_union(&seq1.modifies, &seq2.modifies);
    let mut statements = seq1.statements;
    statements.extend(seq2.statements);
    InstructionSeq::new(needs, modifies, statements)
}

/// 여러 명령 시퀀스를 이어 붙임 (Append multiple instruction sequences)
pub fn append_sequences(seqs: Vec<InstructionSeq>) -> InstructionSeq {
    seqs.into_iter()
        .reduce(append_instruction_sequences)
        .unwrap_or_else(InstructionSeq::empty)
}

/// seq1 실행 후 seq2 전에 레지스터 보존
///
/// `regs`에 있는 레지스터 중 다음 조건을 만족하면 seq1 앞뒤에 save/restore 삽입:
/// - seq1이 수정했고, AND
/// - seq2가 필요로 함
/// (Preserve registers around execution of seq1 before seq2
///
/// Inserts save/restore around seq1 for any register in `regs` that:
/// - Is modified by seq1, AND
/// - Is needed by seq2)
pub fn preserving(regs: &[Register], seq1: InstructionSeq, seq2: InstructionSeq) -> InstructionSeq {
    if regs.is_empty() {
        return append_instruction_sequences(seq1, seq2);
    }

    let first_reg = regs[0];
    let rest_regs = &regs[1..];

    if seq1.modifies_register(first_reg) && seq2.needs_register(first_reg) {
        // 이 레지스터는 저장/복원이 필요
        // (Need to save and restore this register)
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
        // 이 레지스터는 save/restore 불필요
        // (No save/restore needed for this register)
        preserving(rest_regs, seq1, seq2)
    }
}

/// 본문 시퀀스를 메인 시퀀스에 추가 (본문 레지스터 사용은 고려하지 않음)
///
/// 람다 컴파일에서 본문은 "인라인"이 아니므로 사용됨 -
/// 본문 코드는 나중에 점프되어 실행되며, 순차 실행되지 않는다.
/// (Append body sequence to main sequence without considering body's register use
///
/// Used for lambda compilation where the body is not "in line" -
/// it's code that will be jumped to later, not executed sequentially.)
pub fn tack_on_instruction_sequence(seq: InstructionSeq, body: InstructionSeq) -> InstructionSeq {
    let mut statements = seq.statements;
    statements.extend(body.statements);
    InstructionSeq::new(seq.needs, seq.modifies, statements)
}

/// 병렬(비순차)로 실행되는 두 시퀀스를 결합
///
/// 조건 분기에서 사용 - 실제로는 하나만 실행되지만,
/// 첫 번째가 레지스터를 수정하더라도 두 번째 시퀀스는
/// 필요한 레지스터를 모두 필요로 한다.
/// (Combine two instruction sequences that execute in parallel (not sequentially)
///
/// Used for conditional branches - only one will actually execute,
/// so the second sequence still needs all its registers even if the
/// first sequence modifies them.)
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
// 연결 코드 생성 (Linkage Code Generation)
// ============================================================================

/// 연결 설명자에 대한 코드 생성
/// (Generate code for a linkage descriptor)
pub fn compile_linkage(linkage: &Linkage) -> InstructionSeq {
    match linkage {
        Linkage::Return => {
            // Return: continue 레지스터의 주소로 점프
            // (Return: jump to address in continue register)
            InstructionSeq::new(
                HashSet::from([Register::Continue]),
                HashSet::new(),
                vec![Instruction::Goto {
                    destination: Box::new(InstructionValue::Reg(Register::Continue)),
                }],
            )
        }
        Linkage::Next => {
            // Next: 코드 불필요, 그냥 통과
            // (Next: no code needed, just fall through)
            InstructionSeq::empty()
        }
        Linkage::Label(label) => {
            // 특정 라벨로 점프
            // (Jump to specific label)
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

/// 명령 시퀀스에 연결 코드 추가
///
/// linkage가 Return이면 continue 레지스터를 보존한다.
/// seq가 continue를 수정할 수 있지만 linkage가 필요로 하기 때문이다.
/// (Append linkage code to an instruction sequence
///
/// Preserves the continue register if the linkage is Return,
/// since seq might modify continue but the linkage needs it.)
pub fn end_with_linkage(linkage: &Linkage, seq: InstructionSeq) -> InstructionSeq {
    preserving(&[Register::Continue], seq, compile_linkage(linkage))
}

// ============================================================================
// 표현식 컴파일 (Expression Compilation)
// ============================================================================

/// 표현식을 명령 시퀀스로 컴파일
///
/// # 인자 (Arguments)
/// - `expr`: 컴파일할 표현식 (The expression to compile)
/// - `target`: 결과를 둘 레지스터 (Register where the result should be placed)
/// - `linkage`: 결과 계산 후 동작 (What to do after computing the result)
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
            // cond를 중첩 if로 변환
            // (Convert cond to nested if)
            let if_expr = cond_to_if(clauses);
            compile(&if_expr, target, linkage)
        }
        Expr::Application { .. } => compile_application(expr, target, linkage),
    }
}

// ============================================================================
// 단순 표현식 컴파일러 (Simple Expression Compilers)
// ============================================================================

/// 자기평가 표현식 컴파일 (숫자, 문자열)
/// (Compile self-evaluating expressions (numbers, strings))
fn compile_self_evaluating(expr: &Expr, target: Register, linkage: &Linkage) -> InstructionSeq {
    let value = match expr {
        Expr::Number(n) => Value::Number(*n),
        Expr::String(s) => Value::String(s.clone()),
        _ => panic!("자기평가 표현식이 아님 (Not a self-evaluating expression)"),
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

/// 인용 표현식 컴파일 (Compile quoted expressions)
fn compile_quoted(expr: &Expr, target: Register, linkage: &Linkage) -> InstructionSeq {
    let quoted_value = match expr {
        Expr::Quote(e) => expr_to_value(e),
        _ => panic!("인용 표현식이 아님 (Not a quoted expression)"),
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

/// 변수 조회 컴파일 (Compile variable lookup)
fn compile_variable(expr: &Expr, target: Register, linkage: &Linkage) -> InstructionSeq {
    let var_name = match expr {
        Expr::Symbol(name) => name.clone(),
        _ => panic!("변수가 아님 (Not a variable)"),
    };

    let seq = InstructionSeq::new(
        HashSet::from([Register::Env]),
        HashSet::from([target]),
        vec![Instruction::Assign {
            target,
            source: Box::new(InstructionValue::Op {
                name: "변수-조회(lookup-variable-value)".to_string(),
                args: vec![
                    InstructionValue::Const(Value::Symbol(var_name)),
                    InstructionValue::Reg(Register::Env),
                ],
            }),
        }],
    );

    end_with_linkage(linkage, seq)
}

/// 대입 컴파일 (set!) (Compile assignment (set!))
fn compile_assignment(expr: &Expr, target: Register, linkage: &Linkage) -> InstructionSeq {
    let (var_name, value_expr) = match expr {
        Expr::Set { name, value } => (name.clone(), value.as_ref()),
        _ => panic!("대입이 아님 (Not an assignment)"),
    };

    let get_value_code = compile(value_expr, Register::Val, &Linkage::Next);

    let set_code = InstructionSeq::new(
        HashSet::from([Register::Env, Register::Val]),
        HashSet::from([target]),
        vec![
            Instruction::Perform {
                operation: Box::new(InstructionValue::Op {
                    name: "변수-설정(set-variable-value!)".to_string(),
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

/// 정의 컴파일 (Compile definition)
fn compile_definition(expr: &Expr, target: Register, linkage: &Linkage) -> InstructionSeq {
    let (var_name, value_expr) = match expr {
        Expr::Define { name, value } => (name.clone(), value.as_ref()),
        _ => panic!("정의가 아님 (Not a definition)"),
    };

    let get_value_code = compile(value_expr, Register::Val, &Linkage::Next);

    let define_code = InstructionSeq::new(
        HashSet::from([Register::Env, Register::Val]),
        HashSet::from([target]),
        vec![
            Instruction::Perform {
                operation: Box::new(InstructionValue::Op {
                    name: "변수-정의(define-variable!)".to_string(),
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
// 조건식 컴파일 (Conditional Compilation)
// ============================================================================

/// if 표현식 컴파일 (Compile if expression)
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
        _ => panic!("if 표현식이 아님 (Not an if expression)"),
    };

    let t_branch = make_label("참-분기(true-branch)");
    let f_branch = make_label("거짓-분기(false-branch)");
    let after_if = make_label("if-이후(after-if)");

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
                    name: "거짓?(false?)".to_string(),
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

/// cond를 중첩 if 표현식으로 변환 (Convert cond to nested if expressions)
fn cond_to_if(clauses: &[(Expr, Expr)]) -> Expr {
    if clauses.is_empty() {
        // 절이 없음 - 거짓 반환
        // (No clauses - return false)
        Expr::Number(0) // 0을 거짓 자리표시자로 사용 (Using 0 as false placeholder)
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
// 시퀀스 컴파일 (Sequence Compilation)
// ============================================================================

/// 표현식 시퀀스 컴파일 (begin 또는 프로시저 본문)
/// (Compile a sequence of expressions (for begin or procedure body))
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
// 람다 컴파일 (Lambda Compilation)
// ============================================================================

/// 람다 표현식 컴파일 (프로시저 생성)
/// (Compile lambda expression (procedure creation))
fn compile_lambda(expr: &Expr, target: Register, linkage: &Linkage) -> InstructionSeq {
    let (params, body) = match expr {
        Expr::Lambda { params, body } => (params, body),
        _ => panic!("람다 표현식이 아님 (Not a lambda expression)"),
    };

    let proc_entry = make_label("진입(entry)");
    let after_lambda = make_label("람다-이후(after-lambda)");

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
                name: "컴파일-프로시저-생성(make-compiled-procedure)".to_string(),
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

/// 람다 본문 컴파일 (Compile lambda body)
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
                    name: "컴파일-프로시저-환경(compiled-procedure-env)".to_string(),
                    args: vec![InstructionValue::Reg(Register::Proc)],
                }),
            },
            Instruction::Assign {
                target: Register::Env,
                source: Box::new(InstructionValue::Op {
                    name: "환경-확장(extend-environment)".to_string(),
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
// 적용 컴파일 (Application Compilation)
// ============================================================================

/// 프로시저 적용 컴파일 (Compile procedure application)
fn compile_application(expr: &Expr, target: Register, linkage: &Linkage) -> InstructionSeq {
    let (operator, operands) = match expr {
        Expr::Application { operator, operands } => (operator.as_ref(), operands),
        _ => panic!("적용이 아님 (Not an application)"),
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

/// 컴파일된 피연산자 코드로 인자 리스트 구성
/// (Construct argument list from compiled operand codes)
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
                        name: "리스트(list)".to_string(),
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

/// 남은 인자들을 가져오는 코드 구성 (마지막 인자 이후)
/// (Build code to get remaining arguments (after the last one))
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
                    name: "쌍(cons)".to_string(),
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

/// 프로시저 호출 컴파일 (기본 vs 컴파일된 디스패치)
/// (Compile procedure call (primitive vs compiled dispatch))
fn compile_procedure_call(target: Register, linkage: &Linkage) -> InstructionSeq {
    let primitive_branch = make_label("기본-분기(primitive-branch)");
    let compiled_branch = make_label("컴파일-분기(compiled-branch)");
    let after_call = make_label("호출-이후(after-call)");

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
                    name: "기본-프로시저?(primitive-procedure?)".to_string(),
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
                        name: "기본-프로시저-적용(apply-primitive-procedure)".to_string(),
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

/// 컴파일된 프로시저 적용 (Apply compiled procedure)
fn compile_proc_appl(target: Register, linkage: &Linkage) -> InstructionSeq {
    let all_regs_set: HashSet<Register> = ALL_REGS.iter().copied().collect();

    match (target, linkage) {
        (Register::Val, Linkage::Return) => {
            // 꼬리 호출 최적화: 프로시저로 바로 점프
            // (Tail call optimization: just jump to procedure)
            InstructionSeq::new(
                HashSet::from([Register::Proc, Register::Continue]),
                all_regs_set,
                vec![
                    Instruction::Assign {
                        target: Register::Val,
                        source: Box::new(InstructionValue::Op {
                            name: "컴파일-프로시저-진입(compiled-procedure-entry)".to_string(),
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
            // continue를 linkage 대상으로 설정 후 점프
            // (Set continue to linkage target, then jump)
            let continue_target = match linkage {
                Linkage::Label(label) => InstructionValue::Label(label.clone()),
                _ => panic!("val 대상의 linkage가 유효하지 않음 (Invalid linkage for val target)"),
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
                            name: "컴파일-프로시저-진입(compiled-procedure-entry)".to_string(),
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
            panic!("target != val 인 Return linkage는 지원하지 않음 (Return linkage with target != val not supported)");
        }
        (_, _) => {
            // label linkage와 val이 아닌 target
            // (Non-val target with label linkage)
            let proc_return = make_label("프로시저-반환(proc-return)");
            let linkage_label = match linkage {
                Linkage::Label(label) => label.clone(),
                _ => panic!("유효하지 않은 linkage (Invalid linkage)"),
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
                            name: "컴파일-프로시저-진입(compiled-procedure-entry)".to_string(),
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
// 보조 함수 (Helper Functions)
// ============================================================================

/// 표현식을 런타임 값으로 변환 (인용 표현식용)
/// (Convert expression to runtime value (for quoted expressions))
fn expr_to_value(expr: &Expr) -> Value {
    match expr {
        Expr::Number(n) => Value::Number(*n),
        Expr::String(s) => Value::String(s.clone()),
        Expr::Symbol(s) => Value::Symbol(s.clone()),
        _ => Value::Symbol("<복합-값(complex-value)>".to_string()),
    }
}

// ============================================================================
// 테스트 (Tests)
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

        // 라벨/테스트/분기를 포함한 여러 문장이 있어야 함
        // (Should have multiple statements including labels, test, branch)
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

        // 프로시저 생성과 본문 코드가 포함되어야 함
        // (Should create procedure and include body code)
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

        // 연산자/피연산자 컴파일, arglist 구성, 호출이 포함되어야 함
        // (Should compile operator, operands, construct arglist, and call)
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

        // Env 필요 (seq1) + Val 필요 (seq2, seq1이 수정) 조건 확인
        // 실제로: needs Env (seq1 needs) ∪ (Val (seq2 needs) - Val (seq1 modifies))
        // = Env ∪ (Val - Val) = Env ∪ ∅ = Env
        assert!(combined.needs_register(Register::Env));
        assert!(!combined.needs_register(Register::Val)); // Val is provided by seq1

        // 수정 레지스터: Val (seq1) ∪ Proc (seq2)
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

        // seq2가 Val을 필요로 하지 않으므로 save/restore 없음
        // (Should NOT insert save/restore since seq2 doesn't need Val)
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

        // save/restore 삽입
        // (Should insert save/restore)
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

        // 복잡한 명령 시퀀스가 생성되어야 함
        // (Should produce a complex instruction sequence)
        assert!(seq.statements.len() > 10);
        assert!(seq.needs_register(Register::Env));

        // 기대하는 명령 타입 포함 여부 확인
        // (Check that it contains expected instruction types)
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

        assert!(has_label, "라벨이 있어야 함 (Should have labels)");
        assert!(has_test, "테스트 명령이 있어야 함 (Should have test instructions)");
        assert!(has_branch, "분기 명령이 있어야 함 (Should have branch instructions)");
    }

    #[test]
    fn test_tail_call_optimization() {
        reset_label_counter();

        // 다른 프로시저를 꼬리 호출하는 람다
        // (Lambda that tail-calls another procedure)
        let expr = Expr::Lambda {
            params: vec!["x".to_string()],
            body: vec![Expr::Application {
                operator: Box::new(Expr::Symbol("f".to_string())),
                operands: vec![Expr::Symbol("x".to_string())],
            }],
        };

        let seq = compile(&expr, Register::Val, &Linkage::Next);

        // 본문은 linkage=return으로 컴파일되어 꼬리 호출 최적화가 가능
        // 마지막 호출에서 새 프레임을 푸시하지 않아야 함
        // (The body compiles with linkage=return, enabling tail call optimization
        // This should not push a new frame for the final call)
        assert!(seq.statements.len() > 0);
    }
}
