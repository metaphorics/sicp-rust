//! SICP 5.2절: 레지스터 기계 시뮬레이터 (A Register-Machine Simulator)
//!
//! 이 모듈은 레지스터 기계 시뮬레이터를 구현하여
//! 기계 설계를 테스트하고 성능 특성을 측정할 수 있게 한다
//! (This module implements a simulator for register machines, allowing us to
//! test machine designs and measure their performance characteristics).
//!
//! # 아키텍처 (Architecture)
//!
//! 시뮬레이터는 다음 주요 구성요소로 이루어진다:
//! (The simulator consists of several key components:)
//!
//! - **Machine**: 레지스터, 스택, 명령 시퀀스를 포함하는 메인 구조
//! - **Assembler**: 컨트롤러 텍스트를 실행 가능한 명령으로 변환
//! - **Instructions**: 파싱되고 해석된 명령 표현
//! - **Stack**: 성능 모니터링과 함께 값을 추적
//!
//! # 메모리 모델 (Memory Model)
//!
//! ```text
//! Machine (모든 상태 소유) (owns all state)
//!   ├── registers: HashMap<String, Value>
//!   ├── stack: Stack (Vec<Value> + 통계 (statistics))
//!   ├── instructions: Vec<Instruction>
//!   ├── pc: usize (프로그램 카운터 인덱스) (program counter as index)
//!   └── flag: bool (test/branch용) (for test/branch)
//! ```
//!
//! # 예시 (Example)
//!
//! ```ignore
//! use sicp_chapter5::section_5_2::*;
//!
//! // 간단한 GCD 기계 생성 (Create a simple GCD machine)
//! let mut machine = MachineBuilder::new()
//!     .register("a")
//!     .register("b")
//!     .register("t")
//!     .operation("=", |args| Value::Bool(args[0] == args[1]))
//!     .operation("rem", |args| {
//!         if let (Value::Number(a), Value::Number(b)) = (&args[0], &args[1]) {
//!             Value::Number(a % b)
//!         } else {
//!             panic!("rem 은 숫자가 필요함 (rem requires numbers)")
//!         }
//!     })
//!     .controller(vec![
//!         Inst::Label("test-b".to_string()),
//!         Inst::Test(OpExp::new("=", vec![VExp::Reg("b".to_string()), VExp::Const(Value::Number(0))])),
//!         Inst::Branch("gcd-done".to_string()),
//!         Inst::Assign("t".to_string(), VExp::Op(OpExp::new("rem", vec![VExp::Reg("a".to_string()), VExp::Reg("b".to_string())]))),
//!         Inst::Assign("a".to_string(), VExp::Reg("b".to_string())),
//!         Inst::Assign("b".to_string(), VExp::Reg("t".to_string())),
//!         Inst::Goto(GotoDest::Label("test-b".to_string())),
//!         Inst::Label("gcd-done".to_string()),
//!     ])
//!     .build();
//!
//! machine.set_register("a", Value::Number(206));
//! machine.set_register("b", Value::Number(40));
//! machine.start();
//! assert_eq!(machine.get_register("a"), Value::Number(2));
//! ```

use std::collections::HashMap;
use std::fmt;

// ============================================================================
// 값 타입 (Value Type)
// ============================================================================

/// 레지스터나 스택에 저장될 수 있는 값
/// (Values that can be stored in registers or on the stack)
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// 미할당 레지스터 값 (Unassigned register value)
    Unassigned,
    /// 정수 (Integer number)
    Number(i64),
    /// 불리언 값 (test 결과용) (Boolean value (for test results))
    Bool(bool),
    /// 명령 포인터 (라벨 값용) (Instruction pointer (for label values))
    InstructionPointer(usize),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Unassigned => write!(f, "*미할당 (unassigned)*"),
            Value::Number(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::InstructionPointer(ip) => write!(f, "@{}", ip),
        }
    }
}

impl Value {
    /// 분기용으로 값을 불리언으로 변환
    /// (Convert value to boolean for branching)
    pub fn as_bool(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Number(0) => false,
            Value::Number(_) => true,
            _ => false,
        }
    }

    /// 숫자를 추출하거나 패닉 (Extract number or panic)
    pub fn as_number(&self) -> i64 {
        match self {
            Value::Number(n) => *n,
            _ => panic!(
                "숫자를 기대했지만 {:?}를 받음 (Expected number, got {:?})",
                self, self
            ),
        }
    }
}

// ============================================================================
// 성능 모니터링이 있는 스택 (Stack with Performance Monitoring)
// ============================================================================

/// 성능 통계를 추적하는 스택
/// (Stack that tracks performance statistics)
#[derive(Debug)]
pub struct Stack {
    data: Vec<Value>,
    pushes: usize,
    max_depth: usize,
}

impl Stack {
    /// 새 빈 스택 생성 (Create a new empty stack)
    pub fn new() -> Self {
        Stack {
            data: Vec::new(),
            pushes: 0,
            max_depth: 0,
        }
    }

    /// 스택에 값 푸시 (Push a value onto the stack)
    pub fn push(&mut self, value: Value) {
        self.data.push(value);
        self.pushes += 1;
        if self.data.len() > self.max_depth {
            self.max_depth = self.data.len();
        }
    }

    /// 스택에서 값 팝 (Pop a value from the stack)
    pub fn pop(&mut self) -> Value {
        self.data.pop().expect("빈 스택: POP (Empty stack: POP)")
    }

    /// 스택이 비었는지 확인 (Check if stack is empty)
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// 현재 깊이 반환 (Get current depth)
    pub fn depth(&self) -> usize {
        self.data.len()
    }

    /// 스택 초기화 (clear 및 통계 리셋)
    /// (Initialize stack (clear and reset statistics))
    pub fn initialize(&mut self) {
        self.data.clear();
        self.pushes = 0;
        self.max_depth = 0;
    }

    /// 총 push 횟수 반환 (Get total number of pushes)
    pub fn total_pushes(&self) -> usize {
        self.pushes
    }

    /// 최대 깊이 반환 (Get maximum depth reached)
    pub fn maximum_depth(&self) -> usize {
        self.max_depth
    }

    /// 스택 통계 출력 (Print stack statistics)
    pub fn print_statistics(&self) {
        println!(
            "(total-pushes = {} maximum-depth = {})",
            self.pushes, self.max_depth
        );
    }
}

impl Default for Stack {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 값 표현식 (해석 전) (Value Expressions (before resolution))
// ============================================================================

/// 값 표현식 (해석 전) (Value expression (unresolved))
#[derive(Debug, Clone, PartialEq)]
pub enum VExp {
    /// 상수 값 (Constant value)
    Const(Value),
    /// 레지스터 참조 (Register reference)
    Reg(String),
    /// 라벨 참조 (Label reference)
    Label(String),
    /// 연산 적용 (Operation application)
    Op(OpExp),
}

/// 연산 표현식 (Operation expression)
#[derive(Debug, Clone, PartialEq)]
pub struct OpExp {
    pub op_name: String,
    pub operands: Vec<VExp>,
}

impl OpExp {
    pub fn new(op_name: &str, operands: Vec<VExp>) -> Self {
        OpExp {
            op_name: op_name.to_string(),
            operands,
        }
    }
}

// ============================================================================
// 해석된 값 표현식 (Resolved Value Expressions)
// ============================================================================

/// 해석된 값 표현식 (어셈블 후)
/// (Resolved value expression (after assembly))
#[derive(Debug, Clone)]
enum ResolvedVExp {
    /// 상수 값 (Constant value)
    Const(Value),
    /// 레지스터 인덱스 (Register index)
    Reg(usize),
    /// 라벨 (명령 포인터) (Label (instruction pointer))
    Label(usize),
    /// 해석된 피연산자를 가진 연산
    /// (Operation with resolved operands)
    Op(String, Vec<ResolvedVExp>),
}

// ============================================================================
// 명령 (해석 전) (Instructions (before resolution))
// ============================================================================

/// 명령 텍스트 (어셈블 전) (Instruction text (before assembly))
#[derive(Debug, Clone, PartialEq)]
pub enum Inst {
    /// 라벨 마커 (Label marker)
    Label(String),
    /// 레지스터에 값 할당 (Assign value to register)
    Assign(String, VExp),
    /// 조건 테스트 후 플래그 설정 (Test condition and set flag)
    Test(OpExp),
    /// 플래그가 true면 분기 (Branch if flag is true)
    Branch(String),
    /// 라벨 또는 레지스터로 goto (Goto label or register)
    Goto(GotoDest),
    /// 레지스터를 스택에 저장 (Save register to stack)
    Save(String),
    /// 스택에서 레지스터 복원 (Restore register from stack)
    Restore(String),
    /// 연산 수행 (부수 효과용) (Perform operation (for side effects))
    Perform(OpExp),
}

/// Goto 대상 (Goto destination)
#[derive(Debug, Clone, PartialEq)]
pub enum GotoDest {
    Label(String),
    Reg(String),
}

// ============================================================================
// 해석된 명령 (Resolved Instructions)
// ============================================================================

/// 해석된 명령 (어셈블 후) (Resolved instruction (after assembly))
#[derive(Debug, Clone)]
enum ResolvedInst {
    /// 레지스터에 값 할당 (Assign value to register)
    Assign {
        target_reg: usize,
        value: ResolvedVExp,
    },
    /// 조건 테스트 후 플래그 설정 (Test condition and set flag)
    Test { condition: ResolvedVExp },
    /// 플래그가 true면 분기 (Branch if flag is true)
    Branch { destination: usize },
    /// goto 대상 (Goto destination)
    Goto { destination: GotoDestResolved },
    /// 레지스터를 스택에 저장 (Save register to stack)
    Save { reg: usize },
    /// 스택에서 레지스터 복원 (Restore register from stack)
    Restore { reg: usize },
    /// 연산 수행 (Perform operation)
    Perform { action: ResolvedVExp },
}

#[derive(Debug, Clone)]
enum GotoDestResolved {
    Label(usize),
    Reg(usize),
}

// ============================================================================
// 기계 (Machine)
// ============================================================================

/// 연산 함수 타입 (Operation function type)
pub type OpFn = Box<dyn Fn(&[Value]) -> Value>;

/// 레지스터 기계 (Register machine)
pub struct Machine {
    /// 레지스터 저장소 (레지스터 인덱스로 접근)
    /// (Register storage (indexed by register index))
    registers: Vec<Value>,
    /// 레지스터 이름 -> 인덱스 매핑
    /// (Register name to index mapping)
    register_map: HashMap<String, usize>,
    /// 스택 (Stack)
    stack: Stack,
    /// 프로그램 카운터 (명령 인덱스)
    /// (Program counter (index into instructions))
    pc: usize,
    /// 플래그 레지스터 (test/branch용)
    /// (Flag register (for test/branch))
    flag: bool,
    /// 명령 시퀀스 (Instruction sequence)
    instructions: Vec<ResolvedInst>,
    /// 연산 테이블 (Operations table)
    operations: HashMap<String, OpFn>,
    /// 명령 카운트 (Instruction count)
    instruction_count: usize,
}

impl Machine {
    /// 처음부터 기계 실행 (Execute the machine starting from the beginning)
    pub fn start(&mut self) {
        self.pc = 0;
        self.instruction_count = 0;
        self.execute();
    }

    /// 완료될 때까지 명령 실행 (Execute instructions until done)
    fn execute(&mut self) {
        while self.pc < self.instructions.len() {
            self.instruction_count += 1;
            let inst = self.instructions[self.pc].clone();
            self.execute_instruction(inst);
        }
    }

    /// 단일 명령 실행 (Execute a single instruction)
    fn execute_instruction(&mut self, inst: ResolvedInst) {
        match inst {
            ResolvedInst::Assign { target_reg, value } => {
                let val = self.eval_value_exp(&value);
                self.registers[target_reg] = val;
                self.pc += 1;
            }
            ResolvedInst::Test { condition } => {
                let result = self.eval_value_exp(&condition);
                self.flag = result.as_bool();
                self.pc += 1;
            }
            ResolvedInst::Branch { destination } => {
                if self.flag {
                    self.pc = destination;
                } else {
                    self.pc += 1;
                }
            }
            ResolvedInst::Goto { destination } => match destination {
                GotoDestResolved::Label(dest) => {
                    self.pc = dest;
                }
                GotoDestResolved::Reg(reg) => {
                    if let Value::InstructionPointer(ip) = self.registers[reg] {
                        self.pc = ip;
                    } else {
                        panic!(
                            "Goto 레지스터에는 명령 포인터가 있어야 함 (Goto register must contain instruction pointer)"
                        );
                    }
                }
            },
            ResolvedInst::Save { reg } => {
                let value = self.registers[reg].clone();
                self.stack.push(value);
                self.pc += 1;
            }
            ResolvedInst::Restore { reg } => {
                let value = self.stack.pop();
                self.registers[reg] = value;
                self.pc += 1;
            }
            ResolvedInst::Perform { action } => {
                self.eval_value_exp(&action);
                self.pc += 1;
            }
        }
    }

    /// 해석된 값 표현식을 평가
    /// (Evaluate a resolved value expression)
    fn eval_value_exp(&self, exp: &ResolvedVExp) -> Value {
        match exp {
            ResolvedVExp::Const(v) => v.clone(),
            ResolvedVExp::Reg(idx) => self.registers[*idx].clone(),
            ResolvedVExp::Label(ip) => Value::InstructionPointer(*ip),
            ResolvedVExp::Op(op_name, operands) => {
                let args: Vec<Value> = operands.iter().map(|e| self.eval_value_exp(e)).collect();
                let op = self
                    .operations
                    .get(op_name)
                    .unwrap_or_else(|| panic!("알 수 없는 연산 (Unknown operation): {}", op_name));
                op(&args)
            }
        }
    }

    /// 이름으로 레지스터 값 가져오기 (Get register value by name)
    pub fn get_register(&self, name: &str) -> Value {
        let idx = self
            .register_map
            .get(name)
            .unwrap_or_else(|| panic!("알 수 없는 레지스터 (Unknown register): {}", name));
        self.registers[*idx].clone()
    }

    /// 이름으로 레지스터 값 설정 (Set register value by name)
    pub fn set_register(&mut self, name: &str, value: Value) {
        let idx = self
            .register_map
            .get(name)
            .unwrap_or_else(|| panic!("알 수 없는 레지스터 (Unknown register): {}", name));
        self.registers[*idx] = value;
    }

    /// 스택 통계 가져오기 (Get stack statistics)
    pub fn stack_statistics(&self) -> (usize, usize) {
        (self.stack.total_pushes(), self.stack.maximum_depth())
    }

    /// 스택 통계 출력 (Print stack statistics)
    pub fn print_stack_statistics(&self) {
        self.stack.print_statistics();
    }

    /// 명령 카운트 반환 (Get instruction count)
    pub fn instruction_count(&self) -> usize {
        self.instruction_count
    }

    /// 명령 카운트 리셋 (Reset instruction count)
    pub fn reset_instruction_count(&mut self) {
        self.instruction_count = 0;
    }

    /// 스택 초기화 (Initialize stack)
    pub fn initialize_stack(&mut self) {
        self.stack.initialize();
    }
}

// ============================================================================
// 어셈블러 (Assembler)
// ============================================================================

/// 컨트롤러 텍스트를 실행 가능한 명령으로 어셈블
/// (Assemble controller text into executable instructions)
fn assemble(
    controller: Vec<Inst>,
    register_map: &HashMap<String, usize>,
    operations: &HashMap<String, OpFn>,
) -> Vec<ResolvedInst> {
    // 1단계: 라벨 추출 및 명령 리스트 구축
    // (Pass 1: Extract labels and build instruction list)
    let (insts, labels) = extract_labels(controller);

    // 2단계: 모든 참조를 해석하고 실행 절차 생성
    // (Pass 2: Resolve all references and create execution procedures)
    update_insts(insts, &labels, register_map, operations)
}

/// 컨트롤러 텍스트에서 라벨 추출
/// (Extract labels from controller text)
fn extract_labels(text: Vec<Inst>) -> (Vec<Inst>, HashMap<String, usize>) {
    let mut instructions = Vec::new();
    let mut labels = HashMap::new();

    for inst in text {
        match inst {
            Inst::Label(name) => {
                if labels.contains_key(&name) {
                    panic!("중복 정의된 라벨 (Multiply defined label): {}", name);
                }
                labels.insert(name, instructions.len());
            }
            _ => {
                instructions.push(inst);
            }
        }
    }

    (instructions, labels)
}

/// 해석된 참조로 명령 갱신
/// (Update instructions with resolved references)
fn update_insts(
    insts: Vec<Inst>,
    labels: &HashMap<String, usize>,
    register_map: &HashMap<String, usize>,
    _operations: &HashMap<String, OpFn>,
) -> Vec<ResolvedInst> {
    insts
        .into_iter()
        .map(|inst| resolve_instruction(inst, labels, register_map))
        .collect()
}

/// 단일 명령 해석 (Resolve a single instruction)
fn resolve_instruction(
    inst: Inst,
    labels: &HashMap<String, usize>,
    register_map: &HashMap<String, usize>,
) -> ResolvedInst {
    match inst {
        Inst::Assign(reg_name, value_exp) => {
            let target_reg = *register_map
                .get(&reg_name)
                .unwrap_or_else(|| panic!("알 수 없는 레지스터 (Unknown register): {}", reg_name));
            let value = resolve_value_exp(value_exp, labels, register_map);
            ResolvedInst::Assign { target_reg, value }
        }
        Inst::Test(op_exp) => {
            let condition = resolve_op_exp(op_exp, labels, register_map);
            ResolvedInst::Test { condition }
        }
        Inst::Branch(label_name) => {
            let destination = *labels
                .get(&label_name)
                .unwrap_or_else(|| panic!("정의되지 않은 라벨 (Undefined label): {}", label_name));
            ResolvedInst::Branch { destination }
        }
        Inst::Goto(dest) => {
            let destination = match dest {
                GotoDest::Label(label_name) => {
                    let ip = *labels.get(&label_name).unwrap_or_else(|| {
                        panic!("정의되지 않은 라벨 (Undefined label): {}", label_name)
                    });
                    GotoDestResolved::Label(ip)
                }
                GotoDest::Reg(reg_name) => {
                    let idx = *register_map.get(&reg_name).unwrap_or_else(|| {
                        panic!("알 수 없는 레지스터 (Unknown register): {}", reg_name)
                    });
                    GotoDestResolved::Reg(idx)
                }
            };
            ResolvedInst::Goto { destination }
        }
        Inst::Save(reg_name) => {
            let reg = *register_map
                .get(&reg_name)
                .unwrap_or_else(|| panic!("알 수 없는 레지스터 (Unknown register): {}", reg_name));
            ResolvedInst::Save { reg }
        }
        Inst::Restore(reg_name) => {
            let reg = *register_map
                .get(&reg_name)
                .unwrap_or_else(|| panic!("알 수 없는 레지스터 (Unknown register): {}", reg_name));
            ResolvedInst::Restore { reg }
        }
        Inst::Perform(op_exp) => {
            let action = resolve_op_exp(op_exp, labels, register_map);
            ResolvedInst::Perform { action }
        }
        Inst::Label(_) => panic!("Labels should have been filtered out"),
    }
}

/// 값 표현식 해석 (Resolve a value expression)
fn resolve_value_exp(
    exp: VExp,
    labels: &HashMap<String, usize>,
    register_map: &HashMap<String, usize>,
) -> ResolvedVExp {
    match exp {
        VExp::Const(v) => ResolvedVExp::Const(v),
        VExp::Reg(name) => {
            let idx = *register_map
                .get(&name)
                .unwrap_or_else(|| panic!("알 수 없는 레지스터 (Unknown register): {}", name));
            ResolvedVExp::Reg(idx)
        }
        VExp::Label(name) => {
            let ip = *labels
                .get(&name)
                .unwrap_or_else(|| panic!("정의되지 않은 라벨 (Undefined label): {}", name));
            ResolvedVExp::Label(ip)
        }
        VExp::Op(op_exp) => resolve_op_exp(op_exp, labels, register_map),
    }
}

/// 연산 표현식 해석 (Resolve an operation expression)
fn resolve_op_exp(
    op_exp: OpExp,
    labels: &HashMap<String, usize>,
    register_map: &HashMap<String, usize>,
) -> ResolvedVExp {
    let operands = op_exp
        .operands
        .into_iter()
        .map(|e| resolve_value_exp(e, labels, register_map))
        .collect();
    ResolvedVExp::Op(op_exp.op_name, operands)
}

// ============================================================================
// 기계 빌더 (Machine Builder)
// ============================================================================

/// 기계 구축용 빌더 (Builder for constructing machines)
pub struct MachineBuilder {
    register_names: Vec<String>,
    operations: HashMap<String, OpFn>,
    controller: Option<Vec<Inst>>,
}

impl MachineBuilder {
    /// 새 기계 빌더 생성 (Create a new machine builder)
    pub fn new() -> Self {
        MachineBuilder {
            register_names: vec!["pc".to_string(), "flag".to_string()],
            operations: HashMap::new(),
            controller: None,
        }
    }

    /// 레지스터 추가 (Add a register)
    pub fn register(mut self, name: &str) -> Self {
        self.register_names.push(name.to_string());
        self
    }

    /// 연산 추가 (Add an operation)
    pub fn operation<F>(mut self, name: &str, op: F) -> Self
    where
        F: Fn(&[Value]) -> Value + 'static,
    {
        self.operations.insert(name.to_string(), Box::new(op));
        self
    }

    /// 컨트롤러 설정 (Set controller)
    pub fn controller(mut self, controller: Vec<Inst>) -> Self {
        self.controller = Some(controller);
        self
    }

    /// 기계 생성 (Build the machine)
    pub fn build(self) -> Machine {
        let controller = self
            .controller
            .expect("컨트롤러가 설정되지 않음 (Controller not set)");

        // 레지스터 맵 구축 (Build register map)
        let mut register_map = HashMap::new();
        for (idx, name) in self.register_names.iter().enumerate() {
            register_map.insert(name.clone(), idx);
        }

        // 레지스터 초기화 (Initialize registers)
        let registers = vec![Value::Unassigned; self.register_names.len()];

        // 내장 연산 추가 (Add built-in operations)
        let mut operations = self.operations;
        operations.insert(
            "initialize-stack".to_string(),
            Box::new(|_| Value::Unassigned),
        );

        // 명령 어셈블 (Assemble instructions)
        let instructions = assemble(controller, &register_map, &operations);

        Machine {
            registers,
            register_map,
            stack: Stack::new(),
            pc: 0,
            flag: false,
            instructions,
            operations,
            instruction_count: 0,
        }
    }
}

impl Default for MachineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 편의 함수 (Convenience Functions)
// ============================================================================

/// 주어진 명세로 기계를 생성
/// (Create a machine with the given specification)
pub fn make_machine(
    register_names: &[&str],
    operations: Vec<(&str, OpFn)>,
    controller: Vec<Inst>,
) -> Machine {
    let mut builder = MachineBuilder::new();

    for name in register_names {
        builder = builder.register(name);
    }

    for (name, op) in operations {
        builder = builder.operation(name, op);
    }

    builder.controller(controller).build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_operations() {
        let mut stack = Stack::new();
        assert!(stack.is_empty());

        stack.push(Value::Number(1));
        stack.push(Value::Number(2));
        stack.push(Value::Number(3));

        assert_eq!(stack.depth(), 3);
        assert_eq!(stack.maximum_depth(), 3);
        assert_eq!(stack.total_pushes(), 3);

        assert_eq!(stack.pop(), Value::Number(3));
        assert_eq!(stack.pop(), Value::Number(2));
        assert_eq!(stack.depth(), 1);

        stack.push(Value::Number(4));
        assert_eq!(stack.maximum_depth(), 3); // 최대값은 3 유지 (Max remains 3)

        stack.initialize();
        assert!(stack.is_empty());
        assert_eq!(stack.total_pushes(), 0);
        assert_eq!(stack.maximum_depth(), 0);
    }

    #[test]
    fn test_simple_gcd_machine() {
        let mut machine = MachineBuilder::new()
            .register("a")
            .register("b")
            .register("t")
            .operation("=", |args| Value::Bool(args[0] == args[1]))
            .operation("rem", |args| {
                if let (Value::Number(a), Value::Number(b)) = (&args[0], &args[1]) {
                    Value::Number(a % b)
                } else {
                    panic!("rem 은 숫자가 필요함 (rem requires numbers)")
                }
            })
            .controller(vec![
                Inst::Label("test-b".to_string()),
                Inst::Test(OpExp::new(
                    "=",
                    vec![VExp::Reg("b".to_string()), VExp::Const(Value::Number(0))],
                )),
                Inst::Branch("gcd-done".to_string()),
                Inst::Assign(
                    "t".to_string(),
                    VExp::Op(OpExp::new(
                        "rem",
                        vec![VExp::Reg("a".to_string()), VExp::Reg("b".to_string())],
                    )),
                ),
                Inst::Assign("a".to_string(), VExp::Reg("b".to_string())),
                Inst::Assign("b".to_string(), VExp::Reg("t".to_string())),
                Inst::Goto(GotoDest::Label("test-b".to_string())),
                Inst::Label("gcd-done".to_string()),
            ])
            .build();

        machine.set_register("a", Value::Number(206));
        machine.set_register("b", Value::Number(40));
        machine.start();

        assert_eq!(machine.get_register("a"), Value::Number(2));
        assert!(machine.instruction_count() > 0);
    }

    #[test]
    fn test_factorial_machine() {
        let mut machine = MachineBuilder::new()
            .register("n")
            .register("val")
            .register("continue")
            .operation("=", |args| Value::Bool(args[0] == args[1]))
            .operation("-", |args| {
                Value::Number(args[0].as_number() - args[1].as_number())
            })
            .operation("*", |args| {
                Value::Number(args[0].as_number() * args[1].as_number())
            })
            .controller(vec![
                Inst::Assign("continue".to_string(), VExp::Label("fact-done".to_string())),
                Inst::Label("fact-loop".to_string()),
                Inst::Test(OpExp::new(
                    "=",
                    vec![VExp::Reg("n".to_string()), VExp::Const(Value::Number(1))],
                )),
                Inst::Branch("base-case".to_string()),
                Inst::Save("continue".to_string()),
                Inst::Save("n".to_string()),
                Inst::Assign(
                    "n".to_string(),
                    VExp::Op(OpExp::new(
                        "-",
                        vec![VExp::Reg("n".to_string()), VExp::Const(Value::Number(1))],
                    )),
                ),
                Inst::Assign(
                    "continue".to_string(),
                    VExp::Label("after-fact".to_string()),
                ),
                Inst::Goto(GotoDest::Label("fact-loop".to_string())),
                Inst::Label("after-fact".to_string()),
                Inst::Restore("n".to_string()),
                Inst::Restore("continue".to_string()),
                Inst::Assign(
                    "val".to_string(),
                    VExp::Op(OpExp::new(
                        "*",
                        vec![VExp::Reg("n".to_string()), VExp::Reg("val".to_string())],
                    )),
                ),
                Inst::Goto(GotoDest::Reg("continue".to_string())),
                Inst::Label("base-case".to_string()),
                Inst::Assign("val".to_string(), VExp::Const(Value::Number(1))),
                Inst::Goto(GotoDest::Reg("continue".to_string())),
                Inst::Label("fact-done".to_string()),
            ])
            .build();

        // 5! 테스트 (Test factorial of 5)
        machine.set_register("n", Value::Number(5));
        machine.start();
        assert_eq!(machine.get_register("val"), Value::Number(120));

        let (pushes, max_depth) = machine.stack_statistics();
        // n=5: pushes는 2*(n-1)=8, max_depth도 2*(n-1)=8
        // (For n=5: pushes should be 2*(n-1) = 8, max_depth should be 2*(n-1) = 8)
        assert_eq!(pushes, 8);
        assert_eq!(max_depth, 8);
    }

    #[test]
    fn test_fibonacci_machine() {
        let mut machine = MachineBuilder::new()
            .register("n")
            .register("val")
            .register("continue")
            .operation("<", |args| {
                Value::Bool(args[0].as_number() < args[1].as_number())
            })
            .operation("-", |args| {
                Value::Number(args[0].as_number() - args[1].as_number())
            })
            .operation("+", |args| {
                Value::Number(args[0].as_number() + args[1].as_number())
            })
            .controller(vec![
                Inst::Assign("continue".to_string(), VExp::Label("fib-done".to_string())),
                Inst::Label("fib-loop".to_string()),
                Inst::Test(OpExp::new(
                    "<",
                    vec![VExp::Reg("n".to_string()), VExp::Const(Value::Number(2))],
                )),
                Inst::Branch("immediate-answer".to_string()),
                Inst::Save("continue".to_string()),
                Inst::Assign(
                    "continue".to_string(),
                    VExp::Label("afterfib-n-1".to_string()),
                ),
                Inst::Save("n".to_string()),
                Inst::Assign(
                    "n".to_string(),
                    VExp::Op(OpExp::new(
                        "-",
                        vec![VExp::Reg("n".to_string()), VExp::Const(Value::Number(1))],
                    )),
                ),
                Inst::Goto(GotoDest::Label("fib-loop".to_string())),
                Inst::Label("afterfib-n-1".to_string()),
                Inst::Restore("n".to_string()),
                Inst::Save("val".to_string()),
                Inst::Assign(
                    "n".to_string(),
                    VExp::Op(OpExp::new(
                        "-",
                        vec![VExp::Reg("n".to_string()), VExp::Const(Value::Number(2))],
                    )),
                ),
                Inst::Assign(
                    "continue".to_string(),
                    VExp::Label("afterfib-n-2".to_string()),
                ),
                Inst::Goto(GotoDest::Label("fib-loop".to_string())),
                Inst::Label("afterfib-n-2".to_string()),
                Inst::Restore("n".to_string()),
                Inst::Restore("continue".to_string()),
                Inst::Assign(
                    "val".to_string(),
                    VExp::Op(OpExp::new(
                        "+",
                        vec![VExp::Reg("val".to_string()), VExp::Reg("n".to_string())],
                    )),
                ),
                Inst::Goto(GotoDest::Reg("continue".to_string())),
                Inst::Label("immediate-answer".to_string()),
                Inst::Assign("val".to_string(), VExp::Reg("n".to_string())),
                Inst::Goto(GotoDest::Reg("continue".to_string())),
                Inst::Label("fib-done".to_string()),
            ])
            .build();

        // 피보나치 5 테스트 (결과 5) (Test fibonacci of 5 (should be 5))
        machine.set_register("n", Value::Number(5));
        machine.start();
        assert_eq!(machine.get_register("val"), Value::Number(5));
    }

    #[test]
    #[should_panic(expected = "중복 정의된 라벨 (Multiply defined label)")]
    fn test_duplicate_label_detection() {
        MachineBuilder::new()
            .register("a")
            .controller(vec![
                Inst::Label("start".to_string()),
                Inst::Assign("a".to_string(), VExp::Const(Value::Number(3))),
                Inst::Label("start".to_string()), // 중복! (Duplicate!)
            ])
            .build();
    }

    #[test]
    fn test_instruction_counting() {
        let mut machine = MachineBuilder::new()
            .register("a")
            .operation("+", |args| {
                Value::Number(args[0].as_number() + args[1].as_number())
            })
            .controller(vec![
                Inst::Assign("a".to_string(), VExp::Const(Value::Number(0))),
                Inst::Assign(
                    "a".to_string(),
                    VExp::Op(OpExp::new(
                        "+",
                        vec![VExp::Reg("a".to_string()), VExp::Const(Value::Number(1))],
                    )),
                ),
                Inst::Assign(
                    "a".to_string(),
                    VExp::Op(OpExp::new(
                        "+",
                        vec![VExp::Reg("a".to_string()), VExp::Const(Value::Number(1))],
                    )),
                ),
            ])
            .build();

        machine.start();
        assert_eq!(machine.instruction_count(), 3);
        assert_eq!(machine.get_register("a"), Value::Number(2));
    }
}
