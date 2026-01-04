//! SICP Section 5.2: A Register-Machine Simulator
//!
//! This module implements a simulator for register machines, allowing us to
//! test machine designs and measure their performance characteristics.
//!
//! # Architecture
//!
//! The simulator consists of several key components:
//!
//! - **Machine**: The main structure containing registers, stack, and instruction sequence
//! - **Assembler**: Converts controller text into executable instructions
//! - **Instructions**: Parsed and resolved instruction representations
//! - **Stack**: Tracks values with performance monitoring
//!
//! # Memory Model
//!
//! ```text
//! Machine (owns all state)
//!   ├── registers: HashMap<String, Value>
//!   ├── stack: Stack (Vec<Value> + statistics)
//!   ├── instructions: Vec<Instruction>
//!   ├── pc: usize (program counter as index)
//!   └── flag: bool (for test/branch)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use sicp_chapter5::section_5_2::*;
//!
//! // Create a simple GCD machine
//! let mut machine = MachineBuilder::new()
//!     .register("a")
//!     .register("b")
//!     .register("t")
//!     .operation("=", |args| Value::Bool(args[0] == args[1]))
//!     .operation("rem", |args| {
//!         if let (Value::Number(a), Value::Number(b)) = (&args[0], &args[1]) {
//!             Value::Number(a % b)
//!         } else {
//!             panic!("rem requires numbers")
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
// Value Type
// ============================================================================

/// Values that can be stored in registers or on the stack
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Unassigned register value
    Unassigned,
    /// Integer number
    Number(i64),
    /// Boolean value (for test results)
    Bool(bool),
    /// Instruction pointer (for label values)
    InstructionPointer(usize),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Unassigned => write!(f, "*unassigned*"),
            Value::Number(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::InstructionPointer(ip) => write!(f, "@{}", ip),
        }
    }
}

impl Value {
    /// Convert value to boolean for branching
    pub fn as_bool(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Number(0) => false,
            Value::Number(_) => true,
            _ => false,
        }
    }

    /// Extract number or panic
    pub fn as_number(&self) -> i64 {
        match self {
            Value::Number(n) => *n,
            _ => panic!("Expected number, got {:?}", self),
        }
    }
}

// ============================================================================
// Stack with Performance Monitoring
// ============================================================================

/// Stack that tracks performance statistics
#[derive(Debug)]
pub struct Stack {
    data: Vec<Value>,
    pushes: usize,
    max_depth: usize,
}

impl Stack {
    /// Create a new empty stack
    pub fn new() -> Self {
        Stack {
            data: Vec::new(),
            pushes: 0,
            max_depth: 0,
        }
    }

    /// Push a value onto the stack
    pub fn push(&mut self, value: Value) {
        self.data.push(value);
        self.pushes += 1;
        if self.data.len() > self.max_depth {
            self.max_depth = self.data.len();
        }
    }

    /// Pop a value from the stack
    pub fn pop(&mut self) -> Value {
        self.data.pop().expect("Empty stack: POP")
    }

    /// Check if stack is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get current depth
    pub fn depth(&self) -> usize {
        self.data.len()
    }

    /// Initialize stack (clear and reset statistics)
    pub fn initialize(&mut self) {
        self.data.clear();
        self.pushes = 0;
        self.max_depth = 0;
    }

    /// Get total number of pushes
    pub fn total_pushes(&self) -> usize {
        self.pushes
    }

    /// Get maximum depth reached
    pub fn maximum_depth(&self) -> usize {
        self.max_depth
    }

    /// Print stack statistics
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
// Value Expressions (before resolution)
// ============================================================================

/// Value expression (unresolved)
#[derive(Debug, Clone, PartialEq)]
pub enum VExp {
    /// Constant value
    Const(Value),
    /// Register reference
    Reg(String),
    /// Label reference
    Label(String),
    /// Operation application
    Op(OpExp),
}

/// Operation expression
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
// Resolved Value Expressions
// ============================================================================

/// Resolved value expression (after assembly)
#[derive(Debug, Clone)]
enum ResolvedVExp {
    /// Constant value
    Const(Value),
    /// Register index
    Reg(usize),
    /// Label (instruction pointer)
    Label(usize),
    /// Operation with resolved operands
    Op(String, Vec<ResolvedVExp>),
}

// ============================================================================
// Instructions (before resolution)
// ============================================================================

/// Instruction text (before assembly)
#[derive(Debug, Clone, PartialEq)]
pub enum Inst {
    /// Label marker
    Label(String),
    /// Assign value to register
    Assign(String, VExp),
    /// Test condition and set flag
    Test(OpExp),
    /// Branch if flag is true
    Branch(String),
    /// Goto label or register
    Goto(GotoDest),
    /// Save register to stack
    Save(String),
    /// Restore register from stack
    Restore(String),
    /// Perform operation (for side effects)
    Perform(OpExp),
}

/// Goto destination
#[derive(Debug, Clone, PartialEq)]
pub enum GotoDest {
    Label(String),
    Reg(String),
}

// ============================================================================
// Resolved Instructions
// ============================================================================

/// Resolved instruction (after assembly)
#[derive(Debug, Clone)]
enum ResolvedInst {
    /// Assign value to register
    Assign {
        target_reg: usize,
        value: ResolvedVExp,
    },
    /// Test condition and set flag
    Test { condition: ResolvedVExp },
    /// Branch if flag is true
    Branch { destination: usize },
    /// Goto destination
    Goto { destination: GotoDestResolved },
    /// Save register to stack
    Save { reg: usize },
    /// Restore register from stack
    Restore { reg: usize },
    /// Perform operation
    Perform { action: ResolvedVExp },
}

#[derive(Debug, Clone)]
enum GotoDestResolved {
    Label(usize),
    Reg(usize),
}

// ============================================================================
// Machine
// ============================================================================

/// Operation function type
pub type OpFn = Box<dyn Fn(&[Value]) -> Value>;

/// Register machine
pub struct Machine {
    /// Register storage (indexed by register index)
    registers: Vec<Value>,
    /// Register name to index mapping
    register_map: HashMap<String, usize>,
    /// Stack
    stack: Stack,
    /// Program counter (index into instructions)
    pc: usize,
    /// Flag register (for test/branch)
    flag: bool,
    /// Instruction sequence
    instructions: Vec<ResolvedInst>,
    /// Operations table
    operations: HashMap<String, OpFn>,
    /// Instruction count
    instruction_count: usize,
}

impl Machine {
    /// Execute the machine starting from the beginning
    pub fn start(&mut self) {
        self.pc = 0;
        self.instruction_count = 0;
        self.execute();
    }

    /// Execute instructions until done
    fn execute(&mut self) {
        while self.pc < self.instructions.len() {
            self.instruction_count += 1;
            let inst = self.instructions[self.pc].clone();
            self.execute_instruction(inst);
        }
    }

    /// Execute a single instruction
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
                        panic!("Goto register must contain instruction pointer");
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

    /// Evaluate a resolved value expression
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
                    .unwrap_or_else(|| panic!("Unknown operation: {}", op_name));
                op(&args)
            }
        }
    }

    /// Get register value by name
    pub fn get_register(&self, name: &str) -> Value {
        let idx = self
            .register_map
            .get(name)
            .unwrap_or_else(|| panic!("Unknown register: {}", name));
        self.registers[*idx].clone()
    }

    /// Set register value by name
    pub fn set_register(&mut self, name: &str, value: Value) {
        let idx = self
            .register_map
            .get(name)
            .unwrap_or_else(|| panic!("Unknown register: {}", name));
        self.registers[*idx] = value;
    }

    /// Get stack statistics
    pub fn stack_statistics(&self) -> (usize, usize) {
        (self.stack.total_pushes(), self.stack.maximum_depth())
    }

    /// Print stack statistics
    pub fn print_stack_statistics(&self) {
        self.stack.print_statistics();
    }

    /// Get instruction count
    pub fn instruction_count(&self) -> usize {
        self.instruction_count
    }

    /// Reset instruction count
    pub fn reset_instruction_count(&mut self) {
        self.instruction_count = 0;
    }

    /// Initialize stack
    pub fn initialize_stack(&mut self) {
        self.stack.initialize();
    }
}

// ============================================================================
// Assembler
// ============================================================================

/// Assemble controller text into executable instructions
fn assemble(
    controller: Vec<Inst>,
    register_map: &HashMap<String, usize>,
    operations: &HashMap<String, OpFn>,
) -> Vec<ResolvedInst> {
    // Pass 1: Extract labels and build instruction list
    let (insts, labels) = extract_labels(controller);

    // Pass 2: Resolve all references and create execution procedures
    update_insts(insts, &labels, register_map, operations)
}

/// Extract labels from controller text
fn extract_labels(text: Vec<Inst>) -> (Vec<Inst>, HashMap<String, usize>) {
    let mut instructions = Vec::new();
    let mut labels = HashMap::new();

    for inst in text {
        match inst {
            Inst::Label(name) => {
                if labels.contains_key(&name) {
                    panic!("Multiply defined label: {}", name);
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

/// Update instructions with resolved references
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

/// Resolve a single instruction
fn resolve_instruction(
    inst: Inst,
    labels: &HashMap<String, usize>,
    register_map: &HashMap<String, usize>,
) -> ResolvedInst {
    match inst {
        Inst::Assign(reg_name, value_exp) => {
            let target_reg = *register_map
                .get(&reg_name)
                .unwrap_or_else(|| panic!("Unknown register: {}", reg_name));
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
                .unwrap_or_else(|| panic!("Undefined label: {}", label_name));
            ResolvedInst::Branch { destination }
        }
        Inst::Goto(dest) => {
            let destination = match dest {
                GotoDest::Label(label_name) => {
                    let ip = *labels
                        .get(&label_name)
                        .unwrap_or_else(|| panic!("Undefined label: {}", label_name));
                    GotoDestResolved::Label(ip)
                }
                GotoDest::Reg(reg_name) => {
                    let idx = *register_map
                        .get(&reg_name)
                        .unwrap_or_else(|| panic!("Unknown register: {}", reg_name));
                    GotoDestResolved::Reg(idx)
                }
            };
            ResolvedInst::Goto { destination }
        }
        Inst::Save(reg_name) => {
            let reg = *register_map
                .get(&reg_name)
                .unwrap_or_else(|| panic!("Unknown register: {}", reg_name));
            ResolvedInst::Save { reg }
        }
        Inst::Restore(reg_name) => {
            let reg = *register_map
                .get(&reg_name)
                .unwrap_or_else(|| panic!("Unknown register: {}", reg_name));
            ResolvedInst::Restore { reg }
        }
        Inst::Perform(op_exp) => {
            let action = resolve_op_exp(op_exp, labels, register_map);
            ResolvedInst::Perform { action }
        }
        Inst::Label(_) => panic!("Labels should have been filtered out"),
    }
}

/// Resolve a value expression
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
                .unwrap_or_else(|| panic!("Unknown register: {}", name));
            ResolvedVExp::Reg(idx)
        }
        VExp::Label(name) => {
            let ip = *labels
                .get(&name)
                .unwrap_or_else(|| panic!("Undefined label: {}", name));
            ResolvedVExp::Label(ip)
        }
        VExp::Op(op_exp) => resolve_op_exp(op_exp, labels, register_map),
    }
}

/// Resolve an operation expression
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
// Machine Builder
// ============================================================================

/// Builder for constructing machines
pub struct MachineBuilder {
    register_names: Vec<String>,
    operations: HashMap<String, OpFn>,
    controller: Option<Vec<Inst>>,
}

impl MachineBuilder {
    /// Create a new machine builder
    pub fn new() -> Self {
        MachineBuilder {
            register_names: vec!["pc".to_string(), "flag".to_string()],
            operations: HashMap::new(),
            controller: None,
        }
    }

    /// Add a register
    pub fn register(mut self, name: &str) -> Self {
        self.register_names.push(name.to_string());
        self
    }

    /// Add an operation
    pub fn operation<F>(mut self, name: &str, op: F) -> Self
    where
        F: Fn(&[Value]) -> Value + 'static,
    {
        self.operations.insert(name.to_string(), Box::new(op));
        self
    }

    /// Set controller
    pub fn controller(mut self, controller: Vec<Inst>) -> Self {
        self.controller = Some(controller);
        self
    }

    /// Build the machine
    pub fn build(self) -> Machine {
        let controller = self.controller.expect("Controller not set");

        // Build register map
        let mut register_map = HashMap::new();
        for (idx, name) in self.register_names.iter().enumerate() {
            register_map.insert(name.clone(), idx);
        }

        // Initialize registers
        let registers = vec![Value::Unassigned; self.register_names.len()];

        // Add built-in operations
        let mut operations = self.operations;
        operations.insert(
            "initialize-stack".to_string(),
            Box::new(|_| Value::Unassigned),
        );

        // Assemble instructions
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
// Convenience Functions
// ============================================================================

/// Create a machine with the given specification
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
        assert_eq!(stack.maximum_depth(), 3); // Max remains 3

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
                    panic!("rem requires numbers")
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

        // Test factorial of 5
        machine.set_register("n", Value::Number(5));
        machine.start();
        assert_eq!(machine.get_register("val"), Value::Number(120));

        let (pushes, max_depth) = machine.stack_statistics();
        // For n=5: pushes should be 2*(n-1) = 8, max_depth should be 2*(n-1) = 8
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

        // Test fibonacci of 5 (should be 5)
        machine.set_register("n", Value::Number(5));
        machine.start();
        assert_eq!(machine.get_register("val"), Value::Number(5));
    }

    #[test]
    #[should_panic(expected = "Multiply defined label")]
    fn test_duplicate_label_detection() {
        MachineBuilder::new()
            .register("a")
            .controller(vec![
                Inst::Label("start".to_string()),
                Inst::Assign("a".to_string(), VExp::Const(Value::Number(3))),
                Inst::Label("start".to_string()), // Duplicate!
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
