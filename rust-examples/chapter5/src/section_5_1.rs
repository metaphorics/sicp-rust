//! Section 5.1: Designing Register Machines
//!
//! This module implements register machines as described in SICP Chapter 5.1.
//! A register machine consists of:
//! - Registers: Named storage locations holding integer values
//! - Stack: LIFO data structure for save/restore operations
//! - Instructions: Sequence of operations (assign, test, branch, goto, save, restore, perform)
//! - Operations: Primitive operations like arithmetic and comparisons
//!
//! ## Rust Mapping
//!
//! | Scheme Concept | Rust Implementation |
//! |----------------|---------------------|
//! | Register machine language | `Instruction` enum with variants |
//! | Registers | `HashMap<String, i64>` |
//! | Stack | `Vec<i64>` |
//! | Controller | `Vec<Instruction>` with labels |
//! | Operations | Function pointers `fn(&[i64]) -> i64` |
//!
//! ## Memory Layout
//!
//! ```text
//! Machine {
//!     registers: HashMap<String, i64>  // Owned values
//!     stack: Vec<i64>                   // Owned stack
//!     instructions: Vec<Instruction>    // Owned instruction sequence
//!     operations: HashMap<String, fn(&[i64]) -> i64>  // Function pointers
//!     labels: HashMap<String, usize>    // Label -> PC mapping
//!     pc: usize                         // Program counter
//!     test_flag: bool                   // Result of last test
//! }
//! ```

use std::collections::HashMap;

/// Register values are 64-bit signed integers
pub type Value = i64;

/// Operation function type: takes slice of values, returns single value
/// Boxed to allow different closure types in the same HashMap
pub type Operation = Box<dyn Fn(&[Value]) -> Value>;

/// Source of a value in the register machine
#[derive(Debug, Clone, PartialEq)]
pub enum Source {
    /// Value from a register: (reg a)
    Reg(String),
    /// Constant value: (const 0)
    Const(Value),
    /// Result of operation: (op rem) with inputs
    Op(String, Vec<Source>),
    /// Label reference (for assign continue): (label after-gcd)
    Label(String),
}

/// Target for goto instruction
#[derive(Debug, Clone, PartialEq)]
pub enum Target {
    /// Jump to label: (goto (label test-b))
    Label(String),
    /// Jump to address in register: (goto (reg continue))
    Reg(String),
}

/// Register machine instructions
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    /// Assign value to register: (assign a (reg b))
    Assign(String, Source),
    /// Test condition: (test (op =) (reg b) (const 0))
    Test(Source),
    /// Branch if test succeeded: (branch (label gcd-done))
    Branch(String),
    /// Unconditional jump: (goto (label test-b))
    Goto(Target),
    /// Save register to stack: (save n)
    Save(String),
    /// Restore register from stack: (restore n)
    Restore(String),
    /// Perform action (side effect): (perform (op print) (reg a))
    Perform(Source),
}

/// A register machine with state and controller
pub struct Machine {
    /// Register bank: register name -> value
    registers: HashMap<String, Value>,
    /// Stack for save/restore operations
    stack: Vec<Value>,
    /// Instruction sequence (with optional labels)
    instructions: Vec<Instruction>,
    /// Available operations: name -> function
    operations: HashMap<String, Operation>,
    /// Label map: label name -> instruction index
    labels: HashMap<String, usize>,
    /// Program counter
    pc: usize,
    /// Result of last test instruction
    test_flag: bool,
    /// Halted flag
    halted: bool,
}

impl Machine {
    /// Create a new register machine
    ///
    /// # Arguments
    /// * `register_names` - Names of registers to initialize (all start at 0)
    /// * `operations` - Map of operation names to functions
    /// * `instructions` - Instruction sequence with embedded labels
    ///
    /// # Example
    /// ```ignore
    /// use sicp_chapter5::section_5_1::{Machine, Instruction, Source, Target, Operation};
    /// use std::collections::HashMap;
    ///
    /// let mut ops = HashMap::new();
    /// ops.insert(
    ///     "=".to_string(),
    ///     Box::new(|args: &[i64]| (args[0] == args[1]) as i64) as Operation,
    /// );
    ///
    /// let machine = Machine::new(
    ///     vec!["a".to_string(), "b".to_string()],
    ///     ops,
    ///     vec![
    ///         Instruction::Test(Source::Op("=".to_string(), vec![
    ///             Source::Reg("b".to_string()),
    ///             Source::Const(0),
    ///         ])),
    ///     ],
    /// );
    /// ```
    pub fn new(
        register_names: Vec<String>,
        operations: HashMap<String, Operation>,
        instructions: Vec<Instruction>,
    ) -> Self {
        let mut registers = HashMap::new();
        for name in register_names {
            registers.insert(name, 0);
        }

        // Build label map by scanning instructions
        let labels = HashMap::new(); // Labels are embedded in instruction flow

        Machine {
            registers,
            stack: Vec::new(),
            instructions,
            operations,
            labels,
            pc: 0,
            test_flag: false,
            halted: false,
        }
    }

    /// Create machine with explicit label positions
    pub fn new_with_labels(
        register_names: Vec<String>,
        operations: HashMap<String, Operation>,
        instructions: Vec<Instruction>,
        labels: HashMap<String, usize>,
    ) -> Self {
        let mut machine = Self::new(register_names, operations, instructions);
        machine.labels = labels;
        machine
    }

    /// Set a register's value
    pub fn set_register(&mut self, name: &str, value: Value) {
        if let Some(reg) = self.registers.get_mut(name) {
            *reg = value;
        } else {
            panic!("Unknown register: {}", name);
        }
    }

    /// Get a register's value
    pub fn get_register(&self, name: &str) -> Value {
        *self
            .registers
            .get(name)
            .unwrap_or_else(|| panic!("Unknown register: {}", name))
    }

    /// Execute the machine until halt
    pub fn run(&mut self) {
        self.pc = 0;
        self.halted = false;

        while !self.halted && self.pc < self.instructions.len() {
            let instruction = self.instructions[self.pc].clone();
            self.execute(&instruction);
        }
    }

    /// Execute a single instruction
    fn execute(&mut self, instruction: &Instruction) {
        match instruction {
            Instruction::Assign(reg, source) => {
                let value = self.eval_source(source);
                self.set_register(reg, value);
                self.pc += 1;
            }
            Instruction::Test(source) => {
                let result = self.eval_source(source);
                self.test_flag = result != 0;
                self.pc += 1;
            }
            Instruction::Branch(label) => {
                if self.test_flag {
                    self.pc = *self
                        .labels
                        .get(label)
                        .unwrap_or_else(|| panic!("Unknown label: {}", label));
                } else {
                    self.pc += 1;
                }
            }
            Instruction::Goto(target) => match target {
                Target::Label(label) => {
                    self.pc = *self
                        .labels
                        .get(label)
                        .unwrap_or_else(|| panic!("Unknown label: {}", label));
                }
                Target::Reg(reg) => {
                    let addr = self.get_register(reg) as usize;
                    self.pc = addr;
                }
            },
            Instruction::Save(reg) => {
                let value = self.get_register(reg);
                self.stack.push(value);
                self.pc += 1;
            }
            Instruction::Restore(reg) => {
                let value = self.stack.pop().expect("Stack underflow");
                self.set_register(reg, value);
                self.pc += 1;
            }
            Instruction::Perform(source) => {
                self.eval_source(source);
                self.pc += 1;
            }
        }

        // Check for halt condition (pc beyond instructions)
        if self.pc >= self.instructions.len() {
            self.halted = true;
        }
    }

    /// Evaluate a source to get its value
    fn eval_source(&self, source: &Source) -> Value {
        match source {
            Source::Reg(name) => self.get_register(name),
            Source::Const(val) => *val,
            Source::Op(op_name, inputs) => {
                let op = self
                    .operations
                    .get(op_name)
                    .unwrap_or_else(|| panic!("Unknown operation: {}", op_name));
                let args: Vec<Value> = inputs.iter().map(|src| self.eval_source(src)).collect();
                op(&args)
            }
            Source::Label(label) => {
                // Return label address as value (for assign continue)
                *self
                    .labels
                    .get(label)
                    .unwrap_or_else(|| panic!("Unknown label: {}", label)) as Value
            }
        }
    }

    /// Get current stack depth (for debugging/testing)
    pub fn stack_depth(&self) -> usize {
        self.stack.len()
    }

    /// Check if machine has halted
    pub fn is_halted(&self) -> bool {
        self.halted
    }
}

/// Build the GCD machine from Figure 5.4
///
/// Computes GCD of values in registers a and b.
/// Result is left in register a.
///
/// Controller:
/// ```scheme
/// test-b
///   (test (op =) (reg b) (const 0))
///   (branch (label gcd-done))
///   (assign t (op rem) (reg a) (reg b))
///   (assign a (reg b))
///   (assign b (reg t))
///   (goto (label test-b))
/// gcd-done
/// ```
pub fn make_gcd_machine() -> Machine {
    let mut ops = HashMap::new();
    ops.insert(
        "=".to_string(),
        Box::new(|args: &[Value]| (args[0] == args[1]) as Value) as Operation,
    );
    ops.insert(
        "rem".to_string(),
        Box::new(|args: &[Value]| args[0] % args[1]) as Operation,
    );

    let instructions = vec![
        // test-b (label at index 0)
        Instruction::Test(Source::Op(
            "=".to_string(),
            vec![Source::Reg("b".to_string()), Source::Const(0)],
        )),
        Instruction::Branch("gcd-done".to_string()),
        Instruction::Assign(
            "t".to_string(),
            Source::Op(
                "rem".to_string(),
                vec![Source::Reg("a".to_string()), Source::Reg("b".to_string())],
            ),
        ),
        Instruction::Assign("a".to_string(), Source::Reg("b".to_string())),
        Instruction::Assign("b".to_string(), Source::Reg("t".to_string())),
        Instruction::Goto(Target::Label("test-b".to_string())),
        // gcd-done (label at index 6)
    ];

    let mut labels = HashMap::new();
    labels.insert("test-b".to_string(), 0);
    labels.insert("gcd-done".to_string(), 6);

    Machine::new_with_labels(
        vec!["a".to_string(), "b".to_string(), "t".to_string()],
        ops,
        instructions,
        labels,
    )
}

/// Build iterative factorial machine (Exercise 5.1)
///
/// Computes factorial of n using iterative algorithm:
/// ```scheme
/// (define (factorial n)
///   (define (iter product counter)
///     (if (> counter n)
///         product
///         (iter (* counter product) (+ counter 1))))
///   (iter 1 1))
/// ```
///
/// Registers: n, product, counter
/// Result in product register.
pub fn make_factorial_iterative_machine() -> Machine {
    let mut ops = HashMap::new();
    ops.insert(
        ">".to_string(),
        Box::new(|args: &[Value]| (args[0] > args[1]) as Value) as Operation,
    );
    ops.insert(
        "*".to_string(),
        Box::new(|args: &[Value]| args[0] * args[1]) as Operation,
    );
    ops.insert(
        "+".to_string(),
        Box::new(|args: &[Value]| args[0] + args[1]) as Operation,
    );

    let instructions = vec![
        // Initialize product = 1, counter = 1
        Instruction::Assign("product".to_string(), Source::Const(1)),
        Instruction::Assign("counter".to_string(), Source::Const(1)),
        // fact-loop (label at index 2)
        Instruction::Test(Source::Op(
            ">".to_string(),
            vec![
                Source::Reg("counter".to_string()),
                Source::Reg("n".to_string()),
            ],
        )),
        Instruction::Branch("fact-done".to_string()),
        Instruction::Assign(
            "product".to_string(),
            Source::Op(
                "*".to_string(),
                vec![
                    Source::Reg("counter".to_string()),
                    Source::Reg("product".to_string()),
                ],
            ),
        ),
        Instruction::Assign(
            "counter".to_string(),
            Source::Op(
                "+".to_string(),
                vec![Source::Reg("counter".to_string()), Source::Const(1)],
            ),
        ),
        Instruction::Goto(Target::Label("fact-loop".to_string())),
        // fact-done (label at index 7)
    ];

    let mut labels = HashMap::new();
    labels.insert("fact-loop".to_string(), 2);
    labels.insert("fact-done".to_string(), 7);

    Machine::new_with_labels(
        vec![
            "n".to_string(),
            "product".to_string(),
            "counter".to_string(),
        ],
        ops,
        instructions,
        labels,
    )
}

/// Build recursive factorial machine (Figure 5.11)
///
/// Computes factorial recursively using stack.
/// ```scheme
/// (define (factorial n)
///   (if (= n 1)
///       1
///       (* (factorial (- n 1)) n)))
/// ```
///
/// Registers: n, val, continue
/// Uses stack to save n and continue.
/// Result in val register.
pub fn make_factorial_recursive_machine() -> Machine {
    let mut ops = HashMap::new();
    ops.insert(
        "=".to_string(),
        Box::new(|args: &[Value]| (args[0] == args[1]) as Value) as Operation,
    );
    ops.insert(
        "-".to_string(),
        Box::new(|args: &[Value]| args[0] - args[1]) as Operation,
    );
    ops.insert(
        "*".to_string(),
        Box::new(|args: &[Value]| args[0] * args[1]) as Operation,
    );

    let instructions = vec![
        // Initialize continue to fact-done
        Instruction::Assign(
            "continue".to_string(),
            Source::Label("fact-done".to_string()),
        ),
        // fact-loop (label at index 1)
        Instruction::Test(Source::Op(
            "=".to_string(),
            vec![Source::Reg("n".to_string()), Source::Const(1)],
        )),
        Instruction::Branch("base-case".to_string()),
        // Recursive case: save continue and n
        Instruction::Save("continue".to_string()),
        Instruction::Save("n".to_string()),
        Instruction::Assign(
            "n".to_string(),
            Source::Op(
                "-".to_string(),
                vec![Source::Reg("n".to_string()), Source::Const(1)],
            ),
        ),
        Instruction::Assign(
            "continue".to_string(),
            Source::Label("after-fact".to_string()),
        ),
        Instruction::Goto(Target::Label("fact-loop".to_string())),
        // after-fact (label at index 8)
        Instruction::Restore("n".to_string()),
        Instruction::Restore("continue".to_string()),
        Instruction::Assign(
            "val".to_string(),
            Source::Op(
                "*".to_string(),
                vec![Source::Reg("n".to_string()), Source::Reg("val".to_string())],
            ),
        ),
        Instruction::Goto(Target::Reg("continue".to_string())),
        // base-case (label at index 12)
        Instruction::Assign("val".to_string(), Source::Const(1)),
        Instruction::Goto(Target::Reg("continue".to_string())),
        // fact-done (label at index 14)
    ];

    let mut labels = HashMap::new();
    labels.insert("fact-loop".to_string(), 1);
    labels.insert("base-case".to_string(), 12);
    labels.insert("after-fact".to_string(), 8);
    labels.insert("fact-done".to_string(), 14);

    Machine::new_with_labels(
        vec!["n".to_string(), "val".to_string(), "continue".to_string()],
        ops,
        instructions,
        labels,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd_machine() {
        let mut machine = make_gcd_machine();
        machine.set_register("a", 206);
        machine.set_register("b", 40);
        machine.run();
        assert_eq!(machine.get_register("a"), 2);
    }

    #[test]
    fn test_gcd_machine_coprime() {
        let mut machine = make_gcd_machine();
        machine.set_register("a", 17);
        machine.set_register("b", 13);
        machine.run();
        assert_eq!(machine.get_register("a"), 1);
    }

    #[test]
    fn test_gcd_machine_equal() {
        let mut machine = make_gcd_machine();
        machine.set_register("a", 12);
        machine.set_register("b", 12);
        machine.run();
        assert_eq!(machine.get_register("a"), 12);
    }

    #[test]
    fn test_factorial_iterative() {
        let mut machine = make_factorial_iterative_machine();
        machine.set_register("n", 5);
        machine.run();
        assert_eq!(machine.get_register("product"), 120);
    }

    #[test]
    fn test_factorial_iterative_base_case() {
        let mut machine = make_factorial_iterative_machine();
        machine.set_register("n", 1);
        machine.run();
        assert_eq!(machine.get_register("product"), 1);
    }

    #[test]
    fn test_factorial_iterative_zero() {
        let mut machine = make_factorial_iterative_machine();
        machine.set_register("n", 0);
        machine.run();
        assert_eq!(machine.get_register("product"), 1);
    }

    #[test]
    fn test_factorial_recursive() {
        let mut machine = make_factorial_recursive_machine();
        machine.set_register("n", 5);
        machine.run();
        assert_eq!(machine.get_register("val"), 120);
        assert_eq!(machine.stack_depth(), 0); // Stack should be empty after completion
    }

    #[test]
    fn test_factorial_recursive_base_case() {
        let mut machine = make_factorial_recursive_machine();
        machine.set_register("n", 1);
        machine.run();
        assert_eq!(machine.get_register("val"), 1);
        assert_eq!(machine.stack_depth(), 0);
    }

    #[test]
    fn test_factorial_recursive_large() {
        let mut machine = make_factorial_recursive_machine();
        machine.set_register("n", 10);
        machine.run();
        assert_eq!(machine.get_register("val"), 3628800);
        assert_eq!(machine.stack_depth(), 0);
    }

    #[test]
    fn test_stack_operations() {
        let mut ops = HashMap::new();
        ops.insert(
            "=".to_string(),
            Box::new(|args: &[Value]| (args[0] == args[1]) as Value) as Operation,
        );

        let instructions = vec![
            Instruction::Assign("a".to_string(), Source::Const(42)),
            Instruction::Save("a".to_string()),
            Instruction::Assign("a".to_string(), Source::Const(99)),
            Instruction::Restore("b".to_string()),
        ];

        let mut machine = Machine::new(vec!["a".to_string(), "b".to_string()], ops, instructions);

        machine.run();
        assert_eq!(machine.get_register("a"), 99);
        assert_eq!(machine.get_register("b"), 42);
        assert_eq!(machine.stack_depth(), 0);
    }

    #[test]
    fn test_branch_instruction() {
        let mut ops = HashMap::new();
        ops.insert(
            "=".to_string(),
            Box::new(|args: &[Value]| (args[0] == args[1]) as Value) as Operation,
        );

        let instructions = vec![
            Instruction::Test(Source::Op(
                "=".to_string(),
                vec![Source::Reg("a".to_string()), Source::Const(0)],
            )),
            Instruction::Branch("skip".to_string()),
            Instruction::Assign("b".to_string(), Source::Const(99)), // Skipped if a == 0
            // skip (index 3)
            Instruction::Assign("c".to_string(), Source::Const(1)),
        ];

        let mut labels = HashMap::new();
        labels.insert("skip".to_string(), 3);

        let mut machine = Machine::new_with_labels(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            ops,
            instructions,
            labels,
        );

        machine.set_register("a", 0);
        machine.run();
        assert_eq!(machine.get_register("b"), 0); // Not modified
        assert_eq!(machine.get_register("c"), 1);
    }

    #[test]
    fn test_goto_register() {
        let mut ops = HashMap::new();
        ops.insert(
            "=".to_string(),
            Box::new(|args: &[Value]| (args[0] == args[1]) as Value) as Operation,
        );

        let instructions = vec![
            Instruction::Assign(
                "target".to_string(),
                Source::Label("destination".to_string()),
            ),
            Instruction::Goto(Target::Reg("target".to_string())),
            Instruction::Assign("a".to_string(), Source::Const(99)), // Skipped
            // destination (index 3)
            Instruction::Assign("b".to_string(), Source::Const(42)),
        ];

        let mut labels = HashMap::new();
        labels.insert("destination".to_string(), 3);

        let mut machine = Machine::new_with_labels(
            vec!["a".to_string(), "b".to_string(), "target".to_string()],
            ops,
            instructions,
            labels,
        );

        machine.run();
        assert_eq!(machine.get_register("a"), 0); // Not modified
        assert_eq!(machine.get_register("b"), 42);
    }

    /// Exercise 5.1: Iterative factorial machine specification
    #[test]
    fn test_exercise_5_1() {
        // Test the iterative factorial machine
        let test_cases = vec![(0, 1), (1, 1), (5, 120), (6, 720), (7, 5040)];

        for (n, expected) in test_cases {
            let mut machine = make_factorial_iterative_machine();
            machine.set_register("n", n);
            machine.run();
            assert_eq!(
                machine.get_register("product"),
                expected,
                "factorial({}) should be {}",
                n,
                expected
            );
        }
    }

    /// Exercise 5.5: Hand-simulate factorial and Fibonacci machines
    /// This test verifies stack behavior during recursive execution
    #[test]
    fn test_exercise_5_5_factorial_stack_trace() {
        let mut machine = make_factorial_recursive_machine();
        machine.set_register("n", 3);

        // Manually step through to observe stack behavior
        // Initial: n=3, continue=fact-done
        // After saves: stack=[fact-done, 3]
        // Recursive call: n=2, continue=after-fact
        // After saves: stack=[fact-done, 3, after-fact, 2]
        // And so on...

        machine.run();

        // Final state: val should be 6, stack should be empty
        assert_eq!(machine.get_register("val"), 6);
        assert_eq!(
            machine.stack_depth(),
            0,
            "Stack should be empty after completion"
        );
    }
}
