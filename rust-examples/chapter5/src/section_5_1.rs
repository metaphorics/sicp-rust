//! 5.1절: 레지스터 기계 설계 (Designing Register Machines)
//!
//! 이 모듈은 SICP 5.1장에서 설명한 레지스터 기계를 구현한다
//! (This module implements register machines as described in SICP Chapter 5.1).
//! 레지스터 기계는 다음으로 구성된다:
//! (A register machine consists of:)
//! - 레지스터 (Registers): 정수 값을 담는 이름 있는 저장 위치
//! - 스택 (Stack): save/restore를 위한 LIFO 자료구조
//! - 명령 (Instructions): 연산 시퀀스 (assign, test, branch, goto, save, restore, perform)
//! - 연산 (Operations): 산술/비교 같은 기본 연산
//!
//! ## 러스트 매핑 (Rust Mapping)
//!
//! | Scheme 개념 (Scheme Concept) | Rust 구현 (Rust Implementation) |
//! |------------------------------|-------------------------------|
//! | Register machine language | `Instruction` enum with variants |
//! | Registers | `HashMap<String, i64>` |
//! | Stack | `Vec<i64>` |
//! | Controller | `Vec<Instruction>` with labels |
//! | Operations | Function pointers `fn(&[i64]) -> i64` |
//!
//! ## 메모리 레이아웃 (Memory Layout)
//!
//! ```text
//! Machine {
//!     registers: HashMap<String, i64>  // 소유된 값 (Owned values)
//!     stack: Vec<i64>                   // 소유된 스택 (Owned stack)
//!     instructions: Vec<Instruction>    // 소유된 명령 시퀀스 (Owned instruction sequence)
//!     operations: HashMap<String, fn(&[i64]) -> i64>  // 함수 포인터 (Function pointers)
//!     labels: HashMap<String, usize>    // 라벨 -> PC 매핑 (Label -> PC mapping)
//!     pc: usize                         // 프로그램 카운터 (Program counter)
//!     test_flag: bool                   // 마지막 test 결과 (Result of last test)
//! }
//! ```

use std::collections::HashMap;

/// 레지스터 값은 64비트 부호 정수 (Register values are 64-bit signed integers)
pub type Value = i64;

/// 연산 함수 타입: 값 슬라이스를 받아 단일 값을 반환
/// (Operation function type: takes slice of values, returns single value)
/// 같은 HashMap에 다른 클로저 타입을 넣기 위해 박스 처리
/// (Boxed to allow different closure types in the same HashMap)
pub type Operation = Box<dyn Fn(&[Value]) -> Value>;

/// 레지스터 기계에서 값의 출처 (Source of a value in the register machine)
#[derive(Debug, Clone, PartialEq)]
pub enum Source {
    /// 레지스터 값: (reg a) (Value from a register: (reg a))
    Reg(String),
    /// 상수 값: (const 0) (Constant value: (const 0))
    Const(Value),
    /// 연산 결과: (op rem) + 입력 (Result of operation: (op rem) with inputs)
    Op(String, Vec<Source>),
    /// 라벨 참조 (assign continue용): (label after-gcd)
    /// (Label reference (for assign continue): (label after-gcd))
    Label(String),
}

/// goto 명령의 대상 (Target for goto instruction)
#[derive(Debug, Clone, PartialEq)]
pub enum Target {
    /// 라벨로 점프: (goto (label test-b)) (Jump to label: (goto (label test-b)))
    Label(String),
    /// 레지스터 주소로 점프: (goto (reg continue))
    /// (Jump to address in register: (goto (reg continue)))
    Reg(String),
}

/// 레지스터 기계 명령 (Register machine instructions)
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    /// 레지스터에 값 할당: (assign a (reg b))
    /// (Assign value to register: (assign a (reg b)))
    Assign(String, Source),
    /// 조건 테스트: (test (op =) (reg b) (const 0))
    /// (Test condition: (test (op =) (reg b) (const 0)))
    Test(Source),
    /// 테스트 성공 시 분기: (branch (label gcd-done))
    /// (Branch if test succeeded: (branch (label gcd-done)))
    Branch(String),
    /// 무조건 점프: (goto (label test-b))
    /// (Unconditional jump: (goto (label test-b)))
    Goto(Target),
    /// 레지스터를 스택에 저장: (save n)
    /// (Save register to stack: (save n))
    Save(String),
    /// 스택에서 레지스터 복원: (restore n)
    /// (Restore register from stack: (restore n))
    Restore(String),
    /// 동작 수행(부수 효과): (perform (op print) (reg a))
    /// (Perform action (side effect): (perform (op print) (reg a)))
    Perform(Source),
}

/// 상태와 컨트롤러를 가진 레지스터 기계
/// (A register machine with state and controller)
pub struct Machine {
    /// 레지스터 뱅크: 레지스터 이름 -> 값
    /// (Register bank: register name -> value)
    registers: HashMap<String, Value>,
    /// save/restore를 위한 스택 (Stack for save/restore operations)
    stack: Vec<Value>,
    /// 명령 시퀀스 (선택적 라벨 포함)
    /// (Instruction sequence (with optional labels))
    instructions: Vec<Instruction>,
    /// 사용 가능한 연산: 이름 -> 함수
    /// (Available operations: name -> function)
    operations: HashMap<String, Operation>,
    /// 라벨 맵: 라벨 이름 -> 명령 인덱스
    /// (Label map: label name -> instruction index)
    labels: HashMap<String, usize>,
    /// 프로그램 카운터 (Program counter)
    pc: usize,
    /// 마지막 test 명령 결과 (Result of last test instruction)
    test_flag: bool,
    /// 정지 플래그 (Halted flag)
    halted: bool,
}

impl Machine {
    /// 새 레지스터 기계를 생성한다
    /// (Create a new register machine)
    ///
    /// # 인자 (Arguments)
    /// * `register_names` - 초기화할 레지스터 이름 (모두 0에서 시작)
    /// * `operations` - 연산 이름 -> 함수 맵
    /// * `instructions` - 라벨이 포함된 명령 시퀀스
    ///
    /// # 예시 (Example)
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

        // 명령을 스캔해 라벨 맵 생성 (Build label map by scanning instructions)
        let labels = HashMap::new(); // 라벨은 명령 흐름에 내장됨 (Labels are embedded in instruction flow)

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

    /// 명시적 라벨 위치로 기계를 생성
    /// (Create machine with explicit label positions)
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

    /// 레지스터 값을 설정 (Set a register's value)
    pub fn set_register(&mut self, name: &str, value: Value) {
        if let Some(reg) = self.registers.get_mut(name) {
            *reg = value;
        } else {
            panic!("알 수 없는 레지스터 (Unknown register): {}", name);
        }
    }

    /// 레지스터 값을 가져오기 (Get a register's value)
    pub fn get_register(&self, name: &str) -> Value {
        *self
            .registers
            .get(name)
            .unwrap_or_else(|| panic!("알 수 없는 레지스터 (Unknown register): {}", name))
    }

    /// 정지할 때까지 기계 실행 (Execute the machine until halt)
    pub fn run(&mut self) {
        self.pc = 0;
        self.halted = false;

        while !self.halted && self.pc < self.instructions.len() {
            let instruction = self.instructions[self.pc].clone();
            self.execute(&instruction);
        }
    }

    /// 단일 명령 실행 (Execute a single instruction)
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
                        .unwrap_or_else(|| panic!("알 수 없는 라벨 (Unknown label): {}", label));
                } else {
                    self.pc += 1;
                }
            }
            Instruction::Goto(target) => match target {
                Target::Label(label) => {
                    self.pc = *self
                        .labels
                        .get(label)
                        .unwrap_or_else(|| panic!("알 수 없는 라벨 (Unknown label): {}", label));
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
                let value = self.stack.pop().expect("스택 언더플로 (Stack underflow)");
                self.set_register(reg, value);
                self.pc += 1;
            }
            Instruction::Perform(source) => {
                self.eval_source(source);
                self.pc += 1;
            }
        }

        // 정지 조건 확인 (pc가 명령 길이를 넘음)
        // (Check for halt condition (pc beyond instructions))
        if self.pc >= self.instructions.len() {
            self.halted = true;
        }
    }

    /// 출처(Source)를 평가해 값을 얻는다
    /// (Evaluate a source to get its value)
    fn eval_source(&self, source: &Source) -> Value {
        match source {
            Source::Reg(name) => self.get_register(name),
            Source::Const(val) => *val,
            Source::Op(op_name, inputs) => {
                let op = self
                    .operations
                    .get(op_name)
                    .unwrap_or_else(|| panic!("알 수 없는 연산 (Unknown operation): {}", op_name));
                let args: Vec<Value> = inputs.iter().map(|src| self.eval_source(src)).collect();
                op(&args)
            }
            Source::Label(label) => {
                // 라벨 주소를 값으로 반환 (assign continue용)
                // (Return label address as value (for assign continue))
                *self
                    .labels
                    .get(label)
                    .unwrap_or_else(|| panic!("알 수 없는 라벨 (Unknown label): {}", label)) as Value
            }
        }
    }

    /// 현재 스택 깊이 반환 (디버깅/테스트용)
    /// (Get current stack depth (for debugging/testing))
    pub fn stack_depth(&self) -> usize {
        self.stack.len()
    }

    /// 기계가 정지했는지 확인 (Check if machine has halted)
    pub fn is_halted(&self) -> bool {
        self.halted
    }
}

/// 그림 5.4의 최대공약수(GCD) 기계를 구성한다
/// (Build the GCD machine from Figure 5.4)
///
/// 레지스터 a, b의 최대공약수를 계산한다
/// (Computes GCD of values in registers a and b).
/// 결과는 레지스터 a에 남는다
/// (Result is left in register a).
///
/// 컨트롤러 (Controller):
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
        // test-b (인덱스 0의 라벨) (label at index 0)
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
        // gcd-done (인덱스 6의 라벨) (label at index 6)
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

/// 반복 팩토리얼 기계를 구성한다 (연습문제 5.1)
/// (Build iterative factorial machine (Exercise 5.1))
///
/// 반복 알고리즘으로 n!을 계산한다:
/// (Computes factorial of n using iterative algorithm:)
/// ```scheme
/// (define (factorial n)
///   (define (iter product counter)
///     (if (> counter n)
///         product
///         (iter (* counter product) (+ counter 1))))
///   (iter 1 1))
/// ```
///
/// 레지스터: n, product, counter
/// (Registers: n, product, counter)
/// 결과는 product 레지스터에 저장된다
/// (Result in product register).
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
        // product = 1, counter = 1 초기화 (Initialize product = 1, counter = 1)
        Instruction::Assign("product".to_string(), Source::Const(1)),
        Instruction::Assign("counter".to_string(), Source::Const(1)),
        // fact-loop (인덱스 2의 라벨) (label at index 2)
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
        // fact-done (인덱스 7의 라벨) (label at index 7)
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

/// 재귀 팩토리얼 기계를 구성한다 (그림 5.11)
/// (Build recursive factorial machine (Figure 5.11))
///
/// 스택을 사용해 재귀적으로 팩토리얼을 계산한다
/// (Computes factorial recursively using stack).
/// ```scheme
/// (define (factorial n)
///   (if (= n 1)
///       1
///       (* (factorial (- n 1)) n)))
/// ```
///
/// 레지스터: n, val, continue
/// (Registers: n, val, continue)
/// n과 continue를 저장하기 위해 스택을 사용한다
/// (Uses stack to save n and continue).
/// 결과는 val 레지스터에 저장된다
/// (Result in val register).
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
        // continue를 fact-done으로 초기화 (Initialize continue to fact-done)
        Instruction::Assign(
            "continue".to_string(),
            Source::Label("fact-done".to_string()),
        ),
        // fact-loop (인덱스 1의 라벨) (label at index 1)
        Instruction::Test(Source::Op(
            "=".to_string(),
            vec![Source::Reg("n".to_string()), Source::Const(1)],
        )),
        Instruction::Branch("base-case".to_string()),
        // 재귀 경우: continue와 n 저장
        // (Recursive case: save continue and n)
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
        // after-fact (인덱스 8의 라벨) (label at index 8)
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
        // base-case (인덱스 12의 라벨) (label at index 12)
        Instruction::Assign("val".to_string(), Source::Const(1)),
        Instruction::Goto(Target::Reg("continue".to_string())),
        // fact-done (인덱스 14의 라벨) (label at index 14)
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
        assert_eq!(machine.stack_depth(), 0); // 완료 후 스택은 비어야 함 (Stack should be empty after completion)
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
            Instruction::Assign("b".to_string(), Source::Const(99)), // a == 0이면 건너뜀 (Skipped if a == 0)
            // skip (인덱스 3) (index 3)
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
        assert_eq!(machine.get_register("b"), 0); // 변경되지 않음 (Not modified)
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
            Instruction::Assign("a".to_string(), Source::Const(99)), // 건너뜀 (Skipped)
            // destination (인덱스 3) (index 3)
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
        assert_eq!(machine.get_register("a"), 0); // 변경되지 않음 (Not modified)
        assert_eq!(machine.get_register("b"), 42);
    }

    /// 연습문제 5.1: 반복 팩토리얼 기계 명세
    /// (Exercise 5.1: Iterative factorial machine specification)
    #[test]
    fn test_exercise_5_1() {
        // 반복 팩토리얼 기계 테스트 (Test the iterative factorial machine)
        let test_cases = vec![(0, 1), (1, 1), (5, 120), (6, 720), (7, 5040)];

        for (n, expected) in test_cases {
            let mut machine = make_factorial_iterative_machine();
            machine.set_register("n", n);
            machine.run();
            assert_eq!(
                machine.get_register("product"),
                expected,
                "factorial({})는 {}이어야 함 (factorial({}) should be {})",
                n,
                expected,
                n,
                expected
            );
        }
    }

    /// 연습문제 5.5: 팩토리얼/피보나치 기계 수동 시뮬레이션
    /// (Exercise 5.5: Hand-simulate factorial and Fibonacci machines)
    /// 이 테스트는 재귀 실행 중 스택 동작을 검증한다
    /// (This test verifies stack behavior during recursive execution)
    #[test]
    fn test_exercise_5_5_factorial_stack_trace() {
        let mut machine = make_factorial_recursive_machine();
        machine.set_register("n", 3);

        // 스택 동작을 관찰하기 위해 수동 단계 설명
        // (Manually step through to observe stack behavior)
        // 초기: n=3, continue=fact-done (Initial: n=3, continue=fact-done)
        // 저장 후: stack=[fact-done, 3] (After saves: stack=[fact-done, 3])
        // 재귀 호출: n=2, continue=after-fact (Recursive call: n=2, continue=after-fact)
        // 저장 후: stack=[fact-done, 3, after-fact, 2]
        // 등등... (And so on...)

        machine.run();

        // 최종 상태: val=6, 스택은 비어야 함
        // (Final state: val should be 6, stack should be empty)
        assert_eq!(machine.get_register("val"), 6);
        assert_eq!(
            machine.stack_depth(),
            0,
            "완료 후 스택은 비어야 함 (Stack should be empty after completion)"
        );
    }
}
