//! SICP 5.3절: 저장소 할당과 가비지 컬렉션 (Storage Allocation and Garbage Collection)
//!
//! 이 모듈은 벡터 기반 메모리를 사용해 리스트 구조를 표현하는 방법을 보여주며,
//! SICP 5.3의 개념을 구현한다
//! (This module demonstrates how list structure can be represented using
//! vector-based memory, implementing the concepts from SICP 5.3).
//!
//! # 핵심 개념 (Key Concepts)
//!
//! - **벡터로서의 메모리 (Memory as Vectors)**: `the-cars`와 `the-cdrs` 병렬 벡터로 쌍 표현
//! - **형식화된 포인터 (Typed Pointers)**: 태그로 쌍/숫자/심볼 등을 구분
//! - **가비지 컬렉션 (Garbage Collection)**: stop-and-copy와 mark-sweep 알고리즘
//! - **Rust 소유권 (Rust's Ownership)**: 컴파일 타임 GC vs 런타임 GC
//!
//! # SICP → Rust 매핑 (SICP to Rust Mapping)
//!
//! | SICP 개념 (SICP Concept) | Rust 구현 (Rust Implementation) |
//! |--------------------------|-------------------------------|
//! | `the-cars/the-cdrs` | `Vec<Value>` 병렬 벡터 |
//! | 형식화된 포인터 (Typed pointer) | 판별자를 가진 `Value` enum |
//! | `cons` | `free` 인덱스에 할당 |
//! | Stop-and-copy GC | 두 공간 복사 수집기 |
//! | Broken heart | `Value::BrokenHeart(forwarding_addr)` |
//! | 자동 GC (Automatic GC) | Rust 소유권 = 컴파일 타임 GC |

use std::fmt;

/// 메모리에서 쌍의 타입 인덱스
/// (Type index for a pair in memory)
pub type PairIndex = usize;

/// Value는 Lisp 메모리 시스템에서의 태그 포인터를 나타낸다
/// (Value represents a tagged pointer in the Lisp memory system).
///
/// 이는 데이터 타입 정보와 실제 값/인덱스를 함께 담는 SICP의
/// "typed pointers"에 해당한다
/// (This corresponds to SICP's "typed pointers" that include both
/// data type information and the actual value or index).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    /// 숫자 값 (예: n4는 숫자 4를 의미)
    /// (Numeric value (e.g., n4 represents the number 4))
    Number(i64),

    /// 심볼 (인터닝된 문자열) (Symbol (interned string))
    Symbol(String),

    /// 쌍 포인터 (예: p5는 cars/cdrs의 인덱스 5)
    /// (Pair pointer (e.g., p5 represents index 5 into cars/cdrs))
    Pair(PairIndex),

    /// 빈 리스트 (Empty list)
    Nil,

    /// 전달 주소를 가진 브로큰 하트 마커 (GC용)
    /// (Broken heart marker with forwarding address (for GC))
    BrokenHeart(usize),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Number(n) => write!(f, "숫자(n){}", n),
            Value::Symbol(s) => write!(f, "심볼(symbol) '{}", s),
            Value::Pair(idx) => write!(f, "쌍(p){}", idx),
            Value::Nil => write!(f, "빈리스트(e0)"),
            Value::BrokenHeart(addr) => write!(f, "부서진-심장 (broken-heart)->{}", addr),
        }
    }
}

impl Value {
    /// 값이 쌍 포인터인지 확인 (Check if this value is a pair pointer)
    pub fn is_pair(&self) -> bool {
        matches!(self, Value::Pair(_))
    }

    /// 값이 nil인지 확인 (Check if this value is nil)
    pub fn is_nil(&self) -> bool {
        matches!(self, Value::Nil)
    }

    /// 브로큰 하트인지 확인 (이동된 쌍)
    /// (Check if this is a broken heart (relocated pair))
    pub fn is_broken_heart(&self) -> bool {
        matches!(self, Value::BrokenHeart(_))
    }
}

/// Memory는 리스트 구조를 위한 벡터 기반 저장소를 나타낸다
/// (Memory represents the vector-based storage for list structures).
///
/// 이는 SICP 그림 5.14의 메모리 모델로, 쌍이 쌍 포인터로 인덱싱되는
/// 병렬 벡터에 저장된다
/// (This implements the memory model from SICP Figure 5.14, where
/// pairs are stored in parallel vectors indexed by pair pointers).
#[derive(Debug, Clone)]
pub struct Memory {
    /// the-cars 벡터: 쌍의 car 필드를 저장
    /// (The-cars vector: stores car fields of pairs)
    pub the_cars: Vec<Value>,

    /// the-cdrs 벡터: 쌍의 cdr 필드를 저장
    /// (The-cdrs vector: stores cdr fields of pairs)
    pub the_cdrs: Vec<Value>,

    /// free 포인터: 다음 cons 할당 인덱스
    /// (Free pointer: next available index for cons)
    pub free: usize,

    /// 메모리 용량 (Capacity of the memory)
    capacity: usize,
}

impl Memory {
    /// 지정된 용량의 새 메모리 생성
    /// (Create a new memory with the specified capacity)
    pub fn new(capacity: usize) -> Self {
        Memory {
            the_cars: vec![Value::Nil; capacity],
            the_cdrs: vec![Value::Nil; capacity],
            free: 0,
            capacity,
        }
    }

    /// 새 쌍을 할당한다 (cons 연산)
    /// (Allocate a new pair (cons operation))
    ///
    /// SICP의 cons 구현에 대응:
    /// (Corresponds to SICP's cons implementation:)
    /// ```scheme
    /// (perform (op vector-set!) (reg the-cars) (reg free) (reg car-val))
    /// (perform (op vector-set!) (reg the-cdrs) (reg free) (reg cdr-val))
    /// (assign result (reg free))
    /// (assign free (op +) (reg free) (const 1))
    /// ```
    pub fn cons(&mut self, car: Value, cdr: Value) -> Result<Value, &'static str> {
        if self.free >= self.capacity {
            return Err("메모리 부족 (Out of memory)");
        }

        let index = self.free;
        self.the_cars[index] = car;
        self.the_cdrs[index] = cdr;
        self.free += 1;

        Ok(Value::Pair(index))
    }

    /// 쌍의 car를 가져온다
    /// (Get the car of a pair)
    ///
    /// 다음에 대응:
    /// (Corresponds to:)
    /// ```scheme
    /// (assign result (op vector-ref) (reg the-cars) (reg pair-index))
    /// ```
    pub fn car(&self, pair: &Value) -> Result<&Value, &'static str> {
        match pair {
            Value::Pair(index) => {
                if *index < self.the_cars.len() {
                    Ok(&self.the_cars[*index])
                } else {
                    Err("잘못된 쌍 인덱스 (Invalid pair index)")
                }
            }
            _ => Err("쌍이 아님 (Not a pair)"),
        }
    }

    /// 쌍의 cdr를 가져온다 (Get the cdr of a pair)
    pub fn cdr(&self, pair: &Value) -> Result<&Value, &'static str> {
        match pair {
            Value::Pair(index) => {
                if *index < self.the_cdrs.len() {
                    Ok(&self.the_cdrs[*index])
                } else {
                    Err("잘못된 쌍 인덱스 (Invalid pair index)")
                }
            }
            _ => Err("쌍이 아님 (Not a pair)"),
        }
    }

    /// 쌍의 car를 설정 (set-car!)
    /// (Set the car of a pair (set-car!))
    pub fn set_car(&mut self, pair: &Value, value: Value) -> Result<(), &'static str> {
        match pair {
            Value::Pair(index) => {
                if *index < self.the_cars.len() {
                    self.the_cars[*index] = value;
                    Ok(())
                } else {
                    Err("잘못된 쌍 인덱스 (Invalid pair index)")
                }
            }
            _ => Err("쌍이 아님 (Not a pair)"),
        }
    }

    /// 쌍의 cdr를 설정 (set-cdr!)
    /// (Set the cdr of a pair (set-cdr!))
    pub fn set_cdr(&mut self, pair: &Value, value: Value) -> Result<(), &'static str> {
        match pair {
            Value::Pair(index) => {
                if *index < self.the_cdrs.len() {
                    self.the_cdrs[*index] = value;
                    Ok(())
                } else {
                    Err("잘못된 쌍 인덱스 (Invalid pair index)")
                }
            }
            _ => Err("쌍이 아님 (Not a pair)"),
        }
    }

    /// 값들로 리스트를 구성하는 헬퍼
    /// (Build a list from values (helper function))
    pub fn list(&mut self, values: Vec<Value>) -> Result<Value, &'static str> {
        let mut result = Value::Nil;
        for value in values.into_iter().rev() {
            result = self.cons(value, result)?;
        }
        Ok(result)
    }
}

/// stop-and-copy 가비지 컬렉터
/// (Stop-and-copy garbage collector)
///
/// SICP 5.3.2의 알고리즘을 구현한다. 메모리를 두 부분으로 나누며,
/// 작업 메모리와 여유 메모리를 사용한다. GC 동안 도달 가능한 모든 쌍을
/// 작업 메모리에서 여유 메모리로 복사한 뒤 역할을 교체한다
/// (Implements the algorithm from SICP 5.3.2, which divides memory into
/// two halves: working memory and free memory. During GC, all reachable
/// pairs are copied from working memory to free memory, then the roles
/// are swapped).
pub struct StopAndCopyGC {
    /// 작업 메모리 (현재 할당 공간)
    /// (Working memory (current allocation space))
    working: Memory,

    /// 여유 메모리 (GC 대상)
    /// (Free memory (target for GC))
    free_space: Memory,

    /// 루트 집합 (라이브 포인터를 담는 레지스터)
    /// (Root set (registers containing live pointers))
    roots: Vec<Value>,
}

impl StopAndCopyGC {
    /// 공간별 메모리 크기를 지정해 새 GC 생성
    /// (Create a new GC with specified memory size per space)
    pub fn new(size: usize) -> Self {
        StopAndCopyGC {
            working: Memory::new(size),
            free_space: Memory::new(size),
            roots: Vec::new(),
        }
    }

    /// 루트 추가 (수집되지 않아야 하는 라이브 포인터)
    /// (Add a root (live pointer that should not be collected))
    pub fn add_root(&mut self, value: Value) {
        self.roots.push(value);
    }

    /// 작업 메모리에 쌍 할당 (Allocate a pair in working memory)
    pub fn cons(&mut self, car: Value, cdr: Value) -> Result<Value, &'static str> {
        self.working.cons(car, cdr)
    }

    /// 가비지 컬렉션 수행 (Perform garbage collection)
    ///
    /// SICP의 stop-and-copy 알고리즘을 구현한다:
    /// (This implements the stop-and-copy algorithm from SICP:)
    /// 1. scan과 free 포인터를 0으로 초기화
    ///    (Initialize scan and free pointers to 0)
    /// 2. 모든 루트를 재배치
    ///    (Relocate all roots)
    /// 3. 복사된 쌍을 스캔하고 car/cdr 재배치
    ///    (Scan copied pairs and relocate their car/cdr)
    /// 4. 작업/여유 메모리 역할 교체
    ///    (Flip working and free memory)
    pub fn collect(&mut self) -> Result<(), &'static str> {
        // 새 메모리 초기화 (Initialize new memory)
        self.free_space = Memory::new(self.working.capacity);
        let mut scan = 0;

        // 모든 루트 재배치 (빌림 충돌을 피하려고 먼저 수집)
        // (Relocate all roots (collect first to avoid borrow conflict))
        let roots_to_relocate: Vec<Value> = self.roots.drain(..).collect();
        for root in roots_to_relocate {
            let relocated = self.relocate(root)?;
            self.roots.push(relocated);
        }

        // 도달 가능한 쌍을 스캔하고 재배치
        // (Scan and relocate reachable pairs)
        while scan < self.free_space.free {
            // scan 위치의 쌍에서 car/cdr 가져오기
            // (Get car and cdr of pair at scan position)
            let car = self.free_space.the_cars[scan].clone();
            let cdr = self.free_space.the_cdrs[scan].clone();

            // car/cdr 재배치 (Relocate car and cdr)
            let new_car = self.relocate(car)?;
            let new_cdr = self.relocate(cdr)?;

            // 새 메모리의 쌍 갱신 (Update the pair in new memory)
            self.free_space.the_cars[scan] = new_car;
            self.free_space.the_cdrs[scan] = new_cdr;

            scan += 1;
        }

        // 뒤집기: 작업/여유 메모리 교체
        // (Flip: swap working and free memory)
        std::mem::swap(&mut self.working, &mut self.free_space);

        Ok(())
    }

    /// 값을 새 메모리로 재배치
    /// (Relocate a value to new memory)
    ///
    /// SICP의 relocate-old-result-in-new 서브루틴을 구현한다:
    /// (Implements the relocate-old-result-in-new subroutine from SICP:)
    /// - 쌍이 아닌 값은 그대로 반환
    ///   (Non-pairs are returned unchanged)
    /// - 이미 이동된 쌍은 전달 주소 반환
    ///   (Already-moved pairs return their forwarding address)
    /// - 새 쌍은 복사 후 브로큰 하트 표시
    ///   (Fresh pairs are copied and marked with broken heart)
    fn relocate(&mut self, value: Value) -> Result<Value, &'static str> {
        match value {
            Value::Pair(old_index) => {
                // 이미 이동되었는지 확인 (브로큰 하트)
                // (Check if already moved (broken heart))
                if let Value::BrokenHeart(new_index) = self.working.the_cars[old_index] {
                    return Ok(Value::Pair(new_index));
                }

                // 새 메모리로 복사 (Copy to new memory)
                let car = self.working.the_cars[old_index].clone();
                let cdr = self.working.the_cdrs[old_index].clone();

                let new_index = self.free_space.free;
                if new_index >= self.free_space.capacity {
                    return Err("GC 중 메모리 부족 (Out of memory during GC)");
                }

                self.free_space.the_cars[new_index] = car;
                self.free_space.the_cdrs[new_index] = cdr;
                self.free_space.free += 1;

                // 기존 위치를 브로큰 하트로 표시
                // (Mark old location with broken heart)
                self.working.the_cars[old_index] = Value::BrokenHeart(new_index);

                Ok(Value::Pair(new_index))
            }
            // 쌍이 아닌 값은 재배치하지 않음
            // (Non-pairs are not relocated)
            _ => Ok(value),
        }
    }

    /// 쌍 접근용 메모리 참조 반환
    /// (Get memory reference for accessing pairs)
    pub fn memory(&self) -> &Memory {
        &self.working
    }
}

/// 마크-스윕 가비지 컬렉터 (stop-and-copy 대안)
/// (Mark-sweep garbage collector (alternative to stop-and-copy))
///
/// SICP 각주에 언급된 mark-sweep 알고리즘을 구현한다:
/// (Implements the mark-sweep algorithm mentioned in SICP footnote:)
/// 1. 마크 단계: 루트에서 추적해 도달 가능한 쌍을 표시
///    (Mark phase: trace from roots and mark all reachable pairs)
/// 2. 스윕 단계: 모든 메모리를 스캔해 표시되지 않은 쌍을 회수
///    (Sweep phase: scan all memory and reclaim unmarked pairs)
pub struct MarkSweepGC {
    /// 메모리 저장소 (Memory storage)
    memory: Memory,

    /// 마크 비트 (도달 가능하면 true)
    /// (Mark bits (true if reachable))
    marked: Vec<bool>,

    /// 루트 집합 (Root set)
    roots: Vec<Value>,

    /// 프리 리스트 (사용 가능한 셀 인덱스)
    /// (Free list (indices of available cells))
    free_list: Vec<usize>,
}

impl MarkSweepGC {
    /// 새 mark-sweep 컬렉터 생성
    /// (Create a new mark-sweep collector)
    pub fn new(size: usize) -> Self {
        MarkSweepGC {
            memory: Memory::new(size),
            marked: vec![false; size],
            roots: Vec::new(),
            free_list: (0..size).collect(),
        }
    }

    /// 루트 추가 (Add a root)
    pub fn add_root(&mut self, value: Value) {
        self.roots.push(value);
    }

    /// 프리 리스트로 쌍 할당
    /// (Allocate a pair using free list)
    pub fn cons(&mut self, car: Value, cdr: Value) -> Result<Value, &'static str> {
        if let Some(index) = self.free_list.pop() {
            self.memory.the_cars[index] = car;
            self.memory.the_cdrs[index] = cdr;
            Ok(Value::Pair(index))
        } else {
            Err("메모리 부족 (Out of memory)")
        }
    }

    /// mark-sweep 가비지 컬렉션 수행
    /// (Perform mark-sweep garbage collection)
    pub fn collect(&mut self) {
        // 모든 마크 제거 (Clear all marks)
        self.marked.fill(false);

        // 마크 단계: 모든 도달 가능한 쌍 표시 (빌림 충돌을 피하려고 루트 복제)
        // (Mark phase: mark all reachable pairs (clone roots to avoid borrow conflict))
        let roots_to_mark = self.roots.clone();
        for root in &roots_to_mark {
            self.mark(root);
        }

        // 스윕 단계: 표시되지 않은 쌍 회수
        // (Sweep phase: reclaim unmarked pairs)
        self.free_list.clear();
        for i in 0..self.memory.capacity {
            if !self.marked[i] {
                self.free_list.push(i);
            }
        }
    }

    /// 값과 거기서 도달 가능한 모든 쌍을 마크
    /// (Mark a value and all pairs reachable from it)
    fn mark(&mut self, value: &Value) {
        if let Value::Pair(index) = value
            && *index < self.marked.len()
            && !self.marked[*index]
        {
            self.marked[*index] = true;

            // car/cdr를 재귀적으로 마크
            // (Recursively mark car and cdr)
            let car = self.memory.the_cars[*index].clone();
            let cdr = self.memory.the_cdrs[*index].clone();

            self.mark(&car);
            self.mark(&cdr);
        }
    }

    /// 메모리 참조 반환 (Get memory reference)
    pub fn memory(&self) -> &Memory {
        &self.memory
    }
}

/// 연습문제 5.20: (define x (cons 1 2)) (define y (list x x))의 리스트 구조 구성
/// (Exercise 5.20: Build the list structure for (define x (cons 1 2)) (define y (list x x)))
///
/// SICP 그림 5.14의 메모리-벡터 표현을 보여준다
/// (This demonstrates the memory-vector representation from SICP Figure 5.14).
///
/// free = 1에서 시작:
/// (Starting with free = 1:)
/// - x = (cons 1 2) → p1 생성, cars[1]=n1, cdrs[1]=n2, free=2
/// - y = (list x x) = (cons x (cons x nil))
///   - 내부 (cons x nil) → p2 생성, cars[2]=p1, cdrs[2]=e0, free=3
///   - 외부 (cons x p2) → p3 생성, cars[3]=p1, cdrs[3]=p2, free=4
/// - 최종: x=p1, y=p3, free=4
pub fn exercise_5_20() -> Result<(Memory, Value, Value), &'static str> {
    let mut mem = Memory::new(10);

    // 인덱스 0을 건너뜀 (p1부터 시작)
    // (Skip index 0 (start at p1))
    mem.free = 1;

    // x = (cons 1 2)
    let x = mem.cons(Value::Number(1), Value::Number(2))?;

    // y = (list x x) = (cons x (cons x nil))
    let inner = mem.cons(x.clone(), Value::Nil)?;
    let y = mem.cons(x.clone(), inner)?;

    Ok((mem, x, y))
}

// ============================================================================
// RUST 소유권: 컴파일 타임 가비지 컬렉션 (Compile-Time Garbage Collection)
// ============================================================================

// Rust의 소유권 시스템은 본질적으로 컴파일 타임 가비지 컬렉션이다
// (Rust's ownership system is essentially compile-time garbage collection).
//
// SICP는 런타임 GC(stop-and-copy, mark-sweep)를 보여주지만,
// Rust는 런타임 오버헤드 없이 메모리 안전성을 달성한다:
// (While SICP demonstrates runtime GC (stop-and-copy, mark-sweep), Rust
// achieves memory safety without runtime overhead through:)
//
// 1. **소유권 (Ownership)**: 각 값은 정확히 하나의 소유자
// 2. **대여 (Borrowing)**: 참조는 수명 규칙을 따라야 함
// 3. **Drop**: RAII로 소유자가 스코프를 벗어나면 정리
//
// 이 절은 Rust의 접근이 대부분의 경우 런타임 GC를 없애는 방식을 보여준다
// (This section demonstrates how Rust's approach eliminates the need
// for runtime garbage collection in most cases).

/// 전통적인 연결 리스트 (수동 메모리 관리, 개념적)
/// (Traditional linked list with manual memory management (conceptual))
///
/// GC가 있는 언어에서는 임시 구조 생성이 쓰레기를 만든다:
/// (In languages with GC, creating temporary structures creates garbage:)
/// ```scheme
/// (accumulate + 0 (filter odd? (enumerate-interval 0 n)))
/// ```
///
/// 열거 리스트와 필터된 리스트 모두 사용 후 쓰레기가 된다
/// (Both the enumeration list and filtered list become garbage after use).
pub mod ownership_examples {
    #[allow(unused_imports)]
    use super::*;

    /// Rust의 Box<T>는 소유권 기반 메모리 관리를 보여준다
    /// (Rust's Box<T> demonstrates ownership-based memory management)
    ///
    /// GC 불필요 - Box가 스코프를 벗어나면 메모리 해제
    /// (No GC needed - memory freed when Box goes out of scope)
    #[derive(Debug)]
    pub enum List<T> {
        Cons(T, Box<List<T>>),
        Nil,
    }

    impl<T> List<T> {
        pub fn cons(value: T, rest: List<T>) -> Self {
            List::Cons(value, Box::new(rest))
        }

        pub fn nil() -> Self {
            List::Nil
        }
    }

    /// 이터레이터 기반 접근 - 중간 결과에 대한 할당 없음
    /// (Iterator-based approach - zero allocation for intermediate results)
    ///
    /// 임시 리스트를 생성하는 SICP의 filter/enumerate와 대비
    /// (Contrast with SICP's filter/enumerate which creates temporary lists)
    pub fn sum_odd_numbers(n: i32) -> i32 {
        (0..=n)
            .filter(|x| x % 2 != 0) // 할당 없음 - 지연 반복 (No allocation - lazy iteration)
            .sum() // 쓰레기 생성 없음 (No garbage created)
    }

    /// Rust의 자동 정리를 시연 (Drop 트레이트)
    /// (Demonstrates Rust's automatic cleanup (Drop trait))
    pub struct Resource {
        id: usize,
    }

    impl Drop for Resource {
        fn drop(&mut self) {
            // 자동 정리 - GC 불필요
            // (Cleanup happens automatically - no GC needed)
            println!("리소스 {} 정리됨 (Resource {} cleaned up)", self.id, self.id);
        }
    }

    /// GC와 소유권 모델 비교
    /// (Compare GC vs ownership models)
    pub fn demonstrate_ownership() {
        // 스코프 기반 수명 - 결정적 정리
        // (Scope-based lifetime - deterministic cleanup)
        {
            let _r1 = Resource { id: 1 };
            let _r2 = Resource { id: 2 };
            // r1과 r2가 여기서 drop됨 (결정적, GC 일시정지 없음)
            // (r1 and r2 dropped here (deterministic, no GC pause))
        }

        // 소유권 기반 리스트 (List with ownership)
        let list = List::cons(1, List::cons(2, List::cons(3, List::nil())));
        // list가 스코프를 벗어나면 전체 리스트가 해제됨
        // (Entire list freed when 'list' goes out of scope)
        drop(list); // 명시적 drop (대개 자동) (Explicit drop (usually automatic))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_cons_car_cdr() {
        let mut mem = Memory::new(10);

        // cons 테스트 (Test cons)
        let pair = mem.cons(Value::Number(1), Value::Number(2)).unwrap();
        assert!(pair.is_pair());

        // car/cdr 테스트 (Test car and cdr)
        assert_eq!(mem.car(&pair).unwrap(), &Value::Number(1));
        assert_eq!(mem.cdr(&pair).unwrap(), &Value::Number(2));
    }

    #[test]
    fn test_list_construction() {
        let mut mem = Memory::new(10);

        // 그림 5.14의 ((1 2) 3 4) 구성
        // (Build ((1 2) 3 4) from Figure 5.14)
        let inner_pair = mem.cons(Value::Number(1), Value::Number(2)).unwrap();
        let last_pair = mem.cons(Value::Number(4), Value::Nil).unwrap();
        let cdr = mem.cons(Value::Number(3), last_pair).unwrap();
        let result = mem.cons(inner_pair, cdr).unwrap();

        // 구조 검증 (Verify structure)
        assert!(result.is_pair());
        let car = mem.car(&result).unwrap();
        assert!(car.is_pair());
    }

    #[test]
    fn test_stop_and_copy_gc() {
        let mut gc = StopAndCopyGC::new(10);

        // 쌍 생성 (Create some pairs)
        let p1 = gc.cons(Value::Number(1), Value::Number(2)).unwrap();
        let p2 = gc.cons(Value::Number(3), Value::Number(4)).unwrap();
        let p3 = gc.cons(p1.clone(), p2.clone()).unwrap();

        // p3만 루트 (p1/p2는 p3를 통해 도달 가능)
        // (Only p3 is a root (p1 and p2 are reachable through p3))
        gc.add_root(p3.clone());

        // GC 수행 (Perform GC)
        gc.collect().unwrap();

        // p3가 유효한지 확인 (재배치됨)
        // (Verify p3 still valid (relocated))
        assert!(gc.roots[0].is_pair());
    }

    #[test]
    fn test_mark_sweep_gc() {
        let mut gc = MarkSweepGC::new(10);

        // 쌍 생성 (Create some pairs)
        let p1 = gc.cons(Value::Number(1), Value::Number(2)).unwrap();
        let _p2 = gc.cons(Value::Number(3), Value::Number(4)).unwrap(); // 쓰레기 (Garbage)

        // p1만 루트 (Only p1 is root)
        gc.add_root(p1);

        // GC 전: 2셀 사용, 8셀 여유
        // (Before GC: 2 cells used, 8 free)
        assert_eq!(gc.free_list.len(), 8);

        // GC 후: 1셀 생존, 9셀 여유 (p2 수집됨)
        // (After GC: 1 cell live, 9 free (p2 collected))
        gc.collect();
        assert_eq!(gc.free_list.len(), 9);
    }

    #[test]
    fn test_exercise_5_20() {
        let (mem, x, y) = exercise_5_20().unwrap();

        // x는 p1이어야 함 (x should be p1)
        assert_eq!(x, Value::Pair(1));

        // y는 p3이어야 함 (y should be p3)
        assert_eq!(y, Value::Pair(3));

        // free는 4여야 함 (free should be 4)
        assert_eq!(mem.free, 4);

        // 메모리 내용 검증 (Verify memory contents)
        // cars[1] = n1, cdrs[1] = n2
        assert_eq!(mem.the_cars[1], Value::Number(1));
        assert_eq!(mem.the_cdrs[1], Value::Number(2));

        // cars[2] = p1, cdrs[2] = e0
        assert_eq!(mem.the_cars[2], Value::Pair(1));
        assert_eq!(mem.the_cdrs[2], Value::Nil);

        // cars[3] = p1, cdrs[3] = p2
        assert_eq!(mem.the_cars[3], Value::Pair(1));
        assert_eq!(mem.the_cdrs[3], Value::Pair(2));
    }

    #[test]
    fn test_broken_heart() {
        let mut gc = StopAndCopyGC::new(10);

        let p1 = gc.cons(Value::Number(42), Value::Nil).unwrap();
        gc.add_root(p1);

        // GC 전: 브로큰 하트 없음 (Before GC: no broken hearts)
        assert!(!gc.working.the_cars[0].is_broken_heart());

        // GC 후: 기존 위치가 브로큰 하트
        // (After GC: old location has broken heart)
        gc.collect().unwrap();
        assert!(gc.free_space.the_cars[0].is_broken_heart());
    }

    #[test]
    fn test_ownership_list() {
        use ownership_examples::List;

        // 리스트 (1 2 3) 구성 (Build list (1 2 3))
        let list = List::cons(1, List::cons(2, List::cons(3, List::nil())));

        // 리스트가 스코프를 벗어나면 전체 체인이 해제됨
        // (When list goes out of scope, entire chain is freed)
        // GC 불필요 - 소유권이 처리
        // (No GC needed - ownership handles it)
        drop(list);
    }

    #[test]
    fn test_ownership_sum_odd() {
        use ownership_examples::sum_odd_numbers;

        // 쓰레기 생성 없음 - 순수 반복
        // (No garbage created - pure iteration)
        let result = sum_odd_numbers(10);
        assert_eq!(result, 1 + 3 + 5 + 7 + 9);
    }
}
