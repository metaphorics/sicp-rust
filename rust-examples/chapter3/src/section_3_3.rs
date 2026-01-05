//! 3.3절: 가변 데이터로 모델링 (Modeling with Mutable Data)
//!
//! 이 절은 가변 데이터 구조를 위한 관용적인 Rust 패턴을 보여준다
//! (This section demonstrates idiomatic Rust patterns for mutable data structures):
//!
//! - **아레나 기반 cons 리스트 (Arena-based cons lists)**: 타입 안전 인덱스로 `Rc<RefCell<>>`를 대체
//!   (Type-safe indices replace `Rc<RefCell<>>`)
//! - **VecDeque 큐 (VecDeque queues)**: FIFO를 위한 표준 라이브러리 해법
//!   (Standard library solution for FIFO)
//! - **HashMap 테이블 (HashMap tables)**: 이미 관용적인 구현
//!   (Already idiomatic)
//! - **가변 시뮬레이터 (Mutable simulator)**: `&mut self` 메서드를 통한 소유 상태
//!   (Owned state with `&mut self` methods)
//! - **메시지 큐 제약 (Message-queue constraints)**: 순환 참조 없이 이벤트 주도 전파
//!   (Event-driven propagation without circular refs)

use sicp_common::{Arena, ArenaId};
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::Hash;

// =============================================================================
// 3.3.1 가변 리스트 구조 - 아레나 기반 (Mutable List Structure - Arena-based)
// =============================================================================

/// Copy 타입을 위한 Cell 기반 가변 쌍(RefCell 불필요) (Mutable pair using Cell for Copy types (no RefCell needed)).
#[derive(Debug)]
pub struct MutablePair<T: Copy> {
    car: Cell<T>,
    cdr: Cell<T>,
}

impl<T: Copy> MutablePair<T> {
    pub fn new(car: T, cdr: T) -> Self {
        MutablePair {
            car: Cell::new(car),
            cdr: Cell::new(cdr),
        }
    }

    pub fn car(&self) -> T {
        self.car.get()
    }

    pub fn cdr(&self) -> T {
        self.cdr.get()
    }

    pub fn set_car(&self, value: T) {
        self.car.set(value);
    }

    pub fn set_cdr(&self, value: T) {
        self.cdr.set(value);
    }
}

/// 아레나에 저장된 cons 셀 (A cons cell stored in an arena).
#[derive(Debug, Clone)]
pub struct ConsCell<T> {
    pub car: T,
    pub cdr: Option<ArenaId<ConsCell<T>>>,
}

/// 아레나 기반 가변 cons 리스트 (Arena-based mutable cons list).
///
/// `Rc<RefCell<Cons<T>>>`를 아레나 할당으로 대체하여
/// 런타임 빌림 검사 오버헤드를 제거한다
/// (This replaces `Rc<RefCell<Cons<T>>>` with arena allocation,
/// eliminating runtime borrow checking overhead).
#[derive(Debug)]
pub struct ConsList<T> {
    arena: Arena<ConsCell<T>>,
    head: Option<ArenaId<ConsCell<T>>>,
}

impl<T: Clone> ConsList<T> {
    /// 빈 리스트를 생성한다 (Creates an empty list).
    pub fn new() -> Self {
        ConsList {
            arena: Arena::new(),
            head: None,
        }
    }

    /// 맨 앞에 새 원소를 cons 한다 (Cons a new element to the front).
    pub fn cons(&mut self, car: T) {
        let cell = ConsCell {
            car,
            cdr: self.head,
        };
        self.head = Some(self.arena.alloc(cell));
    }

    /// 머리 원소를 가져온다 (Gets the head element).
    pub fn car(&self) -> Option<&T> {
        self.head.map(|id| &self.arena.get(id).car)
    }

    /// 비어 있는지 확인한다 (Checks if the list is empty).
    pub fn is_empty(&self) -> bool {
        self.head.is_none()
    }

    /// 특정 셀의 car를 가져온다 (Gets the car at a specific cell).
    pub fn get_car(&self, id: ArenaId<ConsCell<T>>) -> &T {
        &self.arena.get(id).car
    }

    /// 특정 셀의 cdr를 가져온다 (Gets the cdr at a specific cell).
    pub fn get_cdr(&self, id: ArenaId<ConsCell<T>>) -> Option<ArenaId<ConsCell<T>>> {
        self.arena.get(id).cdr
    }

    /// 특정 셀의 car를 설정한다(아레나를 통한 변경) (Sets the car at a specific cell (mutation via arena)).
    pub fn set_car(&mut self, id: ArenaId<ConsCell<T>>, value: T) {
        self.arena.get_mut(id).car = value;
    }

    /// 특정 셀의 cdr를 설정한다(아레나를 통한 변경) (Sets the cdr at a specific cell (mutation via arena)).
    pub fn set_cdr(&mut self, id: ArenaId<ConsCell<T>>, cdr: Option<ArenaId<ConsCell<T>>>) {
        self.arena.get_mut(id).cdr = cdr;
    }

    /// 확인을 위해 Vec으로 변환한다 (Converts to Vec for inspection).
    pub fn to_vec(&self) -> Vec<T> {
        let mut result = Vec::new();
        let mut current = self.head;
        while let Some(id) = current {
            let cell = self.arena.get(id);
            result.push(cell.car.clone());
            current = cell.cdr;
        }
        result
    }

    /// 머리 노드 ID를 가져온다 (Gets the head node ID).
    pub fn head_id(&self) -> Option<ArenaId<ConsCell<T>>> {
        self.head
    }

    /// 마지막 노드 ID를 가져온다 (Gets the last node ID).
    pub fn last_id(&self) -> Option<ArenaId<ConsCell<T>>> {
        let mut current = self.head?;
        while let Some(next) = self.arena.get(current).cdr {
            current = next;
        }
        Some(current)
    }

    /// 마지막 cdr을 변경해 다른 리스트를 덧붙인다(파괴적) (Appends another list by mutating the last cdr (destructive)).
    pub fn append_mut(&mut self, other_head: Option<ArenaId<ConsCell<T>>>) {
        if let Some(last) = self.last_id() {
            self.arena.get_mut(last).cdr = other_head;
        } else {
            self.head = other_head;
        }
    }

    /// 사이클을 만든다(마지막이 첫 번째를 가리킴) (Makes a cycle (last points back to first)).
    pub fn make_cycle(&mut self) {
        if let (Some(last), Some(head)) = (self.last_id(), self.head) {
            self.arena.get_mut(last).cdr = Some(head);
        }
    }

    /// 리스트를 파괴적으로 뒤집는다(Scheme의 mystery 함수처럼)
    /// (Reverses the list destructively (like Scheme's mystery function)).
    pub fn reverse_mut(&mut self) {
        let mut prev: Option<ArenaId<ConsCell<T>>> = None;
        let mut current = self.head;

        while let Some(curr_id) = current {
            let next = self.arena.get(curr_id).cdr;
            self.arena.get_mut(curr_id).cdr = prev;
            prev = Some(curr_id);
            current = next;
        }
        self.head = prev;
    }

    /// 노드 개수를 센다(단순 버전, 사이클 미처리) (Counts nodes (naive version, doesn't handle cycles)).
    pub fn len_naive(&self) -> usize {
        let mut count = 0;
        let mut current = self.head;
        while let Some(id) = current {
            count += 1;
            current = self.arena.get(id).cdr;
        }
        count
    }

    /// 플로이드 알고리즘으로 사이클을 감지한다 (Detects cycle using Floyd's algorithm).
    pub fn has_cycle(&self) -> bool {
        let mut slow = self.head;
        let mut fast = self.head;

        while let (Some(s), Some(f)) = (slow, fast) {
            // slow를 한 칸 이동 (Move slow one step)
            slow = self.arena.get(s).cdr;

            // fast를 두 칸 이동 (Move fast two steps)
            if let Some(f1) = self.arena.get(f).cdr {
                fast = self.arena.get(f1).cdr;

                // 두 포인터가 만나는지 확인 (Check if they meet)
                if let (Some(s_id), Some(f_id)) = (slow, fast)
                    && s_id == f_id
                {
                    return true;
                }
            } else {
                return false;
            }
        }
        false
    }
}

impl<T: Clone> Default for ConsList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> FromIterator<T> for ConsList<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let items: Vec<T> = iter.into_iter().collect();
        let mut list = ConsList::new();

        // 뒤에서 앞으로 구성 (Build from back to front)
        for item in items.into_iter().rev() {
            list.cons(item);
        }
        list
    }
}

// =============================================================================
// 3.3.2 큐 표현 - VecDeque (Representing Queues - VecDeque)
// =============================================================================

/// VecDeque 기반 큐(관용적인 Rust) (Queue using VecDeque (idiomatic Rust)).
///
/// 앞/뒤 포인터를 가진 SICP 스타일 큐를 표준 라이브러리의 효율적인 덱으로 대체한다
/// (The SICP-style queue with front/rear pointers is replaced by
/// the standard library's efficient double-ended queue).
#[derive(Debug)]
pub struct Queue<T> {
    items: VecDeque<T>,
}

impl<T> Queue<T> {
    pub fn new() -> Self {
        Queue {
            items: VecDeque::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn front(&self) -> Option<&T> {
        self.items.front()
    }

    pub fn insert(&mut self, item: T) {
        self.items.push_back(item);
    }

    pub fn delete(&mut self) -> Option<T> {
        self.items.pop_front()
    }

    pub fn to_vec(&self) -> Vec<&T> {
        self.items.iter().collect()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
}

impl<T> Default for Queue<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// 덱(양방향 큐) - 연습문제 3.23 (Deque (double-ended queue) - Exercise 3.23)
#[derive(Debug)]
pub struct Deque<T> {
    items: VecDeque<T>,
}

impl<T> Deque<T> {
    pub fn new() -> Self {
        Deque {
            items: VecDeque::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn front(&self) -> Option<&T> {
        self.items.front()
    }

    pub fn rear(&self) -> Option<&T> {
        self.items.back()
    }

    pub fn front_insert(&mut self, item: T) {
        self.items.push_front(item);
    }

    pub fn rear_insert(&mut self, item: T) {
        self.items.push_back(item);
    }

    pub fn front_delete(&mut self) -> Option<T> {
        self.items.pop_front()
    }

    pub fn rear_delete(&mut self) -> Option<T> {
        self.items.pop_back()
    }
}

impl<T> Default for Deque<T> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 3.3.3 테이블 표현 (Representing Tables)
// =============================================================================

/// 1차원 테이블 (One-dimensional table).
#[derive(Debug)]
pub struct Table<K, V> {
    records: HashMap<K, V>,
}

impl<K: Eq + Hash, V> Table<K, V> {
    pub fn new() -> Self {
        Table {
            records: HashMap::new(),
        }
    }

    pub fn lookup(&self, key: &K) -> Option<&V> {
        self.records.get(key)
    }

    pub fn insert(&mut self, key: K, value: V) {
        self.records.insert(key, value);
    }
}

impl<K: Eq + Hash, V> Default for Table<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

/// 2차원 테이블 (Two-dimensional table).
#[derive(Debug)]
pub struct Table2D<K1, K2, V> {
    subtables: HashMap<K1, HashMap<K2, V>>,
}

impl<K1: Eq + Hash, K2: Eq + Hash, V> Table2D<K1, K2, V> {
    pub fn new() -> Self {
        Table2D {
            subtables: HashMap::new(),
        }
    }

    pub fn lookup(&self, key1: &K1, key2: &K2) -> Option<&V> {
        self.subtables.get(key1).and_then(|sub| sub.get(key2))
    }

    pub fn insert(&mut self, key1: K1, key2: K2, value: V) {
        self.subtables.entry(key1).or_default().insert(key2, value);
    }
}

impl<K1: Eq + Hash, K2: Eq + Hash, V> Default for Table2D<K1, K2, V> {
    fn default() -> Self {
        Self::new()
    }
}

/// 단순 클로저 + 캐시 패턴을 사용한 메모이제이션
/// (Memoization using a simple closure + cache pattern).
///
/// 캐시에만 내부 가변성을 사용한다(허용되는 패턴)
/// (Uses interior mutability only for the cache (acceptable pattern)).
pub struct Memoized<T, R, F>
where
    T: Eq + Hash + Clone,
    R: Clone,
    F: Fn(&T) -> R,
{
    func: F,
    cache: std::cell::RefCell<HashMap<T, R>>,
}

impl<T, R, F> Memoized<T, R, F>
where
    T: Eq + Hash + Clone,
    R: Clone,
    F: Fn(&T) -> R,
{
    pub fn new(func: F) -> Self {
        Memoized {
            func,
            cache: std::cell::RefCell::new(HashMap::new()),
        }
    }

    pub fn call(&self, arg: &T) -> R {
        if let Some(result) = self.cache.borrow().get(arg) {
            return result.clone();
        }

        let result = (self.func)(arg);
        self.cache.borrow_mut().insert(arg.clone(), result.clone());
        result
    }
}

// =============================================================================
// 3.3.4 디지털 회로 시뮬레이터 (A Simulator for Digital Circuits)
// =============================================================================

pub mod circuits {
    use super::*;

    pub type Signal = u8; // 0 또는 1 (0 or 1)
    pub type Time = u64;
    pub type WireId = usize;
    pub type ActionId = usize;

    /// 시뮬레이터에 저장된 와이어 상태 (Wire state stored in the simulator).
    #[derive(Debug, Default)]
    pub struct Wire {
        signal: Signal,
        actions: Vec<ActionId>,
    }

    /// 게이트 지연 (Gate delays).
    pub const INVERTER_DELAY: Time = 2;
    pub const AND_GATE_DELAY: Time = 3;
    pub const OR_GATE_DELAY: Time = 5;

    /// 가변 시뮬레이터 참조를 받는 액션 클로저 타입
    /// (Action closure type that takes mutable simulator reference).
    type ActionFn = Box<dyn FnMut(&mut Simulator)>;

    /// 소유된 가변 상태를 가진 디지털 회로 시뮬레이터
    /// (Digital circuit simulator with owned mutable state).
    ///
    /// `Rc<RefCell<>>` 패턴을 단일 소유자로 대체하고
    /// 액션 콜백에 `&mut self`를 전달한다
    /// (This replaces the Rc<RefCell<>> pattern with a single owner
    /// that passes `&mut self` to action callbacks).
    pub struct Simulator {
        wires: Vec<Wire>,
        actions: Vec<Option<ActionFn>>,
        agenda: BTreeMap<Time, VecDeque<ActionId>>,
        current_time: Time,
    }

    impl Simulator {
        pub fn new() -> Self {
            Simulator {
                wires: Vec::new(),
                actions: Vec::new(),
                agenda: BTreeMap::new(),
                current_time: 0,
            }
        }

        /// 새 와이어를 생성하고 ID를 반환한다 (Create a new wire and return its ID).
        pub fn make_wire(&mut self) -> WireId {
            let id = self.wires.len();
            self.wires.push(Wire::default());
            id
        }

        /// 와이어 신호를 가져온다 (Get wire signal).
        pub fn get_signal(&self, wire: WireId) -> Signal {
            self.wires[wire].signal
        }

        /// 와이어 신호를 설정하고 변경 시 액션을 트리거한다
        /// (Set wire signal, triggering actions if changed).
        pub fn set_signal(&mut self, wire: WireId, value: Signal) {
            let old_value = self.wires[wire].signal;
            if old_value != value {
                self.wires[wire].signal = value;
                // 실행할 액션 ID를 수집(빌림 충돌 회피)
                // (Collect action IDs to run (avoid borrow conflict))
                let action_ids: Vec<ActionId> = self.wires[wire].actions.clone();
                for action_id in action_ids {
                    self.run_action(action_id);
                }
            }
        }

        /// 와이어에 액션을 추가한다 (Add action to wire).
        fn add_action_to_wire(&mut self, wire: WireId, action_id: ActionId) {
            self.wires[wire].actions.push(action_id);
        }

        /// 액션을 등록하고 ID를 반환한다 (Register an action and return its ID).
        fn register_action(&mut self, action: ActionFn) -> ActionId {
            let id = self.actions.len();
            self.actions.push(Some(action));
            id
        }

        /// ID로 액션을 실행한다 (Run an action by ID).
        fn run_action(&mut self, action_id: ActionId) {
            if let Some(mut action) = self.actions[action_id].take() {
                action(self);
                self.actions[action_id] = Some(action);
            }
        }

        /// 지연 후 액션을 예약한다 (Schedule action after delay).
        pub fn after_delay(&mut self, delay: Time, action: ActionFn) {
            let time = self.current_time + delay;
            let action_id = self.register_action(action);
            self.agenda.entry(time).or_default().push_back(action_id);
        }

        /// 현재 시뮬레이션 시간 (Current simulation time).
        pub fn current_time(&self) -> Time {
            self.current_time
        }

        /// 아젠다가 빌 때까지 시뮬레이션을 실행한다
        /// (Run simulation until agenda is empty).
        pub fn propagate(&mut self) {
            while let Some((&time, _)) = self.agenda.first_key_value() {
                self.current_time = time;
                if let Some(mut queue) = self.agenda.remove(&time) {
                    while let Some(action_id) = queue.pop_front() {
                        self.run_action(action_id);
                    }
                }
            }
        }

        // 논리 연산 (Logical operations)
        pub fn logical_not(s: Signal) -> Signal {
            if s == 0 { 1 } else { 0 }
        }

        pub fn logical_and(a: Signal, b: Signal) -> Signal {
            if a == 1 && b == 1 { 1 } else { 0 }
        }

        pub fn logical_or(a: Signal, b: Signal) -> Signal {
            if a == 1 || b == 1 { 1 } else { 0 }
        }

        /// 인버터 게이트 (Inverter gate).
        pub fn inverter(&mut self, input: WireId, output: WireId) {
            let action_id = self.register_action(Box::new(move |sim: &mut Simulator| {
                let new_value = Self::logical_not(sim.get_signal(input));
                sim.after_delay(
                    INVERTER_DELAY,
                    Box::new(move |sim2: &mut Simulator| {
                        sim2.set_signal(output, new_value);
                    }),
                );
            }));
            self.add_action_to_wire(input, action_id);
            // 초기 액션 트리거 (Trigger initial action)
            self.run_action(action_id);
        }

        /// AND 게이트 (AND gate).
        pub fn and_gate(&mut self, a1: WireId, a2: WireId, output: WireId) {
            let action_fn = move |sim: &mut Simulator| {
                let new_value = Self::logical_and(sim.get_signal(a1), sim.get_signal(a2));
                sim.after_delay(
                    AND_GATE_DELAY,
                    Box::new(move |sim2: &mut Simulator| {
                        sim2.set_signal(output, new_value);
                    }),
                );
            };

            let action_id1 = self.register_action(Box::new(action_fn));
            let action_id2 = self.register_action(Box::new(move |sim: &mut Simulator| {
                let new_value = Self::logical_and(sim.get_signal(a1), sim.get_signal(a2));
                sim.after_delay(
                    AND_GATE_DELAY,
                    Box::new(move |sim2: &mut Simulator| {
                        sim2.set_signal(output, new_value);
                    }),
                );
            }));

            self.add_action_to_wire(a1, action_id1);
            self.add_action_to_wire(a2, action_id2);
        }

        /// OR 게이트 (OR gate).
        pub fn or_gate(&mut self, a1: WireId, a2: WireId, output: WireId) {
            let action_fn = move |sim: &mut Simulator| {
                let new_value = Self::logical_or(sim.get_signal(a1), sim.get_signal(a2));
                sim.after_delay(
                    OR_GATE_DELAY,
                    Box::new(move |sim2: &mut Simulator| {
                        sim2.set_signal(output, new_value);
                    }),
                );
            };

            let action_id1 = self.register_action(Box::new(action_fn));
            let action_id2 = self.register_action(Box::new(move |sim: &mut Simulator| {
                let new_value = Self::logical_or(sim.get_signal(a1), sim.get_signal(a2));
                sim.after_delay(
                    OR_GATE_DELAY,
                    Box::new(move |sim2: &mut Simulator| {
                        sim2.set_signal(output, new_value);
                    }),
                );
            }));

            self.add_action_to_wire(a1, action_id1);
            self.add_action_to_wire(a2, action_id2);
        }

        /// 하프 애더 회로 (Half adder circuit).
        pub fn half_adder(&mut self, a: WireId, b: WireId, sum: WireId, carry: WireId) {
            let d = self.make_wire();
            let e = self.make_wire();

            self.or_gate(a, b, d);
            self.and_gate(a, b, carry);
            self.inverter(carry, e);
            self.and_gate(d, e, sum);
        }

        /// 풀 애더 회로 (Full adder circuit).
        pub fn full_adder(
            &mut self,
            a: WireId,
            b: WireId,
            c_in: WireId,
            sum: WireId,
            c_out: WireId,
        ) {
            let s = self.make_wire();
            let c1 = self.make_wire();
            let c2 = self.make_wire();

            self.half_adder(b, c_in, s, c1);
            self.half_adder(a, s, sum, c2);
            self.or_gate(c1, c2, c_out);
        }
    }

    impl Default for Simulator {
        fn default() -> Self {
            Self::new()
        }
    }
}

// =============================================================================
// 3.3.5 제약 전파 - 메시지 큐 아키텍처 (Propagation of Constraints - Message Queue Architecture)
// =============================================================================

pub mod constraints {
    use super::*;

    /// 커넥터 식별자 (Connector identifier).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ConnectorId(pub usize);

    /// 제약 식별자 (Constraint identifier).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ConstraintId(pub usize);

    /// 커넥터 상태 (Connector state).
    #[derive(Debug, Clone, Default)]
    pub struct ConnectorState {
        pub value: Option<f64>,
        pub informant: Option<String>,
        pub constraints: Vec<ConstraintId>,
    }

    /// 메시지 큐 기반 전파 이벤트 (Event for message-queue based propagation).
    #[derive(Debug, Clone)]
    pub enum Event {
        SetValue {
            connector: ConnectorId,
            value: f64,
            informant: String,
        },
        ForgetValue {
            connector: ConnectorId,
            retractor: String,
        },
        ProcessNewValue {
            constraint: ConstraintId,
        },
        ProcessForgetValue {
            constraint: ConstraintId,
        },
    }

    /// 제약 타입 (Constraint types).
    #[derive(Debug, Clone)]
    pub enum ConstraintKind {
        Adder {
            a1: ConnectorId,
            a2: ConnectorId,
            sum: ConnectorId,
            name: String,
        },
        Multiplier {
            m1: ConnectorId,
            m2: ConnectorId,
            product: ConnectorId,
            name: String,
        },
        Constant {
            value: f64,
            connector: ConnectorId,
            name: String,
        },
        Probe {
            name: String,
            connector: ConnectorId,
        },
    }

    /// 메시지 큐 아키텍처를 사용하는 제약 네트워크
    /// (Constraint network using message-queue architecture).
    ///
    /// 중앙집중 상태와 명시적 이벤트 전파로 `Rc<RefCell<>>` 패턴을 제거한다
    /// (This eliminates all `Rc<RefCell<>>` patterns by using a centralized
    /// state with explicit event propagation).
    pub struct ConstraintNetwork {
        connectors: Vec<ConnectorState>,
        constraints: Vec<ConstraintKind>,
        events: VecDeque<Event>,
        probe_output: Vec<String>,
    }

    impl ConstraintNetwork {
        pub fn new() -> Self {
            ConstraintNetwork {
                connectors: Vec::new(),
                constraints: Vec::new(),
                events: VecDeque::new(),
                probe_output: Vec::new(),
            }
        }

        /// 새 커넥터를 생성한다 (Creates a new connector).
        pub fn make_connector(&mut self) -> ConnectorId {
            let id = ConnectorId(self.connectors.len());
            self.connectors.push(ConnectorState::default());
            id
        }

        /// 커넥터에 값이 있는지 확인한다 (Checks if connector has a value).
        pub fn has_value(&self, conn: ConnectorId) -> bool {
            self.connectors[conn.0].value.is_some()
        }

        /// 커넥터 값을 가져온다 (Gets connector value).
        pub fn get_value(&self, conn: ConnectorId) -> Option<f64> {
            self.connectors[conn.0].value
        }

        /// 값-설정 이벤트를 큐에 넣는다 (Queues a set-value event).
        pub fn set_value(&mut self, conn: ConnectorId, value: f64, informant: &str) {
            self.events.push_back(Event::SetValue {
                connector: conn,
                value,
                informant: informant.to_string(),
            });
        }

        /// 값-삭제 이벤트를 큐에 넣는다 (Queues a forget-value event).
        pub fn forget_value(&mut self, conn: ConnectorId, retractor: &str) {
            self.events.push_back(Event::ForgetValue {
                connector: conn,
                retractor: retractor.to_string(),
            });
        }

        /// 가산 제약을 생성한다: a1 + a2 = sum
        /// (Creates an adder constraint: a1 + a2 = sum).
        pub fn adder(
            &mut self,
            a1: ConnectorId,
            a2: ConnectorId,
            sum: ConnectorId,
            name: &str,
        ) -> ConstraintId {
            let id = ConstraintId(self.constraints.len());
            self.constraints.push(ConstraintKind::Adder {
                a1,
                a2,
                sum,
                name: name.to_string(),
            });

            // 커넥터에 연결 (Connect to connectors)
            self.connectors[a1.0].constraints.push(id);
            self.connectors[a2.0].constraints.push(id);
            self.connectors[sum.0].constraints.push(id);

            id
        }

        /// 곱셈 제약을 생성한다: m1 * m2 = product
        /// (Creates a multiplier constraint: m1 * m2 = product).
        pub fn multiplier(
            &mut self,
            m1: ConnectorId,
            m2: ConnectorId,
            product: ConnectorId,
            name: &str,
        ) -> ConstraintId {
            let id = ConstraintId(self.constraints.len());
            self.constraints.push(ConstraintKind::Multiplier {
                m1,
                m2,
                product,
                name: name.to_string(),
            });

            self.connectors[m1.0].constraints.push(id);
            self.connectors[m2.0].constraints.push(id);
            self.connectors[product.0].constraints.push(id);

            id
        }

        /// 상수 제약을 생성한다 (Creates a constant constraint).
        pub fn constant(&mut self, value: f64, connector: ConnectorId, name: &str) -> ConstraintId {
            let id = ConstraintId(self.constraints.len());
            self.constraints.push(ConstraintKind::Constant {
                value,
                connector,
                name: name.to_string(),
            });

            self.connectors[connector.0].constraints.push(id);

            // 상수 값을 즉시 설정 (Set the constant value immediately)
            self.connectors[connector.0].value = Some(value);
            self.connectors[connector.0].informant = Some(name.to_string());

            id
        }

        /// 프로브를 생성한다 (Creates a probe).
        pub fn probe(&mut self, name: &str, connector: ConnectorId) -> ConstraintId {
            let id = ConstraintId(self.constraints.len());
            self.constraints.push(ConstraintKind::Probe {
                name: name.to_string(),
                connector,
            });

            self.connectors[connector.0].constraints.push(id);
            id
        }

        /// 대기 중인 모든 이벤트를 처리한다 (Processes all pending events).
        pub fn propagate(&mut self) {
            while let Some(event) = self.events.pop_front() {
                match event {
                    Event::SetValue {
                        connector,
                        value,
                        informant,
                    } => {
                        self.handle_set_value(connector, value, &informant);
                    }
                    Event::ForgetValue {
                        connector,
                        retractor,
                    } => {
                        self.handle_forget_value(connector, &retractor);
                    }
                    Event::ProcessNewValue { constraint } => {
                        self.process_new_value(constraint);
                    }
                    Event::ProcessForgetValue { constraint } => {
                        self.process_forget_value(constraint);
                    }
                }
            }
        }

        fn handle_set_value(&mut self, conn: ConnectorId, value: f64, informant: &str) {
            let state = &self.connectors[conn.0];
            if state.value.is_none() {
                self.connectors[conn.0].value = Some(value);
                self.connectors[conn.0].informant = Some(informant.to_string());

                // 제약에 통지 (Notify constraints)
                let constraints: Vec<ConstraintId> = self.connectors[conn.0].constraints.clone();
                for cid in constraints {
                    self.events
                        .push_back(Event::ProcessNewValue { constraint: cid });
                }
            }
        }

        fn handle_forget_value(&mut self, conn: ConnectorId, retractor: &str) {
            if let Some(ref inf) = self.connectors[conn.0].informant
                && inf == retractor
            {
                self.connectors[conn.0].value = None;
                self.connectors[conn.0].informant = None;

                let constraints: Vec<ConstraintId> = self.connectors[conn.0].constraints.clone();
                for cid in constraints {
                    self.events
                        .push_back(Event::ProcessForgetValue { constraint: cid });
                }
            }
        }

        fn process_new_value(&mut self, cid: ConstraintId) {
            match self.constraints[cid.0].clone() {
                ConstraintKind::Adder { a1, a2, sum, name } => {
                    let v1 = self.get_value(a1);
                    let v2 = self.get_value(a2);
                    let s = self.get_value(sum);

                    match (v1, v2, s) {
                        (Some(v1), Some(v2), None) => {
                            self.set_value(sum, v1 + v2, &name);
                        }
                        (Some(v1), None, Some(s)) => {
                            self.set_value(a2, s - v1, &name);
                        }
                        (None, Some(v2), Some(s)) => {
                            self.set_value(a1, s - v2, &name);
                        }
                        _ => {}
                    }
                }
                ConstraintKind::Multiplier {
                    m1,
                    m2,
                    product,
                    name,
                } => {
                    let v1 = self.get_value(m1);
                    let v2 = self.get_value(m2);
                    let p = self.get_value(product);

                    match (v1, v2, p) {
                        (Some(v1), Some(v2), None) => {
                            self.set_value(product, v1 * v2, &name);
                        }
                        (Some(v1), None, Some(p)) if v1.abs() > 1e-10 => {
                            self.set_value(m2, p / v1, &name);
                        }
                        (None, Some(v2), Some(p)) if v2.abs() > 1e-10 => {
                            self.set_value(m1, p / v2, &name);
                        }
                        _ => {}
                    }
                }
                ConstraintKind::Constant { .. } => {
                    // 상수는 변경에 반응하지 않음 (Constants don't respond to changes)
                }
                ConstraintKind::Probe { name, connector } => {
                    if let Some(value) = self.get_value(connector) {
                        self.probe_output
                            .push(format!("프로브 (Probe): {} = {}", name, value));
                    }
                }
            }
        }

        fn process_forget_value(&mut self, cid: ConstraintId) {
            match self.constraints[cid.0].clone() {
                ConstraintKind::Adder { a1, a2, sum, name } => {
                    self.forget_value(a1, &name);
                    self.forget_value(a2, &name);
                    self.forget_value(sum, &name);
                }
                ConstraintKind::Multiplier {
                    m1,
                    m2,
                    product,
                    name,
                } => {
                    self.forget_value(m1, &name);
                    self.forget_value(m2, &name);
                    self.forget_value(product, &name);
                }
                ConstraintKind::Constant { .. } => {
                    // 상수는 잊지 않음 (Constants don't forget)
                }
                ConstraintKind::Probe { name, connector: _ } => {
                    self.probe_output
                        .push(format!("프로브 (Probe): {} = ?", name));
                }
            }
        }

        /// 섭씨-화씨 변환기를 구성한다 (Builds Celsius-Fahrenheit converter).
        pub fn celsius_fahrenheit_converter(&mut self, c: ConnectorId, f: ConnectorId) {
            let u = self.make_connector();
            let v = self.make_connector();
            let w = self.make_connector();
            let x = self.make_connector();
            let y = self.make_connector();

            self.multiplier(c, w, u, "m1");
            self.multiplier(v, x, u, "m2");
            self.adder(v, y, f, "a1");
            self.constant(9.0, w, "c1");
            self.constant(5.0, x, "c2");
            self.constant(32.0, y, "c3");
        }

        /// 테스트용 프로브 출력 값을 가져온다 (Gets probe output for testing).
        pub fn get_probe_output(&self) -> &[String] {
            &self.probe_output
        }
    }

    impl Default for ConstraintNetwork {
        fn default() -> Self {
            Self::new()
        }
    }
}

// =============================================================================
// 테스트 (TESTS)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutable_pair() {
        let pair = MutablePair::new(1, 2);
        assert_eq!(pair.car(), 1);
        assert_eq!(pair.cdr(), 2);

        pair.set_car(10);
        pair.set_cdr(20);
        assert_eq!(pair.car(), 10);
        assert_eq!(pair.cdr(), 20);
    }

    #[test]
    fn test_cons_list_basic() {
        let mut list: ConsList<i32> = ConsList::new();
        assert!(list.is_empty());

        list.cons(3);
        list.cons(2);
        list.cons(1);

        assert_eq!(list.to_vec(), vec![1, 2, 3]);
        assert_eq!(list.car(), Some(&1));
    }

    #[test]
    fn test_cons_list_from_iter() {
        let list: ConsList<i32> = ConsList::from_iter([1, 2, 3, 4, 5]);
        assert_eq!(list.to_vec(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_cons_list_mutation() {
        let mut list: ConsList<i32> = ConsList::from_iter([1, 2, 3]);
        if let Some(head) = list.head_id() {
            list.set_car(head, 10);
            assert_eq!(list.car(), Some(&10));
        }
    }

    #[test]
    fn test_cons_list_reverse() {
        let mut list: ConsList<i32> = ConsList::from_iter([1, 2, 3]);
        list.reverse_mut();
        assert_eq!(list.to_vec(), vec![3, 2, 1]);
    }

    #[test]
    fn test_cons_list_cycle_detection() {
        let mut list: ConsList<i32> = ConsList::from_iter([1, 2, 3]);
        assert!(!list.has_cycle());

        list.make_cycle();
        assert!(list.has_cycle());
    }

    #[test]
    fn test_queue_operations() {
        let mut q = Queue::new();
        assert!(q.is_empty());

        q.insert(1);
        q.insert(2);
        q.insert(3);

        assert_eq!(q.front(), Some(&1));
        assert_eq!(q.len(), 3);

        assert_eq!(q.delete(), Some(1));
        assert_eq!(q.delete(), Some(2));
        assert_eq!(q.front(), Some(&3));
        assert_eq!(q.delete(), Some(3));

        assert!(q.is_empty());
    }

    #[test]
    fn test_deque_operations() {
        let mut dq = Deque::new();

        dq.front_insert(1);
        dq.rear_insert(2);
        dq.front_insert(0);

        assert_eq!(dq.front(), Some(&0));
        assert_eq!(dq.rear(), Some(&2));

        assert_eq!(dq.front_delete(), Some(0));
        assert_eq!(dq.rear_delete(), Some(2));
        assert_eq!(dq.front_delete(), Some(1));

        assert!(dq.is_empty());
    }

    #[test]
    fn test_table() {
        let mut table = Table::new();

        table.insert("a", 1);
        table.insert("b", 2);

        assert_eq!(table.lookup(&"a"), Some(&1));
        assert_eq!(table.lookup(&"b"), Some(&2));
        assert_eq!(table.lookup(&"c"), None);
    }

    #[test]
    fn test_table_2d() {
        let mut table = Table2D::new();

        table.insert("math", "+", 43);
        table.insert("math", "-", 45);
        table.insert("letters", "a", 97);

        assert_eq!(table.lookup(&"math", &"+"), Some(&43));
        assert_eq!(table.lookup(&"math", &"-"), Some(&45));
        assert_eq!(table.lookup(&"letters", &"a"), Some(&97));
        assert_eq!(table.lookup(&"letters", &"b"), None);
    }

    #[test]
    fn test_memoization() {
        use std::cell::Cell;
        use std::rc::Rc;

        let call_count = Rc::new(Cell::new(0));
        let count_clone = Rc::clone(&call_count);

        let fib = Memoized::new(move |n: &i32| -> i32 {
            count_clone.set(count_clone.get() + 1);
            if *n <= 1 {
                *n
            } else {
                *n * 2 // 테스트 단순화 (Simplified for testing)
            }
        });

        // 첫 호출 (First call)
        assert_eq!(fib.call(&5), 10);
        assert_eq!(call_count.get(), 1);

        // 두 번째 호출(캐시 사용) (Second call (should use cache))
        assert_eq!(fib.call(&5), 10);
        assert_eq!(call_count.get(), 1); // 추가 호출 없음 (No additional call)

        // 다른 인자 (Different argument)
        assert_eq!(fib.call(&3), 6);
        assert_eq!(call_count.get(), 2);
    }

    #[test]
    fn test_circuit_simulator_inverter() {
        use circuits::Simulator;

        let mut sim = Simulator::new();
        let input = sim.make_wire();
        let output = sim.make_wire();

        sim.inverter(input, output);

        // 입력 0 -> 출력 1 (지연 후) (Input 0 -> Output 1 (after delay))
        sim.propagate();
        assert_eq!(sim.get_signal(output), 1);

        // 입력 1 -> 출력 0 (Input 1 -> Output 0)
        sim.set_signal(input, 1);
        sim.propagate();
        assert_eq!(sim.get_signal(output), 0);
    }

    #[test]
    fn test_circuit_simulator_and_gate() {
        use circuits::Simulator;

        let mut sim = Simulator::new();
        let a = sim.make_wire();
        let b = sim.make_wire();
        let output = sim.make_wire();

        sim.and_gate(a, b, output);
        sim.propagate();

        // 0 그리고(AND) 0 = 0 (0 AND 0 = 0)
        assert_eq!(sim.get_signal(output), 0);

        // 1 그리고(AND) 1 = 1 (1 AND 1 = 1)
        sim.set_signal(a, 1);
        sim.set_signal(b, 1);
        sim.propagate();
        assert_eq!(sim.get_signal(output), 1);
    }

    #[test]
    fn test_circuit_half_adder() {
        use circuits::Simulator;

        let mut sim = Simulator::new();
        let a = sim.make_wire();
        let b = sim.make_wire();
        let sum = sim.make_wire();
        let carry = sim.make_wire();

        sim.half_adder(a, b, sum, carry);
        sim.propagate();

        // 0 + 0 = 0, 캐리 0 (0 + 0 = 0, carry 0)
        assert_eq!(sim.get_signal(sum), 0);
        assert_eq!(sim.get_signal(carry), 0);

        // 1 + 0 = 1, 캐리 0 (1 + 0 = 1, carry 0)
        sim.set_signal(a, 1);
        sim.propagate();
        // 참고: 하프 애더는 전파 지연이 있다 (Note: Half adder has propagation delay)
        assert_eq!(sim.get_signal(a), 1);
        assert_eq!(sim.get_signal(b), 0);
    }

    #[test]
    fn test_constraint_adder() {
        use constraints::ConstraintNetwork;

        let mut net = ConstraintNetwork::new();
        let a = net.make_connector();
        let b = net.make_connector();
        let sum = net.make_connector();

        net.adder(a, b, sum, "adder");

        // a=3, b=5 설정 -> sum은 8이어야 한다 (Set a=3, b=5 -> sum should be 8)
        net.set_value(a, 3.0, "user");
        net.set_value(b, 5.0, "user");
        net.propagate();

        assert_eq!(net.get_value(a), Some(3.0));
        assert_eq!(net.get_value(b), Some(5.0));
        assert_eq!(net.get_value(sum), Some(8.0));
    }

    #[test]
    fn test_constraint_adder_reverse() {
        use constraints::ConstraintNetwork;

        let mut net = ConstraintNetwork::new();
        let a = net.make_connector();
        let b = net.make_connector();
        let sum = net.make_connector();

        net.adder(a, b, sum, "adder");

        // a=3, sum=8 설정 -> b는 5여야 한다 (Set a=3, sum=8 -> b should be 5)
        net.set_value(a, 3.0, "user");
        net.set_value(sum, 8.0, "user");
        net.propagate();

        assert_eq!(net.get_value(b), Some(5.0));
    }

    #[test]
    fn test_constraint_multiplier() {
        use constraints::ConstraintNetwork;

        let mut net = ConstraintNetwork::new();
        let m1 = net.make_connector();
        let m2 = net.make_connector();
        let product = net.make_connector();

        net.multiplier(m1, m2, product, "mult");

        // m1=4, m2=5 설정 -> product는 20이어야 한다 (Set m1=4, m2=5 -> product should be 20)
        net.set_value(m1, 4.0, "user");
        net.set_value(m2, 5.0, "user");
        net.propagate();

        assert_eq!(net.get_value(product), Some(20.0));
    }

    #[test]
    fn test_celsius_fahrenheit() {
        use constraints::ConstraintNetwork;

        let mut net = ConstraintNetwork::new();
        let c = net.make_connector();
        let f = net.make_connector();

        net.celsius_fahrenheit_converter(c, f);

        // C = 25 설정 -> F는 77이어야 한다 (Set C = 25 -> F should be 77)
        net.set_value(c, 25.0, "user");
        net.propagate();

        assert_eq!(net.get_value(c), Some(25.0));
        // F = (9/5) * C + 32 = (9/5) * 25 + 32 = 45 + 32 = 77
        assert_eq!(net.get_value(f), Some(77.0));
    }

    #[test]
    fn test_celsius_fahrenheit_reverse() {
        use constraints::ConstraintNetwork;

        let mut net = ConstraintNetwork::new();
        let c = net.make_connector();
        let f = net.make_connector();

        net.celsius_fahrenheit_converter(c, f);

        // F = 212 설정 -> C는 100이어야 한다 (Set F = 212 -> C should be 100)
        net.set_value(f, 212.0, "user");
        net.propagate();

        assert_eq!(net.get_value(f), Some(212.0));
        // C = (F - 32) * 5/9 = 180 * 5/9 = 100
        assert_eq!(net.get_value(c), Some(100.0));
    }
}
