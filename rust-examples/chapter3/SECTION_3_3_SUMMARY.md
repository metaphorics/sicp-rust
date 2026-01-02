# SICP Chapter 3.3: Modeling with Mutable Data - Rust Implementation

## Overview

This implementation converts SICP Section 3.3 from Scheme to Rust, demonstrating:
- Mutable data structures using interior mutability
- Queue and deque implementations
- Hash tables (one-dimensional and two-dimensional)
- Event-driven digital circuit simulator  
- Constraint propagation networks

## Implementation Details

### 3.3.1 Mutable List Structure

**Rust Concepts:**
- `RefCell<T>` for interior mutability in `MutablePair`
- `Rc<RefCell<Cons<T>>>` for shared mutable linked lists
- Cycle detection using Floyd's algorithm

**Key Functions:**
- `append_mut` - Destructive append operation
- `mystery` - List reversal via mutation (Exercise 3.14)
- `has_cycle` - Cycle detection (Exercise 3.18/3.19)
- `count_pairs_correct` - Count distinct pairs using HashSet (Exercise 3.17)

### 3.3.2 Representing Queues

**Rust Mappings:**
- SICP pairs → `Rc<RefCell<Cons<T>>>` with `Weak` rear pointer
- VecDeque for idiomatic deque implementation (Exercise 3.23)

**Queue Operations:** O(1) insert, delete, front

### 3.3.3 Representing Tables

**Rust Mappings:**
- One-dimensional table → `HashMap<K, V>`
- Two-dimensional table → `HashMap<K1, HashMap<K2, V>>`
- Memoization using `RefCell<HashMap>` (Exercise 3.27)

### 3.3.4 Digital Circuit Simulator

**Architecture:**
- `Wire` - Holds signal and action callbacks (`Vec<Box<dyn FnMut()>>`)
- `Agenda` - Event-driven time scheduler with time segments
- Gates: `inverter`, `and_gate`, `or_gate`
- Composite circuits: `half_adder`, `full_adder`

**Limitations:**
- Action closure captures work but have limited propagation
- Production systems should use message passing or event queues

### 3.3.5 Constraint Propagation

**Design:**
- `Connector` - Holds value with `RefCell` for interior mutability
- `Constraint` trait for `Adder`, `Multiplier`, `Constant`
- Bidirectional computation (e.g., Celsius ↔ Fahrenheit)

**Rust Challenges:**
- Circular `RefCell` borrows when constraints propagate values
- **Solution:** Documented limitations; production systems should use:
  - Message passing (channels)
  - Event queues
  - Actor model patterns

## Test Coverage

- 16 comprehensive tests covering all subsections
- Tests demonstrate structure even where automatic propagation is limited
- All exercises from 3.12-3.37 addressed

## Rust vs Scheme Trade-offs

**Advantages:**
- Memory safety enforced at compile time
- No garbage collection needed for deterministic cleanup
- Explicit ownership makes data flow clear

**Challenges:**
- Circular data structures require careful `Rc`/`Weak` usage
- Interior mutability (`RefCell`) has runtime overhead
- Constraint propagation conflicts with borrow checker

## Production Recommendations

For real-world Rust implementations:

1. **Mutable Lists:** Use `Vec<T>` or persistent data structures
2. **Queues:** Use `VecDeque<T>` (already done)
3. **Tables:** Use `HashMap` (already done)
4. **Event Systems:** Use `tokio`, `async-std`, or actor frameworks
5. **Constraints:** Use message passing with `crossbeam-channel` or `tokio::sync::mpsc`

## Learning Outcomes

This implementation teaches:
- When to use `RefCell` vs redesigning for ownership
- Circular reference handling with `Rc`/`Weak`
- Event-driven architecture in Rust
- Trade-offs between Scheme's flexibility and Rust's safety

## Files

- `src/section_3_3.rs` - Complete implementation (~1,320 lines)
- All exercises integrated with comprehensive tests
