//! SICP Section 5.3: Storage Allocation and Garbage Collection
//!
//! This module demonstrates how list structure can be represented using
//! vector-based memory, implementing the concepts from SICP 5.3.
//!
//! # Key Concepts
//!
//! - **Memory as Vectors**: Representing pairs using parallel `the-cars` and `the-cdrs` vectors
//! - **Typed Pointers**: Tags distinguish pairs, numbers, symbols, etc.
//! - **Garbage Collection**: Stop-and-copy and mark-sweep algorithms
//! - **Rust's Ownership**: Compile-time garbage collection vs runtime GC
//!
//! # SICP to Rust Mapping
//!
//! | SICP Concept | Rust Implementation |
//! |--------------|---------------------|
//! | `the-cars/the-cdrs` | `Vec<Value>` parallel vectors |
//! | Typed pointer | `Value` enum with discriminants |
//! | `cons` | Allocate at `free` index |
//! | Stop-and-copy GC | Two-space copying collector |
//! | Broken heart | `Value::BrokenHeart(forwarding_addr)` |
//! | Automatic GC | Rust's ownership = compile-time GC |

use std::fmt;

/// Type index for a pair in memory
pub type PairIndex = usize;

/// Value represents a tagged pointer in the Lisp memory system.
///
/// This corresponds to SICP's "typed pointers" that include both
/// data type information and the actual value or index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    /// Numeric value (e.g., n4 represents the number 4)
    Number(i64),

    /// Symbol (interned string)
    Symbol(String),

    /// Pair pointer (e.g., p5 represents index 5 into cars/cdrs)
    Pair(PairIndex),

    /// Empty list
    Nil,

    /// Broken heart marker with forwarding address (for GC)
    BrokenHeart(usize),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Number(n) => write!(f, "n{}", n),
            Value::Symbol(s) => write!(f, "'{}", s),
            Value::Pair(idx) => write!(f, "p{}", idx),
            Value::Nil => write!(f, "e0"),
            Value::BrokenHeart(addr) => write!(f, "broken-heart->{}", addr),
        }
    }
}

impl Value {
    /// Check if this value is a pair pointer
    pub fn is_pair(&self) -> bool {
        matches!(self, Value::Pair(_))
    }

    /// Check if this value is nil
    pub fn is_nil(&self) -> bool {
        matches!(self, Value::Nil)
    }

    /// Check if this is a broken heart (relocated pair)
    pub fn is_broken_heart(&self) -> bool {
        matches!(self, Value::BrokenHeart(_))
    }
}

/// Memory represents the vector-based storage for list structures.
///
/// This implements the memory model from SICP Figure 5.14, where
/// pairs are stored in parallel vectors indexed by pair pointers.
#[derive(Debug, Clone)]
pub struct Memory {
    /// The-cars vector: stores car fields of pairs
    pub the_cars: Vec<Value>,

    /// The-cdrs vector: stores cdr fields of pairs
    pub the_cdrs: Vec<Value>,

    /// Free pointer: next available index for cons
    pub free: usize,

    /// Capacity of the memory
    capacity: usize,
}

impl Memory {
    /// Create a new memory with the specified capacity
    pub fn new(capacity: usize) -> Self {
        Memory {
            the_cars: vec![Value::Nil; capacity],
            the_cdrs: vec![Value::Nil; capacity],
            free: 0,
            capacity,
        }
    }

    /// Allocate a new pair (cons operation)
    ///
    /// Corresponds to SICP's cons implementation:
    /// ```scheme
    /// (perform (op vector-set!) (reg the-cars) (reg free) (reg car-val))
    /// (perform (op vector-set!) (reg the-cdrs) (reg free) (reg cdr-val))
    /// (assign result (reg free))
    /// (assign free (op +) (reg free) (const 1))
    /// ```
    pub fn cons(&mut self, car: Value, cdr: Value) -> Result<Value, &'static str> {
        if self.free >= self.capacity {
            return Err("Out of memory");
        }

        let index = self.free;
        self.the_cars[index] = car;
        self.the_cdrs[index] = cdr;
        self.free += 1;

        Ok(Value::Pair(index))
    }

    /// Get the car of a pair
    ///
    /// Corresponds to:
    /// ```scheme
    /// (assign result (op vector-ref) (reg the-cars) (reg pair-index))
    /// ```
    pub fn car(&self, pair: &Value) -> Result<&Value, &'static str> {
        match pair {
            Value::Pair(index) => {
                if *index < self.the_cars.len() {
                    Ok(&self.the_cars[*index])
                } else {
                    Err("Invalid pair index")
                }
            }
            _ => Err("Not a pair"),
        }
    }

    /// Get the cdr of a pair
    pub fn cdr(&self, pair: &Value) -> Result<&Value, &'static str> {
        match pair {
            Value::Pair(index) => {
                if *index < self.the_cdrs.len() {
                    Ok(&self.the_cdrs[*index])
                } else {
                    Err("Invalid pair index")
                }
            }
            _ => Err("Not a pair"),
        }
    }

    /// Set the car of a pair (set-car!)
    pub fn set_car(&mut self, pair: &Value, value: Value) -> Result<(), &'static str> {
        match pair {
            Value::Pair(index) => {
                if *index < self.the_cars.len() {
                    self.the_cars[*index] = value;
                    Ok(())
                } else {
                    Err("Invalid pair index")
                }
            }
            _ => Err("Not a pair"),
        }
    }

    /// Set the cdr of a pair (set-cdr!)
    pub fn set_cdr(&mut self, pair: &Value, value: Value) -> Result<(), &'static str> {
        match pair {
            Value::Pair(index) => {
                if *index < self.the_cdrs.len() {
                    self.the_cdrs[*index] = value;
                    Ok(())
                } else {
                    Err("Invalid pair index")
                }
            }
            _ => Err("Not a pair"),
        }
    }

    /// Build a list from values (helper function)
    pub fn list(&mut self, values: Vec<Value>) -> Result<Value, &'static str> {
        let mut result = Value::Nil;
        for value in values.into_iter().rev() {
            result = self.cons(value, result)?;
        }
        Ok(result)
    }
}

/// Stop-and-copy garbage collector
///
/// Implements the algorithm from SICP 5.3.2, which divides memory into
/// two halves: working memory and free memory. During GC, all reachable
/// pairs are copied from working memory to free memory, then the roles
/// are swapped.
pub struct StopAndCopyGC {
    /// Working memory (current allocation space)
    working: Memory,

    /// Free memory (target for GC)
    free_space: Memory,

    /// Root set (registers containing live pointers)
    roots: Vec<Value>,
}

impl StopAndCopyGC {
    /// Create a new GC with specified memory size per space
    pub fn new(size: usize) -> Self {
        StopAndCopyGC {
            working: Memory::new(size),
            free_space: Memory::new(size),
            roots: Vec::new(),
        }
    }

    /// Add a root (live pointer that should not be collected)
    pub fn add_root(&mut self, value: Value) {
        self.roots.push(value);
    }

    /// Allocate a pair in working memory
    pub fn cons(&mut self, car: Value, cdr: Value) -> Result<Value, &'static str> {
        self.working.cons(car, cdr)
    }

    /// Perform garbage collection
    ///
    /// This implements the stop-and-copy algorithm from SICP:
    /// 1. Initialize scan and free pointers to 0
    /// 2. Relocate all roots
    /// 3. Scan copied pairs and relocate their car/cdr
    /// 4. Flip working and free memory
    pub fn collect(&mut self) -> Result<(), &'static str> {
        // Initialize new memory
        self.free_space = Memory::new(self.working.capacity);
        let mut scan = 0;

        // Relocate all roots (collect first to avoid borrow conflict)
        let roots_to_relocate: Vec<Value> = self.roots.drain(..).collect();
        for root in roots_to_relocate {
            let relocated = self.relocate(root)?;
            self.roots.push(relocated);
        }

        // Scan and relocate reachable pairs
        while scan < self.free_space.free {
            // Get car and cdr of pair at scan position
            let car = self.free_space.the_cars[scan].clone();
            let cdr = self.free_space.the_cdrs[scan].clone();

            // Relocate car and cdr
            let new_car = self.relocate(car)?;
            let new_cdr = self.relocate(cdr)?;

            // Update the pair in new memory
            self.free_space.the_cars[scan] = new_car;
            self.free_space.the_cdrs[scan] = new_cdr;

            scan += 1;
        }

        // Flip: swap working and free memory
        std::mem::swap(&mut self.working, &mut self.free_space);

        Ok(())
    }

    /// Relocate a value to new memory
    ///
    /// Implements the relocate-old-result-in-new subroutine from SICP:
    /// - Non-pairs are returned unchanged
    /// - Already-moved pairs return their forwarding address
    /// - Fresh pairs are copied and marked with broken heart
    fn relocate(&mut self, value: Value) -> Result<Value, &'static str> {
        match value {
            Value::Pair(old_index) => {
                // Check if already moved (broken heart)
                if let Value::BrokenHeart(new_index) = self.working.the_cars[old_index] {
                    return Ok(Value::Pair(new_index));
                }

                // Copy to new memory
                let car = self.working.the_cars[old_index].clone();
                let cdr = self.working.the_cdrs[old_index].clone();

                let new_index = self.free_space.free;
                if new_index >= self.free_space.capacity {
                    return Err("Out of memory during GC");
                }

                self.free_space.the_cars[new_index] = car;
                self.free_space.the_cdrs[new_index] = cdr;
                self.free_space.free += 1;

                // Mark old location with broken heart
                self.working.the_cars[old_index] = Value::BrokenHeart(new_index);

                Ok(Value::Pair(new_index))
            }
            // Non-pairs are not relocated
            _ => Ok(value),
        }
    }

    /// Get memory reference for accessing pairs
    pub fn memory(&self) -> &Memory {
        &self.working
    }
}

/// Mark-sweep garbage collector (alternative to stop-and-copy)
///
/// Implements the mark-sweep algorithm mentioned in SICP footnote:
/// 1. Mark phase: trace from roots and mark all reachable pairs
/// 2. Sweep phase: scan all memory and reclaim unmarked pairs
pub struct MarkSweepGC {
    /// Memory storage
    memory: Memory,

    /// Mark bits (true if reachable)
    marked: Vec<bool>,

    /// Root set
    roots: Vec<Value>,

    /// Free list (indices of available cells)
    free_list: Vec<usize>,
}

impl MarkSweepGC {
    /// Create a new mark-sweep collector
    pub fn new(size: usize) -> Self {
        MarkSweepGC {
            memory: Memory::new(size),
            marked: vec![false; size],
            roots: Vec::new(),
            free_list: (0..size).collect(),
        }
    }

    /// Add a root
    pub fn add_root(&mut self, value: Value) {
        self.roots.push(value);
    }

    /// Allocate a pair using free list
    pub fn cons(&mut self, car: Value, cdr: Value) -> Result<Value, &'static str> {
        if let Some(index) = self.free_list.pop() {
            self.memory.the_cars[index] = car;
            self.memory.the_cdrs[index] = cdr;
            Ok(Value::Pair(index))
        } else {
            Err("Out of memory")
        }
    }

    /// Perform mark-sweep garbage collection
    pub fn collect(&mut self) {
        // Clear all marks
        self.marked.fill(false);

        // Mark phase: mark all reachable pairs (clone roots to avoid borrow conflict)
        let roots_to_mark = self.roots.clone();
        for root in &roots_to_mark {
            self.mark(root);
        }

        // Sweep phase: reclaim unmarked pairs
        self.free_list.clear();
        for i in 0..self.memory.capacity {
            if !self.marked[i] {
                self.free_list.push(i);
            }
        }
    }

    /// Mark a value and all pairs reachable from it
    fn mark(&mut self, value: &Value) {
        if let Value::Pair(index) = value {
            if *index < self.marked.len() && !self.marked[*index] {
                self.marked[*index] = true;

                // Recursively mark car and cdr
                let car = self.memory.the_cars[*index].clone();
                let cdr = self.memory.the_cdrs[*index].clone();

                self.mark(&car);
                self.mark(&cdr);
            }
        }
    }

    /// Get memory reference
    pub fn memory(&self) -> &Memory {
        &self.memory
    }
}

/// Exercise 5.20: Build the list structure for (define x (cons 1 2)) (define y (list x x))
///
/// This demonstrates the memory-vector representation from SICP Figure 5.14.
///
/// Starting with free = 1:
/// - x = (cons 1 2) creates p1 with cars[1]=n1, cdrs[1]=n2, free becomes 2
/// - y = (list x x) = (cons x (cons x nil))
///   - Inner (cons x nil) creates p2 with cars[2]=p1, cdrs[2]=e0, free becomes 3
///   - Outer (cons x p2) creates p3 with cars[3]=p1, cdrs[3]=p2, free becomes 4
/// - Final: x=p1, y=p3, free=4
pub fn exercise_5_20() -> Result<(Memory, Value, Value), &'static str> {
    let mut mem = Memory::new(10);

    // Skip index 0 (start at p1)
    mem.free = 1;

    // x = (cons 1 2)
    let x = mem.cons(Value::Number(1), Value::Number(2))?;

    // y = (list x x) = (cons x (cons x nil))
    let inner = mem.cons(x.clone(), Value::Nil)?;
    let y = mem.cons(x.clone(), inner)?;

    Ok((mem, x, y))
}

// ============================================================================
// RUST OWNERSHIP: Compile-Time Garbage Collection
// ============================================================================

/// Rust's ownership system is essentially compile-time garbage collection.
///
/// While SICP demonstrates runtime GC (stop-and-copy, mark-sweep), Rust
/// achieves memory safety without runtime overhead through:
///
/// 1. **Ownership**: Each value has exactly one owner
/// 2. **Borrowing**: References must follow lifetime rules
/// 3. **Drop**: RAII ensures cleanup when owner goes out of scope
///
/// This section demonstrates how Rust's approach eliminates the need
/// for runtime garbage collection in most cases.

/// Traditional linked list with manual memory management (conceptual)
///
/// In languages with GC, creating temporary structures creates garbage:
/// ```scheme
/// (accumulate + 0 (filter odd? (enumerate-interval 0 n)))
/// ```
///
/// Both the enumeration list and filtered list become garbage after use.
pub mod ownership_examples {
    #[allow(unused_imports)]
    use super::*;

    /// Rust's Box<T> demonstrates ownership-based memory management
    ///
    /// No GC needed - memory freed when Box goes out of scope
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

    /// Iterator-based approach - zero allocation for intermediate results
    ///
    /// Contrast with SICP's filter/enumerate which creates temporary lists
    pub fn sum_odd_numbers(n: i32) -> i32 {
        (0..=n)
            .filter(|x| x % 2 != 0) // No allocation - lazy iteration
            .sum() // No garbage created
    }

    /// Demonstrates Rust's automatic cleanup (Drop trait)
    pub struct Resource {
        id: usize,
    }

    impl Drop for Resource {
        fn drop(&mut self) {
            // Cleanup happens automatically - no GC needed
            println!("Resource {} cleaned up", self.id);
        }
    }

    /// Compare GC vs ownership models
    pub fn demonstrate_ownership() {
        // Scope-based lifetime - deterministic cleanup
        {
            let _r1 = Resource { id: 1 };
            let _r2 = Resource { id: 2 };
            // r1 and r2 dropped here (deterministic, no GC pause)
        }

        // List with ownership
        let list = List::cons(1, List::cons(2, List::cons(3, List::nil())));
        // Entire list freed when 'list' goes out of scope
        drop(list); // Explicit drop (usually automatic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_cons_car_cdr() {
        let mut mem = Memory::new(10);

        // Test cons
        let pair = mem.cons(Value::Number(1), Value::Number(2)).unwrap();
        assert!(pair.is_pair());

        // Test car and cdr
        assert_eq!(mem.car(&pair).unwrap(), &Value::Number(1));
        assert_eq!(mem.cdr(&pair).unwrap(), &Value::Number(2));
    }

    #[test]
    fn test_list_construction() {
        let mut mem = Memory::new(10);

        // Build ((1 2) 3 4) from Figure 5.14
        let inner_pair = mem.cons(Value::Number(1), Value::Number(2)).unwrap();
        let last_pair = mem.cons(Value::Number(4), Value::Nil).unwrap();
        let cdr = mem.cons(Value::Number(3), last_pair).unwrap();
        let result = mem.cons(inner_pair, cdr).unwrap();

        // Verify structure
        assert!(result.is_pair());
        let car = mem.car(&result).unwrap();
        assert!(car.is_pair());
    }

    #[test]
    fn test_stop_and_copy_gc() {
        let mut gc = StopAndCopyGC::new(10);

        // Create some pairs
        let p1 = gc.cons(Value::Number(1), Value::Number(2)).unwrap();
        let p2 = gc.cons(Value::Number(3), Value::Number(4)).unwrap();
        let p3 = gc.cons(p1.clone(), p2.clone()).unwrap();

        // Only p3 is a root (p1 and p2 are reachable through p3)
        gc.add_root(p3.clone());

        // Perform GC
        gc.collect().unwrap();

        // Verify p3 still valid (relocated)
        assert!(gc.roots[0].is_pair());
    }

    #[test]
    fn test_mark_sweep_gc() {
        let mut gc = MarkSweepGC::new(10);

        // Create some pairs
        let p1 = gc.cons(Value::Number(1), Value::Number(2)).unwrap();
        let _p2 = gc.cons(Value::Number(3), Value::Number(4)).unwrap(); // Garbage

        // Only p1 is root
        gc.add_root(p1);

        // Before GC: 2 cells used, 8 free
        assert_eq!(gc.free_list.len(), 8);

        // After GC: 1 cell live, 9 free (p2 collected)
        gc.collect();
        assert_eq!(gc.free_list.len(), 9);
    }

    #[test]
    fn test_exercise_5_20() {
        let (mem, x, y) = exercise_5_20().unwrap();

        // x should be p1
        assert_eq!(x, Value::Pair(1));

        // y should be p3
        assert_eq!(y, Value::Pair(3));

        // free should be 4
        assert_eq!(mem.free, 4);

        // Verify memory contents
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

        // Before GC: no broken hearts
        assert!(!gc.working.the_cars[0].is_broken_heart());

        // After GC: old location has broken heart
        gc.collect().unwrap();
        assert!(gc.free_space.the_cars[0].is_broken_heart());
    }

    #[test]
    fn test_ownership_list() {
        use ownership_examples::List;

        // Build list (1 2 3)
        let list = List::cons(1, List::cons(2, List::cons(3, List::nil())));

        // When list goes out of scope, entire chain is freed
        // No GC needed - ownership handles it
        drop(list);
    }

    #[test]
    fn test_ownership_sum_odd() {
        use ownership_examples::sum_odd_numbers;

        // No garbage created - pure iteration
        let result = sum_odd_numbers(10);
        assert_eq!(result, 1 + 3 + 5 + 7 + 9);
    }
}
