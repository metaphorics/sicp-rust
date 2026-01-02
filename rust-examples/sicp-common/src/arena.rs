//! # Arena Allocation
//!
//! Type-safe arena allocation with index-based references, replacing
//! `Rc<RefCell<T>>` patterns for graph structures.
//!
//! ## Benefits over `Rc<RefCell<T>>`
//!
//! - No runtime borrow checking overhead
//! - Better cache locality (contiguous memory)
//! - Trivially `Send + Sync` (indices are just integers)
//! - Cycles are easy: just store indices
//!
//! ## Example
//!
//! ```
//! use sicp_common::arena::{Arena, ArenaId};
//!
//! // Define a node that can reference other nodes
//! #[derive(Debug)]
//! struct Node {
//!     value: i64,
//!     next: Option<ArenaId<Node>>,
//! }
//!
//! let mut arena: Arena<Node> = Arena::new();
//!
//! // Allocate nodes
//! let first = arena.alloc(Node { value: 1, next: None });
//! let second = arena.alloc(Node { value: 2, next: Some(first) });
//!
//! // Access nodes
//! assert_eq!(arena.get(first).value, 1);
//! assert_eq!(arena.get(second).next, Some(first));
//! ```

use std::marker::PhantomData;

/// A type-safe index into an [`Arena`].
///
/// `ArenaId<T>` is a lightweight handle (just a `usize`) that references
/// an element of type `T` in an arena. The `PhantomData<T>` ensures type
/// safety at compile time.
#[derive(Debug)]
pub struct ArenaId<T> {
    index: usize,
    _marker: PhantomData<T>,
}

// Manual implementations to avoid requiring T: Clone/Copy/etc.
impl<T> Clone for ArenaId<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ArenaId<T> {}

impl<T> PartialEq for ArenaId<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> Eq for ArenaId<T> {}

impl<T> std::hash::Hash for ArenaId<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

impl<T> ArenaId<T> {
    /// Returns the raw index of this arena ID.
    ///
    /// This is useful for debugging or serialization, but generally
    /// you should use the Arena methods to access elements.
    #[must_use]
    pub fn index(self) -> usize {
        self.index
    }
}

/// A simple arena allocator for type `T`.
///
/// Elements are stored contiguously and referenced by [`ArenaId<T>`].
/// Once allocated, elements cannot be removed (append-only).
///
/// ## Thread Safety
///
/// `Arena<T>` is `Send` and `Sync` if `T` is, since it just wraps a `Vec<T>`.
/// The `ArenaId<T>` indices are trivially thread-safe as they're just integers.
#[derive(Debug)]
pub struct Arena<T> {
    items: Vec<T>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Arena<T> {
    /// Creates a new empty arena.
    #[must_use]
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Creates a new arena with the specified capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
        }
    }

    /// Allocates a new element in the arena and returns its ID.
    ///
    /// # Example
    ///
    /// ```
    /// use sicp_common::arena::Arena;
    ///
    /// let mut arena = Arena::new();
    /// let id = arena.alloc(42);
    /// assert_eq!(arena.get(id), &42);
    /// ```
    pub fn alloc(&mut self, value: T) -> ArenaId<T> {
        let index = self.items.len();
        self.items.push(value);
        ArenaId {
            index,
            _marker: PhantomData,
        }
    }

    /// Returns a reference to the element at the given ID.
    ///
    /// # Panics
    ///
    /// Panics if the ID is out of bounds (invalid ID from a different arena).
    #[must_use]
    pub fn get(&self, id: ArenaId<T>) -> &T {
        &self.items[id.index]
    }

    /// Returns a mutable reference to the element at the given ID.
    ///
    /// # Panics
    ///
    /// Panics if the ID is out of bounds.
    #[must_use]
    pub fn get_mut(&mut self, id: ArenaId<T>) -> &mut T {
        &mut self.items[id.index]
    }

    /// Tries to get a reference to the element, returning `None` if invalid.
    #[must_use]
    pub fn try_get(&self, id: ArenaId<T>) -> Option<&T> {
        self.items.get(id.index)
    }

    /// Tries to get a mutable reference to the element, returning `None` if invalid.
    #[must_use]
    pub fn try_get_mut(&mut self, id: ArenaId<T>) -> Option<&mut T> {
        self.items.get_mut(id.index)
    }

    /// Returns the number of elements in the arena.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns true if the arena is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Returns an iterator over all elements in the arena.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    /// Returns an iterator over all elements with their IDs.
    pub fn iter_with_ids(&self) -> impl Iterator<Item = (ArenaId<T>, &T)> {
        self.items.iter().enumerate().map(|(index, item)| {
            (
                ArenaId {
                    index,
                    _marker: PhantomData,
                },
                item,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_and_get() {
        let mut arena: Arena<i64> = Arena::new();
        let id1 = arena.alloc(10);
        let id2 = arena.alloc(20);
        let id3 = arena.alloc(30);

        assert_eq!(arena.get(id1), &10);
        assert_eq!(arena.get(id2), &20);
        assert_eq!(arena.get(id3), &30);
    }

    #[test]
    fn test_get_mut() {
        let mut arena = Arena::new();
        let id = arena.alloc(String::from("hello"));

        arena.get_mut(id).push_str(" world");
        assert_eq!(arena.get(id), "hello world");
    }

    #[test]
    fn test_cyclic_reference() {
        #[derive(Debug)]
        struct Node {
            value: i64,
            next: Option<ArenaId<Node>>,
        }

        let mut arena: Arena<Node> = Arena::new();

        // Create a cycle: a -> b -> a
        let a = arena.alloc(Node {
            value: 1,
            next: None,
        });
        let b = arena.alloc(Node {
            value: 2,
            next: Some(a),
        });

        // Update a to point to b (creating cycle)
        arena.get_mut(a).next = Some(b);

        // Verify the cycle
        assert_eq!(arena.get(a).next, Some(b));
        assert_eq!(arena.get(b).next, Some(a));
    }

    #[test]
    fn test_arena_id_is_copy() {
        let mut arena: Arena<i64> = Arena::new();
        let id = arena.alloc(42);

        // ArenaId should be Copy
        let id_copy = id;
        assert_eq!(arena.get(id), arena.get(id_copy));
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut arena: Arena<i64> = Arena::new();
        assert!(arena.is_empty());
        assert_eq!(arena.len(), 0);

        arena.alloc(1);
        assert!(!arena.is_empty());
        assert_eq!(arena.len(), 1);

        arena.alloc(2);
        assert_eq!(arena.len(), 2);
    }

    #[test]
    fn test_iter_with_ids() {
        let mut arena: Arena<&str> = Arena::new();
        let id_a = arena.alloc("a");
        let id_b = arena.alloc("b");

        let collected: Vec<_> = arena.iter_with_ids().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0], (id_a, &"a"));
        assert_eq!(collected[1], (id_b, &"b"));
    }
}
