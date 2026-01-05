//! # 아레나 할당 (Arena Allocation)
//!
//! 인덱스 기반 참조를 사용하는 타입 안전 아레나 할당으로,
//! 그래프 구조에서 `Rc<RefCell<T>>` 패턴을 대체한다
//! (Type-safe arena allocation with index-based references, replacing
//! `Rc<RefCell<T>>` patterns for graph structures).
//!
//! ## `Rc<RefCell<T>>` 대비 장점 (Benefits over `Rc<RefCell<T>>`)
//!
//! - 런타임 빌림 검사 오버헤드 없음 (No runtime borrow checking overhead)
//! - 더 나은 캐시 지역성(연속 메모리) (Better cache locality (contiguous memory))
//! - 자명하게 `Send + Sync` (인덱스는 단순 정수) (Trivially `Send + Sync` (indices are just integers))
//! - 사이클이 쉬움: 인덱스만 저장 (Cycles are easy: just store indices)
//!
//! ## 예시 (Example)
//!
//! ```
//! use sicp_common::arena::{Arena, ArenaId};
//!
//! // 다른 노드를 참조할 수 있는 노드를 정의 (Define a node that can reference other nodes)
//! #[derive(Debug)]
//! struct Node {
//!     value: i64,
//!     next: Option<ArenaId<Node>>,
//! }
//!
//! let mut arena: Arena<Node> = Arena::new();
//!
//! // 노드 할당 (Allocate nodes)
//! let first = arena.alloc(Node { value: 1, next: None });
//! let second = arena.alloc(Node { value: 2, next: Some(first) });
//!
//! // 노드 접근 (Access nodes)
//! assert_eq!(arena.get(first).value, 1);
//! assert_eq!(arena.get(second).next, Some(first));
//! ```

use std::marker::PhantomData;

/// [`Arena`]에 대한 타입 안전 인덱스 (A type-safe index into an [`Arena`]).
///
/// `ArenaId<T>`는 아레나의 `T` 타입 원소를 참조하는 가벼운 핸들(단지 `usize`)이다
/// (`ArenaId<T>` is a lightweight handle (just a `usize`) that references
/// an element of type `T` in an arena).
/// `PhantomData<T>`는 컴파일 타임 타입 안전을 보장한다
/// (The `PhantomData<T>` ensures type safety at compile time).
#[derive(Debug)]
pub struct ArenaId<T> {
    index: usize,
    _marker: PhantomData<T>,
}

// T: Clone/Copy 등을 요구하지 않기 위한 수동 구현 (Manual implementations to avoid requiring T: Clone/Copy/etc.).
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
    /// 이 아레나 ID의 원시 인덱스를 반환한다 (Returns the raw index of this arena ID).
    ///
    /// 디버깅이나 직렬화에 유용하지만, 일반적으로는
    /// 아레나 메서드를 사용해 원소에 접근해야 한다
    /// (This is useful for debugging or serialization, but generally
    /// you should use the Arena methods to access elements).
    #[must_use]
    pub fn index(self) -> usize {
        self.index
    }
}

/// 타입 `T`를 위한 단순 아레나 할당기 (A simple arena allocator for type `T`).
///
/// 원소는 연속으로 저장되고 [`ArenaId<T>`]로 참조된다
/// (Elements are stored contiguously and referenced by [`ArenaId<T>`]).
/// 한 번 할당되면 원소를 제거할 수 없다(append-only)
/// (Once allocated, elements cannot be removed (append-only)).
///
/// ## 스레드 안전성 (Thread Safety)
///
/// `Arena<T>`는 `Vec<T>`를 감싸기 때문에 `T`가 `Send`/`Sync`이면 동일하게 보장된다
/// (`Arena<T>` is `Send` and `Sync` if `T` is, since it just wraps a `Vec<T>`).
/// `ArenaId<T>` 인덱스는 단순 정수이므로 자명하게 스레드 안전하다
/// (The `ArenaId<T>` indices are trivially thread-safe as they're just integers).
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
    /// 새로운 빈 아레나를 생성한다 (Creates a new empty arena).
    #[must_use]
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// 지정된 용량으로 새 아레나를 생성한다 (Creates a new arena with the specified capacity).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
        }
    }

    /// 아레나에 새 원소를 할당하고 그 ID를 반환한다
    /// (Allocates a new element in the arena and returns its ID).
    ///
    /// # 예시 (Example)
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

    /// 주어진 ID의 원소에 대한 참조를 반환한다 (Returns a reference to the element at the given ID).
    ///
    /// # 패닉 (Panics)
    ///
    /// ID가 범위를 벗어나면 패닉한다(다른 아레나의 잘못된 ID)
    /// (Panics if the ID is out of bounds (invalid ID from a different arena)).
    #[must_use]
    pub fn get(&self, id: ArenaId<T>) -> &T {
        &self.items[id.index]
    }

    /// 주어진 ID의 원소에 대한 가변 참조를 반환한다
    /// (Returns a mutable reference to the element at the given ID).
    ///
    /// # 패닉 (Panics)
    ///
    /// ID가 범위를 벗어나면 패닉한다 (Panics if the ID is out of bounds).
    #[must_use]
    pub fn get_mut(&mut self, id: ArenaId<T>) -> &mut T {
        &mut self.items[id.index]
    }

    /// 원소 참조를 시도하고, 유효하지 않으면 `None`을 반환한다
    /// (Tries to get a reference to the element, returning `None` if invalid).
    #[must_use]
    pub fn try_get(&self, id: ArenaId<T>) -> Option<&T> {
        self.items.get(id.index)
    }

    /// 가변 참조를 시도하고, 유효하지 않으면 `None`을 반환한다
    /// (Tries to get a mutable reference to the element, returning `None` if invalid).
    #[must_use]
    pub fn try_get_mut(&mut self, id: ArenaId<T>) -> Option<&mut T> {
        self.items.get_mut(id.index)
    }

    /// 아레나의 원소 개수를 반환한다 (Returns the number of elements in the arena).
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// 아레나가 비어 있으면 true를 반환한다 (Returns true if the arena is empty).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// 아레나의 모든 원소에 대한 이터레이터를 반환한다
    /// (Returns an iterator over all elements in the arena).
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    /// ID와 함께 모든 원소의 이터레이터를 반환한다
    /// (Returns an iterator over all elements with their IDs).
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
        let id = arena.alloc(String::from("안녕 (hello)"));

        arena.get_mut(id).push_str(" 세계 (world)");
        assert_eq!(arena.get(id), "안녕 (hello) 세계 (world)");
    }

    #[test]
    fn test_cyclic_reference() {
        #[derive(Debug)]
        struct Node {
            value: i64,
            next: Option<ArenaId<Node>>,
        }

        let mut arena: Arena<Node> = Arena::new();

        // 사이클 생성: a -> b -> a (Create a cycle: a -> b -> a)
        let a = arena.alloc(Node {
            value: 1,
            next: None,
        });
        let b = arena.alloc(Node {
            value: 2,
            next: Some(a),
        });

        // a가 b를 가리키도록 갱신(사이클 생성) (Update a to point to b (creating cycle))
        arena.get_mut(a).next = Some(b);

        // 사이클 검증 (Verify the cycle)
        assert_eq!(arena.get(a).next, Some(b));
        assert_eq!(arena.get(b).next, Some(a));
    }

    #[test]
    fn test_arena_id_is_copy() {
        let mut arena: Arena<i64> = Arena::new();
        let id = arena.alloc(42);

        // ArenaId는 Copy여야 한다 (ArenaId should be Copy)
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
