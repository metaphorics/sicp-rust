//! # Functional List Operations
//!
//! Idiomatic Rust patterns for Scheme-style list operations.
//! These replace `cons`/`car`/`cdr` with iterator-based approaches.
//!
//! ## Philosophy
//!
//! In Scheme, lists are built from cons cells and manipulated with
//! `car`, `cdr`, `map`, `filter`, `fold`, etc. In idiomatic Rust:
//!
//! - Use `Vec<T>` or slices `&[T]` for sequences
//! - Use iterators (`.iter()`, `.map()`, `.filter()`, `.fold()`) for transformations
//! - Use tuples `(A, B)` for pairs
//! - Use pattern matching for destructuring
//!
//! This module provides helper functions that bridge Scheme patterns to Rust idioms.
//!
//! ## Example
//!
//! ```
//! use sicp_common::list::{cons, car, cdr, list_ref, length};
//!
//! // Scheme: (cons 1 (cons 2 (cons 3 '())))
//! let lst = vec![1, 2, 3];
//!
//! // Scheme: (car lst) => 1
//! assert_eq!(car(&lst), Some(&1));
//!
//! // Scheme: (cdr lst) => (2 3)
//! assert_eq!(cdr(&lst), &[2, 3]);
//!
//! // Scheme: (list-ref lst 1) => 2
//! assert_eq!(list_ref(&lst, 1), Some(&2));
//!
//! // Scheme: (length lst) => 3
//! assert_eq!(length(&lst), 3);
//! ```

/// A cons pair (head, tail).
///
/// In Scheme, `(cons a b)` creates a pair. In Rust, we use tuples.
/// This type alias makes the connection explicit.
pub type Pair<H, T> = (H, T);

/// Creates a pair (cons cell).
///
/// # Scheme equivalent
/// ```scheme
/// (cons 1 2) => (1 . 2)
/// ```
///
/// # Example
/// ```
/// use sicp_common::list::cons;
/// let pair = cons(1, 2);
/// assert_eq!(pair, (1, 2));
/// ```
#[inline]
pub fn cons<H, T>(head: H, tail: T) -> Pair<H, T> {
    (head, tail)
}

/// Returns the first element of a slice (car).
///
/// # Scheme equivalent
/// ```scheme
/// (car '(1 2 3)) => 1
/// ```
///
/// # Example
/// ```
/// use sicp_common::list::car;
/// assert_eq!(car(&[1, 2, 3]), Some(&1));
/// assert_eq!(car::<i32>(&[]), None);
/// ```
#[inline]
pub fn car<T>(list: &[T]) -> Option<&T> {
    list.first()
}

/// Returns all but the first element of a slice (cdr).
///
/// # Scheme equivalent
/// ```scheme
/// (cdr '(1 2 3)) => (2 3)
/// ```
///
/// # Example
/// ```
/// use sicp_common::list::cdr;
/// assert_eq!(cdr(&[1, 2, 3]), &[2, 3]);
/// assert_eq!(cdr(&[1]), &[] as &[i32]);
/// assert_eq!(cdr::<i32>(&[]), &[] as &[i32]);
/// ```
#[inline]
pub fn cdr<T>(list: &[T]) -> &[T] {
    if list.is_empty() { &[] } else { &list[1..] }
}

/// Returns the element at the given index.
///
/// # Scheme equivalent
/// ```scheme
/// (list-ref '(a b c) 1) => b
/// ```
///
/// # Example
/// ```
/// use sicp_common::list::list_ref;
/// assert_eq!(list_ref(&['a', 'b', 'c'], 1), Some(&'b'));
/// assert_eq!(list_ref(&['a', 'b', 'c'], 5), None);
/// ```
#[inline]
pub fn list_ref<T>(list: &[T], index: usize) -> Option<&T> {
    list.get(index)
}

/// Returns the length of a list.
///
/// # Scheme equivalent
/// ```scheme
/// (length '(1 2 3)) => 3
/// ```
#[inline]
pub fn length<T>(list: &[T]) -> usize {
    list.len()
}

/// Checks if a list is null (empty).
///
/// # Scheme equivalent
/// ```scheme
/// (null? '()) => #t
/// (null? '(1)) => #f
/// ```
#[inline]
pub fn is_null<T>(list: &[T]) -> bool {
    list.is_empty()
}

/// Appends two lists.
///
/// # Scheme equivalent
/// ```scheme
/// (append '(1 2) '(3 4)) => (1 2 3 4)
/// ```
///
/// # Example
/// ```
/// use sicp_common::list::append;
/// let result = append(&[1, 2], &[3, 4]);
/// assert_eq!(result, vec![1, 2, 3, 4]);
/// ```
pub fn append<T: Clone>(list1: &[T], list2: &[T]) -> Vec<T> {
    let mut result = list1.to_vec();
    result.extend_from_slice(list2);
    result
}

/// Reverses a list.
///
/// # Scheme equivalent
/// ```scheme
/// (reverse '(1 2 3)) => (3 2 1)
/// ```
///
/// # Example
/// ```
/// use sicp_common::list::reverse;
/// assert_eq!(reverse(&[1, 2, 3]), vec![3, 2, 1]);
/// ```
pub fn reverse<T: Clone>(list: &[T]) -> Vec<T> {
    list.iter().rev().cloned().collect()
}

/// Maps a function over a list.
///
/// # Scheme equivalent
/// ```scheme
/// (map square '(1 2 3)) => (1 4 9)
/// ```
///
/// # Note
/// Prefer using `.iter().map(f).collect()` directly in idiomatic Rust.
///
/// # Example
/// ```
/// use sicp_common::list::map;
/// let squares = map(|x| x * x, &[1, 2, 3]);
/// assert_eq!(squares, vec![1, 4, 9]);
/// ```
pub fn map<T, U, F>(f: F, list: &[T]) -> Vec<U>
where
    F: Fn(&T) -> U,
{
    list.iter().map(f).collect()
}

/// Filters a list by a predicate.
///
/// # Scheme equivalent
/// ```scheme
/// (filter even? '(1 2 3 4)) => (2 4)
/// ```
///
/// # Note
/// Prefer using `.iter().filter(p).cloned().collect()` directly in idiomatic Rust.
///
/// # Example
/// ```
/// use sicp_common::list::filter;
/// let evens = filter(|x| x % 2 == 0, &[1, 2, 3, 4]);
/// assert_eq!(evens, vec![2, 4]);
/// ```
pub fn filter<T: Clone, F>(predicate: F, list: &[T]) -> Vec<T>
where
    F: Fn(&T) -> bool,
{
    list.iter().filter(|x| predicate(x)).cloned().collect()
}

/// Left fold (accumulate) over a list.
///
/// # Scheme equivalent
/// ```scheme
/// (fold-left + 0 '(1 2 3)) => 6
/// ```
///
/// # Note
/// Prefer using `.iter().fold(init, f)` directly in idiomatic Rust.
///
/// # Example
/// ```
/// use sicp_common::list::fold_left;
/// let sum = fold_left(|acc, x| acc + x, 0, &[1, 2, 3]);
/// assert_eq!(sum, 6);
/// ```
pub fn fold_left<T, U, F>(f: F, init: U, list: &[T]) -> U
where
    F: Fn(U, &T) -> U,
{
    list.iter().fold(init, f)
}

/// Right fold over a list.
///
/// # Scheme equivalent
/// ```scheme
/// (fold-right cons '() '(1 2 3)) => (1 2 3)
/// ```
///
/// # Example
/// ```
/// use sicp_common::list::fold_right;
/// let result = fold_right(|x, acc| format!("({x} . {acc})"), "nil".to_string(), &[1, 2, 3]);
/// assert_eq!(result, "(1 . (2 . (3 . nil)))");
/// ```
pub fn fold_right<T, U, F>(f: F, init: U, list: &[T]) -> U
where
    F: Fn(&T, U) -> U,
{
    list.iter().rfold(init, |acc, x| f(x, acc))
}

/// Enumerates a range of integers.
///
/// # Scheme equivalent
/// ```scheme
/// (enumerate-interval 2 7) => (2 3 4 5 6 7)
/// ```
///
/// # Example
/// ```
/// use sicp_common::list::enumerate_interval;
/// assert_eq!(enumerate_interval(2, 5), vec![2, 3, 4, 5]);
/// ```
pub fn enumerate_interval(low: i64, high: i64) -> Vec<i64> {
    (low..=high).collect()
}

/// Flat-maps a function over a list (map then flatten).
///
/// # Scheme equivalent
/// ```scheme
/// (flatmap f seq) = (accumulate append '() (map f seq))
/// ```
///
/// # Example
/// ```
/// use sicp_common::list::flatmap;
/// let result = flatmap(|x| vec![*x, *x * 10], &[1, 2, 3]);
/// assert_eq!(result, vec![1, 10, 2, 20, 3, 30]);
/// ```
pub fn flatmap<T, U, F>(f: F, list: &[T]) -> Vec<U>
where
    F: Fn(&T) -> Vec<U>,
{
    list.iter().flat_map(f).collect()
}

/// Checks if an element is in the list.
///
/// # Scheme equivalent
/// ```scheme
/// (memq 'a '(a b c)) => (a b c)
/// (memq 'b '(a b c)) => (b c)
/// (memq 'd '(a b c)) => #f
/// ```
///
/// This version returns `Option<usize>` (the index) for Rust idioms.
///
/// # Example
/// ```
/// use sicp_common::list::memq;
/// assert_eq!(memq(&'b', &['a', 'b', 'c']), Some(1));
/// assert_eq!(memq(&'d', &['a', 'b', 'c']), None);
/// ```
pub fn memq<T: PartialEq>(item: &T, list: &[T]) -> Option<usize> {
    list.iter().position(|x| x == item)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cons_car_cdr() {
        let pair = cons(1, vec![2, 3]);
        assert_eq!(pair.0, 1);
        assert_eq!(pair.1, vec![2, 3]);

        let list = vec![1, 2, 3];
        assert_eq!(car(&list), Some(&1));
        assert_eq!(cdr(&list), &[2, 3]);
    }

    #[test]
    fn test_empty_list_operations() {
        let empty: &[i32] = &[];
        assert_eq!(car(empty), None);
        assert_eq!(cdr(empty), &[] as &[i32]);
        assert!(is_null(empty));
        assert_eq!(length(empty), 0);
    }

    #[test]
    fn test_list_ref() {
        let list = vec!['a', 'b', 'c', 'd'];
        assert_eq!(list_ref(&list, 0), Some(&'a'));
        assert_eq!(list_ref(&list, 2), Some(&'c'));
        assert_eq!(list_ref(&list, 10), None);
    }

    #[test]
    fn test_append_and_reverse() {
        assert_eq!(append(&[1, 2], &[3, 4]), vec![1, 2, 3, 4]);
        assert_eq!(reverse(&[1, 2, 3]), vec![3, 2, 1]);
    }

    #[test]
    fn test_map_filter_fold() {
        let list = vec![1, 2, 3, 4, 5];

        let squared = map(|x| x * x, &list);
        assert_eq!(squared, vec![1, 4, 9, 16, 25]);

        let evens = filter(|x| x % 2 == 0, &list);
        assert_eq!(evens, vec![2, 4]);

        let sum = fold_left(|acc, x| acc + x, 0, &list);
        assert_eq!(sum, 15);
    }

    #[test]
    fn test_fold_right() {
        // Build a string showing the cons structure
        let result = fold_right(
            |x, acc| format!("({x} . {acc})"),
            "nil".to_string(),
            &[1, 2, 3],
        );
        assert_eq!(result, "(1 . (2 . (3 . nil)))");
    }

    #[test]
    fn test_enumerate_interval() {
        assert_eq!(enumerate_interval(1, 5), vec![1, 2, 3, 4, 5]);
        assert_eq!(enumerate_interval(5, 5), vec![5]);
        assert!(enumerate_interval(5, 4).is_empty());
    }

    #[test]
    fn test_flatmap() {
        let result = flatmap(|x| vec![*x, *x * 2], &[1, 2, 3]);
        assert_eq!(result, vec![1, 2, 2, 4, 3, 6]);
    }

    #[test]
    fn test_memq() {
        let list = vec!["apple", "banana", "cherry"];
        assert_eq!(memq(&"banana", &list), Some(1));
        assert_eq!(memq(&"grape", &list), None);
    }
}
