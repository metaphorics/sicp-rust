//! # 함수형 리스트 연산 (Functional List Operations)
//!
//! Scheme 스타일 리스트 연산을 위한 관용적인 Rust 패턴 (Idiomatic Rust patterns for Scheme-style list operations).
//! 이는 `cons`/`car`/`cdr`를 이터레이터 기반 접근으로 대체한다 (These replace `cons`/`car`/`cdr` with iterator-based approaches).
//!
//! ## 철학 (Philosophy)
//!
//! Scheme에서는 리스트가 cons 셀로 구성되고 `car`, `cdr`, `map`, `filter`, `fold` 등으로 조작된다
//! (In Scheme, lists are built from cons cells and manipulated with `car`, `cdr`, `map`, `filter`, `fold`, etc.).
//! 관용적인 Rust에서는 다음과 같이 대응한다 (In idiomatic Rust):
//!
//! - 시퀀스에는 `Vec<T>` 또는 슬라이스 `&[T]`를 사용한다 (Use `Vec<T>` or slices `&[T]` for sequences)
//! - 변환에는 이터레이터(`.iter()`, `.map()`, `.filter()`, `.fold()`)를 사용한다
//!   (Use iterators (`.iter()`, `.map()`, `.filter()`, `.fold()`) for transformations)
//! - 쌍에는 튜플 `(A, B)`를 사용한다 (Use tuples `(A, B)` for pairs)
//! - 구조 분해에는 패턴 매칭을 사용한다 (Use pattern matching for destructuring)
//!
//! 이 모듈은 Scheme 패턴을 Rust 관용구로 잇는 보조 함수를 제공한다
//! (This module provides helper functions that bridge Scheme patterns to Rust idioms).
//!
//! ## 예시 (Example)
//!
//! ```
//! use sicp_common::list::{cons, car, cdr, list_ref, length};
//!
//! // 스킴 (Scheme): (cons 1 (cons 2 (cons 3 '())))
//! let lst = vec![1, 2, 3];
//!
//! // 스킴 (Scheme): (car lst) => 1
//! assert_eq!(car(&lst), Some(&1));
//!
//! // 스킴 (Scheme): (cdr lst) => (2 3)
//! assert_eq!(cdr(&lst), &[2, 3]);
//!
//! // 스킴 (Scheme): (list-ref lst 1) => 2
//! assert_eq!(list_ref(&lst, 1), Some(&2));
//!
//! // 스킴 (Scheme): (length lst) => 3
//! assert_eq!(length(&lst), 3);
//! ```

/// cons 쌍 (head, tail) (A cons pair (head, tail)).
///
/// Scheme에서는 `(cons a b)`가 쌍을 만든다; Rust에서는 튜플을 사용한다
/// (In Scheme, `(cons a b)` creates a pair. In Rust, we use tuples).
/// 이 타입 별칭은 그 대응을 명확히 한다 (This type alias makes the connection explicit).
pub type Pair<H, T> = (H, T);

/// 쌍(컨스 셀)을 생성한다 (Creates a pair (cons cell)).
///
/// # 스킴 동등식 (Scheme equivalent)
/// ```scheme
/// (cons 1 2) => (1 . 2)
/// ```
///
/// # 예시 (Example)
/// ```
/// use sicp_common::list::cons;
/// let pair = cons(1, 2);
/// assert_eq!(pair, (1, 2));
/// ```
#[inline]
pub fn cons<H, T>(head: H, tail: T) -> Pair<H, T> {
    (head, tail)
}

/// 슬라이스의 첫 원소를 반환한다 (car) (Returns the first element of a slice (car)).
///
/// # 스킴 동등식 (Scheme equivalent)
/// ```scheme
/// (car '(1 2 3)) => 1
/// ```
///
/// # 예시 (Example)
/// ```
/// use sicp_common::list::car;
/// assert_eq!(car(&[1, 2, 3]), Some(&1));
/// assert_eq!(car::<i32>(&[]), None);
/// ```
#[inline]
pub fn car<T>(list: &[T]) -> Option<&T> {
    list.first()
}

/// 슬라이스의 첫 원소를 제외한 나머지를 반환한다 (cdr)
/// (Returns all but the first element of a slice (cdr)).
///
/// # 스킴 동등식 (Scheme equivalent)
/// ```scheme
/// (cdr '(1 2 3)) => (2 3)
/// ```
///
/// # 예시 (Example)
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

/// 주어진 인덱스의 원소를 반환한다 (Returns the element at the given index).
///
/// # 스킴 동등식 (Scheme equivalent)
/// ```scheme
/// (list-ref '(a b c) 1) => b
/// ```
///
/// # 예시 (Example)
/// ```
/// use sicp_common::list::list_ref;
/// assert_eq!(list_ref(&['a', 'b', 'c'], 1), Some(&'b'));
/// assert_eq!(list_ref(&['a', 'b', 'c'], 5), None);
/// ```
#[inline]
pub fn list_ref<T>(list: &[T], index: usize) -> Option<&T> {
    list.get(index)
}

/// 리스트의 길이를 반환한다 (Returns the length of a list).
///
/// # 스킴 동등식 (Scheme equivalent)
/// ```scheme
/// (length '(1 2 3)) => 3
/// ```
#[inline]
pub fn length<T>(list: &[T]) -> usize {
    list.len()
}

/// 리스트가 null(빈 목록)인지 검사한다 (Checks if a list is null (empty)).
///
/// # 스킴 동등식 (Scheme equivalent)
/// ```scheme
/// (null? '()) => #t
/// (null? '(1)) => #f
/// ```
#[inline]
pub fn is_null<T>(list: &[T]) -> bool {
    list.is_empty()
}

/// 두 리스트를 이어 붙인다 (Appends two lists).
///
/// # 스킴 동등식 (Scheme equivalent)
/// ```scheme
/// (append '(1 2) '(3 4)) => (1 2 3 4)
/// ```
///
/// # 예시 (Example)
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

/// 리스트를 뒤집는다 (Reverses a list).
///
/// # 스킴 동등식 (Scheme equivalent)
/// ```scheme
/// (reverse '(1 2 3)) => (3 2 1)
/// ```
///
/// # 예시 (Example)
/// ```
/// use sicp_common::list::reverse;
/// assert_eq!(reverse(&[1, 2, 3]), vec![3, 2, 1]);
/// ```
pub fn reverse<T: Clone>(list: &[T]) -> Vec<T> {
    list.iter().rev().cloned().collect()
}

/// 리스트에 함수를 매핑한다 (Maps a function over a list).
///
/// # 스킴 동등식 (Scheme equivalent)
/// ```scheme
/// (map square '(1 2 3)) => (1 4 9)
/// ```
///
/// # 참고 (Note)
/// 관용적인 Rust에서는 `.iter().map(f).collect()`를 직접 사용하는 것을 선호한다
/// (Prefer using `.iter().map(f).collect()` directly in idiomatic Rust).
///
/// # 예시 (Example)
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

/// 술어로 리스트를 필터링한다 (Filters a list by a predicate).
///
/// # 스킴 동등식 (Scheme equivalent)
/// ```scheme
/// (filter even? '(1 2 3 4)) => (2 4)
/// ```
///
/// # 참고 (Note)
/// 관용적인 Rust에서는 `.iter().filter(p).cloned().collect()`를 직접 사용하는 것을 선호한다
/// (Prefer using `.iter().filter(p).cloned().collect()` directly in idiomatic Rust).
///
/// # 예시 (Example)
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

/// 리스트에 대한 왼쪽 폴드(누적) (Left fold (accumulate) over a list).
///
/// # 스킴 동등식 (Scheme equivalent)
/// ```scheme
/// (fold-left + 0 '(1 2 3)) => 6
/// ```
///
/// # 참고 (Note)
/// 관용적인 Rust에서는 `.iter().fold(init, f)`를 직접 사용하는 것을 선호한다
/// (Prefer using `.iter().fold(init, f)` directly in idiomatic Rust).
///
/// # 예시 (Example)
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

/// 리스트에 대한 오른쪽 폴드 (Right fold over a list).
///
/// # 스킴 동등식 (Scheme equivalent)
/// ```scheme
/// (fold-right cons '() '(1 2 3)) => (1 2 3)
/// ```
///
/// # 예시 (Example)
/// ```
/// use sicp_common::list::fold_right;
/// let result = fold_right(|x, acc| format!("({x} . {acc})"), "닐 (nil)".to_string(), &[1, 2, 3]);
/// assert_eq!(result, "(1 . (2 . (3 . 닐 (nil))))");
/// ```
pub fn fold_right<T, U, F>(f: F, init: U, list: &[T]) -> U
where
    F: Fn(&T, U) -> U,
{
    list.iter().rfold(init, |acc, x| f(x, acc))
}

/// 정수 범위를 열거한다 (Enumerates a range of integers).
///
/// # 스킴 동등식 (Scheme equivalent)
/// ```scheme
/// (enumerate-interval 2 7) => (2 3 4 5 6 7)
/// ```
///
/// # 예시 (Example)
/// ```
/// use sicp_common::list::enumerate_interval;
/// assert_eq!(enumerate_interval(2, 5), vec![2, 3, 4, 5]);
/// ```
pub fn enumerate_interval(low: i64, high: i64) -> Vec<i64> {
    (low..=high).collect()
}

/// 리스트에 함수를 flat-map한다(매핑 후 평탄화) (Flat-maps a function over a list (map then flatten)).
///
/// # 스킴 동등식 (Scheme equivalent)
/// ```scheme
/// (flatmap f seq) = (accumulate append '() (map f seq))
/// ```
///
/// # 예시 (Example)
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

/// 리스트에 원소가 있는지 검사한다 (Checks if an element is in the list).
///
/// # 스킴 동등식 (Scheme equivalent)
/// ```scheme
/// (memq 'a '(a b c)) => (a b c)
/// (memq 'b '(a b c)) => (b c)
/// (memq 'd '(a b c)) => #f
/// ```
///
/// 이 버전은 Rust 관용구에 맞게 `Option<usize>`(인덱스)를 반환한다
/// (This version returns `Option<usize>` (the index) for Rust idioms).
///
/// # 예시 (Example)
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
        // cons 구조를 보여주는 문자열을 구성한다 (Build a string showing the cons structure)
        let result = fold_right(
            |x, acc| format!("({x} . {acc})"),
            "닐 (nil)".to_string(),
            &[1, 2, 3],
        );
        assert_eq!(result, "(1 . (2 . (3 . 닐 (nil))))");
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
        let list = vec!["사과 (apple)", "바나나 (banana)", "체리 (cherry)"];
        assert_eq!(memq(&"바나나 (banana)", &list), Some(1));
        assert_eq!(memq(&"포도 (grape)", &list), None);
    }
}
