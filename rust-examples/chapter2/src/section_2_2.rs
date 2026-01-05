//! 2.2절: 계층적 데이터와 클로저 속성
//! (Section 2.2: Hierarchical Data and the Closure Property)
//!
//! 이 섹션에서는 다음을 보여줍니다:
//! - Vec을 사용한 시퀀스 표현 (Sequence representation using Vec)
//! - 계층적 구조 (트리)
//! - 관습적인 인터페이스로서의 시퀀스 (map, filter, fold)
//! - 트레이트 기반 페인터(painter)를 사용한 그림 언어 (Picture language using trait-based painters)

use std::ops::{Add, Mul, Sub};

// =============================================================================
// 2.2.1 시퀀스 표현 (Representing Sequences)
// =============================================================================

/// 리스트의 n번째 원소를 가져온다 (Rust는 0부터 시작하지만, 예제들도 0부터 시작한다)
/// (Gets the nth element of a list (Rust is 0-based and examples follow that)).
pub fn list_ref<T: Clone>(items: &[T], n: usize) -> T {
    items[n].clone()
}

/// 재귀적 길이 구현
pub fn length<T>(items: &[T]) -> usize {
    if items.is_empty() {
        0
    } else {
        1 + length(&items[1..])
    }
}

/// 반복적 길이 구현
pub fn length_iter<T>(items: &[T]) -> usize {
    fn length_helper<T>(items: &[T], count: usize) -> usize {
        if items.is_empty() {
            count
        } else {
            length_helper(&items[1..], count + 1)
        }
    }
    length_helper(items, 0)
}

/// 두 벡터 합치기
pub fn append<T: Clone>(list1: &[T], list2: &[T]) -> Vec<T> {
    if list1.is_empty() {
        list2.to_vec()
    } else {
        let mut result = vec![list1[0].clone()];
        result.extend(append(&list1[1..], list2));
        result
    }
}

/// 마지막 원소를 단일 원소 벡터로 가져옵니다
pub fn last_pair<T: Clone>(items: &[T]) -> Vec<T> {
    if items.len() == 1 {
        vec![items[0].clone()]
    } else {
        last_pair(&items[1..])
    }
}

/// 리스트 뒤집기
pub fn reverse<T: Clone>(items: &[T]) -> Vec<T> {
    fn reverse_iter<T: Clone>(items: &[T], result: Vec<T>) -> Vec<T> {
        if items.is_empty() {
            result
        } else {
            let mut new_result = vec![items[0].clone()];
            new_result.extend(result);
            reverse_iter(&items[1..], new_result)
        }
    }
    reverse_iter(items, vec![])
}

/// 리스트의 각 원소를 일정 비율로 스케일링
pub fn scale_list(items: &[i32], factor: i32) -> Vec<i32> {
    if items.is_empty() {
        vec![]
    } else {
        let mut result = vec![items[0] * factor];
        result.extend(scale_list(&items[1..], factor));
        result
    }
}

/// 리스트에 함수 매핑 (Map)
pub fn map<T, U, F>(proc: F, items: &[T]) -> Vec<U>
where
    T: Clone,
    F: Fn(&T) -> U,
{
    if items.is_empty() {
        vec![]
    } else {
        let mut result = vec![proc(&items[0])];
        result.extend(map(proc, &items[1..]));
        result
    }
}

/// map을 사용한 리스트 스케일링 (List scaling using map)
pub fn scale_list_with_map(items: &[i32], factor: i32) -> Vec<i32> {
    map(|x| x * factor, items)
}

/// For-each: 부수 효과(side effects)를 위해 각 원소에 프로시저 적용
/// (For-each: apply a procedure to each element for side effects)
pub fn for_each<T, F>(proc: F, items: &[T])
where
    F: Fn(&T),
{
    if !items.is_empty() {
        proc(&items[0]);
        for_each(proc, &items[1..]);
    }
}

// =============================================================================
// 2.2.2 계층적 구조 (Hierarchical Structures)
// =============================================================================

/// 열거형(enum)을 사용한 트리 데이터 구조 (Tree data structure using enum)
#[derive(Debug, Clone, PartialEq)]
pub enum Tree<T> {
    Leaf(T),
    Branch(Vec<Tree<T>>),
}

impl<T> Tree<T> {
    pub fn is_leaf(&self) -> bool {
        matches!(self, Tree::Leaf(_))
    }
}

/// 트리의 잎(leaf) 개수 세기 (Counting leaves in a tree)
pub fn count_leaves<T>(tree: &Tree<T>) -> usize {
    match tree {
        Tree::Leaf(_) => 1,
        Tree::Branch(children) => children.iter().map(count_leaves).sum(),
    }
}

/// 숫자 트리를 일정 비율로 스케일링
pub fn scale_tree(tree: &Tree<i32>, factor: i32) -> Tree<i32> {
    match tree {
        Tree::Leaf(value) => Tree::Leaf(value * factor),
        Tree::Branch(children) => {
            Tree::Branch(children.iter().map(|t| scale_tree(t, factor)).collect())
        }
    }
}

/// map을 사용한 트리 스케일링 (Tree scaling using map)
pub fn scale_tree_with_map(tree: &Tree<i32>, factor: i32) -> Tree<i32> {
    match tree {
        Tree::Leaf(value) => Tree::Leaf(value * factor),
        Tree::Branch(children) => Tree::Branch(
            children
                .iter()
                .map(|subtree| scale_tree_with_map(subtree, factor))
                .collect(),
        ),
    }
}

/// 트리를 깊게 뒤집기 (Deep reverse)
pub fn deep_reverse<T: Clone>(tree: &Tree<T>) -> Tree<T> {
    match tree {
        Tree::Leaf(value) => Tree::Leaf(value.clone()),
        Tree::Branch(children) => {
            let reversed: Vec<Tree<T>> = children.iter().map(deep_reverse).collect();
            Tree::Branch(reversed.into_iter().rev().collect())
        }
    }
}

/// 트리를 리스트로 평탄화 (Fringe)
pub fn fringe<T: Clone>(tree: &Tree<T>) -> Vec<T> {
    match tree {
        Tree::Leaf(value) => vec![value.clone()],
        Tree::Branch(children) => children.iter().flat_map(fringe).collect(),
    }
}

/// 트리 맵 추상화
pub fn tree_map<T, U, F>(f: F, tree: &Tree<T>) -> Tree<U>
where
    F: Fn(&T) -> U + Clone,
{
    match tree {
        Tree::Leaf(value) => Tree::Leaf(f(value)),
        Tree::Branch(children) => {
            Tree::Branch(children.iter().map(|t| tree_map(f.clone(), t)).collect())
        }
    }
}

/// 집합의 모든 부분집합 생성
pub fn subsets<T: Clone>(s: &[T]) -> Vec<Vec<T>> {
    if s.is_empty() {
        vec![vec![]]
    } else {
        let rest = subsets(&s[1..]);
        let with_first: Vec<Vec<T>> = rest
            .iter()
            .map(|subset| {
                let mut new_subset = vec![s[0].clone()];
                new_subset.extend(subset.clone());
                new_subset
            })
            .collect();
        [rest, with_first].concat()
    }
}

// =============================================================================
// 2.2.3 관습적인 인터페이스로서의 시퀀스 (Sequences as Conventional Interfaces)
// =============================================================================

/// 술어(predicate)를 만족하는 원소 필터링 (Filter elements satisfying a predicate)
pub fn filter<T, P>(predicate: P, sequence: &[T]) -> Vec<T>
where
    T: Clone,
    P: Fn(&T) -> bool,
{
    if sequence.is_empty() {
        vec![]
    } else if predicate(&sequence[0]) {
        let mut result = vec![sequence[0].clone()];
        result.extend(filter(predicate, &sequence[1..]));
        result
    } else {
        filter(predicate, &sequence[1..])
    }
}

/// 누산 (Accumulate, fold right)
pub fn accumulate<T, U, F>(op: F, initial: U, sequence: &[T]) -> U
where
    T: Clone,
    U: Clone,
    F: Fn(&T, U) -> U + Clone,
{
    if sequence.is_empty() {
        initial
    } else {
        op(
            &sequence[0],
            accumulate(op.clone(), initial, &sequence[1..]),
        )
    }
}

/// 범위 내 정수 열거
pub fn enumerate_interval(low: i32, high: i32) -> Vec<i32> {
    if low > high {
        vec![]
    } else {
        let mut result = vec![low];
        result.extend(enumerate_interval(low + 1, high));
        result
    }
}

/// 트리의 잎 열거
pub fn enumerate_tree<T: Clone>(tree: &Tree<T>) -> Vec<T> {
    fringe(tree)
}

/// 플랫맵 (Flatmap) 연산
pub fn flatmap<T, U, F>(proc: F, seq: &[T]) -> Vec<U>
where
    T: Clone,
    U: Clone,
    F: Fn(&T) -> Vec<U>,
{
    accumulate(|x, acc| [proc(x), acc].concat(), vec![], seq)
}

/// 시퀀스에서 항목 제거
pub fn remove<T: Clone + PartialEq>(item: &T, sequence: &[T]) -> Vec<T> {
    filter(|x| x != item, sequence)
}

/// 집합의 모든 순열 생성
pub fn permutations<T: Clone + PartialEq>(s: &[T]) -> Vec<Vec<T>> {
    if s.is_empty() {
        vec![vec![]]
    } else {
        flatmap(
            |x| {
                map(
                    |p| {
                        let mut result = vec![x.clone()];
                        result.extend(p.clone());
                        result
                    },
                    &permutations(&remove(x, s)),
                )
            },
            s,
        )
    }
}

/// accumulate를 사용한 Map
pub fn map_accumulate<T, U, F>(p: F, sequence: &[T]) -> Vec<U>
where
    T: Clone,
    U: Clone,
    F: Fn(&T) -> U,
{
    accumulate(
        |x, acc| {
            let mut result = vec![p(x)];
            result.extend(acc);
            result
        },
        vec![],
        sequence,
    )
}

/// accumulate를 사용한 Append
pub fn append_accumulate<T: Clone>(seq1: &[T], seq2: &[T]) -> Vec<T> {
    accumulate(
        |x, acc| {
            let mut result = vec![x.clone()];
            result.extend(acc);
            result
        },
        seq2.to_vec(),
        seq1,
    )
}

/// accumulate를 사용한 Length
pub fn length_accumulate<T: Clone>(sequence: &[T]) -> usize {
    accumulate(|_x, acc| acc + 1, 0, sequence)
}

/// Fold left (반복적 누산)
pub fn fold_left<T, U, F>(op: F, initial: U, sequence: &[T]) -> U
where
    T: Clone,
    U: Clone,
    F: Fn(U, &T) -> U + Clone,
{
    fn iter<T, U, F>(op: &F, result: U, rest: &[T]) -> U
    where
        T: Clone,
        U: Clone,
        F: Fn(U, &T) -> U,
    {
        if rest.is_empty() {
            result
        } else {
            iter(op, op(result, &rest[0]), &rest[1..])
        }
    }
    iter(&op, initial, sequence)
}

// =============================================================================
// 2.2.4 그림 언어 (Picture Language)
// =============================================================================

/// 2D 벡터 표현 (2D vector representation)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Vec2 { x, y }
    }

    pub fn xcor(&self) -> f64 {
        self.x
    }

    pub fn ycor(&self) -> f64 {
        self.y
    }
}

impl Add for Vec2 {
    type Output = Vec2;

    fn add(self, other: Vec2) -> Vec2 {
        Vec2::new(self.x + other.x, self.y + other.y)
    }
}

impl Sub for Vec2 {
    type Output = Vec2;

    fn sub(self, other: Vec2) -> Vec2 {
        Vec2::new(self.x - other.x, self.y - other.y)
    }
}

impl Mul<f64> for Vec2 {
    type Output = Vec2;

    fn mul(self, scalar: f64) -> Vec2 {
        Vec2::new(self.x * scalar, self.y * scalar)
    }
}

/// 원점과 두 개의 가장자리 벡터로 정의된 프레임
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    pub origin: Vec2,
    pub edge1: Vec2,
    pub edge2: Vec2,
}

impl Frame {
    pub fn new(origin: Vec2, edge1: Vec2, edge2: Vec2) -> Self {
        Frame {
            origin,
            edge1,
            edge2,
        }
    }

    /// 단위 정사각형의 점을 프레임 좌표로 매핑
    pub fn coord_map(&self, v: Vec2) -> Vec2 {
        self.origin + self.edge1 * v.x + self.edge2 * v.y
    }
}

/// 그리기를 위한 선분
#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub start: Vec2,
    pub end: Vec2,
}

impl Segment {
    pub fn new(start: Vec2, end: Vec2) -> Self {
        Segment { start, end }
    }
}

/// 페인터(Painter) 트레이트 - 프레임 안에 이미지를 그린다
/// (Painter trait - draws an image inside a frame)
pub trait Painter {
    fn paint(&self, frame: &Frame);
}

/// 선분들을 사용하는 페인터 구현
pub struct SegmentPainter {
    pub segments: Vec<Segment>,
}

impl SegmentPainter {
    pub fn new(segments: Vec<Segment>) -> Self {
        SegmentPainter { segments }
    }
}

impl Painter for SegmentPainter {
    fn paint(&self, frame: &Frame) {
        for segment in &self.segments {
            let start = frame.coord_map(segment.start);
            let end = frame.coord_map(segment.end);
            // 실제 구현에서는 start에서 end까지 선을 그린다
            // (In a real implementation, draw a line from start to end)
            println!(
                "Draw line from ({}, {}) to ({}, {})",
                start.x, start.y, end.x, end.y
            );
        }
    }
}

/// 페인터 변환
pub fn transform_painter<P: Painter>(
    painter: P,
    origin: Vec2,
    corner1: Vec2,
    corner2: Vec2,
) -> impl Painter {
    struct TransformedPainter<P> {
        painter: P,
        origin: Vec2,
        corner1: Vec2,
        corner2: Vec2,
    }

    impl<P: Painter> Painter for TransformedPainter<P> {
        fn paint(&self, frame: &Frame) {
            let m = |v: Vec2| frame.coord_map(v);
            let new_origin = m(self.origin);
            let new_frame = Frame::new(
                new_origin,
                m(self.corner1) - new_origin,
                m(self.corner2) - new_origin,
            );
            self.painter.paint(&new_frame);
        }
    }

    TransformedPainter {
        painter,
        origin,
        corner1,
        corner2,
    }
}

/// 페인터를 수직으로 뒤집기
pub fn flip_vert<P: Painter>(painter: P) -> impl Painter {
    transform_painter(
        painter,
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 0.0),
    )
}

/// 페인터를 수평으로 뒤집기
pub fn flip_horiz<P: Painter>(painter: P) -> impl Painter {
    transform_painter(
        painter,
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 1.0),
    )
}

/// 페인터를 시계 반대 방향으로 90도 회전
pub fn rotate90<P: Painter>(painter: P) -> impl Painter {
    transform_painter(
        painter,
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 0.0),
    )
}

/// 페인터를 180도 회전
pub fn rotate180<P: Painter>(painter: P) -> impl Painter {
    transform_painter(
        painter,
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 0.0),
    )
}

/// 페인터를 시계 반대 방향으로 270도 회전
pub fn rotate270<P: Painter>(painter: P) -> impl Painter {
    transform_painter(
        painter,
        Vec2::new(0.0, 1.0),
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 1.0),
    )
}

// 참고: beside와 below는 동적 디스패치(Box<dyn Painter>)나
// 제네릭 타입 파라미터가 필요합니다. 예제의 단순화를 위해
// 책에 나온 변환 접근 방식을 사용합니다.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_ref() {
        let squares = vec![1, 4, 9, 16, 25];
        assert_eq!(list_ref(&squares, 3), 16);
    }

    #[test]
    fn test_length() {
        let odds = vec![1, 3, 5, 7];
        assert_eq!(length(&odds), 4);
        assert_eq!(length_iter(&odds), 4);
    }

    #[test]
    fn test_append() {
        let squares = vec![1, 4, 9, 16, 25];
        let odds = vec![1, 3, 5, 7];
        assert_eq!(append(&squares, &odds), vec![1, 4, 9, 16, 25, 1, 3, 5, 7]);
    }

    #[test]
    fn test_last_pair() {
        let list = vec![23, 72, 149, 34];
        assert_eq!(last_pair(&list), vec![34]);
    }

    #[test]
    fn test_reverse() {
        let list = vec![1, 4, 9, 16, 25];
        assert_eq!(reverse(&list), vec![25, 16, 9, 4, 1]);
    }

    #[test]
    fn test_map() {
        let list = vec![-10, 2, -11, 17];
        let result = map(|x: &i32| x.abs(), &list);
        assert_eq!(result, vec![10, 2, 11, 17]);
    }

    #[test]
    fn test_scale_list() {
        let list = vec![1, 2, 3, 4, 5];
        assert_eq!(scale_list(&list, 10), vec![10, 20, 30, 40, 50]);
        assert_eq!(scale_list_with_map(&list, 10), vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_count_leaves() {
        let x = Tree::Branch(vec![
            Tree::Branch(vec![Tree::Leaf(1), Tree::Leaf(2)]),
            Tree::Branch(vec![Tree::Leaf(3), Tree::Leaf(4)]),
        ]);
        assert_eq!(count_leaves(&x), 4);

        let xx = Tree::Branch(vec![x.clone(), x.clone()]);
        assert_eq!(count_leaves(&xx), 8);
    }

    #[test]
    fn test_scale_tree() {
        let tree = Tree::Branch(vec![
            Tree::Leaf(1),
            Tree::Branch(vec![
                Tree::Leaf(2),
                Tree::Branch(vec![Tree::Leaf(3), Tree::Leaf(4)]),
                Tree::Leaf(5),
            ]),
            Tree::Branch(vec![Tree::Leaf(6), Tree::Leaf(7)]),
        ]);

        let expected = Tree::Branch(vec![
            Tree::Leaf(10),
            Tree::Branch(vec![
                Tree::Leaf(20),
                Tree::Branch(vec![Tree::Leaf(30), Tree::Leaf(40)]),
                Tree::Leaf(50),
            ]),
            Tree::Branch(vec![Tree::Leaf(60), Tree::Leaf(70)]),
        ]);

        assert_eq!(scale_tree(&tree, 10), expected);
        assert_eq!(scale_tree_with_map(&tree, 10), expected);
    }

    #[test]
    fn test_subsets() {
        let s = vec![1, 2, 3];
        let result = subsets(&s);
        assert_eq!(result.len(), 8);
        assert!(result.contains(&vec![]));
        assert!(result.contains(&vec![1, 2, 3]));
    }

    #[test]
    fn test_filter() {
        let list = vec![1, 2, 3, 4, 5];
        let result = filter(|x: &i32| x % 2 == 1, &list);
        assert_eq!(result, vec![1, 3, 5]);
    }

    #[test]
    fn test_accumulate() {
        let list = vec![1, 2, 3, 4, 5];
        assert_eq!(accumulate(|x, acc| x + acc, 0, &list), 15);
        assert_eq!(accumulate(|x, acc| x * acc, 1, &list), 120);
    }

    #[test]
    fn test_enumerate_interval() {
        assert_eq!(enumerate_interval(2, 7), vec![2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_flatmap() {
        let result = flatmap(|i| vec![*i, i * 2], &vec![1, 2, 3]);
        assert_eq!(result, vec![1, 2, 2, 4, 3, 6]);
    }

    #[test]
    fn test_permutations() {
        let s = vec![1, 2, 3];
        let result = permutations(&s);
        assert_eq!(result.len(), 6);
        assert!(result.contains(&vec![1, 2, 3]));
        assert!(result.contains(&vec![3, 2, 1]));
    }

    #[test]
    fn test_fold_left() {
        let list = vec![1, 2, 3];
        assert_eq!(fold_left(|acc, x| acc - x, 0, &list), -6); // 0 - 1 - 2 - 3
    }

    #[test]
    fn test_vec2_operations() {
        let v1 = Vec2::new(1.0, 2.0);
        let v2 = Vec2::new(3.0, 4.0);

        let sum = v1 + v2;
        assert_eq!(sum.x, 4.0);
        assert_eq!(sum.y, 6.0);

        let diff = v2 - v1;
        assert_eq!(diff.x, 2.0);
        assert_eq!(diff.y, 2.0);

        let scaled = v1 * 2.0;
        assert_eq!(scaled.x, 2.0);
        assert_eq!(scaled.y, 4.0);
    }

    #[test]
    fn test_frame_coord_map() {
        let frame = Frame::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
        );

        let origin = frame.coord_map(Vec2::new(0.0, 0.0));
        assert_eq!(origin.x, 0.0);
        assert_eq!(origin.y, 0.0);

        let center = frame.coord_map(Vec2::new(0.5, 0.5));
        assert_eq!(center.x, 0.5);
        assert_eq!(center.y, 0.5);
    }
}
