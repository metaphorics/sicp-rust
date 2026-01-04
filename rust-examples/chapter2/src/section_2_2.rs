//! Section 2.2: Hierarchical Data and the Closure Property
//!
//! This section demonstrates:
//! - Representing sequences with Vec
//! - Hierarchical structures (trees)
//! - Sequences as conventional interfaces (map, filter, fold)
//! - Picture language with trait-based painters

use std::ops::{Add, Mul, Sub};

// =============================================================================
// 2.2.1 Representing Sequences
// =============================================================================

/// Get the nth element of a list (0-indexed in Rust, but examples use 0-indexed too)
pub fn list_ref<T: Clone>(items: &[T], n: usize) -> T {
    items[n].clone()
}

/// Recursive length implementation
pub fn length<T>(items: &[T]) -> usize {
    if items.is_empty() {
        0
    } else {
        1 + length(&items[1..])
    }
}

/// Iterative length implementation
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

/// Append two vectors
pub fn append<T: Clone>(list1: &[T], list2: &[T]) -> Vec<T> {
    if list1.is_empty() {
        list2.to_vec()
    } else {
        let mut result = vec![list1[0].clone()];
        result.extend(append(&list1[1..], list2));
        result
    }
}

/// Get the last element as a single-element vector
pub fn last_pair<T: Clone>(items: &[T]) -> Vec<T> {
    if items.len() == 1 {
        vec![items[0].clone()]
    } else {
        last_pair(&items[1..])
    }
}

/// Reverse a list
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

/// Scale each element in a list by a factor
pub fn scale_list(items: &[i32], factor: i32) -> Vec<i32> {
    if items.is_empty() {
        vec![]
    } else {
        let mut result = vec![items[0] * factor];
        result.extend(scale_list(&items[1..], factor));
        result
    }
}

/// Map a function over a list
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

/// Scale list using map
pub fn scale_list_with_map(items: &[i32], factor: i32) -> Vec<i32> {
    map(|x| x * factor, items)
}

/// For-each: apply procedure to each element for side effects
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
// 2.2.2 Hierarchical Structures
// =============================================================================

/// Tree data structure using enum
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

/// Count leaves in a tree
pub fn count_leaves<T>(tree: &Tree<T>) -> usize {
    match tree {
        Tree::Leaf(_) => 1,
        Tree::Branch(children) => children.iter().map(count_leaves).sum(),
    }
}

/// Scale a numeric tree by a factor
pub fn scale_tree(tree: &Tree<i32>, factor: i32) -> Tree<i32> {
    match tree {
        Tree::Leaf(value) => Tree::Leaf(value * factor),
        Tree::Branch(children) => {
            Tree::Branch(children.iter().map(|t| scale_tree(t, factor)).collect())
        }
    }
}

/// Scale tree using map
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

/// Deep reverse a tree
pub fn deep_reverse<T: Clone>(tree: &Tree<T>) -> Tree<T> {
    match tree {
        Tree::Leaf(value) => Tree::Leaf(value.clone()),
        Tree::Branch(children) => {
            let reversed: Vec<Tree<T>> = children.iter().map(deep_reverse).collect();
            Tree::Branch(reversed.into_iter().rev().collect())
        }
    }
}

/// Flatten a tree to a list (fringe)
pub fn fringe<T: Clone>(tree: &Tree<T>) -> Vec<T> {
    match tree {
        Tree::Leaf(value) => vec![value.clone()],
        Tree::Branch(children) => children.iter().flat_map(fringe).collect(),
    }
}

/// Tree map abstraction
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

/// Generate all subsets of a set
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
// 2.2.3 Sequences as Conventional Interfaces
// =============================================================================

/// Filter elements that satisfy a predicate
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

/// Accumulate (fold right)
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

/// Enumerate integers in a range
pub fn enumerate_interval(low: i32, high: i32) -> Vec<i32> {
    if low > high {
        vec![]
    } else {
        let mut result = vec![low];
        result.extend(enumerate_interval(low + 1, high));
        result
    }
}

/// Enumerate leaves of a tree
pub fn enumerate_tree<T: Clone>(tree: &Tree<T>) -> Vec<T> {
    fringe(tree)
}

/// Flatmap operation
pub fn flatmap<T, U, F>(proc: F, seq: &[T]) -> Vec<U>
where
    T: Clone,
    U: Clone,
    F: Fn(&T) -> Vec<U>,
{
    accumulate(|x, acc| [proc(x), acc].concat(), vec![], seq)
}

/// Remove an item from a sequence
pub fn remove<T: Clone + PartialEq>(item: &T, sequence: &[T]) -> Vec<T> {
    filter(|x| x != item, sequence)
}

/// Generate all permutations of a set
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

/// Map using accumulate
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

/// Append using accumulate
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

/// Length using accumulate
pub fn length_accumulate<T: Clone>(sequence: &[T]) -> usize {
    accumulate(|_x, acc| acc + 1, 0, sequence)
}

/// Fold left (iterative accumulation)
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
// 2.2.4 Picture Language
// =============================================================================

/// 2D Vector representation
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

/// Frame defined by origin and two edge vectors
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

    /// Map a point in unit square to frame coordinates
    pub fn coord_map(&self, v: Vec2) -> Vec2 {
        self.origin + self.edge1 * v.x + self.edge2 * v.y
    }
}

/// Line segment for drawing
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

/// Painter trait - draws an image in a frame
pub trait Painter {
    fn paint(&self, frame: &Frame);
}

/// Painter implementation using segments
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
            // In real implementation, would draw line from start to end
            println!(
                "Draw line from ({}, {}) to ({}, {})",
                start.x, start.y, end.x, end.y
            );
        }
    }
}

/// Transform a painter
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

/// Flip painter vertically
pub fn flip_vert<P: Painter>(painter: P) -> impl Painter {
    transform_painter(
        painter,
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 0.0),
    )
}

/// Flip painter horizontally
pub fn flip_horiz<P: Painter>(painter: P) -> impl Painter {
    transform_painter(
        painter,
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 1.0),
    )
}

/// Rotate painter 90 degrees counterclockwise
pub fn rotate90<P: Painter>(painter: P) -> impl Painter {
    transform_painter(
        painter,
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 0.0),
    )
}

/// Rotate painter 180 degrees
pub fn rotate180<P: Painter>(painter: P) -> impl Painter {
    transform_painter(
        painter,
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 0.0),
    )
}

/// Rotate painter 270 degrees counterclockwise
pub fn rotate270<P: Painter>(painter: P) -> impl Painter {
    transform_painter(
        painter,
        Vec2::new(0.0, 1.0),
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 1.0),
    )
}

// Note: beside and below require Box<dyn Painter> for dynamic dispatch
// or generic type parameters. For simplicity in examples, we use the
// transformation approach shown in the book.

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
