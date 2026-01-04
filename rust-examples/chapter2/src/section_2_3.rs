//! Section 2.3: Symbolic Data
//!
//! This section introduces the ability to work with arbitrary symbols as data.
//! We explore:
//! - Quotation and symbol manipulation
//! - Symbolic differentiation of algebraic expressions
//! - Multiple representations of sets (unordered, ordered, binary trees)
//! - Huffman encoding trees for variable-length codes

use std::fmt;

// ============================================================================
// 2.3.1: Quotation
// ============================================================================

/// Searches for an item in a list, returning the sublist starting with the first occurrence.
/// Similar to Scheme's `memq` procedure.
///
/// # Example
/// ```
/// # use sicp_chapter2::section_2_3::memq;
/// let list = vec!["x", "apple", "sauce", "y", "apple", "pear"];
/// assert_eq!(memq("apple", &list), Some(&["apple", "sauce", "y", "apple", "pear"][..]));
/// assert_eq!(memq("banana", &list), None);
/// ```
pub fn memq<T: PartialEq>(item: T, list: &[T]) -> Option<&[T]> {
    for (i, elem) in list.iter().enumerate() {
        if *elem == item {
            return Some(&list[i..]);
        }
    }
    None
}

/// Recursive equality test for nested structures.
/// In Rust, we can use PartialEq directly, but this demonstrates the concept.
pub fn equal<T: PartialEq>(a: &[T], b: &[T]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

// ============================================================================
// 2.3.2: Symbolic Differentiation
// ============================================================================

/// Algebraic expression representation
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A constant number
    Const(f64),
    /// A variable (e.g., 'x', 'y')
    Var(&'static str),
    /// Sum of two expressions
    Sum(Box<Expr>, Box<Expr>),
    /// Product of two expressions
    Product(Box<Expr>, Box<Expr>),
    /// Exponentiation: base^exponent
    Exponent(Box<Expr>, i32),
}

impl Expr {
    /// Checks if expression is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self, Expr::Var(_))
    }

    /// Checks if two expressions are the same variable
    pub fn same_variable(&self, other: &Expr) -> bool {
        match (self, other) {
            (Expr::Var(v1), Expr::Var(v2)) => v1 == v2,
            _ => false,
        }
    }

    /// Checks if expression equals a specific number
    pub fn is_number(&self, n: f64) -> bool {
        match self {
            Expr::Const(val) => (*val - n).abs() < f64::EPSILON,
            _ => false,
        }
    }

    /// Constructor for sum with simplification
    pub fn make_sum(a1: Expr, a2: Expr) -> Expr {
        match (&a1, &a2) {
            // 0 + a2 = a2
            (Expr::Const(n), _) if (*n - 0.0).abs() < f64::EPSILON => a2,
            // a1 + 0 = a1
            (_, Expr::Const(n)) if (*n - 0.0).abs() < f64::EPSILON => a1,
            // Fold constants
            (Expr::Const(n1), Expr::Const(n2)) => Expr::Const(n1 + n2),
            _ => Expr::Sum(Box::new(a1), Box::new(a2)),
        }
    }

    /// Constructor for product with simplification
    pub fn make_product(m1: Expr, m2: Expr) -> Expr {
        match (&m1, &m2) {
            // 0 * anything = 0
            (Expr::Const(n), _) | (_, Expr::Const(n)) if (*n - 0.0).abs() < f64::EPSILON => {
                Expr::Const(0.0)
            }
            // 1 * m2 = m2
            (Expr::Const(n), _) if (*n - 1.0).abs() < f64::EPSILON => m2,
            // m1 * 1 = m1
            (_, Expr::Const(n)) if (*n - 1.0).abs() < f64::EPSILON => m1,
            // Fold constants
            (Expr::Const(n1), Expr::Const(n2)) => Expr::Const(n1 * n2),
            _ => Expr::Product(Box::new(m1), Box::new(m2)),
        }
    }

    /// Constructor for exponentiation with simplification
    pub fn make_exponent(base: Expr, exp: i32) -> Expr {
        match exp {
            0 => Expr::Const(1.0),
            1 => base,
            _ => Expr::Exponent(Box::new(base), exp),
        }
    }

    /// Extract addend (first term) of a sum
    pub fn addend(&self) -> Option<&Expr> {
        match self {
            Expr::Sum(a, _) => Some(a),
            _ => None,
        }
    }

    /// Extract augend (second term) of a sum
    pub fn augend(&self) -> Option<&Expr> {
        match self {
            Expr::Sum(_, b) => Some(b),
            _ => None,
        }
    }

    /// Extract multiplier (first factor) of a product
    pub fn multiplier(&self) -> Option<&Expr> {
        match self {
            Expr::Product(m, _) => Some(m),
            _ => None,
        }
    }

    /// Extract multiplicand (second factor) of a product
    pub fn multiplicand(&self) -> Option<&Expr> {
        match self {
            Expr::Product(_, m) => Some(m),
            _ => None,
        }
    }

    /// Extract base of an exponent
    pub fn base(&self) -> Option<&Expr> {
        match self {
            Expr::Exponent(b, _) => Some(b),
            _ => None,
        }
    }

    /// Extract exponent
    pub fn exponent(&self) -> Option<i32> {
        match self {
            Expr::Exponent(_, e) => Some(*e),
            _ => None,
        }
    }
}

/// Symbolic differentiation
///
/// Computes the derivative of an algebraic expression with respect to a variable.
///
/// # Rules
/// - dc/dx = 0 (constant)
/// - dx/dx = 1
/// - d(u+v)/dx = du/dx + dv/dx
/// - d(uv)/dx = u(dv/dx) + v(du/dx)
/// - d(u^n)/dx = n*u^(n-1) * du/dx
pub fn deriv(exp: &Expr, var: &str) -> Expr {
    match exp {
        Expr::Const(_) => Expr::Const(0.0),
        Expr::Var(v) => {
            if *v == var {
                Expr::Const(1.0)
            } else {
                Expr::Const(0.0)
            }
        }
        Expr::Sum(a, b) => Expr::make_sum(deriv(a, var), deriv(b, var)),
        Expr::Product(m1, m2) => {
            // d(uv)/dx = u(dv/dx) + v(du/dx)
            Expr::make_sum(
                Expr::make_product((**m1).clone(), deriv(m2, var)),
                Expr::make_product(deriv(m1, var), (**m2).clone()),
            )
        }
        Expr::Exponent(base, n) => {
            // d(u^n)/dx = n * u^(n-1) * du/dx
            Expr::make_product(
                Expr::make_product(
                    Expr::Const(*n as f64),
                    Expr::make_exponent((**base).clone(), n - 1),
                ),
                deriv(base, var),
            )
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Const(n) => write!(f, "{}", n),
            Expr::Var(v) => write!(f, "{}", v),
            Expr::Sum(a, b) => write!(f, "(+ {} {})", a, b),
            Expr::Product(m1, m2) => write!(f, "(* {} {})", m1, m2),
            Expr::Exponent(base, exp) => write!(f, "(** {} {})", base, exp),
        }
    }
}

// ============================================================================
// 2.3.3: Representing Sets
// ============================================================================

// --- Sets as Unordered Lists ---

/// Check if element is in unordered set (O(n))
pub fn element_of_set_unordered<T: PartialEq>(x: &T, set: &[T]) -> bool {
    set.iter().any(|elem| elem == x)
}

/// Add element to unordered set (O(n))
pub fn adjoin_set_unordered<T: PartialEq + Clone>(x: T, set: &[T]) -> Vec<T> {
    if element_of_set_unordered(&x, set) {
        set.to_vec()
    } else {
        let mut result = vec![x];
        result.extend_from_slice(set);
        result
    }
}

/// Intersection of two unordered sets (O(n²))
pub fn intersection_set_unordered<T: PartialEq + Clone>(set1: &[T], set2: &[T]) -> Vec<T> {
    set1.iter()
        .filter(|&x| element_of_set_unordered(x, set2))
        .cloned()
        .collect()
}

/// Union of two unordered sets (O(n²))
pub fn union_set_unordered<T: PartialEq + Clone>(set1: &[T], set2: &[T]) -> Vec<T> {
    let mut result = set1.to_vec();
    for x in set2 {
        if !element_of_set_unordered(x, set1) {
            result.push(x.clone());
        }
    }
    result
}

// --- Sets as Ordered Lists ---

/// Check if element is in ordered set (O(n) average, but can terminate early)
pub fn element_of_set_ordered<T: Ord>(x: &T, set: &[T]) -> bool {
    for elem in set {
        if elem == x {
            return true;
        }
        if elem > x {
            return false;
        }
    }
    false
}

/// Intersection of two ordered sets (O(n))
pub fn intersection_set_ordered<T: Ord + Clone>(set1: &[T], set2: &[T]) -> Vec<T> {
    if set1.is_empty() || set2.is_empty() {
        return vec![];
    }

    let mut result = Vec::new();
    let mut i1 = 0;
    let mut i2 = 0;

    while i1 < set1.len() && i2 < set2.len() {
        let x1 = &set1[i1];
        let x2 = &set2[i2];

        match x1.cmp(x2) {
            std::cmp::Ordering::Equal => {
                result.push(x1.clone());
                i1 += 1;
                i2 += 1;
            }
            std::cmp::Ordering::Less => i1 += 1,
            std::cmp::Ordering::Greater => i2 += 1,
        }
    }

    result
}

/// Add element to ordered set (O(n))
pub fn adjoin_set_ordered<T: Ord + Clone>(x: T, set: &[T]) -> Vec<T> {
    let mut result = Vec::new();
    let mut inserted = false;

    for elem in set {
        if !inserted && x < *elem {
            result.push(x.clone());
            inserted = true;
        }
        if elem == &x {
            // Already in set
            return set.to_vec();
        }
        result.push(elem.clone());
    }

    if !inserted {
        result.push(x);
    }

    result
}

/// Union of two ordered sets (O(n))
pub fn union_set_ordered<T: Ord + Clone>(set1: &[T], set2: &[T]) -> Vec<T> {
    if set1.is_empty() {
        return set2.to_vec();
    }
    if set2.is_empty() {
        return set1.to_vec();
    }

    let mut result = Vec::new();
    let mut i1 = 0;
    let mut i2 = 0;

    while i1 < set1.len() && i2 < set2.len() {
        let x1 = &set1[i1];
        let x2 = &set2[i2];

        match x1.cmp(x2) {
            std::cmp::Ordering::Equal => {
                result.push(x1.clone());
                i1 += 1;
                i2 += 1;
            }
            std::cmp::Ordering::Less => {
                result.push(x1.clone());
                i1 += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(x2.clone());
                i2 += 1;
            }
        }
    }

    // Add remaining elements
    while i1 < set1.len() {
        result.push(set1[i1].clone());
        i1 += 1;
    }
    while i2 < set2.len() {
        result.push(set2[i2].clone());
        i2 += 1;
    }

    result
}

// --- Sets as Binary Trees ---

/// Binary tree representation of a set
#[derive(Debug, Clone, PartialEq)]
pub enum Tree<T> {
    Empty,
    Node {
        entry: T,
        left: Box<Tree<T>>,
        right: Box<Tree<T>>,
    },
}

impl<T: Ord> Tree<T> {
    /// Create an empty tree
    pub fn empty() -> Self {
        Tree::Empty
    }

    /// Create a tree node
    pub fn make_tree(entry: T, left: Tree<T>, right: Tree<T>) -> Self {
        Tree::Node {
            entry,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Check if element is in tree (O(log n) for balanced trees)
    pub fn element_of_set(&self, x: &T) -> bool {
        match self {
            Tree::Empty => false,
            Tree::Node { entry, left, right } => match x.cmp(entry) {
                std::cmp::Ordering::Equal => true,
                std::cmp::Ordering::Less => left.element_of_set(x),
                std::cmp::Ordering::Greater => right.element_of_set(x),
            },
        }
    }

    /// Add element to tree (O(log n) for balanced trees)
    pub fn adjoin_set(self, x: T) -> Self {
        match self {
            Tree::Empty => Tree::make_tree(x, Tree::Empty, Tree::Empty),
            Tree::Node { entry, left, right } => match x.cmp(&entry) {
                std::cmp::Ordering::Equal => Tree::Node { entry, left, right },
                std::cmp::Ordering::Less => Tree::make_tree(entry, (*left).adjoin_set(x), *right),
                std::cmp::Ordering::Greater => {
                    Tree::make_tree(entry, *left, (*right).adjoin_set(x))
                }
            },
        }
    }

    /// Convert tree to sorted list
    pub fn tree_to_list(&self) -> Vec<T>
    where
        T: Clone,
    {
        match self {
            Tree::Empty => vec![],
            Tree::Node { entry, left, right } => {
                let mut result = left.tree_to_list();
                result.push(entry.clone());
                result.extend(right.tree_to_list());
                result
            }
        }
    }

    /// Convert sorted list to balanced binary tree
    pub fn list_to_tree(elements: &[T]) -> Self
    where
        T: Clone,
    {
        fn partial_tree<T: Clone + Ord>(elts: &[T], n: usize) -> (Tree<T>, &[T]) {
            if n == 0 {
                return (Tree::Empty, elts);
            }

            let left_size = (n - 1) / 2;
            let (left_tree, non_left_elts) = partial_tree(elts, left_size);

            let this_entry = non_left_elts[0].clone();
            let right_size = n - left_size - 1;
            let (right_tree, remaining_elts) = partial_tree(&non_left_elts[1..], right_size);

            (
                Tree::make_tree(this_entry, left_tree, right_tree),
                remaining_elts,
            )
        }

        partial_tree(elements, elements.len()).0
    }
}

// Database lookup using binary tree
pub fn lookup<'a, K: Ord, V>(key: &K, records: &'a Tree<(K, V)>) -> Option<&'a V> {
    match records {
        Tree::Empty => None,
        Tree::Node { entry, left, right } => match key.cmp(&entry.0) {
            std::cmp::Ordering::Equal => Some(&entry.1),
            std::cmp::Ordering::Less => lookup(key, left),
            std::cmp::Ordering::Greater => lookup(key, right),
        },
    }
}

// ============================================================================
// 2.3.4: Huffman Encoding Trees
// ============================================================================

/// Huffman tree for variable-length encoding
#[derive(Debug, Clone, PartialEq)]
pub enum HuffmanTree {
    Leaf {
        symbol: char,
        weight: u32,
    },
    Branch {
        left: Box<HuffmanTree>,
        right: Box<HuffmanTree>,
        symbols: Vec<char>,
        weight: u32,
    },
}

impl HuffmanTree {
    /// Create a leaf node
    pub fn make_leaf(symbol: char, weight: u32) -> Self {
        HuffmanTree::Leaf { symbol, weight }
    }

    /// Check if tree is a leaf
    pub fn is_leaf(&self) -> bool {
        matches!(self, HuffmanTree::Leaf { .. })
    }

    /// Get symbol from leaf
    pub fn symbol_leaf(&self) -> Option<char> {
        match self {
            HuffmanTree::Leaf { symbol, .. } => Some(*symbol),
            _ => None,
        }
    }

    /// Get weight
    pub fn weight(&self) -> u32 {
        match self {
            HuffmanTree::Leaf { weight, .. } => *weight,
            HuffmanTree::Branch { weight, .. } => *weight,
        }
    }

    /// Get symbols
    pub fn symbols(&self) -> Vec<char> {
        match self {
            HuffmanTree::Leaf { symbol, .. } => vec![*symbol],
            HuffmanTree::Branch { symbols, .. } => symbols.clone(),
        }
    }

    /// Create a branch node
    pub fn make_code_tree(left: HuffmanTree, right: HuffmanTree) -> Self {
        let mut symbols = left.symbols();
        symbols.extend(right.symbols());
        let weight = left.weight() + right.weight();

        HuffmanTree::Branch {
            left: Box::new(left),
            right: Box::new(right),
            symbols,
            weight,
        }
    }

    /// Get left branch
    pub fn left_branch(&self) -> Option<&HuffmanTree> {
        match self {
            HuffmanTree::Branch { left, .. } => Some(left),
            _ => None,
        }
    }

    /// Get right branch
    pub fn right_branch(&self) -> Option<&HuffmanTree> {
        match self {
            HuffmanTree::Branch { right, .. } => Some(right),
            _ => None,
        }
    }
}

/// Choose branch based on bit (0 = left, 1 = right)
fn choose_branch(bit: u8, branch: &HuffmanTree) -> Result<&HuffmanTree, &'static str> {
    match bit {
        0 => branch.left_branch().ok_or("Invalid branch"),
        1 => branch.right_branch().ok_or("Invalid branch"),
        _ => Err("Invalid bit value"),
    }
}

/// Decode a bit sequence using a Huffman tree
pub fn decode(bits: &[u8], tree: &HuffmanTree) -> Result<Vec<char>, &'static str> {
    fn decode_1(
        bits: &[u8],
        current_branch: &HuffmanTree,
        tree: &HuffmanTree,
    ) -> Result<Vec<char>, &'static str> {
        if bits.is_empty() {
            return Ok(vec![]);
        }

        let next_branch = choose_branch(bits[0], current_branch)?;

        if next_branch.is_leaf() {
            let symbol = next_branch.symbol_leaf().ok_or("Invalid leaf")?;
            let mut result = vec![symbol];
            result.extend(decode_1(&bits[1..], tree, tree)?);
            Ok(result)
        } else {
            decode_1(&bits[1..], next_branch, tree)
        }
    }

    decode_1(bits, tree, tree)
}

/// Encode a symbol using a Huffman tree
fn encode_symbol(symbol: char, tree: &HuffmanTree) -> Result<Vec<u8>, &'static str> {
    match tree {
        HuffmanTree::Leaf { .. } => Ok(vec![]),
        HuffmanTree::Branch { left, right, .. } => {
            if left.symbols().contains(&symbol) {
                let mut result = vec![0];
                result.extend(encode_symbol(symbol, left)?);
                Ok(result)
            } else if right.symbols().contains(&symbol) {
                let mut result = vec![1];
                result.extend(encode_symbol(symbol, right)?);
                Ok(result)
            } else {
                Err("Symbol not in tree")
            }
        }
    }
}

/// Encode a message using a Huffman tree
pub fn encode(message: &[char], tree: &HuffmanTree) -> Result<Vec<u8>, &'static str> {
    let mut result = Vec::new();
    for &ch in message {
        result.extend(encode_symbol(ch, tree)?);
    }
    Ok(result)
}

/// Insert element into ordered set (by weight)
fn adjoin_set_huffman(x: HuffmanTree, set: &[HuffmanTree]) -> Vec<HuffmanTree> {
    if set.is_empty() {
        return vec![x];
    }

    let mut result = Vec::new();
    let mut inserted = false;

    for tree in set {
        if !inserted && x.weight() < tree.weight() {
            result.push(x.clone());
            inserted = true;
        }
        result.push(tree.clone());
    }

    if !inserted {
        result.push(x);
    }

    result
}

/// Create initial leaf set from symbol-frequency pairs
pub fn make_leaf_set(pairs: &[(char, u32)]) -> Vec<HuffmanTree> {
    let mut result = Vec::new();
    for (symbol, weight) in pairs {
        result = adjoin_set_huffman(HuffmanTree::make_leaf(*symbol, *weight), &result);
    }
    result
}

/// Generate Huffman tree from symbol-frequency pairs
pub fn generate_huffman_tree(pairs: &[(char, u32)]) -> HuffmanTree {
    fn successive_merge(leaves: Vec<HuffmanTree>) -> HuffmanTree {
        if leaves.len() == 1 {
            return leaves[0].clone();
        }

        // Take two smallest
        let left = leaves[0].clone();
        let right = leaves[1].clone();
        let merged = HuffmanTree::make_code_tree(left, right);

        // Add merged back to set
        let remaining = adjoin_set_huffman(merged, &leaves[2..]);
        successive_merge(remaining)
    }

    successive_merge(make_leaf_set(pairs))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- 2.3.1 Tests ---

    #[test]
    fn test_memq() {
        let list = vec!["apple", "banana", "pear"];
        assert_eq!(memq("apple", &list), Some(&["apple", "banana", "pear"][..]));
        assert_eq!(memq("banana", &list), Some(&["banana", "pear"][..]));
        assert_eq!(memq("orange", &list), None);
    }

    #[test]
    fn test_equal() {
        let a = vec![1, 2, 3];
        let b = vec![1, 2, 3];
        let c = vec![1, 2, 4];
        assert!(equal(&a, &b));
        assert!(!equal(&a, &c));
    }

    // --- 2.3.2 Tests ---

    #[test]
    fn test_deriv_constant() {
        let exp = Expr::Const(5.0);
        let result = deriv(&exp, "x");
        assert_eq!(result, Expr::Const(0.0));
    }

    #[test]
    fn test_deriv_variable() {
        let exp = Expr::Var("x");
        assert_eq!(deriv(&exp, "x"), Expr::Const(1.0));
        assert_eq!(deriv(&exp, "y"), Expr::Const(0.0));
    }

    #[test]
    fn test_deriv_sum() {
        // d(x + 3)/dx = 1
        let exp = Expr::Sum(Box::new(Expr::Var("x")), Box::new(Expr::Const(3.0)));
        let result = deriv(&exp, "x");
        assert_eq!(result, Expr::Const(1.0));
    }

    #[test]
    fn test_deriv_product() {
        // d(x * y)/dx = y
        let exp = Expr::Product(Box::new(Expr::Var("x")), Box::new(Expr::Var("y")));
        let result = deriv(&exp, "x");
        assert_eq!(result, Expr::Var("y"));
    }

    #[test]
    fn test_deriv_exponent() {
        // d(x^3)/dx = 3*x^2
        let exp = Expr::Exponent(Box::new(Expr::Var("x")), 3);
        let result = deriv(&exp, "x");
        // Result should be 3 * x^2 * 1 = 3 * x^2
        match result {
            Expr::Product(_, _) => {} // Simplified form
            _ => panic!("Expected product"),
        }
    }

    // --- 2.3.3 Tests ---

    #[test]
    fn test_unordered_sets() {
        let set1 = vec![1, 2, 3];
        let set2 = vec![2, 3, 4];

        assert!(element_of_set_unordered(&2, &set1));
        assert!(!element_of_set_unordered(&5, &set1));

        let intersection = intersection_set_unordered(&set1, &set2);
        assert_eq!(intersection, vec![2, 3]);

        let union = union_set_unordered(&set1, &set2);
        assert_eq!(union.len(), 4);
    }

    #[test]
    fn test_ordered_sets() {
        let set1 = vec![1, 2, 3];
        let set2 = vec![2, 3, 4];

        assert!(element_of_set_ordered(&2, &set1));
        assert!(!element_of_set_ordered(&5, &set1));

        let intersection = intersection_set_ordered(&set1, &set2);
        assert_eq!(intersection, vec![2, 3]);

        let union = union_set_ordered(&set1, &set2);
        assert_eq!(union, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_tree_set() {
        let tree = Tree::empty()
            .adjoin_set(5)
            .adjoin_set(3)
            .adjoin_set(7)
            .adjoin_set(1);

        assert!(tree.element_of_set(&5));
        assert!(tree.element_of_set(&3));
        assert!(!tree.element_of_set(&10));

        let list = tree.tree_to_list();
        assert_eq!(list, vec![1, 3, 5, 7]);
    }

    #[test]
    fn test_list_to_tree() {
        let list = vec![1, 3, 5, 7, 9, 11];
        let tree = Tree::list_to_tree(&list);
        let result = tree.tree_to_list();
        assert_eq!(result, list);
    }

    // --- 2.3.4 Tests ---

    #[test]
    fn test_huffman_leaf() {
        let leaf = HuffmanTree::make_leaf('A', 8);
        assert!(leaf.is_leaf());
        assert_eq!(leaf.symbol_leaf(), Some('A'));
        assert_eq!(leaf.weight(), 8);
    }

    #[test]
    fn test_huffman_decode() {
        // Sample tree from Exercise 2.67
        let sample_tree = HuffmanTree::make_code_tree(
            HuffmanTree::make_leaf('A', 4),
            HuffmanTree::make_code_tree(
                HuffmanTree::make_leaf('B', 2),
                HuffmanTree::make_code_tree(
                    HuffmanTree::make_leaf('D', 1),
                    HuffmanTree::make_leaf('C', 1),
                ),
            ),
        );

        let message = vec![0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0];
        let decoded = decode(&message, &sample_tree).unwrap();
        assert_eq!(decoded, vec!['A', 'D', 'A', 'B', 'B', 'C', 'A']);
    }

    #[test]
    fn test_huffman_encode() {
        let sample_tree = HuffmanTree::make_code_tree(
            HuffmanTree::make_leaf('A', 4),
            HuffmanTree::make_code_tree(
                HuffmanTree::make_leaf('B', 2),
                HuffmanTree::make_code_tree(
                    HuffmanTree::make_leaf('D', 1),
                    HuffmanTree::make_leaf('C', 1),
                ),
            ),
        );

        let message = vec!['A', 'D', 'A', 'B', 'B', 'C', 'A'];
        let encoded = encode(&message, &sample_tree).unwrap();
        assert_eq!(encoded, vec![0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_generate_huffman_tree() {
        let pairs = vec![('A', 8), ('B', 3), ('C', 1), ('D', 1)];
        let tree = generate_huffman_tree(&pairs);
        assert_eq!(tree.weight(), 13);

        // Test that we can encode and decode
        let message = vec!['A', 'B', 'C', 'D'];
        let encoded = encode(&message, &tree).unwrap();
        let decoded = decode(&encoded, &tree).unwrap();
        assert_eq!(decoded, message);
    }
}
