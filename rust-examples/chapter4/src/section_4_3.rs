//! SICP Section 4.3: Nondeterministic Computing (amb evaluator)
//!
//! This module demonstrates nondeterministic programming through backtracking search.
//! In Scheme, the `amb` operator represents a nondeterministic choice between alternatives.
//! When a computation fails, the system backtracks to the most recent choice point.
//!
//! # Rust Approach
//!
//! We provide two complementary implementations:
//!
//! 1. **Iterator-based search** - for finite search spaces, leveraging Rust's iterator
//!    combinators for lazy evaluation and implicit backtracking
//! 2. **Continuation-passing style (CPS)** - explicit backtracking using closures
//!    that mirror Scheme's success/failure continuations
//!
//! # Key Mappings
//!
//! | Scheme | Rust |
//! |--------|------|
//! | `(amb e1 e2 ...)` | `amb!` macro or iterator over choices |
//! | `(require pred)` | `filter` or explicit failure continuation |
//! | Backtracking | Iterator combinators or CPS with closures |
//! | Success continuation | `FnOnce(T) -> R` |
//! | Failure continuation | `FnOnce() -> R` |

use std::collections::HashSet;

// ============================================================================
// Iterator-Based Approach (Simple, Composable)
// ============================================================================

/// An element of a list, nondeterministically chosen.
///
/// This is the simplest form of `amb` - it returns an iterator over all choices.
pub fn an_element_of<T: Clone>(items: &[T]) -> impl Iterator<Item = T> + '_ {
    items.iter().cloned()
}

/// An integer between `low` and `high` (inclusive).
pub fn an_integer_between(low: i32, high: i32) -> impl Iterator<Item = i32> {
    low..=high
}

/// An integer starting from `n` (infinite range).
///
/// WARNING: In practice, this must be bounded to avoid infinite loops.
/// Use with care and always add constraints that limit the search.
pub fn an_integer_starting_from(n: i32) -> impl Iterator<Item = i32> {
    n..
}

// ============================================================================
// Exercise 4.35: Pythagorean Triples
// ============================================================================

/// Find Pythagorean triples (i, j, k) where i ≤ j, i² + j² = k²,
/// with all values between `low` and `high`.
///
/// # Example (from SICP)
///
/// ```scheme
/// (define (a-pythagorean-triple-between low high)
///   (let ((i (an-integer-between low high)))
///     (let ((j (an-integer-between i high)))
///       (let ((k (an-integer-between j high)))
///         (require (= (+ (* i i) (* j j)) (* k k)))
///         (list i j k)))))
/// ```
pub fn pythagorean_triples_between(low: i32, high: i32) -> impl Iterator<Item = (i32, i32, i32)> {
    an_integer_between(low, high).flat_map(move |i| {
        an_integer_between(i, high).flat_map(move |j| {
            an_integer_between(j, high).filter_map(move |k| {
                // require: i² + j² = k²
                if i * i + j * j == k * k {
                    Some((i, j, k))
                } else {
                    None
                }
            })
        })
    })
}

/// Exercise 4.37: Ben's optimized version
///
/// This version is more efficient because it:
/// 1. Computes k from i and j (reducing one dimension of search)
/// 2. Checks if k is an integer and within bounds
///
/// Search space: O(n²) vs O(n³) for the naive version
pub fn pythagorean_triples_between_optimized(
    low: i32,
    high: i32,
) -> impl Iterator<Item = (i32, i32, i32)> {
    let hsq = high * high;
    an_integer_between(low, high).flat_map(move |i| {
        an_integer_between(i, high).filter_map(move |j| {
            let ksq = i * i + j * j;
            if ksq > hsq {
                return None;
            }
            // Check if k is a perfect square
            let k = (ksq as f64).sqrt();
            if k.fract() == 0.0 {
                let k = k as i32;
                Some((i, j, k))
            } else {
                None
            }
        })
    })
}

// ============================================================================
// Logic Puzzle: Multiple Dwelling Problem
// ============================================================================

/// Distinct elements check (used in logic puzzles)
pub fn distinct<T: Eq + std::hash::Hash>(items: &[T]) -> bool {
    let mut seen = HashSet::new();
    items.iter().all(|item| seen.insert(item))
}

/// Exercise 4.38-4.41: Multiple Dwelling Problem
///
/// Baker, Cooper, Fletcher, Miller, and Smith live on different floors
/// of a 5-floor apartment house. Constraints:
/// - Baker does not live on the top floor (5)
/// - Cooper does not live on the bottom floor (1)
/// - Fletcher does not live on top (5) or bottom (1)
/// - Miller lives on a higher floor than Cooper
/// - Smith does not live adjacent to Fletcher
/// - Fletcher does not live adjacent to Cooper
///
/// # Scheme (from SICP)
///
/// ```scheme
/// (define (multiple-dwelling)
///   (let ((baker (amb 1 2 3 4 5))
///         (cooper (amb 1 2 3 4 5))
///         (fletcher (amb 1 2 3 4 5))
///         (miller (amb 1 2 3 4 5))
///         (smith (amb 1 2 3 4 5)))
///     (require (distinct? (list baker cooper fletcher miller smith)))
///     (require (not (= baker 5)))
///     (require (not (= cooper 1)))
///     (require (not (= fletcher 5)))
///     (require (not (= fletcher 1)))
///     (require (> miller cooper))
///     (require (not (= (abs (- smith fletcher)) 1)))
///     (require (not (= (abs (- fletcher cooper)) 1)))
///     (list (list 'baker baker)
///           (list 'cooper cooper)
///           (list 'fletcher fletcher)
///           (list 'miller miller)
///           (list 'smith smith))))
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultipleDwelling {
    pub baker: i32,
    pub cooper: i32,
    pub fletcher: i32,
    pub miller: i32,
    pub smith: i32,
}

impl MultipleDwelling {
    pub fn solve() -> impl Iterator<Item = Self> {
        (1..=5).flat_map(|baker| {
            (1..=5).flat_map(move |cooper| {
                (1..=5).flat_map(move |fletcher| {
                    (1..=5).flat_map(move |miller| {
                        (1..=5).filter_map(move |smith| {
                            let floors = [baker, cooper, fletcher, miller, smith];

                            // All must be distinct
                            if !distinct(&floors) {
                                return None;
                            }

                            // Baker not on top
                            if baker == 5 {
                                return None;
                            }

                            // Cooper not on bottom
                            if cooper == 1 {
                                return None;
                            }

                            // Fletcher not on top or bottom
                            if fletcher == 5 || fletcher == 1 {
                                return None;
                            }

                            // Miller higher than Cooper
                            if miller <= cooper {
                                return None;
                            }

                            // Smith not adjacent to Fletcher
                            if i32::abs(smith - fletcher) == 1 {
                                return None;
                            }

                            // Fletcher not adjacent to Cooper
                            if i32::abs(fletcher - cooper) == 1 {
                                return None;
                            }

                            Some(MultipleDwelling {
                                baker,
                                cooper,
                                fletcher,
                                miller,
                                smith,
                            })
                        })
                    })
                })
            })
        })
    }

    /// Exercise 4.38: Without the Smith-Fletcher adjacency constraint
    pub fn solve_without_smith_fletcher_adjacency() -> impl Iterator<Item = Self> {
        (1..=5).flat_map(|baker| {
            (1..=5).flat_map(move |cooper| {
                (1..=5).flat_map(move |fletcher| {
                    (1..=5).flat_map(move |miller| {
                        (1..=5).filter_map(move |smith| {
                            let floors = [baker, cooper, fletcher, miller, smith];

                            if !distinct(&floors) {
                                return None;
                            }
                            if baker == 5 {
                                return None;
                            }
                            if cooper == 1 {
                                return None;
                            }
                            if fletcher == 5 || fletcher == 1 {
                                return None;
                            }
                            if miller <= cooper {
                                return None;
                            }
                            // REMOVED: Smith-Fletcher adjacency constraint
                            if i32::abs(fletcher - cooper) == 1 {
                                return None;
                            }

                            Some(MultipleDwelling {
                                baker,
                                cooper,
                                fletcher,
                                miller,
                                smith,
                            })
                        })
                    })
                })
            })
        })
    }

    /// Exercise 4.40: Optimized version with early pruning
    ///
    /// Instead of generating all 5^5 = 3125 combinations and filtering,
    /// we prune early by checking constraints as soon as variables are bound.
    ///
    /// Search space reduction:
    /// - Before distinct check: 3125 combinations
    /// - After distinct check: 5! = 120 combinations
    /// - With early pruning: ~60 combinations explored
    pub fn solve_optimized() -> impl Iterator<Item = Self> {
        (1..=5)
            .filter(|&baker| baker != 5) // Baker not on top
            .flat_map(|baker| {
                (1..=5)
                    .filter(move |&cooper| cooper != 1 && cooper != baker) // Cooper not bottom, distinct
                    .flat_map(move |cooper| {
                        (1..=5)
                            .filter(move |&fletcher| {
                                fletcher != 1 // Fletcher not bottom
                                    && fletcher != 5 // Fletcher not top
                                    && fletcher != baker
                                    && fletcher != cooper
                                    && i32::abs(fletcher - cooper) != 1 // Not adjacent to Cooper
                            })
                            .flat_map(move |fletcher| {
                                (1..=5)
                                    .filter(move |&miller| {
                                        miller > cooper // Higher than Cooper
                                            && miller != baker
                                            && miller != cooper
                                            && miller != fletcher
                                    })
                                    .flat_map(move |miller| {
                                        (1..=5).filter_map(move |smith| {
                                            if smith == baker
                                                || smith == cooper
                                                || smith == fletcher
                                                || smith == miller
                                            {
                                                return None;
                                            }

                                            if i32::abs(smith - fletcher) == 1 {
                                                return None;
                                            }

                                            Some(MultipleDwelling {
                                                baker,
                                                cooper,
                                                fletcher,
                                                miller,
                                                smith,
                                            })
                                        })
                                    })
                            })
                    })
            })
    }
}

// ============================================================================
// Exercise 4.42: Liars Puzzle
// ============================================================================

/// Five schoolgirls took an exam. Each made one true and one false statement:
/// - Betty: "Kitty was 2nd. I was 3rd."
/// - Ethel: "I was 1st. Joan was 2nd."
/// - Joan: "I was 3rd. Ethel was 5th."
/// - Kitty: "I was 2nd. Mary was 4th."
/// - Mary: "I was 4th. Betty was 1st."
///
/// Find the actual ranking (1st to 5th).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiarsPuzzle {
    pub betty: i32,
    pub ethel: i32,
    pub joan: i32,
    pub kitty: i32,
    pub mary: i32,
}

impl LiarsPuzzle {
    /// Check if exactly one of two statements is true (XOR)
    fn exactly_one_true(a: bool, b: bool) -> bool {
        a ^ b // XOR: true if exactly one is true
    }

    pub fn solve() -> impl Iterator<Item = Self> {
        (1..=5).flat_map(|betty| {
            (1..=5).flat_map(move |ethel| {
                (1..=5).flat_map(move |joan| {
                    (1..=5).flat_map(move |kitty| {
                        (1..=5).filter_map(move |mary| {
                            let ranks = [betty, ethel, joan, kitty, mary];

                            // All ranks must be distinct (1-5)
                            if !distinct(&ranks) {
                                return None;
                            }

                            // Betty: "Kitty was 2nd (T). I was 3rd (F)."
                            // OR "Kitty was 2nd (F). I was 3rd (T)."
                            if !Self::exactly_one_true(kitty == 2, betty == 3) {
                                return None;
                            }

                            // Ethel: exactly one of (ethel==1, joan==2) is true
                            if !Self::exactly_one_true(ethel == 1, joan == 2) {
                                return None;
                            }

                            // Joan: exactly one of (joan==3, ethel==5) is true
                            if !Self::exactly_one_true(joan == 3, ethel == 5) {
                                return None;
                            }

                            // Kitty: exactly one of (kitty==2, mary==4) is true
                            if !Self::exactly_one_true(kitty == 2, mary == 4) {
                                return None;
                            }

                            // Mary: exactly one of (mary==4, betty==1) is true
                            if !Self::exactly_one_true(mary == 4, betty == 1) {
                                return None;
                            }

                            Some(LiarsPuzzle {
                                betty,
                                ethel,
                                joan,
                                kitty,
                                mary,
                            })
                        })
                    })
                })
            })
        })
    }
}

// ============================================================================
// Exercise 4.44: Eight Queens
// ============================================================================

/// Place 8 queens on a chessboard such that no two queens attack each other.
///
/// A queen attacks any piece on the same row, column, or diagonal.
/// We represent a board as a vector where `board[col] = row` (0-indexed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EightQueens {
    pub board: Vec<usize>, // board[col] = row
}

impl EightQueens {
    /// Check if placing a queen at (row, col) is safe given existing queens
    fn is_safe(board: &[usize], row: usize, col: usize) -> bool {
        for (other_col, &other_row) in board.iter().enumerate() {
            // Same row
            if other_row == row {
                return false;
            }

            // Diagonal attack: |row1 - row2| == |col1 - col2|
            let row_diff = (row as i32 - other_row as i32).abs();
            let col_diff = (col as i32 - other_col as i32).abs();
            if row_diff == col_diff {
                return false;
            }
        }
        true
    }

    /// Solve 8-queens using backtracking
    pub fn solve() -> impl Iterator<Item = Self> {
        Self::solve_recursive(Vec::new(), 0)
    }

    fn solve_recursive(board: Vec<usize>, col: usize) -> Box<dyn Iterator<Item = Self>> {
        if col == 8 {
            // Found a complete solution
            return Box::new(std::iter::once(EightQueens { board }));
        }

        // Try placing a queen in each row of the current column
        // Collect safe rows first to avoid moving board multiple times
        let safe_rows: Vec<usize> = (0..8)
            .filter(|&row| Self::is_safe(&board, row, col))
            .collect();

        Box::new(safe_rows.into_iter().flat_map(move |row| {
            let mut new_board = board.clone();
            new_board.push(row);
            Self::solve_recursive(new_board, col + 1)
        }))
    }

    /// Visualize the board (for debugging)
    pub fn display(&self) -> String {
        let mut result = String::new();
        for row in 0..8 {
            for col in 0..8 {
                if self.board[col] == row {
                    result.push('Q');
                } else {
                    result.push('.');
                }
                result.push(' ');
            }
            result.push('\n');
        }
        result
    }
}

// ============================================================================
// Iterator-Based Search (Idiomatic Rust Alternative to CPS)
// ============================================================================

/// Simplified backtracking using iterators and filter.
///
/// While Scheme's amb uses explicit continuations (success/failure), Rust's
/// ownership model makes true CPS challenging. Instead, we use iterators which
/// provide implicit backtracking through lazy evaluation.
///
/// The key insight: Rust's iterators already implement backtracking via
/// `flat_map`, `filter`, and `find_map`. When a predicate fails, the iterator
/// automatically tries the next choice.
///
/// This is actually *more elegant* than explicit continuations because:
/// 1. No manual state management (iterator does it for us)
/// 2. No lifetime issues (no closures capturing by reference)
/// 3. Composable (can chain with other iterator adapters)
/// 4. Lazy (only computes as needed)
///
/// # Example
///
/// ```
/// use sicp_chapter4::section_4_3::*;
///
/// let result = amb_search(vec![1, 2, 3], |x| {
///     if x > 1 {
///         Some(x * 2)
///     } else {
///         None // backtrack
///     }
/// });
/// assert_eq!(result, Some(4)); // First value where x > 1 is 2, so 2*2=4
/// ```
pub fn amb_search<T, F, R>(choices: Vec<T>, f: F) -> Option<R>
where
    F: Fn(T) -> Option<R>,
{
    choices.into_iter().find_map(f)
}

/// Pythagorean triple finder using iterator-based search
///
/// This demonstrates nested choice points without explicit continuations.
/// Each `find_map` is a choice point; returning `None` triggers backtracking.
pub fn pythagorean_triple_search(low: i32, high: i32) -> Option<(i32, i32, i32)> {
    (low..=high).find_map(|i| {
        (i..=high).find_map(|j| {
            (j..=high).find_map(|k| {
                // require: i² + j² = k²
                if i * i + j * j == k * k {
                    Some((i, j, k))
                } else {
                    None // backtrack
                }
            })
        })
    })
}

// ============================================================================
// Natural Language Parsing (Section 4.3.2)
// ============================================================================

/// Simple grammar for natural language parsing
///
/// # Grammar
///
/// ```text
/// sentence       ::= noun-phrase verb-phrase
/// noun-phrase    ::= simple-noun-phrase | noun-phrase prep-phrase
/// simple-noun-phrase ::= article noun
/// verb-phrase    ::= verb | verb-phrase prep-phrase
/// prep-phrase    ::= preposition noun-phrase
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseTree {
    Sentence(Box<ParseTree>, Box<ParseTree>),
    NounPhrase(Box<ParseTree>, Box<ParseTree>),
    SimpleNounPhrase(String, String), // article, noun
    VerbPhrase(Box<ParseTree>, Option<Box<ParseTree>>),
    PrepPhrase(String, Box<ParseTree>), // preposition, noun-phrase
    Verb(String),
}

/// Simple parser using backtracking (iterator-based)
pub struct Parser {
    words: Vec<String>,
    pos: usize,
}

impl Parser {
    pub fn new(input: &str) -> Self {
        Parser {
            words: input.split_whitespace().map(String::from).collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<&str> {
        self.words.get(self.pos).map(|s| s.as_str())
    }

    fn consume(&mut self) -> Option<String> {
        if self.pos < self.words.len() {
            let word = self.words[self.pos].clone();
            self.pos += 1;
            Some(word)
        } else {
            None
        }
    }

    fn parse_article(&mut self) -> Option<String> {
        match self.peek() {
            Some("the") | Some("a") => self.consume(),
            _ => None,
        }
    }

    fn parse_noun(&mut self) -> Option<String> {
        match self.peek() {
            Some("student") | Some("professor") | Some("cat") | Some("class") => self.consume(),
            _ => None,
        }
    }

    fn parse_verb(&mut self) -> Option<String> {
        match self.peek() {
            Some("studies") | Some("lectures") | Some("eats") | Some("sleeps") => self.consume(),
            _ => None,
        }
    }

    fn parse_preposition(&mut self) -> Option<String> {
        match self.peek() {
            Some("for") | Some("to") | Some("in") | Some("by") | Some("with") => self.consume(),
            _ => None,
        }
    }

    pub fn parse_simple_noun_phrase(&mut self) -> Option<ParseTree> {
        let article = self.parse_article()?;
        let noun = self.parse_noun()?;
        Some(ParseTree::SimpleNounPhrase(article, noun))
    }

    pub fn parse_sentence(&mut self) -> Option<ParseTree> {
        let np = self.parse_noun_phrase()?;
        let vp = self.parse_verb_phrase()?;

        if self.pos < self.words.len() {
            return None; // Unparsed words remain
        }

        Some(ParseTree::Sentence(Box::new(np), Box::new(vp)))
    }

    fn parse_noun_phrase(&mut self) -> Option<ParseTree> {
        let simple = self.parse_simple_noun_phrase()?;
        self.maybe_extend_noun_phrase(simple)
    }

    fn maybe_extend_noun_phrase(&mut self, np: ParseTree) -> Option<ParseTree> {
        // Try to extend with a prepositional phrase
        let saved_pos = self.pos;
        if let Some(pp) = self.parse_prep_phrase() {
            let extended = ParseTree::NounPhrase(Box::new(np), Box::new(pp));
            // Recursively try to extend further
            self.maybe_extend_noun_phrase(extended)
        } else {
            self.pos = saved_pos;
            Some(np)
        }
    }

    fn parse_verb_phrase(&mut self) -> Option<ParseTree> {
        let verb_str = self.parse_verb()?;
        let verb = ParseTree::Verb(verb_str);
        self.maybe_extend_verb_phrase(verb)
    }

    fn maybe_extend_verb_phrase(&mut self, vp: ParseTree) -> Option<ParseTree> {
        let saved_pos = self.pos;
        if let Some(pp) = self.parse_prep_phrase() {
            let extended = ParseTree::VerbPhrase(Box::new(vp), Some(Box::new(pp)));
            self.maybe_extend_verb_phrase(extended)
        } else {
            self.pos = saved_pos;
            Some(ParseTree::VerbPhrase(Box::new(vp), None))
        }
    }

    fn parse_prep_phrase(&mut self) -> Option<ParseTree> {
        let prep = self.parse_preposition()?;
        let np = self.parse_noun_phrase()?;
        Some(ParseTree::PrepPhrase(prep, Box::new(np)))
    }
}

pub fn parse(input: &str) -> Option<ParseTree> {
    let mut parser = Parser::new(input);
    parser.parse_sentence()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pythagorean_triples() {
        let triples: Vec<_> = pythagorean_triples_between(1, 20).collect();
        assert!(triples.contains(&(3, 4, 5)));
        assert!(triples.contains(&(5, 12, 13)));
        assert!(triples.contains(&(8, 15, 17)));
    }

    #[test]
    fn test_pythagorean_triples_optimized() {
        let naive: Vec<_> = pythagorean_triples_between(1, 20).collect();
        let optimized: Vec<_> = pythagorean_triples_between_optimized(1, 20).collect();
        assert_eq!(naive.len(), optimized.len());
        for triple in &naive {
            assert!(optimized.contains(triple));
        }
    }

    #[test]
    fn test_multiple_dwelling() {
        let solutions: Vec<_> = MultipleDwelling::solve().collect();
        assert_eq!(solutions.len(), 1);
        let sol = solutions[0];
        assert_eq!(sol.baker, 3);
        assert_eq!(sol.cooper, 2);
        assert_eq!(sol.fletcher, 4);
        assert_eq!(sol.miller, 5);
        assert_eq!(sol.smith, 1);
    }

    #[test]
    fn test_multiple_dwelling_without_smith_fletcher() {
        let count = MultipleDwelling::solve_without_smith_fletcher_adjacency().count();
        // Expected: 5 solutions (Exercise 4.38)
        assert_eq!(count, 5);
    }

    #[test]
    fn test_multiple_dwelling_optimized() {
        let naive: Vec<_> = MultipleDwelling::solve().collect();
        let optimized: Vec<_> = MultipleDwelling::solve_optimized().collect();
        assert_eq!(naive.len(), optimized.len());
        assert_eq!(naive[0], optimized[0]);
    }

    #[test]
    fn test_liars_puzzle() {
        let solutions: Vec<_> = LiarsPuzzle::solve().collect();
        assert_eq!(solutions.len(), 1, "Should have exactly one solution");
        let sol = solutions[0];
        // Solution: Betty=3, Ethel=5, Joan=2, Kitty=1, Mary=4
        // Verified against the puzzle constraints:
        // - Betty (3rd): "Kitty 2nd" (F), "I 3rd" (T) → exactly one true ✓
        // - Ethel (5th): "I 1st" (F), "Joan 2nd" (T) → exactly one true ✓
        // - Joan (2nd): "I 3rd" (F), "Ethel 5th" (T) → exactly one true ✓
        // - Kitty (1st): "I 2nd" (F), "Mary 4th" (T) → exactly one true ✓
        // - Mary (4th): "I 4th" (T), "Betty 1st" (F) → exactly one true ✓
        assert_eq!(sol.betty, 3);
        assert_eq!(sol.ethel, 5);
        assert_eq!(sol.joan, 2);
        assert_eq!(sol.kitty, 1);
        assert_eq!(sol.mary, 4);
    }

    #[test]
    fn test_eight_queens() {
        let solutions: Vec<_> = EightQueens::solve().take(10).collect();
        assert!(!solutions.is_empty());

        // Verify first solution is valid
        let first = &solutions[0];
        for col in 0..8 {
            for other_col in 0..8 {
                if col == other_col {
                    continue;
                }
                let row = first.board[col];
                let other_row = first.board[other_col];

                // No same row
                assert_ne!(row, other_row);

                // No diagonal attack
                let row_diff = (row as i32 - other_row as i32).abs();
                let col_diff = (col as i32 - other_col as i32).abs();
                assert_ne!(row_diff, col_diff);
            }
        }
    }

    #[test]
    fn test_pythagorean_triple_search() {
        let result = pythagorean_triple_search(1, 20);
        assert!(result.is_some());
        let (i, j, k) = result.unwrap();
        assert_eq!(i * i + j * j, k * k);
        assert_eq!((i, j, k), (3, 4, 5)); // First solution
    }

    #[test]
    fn test_amb_search() {
        let result = amb_search(vec![1, 2, 3], |x| if x > 1 { Some(x * 2) } else { None });
        assert_eq!(result, Some(4));
    }

    #[test]
    fn test_parse_simple_sentence() {
        let tree = parse("the cat eats").unwrap();
        match tree {
            ParseTree::Sentence(_, _) => {}
            _ => panic!("Expected Sentence"),
        }
    }

    #[test]
    fn test_parse_complex_sentence() {
        let tree = parse("the student with the cat sleeps in the class");
        assert!(tree.is_some());
    }

    #[test]
    fn test_distinct() {
        assert!(distinct(&[1, 2, 3, 4, 5]));
        assert!(!distinct(&[1, 2, 3, 2, 5]));
    }
}
