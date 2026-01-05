//! SICP 4.3절: 비결정적 계산 (amb 평가기) (Nondeterministic Computing)
//!
//! 이 모듈은 백트래킹 탐색을 통한 비결정적 프로그래밍을 보여준다
//! (This module demonstrates nondeterministic programming through backtracking search).
//! Scheme에서 `amb` 연산자는 대안들 사이의 비결정적 선택을 뜻한다.
//! 계산이 실패하면 시스템은 가장 최근의 선택 지점으로 되돌아간다
//! (In Scheme, the `amb` operator represents a nondeterministic choice between alternatives.
//! When a computation fails, the system backtracks to the most recent choice point).
//!
//! # 러스트 접근 (Rust Approach)
//!
//! 서로 보완적인 두 가지 구현을 제공한다:
//! (We provide two complementary implementations:)
//!
//! 1. **이터레이터 기반 탐색 (Iterator-based search)** - 유한 탐색 공간에서
//!    러스트 이터레이터 컴비네이터를 사용해 지연 평가와 암묵적 백트래킹을 제공
//! 2. **연속 전달 스타일 (CPS)** - Scheme의 성공/실패 연속과 유사한 클로저로
//!    명시적 백트래킹을 구현
//!
//! # 핵심 매핑 (Key Mappings)
//!
//! | Scheme | Rust |
//! |--------|------|
//! | `(amb e1 e2 ...)` | `amb!` 매크로 또는 선택지 이터레이터 |
//! | `(require pred)` | `filter` 또는 명시적 실패 연속 |
//! | 백트래킹 (Backtracking) | 이터레이터 컴비네이터 또는 CPS 클로저 |
//! | 성공 연속 (Success continuation) | `FnOnce(T) -> R` |
//! | 실패 연속 (Failure continuation) | `FnOnce() -> R` |

use std::collections::HashSet;

// ============================================================================
// 이터레이터 기반 접근 (단순, 조합 가능)
// (Iterator-Based Approach (Simple, Composable))
// ============================================================================

/// 리스트 원소를 비결정적으로 선택한다
/// (An element of a list, nondeterministically chosen).
///
/// `amb`의 가장 단순한 형태로, 모든 선택지에 대한 이터레이터를 반환한다
/// (This is the simplest form of `amb` - it returns an iterator over all choices).
pub fn an_element_of<T: Clone>(items: &[T]) -> impl Iterator<Item = T> + '_ {
    items.iter().cloned()
}

/// `low`와 `high` 사이의 정수 (포함)
/// (An integer between `low` and `high` (inclusive)).
pub fn an_integer_between(low: i32, high: i32) -> impl Iterator<Item = i32> {
    low..=high
}

/// `n`부터 시작하는 정수 (무한 범위)
/// (An integer starting from `n` (infinite range)).
///
/// 경고: 실제로는 무한 루프를 피하기 위해 반드시 범위를 제한해야 한다
/// (WARNING: In practice, this must be bounded to avoid infinite loops).
/// 주의해서 사용하고 항상 탐색을 제한하는 제약을 추가하라
/// (Use with care and always add constraints that limit the search).
pub fn an_integer_starting_from(n: i32) -> impl Iterator<Item = i32> {
    n..
}

// ============================================================================
// 연습문제 4.35: 피타고라스 삼중쌍 (Pythagorean Triples)
// ============================================================================

/// i ≤ j, i² + j² = k²를 만족하는 피타고라스 삼중쌍 (i, j, k)을 찾는다
/// (Find Pythagorean triples (i, j, k) where i ≤ j, i² + j² = k²),
/// 모든 값은 `low`와 `high` 사이에 있다
/// (with all values between `low` and `high`).
///
/// # 예시 (SICP에서) (Example (from SICP))
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
                // require: i² + j² = k² (조건) (constraint)
                if i * i + j * j == k * k {
                    Some((i, j, k))
                } else {
                    None
                }
            })
        })
    })
}

/// 연습문제 4.37: Ben의 최적화 버전 (Ben's optimized version)
///
/// 이 버전은 다음 이유로 더 효율적이다:
/// (This version is more efficient because it:)
/// 1. i와 j로 k를 계산해 탐색 차원을 줄인다
///    (Computes k from i and j (reducing one dimension of search))
/// 2. k가 정수이고 범위 내인지 확인한다
///    (Checks if k is an integer and within bounds)
///
/// 탐색 공간: 순진한 버전의 O(n³) 대비 O(n²)
/// (Search space: O(n²) vs O(n³) for the naive version)
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
            // k가 완전제곱인지 확인 (Check if k is a perfect square)
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
// 논리 퍼즐: 다중 거주 문제 (Multiple Dwelling Problem)
// ============================================================================

/// 모든 원소가 서로 다른지 확인 (논리 퍼즐에서 사용)
/// (Distinct elements check (used in logic puzzles))
pub fn distinct<T: Eq + std::hash::Hash>(items: &[T]) -> bool {
    let mut seen = HashSet::new();
    items.iter().all(|item| seen.insert(item))
}

/// 연습문제 4.38-4.41: 다중 거주 문제 (Multiple Dwelling Problem)
///
/// Baker, Cooper, Fletcher, Miller, Smith는 5층 아파트의 서로 다른 층에 산다.
/// 제약:
/// - Baker는 꼭대기 층(5층)에 살지 않는다
/// - Cooper는 1층에 살지 않는다
/// - Fletcher는 5층이나 1층에 살지 않는다
/// - Miller는 Cooper보다 높은 층에 산다
/// - Smith는 Fletcher와 인접한 층에 살지 않는다
/// - Fletcher는 Cooper와 인접한 층에 살지 않는다
///
/// # Scheme (SICP에서) (Scheme (from SICP))
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

                            // 모두 서로 달라야 한다 (All must be distinct)
                            if !distinct(&floors) {
                                return None;
                            }

                            // Baker는 꼭대기 층 금지 (Baker not on top)
                            if baker == 5 {
                                return None;
                            }

                            // Cooper는 1층 금지 (Cooper not on bottom)
                            if cooper == 1 {
                                return None;
                            }

                            // Fletcher는 꼭대기/1층 금지 (Fletcher not on top or bottom)
                            if fletcher == 5 || fletcher == 1 {
                                return None;
                            }

                            // Miller는 Cooper보다 위층 (Miller higher than Cooper)
                            if miller <= cooper {
                                return None;
                            }

                            // Smith는 Fletcher와 인접 금지 (Smith not adjacent to Fletcher)
                            if i32::abs(smith - fletcher) == 1 {
                                return None;
                            }

                            // Fletcher는 Cooper와 인접 금지 (Fletcher not adjacent to Cooper)
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

    /// 연습문제 4.38: Smith-Fletcher 인접 제약을 제거한 경우
    /// (Exercise 4.38: Without the Smith-Fletcher adjacency constraint)
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
                            // 제거됨: Smith-Fletcher 인접 제약
                            // (REMOVED: Smith-Fletcher adjacency constraint)
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

    /// 연습문제 4.40: 조기 가지치기를 적용한 최적화 버전
    /// (Exercise 4.40: Optimized version with early pruning)
    ///
    /// 5^5 = 3125 조합을 전부 생성한 뒤 필터링하는 대신,
    /// 변수가 바인딩되는 즉시 제약을 검사해 조기 가지치기를 한다
    /// (Instead of generating all 5^5 = 3125 combinations and filtering,
    /// we prune early by checking constraints as soon as variables are bound).
    ///
    /// 탐색 공간 축소:
    /// - distinct 검사 전: 3125 조합
    /// - distinct 검사 후: 5! = 120 조합
    /// - 조기 가지치기 적용: 약 60 조합 탐색
    /// (Search space reduction:
    /// - Before distinct check: 3125 combinations
    /// - After distinct check: 5! = 120 combinations
    /// - With early pruning: ~60 combinations explored)
    pub fn solve_optimized() -> impl Iterator<Item = Self> {
        (1..=5)
            .filter(|&baker| baker != 5) // Baker는 꼭대기 층 금지 (Baker not on top)
            .flat_map(|baker| {
                (1..=5)
                    .filter(move |&cooper| cooper != 1 && cooper != baker) // Cooper는 1층 금지, 서로 다름 (Cooper not bottom, distinct)
                    .flat_map(move |cooper| {
                        (1..=5)
                            .filter(move |&fletcher| {
                                fletcher != 1 // Fletcher는 1층 금지 (Fletcher not bottom)
                                    && fletcher != 5 // Fletcher는 꼭대기 금지 (Fletcher not top)
                                    && fletcher != baker
                                    && fletcher != cooper
                                    && i32::abs(fletcher - cooper) != 1 // Cooper와 인접 금지 (Not adjacent to Cooper)
                            })
                            .flat_map(move |fletcher| {
                                (1..=5)
                                    .filter(move |&miller| {
                                        miller > cooper // Cooper보다 위층 (Higher than Cooper)
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
// 연습문제 4.42: 거짓말쟁이 퍼즐 (Liars Puzzle)
// ============================================================================

/// 다섯 명의 여학생이 시험을 봤고, 각자 진실 1개와 거짓 1개를 말했다:
/// - Betty: "Kitty는 2등. 나는 3등."
/// - Ethel: "나는 1등. Joan은 2등."
/// - Joan: "나는 3등. Ethel은 5등."
/// - Kitty: "나는 2등. Mary는 4등."
/// - Mary: "나는 4등. Betty는 1등."
///
/// 실제 순위를 찾아라 (1등~5등).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiarsPuzzle {
    pub betty: i32,
    pub ethel: i32,
    pub joan: i32,
    pub kitty: i32,
    pub mary: i32,
}

impl LiarsPuzzle {
    /// 두 문장 중 정확히 하나만 참인지 확인 (XOR)
    /// (Check if exactly one of two statements is true (XOR))
    fn exactly_one_true(a: bool, b: bool) -> bool {
        a ^ b // XOR: 정확히 하나만 참이면 true (XOR: true if exactly one is true)
    }

    pub fn solve() -> impl Iterator<Item = Self> {
        (1..=5).flat_map(|betty| {
            (1..=5).flat_map(move |ethel| {
                (1..=5).flat_map(move |joan| {
                    (1..=5).flat_map(move |kitty| {
                        (1..=5).filter_map(move |mary| {
                            let ranks = [betty, ethel, joan, kitty, mary];

                            // 모든 순위는 서로 달라야 함 (1-5)
                            // (All ranks must be distinct (1-5))
                            if !distinct(&ranks) {
                                return None;
                            }

                            // Betty: "Kitty는 2등 (T). 나는 3등 (F)."
                            // 또는 "Kitty는 2등 (F). 나는 3등 (T)."
                            if !Self::exactly_one_true(kitty == 2, betty == 3) {
                                return None;
                            }

                            // Ethel: (ethel==1, joan==2) 중 하나만 참
                            if !Self::exactly_one_true(ethel == 1, joan == 2) {
                                return None;
                            }

                            // Joan: (joan==3, ethel==5) 중 하나만 참
                            if !Self::exactly_one_true(joan == 3, ethel == 5) {
                                return None;
                            }

                            // Kitty: (kitty==2, mary==4) 중 하나만 참
                            if !Self::exactly_one_true(kitty == 2, mary == 4) {
                                return None;
                            }

                            // Mary: (mary==4, betty==1) 중 하나만 참
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
// 연습문제 4.44: 8-퀸 (Eight Queens)
// ============================================================================

/// 체스판에 8개의 퀸을 서로 공격하지 않게 배치한다
/// (Place 8 queens on a chessboard such that no two queens attack each other).
///
/// 퀸은 같은 행, 열, 대각선에 있는 어떤 말도 공격할 수 있다.
/// 보드는 `board[col] = row` 형태의 벡터로 표현한다 (0부터 인덱스)
/// (A queen attacks any piece on the same row, column, or diagonal.
/// We represent a board as a vector where `board[col] = row` (0-indexed)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EightQueens {
    pub board: Vec<usize>, // board[col] = row (열=col, 행=row) (board[col] = row)
}

impl EightQueens {
    /// (row, col)에 퀸을 두는 것이 안전한지 검사
    /// (Check if placing a queen at (row, col) is safe given existing queens)
    fn is_safe(board: &[usize], row: usize, col: usize) -> bool {
        for (other_col, &other_row) in board.iter().enumerate() {
            // 같은 행 (Same row)
            if other_row == row {
                return false;
            }

            // 대각선 공격: |row1 - row2| == |col1 - col2|
            // (Diagonal attack: |row1 - row2| == |col1 - col2|)
            let row_diff = (row as i32 - other_row as i32).abs();
            let col_diff = (col as i32 - other_col as i32).abs();
            if row_diff == col_diff {
                return false;
            }
        }
        true
    }

    /// 백트래킹으로 8-퀸 해결
    /// (Solve 8-queens using backtracking)
    pub fn solve() -> impl Iterator<Item = Self> {
        Self::solve_recursive(Vec::new(), 0)
    }

    fn solve_recursive(board: Vec<usize>, col: usize) -> Box<dyn Iterator<Item = Self>> {
        if col == 8 {
            // 완전한 해를 찾음 (Found a complete solution)
            return Box::new(std::iter::once(EightQueens { board }));
        }

        // 현재 열의 각 행에 퀸을 배치해본다
        // (Try placing a queen in each row of the current column)
        // 보드를 여러 번 이동하지 않도록 안전한 행을 먼저 수집
        // (Collect safe rows first to avoid moving board multiple times)
        let safe_rows: Vec<usize> = (0..8)
            .filter(|&row| Self::is_safe(&board, row, col))
            .collect();

        Box::new(safe_rows.into_iter().flat_map(move |row| {
            let mut new_board = board.clone();
            new_board.push(row);
            Self::solve_recursive(new_board, col + 1)
        }))
    }

    /// 보드를 시각화 (디버깅용)
    /// (Visualize the board (for debugging))
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
// 이터레이터 기반 탐색 (CPS 대체, 관용적 러스트)
// (Iterator-Based Search (Idiomatic Rust Alternative to CPS))
// ============================================================================

/// 이터레이터와 filter로 단순화한 백트래킹
/// (Simplified backtracking using iterators and filter).
///
/// Scheme의 amb는 명시적 연속(성공/실패)을 사용하지만,
/// Rust의 소유권 모델에서는 진정한 CPS가 어렵다. 대신 이터레이터를 사용해
/// 지연 평가를 통한 암묵적 백트래킹을 제공한다
/// (While Scheme's amb uses explicit continuations (success/failure), Rust's
/// ownership model makes true CPS challenging. Instead, we use iterators which
/// provide implicit backtracking through lazy evaluation).
///
/// 핵심 통찰: Rust 이터레이터는 `flat_map`, `filter`, `find_map`으로
/// 이미 백트래킹을 구현한다. 술어가 실패하면 다음 선택지를 자동으로 시도한다
/// (The key insight: Rust's iterators already implement backtracking via
/// `flat_map`, `filter`, and `find_map`. When a predicate fails, the iterator
/// automatically tries the next choice).
///
/// 이는 명시적 연속보다 *더 우아*하다:
/// (This is actually *more elegant* than explicit continuations because:)
/// 1. 수동 상태 관리 불필요 (이터레이터가 처리) (No manual state management)
/// 2. 수명 문제 없음 (참조 캡처 불필요) (No lifetime issues)
/// 3. 조합 가능 (다른 어댑터와 체인 가능) (Composable)
/// 4. 지연 평가 (필요한 만큼만 계산) (Lazy)
///
/// # 예시 (Example)
///
/// ```
/// use sicp_chapter4::section_4_3::*;
///
/// let result = amb_search(vec![1, 2, 3], |x| {
///     if x > 1 {
///         Some(x * 2)
///     } else {
///         None // 백트래킹 (backtrack)
///     }
/// });
/// assert_eq!(result, Some(4)); // x > 1의 첫 값은 2 → 2*2=4 (First value where x > 1 is 2, so 2*2=4)
/// ```
pub fn amb_search<T, F, R>(choices: Vec<T>, f: F) -> Option<R>
where
    F: Fn(T) -> Option<R>,
{
    choices.into_iter().find_map(f)
}

/// 이터레이터 기반 탐색으로 피타고라스 삼중쌍을 찾는다
/// (Pythagorean triple finder using iterator-based search)
///
/// 명시적 연속 없이 중첩 선택 지점을 보여준다.
/// 각 `find_map`이 선택 지점이며 `None`을 반환하면 백트래킹된다
/// (This demonstrates nested choice points without explicit continuations.
/// Each `find_map` is a choice point; returning `None` triggers backtracking).
pub fn pythagorean_triple_search(low: i32, high: i32) -> Option<(i32, i32, i32)> {
    (low..=high).find_map(|i| {
        (i..=high).find_map(|j| {
            (j..=high).find_map(|k| {
                // require: i² + j² = k² (조건) (constraint)
                if i * i + j * j == k * k {
                    Some((i, j, k))
                } else {
                    None // 백트래킹 (backtrack)
                }
            })
        })
    })
}

// ============================================================================
// 자연어 파싱 (4.3.2절) (Natural Language Parsing)
// ============================================================================

/// 자연어 파싱을 위한 간단한 문법
/// (Simple grammar for natural language parsing)
///
/// # 문법 (Grammar)
///
/// ```text
/// sentence       ::= noun-phrase verb-phrase   // 문장 (sentence)
/// noun-phrase    ::= simple-noun-phrase | noun-phrase prep-phrase   // 명사구 (noun-phrase)
/// simple-noun-phrase ::= article noun          // 단순 명사구 (simple-noun-phrase)
/// verb-phrase    ::= verb | verb-phrase prep-phrase   // 동사구 (verb-phrase)
/// prep-phrase    ::= preposition noun-phrase   // 전치사구 (prep-phrase)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseTree {
    Sentence(Box<ParseTree>, Box<ParseTree>),
    NounPhrase(Box<ParseTree>, Box<ParseTree>),
    SimpleNounPhrase(String, String), // 관사, 명사 (article, noun)
    VerbPhrase(Box<ParseTree>, Option<Box<ParseTree>>),
    PrepPhrase(String, Box<ParseTree>), // 전치사, 명사구 (preposition, noun-phrase)
    Verb(String),
}

/// 백트래킹을 사용하는 간단한 파서 (이터레이터 기반)
/// (Simple parser using backtracking (iterator-based))
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
            Some("그(the)") | Some("한(a)") => self.consume(),
            _ => None,
        }
    }

    fn parse_noun(&mut self) -> Option<String> {
        match self.peek() {
            Some("학생(student)")
            | Some("교수(professor)")
            | Some("고양이(cat)")
            | Some("수업(class)") => self.consume(),
            _ => None,
        }
    }

    fn parse_verb(&mut self) -> Option<String> {
        match self.peek() {
            Some("공부한다(studies)")
            | Some("강의한다(lectures)")
            | Some("먹는다(eats)")
            | Some("잔다(sleeps)") => self.consume(),
            _ => None,
        }
    }

    fn parse_preposition(&mut self) -> Option<String> {
        match self.peek() {
            Some("위해(for)")
            | Some("에게(to)")
            | Some("안에(in)")
            | Some("에의해(by)")
            | Some("함께(with)") => self.consume(),
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
            return None; // 파싱되지 않은 단어가 남음 (Unparsed words remain)
        }

        Some(ParseTree::Sentence(Box::new(np), Box::new(vp)))
    }

    fn parse_noun_phrase(&mut self) -> Option<ParseTree> {
        let simple = self.parse_simple_noun_phrase()?;
        self.maybe_extend_noun_phrase(simple)
    }

    fn maybe_extend_noun_phrase(&mut self, np: ParseTree) -> Option<ParseTree> {
        // 전치사구로 확장 시도 (Try to extend with a prepositional phrase)
        let saved_pos = self.pos;
        if let Some(pp) = self.parse_prep_phrase() {
            let extended = ParseTree::NounPhrase(Box::new(np), Box::new(pp));
            // 재귀적으로 추가 확장 시도 (Recursively try to extend further)
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
        // 기대값: 해 5개 (연습문제 4.38) (Expected: 5 solutions (Exercise 4.38))
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
        assert_eq!(
            solutions.len(),
            1,
            "정확히 하나의 해가 있어야 함 (Should have exactly one solution)"
        );
        let sol = solutions[0];
        // 해: Betty=3, Ethel=5, Joan=2, Kitty=1, Mary=4
        // 퍼즐 제약과 대조 확인:
        // - Betty (3등): "Kitty 2등" (F), "나는 3등" (T) → 하나만 참 ✓
        // - Ethel (5등): "나는 1등" (F), "Joan 2등" (T) → 하나만 참 ✓
        // - Joan (2등): "나는 3등" (F), "Ethel 5등" (T) → 하나만 참 ✓
        // - Kitty (1등): "나는 2등" (F), "Mary 4등" (T) → 하나만 참 ✓
        // - Mary (4등): "나는 4등" (T), "Betty 1등" (F) → 하나만 참 ✓
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

        // 첫 해가 유효한지 확인 (Verify first solution is valid)
        let first = &solutions[0];
        for col in 0..8 {
            for other_col in 0..8 {
                if col == other_col {
                    continue;
                }
                let row = first.board[col];
                let other_row = first.board[other_col];

                // 같은 행 금지 (No same row)
                assert_ne!(row, other_row);

                // 대각선 공격 금지 (No diagonal attack)
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
        assert_eq!((i, j, k), (3, 4, 5)); // 첫 번째 해 (First solution)
    }

    #[test]
    fn test_amb_search() {
        let result = amb_search(vec![1, 2, 3], |x| if x > 1 { Some(x * 2) } else { None });
        assert_eq!(result, Some(4));
    }

    #[test]
    fn test_parse_simple_sentence() {
        let tree = parse("그(the) 고양이(cat) 먹는다(eats)").unwrap();
        match tree {
            ParseTree::Sentence(_, _) => {}
            _ => panic!("문장을 기대함 (Expected Sentence)"),
        }
    }

    #[test]
    fn test_parse_complex_sentence() {
        let tree = parse("그(the) 학생(student) 함께(with) 그(the) 고양이(cat) 잔다(sleeps) 안에(in) 그(the) 수업(class)");
        assert!(tree.is_some());
    }

    #[test]
    fn test_distinct() {
        assert!(distinct(&[1, 2, 3, 4, 5]));
        assert!(!distinct(&[1, 2, 3, 2, 5]));
    }
}
