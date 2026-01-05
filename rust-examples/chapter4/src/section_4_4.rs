//! SICP 4.4절: 논리 프로그래밍 (Logic Programming)
//!
//! 이 모듈은 Prolog와 유사한 질의 언어 시스템을 구현하며,
//! 패턴 매칭, 통일, 논리적 추론을 시연한다
//! (This module implements a query language system similar to Prolog,
//! demonstrating pattern matching, unification, and logical inference).
//!
//! # 핵심 개념 (Core Concepts)
//!
//! - **패턴 매칭 (Pattern Matching)**: 패턴을 데이터에 단방향으로 매칭
//! - **통일 (Unification)**: 양쪽에 변수가 있어도 가능한 양방향 매칭
//! - **백트래킹 (Backtracking)**: 프레임을 통해 여러 해 경로를 탐색
//! - **규칙 (Rules)**: 연역을 위한 논리적 함의
//!
//! # 아키텍처 (Architecture)
//!
//! ```text
//! 질의 (Query) → qeval → [프레임 (Frames)] → 결과 (Results)
//!   ↓
//! 데이터베이스 (Database) (단언+규칙 (Assertions + Rules))
//!   ↓
//! 통일 / 패턴 매칭 (Unification / Pattern Matching)
//!   ↓
//! 프레임 (변수 바인딩) (Frame (Variable Bindings))
//! ```

use std::collections::HashMap;
use std::fmt;

// ============================================================================
// 핵심 데이터 구조 (Core Data Structures)
// ============================================================================

/// 논리 프로그래밍 시스템에서의 항(term)을 나타낸다
/// (Represents a term in the logic programming system).
/// 항은 원자(상수), 변수, 또는 항의 리스트일 수 있다
/// (Terms can be atoms (constants), variables, or lists of terms).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    /// 원자 상수 값 (예: "ben", "computer", "60000")
    /// (Atomic constant value (e.g., "ben", "computer", "60000"))
    Atom(String),
    /// 패턴 변수 (예: ?x, ?person) (Pattern variable (e.g., ?x, ?person))
    Var(String),
    /// 항의 리스트 (예: (job ?x programmer))
    /// (List of terms (e.g., (job ?x programmer)))
    List(Vec<Term>),
}

impl Term {
    /// 원자 항 생성 (Create an atom term)
    pub fn atom(s: impl Into<String>) -> Self {
        Term::Atom(s.into())
    }

    /// 변수 항 생성 (Create a variable term)
    pub fn var(s: impl Into<String>) -> Self {
        Term::Var(s.into())
    }

    /// 리스트 항 생성 (Create a list term)
    pub fn list(terms: Vec<Term>) -> Self {
        Term::List(terms)
    }

    /// 항이 변수인지 확인 (Check if this term is a variable)
    pub fn is_var(&self) -> bool {
        matches!(self, Term::Var(_))
    }

    /// 변수가 맞다면 변수 이름 반환 (Get variable name if this is a variable)
    pub fn var_name(&self) -> Option<&str> {
        match self {
            Term::Var(name) => Some(name),
            _ => None,
        }
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Atom(s) => write!(f, "{}", s),
            Term::Var(v) => write!(f, "?{}", v),
            Term::List(terms) => {
                write!(f, "(")?;
                for (i, term) in terms.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", term)?;
                }
                write!(f, ")")
            }
        }
    }
}

/// 패턴 타입 별칭 (패턴은 변수를 포함한 항)
/// (Type alias for patterns (patterns are terms with variables))
pub type Pattern = Term;

/// 다양한 질의 타입을 나타낸다 (Represents different types of queries)
#[derive(Debug, Clone, PartialEq)]
pub enum Query {
    /// 단순 패턴 질의 (Simple pattern query)
    Simple(Pattern),
    /// 논리곱 (모두 성공해야 함) (Conjunction (all must succeed))
    And(Vec<Query>),
    /// 논리합 (하나 이상 성공) (Disjunction (at least one must succeed))
    Or(Vec<Query>),
    /// 부정 (실패해야 함) (Negation (must fail))
    Not(Box<Query>),
}

impl Query {
    pub fn simple(pattern: Pattern) -> Self {
        Query::Simple(pattern)
    }

    pub fn and(queries: Vec<Query>) -> Self {
        Query::And(queries)
    }

    pub fn or(queries: Vec<Query>) -> Self {
        Query::Or(queries)
    }

    pub fn negate(query: Query) -> Self {
        Query::Not(Box::new(query))
    }
}

/// 논리 규칙: conclusion :- body (Represents a logical rule: conclusion :- body)
#[derive(Debug, Clone)]
pub struct Rule {
    /// 규칙이 결론으로 내리는 패턴 (The pattern that this rule concludes)
    pub conclusion: Pattern,
    /// 규칙 적용에 필요한 질의 (The query that must be satisfied for this rule to apply)
    pub body: Query,
}

impl Rule {
    pub fn new(conclusion: Pattern, body: Query) -> Self {
        Rule { conclusion, body }
    }
}

/// 프레임: 변수에서 값으로의 매핑
/// (Frame: a mapping from variables to their values)
pub type Frame = HashMap<String, Term>;

// ============================================================================
// 데이터베이스 (Database)
// ============================================================================

/// 단언(사실)과 규칙을 담는 데이터베이스
/// (Database containing assertions (facts) and rules)
#[derive(Debug, Default)]
pub struct Database {
    assertions: Vec<Term>,
    rules: Vec<Rule>,
}

impl Database {
    pub fn new() -> Self {
        Database {
            assertions: Vec::new(),
            rules: Vec::new(),
        }
    }

    /// 단언(사실)을 데이터베이스에 추가 (Add an assertion (fact) to the database)
    pub fn add_assertion(&mut self, assertion: Term) {
        self.assertions.push(assertion);
    }

    /// 규칙을 데이터베이스에 추가 (Add a rule to the database)
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// 모든 단언 가져오기 (Get all assertions)
    pub fn assertions(&self) -> &[Term] {
        &self.assertions
    }

    /// 모든 규칙 가져오기 (Get all rules)
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }
}

// ============================================================================
// 통일 (Unification)
// ============================================================================

/// 프레임에 대해 두 항을 통일한다
/// (Unify two terms with respect to a frame).
/// 통일에 실패하면 None, 성공하면 확장된 프레임을 반환한다
/// (Returns None if unification fails, otherwise returns the extended frame).
///
/// 통일은 대칭적이며 양쪽에 변수 등장 가능
/// (Unification is symmetric - variables can appear on both sides).
pub fn unify(t1: &Term, t2: &Term, frame: &Frame) -> Option<Frame> {
    // 항이 같으면 현재 프레임으로 통일 성공
    // (If terms are equal, unification succeeds with current frame)
    if t1 == t2 {
        return Some(frame.clone());
    }

    match (t1, t2) {
        // 왼쪽이 변수: 프레임 확장 시도
        // (Variable on left: try to extend frame)
        (Term::Var(v1), _) => extend_if_possible(v1, t2, frame),

        // 오른쪽이 변수: 프레임 확장 시도
        // (Variable on right: try to extend frame)
        (_, Term::Var(v2)) => extend_if_possible(v2, t1, frame),

        // 둘 다 리스트면 원소별 통일
        // (Both are lists: unify element by element)
        (Term::List(l1), Term::List(l2)) if l1.len() == l2.len() => {
            let mut current_frame = frame.clone();
            for (term1, term2) in l1.iter().zip(l2.iter()) {
                match unify(term1, term2, &current_frame) {
                    Some(new_frame) => current_frame = new_frame,
                    None => return None,
                }
            }
            Some(current_frame)
        }

        // 그 외는 통일 실패 (Otherwise, unification fails)
        _ => None,
    }
}

/// 가능하다면 변수를 값에 바인딩해 프레임 확장
/// (Extend frame by binding variable to value if possible)
fn extend_if_possible(var: &str, val: &Term, frame: &Frame) -> Option<Frame> {
    // 변수가 이미 바인딩되어 있는지 확인
    // (Check if variable already has a binding)
    if let Some(binding) = frame.get(var) {
        // 이미 바인딩되어 있으면 기존 값과 새 값을 통일
        // (If bound, unify the bound value with the new value)
        return unify(binding, val, frame);
    }

    // val이 변수면 바인딩을 확인
    // (If val is a variable, check if it has a binding)
    if let Term::Var(v) = val
        && let Some(binding) = frame.get(v)
    {
        return unify(&Term::Var(var.to_string()), binding, frame);
    }

    // 순환 의존성 검사 (발생 검사)
    // (Check for circular dependency (occurs check))
    if depends_on(val, var, frame) {
        return None;
    }

    // 바인딩을 추가한 새 프레임 생성
    // (Create new frame with binding)
    let mut new_frame = frame.clone();
    new_frame.insert(var.to_string(), val.clone());
    Some(new_frame)
}

/// 표현식이 변수에 의존하는지 검사 (발생 검사)
/// (Check if expression depends on variable (occurs check))
fn depends_on(term: &Term, var: &str, frame: &Frame) -> bool {
    match term {
        Term::Var(v) => {
            if v == var {
                return true;
            }
            // 이 변수의 바인딩이 var에 의존하는지 확인
            // (Check if this variable's binding depends on var)
            if let Some(binding) = frame.get(v) {
                return depends_on(binding, var, frame);
            }
            false
        }
        Term::List(terms) => terms.iter().any(|t| depends_on(t, var, frame)),
        Term::Atom(_) => false,
    }
}

// ============================================================================
// 패턴 매칭 (Pattern Matching)
// ============================================================================

/// 패턴 매칭: 변수는 패턴에만 등장하는 단방향 매칭
/// (Pattern match: one-way matching where variables only appear in pattern).
/// 매칭 실패 시 None, 성공 시 확장된 프레임 반환
/// (Returns None if match fails, otherwise returns the extended frame).
pub fn pattern_match(pattern: &Term, data: &Term, frame: &Frame) -> Option<Frame> {
    // 패턴과 데이터가 같으면 매칭 성공
    // (If pattern and data are equal, matching succeeds)
    if pattern == data {
        return Some(frame.clone());
    }

    match (pattern, data) {
        // 패턴 쪽 변수: 프레임 확장
        // (Variable in pattern: extend frame)
        (Term::Var(v), _) => extend_if_consistent(v, data, frame),

        // 둘 다 리스트: 원소별 매칭
        // (Both are lists: match element by element)
        (Term::List(p_list), Term::List(d_list)) if p_list.len() == d_list.len() => {
            let mut current_frame = frame.clone();
            for (p, d) in p_list.iter().zip(d_list.iter()) {
                match pattern_match(p, d, &current_frame) {
                    Some(new_frame) => current_frame = new_frame,
                    None => return None,
                }
            }
            Some(current_frame)
        }

        // 그 외는 매칭 실패 (Otherwise, match fails)
        _ => None,
    }
}

/// 기존 바인딩과 일관되면 프레임 확장
/// (Extend frame if consistent with existing bindings)
fn extend_if_consistent(var: &str, data: &Term, frame: &Frame) -> Option<Frame> {
    if let Some(binding) = frame.get(var) {
        // 이미 바인딩된 변수: 일관성 확인
        // (Variable already bound: check consistency)
        pattern_match(binding, data, frame)
    } else {
        // 미바인딩 변수: 바인딩 추가
        // (Variable not bound: add binding)
        let mut new_frame = frame.clone();
        new_frame.insert(var.to_string(), data.clone());
        Some(new_frame)
    }
}

// ============================================================================
// 인스턴스화 (Instantiation)
// ============================================================================

/// 프레임의 값으로 변수를 치환해 항을 인스턴스화
/// (Instantiate a term by replacing variables with their values from the frame)
pub fn instantiate(term: &Term, frame: &Frame) -> Term {
    match term {
        Term::Var(v) => {
            if let Some(value) = frame.get(v) {
                // 값에 변수가 있을 수 있으므로 재귀 인스턴스화
                // (Recursively instantiate in case value contains variables)
                instantiate(value, frame)
            } else {
                term.clone()
            }
        }
        Term::List(terms) => Term::List(terms.iter().map(|t| instantiate(t, frame)).collect()),
        Term::Atom(_) => term.clone(),
    }
}

// ============================================================================
// 질의 평가 (Query Evaluation)
// ============================================================================

/// 데이터베이스에 대해 질의를 평가하고 프레임 벡터를 생성한다
/// (Evaluate a query against the database, producing a vector of frames).
/// 각 프레임은 질의의 하나의 해를 나타낸다
/// (Each frame represents a solution to the query).
pub fn qeval(query: &Query, frames: Vec<Frame>, db: &Database) -> Vec<Frame> {
    match query {
        Query::Simple(pattern) => simple_query(pattern, frames, db),
        Query::And(queries) => conjoin(queries, frames, db),
        Query::Or(queries) => disjoin(queries, frames, db),
        Query::Not(query) => negate(query, frames, db),
    }
}

/// 단순 질의를 처리 (패턴 매칭)
/// (Process a simple query (pattern matching))
fn simple_query(pattern: &Pattern, frames: Vec<Frame>, db: &Database) -> Vec<Frame> {
    frames
        .into_iter()
        .flat_map(|frame| {
            let mut results = Vec::new();

            // 모든 단언과 매칭 시도
            // (Try to match against all assertions)
            for assertion in db.assertions() {
                if let Some(new_frame) = pattern_match(pattern, assertion, &frame) {
                    results.push(new_frame);
                }
            }

            // 모든 규칙 적용 시도 (Try to apply all rules)
            results.extend(apply_rules(pattern, &frame, db));

            results
        })
        .collect()
}

/// 적용 가능한 모든 규칙 적용
/// (Apply all applicable rules)
fn apply_rules(pattern: &Pattern, frame: &Frame, db: &Database) -> Vec<Frame> {
    let mut results = Vec::new();

    for rule in db.rules() {
        // 규칙 내 변수를 이름 변경해 충돌 회피
        // (Rename variables in rule to avoid conflicts)
        let (renamed_conclusion, renamed_body) = rename_rule_variables(rule);

        // 패턴과 규칙 결론 통일 시도
        // (Try to unify pattern with rule conclusion)
        if let Some(unified_frame) = unify(pattern, &renamed_conclusion, frame) {
            // 통일된 프레임에서 규칙 본문 평가
            // (Evaluate rule body in the unified frame)
            let body_results = qeval(&renamed_body, vec![unified_frame], db);
            results.extend(body_results);
        }
    }

    results
}

/// 규칙 내 변수를 이름 변경해 충돌 회피
/// (Rename variables in a rule to avoid conflicts)
/// 실제 구현이라면 카운터로 고유 이름을 만든다
/// (In a real implementation, we'd use a counter to generate unique names)
fn rename_rule_variables(rule: &Rule) -> (Pattern, Query) {
    // 단순화를 위해 기본적인 이름 변경을 사용
    // (For simplicity, we'll use a basic renaming scheme)
    // 실제 시스템은 고유 ID 생성기를 사용해야 함
    // (A production system would use a unique ID generator)
    static mut RULE_COUNTER: usize = 0;

    let id = unsafe {
        RULE_COUNTER += 1;
        RULE_COUNTER
    };

    (
        rename_term_variables(&rule.conclusion, id),
        rename_query_variables(&rule.body, id),
    )
}

fn rename_term_variables(term: &Term, id: usize) -> Term {
    match term {
        Term::Var(v) => Term::Var(format!("{}_r{}", v, id)),
        Term::List(terms) => {
            Term::List(terms.iter().map(|t| rename_term_variables(t, id)).collect())
        }
        Term::Atom(_) => term.clone(),
    }
}

fn rename_query_variables(query: &Query, id: usize) -> Query {
    match query {
        Query::Simple(pattern) => Query::Simple(rename_term_variables(pattern, id)),
        Query::And(queries) => Query::And(
            queries
                .iter()
                .map(|q| rename_query_variables(q, id))
                .collect(),
        ),
        Query::Or(queries) => Query::Or(
            queries
                .iter()
                .map(|q| rename_query_variables(q, id))
                .collect(),
        ),
        Query::Not(query) => Query::Not(Box::new(rename_query_variables(query, id))),
    }
}

/// 논리곱 처리 (AND) (Process conjunction (AND))
fn conjoin(queries: &[Query], frames: Vec<Frame>, db: &Database) -> Vec<Frame> {
    if queries.is_empty() {
        return frames;
    }

    // 첫 질의를 처리한 뒤 나머지를 재귀 처리
    // (Process first query, then recursively process rest)
    let first_results = qeval(&queries[0], frames, db);
    conjoin(&queries[1..], first_results, db)
}

/// 논리합 처리 (OR) (Process disjunction (OR))
fn disjoin(queries: &[Query], frames: Vec<Frame>, db: &Database) -> Vec<Frame> {
    queries
        .iter()
        .flat_map(|query| qeval(query, frames.clone(), db))
        .collect()
}

/// 부정 처리 (NOT) (Process negation (NOT))
fn negate(query: &Query, frames: Vec<Frame>, db: &Database) -> Vec<Frame> {
    frames
        .into_iter()
        .filter(|frame| {
            // 질의가 실패하면 NOT 필터를 통과
            // (A frame passes the NOT filter if the query fails on it)
            qeval(query, vec![frame.clone()], db).is_empty()
        })
        .collect()
}

// ============================================================================
// 헬퍼 함수 (Helper Functions)
// ============================================================================

/// 질의를 실행하고 인스턴스화된 결과를 반환
/// (Execute a query and return instantiated results)
pub fn query(query: &Query, db: &Database) -> Vec<Term> {
    let initial_frame = HashMap::new();
    let result_frames = qeval(query, vec![initial_frame], db);

    // 단순 질의는 패턴을 인스턴스화
    // (For simple queries, instantiate the pattern)
    match query {
        Query::Simple(pattern) => result_frames
            .iter()
            .map(|frame| instantiate(pattern, frame))
            .collect(),
        _ => {
            // 복합 질의는 어떤 패턴을 인스턴스화할지 추적이 필요함
            // (For complex queries, we'd need to track which pattern to instantiate)
            // 현재는 빈 벡터 반환
            // (For now, return empty)
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // 통일 테스트 (Unification Tests)
    // ========================================================================

    #[test]
    fn test_unify_atoms() {
        let frame = HashMap::new();
        let t1 = Term::atom("ben");
        let t2 = Term::atom("ben");

        assert!(unify(&t1, &t2, &frame).is_some());

        let t3 = Term::atom("alyssa");
        assert!(unify(&t1, &t3, &frame).is_none());
    }

    #[test]
    fn test_unify_variable_with_atom() {
        let frame = HashMap::new();
        let var = Term::var("x");
        let atom = Term::atom("ben");

        let result = unify(&var, &atom, &frame).unwrap();
        assert_eq!(result.get("x"), Some(&atom));
    }

    #[test]
    fn test_unify_two_variables() {
        let frame = HashMap::new();
        let var1 = Term::var("x");
        let var2 = Term::var("y");

        let result = unify(&var1, &var2, &frame).unwrap();
        // x->y 또는 y->x 바인딩 모두 유효
        // (Either x->y or y->x binding is valid)
        assert!(result.contains_key("x") || result.contains_key("y"));
    }

    #[test]
    fn test_unify_lists() {
        let frame = HashMap::new();
        let list1 = Term::list(vec![
            Term::atom("job"),
            Term::var("x"),
            Term::atom("programmer"),
        ]);
        let list2 = Term::list(vec![
            Term::atom("job"),
            Term::atom("ben"),
            Term::atom("programmer"),
        ]);

        let result = unify(&list1, &list2, &frame).unwrap();
        assert_eq!(result.get("x"), Some(&Term::atom("ben")));
    }

    #[test]
    fn test_unify_occurs_check() {
        let frame = HashMap::new();
        let var = Term::var("x");
        let recursive = Term::list(vec![Term::atom("f"), Term::var("x")]);

        // 발생 검사로 실패해야 함 (Should fail due to occurs check)
        assert!(unify(&var, &recursive, &frame).is_none());
    }

    // ========================================================================
    // 패턴 매칭 테스트 (Pattern Matching Tests)
    // ========================================================================

    #[test]
    fn test_pattern_match_simple() {
        let frame = HashMap::new();
        let pattern = Term::list(vec![
            Term::atom("job"),
            Term::var("person"),
            Term::atom("programmer"),
        ]);
        let data = Term::list(vec![
            Term::atom("job"),
            Term::atom("alyssa"),
            Term::atom("programmer"),
        ]);

        let result = pattern_match(&pattern, &data, &frame).unwrap();
        assert_eq!(result.get("person"), Some(&Term::atom("alyssa")));
    }

    #[test]
    fn test_pattern_match_consistency() {
        let mut frame = HashMap::new();
        frame.insert("x".to_string(), Term::atom("ben"));

        let pattern = Term::var("x");
        let data1 = Term::atom("ben");
        let data2 = Term::atom("alyssa");

        assert!(pattern_match(&pattern, &data1, &frame).is_some());
        assert!(pattern_match(&pattern, &data2, &frame).is_none());
    }

    // ========================================================================
    // 데이터베이스 및 질의 테스트 (Database and Query Tests)
    // ========================================================================

    fn create_microshaft_db() -> Database {
        let mut db = Database::new();

        // job 단언 추가 (Add job assertions)
        db.add_assertion(Term::list(vec![
            Term::atom("job"),
            Term::list(vec![Term::atom("bitdiddle"), Term::atom("ben")]),
            Term::list(vec![Term::atom("computer"), Term::atom("wizard")]),
        ]));

        db.add_assertion(Term::list(vec![
            Term::atom("job"),
            Term::list(vec![Term::atom("hacker"), Term::atom("alyssa")]),
            Term::list(vec![Term::atom("computer"), Term::atom("programmer")]),
        ]));

        db.add_assertion(Term::list(vec![
            Term::atom("job"),
            Term::list(vec![Term::atom("fect"), Term::atom("cy")]),
            Term::list(vec![Term::atom("computer"), Term::atom("programmer")]),
        ]));

        // salary 단언 추가 (Add salary assertions)
        db.add_assertion(Term::list(vec![
            Term::atom("salary"),
            Term::list(vec![Term::atom("bitdiddle"), Term::atom("ben")]),
            Term::atom("60000"),
        ]));

        db.add_assertion(Term::list(vec![
            Term::atom("salary"),
            Term::list(vec![Term::atom("hacker"), Term::atom("alyssa")]),
            Term::atom("40000"),
        ]));

        db.add_assertion(Term::list(vec![
            Term::atom("salary"),
            Term::list(vec![Term::atom("fect"), Term::atom("cy")]),
            Term::atom("35000"),
        ]));

        // supervisor 단언 추가 (Add supervisor assertions)
        db.add_assertion(Term::list(vec![
            Term::atom("supervisor"),
            Term::list(vec![Term::atom("hacker"), Term::atom("alyssa")]),
            Term::list(vec![Term::atom("bitdiddle"), Term::atom("ben")]),
        ]));

        db.add_assertion(Term::list(vec![
            Term::atom("supervisor"),
            Term::list(vec![Term::atom("fect"), Term::atom("cy")]),
            Term::list(vec![Term::atom("bitdiddle"), Term::atom("ben")]),
        ]));

        db
    }

    #[test]
    fn test_simple_query() {
        let db = create_microshaft_db();

        // 질의: (job ?x (computer programmer)) (Query)
        let q = Query::simple(Term::list(vec![
            Term::atom("job"),
            Term::var("x"),
            Term::list(vec![Term::atom("computer"), Term::atom("programmer")]),
        ]));

        let results = query(&q, &db);
        assert_eq!(results.len(), 2); // Alyssa와 Cy (Alyssa and Cy)
    }

    #[test]
    fn test_and_query() {
        let db = create_microshaft_db();

        // 질의: (and (job ?x (computer programmer))
        //             (salary ?x ?amount))
        let query = Query::and(vec![
            Query::simple(Term::list(vec![
                Term::atom("job"),
                Term::var("x"),
                Term::list(vec![Term::atom("computer"), Term::atom("programmer")]),
            ])),
            Query::simple(Term::list(vec![
                Term::atom("salary"),
                Term::var("x"),
                Term::var("amount"),
            ])),
        ]);

        let initial_frame = HashMap::new();
        let results = qeval(&query, vec![initial_frame], &db);
        assert_eq!(results.len(), 2); // Alyssa와 Cy (Alyssa and Cy)
    }

    #[test]
    fn test_rule_application() {
        let mut db = Database::new();

        // 사실 추가 (Add facts)
        db.add_assertion(Term::list(vec![
            Term::atom("parent"),
            Term::atom("adam"),
            Term::atom("cain"),
        ]));

        db.add_assertion(Term::list(vec![
            Term::atom("parent"),
            Term::atom("cain"),
            Term::atom("enoch"),
        ]));

        // 규칙 추가: (grandparent ?g ?gc) :- (and (parent ?g ?p) (parent ?p ?gc))
        db.add_rule(Rule::new(
            Term::list(vec![
                Term::atom("grandparent"),
                Term::var("g"),
                Term::var("gc"),
            ]),
            Query::and(vec![
                Query::simple(Term::list(vec![
                    Term::atom("parent"),
                    Term::var("g"),
                    Term::var("p"),
                ])),
                Query::simple(Term::list(vec![
                    Term::atom("parent"),
                    Term::var("p"),
                    Term::var("gc"),
                ])),
            ]),
        ));

        // 질의: (grandparent ?x ?y) (Query)
        let q = Query::simple(Term::list(vec![
            Term::atom("grandparent"),
            Term::var("x"),
            Term::var("y"),
        ]));

        let results = query(&q, &db);
        assert_eq!(results.len(), 1); // Adam은 Enoch의 조부모 (Adam is grandparent of Enoch)
    }

    #[test]
    fn test_not_query() {
        let db = create_microshaft_db();

        // 질의: (and (job ?x ?job)
        //             (not (job ?x (computer programmer))))
        let query = Query::and(vec![
            Query::simple(Term::list(vec![
                Term::atom("job"),
                Term::var("x"),
                Term::var("job"),
            ])),
            Query::negate(Query::simple(Term::list(vec![
                Term::atom("job"),
                Term::var("x"),
                Term::list(vec![Term::atom("computer"), Term::atom("programmer")]),
            ]))),
        ]);

        let initial_frame = HashMap::new();
        let results = qeval(&query, vec![initial_frame], &db);
        assert_eq!(results.len(), 1); // Ben만 해당 (마법사) (Only Ben (the wizard))
    }

    #[test]
    fn test_instantiate() {
        let mut frame = HashMap::new();
        frame.insert("x".to_string(), Term::atom("ben"));
        frame.insert("y".to_string(), Term::atom("programmer"));

        let pattern = Term::list(vec![Term::atom("job"), Term::var("x"), Term::var("y")]);

        let result = instantiate(&pattern, &frame);
        assert_eq!(
            result,
            Term::list(vec![
                Term::atom("job"),
                Term::atom("ben"),
                Term::atom("programmer"),
            ])
        );
    }
}
