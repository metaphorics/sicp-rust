//! SICP Section 4.4: Logic Programming
//!
//! This module implements a query language system similar to Prolog,
//! demonstrating pattern matching, unification, and logical inference.
//!
//! # Core Concepts
//!
//! - **Pattern Matching**: One-way matching of patterns against data
//! - **Unification**: Two-way matching allowing variables on both sides
//! - **Backtracking**: Exploring multiple solution paths via frames
//! - **Rules**: Logical implications for deductive reasoning
//!
//! # Architecture
//!
//! ```text
//! Query → qeval → [Frames] → Results
//!   ↓
//! Database (Assertions + Rules)
//!   ↓
//! Unification / Pattern Matching
//!   ↓
//! Frame (Variable Bindings)
//! ```

use std::collections::HashMap;
use std::fmt;

// ============================================================================
// Core Data Structures
// ============================================================================

/// Represents a term in the logic programming system.
/// Terms can be atoms (constants), variables, or lists of terms.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    /// Atomic constant value (e.g., "ben", "computer", "60000")
    Atom(String),
    /// Pattern variable (e.g., ?x, ?person)
    Var(String),
    /// List of terms (e.g., (job ?x programmer))
    List(Vec<Term>),
}

impl Term {
    /// Create an atom term
    pub fn atom(s: impl Into<String>) -> Self {
        Term::Atom(s.into())
    }

    /// Create a variable term
    pub fn var(s: impl Into<String>) -> Self {
        Term::Var(s.into())
    }

    /// Create a list term
    pub fn list(terms: Vec<Term>) -> Self {
        Term::List(terms)
    }

    /// Check if this term is a variable
    pub fn is_var(&self) -> bool {
        matches!(self, Term::Var(_))
    }

    /// Get variable name if this is a variable
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

/// Type alias for patterns (patterns are terms with variables)
pub type Pattern = Term;

/// Represents different types of queries
#[derive(Debug, Clone, PartialEq)]
pub enum Query {
    /// Simple pattern query
    Simple(Pattern),
    /// Conjunction (all must succeed)
    And(Vec<Query>),
    /// Disjunction (at least one must succeed)
    Or(Vec<Query>),
    /// Negation (must fail)
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

/// Represents a logical rule: conclusion :- body
#[derive(Debug, Clone)]
pub struct Rule {
    /// The pattern that this rule concludes
    pub conclusion: Pattern,
    /// The query that must be satisfied for this rule to apply
    pub body: Query,
}

impl Rule {
    pub fn new(conclusion: Pattern, body: Query) -> Self {
        Rule { conclusion, body }
    }
}

/// Frame: a mapping from variables to their values
pub type Frame = HashMap<String, Term>;

// ============================================================================
// Database
// ============================================================================

/// Database containing assertions (facts) and rules
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

    /// Add an assertion (fact) to the database
    pub fn add_assertion(&mut self, assertion: Term) {
        self.assertions.push(assertion);
    }

    /// Add a rule to the database
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Get all assertions
    pub fn assertions(&self) -> &[Term] {
        &self.assertions
    }

    /// Get all rules
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }
}

// ============================================================================
// Unification
// ============================================================================

/// Unify two terms with respect to a frame.
/// Returns None if unification fails, otherwise returns the extended frame.
///
/// Unification is symmetric - variables can appear on both sides.
pub fn unify(t1: &Term, t2: &Term, frame: &Frame) -> Option<Frame> {
    // If terms are equal, unification succeeds with current frame
    if t1 == t2 {
        return Some(frame.clone());
    }

    match (t1, t2) {
        // Variable on left: try to extend frame
        (Term::Var(v1), _) => extend_if_possible(v1, t2, frame),

        // Variable on right: try to extend frame
        (_, Term::Var(v2)) => extend_if_possible(v2, t1, frame),

        // Both are lists: unify element by element
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

        // Otherwise, unification fails
        _ => None,
    }
}

/// Extend frame by binding variable to value if possible
fn extend_if_possible(var: &str, val: &Term, frame: &Frame) -> Option<Frame> {
    // Check if variable already has a binding
    if let Some(binding) = frame.get(var) {
        // If bound, unify the bound value with the new value
        return unify(binding, val, frame);
    }

    // If val is a variable, check if it has a binding
    if let Term::Var(v) = val
        && let Some(binding) = frame.get(v)
    {
        return unify(&Term::Var(var.to_string()), binding, frame);
    }

    // Check for circular dependency (occurs check)
    if depends_on(val, var, frame) {
        return None;
    }

    // Create new frame with binding
    let mut new_frame = frame.clone();
    new_frame.insert(var.to_string(), val.clone());
    Some(new_frame)
}

/// Check if expression depends on variable (occurs check)
fn depends_on(term: &Term, var: &str, frame: &Frame) -> bool {
    match term {
        Term::Var(v) => {
            if v == var {
                return true;
            }
            // Check if this variable's binding depends on var
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
// Pattern Matching
// ============================================================================

/// Pattern match: one-way matching where variables only appear in pattern.
/// Returns None if match fails, otherwise returns the extended frame.
pub fn pattern_match(pattern: &Term, data: &Term, frame: &Frame) -> Option<Frame> {
    // If pattern and data are equal, matching succeeds
    if pattern == data {
        return Some(frame.clone());
    }

    match (pattern, data) {
        // Variable in pattern: extend frame
        (Term::Var(v), _) => extend_if_consistent(v, data, frame),

        // Both are lists: match element by element
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

        // Otherwise, match fails
        _ => None,
    }
}

/// Extend frame if consistent with existing bindings
fn extend_if_consistent(var: &str, data: &Term, frame: &Frame) -> Option<Frame> {
    if let Some(binding) = frame.get(var) {
        // Variable already bound: check consistency
        pattern_match(binding, data, frame)
    } else {
        // Variable not bound: add binding
        let mut new_frame = frame.clone();
        new_frame.insert(var.to_string(), data.clone());
        Some(new_frame)
    }
}

// ============================================================================
// Instantiation
// ============================================================================

/// Instantiate a term by replacing variables with their values from the frame
pub fn instantiate(term: &Term, frame: &Frame) -> Term {
    match term {
        Term::Var(v) => {
            if let Some(value) = frame.get(v) {
                // Recursively instantiate in case value contains variables
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
// Query Evaluation
// ============================================================================

/// Evaluate a query against the database, producing a vector of frames.
/// Each frame represents a solution to the query.
pub fn qeval(query: &Query, frames: Vec<Frame>, db: &Database) -> Vec<Frame> {
    match query {
        Query::Simple(pattern) => simple_query(pattern, frames, db),
        Query::And(queries) => conjoin(queries, frames, db),
        Query::Or(queries) => disjoin(queries, frames, db),
        Query::Not(query) => negate(query, frames, db),
    }
}

/// Process a simple query (pattern matching)
fn simple_query(pattern: &Pattern, frames: Vec<Frame>, db: &Database) -> Vec<Frame> {
    frames
        .into_iter()
        .flat_map(|frame| {
            let mut results = Vec::new();

            // Try to match against all assertions
            for assertion in db.assertions() {
                if let Some(new_frame) = pattern_match(pattern, assertion, &frame) {
                    results.push(new_frame);
                }
            }

            // Try to apply all rules
            results.extend(apply_rules(pattern, &frame, db));

            results
        })
        .collect()
}

/// Apply all applicable rules
fn apply_rules(pattern: &Pattern, frame: &Frame, db: &Database) -> Vec<Frame> {
    let mut results = Vec::new();

    for rule in db.rules() {
        // Rename variables in rule to avoid conflicts
        let (renamed_conclusion, renamed_body) = rename_rule_variables(rule);

        // Try to unify pattern with rule conclusion
        if let Some(unified_frame) = unify(pattern, &renamed_conclusion, frame) {
            // Evaluate rule body in the unified frame
            let body_results = qeval(&renamed_body, vec![unified_frame], db);
            results.extend(body_results);
        }
    }

    results
}

/// Rename variables in a rule to avoid conflicts
/// In a real implementation, we'd use a counter to generate unique names
fn rename_rule_variables(rule: &Rule) -> (Pattern, Query) {
    // For simplicity, we'll use a basic renaming scheme
    // A production system would use a unique ID generator
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

/// Process conjunction (AND)
fn conjoin(queries: &[Query], frames: Vec<Frame>, db: &Database) -> Vec<Frame> {
    if queries.is_empty() {
        return frames;
    }

    // Process first query, then recursively process rest
    let first_results = qeval(&queries[0], frames, db);
    conjoin(&queries[1..], first_results, db)
}

/// Process disjunction (OR)
fn disjoin(queries: &[Query], frames: Vec<Frame>, db: &Database) -> Vec<Frame> {
    queries
        .iter()
        .flat_map(|query| qeval(query, frames.clone(), db))
        .collect()
}

/// Process negation (NOT)
fn negate(query: &Query, frames: Vec<Frame>, db: &Database) -> Vec<Frame> {
    frames
        .into_iter()
        .filter(|frame| {
            // A frame passes the NOT filter if the query fails on it
            qeval(query, vec![frame.clone()], db).is_empty()
        })
        .collect()
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Execute a query and return instantiated results
pub fn query(query: &Query, db: &Database) -> Vec<Term> {
    let initial_frame = HashMap::new();
    let result_frames = qeval(query, vec![initial_frame], db);

    // For simple queries, instantiate the pattern
    match query {
        Query::Simple(pattern) => result_frames
            .iter()
            .map(|frame| instantiate(pattern, frame))
            .collect(),
        _ => {
            // For complex queries, we'd need to track which pattern to instantiate
            // For now, return empty
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Unification Tests
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
        // Either x->y or y->x binding is valid
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

        // Should fail due to occurs check
        assert!(unify(&var, &recursive, &frame).is_none());
    }

    // ========================================================================
    // Pattern Matching Tests
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
    // Database and Query Tests
    // ========================================================================

    fn create_microshaft_db() -> Database {
        let mut db = Database::new();

        // Add job assertions
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

        // Add salary assertions
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

        // Add supervisor assertions
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

        // Query: (job ?x (computer programmer))
        let q = Query::simple(Term::list(vec![
            Term::atom("job"),
            Term::var("x"),
            Term::list(vec![Term::atom("computer"), Term::atom("programmer")]),
        ]));

        let results = query(&q, &db);
        assert_eq!(results.len(), 2); // Alyssa and Cy
    }

    #[test]
    fn test_and_query() {
        let db = create_microshaft_db();

        // Query: (and (job ?x (computer programmer))
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
        assert_eq!(results.len(), 2); // Alyssa and Cy
    }

    #[test]
    fn test_rule_application() {
        let mut db = Database::new();

        // Add facts
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

        // Add rule: (grandparent ?g ?gc) :- (and (parent ?g ?p) (parent ?p ?gc))
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

        // Query: (grandparent ?x ?y)
        let q = Query::simple(Term::list(vec![
            Term::atom("grandparent"),
            Term::var("x"),
            Term::var("y"),
        ]));

        let results = query(&q, &db);
        assert_eq!(results.len(), 1); // Adam is grandparent of Enoch
    }

    #[test]
    fn test_not_query() {
        let db = create_microshaft_db();

        // Query: (and (job ?x ?job)
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
        assert_eq!(results.len(), 1); // Only Ben (the wizard)
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
