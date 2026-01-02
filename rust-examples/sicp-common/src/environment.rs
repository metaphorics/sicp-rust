//! # Persistent Environments
//!
//! Functional environments using `im::HashMap` for O(1) clone with structural sharing.
//! This replaces `Rc<RefCell<HashMap<...>>>` patterns with pure functional semantics.
//!
//! ## Benefits over `Rc<RefCell<Environment>>`
//!
//! - No runtime borrow checking
//! - Thread-safe (`Send + Sync`)
//! - Functional updates return new environments
//! - O(log n) operations with structural sharing (O(1) amortized for many operations)
//! - Closures can own their environment without reference counting
//!
//! ## Example
//!
//! ```
//! use sicp_common::environment::Environment;
//!
//! // Create an empty environment
//! let env: Environment<i64> = Environment::new();
//!
//! // Define returns a NEW environment (functional update)
//! let env = env.define("x".to_string(), 10);
//! let env = env.define("y".to_string(), 20);
//!
//! assert_eq!(env.lookup("x"), Some(&10));
//! assert_eq!(env.lookup("y"), Some(&20));
//!
//! // Extend creates a child environment
//! let child = env.extend([("z".to_string(), 30)]);
//! assert_eq!(child.lookup("z"), Some(&30));
//! assert_eq!(child.lookup("x"), Some(&10)); // Inherits from parent
//! ```

use im::HashMap as ImHashMap;

/// A persistent, immutable environment for variable bindings.
///
/// Environments form a chain where each environment can have a parent.
/// Lookups traverse the chain from child to parent until the binding is found.
///
/// ## Functional Semantics
///
/// All mutating operations return a new `Environment` rather than modifying
/// in place. This is efficient due to structural sharing provided by `im::HashMap`.
///
/// ## Memory Efficiency
///
/// Thanks to structural sharing, creating a child environment or adding
/// a binding only allocates memory for the new/changed parts. The unchanged
/// portions are shared with the original environment.
#[derive(Debug, Clone)]
pub struct Environment<V> {
    /// Current frame's bindings
    bindings: ImHashMap<String, V>,
    /// Parent environment (if any)
    parent: Option<Box<Environment<V>>>,
}

impl<V> Default for Environment<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> Environment<V> {
    /// Creates a new empty environment with no parent.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bindings: ImHashMap::new(),
            parent: None,
        }
    }

    /// Creates a new environment with the given parent.
    #[must_use]
    pub fn with_parent(parent: Environment<V>) -> Self {
        Self {
            bindings: ImHashMap::new(),
            parent: Some(Box::new(parent)),
        }
    }

    /// Returns the number of bindings in this frame (not including parent).
    #[must_use]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Returns true if this frame has no bindings (parent may have bindings).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Returns true if this environment has a parent.
    #[must_use]
    pub fn has_parent(&self) -> bool {
        self.parent.is_some()
    }

    /// Returns a reference to the parent environment, if any.
    #[must_use]
    pub fn parent(&self) -> Option<&Environment<V>> {
        self.parent.as_deref()
    }
}

impl<V: Clone> Environment<V> {
    /// Looks up a variable in this environment and its parents.
    ///
    /// Returns `Some(&value)` if found, `None` if not bound.
    ///
    /// # Example
    ///
    /// ```
    /// use sicp_common::environment::Environment;
    ///
    /// let env = Environment::new()
    ///     .define("x".to_string(), 42);
    ///
    /// assert_eq!(env.lookup("x"), Some(&42));
    /// assert_eq!(env.lookup("y"), None);
    /// ```
    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<&V> {
        self.bindings
            .get(name)
            .or_else(|| self.parent.as_ref().and_then(|p| p.lookup(name)))
    }

    /// Defines a new binding in this environment, returning a new environment.
    ///
    /// If the name already exists in this frame, it is shadowed.
    /// Parent bindings are not affected.
    ///
    /// # Example
    ///
    /// ```
    /// use sicp_common::environment::Environment;
    ///
    /// let env1 = Environment::new().define("x".to_string(), 1);
    /// let env2 = env1.define("x".to_string(), 2);
    ///
    /// // env1 is unchanged (functional update)
    /// assert_eq!(env1.lookup("x"), Some(&1));
    /// assert_eq!(env2.lookup("x"), Some(&2));
    /// ```
    #[must_use]
    pub fn define(&self, name: String, value: V) -> Self {
        Self {
            bindings: self.bindings.update(name, value),
            parent: self.parent.clone(),
        }
    }

    /// Creates a new child environment with the given bindings.
    ///
    /// This is typically used when entering a new scope (e.g., function call).
    ///
    /// # Example
    ///
    /// ```
    /// use sicp_common::environment::Environment;
    ///
    /// let global = Environment::new()
    ///     .define("x".to_string(), 10);
    ///
    /// let local = global.extend([
    ///     ("y".to_string(), 20),
    ///     ("z".to_string(), 30),
    /// ]);
    ///
    /// assert_eq!(local.lookup("x"), Some(&10)); // From parent
    /// assert_eq!(local.lookup("y"), Some(&20)); // Local
    /// assert_eq!(local.lookup("z"), Some(&30)); // Local
    /// ```
    #[must_use]
    pub fn extend(&self, bindings: impl IntoIterator<Item = (String, V)>) -> Self {
        Self {
            bindings: bindings.into_iter().collect(),
            parent: Some(Box::new(self.clone())),
        }
    }

    /// Extends the current frame with additional bindings (same scope level).
    ///
    /// Unlike `extend`, this adds to the current frame rather than creating a child.
    ///
    /// # Example
    ///
    /// ```
    /// use sicp_common::environment::Environment;
    ///
    /// let env = Environment::new()
    ///     .define("x".to_string(), 1)
    ///     .define_all([
    ///         ("y".to_string(), 2),
    ///         ("z".to_string(), 3),
    ///     ]);
    ///
    /// assert_eq!(env.lookup("x"), Some(&1));
    /// assert_eq!(env.lookup("y"), Some(&2));
    /// assert_eq!(env.lookup("z"), Some(&3));
    /// ```
    #[must_use]
    pub fn define_all(&self, bindings: impl IntoIterator<Item = (String, V)>) -> Self {
        let mut new_bindings = self.bindings.clone();
        for (name, value) in bindings {
            new_bindings = new_bindings.update(name, value);
        }
        Self {
            bindings: new_bindings,
            parent: self.parent.clone(),
        }
    }

    /// Checks if a name is bound in this environment (including parents).
    #[must_use]
    pub fn is_bound(&self, name: &str) -> bool {
        self.lookup(name).is_some()
    }

    /// Checks if a name is bound in this frame only (not checking parents).
    #[must_use]
    pub fn is_bound_locally(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }

    /// Returns an iterator over bindings in the current frame.
    pub fn iter_local(&self) -> impl Iterator<Item = (&String, &V)> {
        self.bindings.iter()
    }

    /// Collects all bindings from this environment and its parents.
    ///
    /// In case of shadowing, returns the innermost binding.
    #[must_use]
    pub fn all_bindings(&self) -> ImHashMap<String, V> {
        let parent_bindings = self
            .parent
            .as_ref()
            .map(|p| p.all_bindings())
            .unwrap_or_default();

        // Child bindings override parent bindings
        let mut result = parent_bindings;
        for (k, v) in &self.bindings {
            result = result.update(k.clone(), v.clone());
        }
        result
    }

    /// Returns the depth of this environment (number of frames including this one).
    #[must_use]
    pub fn depth(&self) -> usize {
        1 + self.parent.as_ref().map(|p| p.depth()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_environment() {
        let env: Environment<i64> = Environment::new();
        assert!(env.is_empty());
        assert!(!env.has_parent());
        assert_eq!(env.lookup("x"), None);
    }

    #[test]
    fn test_define_and_lookup() {
        let env = Environment::new()
            .define("x".to_string(), 10)
            .define("y".to_string(), 20);

        assert_eq!(env.lookup("x"), Some(&10));
        assert_eq!(env.lookup("y"), Some(&20));
        assert_eq!(env.lookup("z"), None);
    }

    #[test]
    fn test_functional_update() {
        let env1 = Environment::new().define("x".to_string(), 1);
        let env2 = env1.define("x".to_string(), 2);

        // Original unchanged
        assert_eq!(env1.lookup("x"), Some(&1));
        // New has updated value
        assert_eq!(env2.lookup("x"), Some(&2));
    }

    #[test]
    fn test_extend_creates_child() {
        let parent = Environment::new().define("x".to_string(), 10);

        let child = parent.extend([("y".to_string(), 20), ("z".to_string(), 30)]);

        // Child can see parent bindings
        assert_eq!(child.lookup("x"), Some(&10));
        assert_eq!(child.lookup("y"), Some(&20));
        assert_eq!(child.lookup("z"), Some(&30));

        // Parent doesn't see child bindings
        assert_eq!(parent.lookup("y"), None);
    }

    #[test]
    fn test_shadowing() {
        let outer = Environment::new().define("x".to_string(), 10);

        let inner = outer.extend([("x".to_string(), 20)]);

        // Inner shadows outer
        assert_eq!(inner.lookup("x"), Some(&20));
        // Outer unchanged
        assert_eq!(outer.lookup("x"), Some(&10));
    }

    #[test]
    fn test_depth() {
        let env1 = Environment::<i64>::new();
        assert_eq!(env1.depth(), 1);

        let env2 = env1.extend([("x".to_string(), 1)]);
        assert_eq!(env2.depth(), 2);

        let env3 = env2.extend([("y".to_string(), 2)]);
        assert_eq!(env3.depth(), 3);
    }

    #[test]
    fn test_is_bound_locally() {
        let parent = Environment::new().define("x".to_string(), 10);
        let child = parent.extend([("y".to_string(), 20)]);

        assert!(child.is_bound("x")); // In parent
        assert!(!child.is_bound_locally("x")); // Not in this frame

        assert!(child.is_bound("y")); // In this frame
        assert!(child.is_bound_locally("y")); // In this frame
    }

    #[test]
    fn test_all_bindings() {
        let env = Environment::new()
            .define("x".to_string(), 1)
            .extend([("y".to_string(), 2), ("x".to_string(), 3)]); // x shadows

        let all = env.all_bindings();
        assert_eq!(all.get("x"), Some(&3)); // Shadowed value
        assert_eq!(all.get("y"), Some(&2));
    }

    #[test]
    fn test_define_all() {
        let env = Environment::new()
            .define("a".to_string(), 1)
            .define_all([("b".to_string(), 2), ("c".to_string(), 3)]);

        assert_eq!(env.lookup("a"), Some(&1));
        assert_eq!(env.lookup("b"), Some(&2));
        assert_eq!(env.lookup("c"), Some(&3));
        assert_eq!(env.depth(), 1); // All in same frame
    }
}
