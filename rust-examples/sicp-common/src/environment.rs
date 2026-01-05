//! # 영속 환경 (Persistent Environments)
//!
//! 구조적 공유를 위한 `im::HashMap`을 사용해 O(1) 복제를 제공하는 함수형 환경
//! (Functional environments using `im::HashMap` for O(1) clone with structural sharing).
//! 이는 `Rc<RefCell<HashMap<...>>>` 패턴을 순수 함수형 의미론으로 대체한다
//! (This replaces `Rc<RefCell<HashMap<...>>>` patterns with pure functional semantics).
//!
//! ## `Rc<RefCell<Environment>>` 대비 장점 (Benefits over `Rc<RefCell<Environment>>`)
//!
//! - 런타임 빌림 검사 없음 (No runtime borrow checking)
//! - 스레드 안전 (`Send + Sync`) (Thread-safe (`Send + Sync`))
//! - 함수형 갱신은 새 환경을 반환 (Functional updates return new environments)
//! - 구조적 공유를 통한 O(log n) 연산(많은 연산에서 O(1) 상각)
//!   (O(log n) operations with structural sharing (O(1) amortized for many operations))
//! - 클로저가 참조 카운팅 없이 환경을 소유 가능
//!   (Closures can own their environment without reference counting)
//!
//! ## 예시 (Example)
//!
//! ```
//! use sicp_common::environment::Environment;
//!
//! // 빈 환경 생성 (Create an empty environment)
//! let env: Environment<i64> = Environment::new();
//!
//! // define는 새 환경을 반환한다(함수형 갱신) (Define returns a NEW environment (functional update))
//! let env = env.define("x".to_string(), 10);
//! let env = env.define("y".to_string(), 20);
//!
//! assert_eq!(env.lookup("x"), Some(&10));
//! assert_eq!(env.lookup("y"), Some(&20));
//!
//! // extend는 자식 환경을 생성한다 (Extend creates a child environment)
//! let child = env.extend([("z".to_string(), 30)]);
//! assert_eq!(child.lookup("z"), Some(&30));
//! assert_eq!(child.lookup("x"), Some(&10)); // 부모에서 상속 (Inherits from parent)
//! ```

use im::HashMap as ImHashMap;

/// 변수 바인딩을 위한 영속적 불변 환경 (A persistent, immutable environment for variable bindings).
///
/// 환경은 체인을 이루며 각 환경은 부모를 가질 수 있다
/// (Environments form a chain where each environment can have a parent).
/// 조회는 자식에서 부모로 체인을 따라가며 바인딩을 찾는다
/// (Lookups traverse the chain from child to parent until the binding is found).
///
/// ## 함수형 의미론 (Functional Semantics)
///
/// 모든 변경 연산은 제자리 수정 대신 새 `Environment`를 반환한다
/// (All mutating operations return a new `Environment` rather than modifying
/// in place). 이는 `im::HashMap`의 구조적 공유로 효율적이다
/// (This is efficient due to structural sharing provided by `im::HashMap`).
///
/// ## 메모리 효율성 (Memory Efficiency)
///
/// 구조적 공유 덕분에, 자식 환경 생성이나 바인딩 추가는
/// 새로 바뀐 부분만 메모리를 할당한다
/// (Thanks to structural sharing, creating a child environment or adding
/// a binding only allocates memory for the new/changed parts).
/// 변경되지 않은 부분은 원본 환경과 공유된다
/// (The unchanged portions are shared with the original environment).
#[derive(Debug, Clone)]
pub struct Environment<V> {
    /// 현재 프레임의 바인딩 (Current frame's bindings)
    bindings: ImHashMap<String, V>,
    /// 부모 환경(있다면) (Parent environment (if any))
    parent: Option<Box<Environment<V>>>,
}

impl<V> Default for Environment<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> Environment<V> {
    /// 부모가 없는 새 빈 환경을 생성한다 (Creates a new empty environment with no parent).
    #[must_use]
    pub fn new() -> Self {
        Self {
            bindings: ImHashMap::new(),
            parent: None,
        }
    }

    /// 주어진 부모를 가진 새 환경을 생성한다 (Creates a new environment with the given parent).
    #[must_use]
    pub fn with_parent(parent: Environment<V>) -> Self {
        Self {
            bindings: ImHashMap::new(),
            parent: Some(Box::new(parent)),
        }
    }

    /// 이 프레임의 바인딩 개수를 반환한다(부모 제외)
    /// (Returns the number of bindings in this frame (not including parent)).
    #[must_use]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// 이 프레임이 비어 있으면 true를 반환한다(부모는 바인딩을 가질 수 있음)
    /// (Returns true if this frame has no bindings (parent may have bindings)).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// 이 환경에 부모가 있으면 true를 반환한다 (Returns true if this environment has a parent).
    #[must_use]
    pub fn has_parent(&self) -> bool {
        self.parent.is_some()
    }

    /// 부모 환경 참조를 반환한다(있다면) (Returns a reference to the parent environment, if any).
    #[must_use]
    pub fn parent(&self) -> Option<&Environment<V>> {
        self.parent.as_deref()
    }
}

impl<V: Clone> Environment<V> {
    /// 이 환경과 부모들에서 변수를 조회한다
    /// (Looks up a variable in this environment and its parents).
    ///
    /// 찾으면 `Some(&value)`, 없으면 `None`을 반환한다
    /// (Returns `Some(&value)` if found, `None` if not bound).
    ///
    /// # 예시 (Example)
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

    /// 이 환경에 새 바인딩을 정의하고 새 환경을 반환한다
    /// (Defines a new binding in this environment, returning a new environment).
    ///
    /// 이 프레임에 이름이 이미 있으면 섀도잉된다; 부모 바인딩은 영향 없음
    /// (If the name already exists in this frame, it is shadowed.
    /// Parent bindings are not affected).
    ///
    /// # 예시 (Example)
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

    /// 주어진 바인딩으로 새 자식 환경을 생성한다
    /// (Creates a new child environment with the given bindings).
    ///
    /// 일반적으로 새 스코프(예: 함수 호출)에 들어갈 때 사용한다
    /// (This is typically used when entering a new scope (e.g., function call)).
    ///
    /// # 예시 (Example)
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

    /// 현재 프레임에 추가 바인딩을 확장한다(동일 스코프)
    /// (Extends the current frame with additional bindings (same scope level)).
    ///
    /// `extend`와 달리 자식을 만들지 않고 현재 프레임에 추가한다
    /// (Unlike `extend`, this adds to the current frame rather than creating a child).
    ///
    /// # 예시 (Example)
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

    /// 이 환경(부모 포함)에 이름이 바인딩되어 있는지 확인한다
    /// (Checks if a name is bound in this environment (including parents)).
    #[must_use]
    pub fn is_bound(&self, name: &str) -> bool {
        self.lookup(name).is_some()
    }

    /// 이 프레임에만 이름이 바인딩되어 있는지 확인한다(부모 제외)
    /// (Checks if a name is bound in this frame only (not checking parents)).
    #[must_use]
    pub fn is_bound_locally(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }

    /// 현재 프레임의 바인딩에 대한 이터레이터를 반환한다
    /// (Returns an iterator over bindings in the current frame).
    pub fn iter_local(&self) -> impl Iterator<Item = (&String, &V)> {
        self.bindings.iter()
    }

    /// 이 환경과 부모들의 모든 바인딩을 수집한다
    /// (Collects all bindings from this environment and its parents).
    ///
    /// 섀도잉이 있으면 가장 안쪽 바인딩을 반환한다
    /// (In case of shadowing, returns the innermost binding).
    #[must_use]
    pub fn all_bindings(&self) -> ImHashMap<String, V> {
        let parent_bindings = self
            .parent
            .as_ref()
            .map(|p| p.all_bindings())
            .unwrap_or_default();

        // 자식 바인딩이 부모 바인딩을 덮어쓴다 (Child bindings override parent bindings)
        let mut result = parent_bindings;
        for (k, v) in &self.bindings {
            result = result.update(k.clone(), v.clone());
        }
        result
    }

    /// 이 환경의 깊이를 반환한다(이 프레임 포함 개수)
    /// (Returns the depth of this environment (number of frames including this one)).
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

        // 원본은 변경되지 않음 (Original unchanged)
        assert_eq!(env1.lookup("x"), Some(&1));
        // 새 환경은 갱신된 값을 가짐 (New has updated value)
        assert_eq!(env2.lookup("x"), Some(&2));
    }

    #[test]
    fn test_extend_creates_child() {
        let parent = Environment::new().define("x".to_string(), 10);

        let child = parent.extend([("y".to_string(), 20), ("z".to_string(), 30)]);

        // 자식은 부모 바인딩을 볼 수 있음 (Child can see parent bindings)
        assert_eq!(child.lookup("x"), Some(&10));
        assert_eq!(child.lookup("y"), Some(&20));
        assert_eq!(child.lookup("z"), Some(&30));

        // 부모는 자식 바인딩을 보지 못함 (Parent doesn't see child bindings)
        assert_eq!(parent.lookup("y"), None);
    }

    #[test]
    fn test_shadowing() {
        let outer = Environment::new().define("x".to_string(), 10);

        let inner = outer.extend([("x".to_string(), 20)]);

        // 내부가 외부를 섀도잉 (Inner shadows outer)
        assert_eq!(inner.lookup("x"), Some(&20));
        // 외부는 변경되지 않음 (Outer unchanged)
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

        assert!(child.is_bound("x")); // 부모에 있음 (In parent)
        assert!(!child.is_bound_locally("x")); // 이 프레임에는 없음 (Not in this frame)

        assert!(child.is_bound("y")); // 이 프레임에 있음 (In this frame)
        assert!(child.is_bound_locally("y")); // 이 프레임에 있음 (In this frame)
    }

    #[test]
    fn test_all_bindings() {
        let env = Environment::new()
            .define("x".to_string(), 1)
            .extend([("y".to_string(), 2), ("x".to_string(), 3)]); // x 섀도잉 (x shadows)

        let all = env.all_bindings();
        assert_eq!(all.get("x"), Some(&3)); // 섀도잉된 값 (Shadowed value)
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
        assert_eq!(env.depth(), 1); // 모두 같은 프레임 (All in same frame)
    }
}
