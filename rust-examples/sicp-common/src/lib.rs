//! # SICP Common Utilities
//!
//! Shared utilities for SICP Rust examples providing idiomatic, functional patterns
//! that avoid `Rc<RefCell<T>>` anti-patterns.
//!
//! ## Modules
//!
//! - [`arena`]: Type-safe arena allocation with index-based references
//! - [`environment`]: Persistent environments using `im::HashMap` for functional scoping
//! - [`list`]: Functional list operations and cons cell patterns
//!
//! ## Design Principles
//!
//! This crate follows these principles from the SICP Rust modernization:
//!
//! 1. **Pure Functional**: Operations return new values instead of mutating
//! 2. **Ownership-based**: Leverage Rust's ownership model instead of GC semantics
//! 3. **Iterator-centric**: Use iterator combinators for sequence operations
//! 4. **No `Rc<RefCell<T>>`**: Avoid runtime borrow checking patterns

pub mod arena;
pub mod environment;
pub mod list;

// Re-export main types for convenience
pub use arena::{Arena, ArenaId};
pub use environment::Environment;
