//! # SICP 공통 유틸리티 (SICP Common Utilities)
//!
//! SICP Rust 예제를 위한 공유 유틸리티로, `Rc<RefCell<T>>` 안티 패턴을 피하는
//! 관용적이고 함수형인 패턴을 제공합니다.
//!
//! ## 모듈 (Modules)
//!
//! - [`arena`]: 인덱스 기반 참조를 사용하는 타입 안전한 아레나 할당
//! - [`environment`]: 함수형 스코핑을 위해 `im::HashMap`을 사용하는 영속적 환경
//! - [`list`]: 함수형 리스트 연산과 cons 셀 패턴
//!
//! ## 설계 원칙 (Design Principles)
//!
//! 이 크레이트는 SICP Rust 현대화 프로젝트의 다음 원칙들을 따릅니다:
//!
//! 1. **순수 함수형 (Pure Functional)**: 연산은 상태를 변경하는 대신 새로운 값을 반환합니다.
//! 2. **소유권 기반 (Ownership-based)**: GC 의미론 대신 Rust의 소유권 모델을 활용합니다.
//! 3. **이터레이터 중심 (Iterator-centric)**: 시퀀스 연산에 이터레이터 콤비네이터를 사용합니다.
//! 4. **`Rc<RefCell<T>>` 없음**: 런타임 빌림 검사 패턴을 피합니다.

pub mod arena;
pub mod environment;
pub mod list;

// Re-export main types for convenience
pub use arena::{Arena, ArenaId};
pub use environment::Environment;
