//! SICP 4장: 메타언어적 추상화 (Metalinguistic Abstraction)
//!
//! 이 장에서는 인터프리터 (interpreter)를 구축합니다:
//! - 메타순환적 평가기 (AST + 패턴 매칭) (Metacircular evaluator)
//! - 지연 평가 (thunk, 메모이제이션) (Lazy evaluation)
//! - 비결정론적 계산 (백트래킹) (Nondeterministic computing)
//! - 논리 프로그래밍 (패턴 매칭, 통일) (Logic programming)

// 구현된 모듈 (Implemented modules)
pub mod section_4_1; // 메타순환적 평가기 (Metacircular Evaluator)
pub mod section_4_2; // 지연 평가 (Lazy Evaluation)
pub mod section_4_3; // 비결정론적 계산 (Nondeterministic Computing)
pub mod section_4_4; // 논리 프로그래밍 (Logic Programming)
