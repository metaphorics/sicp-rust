//! SICP 1장: 프로시저를 이용한 추상화 구축 (Building Abstractions with Procedures)
//!
//! 이 장에서는 프로그래밍의 기본 요소를 소개합니다:
//! - 표현식과 평가 (Expressions and evaluation)
//! - 명명과 환경 (Naming and the environment)
//! - 복합 프로시저 (Compound procedures)
//! - 고차 프로시저 (Higher-order procedures)
//! - 재귀와 반복 (Recursion and iteration)

pub mod section_1_1;
pub mod section_1_2;
pub mod section_1_3;

// 자주 사용되는 항목들을 재수출한다 (Re-export commonly used items).
pub use section_1_1::*;
