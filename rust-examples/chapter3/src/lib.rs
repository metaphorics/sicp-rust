//! SICP 3장: 모듈성, 객체, 그리고 상태 (Modularity, Objects, and State)
//! 
//! 이 장에서는 상태와 변경을 탐구합니다:
//! - 할당과 지역 상태 (RefCell, Mutex) (Assignment and local state)
//! - 환경 모델 (소유권, 수명) (The environment model)
//! - 가변 데이터 (내부 가변성) (Mutable data)
//! - 동시성 (두려움 없는 동시성) (Concurrency)
//! - 스트림 (이터레이터, 비동기) (Streams)

// 모듈
pub mod section_3_1; // 할당과 지역 상태 (Assignment and Local State)
pub mod section_3_2; // 환경 모델 (Environment Model)
pub mod section_3_3; // 가변 데이터 (Mutable Data)
pub mod section_3_4; // 동시성: 시간은 본질적인 요소이다 (Concurrency)
pub mod section_3_5; // 스트림 (Streams)
