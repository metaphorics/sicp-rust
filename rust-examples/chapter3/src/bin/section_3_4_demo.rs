//! SICP 3.4절 데모: 동시성 (Concurrency)
//!
//! 실행 방법: cargo run --bin section_3_4_demo

use sicp_chapter3::section_3_4::*;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      SICP 3.4절: 동시성 - 시간은 본질적인 요소이다           ║");
    println!("║              Rust 구현 버전 (Edition 2024)                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // 데모 1: 경쟁 조건(Race Condition) 방지
    demonstrate_race_condition_prevention();

    // 데모 2: 병렬 실행
    parallel_execute_example();

    // 데모 3: 데드락(Deadlock) 방지
    demonstrate_deadlock_prevention();

    // 데모 4: 원자적(Atomic) 연산
    demonstrate_atomic_operations();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                         요약                                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Rust의 '두려움 없는 동시성(Fearless Concurrency)' 장점:     ║");
    println!("║                                                              ║");
    println!("║  1. 컴파일 타임 안전성: 데이터 레이스 발생 불가능           ║");
    println!("║  2. 타입 시스템이 Send/Sync 트레이트를 강제함                ║");
    println!("║  3. 소유권 시스템이 공유된 가변 상태를 방지함               ║");
    println!("║  4. RAII를 통해 락(lock)이 자동으로 해제됨                  ║");
    println!("║  5. 안전한 공유를 위한 Arc<Mutex<T>> 패턴                   ║");
    println!("║  6. 메모리 순서(ordering)를 보장하는 원자적 연산            ║");
    println!("║                                                              ║");
    println!("║  SICP는 수동 동기화(직렬화기)를 가르칩니다.                 ║");
    println!("║  Rust는 이를 컴파일 타임에 강제합니다!                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
