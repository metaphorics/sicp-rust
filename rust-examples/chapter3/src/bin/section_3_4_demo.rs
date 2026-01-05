//! SICP 3.4절 데모: 동시성 (Concurrency)
//!
//! 실행 방법 (How to run): cargo run --bin section_3_4_demo

use sicp_chapter3::section_3_4::*;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  SICP 3.4절: 동시성 - 시간은 본질이다 (Time Is of the Essence) ║");
    println!("║         Rust 구현 버전 (Rust implementation, Edition 2024)    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // 데모 1: 경쟁 조건 방지 (Race condition prevention)
    demonstrate_race_condition_prevention();

    // 데모 2: 병렬 실행 (Parallel execution)
    parallel_execute_example();

    // 데모 3: 데드락 방지 (Deadlock prevention)
    demonstrate_deadlock_prevention();

    // 데모 4: 원자적 연산 (Atomic operations)
    demonstrate_atomic_operations();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                     요약 (Summary)                           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Rust의 '두려움 없는 동시성' 장점 (Fearless Concurrency):    ║");
    println!("║                                                              ║");
    println!("║  1. 컴파일 타임 안전성: 데이터 레이스 불가 (No data races)   ║");
    println!("║  2. 타입 시스템이 Send/Sync 트레이트를 강제함               ║");
    println!("║     (Type system enforces Send/Sync traits)                 ║");
    println!("║  3. 소유권 시스템이 공유된 가변 상태를 방지함               ║");
    println!("║     (Ownership prevents shared mutable state)               ║");
    println!("║  4. RAII로 락(lock) 자동 해제 (Automatic unlock via RAII)   ║");
    println!("║  5. 안전한 공유를 위한 Arc<Mutex<T>> 패턴                  ║");
    println!("║     (Pattern for safe sharing)                              ║");
    println!("║  6. 메모리 순서 보장 원자 연산 (Atomic ordering guarantees) ║");
    println!("║                                                              ║");
    println!("║  SICP는 수동 동기화(직렬화기)를 가르친다.                   ║");
    println!("║  (SICP teaches manual synchronization/serializers.)         ║");
    println!("║  Rust는 이를 컴파일 타임에 강제한다! (Rust enforces it!)     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
