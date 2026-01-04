//! Demonstration of SICP Section 3.4: Concurrency
//!
//! Run with: cargo run --bin section_3_4_demo

use sicp_chapter3::section_3_4::*;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   SICP Section 3.4: Concurrency - Time Is of the Essence    ║");
    println!("║              Rust Implementation (Edition 2024)              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Demo 1: Race Condition Prevention
    demonstrate_race_condition_prevention();

    // Demo 2: Parallel Execution
    parallel_execute_example();

    // Demo 3: Deadlock Prevention
    demonstrate_deadlock_prevention();

    // Demo 4: Atomic Operations
    demonstrate_atomic_operations();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Summary                                   ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Rust's 'Fearless Concurrency' Advantages:                  ║");
    println!("║                                                              ║");
    println!("║  1. Compile-time safety: No data races possible             ║");
    println!("║  2. Type system enforces Send/Sync traits                   ║");
    println!("║  3. Ownership prevents shared mutable state                 ║");
    println!("║  4. RAII ensures automatic lock release                     ║");
    println!("║  5. Arc<Mutex<T>> pattern for safe sharing                  ║");
    println!("║  6. Atomic operations with memory ordering                  ║");
    println!("║                                                              ║");
    println!("║  SICP teaches manual synchronization (serializers).         ║");
    println!("║  Rust enforces it at compile time!                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
