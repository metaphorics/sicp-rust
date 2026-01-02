//! SICP Section 3.5: Streams Demonstration
//!
//! This example demonstrates the key concepts from SICP Chapter 3.5,
//! showing how Rust's Iterator trait provides lazy evaluation similar
//! to Scheme's delay/force mechanism.
//!
//! Run with: cargo run --example streams_demo

use sicp_chapter3::section_3_5::*;

fn main() {
    println!("=== SICP Chapter 3.5: Streams ===\n");

    // =========================================================================
    // Section 3.5.1: Streams Are Delayed Lists
    // =========================================================================
    println!("Section 3.5.1: Lazy Evaluation vs Eager Evaluation");
    println!("{}", "=".repeat(60));

    // Demonstrate the efficiency of lazy evaluation
    println!("Finding the 2nd prime in range [10,000 - 11,000]:");
    println!("  Lazy evaluation only computes until the answer is found.");
    println!("  Eager evaluation builds entire list of primes first.\n");

    let lazy_result = delayed_lists::second_prime_lazy(10_000, 11_000);
    println!("  Result: {:?}", lazy_result);

    // Demonstrate iterator combinators
    println!("\nSum of squares of first 5 odd numbers:");
    println!("  (1² + 3² + 5² + 7² + 9²)");
    let sum = delayed_lists::sum_of_squares_of_odd_numbers(5);
    println!("  Result: {}", sum);

    // =========================================================================
    // Section 3.5.2: Infinite Streams
    // =========================================================================
    println!("\n\nSection 3.5.2: Infinite Streams");
    println!("{}", "=".repeat(60));

    // Infinite stream of integers
    println!("Integers starting from 1:");
    let integers: Vec<i64> = IntegersFrom::new(1).take(10).collect();
    println!("  First 10: {:?}", integers);

    // Fibonacci sequence
    println!("\nFibonacci sequence:");
    let fibs: Vec<u64> = Fibonacci::new().take(15).collect();
    println!("  First 15: {:?}", fibs);

    // Prime numbers using Sieve of Eratosthenes
    println!("\nPrime numbers (Sieve of Eratosthenes):");
    let primes: Vec<u64> = PrimesOptimized::new().take(20).collect();
    println!("  First 20: {:?}", primes);

    // Find the 50th prime
    let prime_50 = PrimesOptimized::new().nth(49).unwrap();
    println!("  50th prime: {}", prime_50);

    // =========================================================================
    // Implicit Stream Definitions
    // =========================================================================
    println!("\n\nImplicit Stream Definitions");
    println!("{}", "=".repeat(60));

    // Stream of ones
    println!("Stream of ones:");
    let ones_vec: Vec<i64> = ones().take(10).collect();
    println!("  {:?}", ones_vec);

    // Doubling stream (powers of 2)
    println!("\nPowers of 2:");
    let powers: Vec<u64> = DoublingStream::new().take(10).collect();
    println!("  {:?}", powers);

    // Adding streams
    println!("\nAdding two streams element-wise:");
    let s1 = vec![1, 2, 3, 4, 5];
    let s2 = vec![10, 20, 30, 40, 50];
    let sum_stream: Vec<i64> = add_streams(s1.into_iter(), s2.into_iter()).collect();
    println!("  [1,2,3,4,5] + [10,20,30,40,50] = {:?}", sum_stream);

    // =========================================================================
    // Section 3.5.3: Exploiting the Stream Paradigm
    // =========================================================================
    println!("\n\nSection 3.5.3: Stream Paradigm Applications");
    println!("{}", "=".repeat(60));

    // Square root approximation using streams
    println!("Square root of 2 using iterative improvement:");
    let sqrt2_approx: Vec<f64> = SqrtStream::new(2.0).take(6).collect();
    for (i, approx) in sqrt2_approx.iter().enumerate() {
        println!("  Iteration {}: {:.10}", i, approx);
    }
    println!("  Actual sqrt(2): {:.10}", 2.0_f64.sqrt());

    // Stream limit
    println!("\nUsing stream_limit to find when approximation is good enough:");
    let sqrt2_limited = stream_limit(SqrtStream::new(2.0), 0.0001);
    println!("  Result within tolerance 0.0001: {:?}", sqrt2_limited);

    // Partial sums
    println!("\nPartial sums of integers 1-10:");
    let integers = (1..=10).map(|x| x as f64);
    let partial_sums: Vec<f64> = PartialSums::new(integers).collect();
    println!("  {:?}", partial_sums);

    // Pi approximation
    println!("\nApproximating π using alternating series:");
    println!("  π/4 = 1 - 1/3 + 1/5 - 1/7 + ...");

    let pi_summands = PiSummands::new()
        .enumerate()
        .map(|(i, term)| if i % 2 == 0 { term } else { -term });

    let partial_sums = PartialSums::new(pi_summands);
    let _pi_approx: Vec<f64> = partial_sums
        .map(|x| x * 4.0)
        .take(10)
        .enumerate()
        .map(|(i, val)| {
            println!("  Term {}: π ≈ {:.6}", i + 1, val);
            val
        })
        .collect();

    println!("  Actual π: {:.6}", std::f64::consts::PI);

    // Euler transform for faster convergence
    println!("\nApplying Euler transform for faster convergence:");
    let pi_summands = PiSummands::new()
        .enumerate()
        .map(|(i, term)| if i % 2 == 0 { term } else { -term });
    let partial_sums = PartialSums::new(pi_summands);
    let pi_values: Vec<f64> = partial_sums.map(|x| x * 4.0).take(20).collect();

    let transformed: Vec<f64> = EulerTransform::new(pi_values.into_iter())
        .take(8)
        .enumerate()
        .map(|(i, val)| {
            println!("  Transformed term {}: π ≈ {:.8}", i + 1, val);
            val
        })
        .collect();

    if let Some(&last) = transformed.last() {
        let error = (last - std::f64::consts::PI).abs();
        println!("  Final error: {:.10}", error);
    }

    // =========================================================================
    // Infinite Streams of Pairs
    // =========================================================================
    println!("\n\nInfinite Streams of Pairs");
    println!("{}", "=".repeat(60));

    println!("First 20 pairs (i, j) where i ≤ j:");
    let pairs: Vec<(usize, usize)> = Pairs::new().take(20).collect();
    for (idx, (i, j)) in pairs.iter().enumerate() {
        print!("({},{}) ", i, j);
        if (idx + 1) % 5 == 0 {
            println!();
        }
    }
    println!();

    // =========================================================================
    // Section 3.5.4: Signal Processing
    // =========================================================================
    println!("\n\nSection 3.5.4: Signal Processing");
    println!("{}", "=".repeat(60));

    // Integration of a constant signal
    println!("Integrating constant signal (value = 1.0):");
    let ones_signal = std::iter::repeat(1.0);
    let integral: Vec<f64> = Integrator::new(ones_signal, 0.0, 0.1).take(11).collect();

    println!("  Time:     0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0");
    print!("  Integral:");
    for value in &integral {
        print!(" {:.1}", value);
    }
    println!();

    // Solving differential equation dy/dt = y, y(0) = 1
    println!("\nSolving dy/dt = y with y(0) = 1:");
    println!("  Analytical solution: y(t) = e^t");

    let _solution: Vec<f64> = solve(|y| y, 1.0, 0.01)
        .take(101)
        .enumerate()
        .filter(|(i, _)| i % 10 == 0)
        .map(|(i, y)| {
            let t = i as f64 * 0.01;
            let exact = t.exp();
            let error = (y - exact).abs();
            println!(
                "  t={:.2}, y={:.6}, exact={:.6}, error={:.8}",
                t, y, exact, error
            );
            y
        })
        .collect();

    let y_at_1 = solve(|y| y, 1.0, 0.001).nth(1000).unwrap();
    println!("  y(1.0) ≈ {:.6} (e ≈ {:.6})", y_at_1, std::f64::consts::E);

    // =========================================================================
    // Section 3.5.5: Monte Carlo Estimation
    // =========================================================================
    println!("\n\nSection 3.5.5: Monte Carlo Estimation of π");
    println!("{}", "=".repeat(60));

    println!("Estimating π using Monte Carlo method:");
    println!("  (Based on GCD test: probability that GCD(a,b) = 1 relates to π)");

    for &trials in &[100, 1000, 10000, 100000] {
        let estimate = monte_carlo::estimate_pi(trials, 42);
        let error = (estimate - std::f64::consts::PI).abs();
        println!(
            "  {:6} trials: π ≈ {:.6}, error = {:.6}",
            trials, estimate, error
        );
    }

    // =========================================================================
    // Conclusion
    // =========================================================================
    println!("\n\n{}", "=".repeat(60));
    println!("Key Insights:");
    println!("{}", "=".repeat(60));
    println!("1. Rust's Iterator trait provides lazy evaluation by default");
    println!("2. No need for explicit delay/force - iterators are lazy");
    println!("3. Iterator combinators (map, filter, take) compose naturally");
    println!("4. Infinite streams are just as natural as finite ones");
    println!("5. State can be managed functionally using iterators");
    println!("6. Signal processing and differential equations map cleanly");
    println!("{}", "=".repeat(60));
}
