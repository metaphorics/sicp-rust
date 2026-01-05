//! SICP 3.5절: 스트림 데모 (Streams Demonstration)
//!
//! 이 예제는 SICP 3.5장의 핵심 개념을 보여주며, 러스트의 Iterator 트레이트가
//! Scheme의 delay/force 메커니즘과 유사하게 어떻게 지연 평가(lazy evaluation)를
//! 제공하는지 보여준다
//! (This example demonstrates key concepts from SICP Section 3.5 and how Rust's
//! Iterator trait provides lazy evaluation similar to Scheme's delay/force).
//!
//! 실행 방법 (How to run): cargo run --example streams_demo

use sicp_chapter3::section_3_5::*;

fn main() {
    println!("=== SICP 3.5장: 스트림 (Streams) ===\n");

    // =========================================================================
    // 3.5.1절: 스트림은 지연 리스트이다 (Streams Are Delayed Lists)
    // =========================================================================
    println!("3.5.1절: 지연 평가 vs 즉시 평가 (Lazy Evaluation vs Eager Evaluation)");
    println!("{}", "=".repeat(60));

    // 지연 평가의 효율성 데모 (Lazy evaluation efficiency demo)
    println!("[10,000 - 11,000] 범위에서 두 번째 소수 찾기 (Find the 2nd prime in range):");
    println!("  지연 평가는 답을 찾을 때까지만 계산한다 (Lazy only computes until it finds the answer).");
    println!("  즉시 평가는 먼저 전체 소수 리스트를 구축한다 (Eager builds the whole prime list first).\n");

    let lazy_result = delayed_lists::second_prime_lazy(10_000, 11_000);
    println!("  결과: {:?}", lazy_result);

    // 이터레이터 콤비네이터 데모 (Iterator combinator demo)
    println!("\n처음 5개 홀수의 제곱 합 (Sum of squares of first 5 odd numbers):");
    println!("  (1² + 3² + 5² + 7² + 9²)");
    let sum = delayed_lists::sum_of_squares_of_odd_numbers(5);
    println!("  결과: {}", sum);

    // =========================================================================
    // 3.5.2절: 무한 스트림 (Infinite Streams)
    // =========================================================================
    println!("\n\n3.5.2절: 무한 스트림 (Infinite Streams)");
    println!("{}", "=".repeat(60));

    // 무한 정수 스트림 (Infinite integer stream)
    println!("1부터 시작하는 정수들 (Integers starting at 1):");
    let integers: Vec<i64> = IntegersFrom::new(1).take(10).collect();
    println!("  처음 10개: {:?}", integers);

    // 피보나치 수열 (Fibonacci sequence)
    println!("\n피보나치 수열 (Fibonacci sequence):");
    let fibs: Vec<u64> = Fibonacci::new().take(15).collect();
    println!("  처음 15개: {:?}", fibs);

    // 에라토스테네스의 체를 이용한 소수 (Primes via sieve)
    println!("\n소수 (에라토스테네스의 체) (Primes via sieve):");
    let primes: Vec<u64> = PrimesOptimized::new().take(20).collect();
    println!("  처음 20개: {:?}", primes);

    // 50번째 소수 찾기 (Find the 50th prime)
    let prime_50 = PrimesOptimized::new().nth(49).unwrap();
    println!("  50번째 소수 (50th prime): {}", prime_50);

    // =========================================================================
    // 암시적 스트림 정의 (Implicit Stream Definitions)
    // =========================================================================
    println!("\n\n암시적 스트림 정의 (Implicit Stream Definitions)");
    println!("{}", "=".repeat(60));

    // 1로만 이루어진 스트림 (All-ones stream)
    println!("1로 이루어진 스트림 (All-ones stream):");
    let ones_vec: Vec<i64> = ones().take(10).collect();
    println!("  {:?}", ones_vec);

    // 배가 스트림 (2의 거듭제곱) (Doubling stream / powers of 2)
    println!("\n2의 거듭제곱 (Powers of 2):");
    let powers: Vec<u64> = DoublingStream::new().take(10).collect();
    println!("  {:?}", powers);

    // 스트림 더하기 (Add streams)
    println!("\n두 스트림을 요소별로 더하기 (Add two streams element-wise):");
    let s1 = vec![1, 2, 3, 4, 5];
    let s2 = vec![10, 20, 30, 40, 50];
    let sum_stream: Vec<i64> = add_streams(s1.into_iter(), s2.into_iter()).collect();
    println!("  [1,2,3,4,5] + [10,20,30,40,50] = {:?}", sum_stream);

    // =========================================================================
    // 3.5.3절: 스트림 패러다임 활용 (Exploiting the Stream Paradigm)
    // =========================================================================
    println!("\n\n3.5.3절: 스트림 패러다임의 응용 (Exploiting the Stream Paradigm)");
    println!("{}", "=".repeat(60));

    // 스트림을 이용한 제곱근 근사 (Sqrt approximation via streams)
    println!("반복적 개선을 통한 2의 제곱근 구하기 (Sqrt(2) via iterative improvement):");
    let sqrt2_approx: Vec<f64> = SqrtStream::new(2.0).take(6).collect();
    for (i, approx) in sqrt2_approx.iter().enumerate() {
        println!("  반복 {} (Iteration): {:.10}", i, approx);
    }
    println!("  실제 sqrt(2) (Actual): {:.10}", 2.0_f64.sqrt());

    // 스트림 극한 (Stream limit)
    println!("\nstream_limit을 사용해 근사치가 충분히 정확할 때 찾기 (Find when approximation is within tolerance):");
    let sqrt2_limited = stream_limit(SqrtStream::new(2.0), 0.0001);
    println!("  허용 오차 0.0001 내의 결과 (Result within tolerance 0.0001): {:?}", sqrt2_limited);

    // 부분 합 (Partial sums)
    println!("\n정수 1-10의 부분 합 (Partial sums of 1-10):");
    let integers = (1..=10).map(|x| x as f64);
    let partial_sums: Vec<f64> = PartialSums::new(integers).collect();
    println!("  {:?}", partial_sums);

    // 파이(Pi) 근사 (Pi approximation)
    println!("\n교대 급수를 이용한 π 근사 (Pi approximation via alternating series):");
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
            println!("  항 {} (Term): π ≈ {:.6}", i + 1, val);
            val
        })
        .collect();

    println!("  실제 π (Actual π): {:.6}", std::f64::consts::PI);

    // 빠른 수렴을 위한 오일러 변환 (Euler transform for faster convergence)
    println!("\n빠른 수렴을 위해 오일러 변환 적용 (Apply Euler transform):");
    let pi_summands = PiSummands::new()
        .enumerate()
        .map(|(i, term)| if i % 2 == 0 { term } else { -term });
    let partial_sums = PartialSums::new(pi_summands);
    let pi_values: Vec<f64> = partial_sums.map(|x| x * 4.0).take(20).collect();

    let transformed: Vec<f64> = EulerTransform::new(pi_values.into_iter())
        .take(8)
        .enumerate()
        .map(|(i, val)| {
            println!("  변환된 항 {} (Transformed term): π ≈ {:.8}", i + 1, val);
            val
        })
        .collect();

    if let Some(&last) = transformed.last() {
        let error = (last - std::f64::consts::PI).abs();
        println!("  최종 오차 (Final error): {:.10}", error);
    }

    // =========================================================================
    // 무한 쌍 스트림 (Infinite Streams of Pairs)
    // =========================================================================
    println!("\n\n무한 쌍 스트림 (Infinite Streams of Pairs)");
    println!("{}", "=".repeat(60));

    println!("i ≤ j 인 처음 20개의 쌍 (i, j) (First 20 pairs with i ≤ j):");
    let pairs: Vec<(usize, usize)> = Pairs::new().take(20).collect();
    for (idx, (i, j)) in pairs.iter().enumerate() {
        print!("({},{}) ", i, j);
        if (idx + 1) % 5 == 0 {
            println!();
        }
    }
    println!();

    // =========================================================================
    // 3.5.4절: 신호 처리 (Signal Processing)
    // =========================================================================
    println!("\n\n3.5.4절: 신호 처리 (Signal Processing)");
    println!("{}", "=".repeat(60));

    // 상수 신호의 적분 (Integrate constant signal)
    println!("상수 신호(값 = 1.0) 적분하기 (Integrate constant signal):");
    let ones_signal = std::iter::repeat(1.0);
    let integral: Vec<f64> = Integrator::new(ones_signal, 0.0, 0.1).take(11).collect();

    println!("  시간 (Time): 0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0");
    print!("  적분값 (Integral):");
    for value in &integral {
        print!(" {:.1}", value);
    }
    println!();

    // 미분 방정식 dy/dt = y, y(0) = 1 풀이 (Solve dy/dt = y, y(0) = 1)
    println!("\ny(0) = 1 인 dy/dt = y 풀이 (Solve dy/dt = y, y(0) = 1):");
    println!("  해석적 해 (Analytical solution): y(t) = e^t");

    let _solution: Vec<f64> = solve(|y| y, 1.0, 0.01)
        .take(101)
        .enumerate()
        .filter(|(i, _)| i % 10 == 0)
        .map(|(i, y)| {
            let t = i as f64 * 0.01;
            let exact = t.exp();
            let error = (y - exact).abs();
            println!(
                "  t={:.2}, y={:.6}, exact={:.6}, 오차={:.8} (error)",
                t, y, exact, error
            );
            y
        })
        .collect();

    let y_at_1 = solve(|y| y, 1.0, 0.001).nth(1000).unwrap();
    println!(
        "  y(1.0) ≈ {:.6} (e ≈ {:.6})",
        y_at_1,
        std::f64::consts::E
    );

    // =========================================================================
    // 3.5.5절: 몬테카를로 추정 (Monte Carlo Estimation)
    // =========================================================================
    println!("\n\n3.5.5절: π의 몬테카를로 추정 (Monte Carlo estimation of π)");
    println!("{}", "=".repeat(60));

    println!("몬테카를로 방법으로 π 추정하기 (Estimate π with Monte Carlo):");
    println!("  (GCD 테스트 기반: GCD(a,b) = 1 확률은 π와 관련)");
    println!("  (Based on GCD test: P(GCD(a,b)=1) relates to π)");

    for &trials in &[100, 1000, 10000, 100000] {
        let estimate = monte_carlo::estimate_pi(trials, 42);
        let error = (estimate - std::f64::consts::PI).abs();
        println!(
            "  {:6} 회 시행 (trials): π ≈ {:.6}, 오차 = {:.6} (error)",
            trials, estimate, error
        );
    }

    // =========================================================================
    // 결론 (Conclusion)
    // =========================================================================
    println!("\n\n{}", "=".repeat(60));
    println!("핵심 통찰 (Key takeaways):");
    println!("{}", "=".repeat(60));
    println!("1. Rust의 Iterator 트레이트는 기본적으로 지연 평가 제공");
    println!("   (Iterator trait is lazy by default)");
    println!("2. 명시적인 delay/force가 불필요함 - 이터레이터는 이미 지연 방식");
    println!("   (No explicit delay/force needed; iterators are already lazy)");
    println!("3. 이터레이터 콤비네이터(map, filter, take)는 자연스럽게 합성됨");
    println!("   (Iterator combinators compose naturally)");
    println!("4. 무한 스트림도 유한 스트림만큼 자연스럽다");
    println!("   (Infinite streams feel as natural as finite ones)");
    println!("5. 이터레이터로 상태를 함수형으로 관리 가능");
    println!("   (State can be managed functionally with iterators)");
    println!("6. 신호 처리 및 미분 방정식이 깔끔하게 매핑됨");
    println!("   (Signal processing and ODEs map cleanly)");
    println!("{}", "=".repeat(60));
}
