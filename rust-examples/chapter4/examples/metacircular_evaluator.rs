//! 예제: 메타순환적 평가기 사용 (Using the Metacircular Evaluator)
//!
//! Rust로 구현된 SICP 메타순환적 평가기를 보여준다
//! (Demonstrates the SICP metacircular evaluator implemented in Rust).
//!
//! 실행 방법 (How to run): cargo run --example metacircular_evaluator

use sicp_chapter4::section_4_1::{CondClause, Expr, eval, setup_environment};

fn main() {
    println!("=== Rust로 구현된 SICP 메타순환적 평가기 (Metacircular Evaluator) ===\n");

    let env = setup_environment();

    // 예제 1: 단순 산술 연산 (Example 1: simple arithmetic)
    println!("1. 산술 연산 (Arithmetic): (+ (* 2 3) (- 10 5))");
    let expr = Expr::Application {
        operator: Box::new(Expr::Symbol("+".to_string())),
        operands: vec![
            Expr::Application {
                operator: Box::new(Expr::Symbol("*".to_string())),
                operands: vec![Expr::Number(2), Expr::Number(3)],
            },
            Expr::Application {
                operator: Box::new(Expr::Symbol("-".to_string())),
                operands: vec![Expr::Number(10), Expr::Number(5)],
            },
        ],
    };
    match eval(&expr, env.clone()) {
        Ok((result, _)) => println!("   결과 (Result): {}\n", result),
        Err(e) => println!("   오류 (Error): {}\n", e),
    }

    // 예제 2: 함수 정의 (팩토리얼) (Example 2: factorial definition)
    println!("2. 팩토리얼 정의 (Factorial definition):");
    println!("   (define factorial");
    println!("     (lambda (n)");
    println!("       (if (= n 0) 1 (* n (factorial (- n 1))))))");

    let factorial_def = Expr::Define {
        name: "factorial".to_string(),
        value: Box::new(Expr::Lambda {
            params: vec!["n".to_string()],
            body: vec![Expr::If {
                predicate: Box::new(Expr::Application {
                    operator: Box::new(Expr::Symbol("=".to_string())),
                    operands: vec![Expr::Symbol("n".to_string()), Expr::Number(0)],
                }),
                consequent: Box::new(Expr::Number(1)),
                alternative: Box::new(Expr::Application {
                    operator: Box::new(Expr::Symbol("*".to_string())),
                    operands: vec![
                        Expr::Symbol("n".to_string()),
                        Expr::Application {
                            operator: Box::new(Expr::Symbol("factorial".to_string())),
                            operands: vec![Expr::Application {
                                operator: Box::new(Expr::Symbol("-".to_string())),
                                operands: vec![Expr::Symbol("n".to_string()), Expr::Number(1)],
                            }],
                        },
                    ],
                }),
            }],
        }),
    };

    // 정의를 통해 환경 업데이트 (Update environment via definition)
    let (_, env) = eval(&factorial_def, env).unwrap();

    println!("   (factorial 6)");
    let factorial_call = Expr::Application {
        operator: Box::new(Expr::Symbol("factorial".to_string())),
        operands: vec![Expr::Number(6)],
    };
    match eval(&factorial_call, env.clone()) {
        Ok((result, _)) => println!("   결과 (Result): {}\n", result),
        Err(e) => println!("   오류 (Error): {}\n", e),
    }

    // 예제 3: 클로저와 고차 함수 (Example 3: closures and higher-order functions)
    println!("3. 클로저 (Closure) - make-adder:");
    println!("   (define make-adder (lambda (x) (lambda (y) (+ x y))))");

    let make_adder_def = Expr::Define {
        name: "make-adder".to_string(),
        value: Box::new(Expr::Lambda {
            params: vec!["x".to_string()],
            body: vec![Expr::Lambda {
                params: vec!["y".to_string()],
                body: vec![Expr::Application {
                    operator: Box::new(Expr::Symbol("+".to_string())),
                    operands: vec![Expr::Symbol("x".to_string()), Expr::Symbol("y".to_string())],
                }],
            }],
        }),
    };
    let (_, env) = eval(&make_adder_def, env).unwrap();

    println!("   (define add10 (make-adder 10))");
    let add10_def = Expr::Define {
        name: "add10".to_string(),
        value: Box::new(Expr::Application {
            operator: Box::new(Expr::Symbol("make-adder".to_string())),
            operands: vec![Expr::Number(10)],
        }),
    };
    let (_, env) = eval(&add10_def, env).unwrap();

    println!("   (add10 32)");
    let add10_call = Expr::Application {
        operator: Box::new(Expr::Symbol("add10".to_string())),
        operands: vec![Expr::Number(32)],
    };
    match eval(&add10_call, env.clone()) {
        Ok((result, _)) => println!("   결과 (Result): {}\n", result),
        Err(e) => println!("   오류 (Error): {}\n", e),
    }

    // 예제 4: Cond (파생 표현식) (Example 4: cond)
    println!("4. Cond 표현식 (Derived expression):");
    println!("   (cond ((< 5 3) 'less)");
    println!("         ((> 5 3) 'greater)");
    println!("         (else 'equal))");

    let cond_expr = Expr::Cond(vec![
        CondClause {
            predicate: Expr::Application {
                operator: Box::new(Expr::Symbol("<".to_string())),
                operands: vec![Expr::Number(5), Expr::Number(3)],
            },
            actions: vec![Expr::Quote(Box::new(Expr::Symbol("less".to_string())))],
        },
        CondClause {
            predicate: Expr::Application {
                operator: Box::new(Expr::Symbol(">".to_string())),
                operands: vec![Expr::Number(5), Expr::Number(3)],
            },
            actions: vec![Expr::Quote(Box::new(Expr::Symbol("greater".to_string())))],
        },
        CondClause {
            predicate: Expr::Symbol("else".to_string()),
            actions: vec![Expr::Quote(Box::new(Expr::Symbol("equal".to_string())))],
        },
    ]);

    match eval(&cond_expr, env.clone()) {
        Ok((result, _)) => println!("   결과 (Result): {}\n", result),
        Err(e) => println!("   오류 (Error): {}\n", e),
    }

    // 예제 5: Let (파생 표현식) (Example 5: let)
    println!("5. Let 표현식 (Derived expression):");
    println!("   (let ((x 10) (y 20))");
    println!("     (* (+ x y) 2))");

    let let_expr = Expr::Let {
        bindings: vec![
            ("x".to_string(), Expr::Number(10)),
            ("y".to_string(), Expr::Number(20)),
        ],
        body: vec![Expr::Application {
            operator: Box::new(Expr::Symbol("*".to_string())),
            operands: vec![
                Expr::Application {
                    operator: Box::new(Expr::Symbol("+".to_string())),
                    operands: vec![Expr::Symbol("x".to_string()), Expr::Symbol("y".to_string())],
                },
                Expr::Number(2),
            ],
        }],
    };

    match eval(&let_expr, env) {
        Ok((result, _)) => println!("   결과 (Result): {}\n", result),
        Err(e) => println!("   오류 (Error): {}\n", e),
    }

    println!("=== 주요 Rust 개념 데모 (Key Rust concepts) ===");
    println!("1. 대수적 데이터 타입(ADT): 철저한 매칭(exhaustive matching)이 가능한 Expr/Value 열거형");
    println!("2. 영속적 데이터 구조: O(1) 클론 환경을 위한 im::HashMap");
    println!("3. 함수형 상태 전달: eval은 (Value, Environment)를 반환");
    println!("4. 소유된 클로저: 람다는 참조가 아닌 복제로 환경을 캡처");
    println!("5. 에러 처리: 타입 안전한 에러를 위한 Result<(Value, Environment), EvalError>");
}
