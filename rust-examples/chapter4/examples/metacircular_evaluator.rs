//! Example: Using the Metacircular Evaluator
//!
//! Demonstrates the SICP metacircular evaluator implemented in Rust.
//!
//! Run with: cargo run --example metacircular_evaluator

use sicp_chapter4::section_4_1::{CondClause, Expr, eval, setup_environment};

fn main() {
    println!("=== SICP Metacircular Evaluator in Rust ===\n");

    let env = setup_environment();

    // Example 1: Simple arithmetic
    println!("1. Arithmetic: (+ (* 2 3) (- 10 5))");
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
        Ok(result) => println!("   Result: {}\n", result),
        Err(e) => println!("   Error: {}\n", e),
    }

    // Example 2: Define a function (factorial)
    println!("2. Define factorial:");
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

    eval(&factorial_def, env.clone()).unwrap();

    println!("   (factorial 6)");
    let factorial_call = Expr::Application {
        operator: Box::new(Expr::Symbol("factorial".to_string())),
        operands: vec![Expr::Number(6)],
    };
    match eval(&factorial_call, env.clone()) {
        Ok(result) => println!("   Result: {}\n", result),
        Err(e) => println!("   Error: {}\n", e),
    }

    // Example 3: Closures and higher-order functions
    println!("3. Closures - make-adder:");
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
    eval(&make_adder_def, env.clone()).unwrap();

    println!("   (define add10 (make-adder 10))");
    let add10_def = Expr::Define {
        name: "add10".to_string(),
        value: Box::new(Expr::Application {
            operator: Box::new(Expr::Symbol("make-adder".to_string())),
            operands: vec![Expr::Number(10)],
        }),
    };
    eval(&add10_def, env.clone()).unwrap();

    println!("   (add10 32)");
    let add10_call = Expr::Application {
        operator: Box::new(Expr::Symbol("add10".to_string())),
        operands: vec![Expr::Number(32)],
    };
    match eval(&add10_call, env.clone()) {
        Ok(result) => println!("   Result: {}\n", result),
        Err(e) => println!("   Error: {}\n", e),
    }

    // Example 4: Cond (derived expression)
    println!("4. Cond expression:");
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
        Ok(result) => println!("   Result: {}\n", result),
        Err(e) => println!("   Error: {}\n", e),
    }

    // Example 5: Let (derived expression)
    println!("5. Let expression:");
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
        Ok(result) => println!("   Result: {}\n", result),
        Err(e) => println!("   Error: {}\n", e),
    }

    println!("=== Key Rust Concepts Demonstrated ===");
    println!("1. Algebraic Data Types: Expr and Value enums with exhaustive matching");
    println!("2. Ownership & Borrowing: Rc<RefCell<Environment>> for shared mutable state");
    println!("3. Pattern Matching: eval() dispatches on Expr variants");
    println!("4. Closures: Lambda captures environment via Rc clone");
    println!("5. Error Handling: Result<Value, EvalError> for type-safe errors");
}
