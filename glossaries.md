# SICP-Rust Translation Glossary

## Scheme/Lisp Terms
*   **Procedure**: 프로시저 (Procedure)
*   **Abstraction**: 추상화 (Abstraction)
*   **Compound Data**: 복합 데이터 (Compound Data)
*   **Environment**: 환경 (Environment)
*   **Closure**: 클로저 (Closure)
*   **Cons**: Cons (Cons) - usually left as is or "Cons 셀"
*   **Car**: Car
*   **Cdr**: Cdr
*   **List**: 리스트 (List)
*   **Stream**: 스트림 (Stream)
*   **Higher-order procedure**: 고차 프로시저 (Higher-order procedure)
*   **Expression**: 표현식 (Expression)
*   **Evaluation**: 평가 (Evaluation)
*   **Applicative Order**: 적용적 순서 (Applicative Order)
*   **Normal Order**: 정규 순서 (Normal Order)
*   **Metalinguistic Abstraction**: 메타언어적 추상화 (Metalinguistic Abstraction)
*   **Continuation**: 연속성/컨티뉴에이션 (Continuation) - context dependent
*   **Register Machine**: 레지스터 기계 (Register Machine)

## Rust Terms
*   **Trait**: 트레이트 (Trait)
*   **Ownership**: 소유권 (Ownership)
*   **Borrow/Borrowing**: 빌림 (Borrowing)
*   **Lifetime**: 수명/라이프타임 (Lifetime)
*   **Crate**: 크레이트 (Crate)
*   **Struct**: 구조체 (Struct)
*   **Enum**: 열거형 (Enum)
*   **Iterator**: 반복자 (Iterator)
*   **Result**: 결과/Result (Result) - often kept as Result type
*   **Option**: 옵션/Option (Option) - often kept as Option type
*   **Pattern Matching**: 패턴 매칭 (Pattern Matching)
*   **Macro**: 매크로 (Macro)
*   **Smart Pointer**: 스마트 포인터 (Smart Pointer)
*   **Reference**: 참조 (Reference)

## General Computer Science
*   **Iterate**: 반복 (Iterate)
*   **Recursion**: 재귀 (Recursion)
*   **State**: 상태 (State)
*   **Mutation**: 변경 (Mutation)
*   **Concurrency**: 동시성 (Concurrency)
*   **Object**: 객체 (Object)
*   **Syntax**: 문법 (Syntax)
*   **Semantics**: 의미론 (Semantics)
*   **Interpreter**: 인터프리터 (Interpreter)
*   **Compiler**: 컴파일러 (Compiler)
*   **Garbage Collection**: 가비지 컬렉션 (Garbage Collection)

## Style Guide
*   **Main Text**: Plain form (~한다).
*   **Guides/READMEs**: Polite form (~합니다/해요).
*   **Every Occurrence**: Use "Korean Term (English Term)" every time.
*   **Scope**: Apply to narrative text, headings, comments, and bibliography entries.
*   **Code/Literals**: Translate code and literals to Korean with English in parentheses, while preserving Texinfo commands, @ref keys, and node identifiers.
*   **Rust Source**: Translate doc comments, inline comments, and string literals; keep identifiers and Rust keywords unchanged for compilability.
*   **Markdown Code Blocks**: Translate comments and string literals inside fenced blocks; keep identifiers and syntax intact.
