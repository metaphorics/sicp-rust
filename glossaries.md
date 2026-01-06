# SICP-Rust Translation Glossary

## Scheme/Lisp Terms

- **Procedure**: 프로시저 (Procedure)
- **Abstraction**: 추상화 (Abstraction)
- **Compound Data**: 복합 데이터 (Compound Data)
- **Environment**: 환경 (Environment)
- **Closure**: 클로저 (Closure)
- **Cons**: Cons (Cons) - usually left as is or "Cons 셀"
- **Car**: Car
- **Cdr**: Cdr
- **List**: 리스트 (List)
- **Stream**: 스트림 (Stream)
- **Higher-order procedure**: 고차 프로시저 (Higher-order procedure)
- **Expression**: 표현식 (Expression)
- **Evaluation**: 평가 (Evaluation)
- **Applicative Order**: 적용적 순서 (Applicative Order)
- **Normal Order**: 정규 순서 (Normal Order)
- **Metalinguistic Abstraction**: 메타언어적 추상화 (Metalinguistic Abstraction)
- **Continuation**: 연속성/컨티뉴에이션 (Continuation) - context dependent
- **Register Machine**: 레지스터 기계 (Register Machine)

## Rust Terms

- **Trait**: 트레이트 (Trait)
- **Ownership**: 소유권 (Ownership)
- **Borrow/Borrowing**: 빌림 (Borrowing)
- **Lifetime**: 수명/라이프타임 (Lifetime)
- **Crate**: 크레이트 (Crate)
- **Struct**: 구조체 (Struct)
- **Enum**: 열거형 (Enum)
- **Iterator**: 반복자 (Iterator)
- **Result**: 결과/Result (Result) - often kept as Result type
- **Option**: 옵션/Option (Option) - often kept as Option type
- **Pattern Matching**: 패턴 매칭 (Pattern Matching)
- **Macro**: 매크로 (Macro)
- **Smart Pointer**: 스마트 포인터 (Smart Pointer)
- **Reference**: 참조 (Reference)
- **Newtype Pattern**: 뉴타입 패턴 (Newtype Pattern)
- **Zero-Cost Type Safety**: 제로 비용 타입 안전성 (Zero-Cost Type Safety)
- **Orphan Rule**: 고아 규칙 (Orphan Rule)
- **Type State Pattern**: 타입 상태 패턴 (Type State Pattern)
- **Send**: Send (Send)
- **Sync**: Sync (Sync)
- **Async/Await**: 비동기/대기 (Async/Await)
- **Future**: 퓨처 (Future)
- **Executor**: 실행자 (Executor)
- **Pin**: 핀 (Pin)
- **Stream**: 스트림 (Stream) - Rust trait context
- **Atomic**: 원자적 (Atomic)
- **Memory Ordering**: 메모리 순서 (Memory Ordering)
- **RAII**: RAII (자원 획득이 초기화)
- **Drop**: Drop (Drop)
- **Rc (Reference Counted)**: Rc (참조 카운트)
- **Arc (Atomic Reference Counted)**: Arc (원자적 참조 카운트)
- **Borrow Checker**: 빌림 검사기 (Borrow Checker)
- **Declarative Macro**: 선언적 매크로 (Declarative Macro)
- **Procedural Macro**: 절차적 매크로 (Procedural Macro)

## General Computer Science

- **Iterate**: 반복 (Iterate)
- **Recursion**: 재귀 (Recursion)
- **State**: 상태 (State)
- **Mutation**: 변경 (Mutation)
- **Concurrency**: 동시성 (Concurrency)
- **Object**: 객체 (Object)
- **Syntax**: 문법 (Syntax)
- **Semantics**: 의미론 (Semantics)
- **Interpreter**: 인터프리터 (Interpreter)
- **Compiler**: 컴파일러 (Compiler)
- **Garbage Collection**: 가비지 컬렉션 (Garbage Collection)
- **Coercion**: 강제 변환 (Coercion)
- **Hierarchy of types**: 타입 계층 (Hierarchy of types)
- **Serializer**: 직렬화기 (Serializer)
- **Mutex**: 뮤텍스 (Mutex)
- **Deadlock**: 데드락/교착 상태 (Deadlock)
- **Barrier synchronization**: 장벽 동기화 (Barrier synchronization)
- **Fearless concurrency**: 두려움 없는 동시성 (Fearless concurrency)
- **Data race**: 데이터 레이스/경합 (Data race)
- **Cooperative concurrency**: 협력적 동시성 (Cooperative concurrency)
- **Preemptive multitasking**: 선점형 멀티태스킹 (Preemptive multitasking)
- **Cancellation safety**: 취소 안전성 (Cancellation safety)
- **Lock-free**: 락 프리 (Lock-free)
- **Compare-and-Swap (CAS)**: 비교 교환 (Compare-and-Swap)
- **ABA problem**: ABA 문제 (ABA problem)
- **Epoch-based reclamation**: 에포크 기반 회수 (Epoch-based reclamation)
- **Metacircular evaluator**: 메타순환 평가자 (Metacircular evaluator)
- **Abstract syntax**: 추상 구문 (Abstract syntax)
- **Thunk**: 썽크 (Thunk)
- **Memoization**: 메모이제이션 (Memoization)
- **Persistent data structure**: 영속적 데이터 구조 (Persistent data structure)
- **Structural sharing**: 구조적 공유 (Structural sharing)
- **Copy-on-write**: 쓰기 시 복사 (Copy-on-write)
- **Nondeterministic computing**: 비결정적 컴퓨팅 (Nondeterministic computing)
- **Backtracking**: 백트래킹 (Backtracking)
- **Unification**: 단일화 (Unification)
- **Logic programming**: 논리 프로그래밍 (Logic programming)
- **Query language**: 질의 언어 (Query language)
- **Register machine**: 레지스터 기계 (Register machine)
- **Data path**: 데이터 경로 (Data path)
- **Controller**: 컨트롤러 (Controller)
- **Assembler**: 어셈블러 (Assembler)
- **Stack machine**: 스택 기계 (Stack machine)
- **WebAssembly (WASM)**: 웹어셈블리 (WebAssembly)
- **Linear memory**: 선형 메모리 (Linear memory)
- **Stop-and-copy**: 정지 후 복사 (Stop-and-copy)
- **Root**: 루트 (Root) - in GC context
- **Working memory**: 작업 메모리 (Working memory)
- **Free memory**: 자유 메모리 (Free memory)

## Style Guide

- **Main Text**: Plain form (~한다).
- **Guides/READMEs**: Polite form (~합니다/해요).
- **Every Occurrence**: Use "Korean Term (English Term)" every time.
- **Scope**: Apply to narrative text, headings, comments, and bibliography entries.
- **Code/Literals**: Translate code and literals to Korean with English in parentheses, while preserving Texinfo commands, @ref keys, and node identifiers.
- **Rust Source**: Translate doc comments, inline comments, and string literals; keep identifiers and Rust keywords unchanged for compilability.
- **Markdown Code Blocks**: Translate comments and string literals inside fenced blocks; keep identifiers and syntax intact.
- **Identifiers**: Keep variable/function/type/module names in English across all code and code examples.
