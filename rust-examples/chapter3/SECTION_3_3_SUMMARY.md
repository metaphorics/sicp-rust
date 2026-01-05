# SICP 3.3장: 가변 데이터를 이용한 모델링 - Rust 구현 (Modeling with Mutable Data - Rust Implementation)

## 개요 (Overview)

이 구현은 SICP 3.3절을 Scheme에서 Rust로 변환하여 다음을 보여줍니다:

- 내부 가변성(interior mutability)을 사용한 가변 데이터 구조
- 큐(Queue)와 데크(Deque) 구현
- 해시 테이블 (1차원 및 2차원)
- 이벤트 주도 디지털 회로 시뮬레이터
- 제약 조건 전파(Constraint propagation) 네트워크

## 구현 세부 사항 (Implementation Details)

### 3.3.1 가변 리스트 구조 (Mutable List Structure)

**Rust 개념:**

- `MutablePair`에서 내부 가변성을 위해 `RefCell<T>` 사용
- 공유 가능한 가변 연결 리스트를 위해 `Rc<RefCell<Cons<T>>>` 사용
- 플로이드 알고리즘(Floyd's algorithm)을 사용한 순환(cycle) 탐지

**핵심 함수:**

- `append_mut` - 파괴적(destructive) 추가 연산
- `mystery` - 변경을 통한 리스트 뒤집기 (연습문제 3.14)
- `has_cycle` - 순환 탐지 (연습문제 3.18/3.19)
- `count_pairs_correct` - HashSet을 사용하여 고유한 쌍의 개수 세기 (연습문제 3.17)

### 3.3.2 큐 표현 (Representing Queues)

**Rust 매핑:**

- SICP 쌍(pair) → `Rc<RefCell<Cons<T>>>`와 `Weak` 후방(rear) 포인터 사용
- 관용적인 데크 구현을 위해 `VecDeque` 사용 (연습문제 3.23)

**큐 연산:** O(1) 삽입, 삭제, 전방(front) 조회

### 3.3.3 테이블 표현 (Representing Tables)

**Rust 매핑:**

- 1차원 테이블 → `HashMap<K, V>`
- 2차원 테이블 → `HashMap<K1, HashMap<K2, V>>`
- `RefCell<HashMap>`을 사용한 메모이제이션(Memoization) (연습문제 3.27)

### 3.3.4 디지털 회로 시뮬레이터 (Digital Circuit Simulator)

**아키텍처:**

- `Wire` - 신호값과 액션 콜백(`Vec<Box<dyn FnMut()>>`)을 보유
- `Agenda` - 시간 세그먼트를 가진 이벤트 주도 시간 스케줄러
- 게이트: `inverter`, `and_gate`, `or_gate`
- 복합 회로: `half_adder`, `full_adder`

**한계점:**

- 액션 클로저가 작업을 캡처하지만 전파에는 한계가 있음
- 실제 프로덕션 시스템에서는 메시지 전달이나 이벤트 큐를 사용해야 함

### 3.3.5 제약 조건 전파 (Constraint Propagation)

**설계:**

- `Connector` - 내부 가변성을 위해 `RefCell`로 값을 보유
- `Adder`, `Multiplier`, `Constant`를 위한 `Constraint` 트레이트
- 양방향 계산 (예: 섭씨 ↔ 화씨)

**Rust에서의 도전 과제:**

- 제약 조건이 값을 전파할 때 순환 `RefCell` 빌림 문제 발생
- **해결책:** 문서화된 한계점; 프로덕션 시스템은 다음을 사용해야 함:
  - 메시지 전달 (채널)
  - 이벤트 큐
  - 액터 모델 패턴

## 테스트 커버리지 (Test Coverage)

- 모든 하위 섹션을 아우르는 16개의 포괄적인 테스트
- 자동 전파가 제한적인 부분에서도 구조를 보여주는 테스트
- 연습문제 3.12-3.37 모두 해결

## Rust와 Scheme의 트레이드오프 (Rust vs Scheme Trade-offs)

**장점:**

- 컴파일 타임에 강제되는 메모리 안전성
- 결정론적 정리를 위해 가비지 컬렉션이 필요 없음
- 명시적 소유권으로 데이터 흐름이 명확해짐

**도전 과제:**

- 순환 데이터 구조는 세심한 `Rc`/`Weak` 사용이 필요함
- 내부 가변성(`RefCell`)은 런타임 오버헤드가 있음
- 제약 조건 전파는 빌림 검사기(borrow checker)와 충돌함

## 프로덕션 권장 사항 (Production Recommendations)

실제 Rust 구현을 위한 조언:

1. **가변 리스트:** `Vec<T>` 또는 영속적 데이터 구조 사용
2. **큐:** `VecDeque<T>` 사용 (이미 완료됨)
3. **테이블:** `HashMap` 사용 (이미 완료됨)
4. **이벤트 시스템:** `tokio`, `async-std` 또는 액터 프레임워크 사용
5. **제약 조건:** `crossbeam-channel` 또는 `tokio::sync::mpsc`를 통한 메시지 전달 사용

## 학습 성과 (Learning Outcomes)

이 구현을 통해 다음을 배울 수 있습니다:

- `RefCell`을 사용할 때와 소유권 중심으로 재설계할 때를 구분하는 법
- `Rc`/`Weak`를 사용한 순환 참조 처리
- Rust에서의 이벤트 주도 아키텍처
- Scheme의 유연성과 Rust의 안전성 사이의 트레이드오프

## 파일 (Files)

- `src/section_3_3.rs` - 전체 구현 (약 1,320행)
- 포괄적인 테스트와 통합된 모든 연습문제
