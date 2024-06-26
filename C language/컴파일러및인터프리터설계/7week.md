### 7주차: 코드 생성 개요

**강의 목표:**  
코드 생성의 기본 개념을 이해하고, 중간 코드를 생성하는 방법을 학습합니다.

---

**강의 내용:**

1. **코드 생성의 기본 개념 (15분)**
   - 코드 생성의 역할과 중요성
   - 중간 코드와 타깃 코드의 개념
   - 중간 코드의 필요성 및 장점

2. **중간 코드 생성 (20분)**
   - 중간 코드의 형태와 예제
   - 중간 코드 생성 규칙
   - 중간 코드에서 타깃 코드로의 변환

3. **중간 코드의 유형 (20분)**
   - 세 주소 코드 (Three-Address Code)
   - 스택 머신 코드
   - 트리 구조 코드
   - 각 유형의 예제와 사용 사례

---

**강의 진행:**

1. **강의 시작 (5분)**
   - 강의 목표와 주제를 소개합니다.
   - 이번 주차에 다룰 내용에 대한 개요를 설명합니다.

2. **코드 생성의 기본 개념 (15분)**
   - 코드 생성의 역할과 중요성을 설명합니다.
   - 중간 코드와 타깃 코드의 개념을 설명합니다.
   - 중간 코드의 필요성과 장점을 설명합니다.
   - Q&A를 통해 개념을 확인합니다.

3. **중간 코드 생성 (20분)**
   - 중간 코드의 형태와 예제를 설명합니다.
   - 중간 코드 생성 규칙을 설명합니다.
   - 중간 코드에서 타깃 코드로의 변환 과정을 설명합니다.
   - 각 과정에 대한 예제를 시연합니다.

4. **중간 코드의 유형 (20분)**
   - 세 주소 코드 (Three-Address Code)의 개념과 예제를 설명합니다.
   - 스택 머신 코드의 개념과 예제를 설명합니다.
   - 트리 구조 코드의 개념과 예제를 설명합니다.
   - 각 유형의 예제와 사용 사례를 시연합니다.

5. **실습 준비 및 안내 (10분)**
   - 실습 내용을 안내하고 실습 방법을 설명합니다.
   - 실습 과제를 발표하고 Q&A 시간을 갖습니다.

---

**실습 내용:**

1. **중간 코드 생성 실습**

**실습 과제:**
- 주어진 소스 코드에서 중간 코드를 생성합니다.
- 세 주소 코드, 스택 머신 코드, 트리 구조 코드 중 하나를 선택하여 중간 코드를 작성합니다.

**실습 예제:**

**세 주소 코드 예제:**

소스 코드:
```c
int a = 5;
int b = 10;
int c = a + b * 2;
```

세 주소 코드:
```
t1 = 5
t2 = 10
t3 = t2 * 2
t4 = t1 + t3
c = t4
```

**스택 머신 코드 예제:**

소스 코드:
```c
int a = 5;
int b = 10;
int c = a + b * 2;
```

스택 머신 코드:
```
PUSH 5
STORE a
PUSH 10
STORE b
LOAD b
PUSH 2
MUL
LOAD a
ADD
STORE c
```

**트리 구조 코드 예제:**

소스 코드:
```c
int a = 5;
int b = 10;
int c = a + b * 2;
```

트리 구조 코드:
```
     +
    / \
   a   *
      / \
     b   2
```

**실습 목표:**
- 학생들이 중간 코드를 생성하고, 다양한 유형의 중간 코드를 이해할 수 있도록 합니다.
- 중간 코드 생성 규칙을 적용하여 실제 예제를 통해 중간 코드를 작성하는 경험을 합니다.

**제출물:**
- 작성한 중간 코드
- 중간 코드 설명 문서

이 강의 계획을 통해 학생들은 코드 생성의 기본 개념과 중간 코드 생성 방법을 학습하고, 다양한 유형의 중간 코드를 작성하는 경험을 통해 코드 생성의 중요성을 이해할 수 있습니다.