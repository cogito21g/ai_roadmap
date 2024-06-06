### 13주차: 인터프리터 개요

**강의 목표:**  
인터프리터의 개념을 이해하고, 인터프리터와 컴파일러의 차이점을 학습합니다. 간단한 인터프리터를 설계하고 구현하는 방법을 익힙니다.

---

**강의 내용:**

1. **인터프리터의 개념 (15분)**
   - 인터프리터의 정의와 역할
   - 인터프리터와 컴파일러의 차이점
   - 인터프리터의 장단점

2. **인터프리터 구조 (20분)**
   - 인터프리터의 기본 구조
   - 소스 코드 분석 및 실행 과정
   - 주요 컴포넌트 (어휘 분석기, 구문 분석기, 실행 엔진)

3. **간단한 인터프리터 설계 (20분)**
   - 인터프리터 설계 원칙
   - 주요 데이터 구조와 알고리즘
   - 예제 인터프리터 설계

---

**강의 진행:**

1. **강의 시작 (5분)**
   - 강의 목표와 주제를 소개합니다.
   - 이번 주차에 다룰 내용에 대한 개요를 설명합니다.

2. **인터프리터의 개념 (15분)**
   - 인터프리터의 정의와 역할을 설명합니다.
   - 인터프리터와 컴파일러의 차이점을 비교 설명합니다.
   - 인터프리터의 장단점을 설명합니다.
   - Q&A를 통해 개념을 확인합니다.

3. **인터프리터 구조 (20분)**
   - 인터프리터의 기본 구조를 설명합니다.
   - 소스 코드 분석 및 실행 과정을 설명합니다.
   - 인터프리터의 주요 컴포넌트 (어휘 분석기, 구문 분석기, 실행 엔진)를 설명합니다.

4. **간단한 인터프리터 설계 (20분)**
   - 인터프리터 설계 원칙을 설명합니다.
   - 주요 데이터 구조와 알고리즘을 설명합니다.
   - 간단한 인터프리터 설계 예제를 시연합니다.

5. **실습 준비 및 안내 (10분)**
   - 실습 내용을 안내하고 실습 방법을 설명합니다.
   - 실습 과제를 발표하고 Q&A 시간을 갖습니다.

---

**실습 내용:**

1. **간단한 인터프리터 구현**

**실습 과제:**
- 간단한 인터프리터를 설계하고 구현합니다.
- 주어진 소스 코드를 분석하고 실행하는 인터프리터를 작성합니다.

**실습 예제:**

**인터프리터 예제 코드:**

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { NUMBER, PLUS, MINUS, END } TokenType;

typedef struct {
    TokenType type;
    int value;
} Token;

Token get_next_token(const char **input) {
    while (**input == ' ') (*input)++;

    if (**input >= '0' && **input <= '9') {
        int value = 0;
        while (**input >= '0' && **input <= '9') {
            value = value * 10 + (**input - '0');
            (*input)++;
        }
        return (Token){NUMBER, value};
    }

    if (**input == '+') {
        (*input)++;
        return (Token){PLUS, 0};
    }

    if (**input == '-') {
        (*input)++;
        return (Token){MINUS, 0};
    }

    return (Token){END, 0};
}

int eval(const char *input) {
    Token current_token = get_next_token(&input);
    int result = current_token.value;

    while (1) {
        current_token = get_next_token(&input);

        if (current_token.type == END) break;

        Token next_token = get_next_token(&input);

        if (current_token.type == PLUS) {
            result += next_token.value;
        } else if (current_token.type == MINUS) {
            result -= next_token.value;
        }
    }

    return result;
}

int main() {
    const char *program = "10 + 20 - 5";
    int result = eval(program);
    printf("Result: %d\n", result);
    return 0;
}
```

**실습 목표:**
- 학생들이 인터프리터의 개념과 구조를 이해하고, 간단한 인터프리터를 설계하고 구현하는 방법을 익힙니다.
- 주어진 소스 코드를 분석하고 실행하는 인터프리터를 작성하여 그 동작을 이해합니다.

**제출물:**
- 작성한 인터프리터 코드
- 인터프리터 실행 결과 스크린샷

이 강의 계획을 통해 학생들은 인터프리터의 개념과 구조를 학습하고, 간단한 인터프리터를 설계하고 구현하여 그 동작을 이해할 수 있습니다.