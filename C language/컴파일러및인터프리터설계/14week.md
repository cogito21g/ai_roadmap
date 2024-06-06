### 14주차: 인터프리터 구현

**강의 목표:**  
간단한 프로그래밍 언어를 위한 인터프리터를 구현하여, 소스 코드를 실행하는 방법을 학습합니다.

---

**강의 내용:**

1. **인터프리터 복습 (10분)**
   - 인터프리터의 개념과 구조 복습
   - 인터프리터와 컴파일러의 차이점 복습

2. **문법 및 구문 분석 (20분)**
   - 간단한 문법 정의 (BNF 또는 EBNF 사용)
   - 구문 분석기의 역할과 동작 원리
   - 구문 분석기 구현

3. **실행 엔진 구현 (20분)**
   - 실행 엔진의 개념과 역할
   - 주요 데이터 구조와 알고리즘
   - 인터프리터의 실행 과정

---

**강의 진행:**

1. **강의 시작 (5분)**
   - 강의 목표와 주제를 소개합니다.
   - 이번 주차에 다룰 내용에 대한 개요를 설명합니다.

2. **인터프리터 복습 (10분)**
   - 인터프리터의 개념과 구조를 복습합니다.
   - 인터프리터와 컴파일러의 차이점을 복습합니다.
   - Q&A를 통해 개념을 확인합니다.

3. **문법 및 구문 분석 (20분)**
   - 간단한 문법을 정의하는 방법을 설명합니다.
   - 구문 분석기의 역할과 동작 원리를 설명합니다.
   - 간단한 구문 분석기 구현 예제를 시연합니다.

4. **실행 엔진 구현 (20분)**
   - 실행 엔진의 개념과 역할을 설명합니다.
   - 주요 데이터 구조와 알고리즘을 설명합니다.
   - 인터프리터의 실행 과정을 설명하고 예제를 시연합니다.

5. **실습 준비 및 안내 (10분)**
   - 실습 내용을 안내하고 실습 방법을 설명합니다.
   - 실습 과제를 발표하고 Q&A 시간을 갖습니다.

---

**실습 내용:**

1. **간단한 인터프리터 구현**

**실습 과제:**
- 간단한 프로그래밍 언어를 위한 인터프리터를 구현합니다.
- 주어진 소스 코드를 분석하고 실행하는 인터프리터를 작성합니다.

**실습 예제:**

**간단한 문법 정의:**

```plaintext
program   ::= statement+
statement ::= "PRINT" expr
expr      ::= term (("+"|"-") term)*
term      ::= factor (("*"|"/") factor)*
factor    ::= NUMBER | "(" expr ")"
```

**구문 분석기 구현:**

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { NUMBER, PLUS, MINUS, MUL, DIV, LPAREN, RPAREN, PRINT, END } TokenType;

typedef struct {
    TokenType type;
    int value;
} Token;

Token get_next_token(const char **input);
void parse(const char *input);

int main() {
    const char *program = "PRINT 10 + 20 * 3";
    parse(program);
    return 0;
}

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
    if (**input == '+') { (*input)++; return (Token){PLUS, 0}; }
    if (**input == '-') { (*input)++; return (Token){MINUS, 0}; }
    if (**input == '*') { (*input)++; return (Token){MUL, 0}; }
    if (**input == '/') { (*input)++; return (Token){DIV, 0}; }
    if (**input == '(') { (*input)++; return (Token){LPAREN, 0}; }
    if (**input == ')') { (*input)++; return (Token){RPAREN, 0}; }
    if (strncmp(*input, "PRINT", 5) == 0) { *input += 5; return (Token){PRINT, 0}; }
    return (Token){END, 0};
}

void parse(const char *input) {
    Token token = get_next_token(&input);
    if (token.type == PRINT) {
        token = get_next_token(&input);
        int result = expr(&token, &input);
        printf("Result: %d\n", result);
    }
}

int expr(Token *token, const char **input);
int term(Token *token, const char **input);
int factor(Token *token, const char **input);

int expr(Token *token, const char **input) {
    int result = term(token, input);
    while (token->type == PLUS || token->type == MINUS) {
        TokenType op = token->type;
        *token = get_next_token(input);
        int right = term(token, input);
        if (op == PLUS) {
            result += right;
        } else {
            result -= right;
        }
    }
    return result;
}

int term(Token *token, const char **input) {
    int result = factor(token, input);
    while (token->type == MUL || token->type == DIV) {
        TokenType op = token->type;
        *token = get_next_token(input);
        int right = factor(token, input);
        if (op == MUL) {
            result *= right;
        } else {
            result /= right;
        }
    }
    return result;
}

int factor(Token *token, const char **input) {
    int result;
    if (token->type == NUMBER) {
        result = token->value;
        *token = get_next_token(input);
    } else if (token->type == LPAREN) {
        *token = get_next_token(input);
        result = expr(token, input);
        if (token->type != RPAREN) {
            fprintf(stderr, "Error: unmatched parenthesis\n");
            exit(EXIT_FAILURE);
        }
        *token = get_next_token(input);
    } else {
        fprintf(stderr, "Error: unexpected token\n");
        exit(EXIT_FAILURE);
    }
    return result;
}
```

**실습 목표:**
- 학생들이 인터프리터의 개념과 구조를 이해하고, 간단한 인터프리터를 설계하고 구현하는 방법을 익힙니다.
- 주어진 소스 코드를 분석하고 실행하는 인터프리터를 작성하여 그 동작을 이해합니다.

**제출물:**
- 작성한 인터프리터 코드
- 인터프리터 실행 결과 스크린샷

이 강의 계획을 통해 학생들은 인터프리터의 개념과 구조를 학습하고, 간단한 인터프리터를 설계하고 구현하여 그 동작을 이해할 수 있습니다.