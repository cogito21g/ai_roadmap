### 12주차: 가상 머신 구현

**강의 목표:**  
스택 기반 가상 머신을 구현하고, 가상 머신 명령어 집합을 설계하여 실제로 가상 머신을 동작시키는 방법을 학습합니다.

---

**강의 내용:**

1. **스택 기반 가상 머신 복습 (10분)**
   - 스택 기반 가상 머신의 개념과 구조 복습
   - 스택의 역할과 주요 명령어 세트 복습

2. **가상 머신 명령어 집합 설계 (20분)**
   - 기본 명령어 집합 설계 (PUSH, POP, ADD, SUB 등)
   - 명령어 집합 확장 (MUL, DIV, PRINT 등)
   - 명령어 집합 설계 원칙 및 예제

3. **가상 머신 구현 (25분)**
   - 가상 머신 명령어 처리 루프 구현
   - 명령어별 함수 구현 (PUSH, POP, ADD, SUB 등)
   - 예제 프로그램 작성 및 실행

---

**강의 진행:**

1. **강의 시작 (5분)**
   - 강의 목표와 주제를 소개합니다.
   - 이번 주차에 다룰 내용에 대한 개요를 설명합니다.

2. **스택 기반 가상 머신 복습 (10분)**
   - 스택 기반 가상 머신의 개념과 구조를 복습합니다.
   - 스택의 역할과 주요 명령어 세트를 복습합니다.
   - Q&A를 통해 개념을 확인합니다.

3. **가상 머신 명령어 집합 설계 (20분)**
   - 기본 명령어 집합을 설계하는 방법을 설명합니다.
   - 명령어 집합을 확장하는 방법을 설명합니다.
   - 명령어 집합 설계 원칙과 예제를 설명합니다.

4. **가상 머신 구현 (25분)**
   - 가상 머신 명령어 처리 루프를 구현하는 방법을 설명합니다.
   - 각 명령어별 함수를 구현하는 방법을 설명합니다.
   - 예제 프로그램을 작성하고 실행하는 과정을 시연합니다.

5. **실습 준비 및 안내 (10분)**
   - 실습 내용을 안내하고 실습 방법을 설명합니다.
   - 실습 과제를 발표하고 Q&A 시간을 갖습니다.

---

**실습 내용:**

1. **스택 기반 가상 머신 구현**

**실습 과제:**
- 스택 기반 가상 머신을 구현하고, 주어진 명령어 집합을 처리하는 가상 머신을 작성합니다.
- 예제 프로그램을 작성하여 가상 머신이 올바르게 동작하는지 확인합니다.

**실습 예제:**

**가상 머신 명령어 집합:**

```plaintext
PUSH n    # 스택에 n을 푸시
POP       # 스택에서 값을 팝
ADD       # 스택에서 두 값을 팝하여 더한 후 결과를 푸시
SUB       # 스택에서 두 값을 팝하여 뺀 후 결과를 푸시
MUL       # 스택에서 두 값을 팝하여 곱한 후 결과를 푸시
DIV       # 스택에서 두 값을 팝하여 나눈 후 결과를 푸시
PRINT     # 스택의 최상단 값을 출력
```

**가상 머신 구현 코드:**

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STACK_SIZE 100

int stack[STACK_SIZE];
int sp = -1; // 스택 포인터

void push(int value) {
    if (sp < STACK_SIZE - 1) {
        stack[++sp] = value;
    } else {
        printf("Stack overflow\n");
    }
}

int pop() {
    if (sp >= 0) {
        return stack[sp--];
    } else {
        printf("Stack underflow\n");
        return -1;
    }
}

void add() {
    int b = pop();
    int a = pop();
    push(a + b);
}

void sub() {
    int b = pop();
    int a = pop();
    push(a - b);
}

void mul() {
    int b = pop();
    int a = pop();
    push(a * b);
}

void divide() {
    int b = pop();
    int a = pop();
    push(a / b);
}

void print_stack() {
    if (sp >= 0) {
        printf("Top of stack: %d\n", stack[sp]);
    } else {
        printf("Stack is empty\n");
    }
}

void execute(const char *code) {
    char instruction[10];
    int num;
    const char *ptr = code;

    while (sscanf(ptr, "%s", instruction) != EOF) {
        ptr += strlen(instruction) + 1;

        if (strcmp(instruction, "PUSH") == 0) {
            sscanf(ptr, "%d", &num);
            push(num);
            while (*ptr != ' ' && *ptr != '\0') ptr++;
            if (*ptr == ' ') ptr++;
        } else if (strcmp(instruction, "POP") == 0) {
            pop();
        } else if (strcmp(instruction, "ADD") == 0) {
            add();
        } else if (strcmp(instruction, "SUB") == 0) {
            sub();
        } else if (strcmp(instruction, "MUL") == 0) {
            mul();
        } else if (strcmp(instruction, "DIV") == 0) {
            divide();
        } else if (strcmp(instruction, "PRINT") == 0) {
            print_stack();
        }
    }
}

int main() {
    const char *program = "PUSH 5 PUSH 10 ADD PRINT PUSH 20 SUB PRINT";
    execute(program);
    return 0;
}
```

**실습 목표:**
- 학생들이 스택 기반 가상 머신의 명령어 집합을 설계하고, 이를 실제로 구현하는 방법을 익힙니다.
- 가상 머신 명령어 집합을 처리하는 가상 머신을 작성하고, 예제 프로그램을 실행하여 결과를 확인합니다.

**제출물:**
- 작성한 스택 기반 가상 머신 코드
- 가상 머신 실행 결과 스크린샷

이 강의 계획을 통해 학생들은 가상 머신의 개념과 스택 기반 가상 머신의 구조 및 동작 원리를 학습하고, 간단한 가상 머신을 구현하여 그 동작을 이해할 수 있습니다.