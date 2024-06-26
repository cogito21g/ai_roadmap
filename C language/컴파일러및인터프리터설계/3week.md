### 3주차: 어휘 분석기 구현

**강의 목표:**  
Lex 도구를 사용하여 어휘 분석기를 구현하고, 이를 통해 소스 코드에서 토큰을 추출하는 방법을 학습합니다.

---

**강의 내용:**

1. **Lex 도구 소개 (15분)**
   - Lex 도구의 개념과 기능
   - Lex 도구 설치 및 설정 방법
   - Lex 파일 구조와 문법

2. **Lex 파일 작성 (20분)**
   - Lex 파일의 기본 구조
   - Lex 파일에 정규 표현식과 액션 추가
   - Lex 파일 컴파일 및 실행 방법

3. **어휘 분석기 구현 (20분)**
   - Lex 파일 작성 실습
   - 다양한 토큰 패턴 정의
   - 토큰 인식 후의 액션 정의

---

**강의 진행:**

1. **강의 시작 (5분)**
   - 강의 목표와 주제를 소개합니다.
   - 이번 주차에 다룰 내용에 대한 개요를 설명합니다.

2. **Lex 도구 소개 (15분)**
   - Lex 도구의 개념과 기능을 설명합니다.
   - Lex 도구 설치 및 설정 방법을 시연합니다.
   - Lex 파일 구조와 문법을 설명합니다.
   - 간단한 Lex 파일 예제를 통해 구조를 이해합니다.

3. **Lex 파일 작성 (20분)**
   - Lex 파일의 기본 구조를 설명합니다.
   - 정규 표현식과 액션을 추가하는 방법을 설명합니다.
   - Lex 파일을 컴파일하고 실행하는 방법을 시연합니다.

4. **어휘 분석기 구현 (20분)**
   - Lex 파일을 작성하여 어휘 분석기를 구현하는 과정을 시연합니다.
   - 다양한 토큰 패턴을 정의하고 인식된 토큰에 대한 액션을 정의합니다.
   - 작성한 Lex 파일을 컴파일하고 실행하여 결과를 확인합니다.

5. **실습 준비 및 안내 (10분)**
   - 실습 내용을 안내하고 실습 방법을 설명합니다.
   - 실습 과제를 발표하고 Q&A 시간을 갖습니다.

---

**실습 내용:**

1. **Lex 도구를 사용한 어휘 분석기 구현**

**실습 과제:**
- Lex 도구를 사용하여 간단한 어휘 분석기를 구현합니다.
- 다양한 토큰 패턴을 정의하고, 인식된 토큰에 대한 액션을 정의합니다.

**실습 예제:**

**Lex 파일 예제:**

```lex
%{
#include <stdio.h>
%}

digit      [0-9]
letter     [a-zA-Z]
identifier {letter}({letter}|{digit})*

%%

{identifier}   { printf("IDENTIFIER: %s\n", yytext); }
{digit}+       { printf("NUMBER: %s\n", yytext); }
"+"|"-"|"*"|"/" { printf("OPERATOR: %s\n", yytext); }
[ \t\n]+       { /* 무시 */ }
.              { printf("UNKNOWN: %s\n", yytext); }

%%

int main(int argc, char **argv) {
    yylex();
    return 0;
}
```

**Lex 파일 컴파일 및 실행:**

1. Lex 파일을 작성하여 저장합니다 (예: lexer.l).
2. Lex 파일을 컴파일합니다.
   ```sh
   flex lexer.l
   gcc lex.yy.c -o lexer -ll
   ```
3. 실행하여 결과를 확인합니다.
   ```sh
   ./lexer
   ```

**실습 목표:**
- 학생들이 Lex 도구를 사용하여 어휘 분석기를 구현하고, 다양한 토큰 패턴을 정의하는 방법을 익힙니다.
- Lex 파일을 컴파일하고 실행하여 소스 코드에서 토큰을 추출하는 과정을 경험합니다.

**제출물:**
- 작성한 Lex 파일
- 어휘 분석기 실행 결과 스크린샷

이 강의 계획을 통해 학생들은 Lex 도구를 사용하여 어휘 분석기를 구현하는 방법을 학습하고, 다양한 토큰 패턴을 정의하며, 실제로 컴파일하고 실행하여 소스 코드에서 토큰을 추출하는 과정을 경험할 수 있습니다.