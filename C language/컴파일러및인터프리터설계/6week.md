### 6주차: 중간 프로젝트

**강의 목표:**  
어휘 분석기 및 구문 분석기를 통합하여 간단한 프로그래밍 언어를 위한 파서 컴파일러를 구현합니다. 학생들은 지금까지 배운 내용을 바탕으로 중간 프로젝트를 완성합니다.

---

**강의 내용:**

1. **중간 프로젝트 개요 (15분)**
   - 중간 프로젝트의 목표와 기대 결과
   - 프로젝트 요구사항 설명
   - 프로젝트 평가 기준 안내

2. **어휘 분석기 및 구문 분석기 통합 (20분)**
   - 어휘 분석기와 구문 분석기의 역할 복습
   - 어휘 분석기와 구문 분석기를 통합하는 방법 설명
   - 통합 과정 시연

3. **프로젝트 계획 수립 및 진행 (25분)**
   - 프로젝트 계획 수립 방법 설명
   - 역할 분담 및 작업 일정 작성
   - 실시간 프로젝트 진행 및 지원

---

**강의 진행:**

1. **강의 시작 (5분)**
   - 중간 프로젝트의 목표와 주제를 소개합니다.
   - 이번 주차에 다룰 내용에 대한 개요를 설명합니다.

2. **중간 프로젝트 개요 (15분)**
   - 중간 프로젝트의 목표와 기대 결과를 설명합니다.
   - 프로젝트 요구사항을 자세히 설명합니다.
   - 프로젝트 평가 기준을 안내합니다.

3. **어휘 분석기 및 구문 분석기 통합 (20분)**
   - 어휘 분석기와 구문 분석기의 역할을

   복습합니다.
   - 어휘 분석기와 구문 분석기를 통합하는 방법을 설명합니다.
   - 통합 과정 및 주의사항을 시연합니다.

4. **프로젝트 계획 수립 및 진행 (25분)**
   - 프로젝트 계획 수립 방법을 설명합니다.
   - 역할 분담 및 작업 일정을 작성하는 방법을 안내합니다.
   - 각 팀별로 프로젝트 계획을 수립하고 실시간으로 프로젝트를 진행합니다.
   - 진행 중에 발생하는 문제에 대한 지원을 제공합니다.

5. **Q&A 및 실습 준비 (10분)**
   - Q&A 시간을 통해 학생들의 질문에 답변하고, 이해도를 확인합니다.
   - 실습 내용을 안내하고 실습 방법을 설명합니다.

---

**실습 내용:**

1. **어휘 분석기 및 구문 분석기 통합 프로젝트**

**실습 과제:**
- 이전에 작성한 어휘 분석기와 구문 분석기를 통합하여 간단한 프로그래밍 언어를 위한 파서 컴파일러를 구현합니다.
- 프로젝트 계획을 수립하고 역할을 분담하여 작업을 진행합니다.

**실습 목표:**
- 학생들이 지금까지 배운 내용을 종합적으로 활용하여 중간 프로젝트를 완성합니다.
- 어휘 분석기와 구문 분석기를 통합하고, 통합된 시스템에서 소스 코드를 파싱하는 경험을 합니다.

**프로젝트 요구사항:**
- 변수 선언, 연산자, 조건문, 반복문 등의 기본 구문을 지원하는 간단한 프로그래밍 언어를 설계합니다.
- 어휘 분석기와 구문 분석기를 통합하여 전체 파서 컴파일러를 구현합니다.
- 예제 코드를 통해 컴파일러가 올바르게 동작하는지 테스트합니다.

**프로젝트 평가 기준:**
- 프로젝트 계획서: 역할 분담, 작업 일정, 목표
- 코드의 완성도: 어휘 분석기, 구문 분석기, 통합된 시스템의 동작 여부
- 테스트 결과: 예제 코드의 파싱 결과
- 문서화: 코드 주석, 프로젝트 설명서

**제출물:**
- 프로젝트 계획서
- 어휘 분석기 및 구문 분석기 코드
- 통합된 파서 컴파일러 코드
- 테스트 결과
- 프로젝트 설명서

---

**예제 코드:**

**Yacc/Bison 파일 (parser.y):**

```yacc
%{
#include <stdio.h>
#include <stdlib.h>
%}

%token NUMBER IDENTIFIER

%%
program:
    program statement
    | /* empty */
    ;

statement:
    declaration
    | assignment
    | if_statement
    | for_loop
    ;

declaration:
    "int" IDENTIFIER ';'
    {
        printf("Declaration: int %s\n", $2);
    }
    ;

assignment:
    IDENTIFIER '=' expr ';'
    {
        printf("Assignment: %s = %d\n", $1, $3);
    }
    ;

if_statement:
    "if" '(' expr ')' '{' statement '}'
    {
        printf("If statement with condition %d\n", $3);
    }
    ;

for_loop:
    "for" '(' declaration expr ';' expr ')' '{' statement '}'
    {
        printf("For loop with condition %d\n", $4);
    }
    ;

expr:
    NUMBER
    | IDENTIFIER
    | expr '+' expr
    | expr '-' expr
    | expr '*' expr
    | expr '/' expr
    ;
%%

int main(void) {
    yyparse();
    return 0;
}

int yyerror(char *s) {
    fprintf(stderr, "error: %s\n", s);
    return 0;
}
```

**Lex 파일 (lexer.l):**

```lex
%{
#include "y.tab.h"
%}

digit      [0-9]
letter     [a-zA-Z]
identifier {letter}({letter}|{digit})*

%%

{identifier}   { yylval = strdup(yytext); return IDENTIFIER; }
{digit}+       { yylval = atoi(yytext); return NUMBER; }
"int"          { return "int"; }
"if"           { return "if"; }
"for"          { return "for"; }
"+"            { return '+'; }
"-"            { return '-'; }
"*"            { return '*'; }
"/"            { return '/'; }
"("            { return '('; }
")"            { return ')'; }
"{"            { return '{'; }
"}"            { return '}'; }
";"            { return ';'; }
"="            { return '='; }
[ \t\n]        { /* ignore whitespace */ }
.              { return yytext[0]; }

%%

int yywrap(void) {
    return 1;
}
```

**구문 분석기 컴파일 및 실행:**

1. Yacc 파일과 Lex 파일을 작성하여 저장합니다 (예: parser.y, lexer.l).
2. Yacc 파일을 컴파일합니다.
   ```sh
   bison -d parser.y
   ```
3. Lex 파일을 컴파일합니다.
   ```sh
   flex lexer.l
   ```
4. 컴파일된 파일을 연결하여 실행 파일을 생성합니다.
   ```sh
   gcc lex.yy.c y.tab.c -o parser -ll
   ```
5. 실행하여 결과를 확인합니다.
   ```sh
   ./parser
   ```

이 강의 계획을 통해 학생들은 어휘 분석기와 구문 분석기를 통합하여 중간 프로젝트를 완성하고, 전체 컴파일러 시스템을 구현하는 경험을 할 수 있습니다. 이를 통해 실무적인 프로젝트 관리 능력과 팀워크를 향상시킬 수 있습니다.