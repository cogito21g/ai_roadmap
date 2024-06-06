### 5주차: 구문 분석기 구현 (Yacc/Bison)

**강의 목표:**  
Yacc/Bison 도구를 사용하여 구문 분석기를 구현하고, 이를 통해 소스 코드를 파싱하는 방법을 학습합니다.

---

**강의 내용:**

1. **Yacc/Bison 도구 소개 (15분)**
   - Yacc/Bison 도구의 개념과 기능
   - Yacc/Bison 도구 설치 및 설정 방법
   - Yacc/Bison 파일 구조와 문법

2. **구문 분석기 구현 준비 (20분)**
   - 구문 분석 규칙 정의
   - Yacc/Bison 파일 작성 방법
   - 토큰과 구문 규칙의 연결

3. **구문 분석기 구현 (20분)**
   - Yacc/Bison 파일 작성 실습
   - 다양한 구문 규칙 정의
   - 구문 분석기 컴파일 및 실행 방법

---

**강의 진행:**

1. **강의 시작 (5분)**
   - 강의 목표와 주제를 소개합니다.
   - 이번 주차에 다룰 내용에 대한 개요를 설명합니다.

2. **Yacc/Bison 도구 소개 (15분)**
   - Yacc/Bison 도구의 개념과 기능을 설명합니다.
   - Yacc/Bison 도구 설치 및 설정 방법을 시연합니다.
   - Yacc/Bison 파일 구조와 문법을 설명합니다.

3. **구문 분석기 구현 준비 (20분)**
   - 구문 분석 규칙을 정의하는 방법을 설명합니다.
   - Yacc/Bison 파일 작성 방법을 시연합니다.
   - 토큰과 구문 규칙을 연결하는 방법을 설명합니다.

4. **구문 분석기 구현 (20분)**
   - Yacc/Bison 파일을 작성하여 구문 분석기를 구현하는 과정을 시연합니다.
   - 다양한 구문 규칙을 정의하고 인식된 구문에 대한 액션을 정의합니다.
   - 작성한 Yacc/Bison 파일을 컴파일하고 실행하여 결과를 확인합니다.

5. **실습 준비 및 안내 (10분)**
   - 실습 내용을 안내하고 실습 방법을 설명합니다.
   - 실습 과제를 발표하고 Q&A 시간을 갖습니다.

---

**실습 내용:**

1. **Yacc/Bison 도구를 사용한 구문 분석기 구현**

**실습 과제:**
- Yacc/Bison 도구를 사용하여 간단한 구문 분석기를 구현합니다.
- 다양한 구문 규칙을 정의하고, 인식된 구문에 대한 액션을 정의합니다.

**실습 예제:**

**Yacc/Bison 파일 예제:**

**Yacc 파일 (parser.y):**

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

**실습 목표:**
- 학생들이 Yacc/Bison 도구를 사용하여 구문 분석기를 구현하고, 다양한 구문 규칙을 정의하는 방법을 익힙니다.
- Yacc/Bison 파일을 컴파일하고 실행하여 소스 코드에서 구문을 파싱하는 과정을 경험합니다.

**제출물:**
- 작성한 Yacc/Bison 파일
- 구문 분석기 실행 결과 스크린샷

이 강의 계획을 통해 학생들은 Yacc/Bison 도구를 사용하여 구문 분석기를 구현하는 방법을 학습하고, 다양한 구문 규칙을 정의하며, 실제로 컴파일하고 실행하여 소스 코드에서 구문을 파싱하는 과정을 경험할 수 있습니다.