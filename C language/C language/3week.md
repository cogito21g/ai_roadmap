### C 언어 20주차 심화 교육과정 - 3주차: 함수

#### 3주차: 함수

**강의 목표:**
3주차의 목표는 함수의 개념과 사용법을 이해하고, 매개변수 전달 방식과 재귀 함수를 활용하여 프로그램을 구조화하는 방법을 배우는 것입니다.

**강의 구성:**

##### 1. 함수 정의 및 호출
- **강의 내용:**
  - 함수의 개념
    - 코드의 재사용성 증가
    - 프로그램의 모듈화
  - 함수 선언과 정의
    - 함수 선언: 반환 타입, 함수 이름, 매개변수
    - 함수 정의: 함수 본문
  - 함수 호출
    - 함수 호출의 기본 구조
    - 예제: 두 수의 합을 계산하는 함수
- **실습:**
  - 함수 선언, 정의 및 호출 예제 작성
    ```c
    #include <stdio.h>

    int add(int a, int b); // 함수 선언

    int main() {
        int result = add(5, 3); // 함수 호출
        printf("Sum: %d\n", result);
        return 0;
    }

    int add(int a, int b) { // 함수 정의
        return a + b;
    }


##### 2. 매개변수 전달 방식
- **강의 내용:**
  - 값에 의한 전달 (Call by Value)
    - 기본 원리: 함수 호출 시 값의 복사본이 전달됨
    - 예제: 두 수의 합을 계산하는 함수
  - 참조에 의한 전달 (Call by Reference)
    - 기본 원리: 함수 호출 시 주소가 전달됨
    - 포인터를 사용하여 참조에 의한 전달 구현
    - 예제: 두 수의 값을 교환하는 함수
- **실습:**
  - 값에 의한 전달 예제 작성
    ```c
    #include <stdio.h>

    void swap(int a, int b); // 함수 선언

    int main() {
        int x = 10, y = 20;
        printf("Before swap: x = %d, y = %d\n", x, y);
        swap(x, y); // 값에 의한 전달
        printf("After swap: x = %d, y = %d\n", x, y);
        return 0;
    }

    void swap(int a, int b) { // 함수 정의
        int temp = a;
        a = b;
        b = temp;
        printf("Inside swap: a = %d, b = %d\n", a, b);
    }
    ```

  - 참조에 의한 전달 예제 작성
    ```c
    #include <stdio.h>

    void swap(int *a, int *b); // 함수 선언

    int main() {
        int x = 10, y = 20;
        printf("Before swap: x = %d, y = %d\n", x, y);
        swap(&x, &y); // 참조에 의한 전달
        printf("After swap: x = %d, y = %d\n", x, y);
        return 0;
    }

    void swap(int *a, int *b) { // 함수 정의
        int temp = *a;
        *a = *b;
        *b = temp;
    }
    ```

##### 3. 재귀 함수
- **강의 내용:**
  - 재귀 함수의 개념
    - 자기 자신을 호출하는 함수
    - 기본 구조와 종료 조건의 중요성
  - 재귀 함수를 사용한 문제 해결
    - 예제: 팩토리얼 계산
    - 예제: 피보나치 수열 계산
- **실습:**
  - 재귀 함수를 사용한 팩토리얼 계산 프로그램 작성
    ```c
    #include <stdio.h>

    int factorial(int n); // 함수 선언

    int main() {
        int number;
        printf("Enter a positive integer: ");
        scanf("%d", &number);
        printf("Factorial of %d is %d\n", number, factorial(number));
        return 0;
    }

    int factorial(int n) { // 함수 정의
        if (n == 0) {
            return 1;
        } else {
            return n * factorial(n - 1);
        }
    }
    ```

  - 재귀 함수를 사용한 피보나치 수열 계산 프로그램 작성
    ```c
    #include <stdio.h>

    int fibonacci(int n); // 함수 선언

    int main() {
        int number;
        printf("Enter a positive integer: ");
        scanf("%d", &number);
        printf("Fibonacci number %d is %d\n", number, fibonacci(number));
        return 0;
    }

    int fibonacci(int n) { // 함수 정의
        if (n == 0) {
            return 0;
        } else if (n == 1) {
            return 1;
        } else {
            return fibonacci(n - 1) + fibonacci(n - 2);
        }
    }
    ```

**과제:**
3주차 과제는 다음과 같습니다.
- 두 수의 최대공약수를 구하는 재귀 함수 작성
- 값에 의한 전달과 참조에 의한 전달의 차이를 설명하는 예제 프로그램 작성
- 재귀를 사용하여 배열의 합을 계산하는 프로그램 작성

**퀴즈:**
1. 함수 선언과 정의의 차이점을 설명하세요.
2. 값에 의한 전달과 참조에 의한 전달의 차이점을 설명하세요.
3. 다음 코드의 출력 결과는 무엇인가요?
    ```c
    void update(int *a) {
        *a = 5;
    }

    int main() {
        int x = 10;
        update(&x);
        printf("%d\n", x);
        return 0;
    }
    ```
4. 재귀 함수가 올바르게 작동하기 위한 필수 조건은 무엇인가요?
5. 재귀 함수를 사용하여 n번째 피보나치 수를 계산하는 코드를 작성하세요.

이 3주차 강의는 학생들이 함수의 개념과 사용법을 이해하고, 매개변수 전달 방식과 재귀 함수를 활용하여 프로그램을 구조화하는 능력을 기를 수 있도록 도와줍니다.