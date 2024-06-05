### C 언어 20주차 심화 교육과정 - 2주차: 제어문

#### 2주차: 제어문

**강의 목표:**
2주차의 목표는 조건문과 반복문을 이해하고 활용하여 프로그램의 흐름을 제어할 수 있도록 하는 것입니다.

**강의 구성:**

##### 1. 조건문
- **강의 내용:**
  - if 문
    - 기본 구조와 사용법
    - 예제: 숫자의 양수/음수/제로 판별
  - else if 문
    - 여러 조건을 순차적으로 검사
    - 예제: 점수에 따른 학점 계산
  - else 문
    - 모든 조건이 거짓일 때 실행
    - 예제: 숫자의 범위에 따른 출력
  - switch 문
    - 여러 값을 검사하여 해당하는 case 실행
    - break와 default의 역할
    - 예제: 메뉴 선택 프로그램

- **실습:**
  - if-else if-else 문을 사용한 프로그램 작성
    ```c
    #include <stdio.h>

    int main() {
        int number;
        printf("Enter an integer: ");
        scanf("%d", &number);

        if (number > 0) {
            printf("The number is positive.\n");
        } else if (number < 0) {
            printf("The number is negative.\n");
        } else {
            printf("The number is zero.\n");
        }

        return 0;
    }
    ```

  - switch 문을 사용한 프로그램 작성
    ```c
    #include <stdio.h>

    int main() {
        int choice;
        printf("Choose an option (1-3):\n1. Option 1\n2. Option 2\n3. Option 3\n");
        scanf("%d", &choice);

        switch (choice) {
            case 1:
                printf("You chose Option 1.\n");
                break;
            case 2:
                printf("You chose Option 2.\n");
                break;
            case 3:
                printf("You chose Option 3.\n");
                break;
            default:
                printf("Invalid choice.\n");
        }

        return 0;
    }
    ```

##### 2. 반복문
- **강의 내용:**
  - for 문
    - 기본 구조 (초기화; 조건; 증감)
    - 예제: 1부터 10까지의 합 계산
  - while 문
    - 조건을 만족하는 동안 반복
    - 예제: 사용자가 0을 입력할 때까지 숫자 합산
  - do-while 문
    - 최소 한 번은 실행 후 조건 검사
    - 예제: 비밀번호 입력 받기

- **실습:**
  - for 문을 사용한 프로그램 작성
    ```c
    #include <stdio.h>

    int main() {
        int sum = 0;

        for (int i = 1; i <= 10; i++) {
            sum += i;
        }

        printf("Sum of numbers from 1 to 10 is %d\n", sum);

        return 0;
    }
    ```

  - while 문을 사용한 프로그램 작성
    ```c
    #include <stdio.h>

    int main() {
        int sum = 0;
        int number;

        printf("Enter numbers to add (enter 0 to stop):\n");

        while (1) {
            scanf("%d", &number);
            if (number == 0) {
                break;
            }
            sum += number;
        }

        printf("The total sum is %d\n", sum);

        return 0;
    }
    ```

  - do-while 문을 사용한 프로그램 작성
    ```c
    #include <stdio.h>

    int main() {
        int password;

        do {
            printf("Enter the password (1234): ");
            scanf("%d", &password);
        } while (password != 1234);

        printf("Access granted!\n");

        return 0;
    }
    ```

**과제:**
2주차 과제는 다음과 같습니다.
- if-else if-else 문을 사용하여 사용자가 입력한 나이에 따라 "미성년자", "성인", "노인"을 출력하는 프로그램 작성
- for 문을 사용하여 1부터 100까지의 짝수 합을 계산하는 프로그램 작성
- while 문을 사용하여 0부터 사용자가 입력한 숫자까지의 합을 계산하는 프로그램 작성
- switch 문을 사용하여 간단한 계산기 프로그램 작성 (덧셈, 뺄셈, 곱셈, 나눗셈)

**퀴즈:**
1. if-else 문과 switch 문을 사용하는 경우의 차이점을 설명하세요.
2. for 문과 while 문의 차이점은 무엇인가요?
3. 다음 코드의 출력 결과는 무엇인가요?
    ```c
    int x = 5;
    switch (x) {
        case 1:
            printf("One");
            break;
        case 5:
            printf("Five");
            break;
        default:
            printf("Default");
    }
    ```
4. while 문과 do-while 문의 차이점을 설명하세요.
5. for 문을 사용하여 1부터 5까지의 숫자를 출력하는 코드를 작성하세요.

이 2주차 강의는 학생들이 조건문과 반복문을 이해하고 이를 활용하여 프로그램의 흐름을 제어하는 능력을 기를 수 있도록 도와줍니다.