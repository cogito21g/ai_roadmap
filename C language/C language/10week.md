### C 언어 20주차 심화 교육과정 - 10주차: 고급 포인터 기술

#### 10주차: 고급 포인터 기술

**강의 목표:**
10주차의 목표는 함수 포인터, 포인터 배열, 그리고 복잡한 포인터 선언을 이해하고 활용하는 것입니다. 이 주차에서는 포인터를 보다 심도 있게 다루어, 프로그램의 유연성을 높이고 효율적인 메모리 관리를 배우는 데 중점을 둡니다.

**강의 구성:**

##### 1. 함수 포인터
- **강의 내용:**
  - 함수 포인터의 개념
    - 함수의 주소를 저장하는 포인터
    - 함수 포인터의 선언과 초기화
  - 함수 포인터를 사용한 함수 호출
    - 예제: 함수 포인터를 사용한 간단한 계산기 프로그램
  - 콜백 함수의 개념과 사용
    - 콜백 함수의 정의와 활용 예제
- **실습:**
  - 함수 포인터 예제 작성
    ```c
    #include <stdio.h>

    int add(int a, int b) { return a + b; }
    int subtract(int a, int b) { return a - b; }
    
    int main() {
        int (*operation)(int, int);
        int a = 10, b = 5;
        
        operation = add;
        printf("Addition: %d\n", operation(a, b));
        
        operation = subtract;
        printf("Subtraction: %d\n", operation(a, b));
        
        return 0;
    }
    ```

  - 콜백 함수 예제 작성
    ```c
    #include <stdio.h>

    void processArray(int *arr, int size, void (*process)(int)) {
        for (int i = 0; i < size; i++) {
            process(arr[i]);
        }
    }

    void printElement(int element) {
        printf("%d\n", element);
    }

    int main() {
        int arr[] = {1, 2, 3, 4, 5};
        processArray(arr, 5, printElement);
        return 0;
    }
    ```

##### 2. 포인터 배열
- **강의 내용:**
  - 포인터 배열의 개념
    - 포인터 배열의 선언과 초기화
    - 포인터 배열의 활용 예제
  - 문자열 배열과 포인터 배열
    - 문자열을 가리키는 포인터 배열
- **실습:**
  - 포인터 배열 예제 작성
    ```c
    #include <stdio.h>

    int main() {
        int a = 10, b = 20, c = 30;
        int *arr[] = {&a, &b, &c};

        for (int i = 0; i < 3; i++) {
            printf("Value of arr[%d]: %d\n", i, *arr[i]);
        }

        return 0;
    }
    ```

  - 문자열을 가리키는 포인터 배열 예제 작성
    ```c
    #include <stdio.h>

    int main() {
        const char *arr[] = {"Hello", "World", "C Programming"};

        for (int i = 0; i < 3; i++) {
            printf("String %d: %s\n", i, arr[i]);
        }

        return 0;
    }
    ```

##### 3. 복잡한 포인터 선언 해석
- **강의 내용:**
  - 복잡한 포인터 선언의 이해
    - 우선순위 규칙: 괄호와 연산자 우선순위
  - 다양한 포인터 선언 예제
    - 포인터 배열, 함수 포인터 배열, 포인터를 반환하는 함수
  - 다중 포인터의 개념과 사용
    - 다중 포인터 선언 및 활용 예제
- **실습:**
  - 복잡한 포인터 선언 예제 작성 및 해석
    ```c
    #include <stdio.h>

    int main() {
        int x = 10;
        int *p = &x;
        int **pp = &p;

        printf("Value of x: %d\n", x);
        printf("Value of *p: %d\n", *p);
        printf("Value of **pp: %d\n", **pp);

        return 0;
    }
    ```

  - 함수 포인터 배열 예제 작성
    ```c
    #include <stdio.h>

    int add(int a, int b) { return a + b; }
    int subtract(int a, int b) { return a - b; }

    int main() {
        int (*operations[])(int, int) = {add, subtract};
        int a = 10, b = 5;

        printf("Addition: %d\n", operations[0](a, b));
        printf("Subtraction: %d\n", operations[1](a, b));

        return 0;
    }
    ```

**과제:**
10주차 과제는 다음과 같습니다.
- 함수 포인터를 사용하여 두 수의 최대공약수를 계산하는 프로그램 작성
- 포인터 배열을 사용하여 여러 문자열을 역순으로 출력하는 프로그램 작성
- 다중 포인터를 사용하여 2차원 배열을 동적으로 할당하고 요소를 초기화하는 프로그램 작성

**퀴즈 및 해설:**

1. **함수 포인터의 정의는 무엇인가요?**
   - 함수 포인터는 함수의 주소를 저장하는 포인터입니다. 이를 통해 함수 포인터를 사용하여 함수를 호출할 수 있습니다.

2. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    int a = 5;
    int b = 10;
    int *arr[] = {&a, &b};
    printf("%d\n", *arr[1]);
    ```
   - 출력 결과는 `10`입니다. `arr[1]`은 `b`의 주소를 가리키며, `*arr[1]`은 `b`의 값을 가져옵니다.

3. **복잡한 포인터 선언을 해석하는 방법은 무엇인가요?**
   - 복잡한 포인터 선언을 해석할 때는 괄호와 연산자 우선순위를 사용합니다. 포인터 선언의 우선순위를 따르며 오른쪽에서 왼쪽으로 읽습니다.

4. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    int x = 10;
    int *p = &x;
    int **pp = &p;
    printf("%d\n", **pp);
    ```
   - 출력 결과는 `10`입니다. `**pp`는 `pp`가 가리키는 `p`가 다시 가리키는 `x`의 값을 가져옵니다.

5. **포인터 배열과 배열 포인터의 차이점은 무엇인가요?**
   - 포인터 배열은 포인터들의 배열이며, 각 포인터는 개별 요소를 가리킵니다. 배열 포인터는 배열 전체를 가리키는 포인터입니다.

**해설:**
1. 함수 포인터는 함수의 주소를 저장하는 포인터로, 이를 통해 함수를 호출할 수 있습니다.
2. `arr[1]`은 `b`의 주소를 가리키며, `*arr[1]`은 `b`의 값을 가져오므로 출력 결과는 `10`입니다.
3. 복잡한 포인터 선언을 해석할 때는 괄호와 연산자 우선순위를 사용하며, 오른쪽에서 왼쪽으로 읽습니다.
4. `**pp`는 `pp`가 가리키는 `p`가 다시 가리키는 `x`의 값을 가져오므로 출력 결과는 `10`입니다.
5. 포인터 배열은 포인터들의 배열이고, 배열 포인터는 배열 전체를 가리키는 포인터입니다.

이 10주차 강의는 학생들이 고급 포인터 기술을 이해하고, 이를 활용하여 복잡한 데이터 구조와 함수를 처리하는 능력을 기를 수 있도록 도와줍니다.