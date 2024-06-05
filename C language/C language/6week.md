### C 언어 20주차 심화 교육과정 - 6주차: 포인터 심화

#### 6주차: 포인터 심화

**강의 목표:**
6주차의 목표는 포인터와 함수를 더욱 깊이 있게 이해하고, 동적 메모리 할당과 다중 포인터의 사용법을 학습하는 것입니다.

**강의 구성:**

##### 1. 포인터와 함수
- **강의 내용:**
  - 함수 포인터
    - 함수 포인터의 개념과 선언
    - 함수 포인터를 사용한 함수 호출
    - 예제: 함수 포인터를 사용한 간단한 계산기 프로그램
  - 포인터를 반환하는 함수
    - 포인터를 반환하는 함수의 정의와 사용법
    - 예제: 배열의 최대값을 찾고 그 포인터를 반환하는 함수
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

  - 포인터를 반환하는 함수 예제 작성
    ```c
    #include <stdio.h>

    int* findMax(int arr[], int size) {
        int *max = &arr[0];
        for (int i = 1; i < size; i++) {
            if (arr[i] > *max) {
                max = &arr[i];
            }
        }
        return max;
    }

    int main() {
        int numbers[5] = {1, 3, 5, 7, 2};
        int *max = findMax(numbers, 5);
        printf("Maximum value: %d\n", *max);
        return 0;
    }
    ```

##### 2. 동적 메모리 할당 (malloc, free)
- **강의 내용:**
  - 동적 메모리 할당의 필요성
    - 컴파일 타임과 런타임 메모리 할당의 차이
    - 동적 메모리 할당의 장점
  - malloc과 free
    - malloc 함수의 사용법: `void* malloc(size_t size);`
    - 메모리 해제: `free(void* ptr);`
    - 예제: 동적 배열 할당 및 해제
  - calloc과 realloc
    - calloc: 메모리 초기화와 함께 할당
    - realloc: 메모리 크기 조정
- **실습:**
  - malloc과 free를 사용한 동적 메모리 할당 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>

    int main() {
        int *arr;
        int n;

        printf("Enter number of elements: ");
        scanf("%d", &n);

        arr = (int*)malloc(n * sizeof(int));
        if (arr == NULL) {
            printf("Memory allocation failed!\n");
            return 1;
        }

        for (int i = 0; i < n; i++) {
            arr[i] = i + 1;
        }

        printf("Array elements: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");

        free(arr);
        return 0;
    }
    ```

  - calloc과 realloc을 사용한 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>

    int main() {
        int *arr;
        int n;

        printf("Enter initial number of elements: ");
        scanf("%d", &n);

        arr = (int*)calloc(n, sizeof(int));
        if (arr == NULL) {
            printf("Memory allocation failed!\n");
            return 1;
        }

        printf("Initial array elements: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");

        printf("Enter new number of elements: ");
        scanf("%d", &n);

        arr = (int*)realloc(arr, n * sizeof(int));
        if (arr == NULL) {
            printf("Memory reallocation failed!\n");
            return 1;
        }

        printf("Array elements after reallocation: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");

        free(arr);
        return 0;
    }
    ```

##### 3. 다중 포인터
- **강의 내용:**
  - 다중 포인터의 개념
    - 포인터를 가리키는 포인터
    - 포인터의 포인터 선언 및 초기화
  - 다중 포인터의 사용 예제
    - 2차원 배열을 다중 포인터로 처리
    - 예제: 다중 포인터를 사용하여 문자열 배열 처리
- **실습:**
  - 다중 포인터 예제 작성
    ```c
    #include <stdio.h>

    int main() {
        int x = 10;
        int *ptr1 = &x;
        int **ptr2 = &ptr1;

        printf("Value of x: %d\n", x);
        printf("Value of *ptr1: %d\n", *ptr1);
        printf("Value of **ptr2: %d\n", **ptr2);

        return 0;
    }
    ```

  - 2차원 배열을 다중 포인터로 처리하는 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>

    int main() {
        int rows = 2, cols = 3;
        int **matrix;

        matrix = (int**)malloc(rows * sizeof(int*));
        for (int i = 0; i < rows; i++) {
            matrix[i] = (int*)malloc(cols * sizeof(int));
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = i * cols + j + 1;
            }
        }

        printf("Matrix elements:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%d ", matrix[i][j]);
            }
            printf("\n");
        }

        for (int i = 0; i < rows; i++) {
            free(matrix[i]);
        }
        free(matrix);

        return 0;
    }
    ```

**과제:**
6주차 과제는 다음과 같습니다.
- 동적 메모리 할당을 사용하여 사용자로부터 문자열을 입력받고 출력하는 프로그램 작성
- 다중 포인터를 사용하여 동적 2차원 배열을 생성하고 초기화하는 프로그램 작성
- 함수 포인터를 사용하여 다양한 수학 연산을 수행하는 프로그램 작성

**퀴즈 및 해설:**

1. **함수 포인터의 정의는 무엇인가요?**
   - 함수 포인터는 함수의 주소를 저장하는 포인터입니다. 이를 통해 함수 포인터를 사용하여 함수를 호출할 수 있습니다.

2. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    int x = 10;
    int *ptr = &x;
    int **pptr = &ptr;
    printf("%d\n", **pptr);
    ```
   - 출력 결과는 `10`입니다. `**pptr`은 포인터를 두 번 역참조하여 `x`의 값을 가져옵니다.

3. **malloc과 calloc의 차이점을 설명하세요.**
   - `malloc`은 지정된 크기의 메모리를 할당하지만 초기화하지 않습니다. `calloc`은 지정된 크기의 메모리를 할당하고 0으로 초기화합니다.

4. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    void set_value(int *a) {
        *a = 100;
    }

    int main() {
        int x = 0;
        set_value(&x);
        printf("%d\n", x);
        return 0;
    }
    ```
   - 출력 결과는 `100`입니다. 함수 `set_value`는 포인터를 통해 `x`의 값을 100으로 설정합니다.

5. **동적 메모리 할당의 장점은 무엇인가요?**
   - 동적 메모리 할당의 장점은 런타임 시에 필요한 만큼의 메모리를 할당할 수 있어 메모리의 효율적 사용이 가능합니다. 또한, 프로그램 실행 중에 메모리 크기를 조정할 수 있습니다.

**해설:**
1. 함수 포인터는 함수의 주소를 저장하는 포인터로, 이를 통해 함수를 호출할 수 있습니다.
2. `**pptr`은 포인터를 두 번 역참조하여 `x`의 값을 가져오므로

 출력 결과는 `10`입니다.
3. `malloc`은 메모리를 초기화하지 않고 할당하며, `calloc`은 메모리를 0으로 초기화하여 할당합니다.
4. 함수 `set_value`는 포인터를 통해 `x`의 값을 100으로 설정하므로, 출력 결과는 `100`입니다.
5. 동적 메모리 할당은 런타임 시에 필요한 만큼의 메모리를 할당할 수 있어 메모리의 효율적 사용이 가능하며, 프로그램 실행 중에 메모리 크기를 조정할 수 있습니다.

이 6주차 강의는 학생들이 포인터와 함수의 심화 개념을 이해하고, 동적 메모리 할당과 다중 포인터를 활용하는 능력을 기를 수 있도록 도와줍니다.