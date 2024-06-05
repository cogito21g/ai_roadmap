### C 언어 20주차 심화 교육과정 - 4주차: 배열

#### 4주차: 배열

**강의 목표:**
4주차의 목표는 1차원 배열과 2차원 배열의 개념을 이해하고, 배열을 함수와 함께 사용하는 방법을 배우는 것입니다.

**강의 구성:**

##### 1. 1차원 배열
- **강의 내용:**
  - 배열의 개념
    - 동일한 데이터 타입의 연속된 메모리 공간
  - 배열 선언 및 초기화
    - 배열 선언: `dataType arrayName[arraySize];`
    - 배열 초기화: `{value1, value2, ..., valueN}`
  - 배열 요소 접근
    - 인덱스를 사용하여 배열 요소 접근
    - 배열의 크기 계산: `sizeof(array) / sizeof(array[0])`
  - 배열의 기본 연산
    - 배열 요소의 합 계산
    - 배열 요소의 최대값 및 최소값 찾기

- **실습:**
  - 배열 선언 및 초기화 예제 작성
    ```c
    #include <stdio.h>

    int main() {
        int numbers[5] = {1, 2, 3, 4, 5};
        for (int i = 0; i < 5; i++) {
            printf("Element at index %d: %d\n", i, numbers[i]);
        }
        return 0;
    }
    ```

  - 배열 요소의 합 계산 프로그램 작성
    ```c
    #include <stdio.h>

    int main() {
        int numbers[5] = {1, 2, 3, 4, 5};
        int sum = 0;
        for (int i = 0; i < 5; i++) {
            sum += numbers[i];
        }
        printf("Sum of array elements: %d\n", sum);
        return 0;
    }
    ```

##### 2. 2차원 배열
- **강의 내용:**
  - 2차원 배열의 개념
    - 행과 열로 구성된 배열
  - 2차원 배열 선언 및 초기화
    - 배열 선언: `dataType arrayName[rows][columns];`
    - 배열 초기화: `{{row1col1, row1col2}, {row2col1, row2col2}}`
  - 2차원 배열 요소 접근
    - 인덱스를 사용하여 배열 요소 접근

- **실습:**
  - 2차원 배열 선언 및 초기화 예제 작성
    ```c
    #include <stdio.h>

    int main() {
        int matrix[2][3] = {{1, 2, 3}, {4, 5, 6}};
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                printf("Element at [%d][%d]: %d\n", i, j, matrix[i][j]);
            }
        }
        return 0;
    }
    ```

##### 3. 배열과 함수
- **강의 내용:**
  - 배열을 함수의 매개변수로 전달
    - 1차원 배열을 함수에 전달하는 방법
    - 2차원 배열을 함수에 전달하는 방법
  - 배열과 포인터
    - 배열 이름이 포인터로 동작하는 원리
    - 포인터를 사용한 배열 요소 접근

- **실습:**
  - 1차원 배열을 함수에 전달하는 예제 작성
    ```c
    #include <stdio.h>

    void printArray(int arr[], int size);

    int main() {
        int numbers[5] = {1, 2, 3, 4, 5};
        printArray(numbers, 5);
        return 0;
    }

    void printArray(int arr[], int size) {
        for (int i = 0; i < size; i++) {
            printf("Element at index %d: %d\n", i, arr[i]);
        }
    }
    ```

  - 2차원 배열을 함수에 전달하는 예제 작성
    ```c
    #include <stdio.h>

    void printMatrix(int matrix[2][3], int rows, int cols);

    int main() {
        int matrix[2][3] = {{1, 2, 3}, {4, 5, 6}};
        printMatrix(matrix, 2, 3);
        return 0;
    }

    void printMatrix(int matrix[2][3], int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("Element at [%d][%d]: %d\n", i, j, matrix[i][j]);
            }
        }
    }
    ```

**과제:**
4주차 과제는 다음과 같습니다.
- 사용자로부터 5개의 정수를 입력받아 배열에 저장하고, 배열의 요소를 출력하는 프로그램 작성
- 2차원 배열을 사용하여 3x3 행렬의 요소를 입력받고, 행렬을 출력하는 프로그램 작성
- 배열을 사용하여 학생들의 점수를 저장하고, 평균 점수를 계산하여 출력하는 프로그램 작성

**퀴즈 및 해설:**

1. **배열의 정의는 무엇인가요?**
   - 배열은 동일한 데이터 타입의 연속된 메모리 공간입니다. 배열은 고정된 크기를 가지며, 인덱스를 사용하여 요소에 접근합니다.

2. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    int arr[3] = {1, 2, 3};
    printf("%d\n", arr[1]);
    ```
   - 출력 결과는 `2`입니다. 배열의 인덱스는 0부터 시작하며, `arr[1]`은 배열의 두 번째 요소인 `2`를 나타냅니다.

3. **배열을 함수의 매개변수로 전달할 때 배열의 크기를 전달해야 하는 이유는 무엇인가요?**
   - 배열을 함수의 매개변수로 전달할 때, 배열의 크기를 전달하지 않으면 함수는 배열의 크기를 알 수 없습니다. 따라서 배열의 크기를 함께 전달하여 함수가 배열의 크기를 알고 적절히 처리할 수 있도록 합니다.

4. **2차원 배열의 요소에 접근하려면 어떻게 해야 하나요?**
   - 2차원 배열의 요소에 접근하려면 두 개의 인덱스를 사용해야 합니다. 첫 번째 인덱스는 행을 나타내고, 두 번째 인덱스는 열을 나타냅니다.
     ```c
     int matrix[2][3] = {{1, 2, 3}, {4, 5, 6}};
     int element = matrix[1][2]; // 2번째 행, 3번째 열의 요소 접근 (값은 6)
     ```

5. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    void modifyArray(int arr[], int size) {
        for (int i = 0; i < size; i++) {
            arr[i] += 10;
        }
    }

    int main() {
        int numbers[3] = {1, 2, 3};
        modifyArray(numbers, 3);
        for (int i = 0; i < 3; i++) {
            printf("%d ", numbers[i]);
        }
        return 0;
    }
    ```
   - 출력 결과는 `11 12 13`입니다. 함수 `modifyArray`는 배열의 각 요소에 10을 더합니다. 배열은 함수에 포인터로 전달되므로, 함수 내에서 배열의 요소를 수정하면 원래 배열에도 영향을 미칩니다.

이 4주차 강의는 학생들이 1차원 배열과 2차원 배열을 이해하고, 배열을 함수와 함께 사용하는 방법을 배우는 능력을 기를 수 있도록 도와줍니다.