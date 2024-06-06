### 자료구조 교육과정 - 1주차: 배열과 연결 리스트

**강의 목표:**
배열과 연결 리스트의 기본 개념을 이해하고, 배열과 연결 리스트의 구현 및 기본 연산을 학습합니다.

**강의 구성:**

#### 1. 배열

**강의 내용:**
- 배열의 개념
  - 배열의 정의 및 특징
  - 배열의 장단점
- 배열의 기본 연산
  - 배열의 삽입, 삭제, 검색

**실습:**
- 배열 구현 예제
  ```c
  #include <stdio.h>

  void printArray(int arr[], int size) {
      for (int i = 0; i < size; i++) {
          printf("%d ", arr[i]);
      }
      printf("\n");
  }

  void insert(int arr[], int* size, int element, int position) {
      for (int i = *size; i > position; i--) {
          arr[i] = arr[i - 1];
      }
      arr[position] = element;
      (*size)++;
  }

  void delete(int arr[], int* size, int position) {
      for (int i = position; i < *size - 1; i++) {
          arr[i] = arr[i + 1];
      }
      (*size)--;
  }

  int main() {
      int arr[10] = {1, 2, 3, 4, 5};
      int size = 5;

      printArray(arr, size);

      insert(arr, &size, 10, 2);
      printArray(arr, size);

      delete(arr, &size, 3);
      printArray(arr, size);

      return 0;
  }
  ```

**과제:**
- 배열에 요소를 삽입하고 삭제하는 함수 구현
  - 배열에 새로운 요소를 삽입하는 함수 작성
  - 배열에서 특정 위치의 요소를 삭제하는 함수 작성

**퀴즈 및 해설:**

1. **배열의 장점과 단점은 무엇인가요?**
   - **장점:**
     - 인덱스를 통해 빠르게 접근 가능 (O(1))
     - 메모리 공간이 연속적이어서 캐시 효율이 높음
   - **단점:**
     - 크기가 고정되어 있어 크기 변경이 불가능함
     - 삽입과 삭제가 비효율적임 (O(n))

2. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    int arr[5] = {1, 2, 3, 4, 5};
    int size = 5;
    printArray(arr, size);
    insert(arr, &size, 10, 2);
    printArray(arr, size);
    delete(arr, &size, 3);
    printArray(arr, size);
    ```
   - 출력 결과:
     ```
     1 2 3 4 5
     1 2 10 3 4 5
     1 2 10 4 5
     ```

**해설:**
1. **배열의 장점과 단점**은 배열의 구조적 특성에서 기인합니다. 배열은 연속된 메모리 공간을 사용하므로 인덱스를 통해 빠르게 접근할 수 있지만, 크기가 고정되어 있어 동적 크기 변경이 불가능합니다. 또한 삽입과 삭제가 배열의 중간에서 일어날 경우 많은 요소를 이동해야 하므로 비효율적입니다.
2. **코드 출력 결과**는 함수 `insert`와 `delete`의 동작을 통해 배열의 변화를 보여줍니다. `insert` 함수는 특정 위치에 요소를 삽입하고, `delete` 함수는 특정 위치의 요소를 삭제합니다.

---

