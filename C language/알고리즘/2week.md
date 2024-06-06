### 알고리즘 교육과정 - 2주차: 정렬 알고리즘

**강의 목표:**
정렬 알고리즘의 기본 개념과 구현 방법을 이해하고, 각 알고리즘의 시간 복잡도를 분석합니다.

**강의 구성:**

#### 2. 정렬 알고리즘

**강의 내용:**
- 정렬의 개념과 중요성
  - 정렬이란 무엇인가?
  - 정렬의 필요성
- 기본 정렬 알고리즘
  - 버블 정렬
  - 선택 정렬
  - 삽입 정렬

**실습:**
- 각 정렬 알고리즘의 구현 및 성능 비교

### 버블 정렬

**강의 내용:**
- 버블 정렬의 개념
  - 인접한 두 요소를 비교하여 정렬하는 방법
- 버블 정렬의 시간 복잡도
  - 평균 및 최악의 시간 복잡도: O(n^2)

**실습:**
- 버블 정렬 구현 예제
  ```c
  #include <stdio.h>

  void bubbleSort(int arr[], int n) {
      for (int i = 0; i < n - 1; i++) {
          for (int j = 0; j < n - i - 1; j++) {
              if (arr[j] > arr[j + 1]) {
                  int temp = arr[j];
                  arr[j] = arr[j + 1];
                  arr[j + 1] = temp;
              }
          }
      }
  }

  void printArray(int arr[], int size) {
      for (int i = 0; i < size; i++) {
          printf("%d ", arr[i]);
      }
      printf("\n");
  }

  int main() {
      int arr[] = {64, 34, 25, 12, 22, 11, 90};
      int n = sizeof(arr) / sizeof(arr[0]);
      bubbleSort(arr, n);
      printf("Sorted array: \n");
      printArray(arr, n);
      return 0;
  }
  ```

### 선택 정렬

**강의 내용:**
- 선택 정렬의 개념
  - 가장 작은 (혹은 가장 큰) 요소를 선택하여 정렬하는 방법
- 선택 정렬의 시간 복잡도
  - 평균 및 최악의 시간 복잡도: O(n^2)

**실습:**
- 선택 정렬 구현 예제
  ```c
  #include <stdio.h>

  void selectionSort(int arr[], int n) {
      for (int i = 0; i < n - 1; i++) {
          int min_idx = i;
          for (int j = i + 1; j < n; j++) {
              if (arr[j] < arr[min_idx]) {
                  min_idx = j;
              }
          }
          int temp = arr[min_idx];
          arr[min_idx] = arr[i];
          arr[i] = temp;
      }
  }

  void printArray(int arr[], int size) {
      for (int i = 0; i < size; i++) {
          printf("%d ", arr[i]);
      }
      printf("\n");
  }

  int main() {
      int arr[] = {64, 25, 12, 22, 11};
      int n = sizeof(arr) / sizeof(arr[0]);
      selectionSort(arr, n);
      printf("Sorted array: \n");
      printArray(arr, n);
      return 0;
  }
  ```

### 삽입 정렬

**강의 내용:**
- 삽입 정렬의 개념
  - 요소를 적절한 위치에 삽입하여 정렬하는 방법
- 삽입 정렬의 시간 복잡도
  - 평균 및 최악의 시간 복잡도: O(n^2)
  - 최선의 시간 복잡도: O(n) (거의 정렬된 경우)

**실습:**
- 삽입 정렬 구현 예제
  ```c
  #include <stdio.h>

  void insertionSort(int arr[], int n) {
      for (int i = 1; i < n; i++) {
          int key = arr[i];
          int j = i - 1;

          while (j >= 0 && arr[j] > key) {
              arr[j + 1] = arr[j];
              j = j - 1;
          }
          arr[j + 1] = key;
      }
  }

  void printArray(int arr[], int size) {
      for (int i = 0; i < size; i++) {
          printf("%d ", arr[i]);
      }
      printf("\n");
  }

  int main() {
      int arr[] = {12, 11, 13, 5, 6};
      int n = sizeof(arr) / sizeof(arr[0]);
      insertionSort(arr, n);
      printf("Sorted array: \n");
      printArray(arr, n);
      return 0;
  }
  ```

**과제:**
- 버블 정렬, 선택 정렬, 삽입 정렬 각각의 알고리즘을 구현하고, 성능을 비교하기 위해 다양한 크기의 배열을 정렬하는 프로그램 작성
- 각 알고리즘의 시간 복잡도를 분석하고, 실제 실행 시간을 측정하여 비교

**퀴즈 및 해설:**

1. **버블 정렬의 시간 복잡도는 무엇인가요?**
   - 버블 정렬의 평균 및 최악의 시간 복잡도는 O(n^2)입니다.

2. **선택 정렬의 기본 원리는 무엇인가요?**
   - 선택 정렬은 배열에서 가장 작은 (혹은 가장 큰) 요소를 찾아 첫 번째 요소와 교환하고, 두 번째로 작은 요소를 찾아 두 번째 요소와 교환하는 방식으로 정렬합니다.

3. **삽입 정렬이 효율적인 경우는 언제인가요?**
   - 삽입 정렬은 배열이 거의 정렬된 경우 효율적입니다. 이때의 시간 복잡도는 O(n)입니다.

**해설:**
1. **버블 정렬의 시간 복잡도**는 인접한 두 요소를 비교하고 교환하는 과정을 반복하기 때문에 O(n^2)입니다.
2. **선택 정렬의 기본 원리**는 배열에서 가장 작은 (혹은 가장 큰) 요소를 선택하여 정렬하는 것입니다. 이를 반복하여 배열을 정렬합니다.
3. **삽입 정렬이 효율적인 경우**는 배열이 거의 정렬된 경우입니다. 이때 각 요소는 거의 올바른 위치에 있으며, 삽입 정렬은 최소한의 비교와 이동만으로 정렬을 완료할 수 있습니다.

---
