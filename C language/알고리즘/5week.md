### 알고리즘 교육과정 - 5주차: 분할 정복 알고리즘

**강의 목표:**
분할 정복 기법을 이해하고, 이를 활용한 다양한 알고리즘을 학습합니다.

**강의 구성:**

#### 5. 분할 정복 알고리즘

**강의 내용:**
- 분할 정복의 개념
- 이진 검색
- 병합 정렬
- 퀵 정렬

**실습:**
- 분할 정복 기법을 이용한 알고리즘 구현

### 분할 정복의 개념

**강의 내용:**
- 분할 정복이란 무엇인가?
  - 문제를 작은 부분 문제로 나누어 해결하는 기법
  - 분할(분해), 정복(해결), 병합(합치기) 단계로 구성
- 분할 정복의 장점
  - 문제를 단순하고 효율적으로 해결
  - 병렬 처리가 가능

**실습:**
- 분할 정복 기법 이해를 위한 간단한 예제

### 이진 검색

**강의 내용:**
- 이진 검색의 개념
  - 정렬된 배열을 절반으로 나누어 검색하는 방법
- 이진 검색의 시간 복잡도
  - O(log n)

**실습:**
- 이진 검색 구현 예제
  ```c
  #include <stdio.h>

  int binarySearch(int arr[], int l, int r, int x) {
      if (r >= l) {
          int mid = l + (r - l) / 2;

          if (arr[mid] == x)
              return mid;

          if (arr[mid] > x)
              return binarySearch(arr, l, mid - 1, x);

          return binarySearch(arr, mid + 1, r, x);
      }
      return -1;
  }

  int main() {
      int arr[] = {2, 3, 4, 10, 40};
      int x = 10;
      int n = sizeof(arr) / sizeof(arr[0]);
      int result = binarySearch(arr, 0, n - 1, x);
      if (result != -1)
          printf("Element is present at index %d\n", result);
      else
          printf("Element is not present in array\n");
      return 0;
  }
  ```

### 병합 정렬

**강의 내용:**
- 병합 정렬의 개념
  - 배열을 반으로 나누고, 정렬한 후 병합하는 방법
- 병합 정렬의 시간 복잡도
  - O(n log n)

**실습:**
- 병합 정렬 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>

  void merge(int arr[], int l, int m, int r) {
      int n1 = m - l + 1;
      int n2 = r - m;

      int L[n1], R[n2];

      for (int i = 0; i < n1; i++)
          L[i] = arr[l + i];
      for (int j = 0; j < n2; j++)
          R[j] = arr[m + 1 + j];

      int i = 0, j = 0, k = l;
      while (i < n1 && j < n2) {
          if (L[i] <= R[j]) {
              arr[k] = L[i];
              i++;
          } else {
              arr[k] = R[j];
              j++;
          }
          k++;
      }

      while (i < n1) {
          arr[k] = L[i];
          i++;
          k++;
      }

      while (j < n2) {
          arr[k] = R[j];
          j++;
          k++;
      }
  }

  void mergeSort(int arr[], int l, int r) {
      if (l < r) {
          int m = l + (r - l) / 2;
          mergeSort(arr, l, m);
          mergeSort(arr, m + 1, r);
          merge(arr, l, m, r);
      }
  }

  void printArray(int arr[], int size) {
      for (int i = 0; i < size; i++) {
          printf("%d ", arr[i]);
      }
      printf("\n");
  }

  int main() {
      int arr[] = {12, 11, 13, 5, 6, 7};
      int arr_size = sizeof(arr) / sizeof(arr[0]);

      printf("Given array is \n");
      printArray(arr, arr_size);

      mergeSort(arr, 0, arr_size - 1);

      printf("\nSorted array is \n");
      printArray(arr, arr_size);
      return 0;
  }
  ```

### 퀵 정렬

**강의 내용:**
- 퀵 정렬의 개념
  - 피벗을 기준으로 배열을 분할하여 정렬하는 방법
- 퀵 정렬의 시간 복잡도
  - 평균: O(n log n)
  - 최악: O(n^2)

**실습:**
- 퀵 정렬 구현 예제
  ```c
  #include <stdio.h>

  void swap(int* a, int* b) {
      int t = *a;
      *a = *b;
      *b = t;
  }

  int partition(int arr[], int low, int high) {
      int pivot = arr[high];
      int i = (low - 1);

      for (int j = low; j < high; j++) {
          if (arr[j] < pivot) {
              i++;
              swap(&arr[i], &arr[j]);
          }
      }
      swap(&arr[i + 1], &arr[high]);
      return (i + 1);
  }

  void quickSort(int arr[], int low, int high) {
      if (low < high) {
          int pi = partition(arr, low, high);
          quickSort(arr, low, pi - 1);
          quickSort(arr, pi + 1, high);
      }
  }

  void printArray(int arr[], int size) {
      for (int i = 0; i < size; i++) {
          printf("%d ", arr[i]);
      }
      printf("\n");
  }

  int main() {
      int arr[] = {10, 7, 8, 9, 1, 5};
      int n = sizeof(arr) / sizeof(arr[0]);
      quickSort(arr, 0, n - 1);
      printf("Sorted array: \n");
      printArray(arr, n);
      return 0;
  }
  ```

**과제:**
- 분할 정복 기법을 사용하여 다양한 문제를 해결하는 알고리즘을 구현하고, 성능을 비교
- 각 알고리즘의 시간 복잡도를 분석하고, 실제 실행 시간을 측정하여 비교

**퀴즈 및 해설:**

1. **분할 정복 기법이란 무엇인가요?**
   - 분할 정복 기법은 문제를 작은 부분 문제로 나누어 해결하고, 각 부분 문제를 해결한 결과를 결합하여 전체 문제를 해결하는 방법입니다.

2. **병합 정렬의 시간 복잡도는 무엇인가요?**
   - 병합 정렬의 시간 복잡도는 O(n log n)입니다.

3. **퀵 정렬의 최악의 경우 시간 복잡도는 무엇인가요?**
   - 퀵 정렬의 최악의 경우 시간 복잡도는 O(n^2)입니다.

**해설:**
1. **분할 정복 기법**은 문제를 작은 부분 문제로 나누어 해결하고, 각 부분 문제를 해결한 결과를 결합하여 전체 문제를 해결하는 방법입니다. 이는 문제를 단순화하고, 병렬 처리가 가능하게 합니다.
2. **병합 정렬의 시간 복잡도**는 배열을 반으로 나누고, 각 부분을 정렬한 후 병합하는 과정이 O(n log n)이기 때문에 O(n log n)입니다.
3. **퀵 정렬의 최악의 경우 시간 복잡도**는 배열이 이미 정렬되어 있거나, 피벗이 항상 가장 크거나 작은 요소를 선택할 때 발생하며, 이 경우 시간 복잡도는 O(n^2)입니다.

---
