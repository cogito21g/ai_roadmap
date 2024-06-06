### 알고리즘 교육과정 - 3주차: 고급 정렬 알고리즘

**강의 목표:**
퀵 정렬, 병합 정렬, 힙 정렬과 같은 고급 정렬 알고리즘을 이해하고, 각 알고리즘의 구현 방법과 시간 복잡도를 학습합니다.

**강의 구성:**

#### 3. 고급 정렬 알고리즘

**강의 내용:**
- 퀵 정렬
- 병합 정렬
- 힙 정렬

**실습:**
- 각 정렬 알고리즘의 구현 및 성능 비교

### 퀵 정렬

**강의 내용:**
- 퀵 정렬의 개념
  - 분할 정복 기법을 사용한 정렬 방법
- 퀵 정렬의 시간 복잡도
  - 평균 시간 복잡도: O(n log n)
  - 최악의 시간 복잡도: O(n^2) (이미 정렬된 경우)

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

### 병합 정렬

**강의 내용:**
- 병합 정렬의 개념
  - 분할 정복 기법을 사용한 정렬 방법
- 병합 정렬의 시간 복잡도
  - 평균 및 최악의 시간 복잡도: O(n log n)

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

### 힙 정렬

**강의 내용:**
- 힙 정렬의 개념
  - 힙을 이용한 정렬 방법
- 힙 정렬의 시간 복잡도
  - 평균 및 최악의 시간 복잡도: O(n log n)

**실습:**
- 힙 정렬 구현 예제
  ```c
  #include <stdio.h>

  void swap(int* a, int* b) {
      int t = *a;
      *a = *b;
      *b = t;
  }

  void heapify(int arr[], int n, int i) {
      int largest = i;
      int left = 2 * i + 1;
      int right = 2 * i + 2;

      if (left < n && arr[left] > arr[largest])
          largest = left;

      if (right < n && arr[right] > arr[largest])
          largest = right;

      if (largest != i) {
          swap(&arr[i], &arr[largest]);
          heapify(arr, n, largest);
      }
  }

  void heapSort(int arr[], int n) {
      for (int i = n / 2 - 1; i >= 0; i--)
          heapify(arr, n, i);

      for (int i = n - 1; i >= 0; i--) {
          swap(&arr[0], &arr[i]);
          heapify(arr, i, 0);
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
      int n = sizeof(arr) / sizeof(arr[0]);

      heapSort(arr, n);

      printf("Sorted array is \n");
      printArray(arr, n);
      return 0;
  }
  ```

**과제:**
- 퀵 정렬, 병합 정렬, 힙 정렬 각각의 알고리즘을 구현하고, 성능을 비교하기 위해 다양한 크기의 배열을 정렬하는 프로그램 작성
- 각 알고리즘의 시간 복잡도를 분석하고, 실제 실행 시간을 측정하여 비교

**퀴즈 및 해설:**

1. **퀵 정렬의 시간 복잡도는 무엇인가요?**
   - 퀵 정렬의 평균 시간 복잡도는 O(n log n)이며, 최악의 경우 시간 복잡도는 O(n^2)입니다. 최악의 경우는 이미 정렬된 배열에서 발생할 수 있습니다.

2. **병합 정렬의 기본 원리는 무엇인가요?**
   - 병합 정렬은 분할 정복 기법을 사용하여 배열을 반으로 나누고, 각 부분을 정렬한 후, 병합하여 전체를 정렬하는 방법입니다.

3. **힙 정렬의 시간 복잡도는 무엇인가요?**
   - 힙 정렬의 평균 및 최악의 시간 복잡도는 모두 O(n log n)입니다.

**해설:**
1. **퀵 정렬의 시간 복잡도**는 분할 정복 기법을 사용하여 평균적으로 O(n log n)이지만, 최악의 경우 O(n^2)입니다. 이는 피벗 선택에 따라 발생할 수 있습니다.
2. **병합 정렬의 기본 원리**는 배열을 반으로 나누고, 각 부분을 정렬한 후, 병합하여 전체를 정렬하는 것입니다. 이 과정은 재귀적으로 수행됩니다.
3. **힙 정렬의 시간 복잡도**는 힙을 구성하는 과정과 힙에서 요소를 제거하는 과정이 모두 O(log n)이기 때문에 전체 정렬 시간 복잡도는 O(n log n)입니다.

---

