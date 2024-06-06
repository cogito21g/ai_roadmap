### 알고리즘 교육과정 - 18주차: 고급 정렬 알고리즘

**강의 목표:**
고급 정렬 알고리즘의 개념과 원리를 이해하고, 이를 활용한 다양한 알고리즘을 학습합니다. 힙 정렬, 기수 정렬, 그리고 병합 정렬 등 복잡한 문제를 효율적으로 해결하는 방법을 학습합니다.

**강의 구성:**

#### 18. 고급 정렬 알고리즘

**강의 내용:**
- 힙 정렬 (Heap Sort)
- 기수 정렬 (Radix Sort)
- 병합 정렬 (Merge Sort)
- 팀소트 (Timsort)

**실습:**
- 고급 정렬 알고리즘 구현 및 성능 분석

### 힙 정렬 (Heap Sort)

**강의 내용:**
- 힙 정렬의 개념
  - 완전 이진 트리 기반의 정렬 알고리즘
- 힙 정렬의 시간 복잡도
  - O(n log n)

**실습:**
- 힙 정렬 구현 예제
  ```c
  #include <stdio.h>

  void heapify(int arr[], int n, int i) {
      int largest = i;
      int left = 2 * i + 1;
      int right = 2 * i + 2;

      if (left < n && arr[left] > arr[largest])
          largest = left;

      if (right < n && arr[right] > arr[largest])
          largest = right;

      if (largest != i) {
          int temp = arr[i];
          arr[i] = arr[largest];
          arr[largest] = temp;
          heapify(arr, n, largest);
      }
  }

  void heapSort(int arr[], int n) {
      for (int i = n / 2 - 1; i >= 0; i--)
          heapify(arr, n, i);

      for (int i = n - 1; i > 0; i--) {
          int temp = arr[0];
          arr[0] = arr[i];
          arr[i] = temp;
          heapify(arr, i, 0);
      }
  }

  void printArray(int arr[], int n) {
      for (int i = 0; i < n; i++)
          printf("%d ", arr[i]);
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

### 기수 정렬 (Radix Sort)

**강의 내용:**
- 기수 정렬의 개념
  - 자릿수 별로 정렬하는 기법
- 기수 정렬의 시간 복잡도
  - O(d * (n + k)) (d는 자릿수, n은 데이터 수, k는 기수)

**실습:**
- 기수 정렬 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>

  int getMax(int arr[], int n) {
      int max = arr[0];
      for (int i = 1; i < n; i++)
          if (arr[i] > max)
              max = arr[i];
      return max;
  }

  void countSort(int arr[], int n, int exp) {
      int *output = (int *)malloc(n * sizeof(int));
      int i, count[10] = {0};

      for (i = 0; i < n; i++)
          count[(arr[i] / exp) % 10]++;

      for (i = 1; i < 10; i++)
          count[i] += count[i - 1];

      for (i = n - 1; i >= 0; i--) {
          output[count[(arr[i] / exp) % 10] - 1] = arr[i];
          count[(arr[i] / exp) % 10]--;
      }

      for (i = 0; i < n; i++)
          arr[i] = output[i];

      free(output);
  }

  void radixSort(int arr[], int n) {
      int m = getMax(arr, n);
      for (int exp = 1; m / exp > 0; exp *= 10)
          countSort(arr, n, exp);
  }

  void printArray(int arr[], int n) {
      for (int i = 0; i < n; i++)
          printf("%d ", arr[i]);
      printf("\n");
  }

  int main() {
      int arr[] = {170, 45, 75, 90, 802, 24, 2, 66};
      int n = sizeof(arr) / sizeof(arr[0]);

      radixSort(arr, n);

      printf("Sorted array is \n");
      printArray(arr, n);

      return 0;
  }
  ```

### 병합 정렬 (Merge Sort)

**강의 내용:**
- 병합 정렬의 개념
  - 분할 정복 기법을 이용한 정렬 알고리즘
- 병합 정렬의 시간 복잡도
  - O(n log n)

**실습:**
- 병합 정렬 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>

  void merge(int arr[], int l, int m, int r) {
      int i, j, k;
      int n1 = m - l + 1;
      int n2 = r - m;

      int *L = (int *)malloc(n1 * sizeof(int));
      int *R = (int *)malloc(n2 * sizeof(int));

      for (i = 0; i < n1; i++)
          L[i] = arr[l + i];
      for (j = 0; j < n2; j++)
          R[j] = arr[m + 1 + j];

      i = 0;
      j = 0;
      k = l;
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

      free(L);
      free(R);
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
      for (int i = 0; i < size; i++)
          printf("%d ", arr[i]);
      printf("\n");
  }

  int main() {
      int arr[] = {12, 11, 13, 5, 6, 7};
      int arr_size = sizeof(arr) / sizeof(arr[0]);

      printf("Given array is \n");
      printArray(arr, arr_size);

      mergeSort(arr, 0, arr_size - 1);

      printf("Sorted array is \n");
      printArray(arr, arr_size);
      return 0;
  }
  ```

### 팀소트 (Timsort)

**강의 내용:**
- 팀소트의 개념
  - 병합 정렬과 삽입 정렬을 결합한 정렬 알고리즘
- 팀소트의 시간 복잡도
  - O(n log n)

**실습:**
- 팀소트 알고리즘의 구현 (C에서는 직접 구현이 복잡하므로 설명에 집중)

**과제:**
- 다양한 고급 정렬 알고리즘을 구현하고, 성능을 비교
- 각 알고리즘의 시간 복잡도를 분석하고, 실제 실행 시간을 측정하여 비교

**퀴즈 및 해설:**

1. **힙 정렬의 시간 복잡도는 무엇인가요?**
   - 힙 정렬의 시간 복잡도는 O(n log n)입니다. 이는 힙 구성과 정렬 과정에서 발생하는 복잡도입니다.

2. **기수 정렬의 기본 원리는 무엇인가요?**
   - 기수 정렬은 자릿수 별로 정렬하는 기법으로, 각 자릿수에 대해 기수 정렬을 수행하여 전체 배열을 정렬합니다.

3. **병합 정렬의 시간 복잡도는 무엇인가요?**
   - 병합 정렬의 시간 복잡도는 O(n log n)입니다. 이는 배열을 분할하고 병합하는 과정에서 발생하는 복잡도입니다.

**해설:**
1. **힙 정렬의 시간 복잡도**는 O(n log n)입니다. 이는 힙 구성과 정렬 과정에서 발생하는 복잡도로, 각 단계에서 힙을 재구성하는 과정이 로그 복잡도를 가지기 때문입니다.
2. **기수

 정렬의 기본 원리**는 자릿수 별로 정렬하는 기법으로, 각 자릿수에 대해 기수 정렬을 수행하여 전체 배열을 정렬합니다. 이는 각 자릿수를 기준으로 정렬하는 과정을 반복하여 전체 배열을 정렬하는 방식입니다.
3. **병합 정렬의 시간 복잡도**는 O(n log n)입니다. 이는 배열을 분할하고 병합하는 과정에서 발생하는 복잡도로, 배열을 반씩 나누고 병합하는 과정에서 로그 복잡도를 가지기 때문입니다.

---
