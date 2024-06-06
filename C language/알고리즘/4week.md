### 알고리즘 교육과정 - 4주차: 검색 알고리즘

**강의 목표:**
검색 알고리즘의 기본 개념과 구현 방법을 이해하고, 각 알고리즘의 시간 복잡도를 분석합니다.

**강의 구성:**

#### 4. 검색 알고리즘

**강의 내용:**
- 선형 검색
- 이진 검색

**실습:**
- 각 검색 알고리즘의 구현 및 성능 비교

### 선형 검색

**강의 내용:**
- 선형 검색의 개념
  - 배열의 첫 번째 요소부터 순차적으로 검색하는 방법
- 선형 검색의 시간 복잡도
  - 평균 및 최악의 시간 복잡도: O(n)

**실습:**
- 선형 검색 구현 예제
  ```c
  #include <stdio.h>

  int linearSearch(int arr[], int n, int x) {
      for (int i = 0; i < n; i++) {
          if (arr[i] == x)
              return i;
      }
      return -1;
  }

  int main() {
      int arr[] = {2, 3, 4, 10, 40};
      int x = 10;
      int n = sizeof(arr) / sizeof(arr[0]);
      int result = linearSearch(arr, n, x);
      if (result != -1)
          printf("Element is present at index %d\n", result);
      else
          printf("Element is not present in array\n");
      return 0;
  }
  ```

### 이진 검색

**강의 내용:**
- 이진 검색의 개념
  - 정렬된 배열을 절반씩 나누어 검색하는 방법
- 이진 검색의 시간 복잡도
  - 평균 및 최악의 시간 복잡도: O(log n)

**실습:**
- 이진 검색 구현 예제
  ```c
  #include <stdio.h>

  int binarySearch(int arr[], int l, int r, int x) {
      while (l <= r) {
          int m = l + (r - l) / 2;

          if (arr[m] == x)
              return m;

          if (arr[m] < x)
              l = m + 1;
          else
              r = m - 1;
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

**과제:**
- 선형 검색과 이진 검색 각각의 알고리즘을 구현하고, 성능을 비교하기 위해 다양한 크기의 배열에서 특정 값을 검색하는 프로그램 작성
- 각 알고리즘의 시간 복잡도를 분석하고, 실제 실행 시간을 측정하여 비교

**퀴즈 및 해설:**

1. **선형 검색의 시간 복잡도는 무엇인가요?**
   - 선형 검색의 평균 및 최악의 시간 복잡도는 O(n)입니다.

2. **이진 검색의 기본 원리는 무엇인가요?**
   - 이진 검색은 정렬된 배열에서 절반씩 나누어 검색하는 방법입니다. 중앙 값을 기준으로 검색 범위를 반으로 줄여가며 값을 찾습니다.

3. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    int arr[] = {2, 3, 4, 10, 40};
    int x = 10;
    int n = sizeof(arr) / sizeof(arr[0]);
    int result = binarySearch(arr, 0, n - 1, x);
    if (result != -1)
        printf("Element is present at index %d\n", result);
    else
        printf("Element is not present in array\n");
    ```
   - 출력 결과:
     ```
     Element is present at index 3
     ```

**해설:**
1. **선형 검색의 시간 복잡도**는 배열의 모든 요소를 확인해야 하므로 O(n)입니다.
2. **이진 검색의 기본 원리**는 정렬된 배열을 절반씩 나누어 검색하는 것입니다. 중앙 값을 기준으로 검색 범위를 반으로 줄여가며 값을 찾습니다.
3. **코드 출력 결과**는 배열에서 10을 이진 검색하여 인덱스 3에서 찾는 것입니다.

---
