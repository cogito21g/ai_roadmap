### 알고리즘 교육과정 - 6주차: 재귀 알고리즘

**강의 목표:**
재귀의 개념과 원리를 이해하고, 재귀를 이용한 알고리즘을 학습합니다. 재귀 알고리즘의 시간 복잡도를 분석하고, 이를 최적화하는 방법을 학습합니다.

**강의 구성:**

#### 6. 재귀 알고리즘

**강의 내용:**
- 재귀의 개념과 원리
- 재귀를 이용한 기본 알고리즘
- 재귀 알고리즘의 최적화

**실습:**
- 재귀 알고리즘 구현 및 성능 분석

### 재귀의 개념과 원리

**강의 내용:**
- 재귀란 무엇인가?
  - 함수가 자기 자신을 호출하는 프로그래밍 기법
- 재귀의 기본 구조
  - 기저 조건 (Base case)
  - 재귀 호출 (Recursive case)
- 재귀의 장단점
  - 문제를 간단하고 직관적으로 표현 가능
  - 높은 메모리 사용량과 성능 저하 가능성

**실습:**
- 재귀 함수 예제
  ```c
  #include <stdio.h>

  int factorial(int n) {
      if (n == 0)
          return 1;
      else
          return n * factorial(n - 1);
  }

  int main() {
      int num = 5;
      printf("Factorial of %d is %d\n", num, factorial(num));
      return 0;
  }
  ```

### 재귀를 이용한 기본 알고리즘

**강의 내용:**
- 피보나치 수열
- 하노이의 탑
- 재귀적 이진 검색

**실습:**
- 피보나치 수열 구현 예제
  ```c
  #include <stdio.h>

  int fibonacci(int n) {
      if (n <= 1)
          return n;
      return fibonacci(n - 1) + fibonacci(n - 2);
  }

  int main() {
      int n = 10;
      printf("Fibonacci series for %d terms: ", n);
      for (int i = 0; i < n; i++) {
          printf("%d ", fibonacci(i));
      }
      printf("\n");
      return 0;
  }
  ```

- 하노이의 탑 구현 예제
  ```c
  #include <stdio.h>

  void towerOfHanoi(int n, char from_rod, char to_rod, char aux_rod) {
      if (n == 1) {
          printf("Move disk 1 from rod %c to rod %c\n", from_rod, to_rod);
          return;
      }
      towerOfHanoi(n - 1, from_rod, aux_rod, to_rod);
      printf("Move disk %d from rod %c to rod %c\n", n, from_rod, to_rod);
      towerOfHanoi(n - 1, aux_rod, to_rod, from_rod);
  }

  int main() {
      int n = 3;
      towerOfHanoi(n, 'A', 'C', 'B');
      return 0;
  }
  ```

- 재귀적 이진 검색 구현 예제
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

### 재귀 알고리즘의 최적화

**강의 내용:**
- 메모이제이션 (Memoization)
- 꼬리 재귀 최적화 (Tail Recursion Optimization)

**실습:**
- 메모이제이션을 이용한 피보나치 수열 최적화
  ```c
  #include <stdio.h>

  int fibonacci(int n, int memo[]) {
      if (n <= 1)
          return n;
      if (memo[n] == -1)
          memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo);
      return memo[n];
  }

  int main() {
      int n = 10;
      int memo[n + 1];
      for (int i = 0; i <= n; i++)
          memo[i] = -1;
      printf("Fibonacci series for %d terms: ", n);
      for (int i = 0; i < n; i++) {
          printf("%d ", fibonacci(i, memo));
      }
      printf("\n");
      return 0;
  }
  ```

**과제:**
- 재귀를 이용한 다양한 문제를 해결하는 알고리즘을 구현하고, 성능을 비교
- 재귀 알고리즘을 메모이제이션을 이용하여 최적화하고, 성능 차이를 분석

**퀴즈 및 해설:**

1. **재귀의 기본 구조는 무엇인가요?**
   - 재귀의 기본 구조는 기저 조건 (Base case)과 재귀 호출 (Recursive case)로 구성됩니다.

2. **피보나치 수열의 재귀적 구현의 시간 복잡도는 무엇인가요?**
   - 피보나치 수열의 재귀적 구현의 시간 복잡도는 O(2^n)입니다.

3. **메모이제이션이란 무엇인가요?**
   - 메모이제이션은 동일한 계산을 반복하지 않도록 계산 결과를 저장하여 재사용하는 기법입니다. 이는 재귀 알고리즘의 성능을 크게 향상시킬 수 있습니다.

**해설:**
1. **재귀의 기본 구조**는 기저 조건과 재귀 호출로 구성됩니다. 기저 조건은 재귀 호출이 멈추는 조건이며, 재귀 호출은 함수가 자기 자신을 호출하는 부분입니다.
2. **피보나치 수열의 재귀적 구현의 시간 복잡도**는 중복된 계산이 많아 O(2^n)입니다. 이는 비효율적이며, 메모이제이션을 통해 최적화할 수 있습니다.
3. **메모이제이션**은 동일한 계산을 반복하지 않도록 계산 결과를 저장하여 재사용하는 기법입니다. 이는 재귀 알고리즘의 성능을 크게 향상시킬 수 있습니다.

---
