### 알고리즘 교육과정 - 7주차: 동적 계획법 (Dynamic Programming)

**강의 목표:**
동적 계획법의 개념과 원리를 이해하고, 이를 활용한 다양한 알고리즘을 학습합니다. 최적 부분 구조와 중복 부분 문제를 이용하여 복잡한 문제를 효율적으로 해결하는 방법을 학습합니다.

**강의 구성:**

#### 7. 동적 계획법 (Dynamic Programming)

**강의 내용:**
- 동적 계획법의 개념과 원리
- 동적 계획법의 기본 패턴
- 대표적인 동적 계획법 문제

**실습:**
- 동적 계획법 알고리즘 구현 및 성능 분석

### 동적 계획법의 개념과 원리

**강의 내용:**
- 동적 계획법이란 무엇인가?
  - 최적 부분 구조와 중복 부분 문제
  - 메모이제이션과 테이블화
- 동적 계획법의 적용 사례
  - 피보나치 수열
  - 최장 공통 부분 수열 (LCS)
  - 배낭 문제

**실습:**
- 피보나치 수열의 동적 계획법 구현 예제
  ```c
  #include <stdio.h>

  int fibonacci(int n) {
      int f[n+2]; // 1 extra to handle case, n = 0
      int i;

      f[0] = 0;
      f[1] = 1;

      for (i = 2; i <= n; i++) {
          f[i] = f[i-1] + f[i-2];
      }

      return f[n];
  }

  int main() {
      int n = 10;
      printf("Fibonacci number %d is %d\n", n, fibonacci(n));
      return 0;
  }
  ```

### 동적 계획법의 기본 패턴

**강의 내용:**
- 하향식 접근법 (Top-Down Approach)
  - 재귀 + 메모이제이션
- 상향식 접근법 (Bottom-Up Approach)
  - 테이블화

**실습:**
- 최장 공통 부분 수열 (LCS) 구현 예제
  ```c
  #include <stdio.h>
  #include <string.h>

  int max(int a, int b) {
      return (a > b)? a : b;
  }

  int lcs(char *X, char *Y, int m, int n) {
      int L[m + 1][n + 1];
      int i, j;

      for (i = 0; i <= m; i++) {
          for (j = 0; j <= n; j++) {
              if (i == 0 || j == 0)
                  L[i][j] = 0;
              else if (X[i - 1] == Y[j - 1])
                  L[i][j] = L[i - 1][j - 1] + 1;
              else
                  L[i][j] = max(L[i - 1][j], L[i][j - 1]);
          }
      }

      return L[m][n];
  }

  int main() {
      char X[] = "AGGTAB";
      char Y[] = "GXTXAYB";
      int m = strlen(X);
      int n = strlen(Y);

      printf("Length of LCS is %d\n", lcs(X, Y, m, n));

      return 0;
  }
  ```

### 대표적인 동적 계획법 문제

**강의 내용:**
- 배낭 문제 (Knapsack Problem)
  - 0/1 배낭 문제
  - 분수 배낭 문제
- 최단 경로 문제
  - 플로이드-워셜 알고리즘

**실습:**
- 0/1 배낭 문제 구현 예제
  ```c
  #include <stdio.h>

  int max(int a, int b) {
      return (a > b)? a : b;
  }

  int knapSack(int W, int wt[], int val[], int n) {
      int i, w;
      int K[n + 1][W + 1];

      for (i = 0; i <= n; i++) {
          for (w = 0; w <= W; w++) {
              if (i == 0 || w == 0)
                  K[i][w] = 0;
              else if (wt[i - 1] <= w)
                  K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w]);
              else
                  K[i][w] = K[i - 1][w];
          }
      }

      return K[n][W];
  }

  int main() {
      int val[] = {60, 100, 120};
      int wt[] = {10, 20, 30};
      int W = 50;
      int n = sizeof(val) / sizeof(val[0]);
      printf("Maximum value in Knapsack = %d\n", knapSack(W, wt, val, n));
      return 0;
  }
  ```

- 플로이드-워셜 알고리즘 구현 예제
  ```c
  #include <stdio.h>

  #define INF 99999
  #define V 4

  void printSolution(int dist[][V]) {
      printf("The following matrix shows the shortest distances between every pair of vertices \n");
      for (int i = 0; i < V; i++) {
          for (int j = 0; j < V; j++) {
              if (dist[i][j] == INF)
                  printf("%7s", "INF");
              else
                  printf("%7d", dist[i][j]);
          }
          printf("\n");
      }
  }

  void floydWarshall(int graph[][V]) {
      int dist[V][V], i, j, k;

      for (i = 0; i < V; i++)
          for (j = 0; j < V; j++)
              dist[i][j] = graph[i][j];

      for (k = 0; k < V; k++) {
          for (i = 0; i < V; i++) {
              for (j = 0; j < V; j++) {
                  if (dist[i][j] > dist[i][k] + dist[k][j])
                      dist[i][j] = dist[i][k] + dist[k][j];
              }
          }
      }

      printSolution(dist);
  }

  int main() {
      int graph[V][V] = {
          {0, 5, INF, 10},
          {INF, 0, 3, INF},
          {INF, INF, 0, 1},
          {INF, INF, INF, 0}
      };

      floydWarshall(graph);
      return 0;
  }
  ```

**과제:**
- 다양한 동적 계획법 문제를 해결하는 알고리즘을 구현하고, 성능을 비교
- 각 알고리즘의 시간 복잡도를 분석하고, 실제 실행 시간을 측정하여 비교

**퀴즈 및 해설:**

1. **동적 계획법이란 무엇인가요?**
   - 동적 계획법은 최적 부분 구조와 중복 부분 문제를 이용하여 복잡한 문제를 효율적으로 해결하는 알고리즘 기법입니다.

2. **동적 계획법의 기본 패턴은 무엇인가요?**
   - 하향식 접근법 (재귀 + 메모이제이션)과 상향식 접근법 (테이블화)이 있습니다.

3. **0/1 배낭 문제의 시간 복잡도는 무엇인가요?**
   - 0/1 배낭 문제의 시간 복잡도는 O(nW)입니다. 여기서 n은 물품의 개수, W는 배낭의 용량입니다.

**해설:**
1. **동적 계획법**은 최적 부분 구조와 중복 부분 문제를 이용하여 복잡한 문제를 효율적으로 해결하는 알고리즘 기법입니다. 이는 큰 문제를 작은 부분 문제로 나누어 해결하고, 해결된 부분 문제의 결과를 저장하여 재사용합니다.
2. **동적 계획법의 기본 패턴**은 하향식 접근법과 상향식 접근법입니다. 하향식 접근법은 재귀와 메모이제이션을 사용하고, 상향식 접근법은 테이블화를 사용하여 문제를 해결합니다.
3. **0/1 배낭 문제의 시간 복잡도**는 O(nW)입니다. 여기서 n은 물품의 개수, W는 배낭의 용량입니다. 이는 동적 계획법을 사용하여 문제를 해결하는 데 필요한 시간입니다.

---

