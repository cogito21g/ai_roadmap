### 알고리즘 교육과정 - 14주차: NP-완전 문제와 근사 알고리즘

**강의 목표:**
NP-완전 문제의 개념과 원리를 이해하고, 이를 해결하기 위한 근사 알고리즘을 학습합니다. 대표적인 NP-완전 문제와 이를 해결하는 근사 알고리즘을 중심으로 복잡한 문제를 효율적으로 해결하는 방법을 학습합니다.

**강의 구성:**

#### 14. NP-완전 문제와 근사 알고리즘

**강의 내용:**
- NP-완전 문제의 개념과 원리
- 대표적인 NP-완전 문제
- 근사 알고리즘

**실습:**
- NP-완전 문제와 근사 알고리즘 구현 및 성능 분석

### NP-완전 문제의 개념과 원리

**강의 내용:**
- NP-완전 문제란 무엇인가?
  - NP(Nondeterministic Polynomial time) 문제
  - NP-완전 문제의 정의와 예
- NP-완전 문제의 응용
  - 그래프 이론, 최적화 문제 등

**실습:**
- NP-완전 문제의 개념 이해를 위한 간단한 예제

### 대표적인 NP-완전 문제

**강의 내용:**
- 배낭 문제 (Knapsack Problem)
  - 0/1 배낭 문제
  - 분수 배낭 문제 (NP-어려움)
- 집합 커버 문제 (Set Cover Problem)
- 여행하는 세일즈맨 문제 (Traveling Salesman Problem)

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

### 근사 알고리즘

**강의 내용:**
- 근사 알고리즘의 개념
  - 최적해를 찾기 어려운 문제에 대해 근사해를 구하는 알고리즘
- 근사 알고리즘의 성능 평가
  - 근사 비율 (Approximation Ratio)
- 대표적인 근사 알고리즘
  - 배낭 문제의 근사 알고리즘
  - 집합 커버 문제의 근사 알고리즘
  - 여행하는 세일즈맨 문제의 근사 알고리즘

**실습:**
- 근사 알고리즘 구현 예제
  ```c
  #include <stdio.h>
  #include <stdbool.h>

  #define V 4
  #define INF 99999

  void printSolution(int path[][V]) {
      printf("Following matrix shows the shortest distances between every pair of vertices\n");
      for (int i = 0; i < V; i++) {
          for (int j = 0; j < V; j++) {
              if (path[i][j] == INF)
                  printf("%7s", "INF");
              else
                  printf("%7d", path[i][j]);
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
                  if (dist[i][k] + dist[k][j] < dist[i][j])
                      dist[i][j] = dist[i][k] + dist[k][j];
              }
          }
      }

      printSolution(dist);
  }

  int main() {
      int graph[V][V] = {{0, 3, INF, 7},
                         {8, 0, 2, INF},
                         {5, INF, 0, 1},
                         {2, INF, INF, 0}};

      floydWarshall(graph);
      return 0;
  }
  ```

### 여행하는 세일즈맨 문제의 근사 알고리즘

**강의 내용:**
- 여행하는 세일즈맨 문제의 개념
  - 모든 도시를 한 번씩 방문하고 시작점으로 돌아오는 최단 경로 찾기
- 근사 알고리즘의 적용
  - 최근접 이웃 알고리즘 (Nearest Neighbor Algorithm)
  - 2-근사 알고리즘

**실습:**
- 최근접 이웃 알고리즘 구현 예제
  ```c
  #include <stdio.h>
  #include <stdbool.h>

  #define V 5

  int minKey(int key[], bool mstSet[]) {
      int min = 99999, min_index;

      for (int v = 0; v < V; v++)
          if (mstSet[v] == false && key[v] < min)
              min = key[v], min_index = v;

      return min_index;
  }

  void printMST(int parent[], int graph[V][V]) {
      printf("Edge \tWeight\n");
      for (int i = 1; i < V; i++)
          printf("%d - %d \t%d \n", parent[i], i, graph[i][parent[i]]);
  }

  void primMST(int graph[V][V]) {
      int parent[V];
      int key[V];
      bool mstSet[V];

      for (int i = 0; i < V; i++) {
          key[i] = 99999;
          mstSet[i] = false;
      }

      key[0] = 0;
      parent[0] = -1;

      for (int count = 0; count < V - 1; count++) {
          int u = minKey(key, mstSet);

          mstSet[u] = true;

          for (int v = 0; v < V; v++)
              if (graph[u][v] && mstSet[v] == false && graph[u][v] < key[v])
                  parent[v] = u, key[v] = graph[u][v];
      }

      printMST(parent, graph);
  }

  int main() {
      int graph[V][V] = {{0, 2, 0, 6, 0},
                         {2, 0, 3, 8, 5},
                         {0, 3, 0, 0, 7},
                         {6, 8, 0, 0, 9},
                         {0, 5, 7, 9, 0}};

      primMST(graph);

      return 0;
  }
  ```

**과제:**
- 다양한 NP-완전 문제를 해결하기 위한 근사 알고리즘을 구현하고, 성능을 비교
- 각 알고리즘의 근사 비율을 분석하고, 실제 실행 시간을 측정하여 비교

**퀴즈 및 해설:**

1. **NP-완전 문제란 무엇인가요?**
   - NP-완전 문제는 결정적 다항 시간 내에 풀 수 없는 문제로, 모든 NP 문제를 다항 시간 내에 변환할 수 있는 문제를 말합니다.

2. **근사 알고리즘의 기본 원리는 무엇인가요?**
   - 근사 알고리즘은 최적해를 찾기 어려운 문제에 대해 근사해를 구하는 알고리즘으로, 최적해에 가까운 해를 빠르게 찾는 것을 목표로 합니다.

3. **배낭 문제의 근사 알고리즘의 근사 비율은 어떻게 평가하나요?**
   - 배낭 문제의 근사 알고리즘의 근사 비율은 구한 해가 최적해에 얼마나 근접한지를 평가하며, 최적해 대비 근사해의 비율로 나타냅니다.

**해설:**
1. **NP-완전 문제**는 결정적 다항 시간 내에 풀 수 없는 문제로, 모든 NP 문제를 다항 시간 내에 변환할 수 있는 문제를 말합니다. 이는 최적해를 찾기 어려운 문제의 집

합으로, 대표적으로 배낭 문제, 집합 커버 문제, 여행하는 세일즈맨 문제가 있습니다.
2. **근사 알고리즘의 기본 원리**는 최적해를 찾기 어려운 문제에 대해 근사해를 구하는 알고리즘으로, 최적해에 가까운 해를 빠르게 찾는 것을 목표로 합니다. 이는 복잡한 문제를 현실적으로 해결하는 데 유용합니다.
3. **배낭 문제의 근사 알고리즘의 근사 비율**은 구한 해가 최적해에 얼마나 근접한지를 평가하며, 최적해 대비 근사해의 비율로 나타냅니다. 이는 알고리즘의 성능을 평가하는 중요한 기준이 됩니다.

---
