### 알고리즘 교육과정 - 15주차: 네트워크 플로우 알고리즘

**강의 목표:**
네트워크 플로우의 개념과 원리를 이해하고, 이를 해결하기 위한 다양한 알고리즘을 학습합니다. 포드-풀커슨 알고리즘과 에드몬드-카프 알고리즘을 중심으로 네트워크 플로우 문제를 효율적으로 해결하는 방법을 학습합니다.

**강의 구성:**

#### 15. 네트워크 플로우 알고리즘

**강의 내용:**
- 네트워크 플로우의 개념과 원리
- 최대 유량 문제 (Maximum Flow Problem)
- 포드-풀커슨 알고리즘
- 에드몬드-카프 알고리즘

**실습:**
- 네트워크 플로우 알고리즘 구현 및 성능 분석

### 네트워크 플로우의 개념과 원리

**강의 내용:**
- 네트워크 플로우란 무엇인가?
  - 유량(Flow)와 용량(Capacity)
  - 소스(Source)와 싱크(Sink)
- 네트워크 플로우 문제의 응용
  - 교통 네트워크, 데이터 네트워크, 공급망 등

**실습:**
- 네트워크 플로우의 기본 개념 이해를 위한 간단한 예제

### 최대 유량 문제 (Maximum Flow Problem)

**강의 내용:**
- 최대 유량 문제의 정의
  - 네트워크에서 소스에서 싱크로 보낼 수 있는 최대 유량을 찾는 문제
- 용어 정리
  - 잔여 그래프(Residual Graph), 증강 경로(Augmenting Path), 컷(Cut)

**실습:**
- 최대 유량 문제 예제

### 포드-풀커슨 알고리즘

**강의 내용:**
- 포드-풀커슨 알고리즘의 개념
  - 증강 경로를 찾아 유량을 증가시키는 방법
- 포드-풀커슨 알고리즘의 시간 복잡도
  - O(E * max_flow) (간선의 수 E, 최대 유량 max_flow)

**실습:**
- 포드-풀커슨 알고리즘 구현 예제
  ```c
  #include <stdio.h>
  #include <limits.h>
  #include <stdbool.h>
  #include <string.h>

  #define V 6

  bool bfs(int rGraph[V][V], int s, int t, int parent[]) {
      bool visited[V];
      memset(visited, 0, sizeof(visited));

      int queue[V], front = 0, rear = 0;
      queue[rear++] = s;
      visited[s] = true;
      parent[s] = -1;

      while (front < rear) {
          int u = queue[front++];

          for (int v = 0; v < V; v++) {
              if (visited[v] == false && rGraph[u][v] > 0) {
                  queue[rear++] = v;
                  parent[v] = u;
                  visited[v] = true;
              }
          }
      }

      return (visited[t] == true);
  }

  int fordFulkerson(int graph[V][V], int s, int t) {
      int u, v;
      int rGraph[V][V];

      for (u = 0; u < V; u++)
          for (v = 0; v < V; v++)
              rGraph[u][v] = graph[u][v];

      int parent[V];
      int max_flow = 0;

      while (bfs(rGraph, s, t, parent)) {
          int path_flow = INT_MAX;
          for (v = t; v != s; v = parent[v]) {
              u = parent[v];
              path_flow = path_flow < rGraph[u][v] ? path_flow : rGraph[u][v];
          }

          for (v = t; v != s; v = parent[v]) {
              u = parent[v];
              rGraph[u][v] -= path_flow;
              rGraph[v][u] += path_flow;
          }

          max_flow += path_flow;
      }

      return max_flow;
  }

  int main() {
      int graph[V][V] = { {0, 16, 13, 0, 0, 0},
                          {0, 0, 10, 12, 0, 0},
                          {0, 4, 0, 0, 14, 0},
                          {0, 0, 9, 0, 0, 20},
                          {0, 0, 0, 7, 0, 4},
                          {0, 0, 0, 0, 0, 0} };

      printf("The maximum possible flow is %d\n", fordFulkerson(graph, 0, 5));
      return 0;
  }
  ```

### 에드몬드-카프 알고리즘

**강의 내용:**
- 에드몬드-카프 알고리즘의 개념
  - 포드-풀커슨 알고리즘의 BFS를 이용한 구현
- 에드몬드-카프 알고리즘의 시간 복잡도
  - O(VE^2) (정점의 수 V, 간선의 수 E)

**실습:**
- 에드몬드-카프 알고리즘 구현 예제
  ```c
  #include <stdio.h>
  #include <limits.h>
  #include <stdbool.h>
  #include <string.h>

  #define V 6

  bool bfs(int rGraph[V][V], int s, int t, int parent[]) {
      bool visited[V];
      memset(visited, 0, sizeof(visited));

      int queue[V], front = 0, rear = 0;
      queue[rear++] = s;
      visited[s] = true;
      parent[s] = -1;

      while (front < rear) {
          int u = queue[front++];

          for (int v = 0; v < V; v++) {
              if (visited[v] == false && rGraph[u][v] > 0) {
                  queue[rear++] = v;
                  parent[v] = u;
                  visited[v] = true;
              }
          }
      }

      return (visited[t] == true);
  }

  int edmondsKarp(int graph[V][V], int s, int t) {
      int u, v;
      int rGraph[V][V];

      for (u = 0; u < V; u++)
          for (v = 0; v < V; v++)
              rGraph[u][v] = graph[u][v];

      int parent[V];
      int max_flow = 0;

      while (bfs(rGraph, s, t, parent)) {
          int path_flow = INT_MAX;
          for (v = t; v != s; v = parent[v]) {
              u = parent[v];
              path_flow = path_flow < rGraph[u][v] ? path_flow : rGraph[u][v];
          }

          for (v = t; v != s; v = parent[v]) {
              u = parent[v];
              rGraph[u][v] -= path_flow;
              rGraph[v][u] += path_flow;
          }

          max_flow += path_flow;
      }

      return max_flow;
  }

  int main() {
      int graph[V][V] = { {0, 16, 13, 0, 0, 0},
                          {0, 0, 10, 12, 0, 0},
                          {0, 4, 0, 0, 14, 0},
                          {0, 0, 9, 0, 0, 20},
                          {0, 0, 0, 7, 0, 4},
                          {0, 0, 0, 0, 0, 0} };

      printf("The maximum possible flow is %d\n", edmondsKarp(graph, 0, 5));
      return 0;
  }
  ```

**과제:**
- 다양한 네트워크 플로우 문제를 해결하기 위한 알고리즘을 구현하고, 성능을 비교
- 각 알고리즘의 시간 복잡도를 분석하고, 실제 실행 시간을 측정하여 비교

**퀴즈 및 해설:**

1. **네트워크 플로우란 무엇인가요?**
   - 네트워크 플로우는 네트워크 내에서 소스에서 싱크로 데이터를 흐르게 하는 것을 의미하며, 각 간선에는 용량이 있습니다.

2. **포드-풀커슨 알고리즘의 기본 원리는 무엇인가요?**
   - 포드-풀커슨 알고리즘은 증강 경로를 찾아 유량을 증가시키는 방법으로, 가능한 경로를 찾고 잔여 용량을 갱신하여 최대 유량을 찾습니다.

3. **에드몬드-카프 알고리즘의 시간 복잡도는 무엇인가요?**
   - 에드몬드-카프 알고리즘의 시간 복잡도는 O(VE^2)입니다. 이는 BFS를 사용하여 증강 경로를 찾기 때문에 포드-풀커슨 알고리즘

보다 더 효율적입니다.

**해설:**
1. **네트워크 플로우**는 네트워크 내에서 소스에서 싱크로 데이터를 흐르게 하는 것을 의미하며, 각 간선에는 용량이 있습니다. 최대 유량 문제는 소스에서 싱크로 보낼 수 있는 최대 유량을 찾는 문제입니다.
2. **포드-풀커슨 알고리즘의 기본 원리**는 증강 경로를 찾아 유량을 증가시키는 방법으로, 가능한 경로를 찾고 잔여 용량을 갱신하여 최대 유량을 찾습니다. 이는 반복적으로 증강 경로를 찾아 유량을 증가시키는 방식입니다.
3. **에드몬드-카프 알고리즘의 시간 복잡도**는 O(VE^2)입니다. 이는 BFS를 사용하여 증강 경로를 찾기 때문에 포드-풀커슨 알고리즘보다 더 효율적입니다. 이는 간선의 수와 정점의 수에 따라 성능이 결정됩니다.

---
