### 알고리즘 교육과정 - 11주차: 최단 경로 알고리즘

**강의 목표:**
최단 경로 알고리즘의 개념과 원리를 이해하고, 이를 활용한 다양한 알고리즘을 학습합니다. 다익스트라 알고리즘, 벨만-포드 알고리즘, 그리고 플로이드-워셜 알고리즘을 중심으로 최단 경로를 효율적으로 구하는 방법을 학습합니다.

**강의 구성:**

#### 11. 최단 경로 알고리즘

**강의 내용:**
- 최단 경로 문제의 개념과 원리
- 다익스트라 알고리즘
- 벨만-포드 알고리즘
- 플로이드-워셜 알고리즘

**실습:**
- 최단 경로 알고리즘 구현 및 성능 분석

### 최단 경로 문제의 개념과 원리

**강의 내용:**
- 최단 경로 문제란 무엇인가?
  - 그래프의 두 정점 사이의 최단 경로를 찾는 문제
- 최단 경로 문제의 응용
  - 네비게이션 시스템, 네트워크 라우팅 등

**실습:**
- 최단 경로 문제의 개념 이해를 위한 간단한 예제

### 다익스트라 알고리즘

**강의 내용:**
- 다익스트라 알고리즘의 개념
  - 시작 정점에서 다른 모든 정점까지의 최단 경로를 구하는 알고리즘
  - 가중치가 음수가 아닌 경우에 사용
- 다익스트라 알고리즘의 시간 복잡도
  - O(V^2) (정점의 수 V, 인접 리스트 사용 시 O(E log V))

**실습:**
- 다익스트라 알고리즘 구현 예제
  ```c
  #include <limits.h>
  #include <stdbool.h>
  #include <stdio.h>

  #define V 9

  int minDistance(int dist[], bool sptSet[]) {
      int min = INT_MAX, min_index;

      for (int v = 0; v < V; v++)
          if (sptSet[v] == false && dist[v] <= min)
              min = dist[v], min_index = v;

      return min_index;
  }

  void printSolution(int dist[], int n) {
      printf("Vertex \t Distance from Source\n");
      for (int i = 0; i < V; i++)
          printf("%d \t\t %d\n", i, dist[i]);
  }

  void dijkstra(int graph[V][V], int src) {
      int dist[V];
      bool sptSet[V];

      for (int i = 0; i < V; i++)
          dist[i] = INT_MAX, sptSet[i] = false;

      dist[src] = 0;

      for (int count = 0; count < V - 1; count++) {
          int u = minDistance(dist, sptSet);
          sptSet[u] = true;

          for (int v = 0; v < V; v++)
              if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX && dist[u] + graph[u][v] < dist[v])
                  dist[v] = dist[u] + graph[u][v];
      }

      printSolution(dist, V);
  }

  int main() {
      int graph[V][V] = {{0, 4, 0, 0, 0, 0, 0, 8, 0},
                         {4, 0, 8, 0, 0, 0, 0, 11, 0},
                         {0, 8, 0, 7, 0, 4, 0, 0, 2},
                         {0, 0, 7, 0, 9, 14, 0, 0, 0},
                         {0, 0, 0, 9, 0, 10, 0, 0, 0},
                         {0, 0, 4, 14, 10, 0, 2, 0, 0},
                         {0, 0, 0, 0, 0, 2, 0, 1, 6},
                         {8, 11, 0, 0, 0, 0, 1, 0, 7},
                         {0, 0, 2, 0, 0, 0, 6, 7, 0}};

      dijkstra(graph, 0);

      return 0;
  }
  ```

### 벨만-포드 알고리즘

**강의 내용:**
- 벨만-포드 알고리즘의 개념
  - 모든 간선에 대해 반복적으로 완화 작업을 수행하여 최단 경로를 찾는 알고리즘
  - 음수 가중치가 있는 경우에도 사용 가능
- 벨만-포드 알고리즘의 시간 복잡도
  - O(VE) (정점의 수 V, 간선의 수 E)

**실습:**
- 벨만-포드 알고리즘 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>
  #include <limits.h>

  struct Edge {
      int src, dest, weight;
  };

  struct Graph {
      int V, E;
      struct Edge* edge;
  };

  struct Graph* createGraph(int V, int E) {
      struct Graph* graph = (struct Graph*) malloc(sizeof(struct Graph));
      graph->V = V;
      graph->E = E;
      graph->edge = (struct Edge*) malloc(graph->E * sizeof(struct Edge));
      return graph;
  }

  void printArr(int dist[], int n) {
      printf("Vertex   Distance from Source\n");
      for (int i = 0; i < n; ++i)
          printf("%d \t\t %d\n", i, dist[i]);
  }

  void BellmanFord(struct Graph* graph, int src) {
      int V = graph->V;
      int E = graph->E;
      int dist[V];

      for (int i = 0; i < V; i++)
          dist[i] = INT_MAX;
      dist[src] = 0;

      for (int i = 1; i <= V - 1; i++) {
          for (int j = 0; j < E; j++) {
              int u = graph->edge[j].src;
              int v = graph->edge[j].dest;
              int weight = graph->edge[j].weight;
              if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
                  dist[v] = dist[u] + weight;
          }
      }

      for (int i = 0; i < E; i++) {
          int u = graph->edge[i].src;
          int v = graph->edge[i].dest;
          int weight = graph->edge[i].weight;
          if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
              printf("Graph contains negative weight cycle\n");
              return;
          }
      }

      printArr(dist, V);
  }

  int main() {
      int V = 5;
      int E = 8;
      struct Graph* graph = createGraph(V, E);

      graph->edge[0].src = 0;
      graph->edge[0].dest = 1;
      graph->edge[0].weight = -1;

      graph->edge[1].src = 0;
      graph->edge[1].dest = 2;
      graph->edge[1].weight = 4;

      graph->edge[2].src = 1;
      graph->edge[2].dest = 2;
      graph->edge[2].weight = 3;

      graph->edge[3].src = 1;
      graph->edge[3].dest = 3;
      graph->edge[3].weight = 2;

      graph->edge[4].src = 1;
      graph->edge[4].dest = 4;
      graph->edge[4].weight = 2;

      graph->edge[5].src = 3;
      graph->edge[5].dest = 2;
      graph->edge[5].weight = 5;

      graph->edge[6].src = 3;
      graph->edge[6].dest = 1;
      graph->edge[6].weight = 1;

      graph->edge[7].src = 4;
      graph->edge[7].dest = 3;
      graph->edge[7].weight = -3;

      BellmanFord(graph, 0);

      return 0;
  }
  ```

### 플로이드-워셜 알고리즘

**강의 내용:**
- 플로이드-워셜 알고리즘의 개념
  - 모든 정점 사이의 최단 경로를 구하는 알고리즘
- 플로이드-워셜 알고리즘의 시간 복잡도
  - O(V^3) (정점의 수 V)

**실습:**
- 플로이드-워셜 알고리즘 구현 예제
  ```c
  #include <stdio.h>

  #define V 4


  #define INF 99999

  void printSolution(int dist[][V]) {
      printf("Following matrix shows the shortest distances between every pair of vertices\n");
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
                  if (dist[i][k] + dist[k][j] < dist[i][j])
                      dist[i][j] = dist[i][k] + dist[k][j];
              }
          }
      }

      printSolution(dist);
  }

  int main() {
      int graph[V][V] = {{0, 5, INF, 10},
                         {INF, 0, 3, INF},
                         {INF, INF, 0, 1},
                         {INF, INF, INF, 0}};

      floydWarshall(graph);
      return 0;
  }
  ```

**과제:**
- 다양한 그래프에 대해 다익스트라 알고리즘, 벨만-포드 알고리즘, 플로이드-워셜 알고리즘을 구현하고, 최단 경로를 구하는 프로그램 작성
- 각 알고리즘의 시간 복잡도를 분석하고, 실제 실행 시간을 측정하여 비교

**퀴즈 및 해설:**

1. **다익스트라 알고리즘의 시간 복잡도는 무엇인가요?**
   - 다익스트라 알고리즘의 시간 복잡도는 O(V^2)이며, 인접 리스트를 사용하면 O(E log V)로 최적화할 수 있습니다.

2. **벨만-포드 알고리즘의 시간 복잡도는 무엇인가요?**
   - 벨만-포드 알고리즘의 시간 복잡도는 O(VE)입니다.

3. **플로이드-워셜 알고리즘의 시간 복잡도는 무엇인가요?**
   - 플로이드-워셜 알고리즘의 시간 복잡도는 O(V^3)입니다.

**해설:**
1. **다익스트라 알고리즘의 시간 복잡도**는 인접 행렬을 사용하면 O(V^2)이며, 인접 리스트와 최소 힙을 사용하면 O(E log V)로 최적화할 수 있습니다. 이는 시작 정점에서 다른 모든 정점까지의 최단 경로를 구하는 데 필요한 시간입니다.
2. **벨만-포드 알고리즘의 시간 복잡도**는 O(VE)입니다. 이는 모든 간선에 대해 반복적으로 완화 작업을 수행하여 최단 경로를 찾는 데 필요한 시간입니다.
3. **플로이드-워셜 알고리즘의 시간 복잡도**는 O(V^3)입니다. 이는 모든 정점 사이의 최단 경로를 구하는 데 필요한 시간입니다.

---

