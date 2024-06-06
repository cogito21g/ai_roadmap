### 알고리즘 교육과정 - 10주차: 최소 신장 트리 (MST) 알고리즘

**강의 목표:**
최소 신장 트리(MST)의 개념과 원리를 이해하고, 이를 활용한 다양한 알고리즘을 학습합니다. 크루스칼 알고리즘과 프림 알고리즘을 중심으로 MST를 효율적으로 구하는 방법을 학습합니다.

**강의 구성:**

#### 10. 최소 신장 트리 (MST) 알고리즘

**강의 내용:**
- MST의 개념과 원리
- 크루스칼 알고리즘
- 프림 알고리즘

**실습:**
- MST 알고리즘 구현 및 성능 분석

### MST의 개념과 원리

**강의 내용:**
- 최소 신장 트리(MST)란 무엇인가?
  - 그래프의 모든 정점을 포함하면서 간선의 가중치 합이 최소가 되는 트리
- MST의 응용
  - 네트워크 설계, 클러스터링 등

**실습:**
- MST의 개념 이해를 위한 간단한 예제

### 크루스칼 알고리즘

**강의 내용:**
- 크루스칼 알고리즘의 개념
  - 간선을 정렬하고 사이클을 이루지 않도록 선택
- 크루스칼 알고리즘의 시간 복잡도
  - O(E log E) (간선의 수 E)

**실습:**
- 크루스칼 알고리즘 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>

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

  struct subset {
      int parent;
      int rank;
  };

  int find(struct subset subsets[], int i) {
      if (subsets[i].parent != i)
          subsets[i].parent = find(subsets, subsets[i].parent);
      return subsets[i].parent;
  }

  void Union(struct subset subsets[], int x, int y) {
      int rootX = find(subsets, x);
      int rootY = find(subsets, y);

      if (subsets[rootX].rank < subsets[rootY].rank)
          subsets[rootX].parent = rootY;
      else if (subsets[rootX].rank > subsets[rootY].rank)
          subsets[rootY].parent = rootX;
      else {
          subsets[rootY].parent = rootX;
          subsets[rootX].rank++;
      }
  }

  int compare(const void* a, const void* b) {
      struct Edge* a1 = (struct Edge*) a;
      struct Edge* b1 = (struct Edge*) b;
      return a1->weight > b1->weight;
  }

  void KruskalMST(struct Graph* graph) {
      int V = graph->V;
      struct Edge result[V];
      int e = 0;
      int i = 0;

      qsort(graph->edge, graph->E, sizeof(graph->edge[0]), compare);

      struct subset* subsets = (struct subset*) malloc(V * sizeof(struct subset));
      for (int v = 0; v < V; ++v) {
          subsets[v].parent = v;
          subsets[v].rank = 0;
      }

      while (e < V - 1 && i < graph->E) {
          struct Edge next_edge = graph->edge[i++];
          int x = find(subsets, next_edge.src);
          int y = find(subsets, next_edge.dest);

          if (x != y) {
              result[e++] = next_edge;
              Union(subsets, x, y);
          }
      }

      printf("Following are the edges in the constructed MST\n");
      for (i = 0; i < e; ++i)
          printf("%d -- %d == %d\n", result[i].src, result[i].dest, result[i].weight);
      return;
  }

  int main() {
      int V = 4;
      int E = 5;
      struct Graph* graph = createGraph(V, E);

      graph->edge[0].src = 0;
      graph->edge[0].dest = 1;
      graph->edge[0].weight = 10;

      graph->edge[1].src = 0;
      graph->edge[1].dest = 2;
      graph->edge[1].weight = 6;

      graph->edge[2].src = 0;
      graph->edge[2].dest = 3;
      graph->edge[2].weight = 5;

      graph->edge[3].src = 1;
      graph->edge[3].dest = 3;
      graph->edge[3].weight = 15;

      graph->edge[4].src = 2;
      graph->edge[4].dest = 3;
      graph->edge[4].weight = 4;

      KruskalMST(graph);

      return 0;
  }
  ```

### 프림 알고리즘

**강의 내용:**
- 프림 알고리즘의 개념
  - 하나의 정점에서 시작하여 최소 비용 간선을 선택
- 프림 알고리즘의 시간 복잡도
  - O(V^2) (정점의 수 V, 인접 리스트 사용 시 O(E log V))

**실습:**
- 프림 알고리즘 구현 예제
  ```c
  #include <stdio.h>
  #include <limits.h>
  #include <stdbool.h>

  #define V 5

  int minKey(int key[], bool mstSet[]) {
      int min = INT_MAX, min_index;

      for (int v = 0; v < V; v++)
          if (mstSet[v] == false && key[v] < min)
              min = key[v], min_index = v;

      return min_index;
  }

  void printMST(int parent[], int n, int graph[V][V]) {
      printf("Edge \tWeight\n");
      for (int i = 1; i < V; i++)
          printf("%d - %d \t%d \n", parent[i], i, graph[i][parent[i]]);
  }

  void primMST(int graph[V][V]) {
      int parent[V];
      int key[V];
      bool mstSet[V];

      for (int i = 0; i < V; i++) {
          key[i] = INT_MAX;
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

      printMST(parent, V, graph);
  }

  int main() {
      int graph[V][V] = {
          {0, 2, 0, 6, 0},
          {2, 0, 3, 8, 5},
          {0, 3, 0, 0, 7},
          {6, 8, 0, 0, 9},
          {0, 5, 7, 9, 0}
      };

      primMST(graph);

      return 0;
  }
  ```

**과제:**
- 다양한 그래프에 대해 크루스칼 알고리즘과 프림 알고리즘을 구현하고, MST를 구하는 프로그램 작성
- 각 알고리즘의 시간 복잡도를 분석하고, 실제 실행 시간을 측정하여 비교

**퀴즈 및 해설:**

1. **최소 신장 트리(MST)란 무엇인가요?**
   - MST는 그래프의 모든 정점을 포함하면서 간선의 가중치 합이 최소가 되는 트리입니다.

2. **크루스칼 알고리즘의 시간 복잡도는 무엇인가요?**
   - 크루스칼 알고리즘의 시간 복잡도는 O(E log E)입니다. 여기서 E는 그래프의 간선 수입니다.

3. **프림 알고리즘의 시간 복잡도는 무엇인가요?**
   - 프림 알고리즘의 시간 복잡도는 O(V^2)입니다. 인접 리스트를 사용하면 O(E log V)로 최적화할 수 있습니다.

**해설:**
1. **MST의 정의**는 그래프의 모든 정점을 포함하면서 간선의 가중치 합이 최소가 되는 트리입니다. 이는 네트워크 설계와 같은 다양한 응용에서 중요한 개념입니다.
2. **크루스칼 알고리즘의 시간 복잡도**는 간선을 정렬하는 과정이 O(E log E)이며, 그 후의 합집합 연산이 거의 O(1

)에 가깝기 때문에 전체 시간 복잡도는 O(E log E)입니다.
3. **프림 알고리즘의 시간 복잡도**는 기본적으로 O(V^2)입니다. 이는 인접 행렬을 사용할 때의 시간 복잡도이며, 인접 리스트와 최소 힙을 사용하면 O(E log V)로 최적화할 수 있습니다.

---
