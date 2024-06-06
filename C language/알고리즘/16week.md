### 알고리즘 교육과정 - 16주차: 고급 그래프 알고리즘

**강의 목표:**
고급 그래프 알고리즘의 개념과 원리를 이해하고, 이를 활용한 다양한 알고리즘을 학습합니다. 위상 정렬, 타잔의 강한 연결 요소 알고리즘, 최소 스패닝 트리 알고리즘 등을 중심으로 복잡한 그래프 문제를 효율적으로 해결하는 방법을 학습합니다.

**강의 구성:**

#### 16. 고급 그래프 알고리즘

**강의 내용:**
- 위상 정렬 (Topological Sort)
- 타잔의 강한 연결 요소 (SCC) 알고리즘
- 최소 스패닝 트리 (MST) 알고리즘

**실습:**
- 고급 그래프 알고리즘 구현 및 성능 분석

### 위상 정렬 (Topological Sort)

**강의 내용:**
- 위상 정렬의 개념
  - 방향 그래프의 정점들을 선형 순서로 나열하는 방법
- 위상 정렬의 응용
  - 작업 스케줄링, 의존성 해결 등
- 위상 정렬의 시간 복잡도
  - O(V + E) (정점의 수 V, 간선의 수 E)

**실습:**
- 위상 정렬 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>

  struct AdjListNode {
      int dest;
      struct AdjListNode* next;
  };

  struct AdjList {
      struct AdjListNode* head;
  };

  struct Graph {
      int V;
      struct AdjList* array;
  };

  struct AdjListNode* newAdjListNode(int dest) {
      struct AdjListNode* newNode = (struct AdjListNode*) malloc(sizeof(struct AdjListNode));
      newNode->dest = dest;
      newNode->next = NULL;
      return newNode;
  }

  struct Graph* createGraph(int V) {
      struct Graph* graph = (struct Graph*) malloc(sizeof(struct Graph));
      graph->V = V;
      graph->array = (struct AdjList*) malloc(V * sizeof(struct AdjList));
      for (int i = 0; i < V; ++i)
          graph->array[i].head = NULL;
      return graph;
  }

  void addEdge(struct Graph* graph, int src, int dest) {
      struct AdjListNode* newNode = newAdjListNode(dest);
      newNode->next = graph->array[src].head;
      graph->array[src].head = newNode;
  }

  void topologicalSortUtil(int v, int visited[], struct Graph* graph, int* stack, int* top) {
      visited[v] = 1;
      struct AdjListNode* node = graph->array[v].head;
      while (node != NULL) {
          if (!visited[node->dest])
              topologicalSortUtil(node->dest, visited, graph, stack, top);
          node = node->next;
      }
      stack[(*top)--] = v;
  }

  void topologicalSort(struct Graph* graph) {
      int* stack = (int*) malloc(graph->V * sizeof(int));
      int top = graph->V - 1;
      int* visited = (int*) malloc(graph->V * sizeof(int));
      for (int i = 0; i < graph->V; i++)
          visited[i] = 0;

      for (int i = 0; i < graph->V; i++)
          if (!visited[i])
              topologicalSortUtil(i, visited, graph, stack, &top);

      printf("Topological Sort: ");
      for (int i = 0; i < graph->V; i++)
          printf("%d ", stack[i]);
      printf("\n");

      free(stack);
      free(visited);
  }

  int main() {
      struct Graph* graph = createGraph(6);
      addEdge(graph, 5, 2);
      addEdge(graph, 5, 0);
      addEdge(graph, 4, 0);
      addEdge(graph, 4, 1);
      addEdge(graph, 2, 3);
      addEdge(graph, 3, 1);

      printf("Topological sorting of the given graph:\n");
      topologicalSort(graph);

      return 0;
  }
  ```

### 타잔의 강한 연결 요소 (SCC) 알고리즘

**강의 내용:**
- 강한 연결 요소 (Strongly Connected Components, SCC)의 개념
  - 방향 그래프에서 서로 도달 가능한 정점들의 최대 집합
- 타잔의 SCC 알고리즘의 원리
  - DFS 기반으로 SCC를 찾는 방법
- 타잔의 SCC 알고리즘의 시간 복잡도
  - O(V + E) (정점의 수 V, 간선의 수 E)

**실습:**
- 타잔의 SCC 알고리즘 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>
  #include <stdbool.h>

  struct AdjListNode {
      int dest;
      struct AdjListNode* next;
  };

  struct AdjList {
      struct AdjListNode* head;
  };

  struct Graph {
      int V;
      int* disc;
      int* low;
      bool* stackMember;
      struct AdjListNode** st;
      struct AdjList* array;
  };

  struct AdjListNode* newAdjListNode(int dest) {
      struct AdjListNode* newNode = (struct AdjListNode*) malloc(sizeof(struct AdjListNode));
      newNode->dest = dest;
      newNode->next = NULL;
      return newNode;
  }

  struct Graph* createGraph(int V) {
      struct Graph* graph = (struct Graph*) malloc(sizeof(struct Graph));
      graph->V = V;
      graph->array = (struct AdjList*) malloc(V * sizeof(struct AdjList));
      for (int i = 0; i < V; ++i)
          graph->array[i].head = NULL;

      graph->disc = (int*) malloc(V * sizeof(int));
      graph->low = (int*) malloc(V * sizeof(int));
      graph->stackMember = (bool*) malloc(V * sizeof(bool));
      graph->st = (struct AdjListNode**) malloc(V * sizeof(struct AdjListNode*));
      return graph;
  }

  void addEdge(struct Graph* graph, int src, int dest) {
      struct AdjListNode* newNode = newAdjListNode(dest);
      newNode->next = graph->array[src].head;
      graph->array[src].head = newNode;
  }

  void SCCUtil(struct Graph* graph, int u, int* time) {
      static int stackIndex = 0;
      graph->disc[u] = graph->low[u] = ++(*time);
      graph->stackMember[u] = true;
      graph->st[stackIndex++] = newAdjListNode(u);

      struct AdjListNode* node = graph->array[u].head;
      while (node != NULL) {
          int v = node->dest;
          if (graph->disc[v] == -1) {
              SCCUtil(graph, v, time);
              graph->low[u] = (graph->low[u] < graph->low[v]) ? graph->low[u] : graph->low[v];
          } else if (graph->stackMember[v] == true) {
              graph->low[u] = (graph->low[u] < graph->disc[v]) ? graph->low[u] : graph->disc[v];
          }
          node = node->next;
      }

      int w = 0;
      if (graph->low[u] == graph->disc[u]) {
          while (graph->st[stackIndex - 1]->dest != u) {
              w = graph->st[--stackIndex]->dest;
              printf("%d ", w);
              graph->stackMember[w] = false;
          }
          w = graph->st[--stackIndex]->dest;
          printf("%d\n", w);
          graph->stackMember[w] = false;
      }
  }

  void SCC(struct Graph* graph) {
      int time = 0;
      for (int i = 0; i < graph->V; i++) {
          graph->disc[i] = -1;
          graph->low[i] = -1;
          graph->stackMember[i] = false;
      }

      for (int i = 0; i < graph->V; i++)
          if (graph->disc[i] == -1)
              SCCUtil(graph, i, &time);
  }

  int main() {
      struct Graph* graph = createGraph(5);
      addEdge(graph, 1, 0);
      addEdge(graph, 0, 2);
      addEdge(graph, 2, 1);
      addEdge(graph, 0, 3);
      addEdge(graph, 3, 4);

      printf("Strongly Connected Components in the given graph:\n");
      SCC(graph);

      return 0;
  }
  ```

### 최소 스패닝 트리 (MST) 알고리즘

**강의 내용:**
- MST의 개념과 원리
  - 모든 정점을 포함하면서 간선의 가중치 합이 최소가 되는 트리
- MST 알고리즘의 응용
  - 네트워크 설계, 클러스터링 등
- 크루스칼 알고리즘과 프림 알고리

즘 복습

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

**과제:**
- 다양한 그래프 문제를 해결하기 위한 고급 그래프 알고리즘을 구현하고, 성능을 비교
- 각 알고리즘의 시간 복잡도를 분석하고, 실제 실행 시간을 측정하여 비교

**퀴즈 및 해설:**

1. **위상 정렬의 시간 복잡도는 무엇인가요?**
   - 위상 정렬의 시간 복잡도는 O(V + E)입니다.

2. **타잔의 강한 연결 요소(SCC) 알고리즘의 기본 원리는 무엇인가요?**
   - 타잔의 SCC 알고리즘은 DFS를 이용하여 각 정점에 대해 도달 가능성을 파악하고, 이를 통해 SCC를 찾는 알고리즘입니다.

3. **크루스칼 알고리즘의 시간 복잡도는 무엇인가요?**
   - 크루스칼 알고리즘의 시간 복잡도는 O(E log E)입니다. 이는 간선을 정렬하는 과정이 주된 복잡도를 차지합니다.

**해설:**
1. **위상 정렬의 시간 복잡도**는 O(V + E)입니다. 이는 모든 정점과 간선을 한 번씩 방문하여 정렬하는 과정에서 발생하는 복잡도입니다.
2. **타잔의 SCC 알고리즘의 기본 원리**는 DFS를 이용하여 각 정점에 대해 도달 가능성을 파악하고, 이를 통해 SCC를 찾는 알고리즘입니다. 이는 DFS의 방문 순서를 이용하여 효율적으로 SCC를 찾을 수 있습니다.
3. **크루스칼 알고리즘의 시간 복잡도**는 O(E log E)입니다. 이는 간선을 정렬하는 과정이 주된 복잡도를 차지하며, 간선의 수 E에 비례하여 로그 복잡도를 가집니다.

---

