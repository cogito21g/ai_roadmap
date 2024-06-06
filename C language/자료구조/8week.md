### 자료구조 교육과정 - 8주차: 그래프 기초와 탐색 알고리즘

**강의 목표:**
그래프의 개념과 기본적인 탐색 알고리즘을 학습하고, 그래프 탐색을 통해 다양한 문제를 해결하는 방법을 이해합니다.

**강의 구성:**

#### 8. 그래프 기초와 탐색 알고리즘

### 그래프 기초

**강의 내용:**
- 그래프의 개념
  - 그래프란 무엇인가?
  - 그래프의 구성 요소: 정점(Vertex)과 간선(Edge)
- 그래프의 표현 방법
  - 인접 행렬
  - 인접 리스트
- 그래프의 종류
  - 무방향 그래프와 방향 그래프
  - 가중치 그래프

**실습:**
- 그래프 구현 예제 (인접 리스트)
  ```c
  #include <stdio.h>
  #include <stdlib.h>

  struct AdjListNode {
      int dest;
      struct AdjListNode* next;
  };

  struct AdjList {
      struct AdjListNode *head;
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

      newNode = newAdjListNode(src);
      newNode->next = graph->array[dest].head;
      graph->array[dest].head = newNode;
  }

  void printGraph(struct Graph* graph) {
      for (int v = 0; v < graph->V; ++v) {
          struct AdjListNode* pCrawl = graph->array[v].head;
          printf("\n Adjacency list of vertex %d\n head ", v);
          while (pCrawl) {
              printf("-> %d", pCrawl->dest);
              pCrawl = pCrawl->next;
          }
          printf("\n");
      }
  }

  int main() {
      int V = 5;
      struct Graph* graph = createGraph(V);
      addEdge(graph, 0, 1);
      addEdge(graph, 0, 4);
      addEdge(graph, 1, 2);
      addEdge(graph, 1, 3);
      addEdge(graph, 1, 4);
      addEdge(graph, 2, 3);
      addEdge(graph, 3, 4);

      printGraph(graph);

      return 0;
  }
  ```

**과제:**
- 그래프를 인접 리스트로 구현하고, 특정 정점에 대한 인접 정점들을 출력하는 함수를 작성
- 그래프에서 특정 정점과 연결된 모든 간선들을 삭제하는 함수를 구현

**퀴즈 및 해설:**

1. **그래프란 무엇인가요?**
   - 그래프는 정점(Vertex)와 간선(Edge)으로 이루어진 데이터 구조로, 정점은 객체를, 간선은 객체 간의 관계를 나타냅니다.

2. **그래프의 인접 리스트 표현 방법은 무엇인가요?**
   - 인접 리스트는 그래프의 각 정점에 연결된 모든 인접 정점들을 리스트로 표현하는 방법입니다. 각 정점은 인접 리스트 배열의 인덱스로 사용되며, 리스트에는 해당 정점과 연결된 모든 정점들이 포함됩니다.

3. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    int V = 5;
    struct Graph* graph = createGraph(V);
    addEdge(graph, 0, 1);
    addEdge(graph, 0, 4);
    addEdge(graph, 1, 2);
    addEdge(graph, 1, 3);
    addEdge(graph, 1, 4);
    addEdge(graph, 2, 3);
    addEdge(graph, 3, 4);

    printGraph(graph);
    ```
   - 출력 결과:
     ```
      Adjacency list of vertex 0
       head -> 4 -> 1

      Adjacency list of vertex 1
       head -> 4 -> 3 -> 2 -> 0

      Adjacency list of vertex 2
       head -> 3 -> 1

      Adjacency list of vertex 3
       head -> 4 -> 2 -> 1

      Adjacency list of vertex 4
       head -> 3 -> 1 -> 0
     ```

**해설:**
1. **그래프의 정의**는 정점(Vertex)와 간선(Edge)으로 이루어진 데이터 구조로, 정점은 객체를, 간선은 객체 간의 관계를 나타냅니다.
2. **그래프의 인접 리스트 표현 방법**은 그래프의 각 정점에 연결된 모든 인접 정점들을 리스트로 표현하는 방법입니다. 각 정점은 인접 리스트 배열의 인덱스로 사용되며, 리스트에는 해당 정점과 연결된 모든 정점들이 포함됩니다.
3. **코드 출력 결과**는 그래프의 인접 리스트를 출력합니다. 각 정점에 연결된 인접 정점들이 리스트 형태로 출력됩니다.

---

### 탐색 알고리즘

**강의 내용:**
- 깊이 우선 탐색 (DFS)
  - DFS의 개념과 구현 방법
  - DFS의 응용: 경로 탐색, 연결 요소 찾기
- 너비 우선 탐색 (BFS)
  - BFS의 개념과 구현 방법
  - BFS의 응용: 최단 경로 찾기, 레벨 순회

**실습:**
- DFS 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>
  #define MAX 100

  struct Graph {
      int V;
      int adj[MAX][MAX];
  };

  struct Graph* createGraph(int V) {
      struct Graph* graph = (struct Graph*) malloc(sizeof(struct Graph));
      graph->V = V;
      for (int i = 0; i < V; i++) {
          for (int j = 0; j < V; j++) {
              graph->adj[i][j] = 0;
          }
      }
      return graph;
  }

  void addEdge(struct Graph* graph, int src, int dest) {
      graph->adj[src][dest] = 1;
      graph->adj[dest][src] = 1;
  }

  void DFSUtil(int v, int visited[], struct Graph* graph) {
      visited[v] = 1;
      printf("%d ", v);
      for (int i = 0; i < graph->V; i++) {
          if (graph->adj[v][i] == 1 && !visited[i]) {
              DFSUtil(i, visited, graph);
          }
      }
  }

  void DFS(struct Graph* graph, int v) {
      int visited[MAX] = {0};
      DFSUtil(v, visited, graph);
  }

  int main() {
      int V = 5;
      struct Graph* graph = createGraph(V);
      addEdge(graph, 0, 1);
      addEdge(graph, 0, 4);
      addEdge(graph, 1, 2);
      addEdge(graph, 1, 3);
      addEdge(graph, 1, 4);
      addEdge(graph, 2, 3);
      addEdge(graph, 3, 4);

      printf("Depth First Traversal starting from vertex 0:\n");
      DFS(graph, 0);

      return 0;
  }
  ```

- BFS 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>
  #define MAX 100

  struct Queue {
      int items[MAX];
      int front, rear;
  };

  struct Queue* createQueue() {
      struct Queue* q = (struct Queue*) malloc(sizeof(struct Queue));
      q->front = q->rear = -1;
      return q;
  }

  int isEmpty(struct Queue* q) {
      return q->front == -1;
  }

  void enqueue(struct Queue* q, int value) {
      if (q->rear == MAX - 1)
          return;
      if (q->front == -1)
          q->front = 0;
      q->rear++;
      q->items[q->rear] = value;
  }

  int dequeue(struct Queue* q) {
      if (isEmpty(q))
          return -1;
      int item = q->items[q->front];
      q->front++;
      if (q->front > q->rear) {
          q->front = q->rear = -1;


      }
      return item;
  }

  struct Graph {
      int V;
      int adj[MAX][MAX];
  };

  struct Graph* createGraph(int V) {
      struct Graph* graph = (struct Graph*) malloc(sizeof(struct Graph));
      graph->V = V;
      for (int i = 0; i < V; i++) {
          for (int j = 0; j < V; j++) {
              graph->adj[i][j] = 0;
          }
      }
      return graph;
  }

  void addEdge(struct Graph* graph, int src, int dest) {
      graph->adj[src][dest] = 1;
      graph->adj[dest][src] = 1;
  }

  void BFS(struct Graph* graph, int startVertex) {
      struct Queue* q = createQueue();
      int visited[MAX] = {0};

      visited[startVertex] = 1;
      enqueue(q, startVertex);

      while (!isEmpty(q)) {
          int currentVertex = dequeue(q);
          printf("%d ", currentVertex);

          for (int i = 0; i < graph->V; i++) {
              if (graph->adj[currentVertex][i] == 1 && !visited[i]) {
                  visited[i] = 1;
                  enqueue(q, i);
              }
          }
      }
  }

  int main() {
      int V = 5;
      struct Graph* graph = createGraph(V);
      addEdge(graph, 0, 1);
      addEdge(graph, 0, 4);
      addEdge(graph, 1, 2);
      addEdge(graph, 1, 3);
      addEdge(graph, 1, 4);
      addEdge(graph, 2, 3);
      addEdge(graph, 3, 4);

      printf("Breadth First Traversal starting from vertex 0:\n");
      BFS(graph, 0);

      return 0;
  }
  ```

**과제:**
- 그래프를 인접 행렬로 구현하고, DFS와 BFS 탐색을 통해 각 정점을 방문하는 순서를 출력하는 함수를 작성
- DFS와 BFS를 이용하여 그래프의 모든 연결 요소를 찾는 함수를 구현

**퀴즈 및 해설:**

1. **깊이 우선 탐색(DFS)이란 무엇인가요?**
   - DFS는 그래프 탐색 알고리즘으로, 시작 정점에서 출발하여 가능한 멀리까지 정점을 탐색한 후, 다시 돌아와 다른 경로를 탐색하는 방법입니다. 재귀적으로 구현할 수 있습니다.

2. **너비 우선 탐색(BFS)이란 무엇인가요?**
   - BFS는 그래프 탐색 알고리즘으로, 시작 정점에서 출발하여 가까운 정점부터 탐색하며, 큐를 이용해 구현할 수 있습니다. 각 정점을 방문한 후, 방문한 정점과 인접한 정점들을 차례로 탐색합니다.

3. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    int V = 5;
    struct Graph* graph = createGraph(V);
    addEdge(graph, 0, 1);
    addEdge(graph, 0, 4);
    addEdge(graph, 1, 2);
    addEdge(graph, 1, 3);
    addEdge(graph, 1, 4);
    addEdge(graph, 2, 3);
    addEdge(graph, 3, 4);

    printf("Depth First Traversal starting from vertex 0:\n");
    DFS(graph, 0);

    printf("Breadth First Traversal starting from vertex 0:\n");
    BFS(graph, 0);
    ```
   - 출력 결과:
     ```
     Depth First Traversal starting from vertex 0:
     0 1 2 3 4 
     Breadth First Traversal starting from vertex 0:
     0 1 4 2 3 
     ```

**해설:**
1. **깊이 우선 탐색(DFS)**은 그래프 탐색 알고리즘으로, 시작 정점에서 출발하여 가능한 멀리까지 정점을 탐색한 후, 다시 돌아와 다른 경로를 탐색하는 방법입니다. 재귀적으로 구현할 수 있습니다.
2. **너비 우선 탐색(BFS)**은 그래프 탐색 알고리즘으로, 시작 정점에서 출발하여 가까운 정점부터 탐색하며, 큐를 이용해 구현할 수 있습니다. 각 정점을 방문한 후, 방문한 정점과 인접한 정점들을 차례로 탐색합니다.
3. **코드 출력 결과**는 그래프의 깊이 우선 탐색(DFS)과 너비 우선 탐색(BFS)의 결과를 보여줍니다. DFS는 가능한 깊이까지 탐색한 후, 다시 돌아와 다른 경로를 탐색하며, BFS는 가까운 정점부터 차례로 탐색합니다.

---
