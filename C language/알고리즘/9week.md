### 알고리즘 교육과정 - 9주차: 그래프 알고리즘 기초

**강의 목표:**
그래프의 기본 개념과 표현 방법을 이해하고, 깊이 우선 탐색(DFS)과 너비 우선 탐색(BFS) 알고리즘을 학습합니다. 이를 통해 그래프 탐색 및 관련 문제 해결 능력을 기릅니다.

**강의 구성:**

#### 9. 그래프 알고리즘 기초

**강의 내용:**
- 그래프의 기본 개념
- 그래프의 표현 방법
- 깊이 우선 탐색 (DFS)
- 너비 우선 탐색 (BFS)

**실습:**
- 그래프의 표현 및 탐색 알고리즘 구현

### 그래프의 기본 개념

**강의 내용:**
- 그래프란 무엇인가?
  - 정점(Vertex)과 간선(Edge)
- 그래프의 종류
  - 방향 그래프(Directed Graph)
  - 무방향 그래프(Undirected Graph)
  - 가중치 그래프(Weighted Graph)
  - 비가중치 그래프(Unweighted Graph)

**실습:**
- 그래프의 기본 개념 이해를 위한 간단한 예제

### 그래프의 표현 방법

**강의 내용:**
- 인접 행렬 (Adjacency Matrix)
- 인접 리스트 (Adjacency List)

**실습:**
- 인접 행렬과 인접 리스트 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>

// 인접 행렬 방식의 그래프 구현
  void printAdjMatrix(int adjMatrix[][5], int vertices) {
      for (int i = 0; i < vertices; i++) {
          for (int j = 0; j < vertices; j++) {
              printf("%d ", adjMatrix[i][j]);
          }
          printf("\n");
      }
  }

  void adjMatrixExample() {
      int vertices = 5;
      int adjMatrix[5][5] = {0};

      adjMatrix[0][1] = 1;
      adjMatrix[0][4] = 1;
      adjMatrix[1][2] = 1;
      adjMatrix[1][3] = 1;
      adjMatrix[1][4] = 1;
      adjMatrix[2][3] = 1;
      adjMatrix[3][4] = 1;

      printf("Adjacency Matrix:\n");
      printAdjMatrix(adjMatrix, vertices);
  }

// 인접 리스트 방식의 그래프 구현
  struct Node {
      int dest;
      struct Node* next;
  };

  struct AdjList {
      struct Node* head;
  };

  struct Graph {
      int vertices;
      struct AdjList* array;
  };

  struct Node* createNode(int dest) {
      struct Node* newNode = (struct Node*) malloc(sizeof(struct Node));
      newNode->dest = dest;
      newNode->next = NULL;
      return newNode;
  }

  struct Graph* createGraph(int vertices) {
      struct Graph* graph = (struct Graph*) malloc(sizeof(struct Graph));
      graph->vertices = vertices;
      graph->array = (struct AdjList*) malloc(vertices * sizeof(struct AdjList));

      for (int i = 0; i < vertices; i++) {
          graph->array[i].head = NULL;
      }
      return graph;
  }

  void addEdge(struct Graph* graph, int src, int dest) {
      struct Node* newNode = createNode(dest);
      newNode->next = graph->array[src].head;
      graph->array[src].head = newNode;

      newNode = createNode(src);
      newNode->next = graph->array[dest].head;
      graph->array[dest].head = newNode;
  }

  void printGraph(struct Graph* graph) {
      for (int v = 0; v < graph->vertices; v++) {
          struct Node* temp = graph->array[v].head;
          printf("\n Adjacency list of vertex %d\n head ", v);
          while (temp) {
              printf("-> %d", temp->dest);
              temp = temp->next;
          }
          printf("\n");
      }
  }

  void adjListExample() {
      int vertices = 5;
      struct Graph* graph = createGraph(vertices);

      addEdge(graph, 0, 1);
      addEdge(graph, 0, 4);
      addEdge(graph, 1, 2);
      addEdge(graph, 1, 3);
      addEdge(graph, 1, 4);
      addEdge(graph, 2, 3);
      addEdge(graph, 3, 4);

      printf("Adjacency List:\n");
      printGraph(graph);
  }

  int main() {
      adjMatrixExample();
      adjListExample();
      return 0;
  }
  ```

### 깊이 우선 탐색 (DFS)

**강의 내용:**
- DFS의 개념
  - 시작 정점에서 출발하여 가능한 깊이까지 탐색한 후, 다시 돌아와 다른 경로를 탐색하는 방법
- DFS의 구현 방법
  - 재귀적 구현
  - 비재귀적 구현 (스택 사용)

**실습:**
- 깊이 우선 탐색 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>

  struct Node {
      int dest;
      struct Node* next;
  };

  struct AdjList {
      struct Node* head;
  };

  struct Graph {
      int vertices;
      struct AdjList* array;
  };

  struct Node* createNode(int dest) {
      struct Node* newNode = (struct Node*) malloc(sizeof(struct Node));
      newNode->dest = dest;
      newNode->next = NULL;
      return newNode;
  }

  struct Graph* createGraph(int vertices) {
      struct Graph* graph = (struct Graph*) malloc(sizeof(struct Graph));
      graph->vertices = vertices;
      graph->array = (struct AdjList*) malloc(vertices * sizeof(struct AdjList));

      for (int i = 0; i < vertices; i++) {
          graph->array[i].head = NULL;
      }
      return graph;
  }

  void addEdge(struct Graph* graph, int src, int dest) {
      struct Node* newNode = createNode(dest);
      newNode->next = graph->array[src].head;
      graph->array[src].head = newNode;

      newNode = createNode(src);
      newNode->next = graph->array[dest].head;
      graph->array[dest].head = newNode;
  }

  void DFSUtil(int v, int visited[], struct Graph* graph) {
      visited[v] = 1;
      printf("%d ", v);
      struct Node* temp = graph->array[v].head;
      while (temp) {
          int connectedVertex = temp->dest;
          if (!visited[connectedVertex]) {
              DFSUtil(connectedVertex, visited, graph);
          }
          temp = temp->next;
      }
  }

  void DFS(struct Graph* graph, int startVertex) {
      int* visited = (int*) malloc(graph->vertices * sizeof(int));
      for (int i = 0; i < graph->vertices; i++)
          visited[i] = 0;

      DFSUtil(startVertex, visited, graph);
  }

  int main() {
      int vertices = 5;
      struct Graph* graph = createGraph(vertices);

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

### 너비 우선 탐색 (BFS)

**강의 내용:**
- BFS의 개념
  - 시작 정점에서 출발하여 가까운 정점부터 탐색하며, 큐를 이용해 구현하는 방법
- BFS의 구현 방법
  - 큐를 사용한 구현

**실습:**
- 너비 우선 탐색 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>

  struct Node {
      int dest;
      struct Node* next;
  };

  struct AdjList {
      struct Node* head;
  };

  struct Graph {
      int vertices;
      struct AdjList* array;
  };

  struct Node* createNode(int dest) {
      struct Node* newNode = (struct Node*) malloc(sizeof(struct Node));
      newNode->dest = dest;
      newNode->next = NULL;
      return newNode;
  }

  struct Graph* createGraph(int vertices) {
      struct Graph* graph = (struct Graph*) malloc(sizeof(struct Graph));
      graph->vertices = vertices;
      graph->array = (struct AdjList*) malloc(vertices * sizeof(struct AdjList));

      for (int i = 0; i < vertices; i++) {
          graph->array[i].head = NULL;
      }
      return graph;
  }

  void addEdge(struct Graph* graph, int src, int dest) {
      struct Node* newNode = createNode(dest);
      newNode->next = graph->array[src].head;
      graph->array[src].head = newNode

;

      newNode = createNode(src);
      newNode->next = graph->array[dest].head;
      graph->array[dest].head = newNode;
  }

  void BFS(int startVertex, struct Graph* graph) {
      int* visited = (int*) malloc(graph->vertices * sizeof(int));
      for (int i = 0; i < graph->vertices; i++)
          visited[i] = 0;

      struct Node** queue = (struct Node**) malloc(graph->vertices * sizeof(struct Node*));
      int front = 0, rear = 0;

      visited[startVertex] = 1;
      queue[rear++] = createNode(startVertex);

      while (front < rear) {
          int currentVertex = queue[front++]->dest;
          printf("%d ", currentVertex);

          struct Node* temp = graph->array[currentVertex].head;
          while (temp) {
              int adjVertex = temp->dest;
              if (!visited[adjVertex]) {
                  queue[rear++] = createNode(adjVertex);
                  visited[adjVertex] = 1;
              }
              temp = temp->next;
          }
      }
      free(queue);
  }

  int main() {
      int vertices = 5;
      struct Graph* graph = createGraph(vertices);

      addEdge(graph, 0, 1);
      addEdge(graph, 0, 4);
      addEdge(graph, 1, 2);
      addEdge(graph, 1, 3);
      addEdge(graph, 1, 4);
      addEdge(graph, 2, 3);
      addEdge(graph, 3, 4);

      printf("Breadth First Traversal starting from vertex 0:\n");
      BFS(0, graph);

      return 0;
  }
  ```

**과제:**
- 그래프를 인접 행렬 및 인접 리스트로 구현하고, DFS와 BFS를 통해 각 정점을 방문하는 순서를 출력하는 프로그램 작성
- DFS와 BFS를 이용하여 그래프의 모든 연결 요소를 찾는 함수를 구현

**퀴즈 및 해설:**

1. **깊이 우선 탐색(DFS)이란 무엇인가요?**
   - DFS는 그래프 탐색 알고리즘으로, 시작 정점에서 출발하여 가능한 깊이까지 탐색한 후, 다시 돌아와 다른 경로를 탐색하는 방법입니다.

2. **너비 우선 탐색(BFS)이란 무엇인가요?**
   - BFS는 그래프 탐색 알고리즘으로, 시작 정점에서 출발하여 가까운 정점부터 탐색하며, 큐를 이용해 구현합니다. 각 정점을 방문한 후, 방문한 정점과 인접한 정점들을 차례로 탐색합니다.

3. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    int vertices = 5;
    struct Graph* graph = createGraph(vertices);

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
    BFS(0, graph);
    ```
   - 출력 결과:
     ```
     Depth First Traversal starting from vertex 0:
     0 1 2 3 4 
     Breadth First Traversal starting from vertex 0:
     0 1 4 2 3 
     ```

**해설:**
1. **깊이 우선 탐색(DFS)**은 그래프 탐색 알고리즘으로, 시작 정점에서 출발하여 가능한 깊이까지 탐색한 후, 다시 돌아와 다른 경로를 탐색하는 방법입니다.
2. **너비 우선 탐색(BFS)**은 그래프 탐색 알고리즘으로, 시작 정점에서 출발하여 가까운 정점부터 탐색하며, 큐를 이용해 구현합니다. 각 정점을 방문한 후, 방문한 정점과 인접한 정점들을 차례로 탐색합니다.
3. **코드 출력 결과**는 그래프의 깊이 우선 탐색(DFS)과 너비 우선 탐색(BFS)의 결과를 보여줍니다. DFS는 가능한 깊이까지 탐색한 후, 다시 돌아와 다른 경로를 탐색하며, BFS는 가까운 정점부터 차례로 탐색합니다.

---

다음 주차 내용을 원하시면 알려주세요. 10주차는 최소 신장 트리(MST) 알고리즘을 다룰 예정입니다.