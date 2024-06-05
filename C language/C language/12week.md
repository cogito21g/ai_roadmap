### C 언어 20주차 심화 교육과정 - 12주차: 데이터 구조 심화

#### 12주차: 데이터 구조 심화

**강의 목표:**
12주차의 목표는 트리, 이진 탐색 트리, 그래프와 같은 고급 데이터 구조를 이해하고, 이를 활용하여 효율적인 데이터 관리를 수행하는 능력을 기르는 것입니다.

**강의 구성:**

##### 1. 트리 (Tree)
- **강의 내용:**
  - 트리의 개념
    - 트리는 계층적 데이터 구조로, 루트 노드와 자식 노드로 구성됨
  - 트리의 용도
    - 파일 시스템, 조직도, 표현식 트리 등
  - 트리의 기본 용어
    - 루트, 부모, 자식, 형제, 리프 노드 등
  - 트리의 순회 방법
    - 전위 순회 (Preorder)
    - 중위 순회 (Inorder)
    - 후위 순회 (Postorder)
- **실습:**
  - 트리의 전위, 중위, 후위 순회 구현 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>

    struct Node {
        int data;
        struct Node* left;
        struct Node* right;
    };

    struct Node* newNode(int data) {
        struct Node* node = (struct Node*)malloc(sizeof(struct Node));
        node->data = data;
        node->left = node->right = NULL;
        return node;
    }

    void printPreorder(struct Node* node) {
        if (node == NULL)
            return;
        printf("%d ", node->data);
        printPreorder(node->left);
        printPreorder(node->right);
    }

    void printInorder(struct Node* node) {
        if (node == NULL)
            return;
        printInorder(node->left);
        printf("%d ", node->data);
        printInorder(node->right);
    }

    void printPostorder(struct Node* node) {
        if (node == NULL)
            return;
        printPostorder(node->left);
        printPostorder(node->right);
        printf("%d ", node->data);
    }

    int main() {
        struct Node* root = newNode(1);
        root->left = newNode(2);
        root->right = newNode(3);
        root->left->left = newNode(4);
        root->left->right = newNode(5);

        printf("Preorder traversal: ");
        printPreorder(root);
        printf("\n");

        printf("Inorder traversal: ");
        printInorder(root);
        printf("\n");

        printf("Postorder traversal: ");
        printPostorder(root);
        printf("\n");

        return 0;
    }
    ```

##### 2. 이진 탐색 트리 (Binary Search Tree)
- **강의 내용:**
  - 이진 탐색 트리의 개념
    - 각 노드의 왼쪽 서브트리에는 작은 값, 오른쪽 서브트리에는 큰 값이 저장됨
  - 이진 탐색 트리의 삽입, 삭제, 탐색 연산
  - 이진 탐색 트리의 장점
    - 효율적인 검색, 삽입, 삭제 연산
- **실습:**
  - 이진 탐색 트리의 삽입, 삭제, 탐색 구현 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>

    struct Node {
        int data;
        struct Node* left;
        struct Node* right;
    };

    struct Node* newNode(int data) {
        struct Node* node = (struct Node*)malloc(sizeof(struct Node));
        node->data = data;
        node->left = node->right = NULL;
        return node;
    }

    struct Node* insert(struct Node* node, int data) {
        if (node == NULL)
            return newNode(data);

        if (data < node->data)
            node->left = insert(node->left, data);
        else if (data > node->data)
            node->right = insert(node->right, data);

        return node;
    }

    struct Node* minValueNode(struct Node* node) {
        struct Node* current = node;

        while (current && current->left != NULL)
            current = current->left;

        return current;
    }

    struct Node* deleteNode(struct Node* root, int data) {
        if (root == NULL)
            return root;

        if (data < root->data)
            root->left = deleteNode(root->left, data);
        else if (data > root->data)
            root->right = deleteNode(root->right, data);
        else {
            if (root->left == NULL) {
                struct Node* temp = root->right;
                free(root);
                return temp;
            } else if (root->right == NULL) {
                struct Node* temp = root->left;
                free(root);
                return temp;
            }

            struct Node* temp = minValueNode(root->right);
            root->data = temp->data;
            root->right = deleteNode(root->right, temp->data);
        }

        return root;
    }

    struct Node* search(struct Node* root, int data) {
        if (root == NULL || root->data == data)
            return root;

        if (root->data < data)
            return search(root->right, data);

        return search(root->left, data);
    }

    void inorder(struct Node* root) {
        if (root != NULL) {
            inorder(root->left);
            printf("%d ", root->data);
            inorder(root->right);
        }
    }

    int main() {
        struct Node* root = NULL;
        root = insert(root, 50);
        insert(root, 30);
        insert(root, 20);
        insert(root, 40);
        insert(root, 70);
        insert(root, 60);
        insert(root, 80);

        printf("Inorder traversal: ");
        inorder(root);
        printf("\n");

        printf("Delete 20\n");
        root = deleteNode(root, 20);
        printf("Inorder traversal: ");
        inorder(root);
        printf("\n");

        printf("Search for 70\n");
        struct Node* result = search(root, 70);
        if (result != NULL)
            printf("Found: %d\n", result->data);
        else
            printf("Not Found\n");

        return 0;
    }
    ```

##### 3. 그래프 (Graph)
- **강의 내용:**
  - 그래프의 개념
    - 정점과 간선으로 구성된 데이터 구조
    - 방향 그래프, 무방향 그래프
  - 그래프의 표현 방법
    - 인접 행렬, 인접 리스트
  - 그래프 순회 알고리즘
    - 깊이 우선 탐색(DFS)
    - 너비 우선 탐색(BFS)
- **실습:**
  - 그래프의 인접 행렬과 인접 리스트 구현 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>

    struct Node {
        int vertex;
        struct Node* next;
    };

    struct Graph {
        int numVertices;
        struct Node** adjLists;
    };

    struct Node* createNode(int v) {
        struct Node* newNode = malloc(sizeof(struct Node));
        newNode->vertex = v;
        newNode->next = NULL;
        return newNode;
    }

    struct Graph* createGraph(int vertices) {
        struct Graph* graph = malloc(sizeof(struct Graph));
        graph->numVertices = vertices;
        graph->adjLists = malloc(vertices * sizeof(struct Node*));

        for (int i = 0; i < vertices; i++)
            graph->adjLists[i] = NULL;

        return graph;
    }

    void addEdge(struct Graph* graph, int src, int dest) {
        struct Node* newNode = createNode(dest);
        newNode->next = graph->adjLists[src];
        graph->adjLists[src] = newNode;

        newNode = createNode(src);
        newNode->next = graph->adjLists[dest];
        graph->adjLists[dest] = newNode;
    }

    void printGraph(struct Graph* graph) {
        for (int v = 0; v < graph->numVertices; v++) {
            struct Node* temp = graph->adjLists[v];
            printf("\n Adjacency list of vertex %d\n ", v);
            while (temp) {
                printf("%d -> ", temp->vertex);
                temp = temp->next;
            }
            printf("\n");
        }
    }

    int main() {
        struct Graph* graph = createGraph(5);
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

  - 그래프의 깊이 우선 탐색(DFS) 및 너비 우선 탐색(BFS) 구현 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h

>

    struct Node {
        int vertex;
        struct Node* next;
    };

    struct Graph {
        int numVertices;
        struct Node** adjLists;
        int* visited;
    };

    struct Node* createNode(int v) {
        struct Node* newNode = malloc(sizeof(struct Node));
        newNode->vertex = v;
        newNode->next = NULL;
        return newNode;
    }

    struct Graph* createGraph(int vertices) {
        struct Graph* graph = malloc(sizeof(struct Graph));
        graph->numVertices = vertices;
        graph->adjLists = malloc(vertices * sizeof(struct Node*));
        graph->visited = malloc(vertices * sizeof(int));

        for (int i = 0; i < vertices; i++) {
            graph->adjLists[i] = NULL;
            graph->visited[i] = 0;
        }

        return graph;
    }

    void addEdge(struct Graph* graph, int src, int dest) {
        struct Node* newNode = createNode(dest);
        newNode->next = graph->adjLists[src];
        graph->adjLists[src] = newNode;

        newNode = createNode(src);
        newNode->next = graph->adjLists[dest];
        graph->adjLists[dest] = newNode;
    }

    void DFS(struct Graph* graph, int vertex) {
        struct Node* adjList = graph->adjLists[vertex];
        struct Node* temp = adjList;

        graph->visited[vertex] = 1;
        printf("Visited %d \n", vertex);

        while (temp != NULL) {
            int connectedVertex = temp->vertex;

            if (graph->visited[connectedVertex] == 0) {
                DFS(graph, connectedVertex);
            }
            temp = temp->next;
        }
    }

    void BFS(struct Graph* graph, int startVertex) {
        struct Node* adjList;
        struct Node* temp;
        int* visited = malloc(graph->numVertices * sizeof(int));

        for (int i = 0; i < graph->numVertices; i++)
            visited[i] = 0;

        int queue[graph->numVertices];
        int front = 0;
        int rear = 0;

        visited[startVertex] = 1;
        queue[rear++] = startVertex;

        while (front < rear) {
            int currentVertex = queue[front++];
            printf("Visited %d\n", currentVertex);

            adjList = graph->adjLists[currentVertex];
            while (adjList) {
                int adjVertex = adjList->vertex;

                if (visited[adjVertex] == 0) {
                    visited[adjVertex] = 1;
                    queue[rear++] = adjVertex;
                }
                adjList = adjList->next;
            }
        }

        free(visited);
    }

    int main() {
        struct Graph* graph = createGraph(6);
        addEdge(graph, 0, 1);
        addEdge(graph, 0, 2);
        addEdge(graph, 1, 2);
        addEdge(graph, 1, 4);
        addEdge(graph, 2, 3);
        addEdge(graph, 3, 4);
        addEdge(graph, 3, 5);

        printf("Depth First Search starting from vertex 0:\n");
        DFS(graph, 0);

        printf("\nBreadth First Search starting from vertex 0:\n");
        BFS(graph, 0);

        return 0;
    }
    ```

**과제:**
12주차 과제는 다음과 같습니다.
- 이진 탐색 트리를 구현하여, 사용자가 입력한 데이터를 삽입, 삭제, 탐색하는 프로그램 작성
- 그래프를 인접 리스트로 구현하고, 깊이 우선 탐색(DFS)과 너비 우선 탐색(BFS)을 수행하는 프로그램 작성
- 트리를 활용하여 표현식 트리(Expression Tree)를 구축하고, 중위 순회(Inorder Traversal)를 통해 수식을 출력하는 프로그램 작성

**퀴즈 및 해설:**

1. **트리와 이진 탐색 트리의 차이점은 무엇인가요?**
   - 트리는 계층적 데이터 구조로, 각 노드가 여러 자식을 가질 수 있습니다. 이진 탐색 트리는 이진 트리의 한 종류로, 각 노드가 최대 두 개의 자식을 가지며, 왼쪽 서브트리에는 작은 값, 오른쪽 서브트리에는 큰 값이 저장됩니다.

2. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    struct Node {
        int data;
        struct Node* left;
        struct Node* right;
    };

    void inorder(struct Node* node) {
        if (node == NULL)
            return;
        inorder(node->left);
        printf("%d ", node->data);
        inorder(node->right);
    }

    int main() {
        struct Node* root = newNode(2);
        root->left = newNode(1);
        root->right = newNode(3);

        inorder(root);

        return 0;
    }
    ```
   - 출력 결과는 `1 2 3`입니다. 중위 순회(Inorder Traversal)는 왼쪽 자식, 루트, 오른쪽 자식 순으로 노드를 방문합니다.

3. **그래프의 인접 행렬과 인접 리스트의 차이점은 무엇인가요?**
   - 인접 행렬은 그래프의 정점 간의 연결 관계를 2차원 배열로 표현하며, 메모리 사용량이 많습니다. 인접 리스트는 각 정점에 연결된 정점들의 리스트를 저장하며, 메모리를 효율적으로 사용합니다.

4. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    struct Node {
        int vertex;
        struct Node* next;
    };

    struct Graph {
        int numVertices;
        struct Node** adjLists;
    };

    void DFS(struct Graph* graph, int vertex) {
        struct Node* adjList = graph->adjLists[vertex];
        struct Node* temp = adjList;

        graph->visited[vertex] = 1;
        printf("Visited %d\n", vertex);

        while (temp != NULL) {
            int connectedVertex = temp->vertex;

            if (graph->visited[connectedVertex] == 0) {
                DFS(graph, connectedVertex);
            }
            temp = temp->next;
        }
    }

    int main() {
        struct Graph* graph = createGraph(4);
        addEdge(graph, 0, 1);
        addEdge(graph, 0, 2);
        addEdge(graph, 1, 2);
        addEdge(graph, 2, 3);

        DFS(graph, 0);

        return 0;
    }
    ```
   - 출력 결과는 `Visited 0`, `Visited 2`, `Visited 3`, `Visited 1`입니다. 깊이 우선 탐색(DFS)은 현재 정점의 모든 인접 정점을 방문한 후 다음 정점을 방문합니다.

5. **이진 탐색 트리의 장점은 무엇인가요?**
   - 이진 탐색 트리는 검색, 삽입, 삭제 연산이 평균적으로 O(log n)의 시간 복잡도를 가지며, 효율적인 데이터 관리가 가능합니다.

**해설:**
1. 트리는 계층적 데이터 구조로 각 노드가 여러 자식을 가질 수 있지만, 이진 탐색 트리는 각 노드가 최대 두 개의 자식을 가지며, 왼쪽 서브트리에는 작은 값, 오른쪽 서브트리에는 큰 값이 저장됩니다.
2. 중위 순회(Inorder Traversal)는 왼쪽 자식, 루트, 오른쪽 자식 순으로 노드를 방문하므로 출력 결과는 `1 2 3`입니다.
3. 인접 행렬은 2차원 배열로 그래프의 정점 간의 연결 관계를 표현하며 메모리 사용량이 많고, 인접 리스트는 각 정점에 연결된 정점들의 리스트를 저장하여 메모리를 효율적으로 사용합니다.
4. 깊이 우선 탐색(DFS)은 현재 정점의 모든 인접 정점을 방문한 후 다음 정점을 방문하므로 출력 결과는 `Visited 0`, `Visited 2`, `Visited 3`, `Visited 1`입니다.
5. 이진 탐색 트리는 검색, 삽입, 삭제 연산이 평균적으로 O(log n)의 시간 복잡도를 가지며, 효율적인 데이터 관리가 가능합니다.

이 12주차 강의는 학생들이 트리, 이진 탐색 트리, 그래프와 같은 고급 데이터 구조를 이해하고, 이를 활용하여 효율적인 데이터 관리를 수행하는 능력을 기를 수 있도록 도와줍니다.