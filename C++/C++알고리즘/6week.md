### 6주차 강의 계획안

#### 강의 주제: 그래프 알고리즘 기초
- 그래프의 개념 및 표현 방법
- 깊이 우선 탐색 (DFS)
- 너비 우선 탐색 (BFS)

---

### 강의 내용

#### 1. 그래프의 개념 및 표현 방법
- **그래프**: 정점(Vertex)과 간선(Edge)으로 이루어진 자료 구조
- **종류**: 방향 그래프, 무방향 그래프, 가중치 그래프 등
- **표현 방법**:
  - 인접 행렬 (Adjacency Matrix)
  - 인접 리스트 (Adjacency List)

**예제**: 그래프의 인접 리스트 표현
```cpp
#include <iostream>
#include <vector>
using namespace std;

class Graph {
    int V;
    vector<int>* adj;

public:
    Graph(int V);
    void addEdge(int v, int w);
    void printGraph();
};

Graph::Graph(int V) {
    this->V = V;
    adj = new vector<int>[V];
}

void Graph::addEdge(int v, int w) {
    adj[v].push_back(w);
}

void Graph::printGraph() {
    for (int v = 0; v < V; ++v) {
        cout << "\n Adjacency list of vertex " << v << "\n head ";
        for (auto x : adj[v])
            cout << "-> " << x;
        printf("\n");
    }
}

int main() {
    Graph g(5);

    g.addEdge(0, 1);
    g.addEdge(0, 4);
    g.addEdge(1, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 3);
    g.addEdge(3, 4);

    g.printGraph();

    return 0;
}
```

#### 2. 깊이 우선 탐색 (DFS)
- **개념**: 그래프의 모든 정점을 깊이 있게 탐색하는 알고리즘
- **특징**: 재귀적, 스택을 이용한 구현
- **시간 복잡도**: O(V + E)

**예제**: DFS
```cpp
#include <iostream>
#include <vector>
using namespace std;

class Graph {
    int V;
    vector<int>* adj;
    void DFSUtil(int v, bool visited[]);

public:
    Graph(int V);
    void addEdge(int v, int w);
    void DFS(int v);
};

Graph::Graph(int V) {
    this->V = V;
    adj = new vector<int>[V];
}

void Graph::addEdge(int v, int w) {
    adj[v].push_back(w);
}

void Graph::DFSUtil(int v, bool visited[]) {
    visited[v] = true;
    cout << v << " ";

    for (int i : adj[v])
        if (!visited[i])
            DFSUtil(i, visited);
}

void Graph::DFS(int v) {
    bool* visited = new bool[V];
    for (int i = 0; i < V; i++)
        visited[i] = false;

    DFSUtil(v, visited);
    delete[] visited;
}

int main() {
    Graph g(4);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 2);
    g.addEdge(2, 0);
    g.addEdge(2, 3);
    g.addEdge(3, 3);

    cout << "Depth First Traversal starting from vertex 2:\n";
    g.DFS(2);

    return 0;
}
```

#### 3. 너비 우선 탐색 (BFS)
- **개념**: 그래프의 모든 정점을 너비 있게 탐색하는 알고리즘
- **특징**: 큐를 이용한 구현
- **시간 복잡도**: O(V + E)

**예제**: BFS
```cpp
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

class Graph {
    int V;
    vector<int>* adj;

public:
    Graph(int V);
    void addEdge(int v, int w);
    void BFS(int s);
};

Graph::Graph(int V) {
    this->V = V;
    adj = new vector<int>[V];
}

void Graph::addEdge(int v, int w) {
    adj[v].push_back(w);
}

void Graph::BFS(int s) {
    bool* visited = new bool[V];
    for (int i = 0; i < V; i++)
        visited[i] = false;

    queue<int> q;
    visited[s] = true;
    q.push(s);

    while (!q.empty()) {
        s = q.front();
        cout << s << " ";
        q.pop();

        for (int i : adj[s]) {
            if (!visited[i]) {
                visited[i] = true;
                q.push(i);
            }
        }
    }

    delete[] visited;
}

int main() {
    Graph g(4);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 2);
    g.addEdge(2, 0);
    g.addEdge(2, 3);
    g.addEdge(3, 3);

    cout << "Breadth First Traversal starting from vertex 2:\n";
    g.BFS(2);

    return 0;
}
```

---

### 과제

#### 과제 1: 그래프 생성 및 DFS 구현
사용자로부터 그래프의 정점과 간선을 입력받아 인접 리스트로 표현하고, DFS 알고리즘을 사용해 탐색 순서를 출력하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
using namespace std;

class Graph {
    int V;
    vector<int>* adj;
    void DFSUtil(int v, bool visited[]);

public:
    Graph(int V);
    void addEdge(int v, int w);
    void DFS(int v);
};

Graph::Graph(int V) {
    this->V = V;
    adj = new vector<int>[V];
}

void Graph::addEdge(int v, int w) {
    adj[v].push_back(w);
}

void Graph::DFSUtil(int v, bool visited[]) {
    visited[v] = true;
    cout << v << " ";

    for (int i : adj[v])
        if (!visited[i])
            DFSUtil(i, visited);
}

void Graph::DFS(int v) {
    bool* visited = new bool[V];
    for (int i = 0; i < V; i++)
        visited[i] = false;

    DFSUtil(v, visited);
    delete[] visited;
}

int main() {
    int V, E;
    cout << "Enter the number of vertices: ";
    cin >> V;
    Graph g(V);

    cout << "Enter the number of edges: ";
    cin >> E;
    cout << "Enter the edges (format: v w):\n";
    for (int i = 0; i < E; i++) {
        int v, w;
        cin >> v >> w;
        g.addEdge(v, w);
    }

    int startVertex;
    cout << "Enter the starting vertex for DFS: ";
    cin >> startVertex;

    cout << "Depth First Traversal starting from vertex " << startVertex << ":\n";
    g.DFS(startVertex);

    return 0;
}
```

**해설**:
1. 사용자로부터 정점과 간선을 입력받아 그래프를 생성합니다.
2. `DFS` 함수를 사용해 시작 정점부터 깊이 우선 탐색을 수행하고, 탐색 순서를 출력합니다.

#### 과제 2: 그래프 생성 및 BFS 구현
사용자로부터 그래프의 정점과 간선을 입력받아 인접 리스트로 표현하고, BFS 알고리즘을 사용해 탐색 순서를 출력하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

class Graph {
    int V;
    vector<int>* adj;

public:
    Graph(int V);
    void addEdge(int v, int w);
    void BFS(int s);
};

Graph::Graph(int V) {
    this->V = V;
    adj = new vector<int>[V];
}

void Graph::addEdge(int v, int w) {
    adj[v].push_back(w);
}

void Graph::BFS(int s) {
    bool* visited = new bool[V];
    for (int i = 0; i < V; i++)
        visited[i] = false;

    queue<int> q;
    visited[s] = true;
    q.push(s);

    while (!q.empty()) {
        s = q.front();
        cout << s << " ";
        q.pop();

        for (int i : adj[s]) {
            if (!visited[i]) {
                visited[i] = true;
                q.push(i);
            }
        }
    }

    delete[] visited;
}

int main() {
    int V, E;
    cout << "Enter the number of vertices: ";
    cin >> V;
    Graph g(V);

    cout << "Enter the number of edges: ";
    cin >> E;
    cout << "Enter the edges (format: v w):\n";
    for (int i = 0; i < E; i++) {
        int v

, w;
        cin >> v >> w;
        g.addEdge(v, w);
    }

    int startVertex;
    cout << "Enter the starting vertex for BFS: ";
    cin >> startVertex;

    cout << "Breadth First Traversal starting from vertex " << startVertex << ":\n";
    g.BFS(startVertex);

    return 0;
}
```

**해설**:
1. 사용자로부터 정점과 간선을 입력받아 그래프를 생성합니다.
2. `BFS` 함수를 사용해 시작 정점부터 너비 우선 탐색을 수행하고, 탐색 순서를 출력합니다.

---

### 퀴즈

#### 퀴즈 1: 다음 중 그래프의 표현 방법이 아닌 것은 무엇인가요?
1. 인접 행렬
2. 인접 리스트
3. 인접 트리
4. 인접 행렬 리스트

**정답**: 3. 인접 트리

#### 퀴즈 2: 깊이 우선 탐색(DFS)에 사용되는 자료 구조는 무엇인가요?
1. 큐
2. 스택
3. 힙
4. 트리

**정답**: 2. 스택

#### 퀴즈 3: 너비 우선 탐색(BFS)의 시간 복잡도는 무엇인가요?
1. O(V)
2. O(E)
3. O(V + E)
4. O(V * E)

**정답**: 3. O(V + E)

이 계획안은 6주차에 필요한 그래프 알고리즘 기초의 개념과 실습을 통해 학습할 수 있도록 구성되었습니다. 과제와 퀴즈를 통해 학습한 내용을 복습하고, 이해도를 높일 수 있습니다.