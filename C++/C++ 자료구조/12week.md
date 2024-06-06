### 12주차 강의 계획: 그래프 (Graph)

#### 강의 목표
- 그래프의 기본 개념과 표현 방식 이해
- 그래프의 탐색 알고리즘 (DFS, BFS) 학습
- 최소 신장 트리(MST)와 최단 경로 알고리즘 학습

#### 강의 구성
- **강의 시간**: 2시간
- **구성**: 그래프 이론 (30분), 그래프 탐색 알고리즘 (30분), MST 및 최단 경로 알고리즘 (30분), 실습 및 과제 안내 (30분)

#### 강의 내용

##### 1. 그래프 이론 (30분)

###### 1.1 그래프의 기본 개념
- **그래프 개요**:
  - 정점(Vertex)와 간선(Edge)
  - 방향 그래프(Directed Graph)와 무방향 그래프(Undirected Graph)
  - 가중치 그래프(Weighted Graph)

###### 1.2 그래프의 표현 방식
- **그래프 표현 방식**:
  - 인접 행렬(Adjacency Matrix)
  - 인접 리스트(Adjacency List)

###### 1.3 그래프의 활용 사례
- **활용 사례**:
  - 소셜 네트워크
  - 경로 탐색 (네비게이션)
  - 전자 회로 설계

##### 2. 그래프 탐색 알고리즘 (30분)

###### 2.1 깊이 우선 탐색(DFS)
- **DFS 개요**:
  - 재귀와 스택을 사용한 깊이 우선 탐색
- **DFS 구현**:
```cpp
#include <iostream>
#include <list>
using namespace std;

class Graph {
    int V;
    list<int>* adj;

    void DFSUtil(int v, bool visited[]);

public:
    Graph(int V);
    void addEdge(int v, int w);
    void DFS(int v);
};

Graph::Graph(int V) {
    this->V = V;
    adj = new list<int>[V];
}

void Graph::addEdge(int v, int w) {
    adj[v].push_back(w);
}

void Graph::DFSUtil(int v, bool visited[]) {
    visited[v] = true;
    cout << v << " ";

    for (auto i = adj[v].begin(); i != adj[v].end(); ++i)
        if (!visited[*i])
            DFSUtil(*i, visited);
}

void Graph::DFS(int v) {
    bool* visited = new bool[V];
    for (int i = 0; i < V; i++)
        visited[i] = false;

    DFSUtil(v, visited);
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

###### 2.2 너비 우선 탐색(BFS)
- **BFS 개요**:
  - 큐를 사용한 너비 우선 탐색
- **BFS 구현**:
```cpp
#include <iostream>
#include <list>
#include <queue>
using namespace std;

class Graph {
    int V;
    list<int>* adj;

public:
    Graph(int V);
    void addEdge(int v, int w);
    void BFS(int s);
};

Graph::Graph(int V) {
    this->V = V;
    adj = new list<int>[V];
}

void Graph::addEdge(int v, int w) {
    adj[v].push_back(w);
}

void Graph::BFS(int s) {
    bool* visited = new bool[V];
    for (int i = 0; i < V; i++)
        visited[i] = false;

    list<int> queue;

    visited[s] = true;
    queue.push_back(s);

    while (!queue.empty()) {
        s = queue.front();
        cout << s << " ";
        queue.pop_front();

        for (auto i = adj[s].begin(); i != adj[s].end(); ++i) {
            if (!visited[*i]) {
                visited[*i] = true;
                queue.push_back(*i);
            }
        }
    }
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

##### 3. MST 및 최단 경로 알고리즘 (30분)

###### 3.1 최소 신장 트리(MST)
- **MST 개요**:
  - 크루스칼 알고리즘(Kruskal's Algorithm)
  - 프림 알고리즘(Prim's Algorithm)

###### 3.2 크루스칼 알고리즘 구현
- **크루스칼 알고리즘**:
  - 간선의 가중치 오름차순 정렬 후 사이클이 생기지 않도록 선택
- **예제**:
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Edge {
public:
    int src, dest, weight;
    Edge(int src, int dest, int weight) {
        this->src = src;
        this->dest = dest;
        this->weight = weight;
    }
};

class Graph {
    int V, E;
    vector<Edge> edges;

public:
    Graph(int V, int E) {
        this->V = V;
        this->E = E;
    }

    void addEdge(int src, int dest, int weight) {
        edges.push_back(Edge(src, dest, weight));
    }

    int find(int parent[], int i) {
        if (parent[i] == i)
            return i;
        return find(parent, parent[i]);
    }

    void Union(int parent[], int rank[], int x, int y) {
        int xroot = find(parent, x);
        int yroot = find(parent, y);

        if (rank[xroot] < rank[yroot])
            parent[xroot] = yroot;
        else if (rank[xroot] > rank[yroot])
            parent[yroot] = xroot;
        else {
            parent[yroot] = xroot;
            rank[xroot]++;
        }
    }

    void KruskalMST() {
        vector<Edge> result;
        int e = 0;
        int i = 0;

        sort(edges.begin(), edges.end(), [](Edge a, Edge b) {
            return a.weight < b.weight;
        });

        int* parent = new int[V];
        int* rank = new int[V];

        for (int v = 0; v < V; v++) {
            parent[v] = v;
            rank[v] = 0;
        }

        while (e < V - 1 && i < E) {
            Edge next_edge = edges[i++];

            int x = find(parent, next_edge.src);
            int y = find(parent, next_edge.dest);

            if (x != y) {
                result.push_back(next_edge);
                e++;
                Union(parent, rank, x, y);
            }
        }

        cout << "Edges in the constructed MST\n";
        for (auto& edge : result)
            cout << edge.src << " -- " << edge.dest << " == " << edge.weight << endl;

        delete[] parent;
        delete[] rank;
    }
};

int main() {
    int V = 4, E = 5;
    Graph g(V, E);

    g.addEdge(0, 1, 10);
    g.addEdge(0, 2, 6);
    g.addEdge(0, 3, 5);
    g.addEdge(1, 3, 15);
    g.addEdge(2, 3, 4);

    g.KruskalMST();

    return 0;
}
```

###### 3.3 다익스트라 알고리즘
- **다익스트라 알고리즘**:
  - 단일 출발점 최단 경로 알고리즘
- **예제**:
```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <limits.h>
using namespace std;

typedef pair<int, int> iPair;

class Graph {
    int V;
    vector<vector<iPair>> adj;

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v, int w) {
        adj[u].push_back(make_pair(v, w));
        adj[v].push_back(make_pair(u, w));
    }

    void dijkstra(int src) {
        priority_queue<iPair, vector<iPair>, greater<iPair>> pq;
        vector<int> dist(V, INT_MAX);

        pq.push(make_pair(0, src));
        dist[src] = 0;

        while (!pq.empty()) {
            int u = pq.top().second;
            pq.pop();

            for (auto x : adj[u]) {
                int v = x.first;
                int weight = x.second;

                if

 (dist[v] > dist[u] + weight) {
                    dist[v] = dist[u] + weight;
                    pq.push(make_pair(dist[v], v));
                }
            }
        }

        cout << "Vertex Distance from Source\n";
        for (int i = 0; i < V; ++i)
            cout << i << "\t\t" << dist[i] << endl;
    }
};

int main() {
    int V = 9;
    Graph g(V);

    g.addEdge(0, 1, 4);
    g.addEdge(0, 7, 8);
    g.addEdge(1, 2, 8);
    g.addEdge(1, 7, 11);
    g.addEdge(2, 3, 7);
    g.addEdge(2, 8, 2);
    g.addEdge(2, 5, 4);
    g.addEdge(3, 4, 9);
    g.addEdge(3, 5, 14);
    g.addEdge(4, 5, 10);
    g.addEdge(5, 6, 2);
    g.addEdge(6, 7, 1);
    g.addEdge(6, 8, 6);
    g.addEdge(7, 8, 7);

    g.dijkstra(0);

    return 0;
}
```

##### 4. 실습 및 과제 안내 (30분)

###### 4.1 실습
- **실습 목표**:
  - 그래프를 사용한 프로그램 작성
- **실습 문제**:
  - DFS와 BFS를 구현하고, 주어진 그래프에서 탐색을 수행하는 프로그램 작성
  - 크루스칼 알고리즘을 사용하여 최소 신장 트리를 구현하는 프로그램 작성
  - 다익스트라 알고리즘을 사용하여 최단 경로를 찾는 프로그램 작성

###### 4.2 과제 안내
- **과제 내용**:
  - DFS와 BFS를 사용하여 그래프 탐색을 수행하는 프로그램 작성
  - 크루스칼 알고리즘을 사용하여 최소 신장 트리를 찾는 프로그램 작성
  - 다익스트라 알고리즘을 사용하여 단일 출발점에서 모든 정점까지의 최단 경로를 찾는 프로그램 작성
- **과제 제출 방법**:
  - 작성한 코드를 주차별 과제 제출 폴더에 업로드
  - 제출 기한: 다음 주 강의 전까지

##### 5. 퀴즈 및 해설 (30분)

###### 퀴즈 문제:
1. 그래프의 표현 방식 중 인접 리스트의 장점은?
   - a) 메모리 사용량이 적음
   - b) 구현이 간단함
   - c) 정점 간의 경로를 빠르게 찾을 수 있음
   - d) 모든 정점 간의 거리를 빠르게 계산할 수 있음
2. BFS에서 사용하는 자료구조는?
   - a) 스택
   - b) 큐
   - c) 우선순위 큐
   - d) 배열
3. 크루스칼 알고리즘에서 사용하는 자료구조는?
   - a) 스택
   - b) 큐
   - c) 유니온-파인드
   - d) 트리

###### 퀴즈 해설:
1. **정답: a) 메모리 사용량이 적음**
   - 인접 리스트는 간선 수가 적은 그래프에서 메모리 사용량이 적다는 장점이 있습니다.
2. **정답: b) 큐**
   - BFS는 너비 우선 탐색으로 큐 자료구조를 사용합니다.
3. **정답: c) 유니온-파인드**
   - 크루스칼 알고리즘에서 사이클을 감지하기 위해 유니온-파인드 자료구조를 사용합니다.

### 준비 자료
- 강의 슬라이드
- 실습 코드 예제
- 과제 안내서

### 과제 해설

##### 과제: DFS와 BFS를 사용하여 그래프 탐색을 수행하는 프로그램 작성
- **문제**: 주어진 그래프에서 DFS와 BFS를 사용하여 탐색을 수행하는 프로그램 작성
- **해설**:
  - DFS와 BFS를 사용하여 그래프를 탐색하는 프로그램을 작성합니다.

```cpp
#include <iostream>
#include <list>
#include <stack>
#include <queue>
using namespace std;

class Graph {
    int V;
    list<int>* adj;

public:
    Graph(int V);
    void addEdge(int v, int w);
    void DFS(int v);
    void BFS(int v);
};

Graph::Graph(int V) {
    this->V = V;
    adj = new list<int>[V];
}

void Graph::addEdge(int v, int w) {
    adj[v].push_back(w);
}

void Graph::DFS(int v) {
    vector<bool> visited(V, false);
    stack<int> stack;

    stack.push(v);

    while (!stack.empty()) {
        v = stack.top();
        stack.pop();

        if (!visited[v]) {
            cout << v << " ";
            visited[v] = true;
        }

        for (auto i = adj[v].rbegin(); i != adj[v].rend(); ++i)
            if (!visited[*i])
                stack.push(*i);
    }
}

void Graph::BFS(int v) {
    vector<bool> visited(V, false);
    queue<int> queue;

    visited[v] = true;
    queue.push(v);

    while (!queue.empty()) {
        v = queue.front();
        cout << v << " ";
        queue.pop();

        for (auto i = adj[v].begin(); i != adj[v].end(); ++i) {
            if (!visited[*i]) {
                visited[*i] = true;
                queue.push(*i);
            }
        }
    }
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
    cout << endl;

    cout << "Breadth First Traversal starting from vertex 2:\n";
    g.BFS(2);
    cout << endl;

    return 0;
}
```
- **설명**:
  - `Graph` 클래스를 사용하여 그래프를 구현합니다.
  - DFS와 BFS를 사용하여 그래프를 탐색합니다.

##### 과제: 크루스칼 알고리즘을 사용하여 최소 신장 트리를 찾는 프로그램 작성
- **문제**: 주어진 그래프에서 크루스칼 알고리즘을 사용하여 최소 신장 트리를 찾는 프로그램 작성
- **해설**:
  - 크루스칼 알고리즘을 사용하여 최소 신장 트리를 찾습니다.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Edge {
public:
    int src, dest, weight;
    Edge(int src, int dest, int weight) {
        this->src = src;
        this->dest = dest;
        this->weight = weight;
    }
};

class Graph {
    int V, E;
    vector<Edge> edges;

public:
    Graph(int V, int E) {
        this->V = V;
        this->E = E;
    }

    void addEdge(int src, int dest, int weight) {
        edges.push_back(Edge(src, dest, weight));
    }

    int find(int parent[], int i) {
        if (parent[i] == i)
            return i;
        return find(parent, parent[i]);
    }

    void Union(int parent[], int rank[], int x, int y) {
        int xroot = find(parent, x);
        int yroot = find(parent, y);

        if (rank[xroot] < rank[yroot])
            parent[xroot] = yroot;
        else if (rank[xroot] > rank[yroot])
            parent[yroot] = xroot;
        else {
            parent[yroot] = xroot;
            rank[xroot]++;
        }
    }

    void KruskalMST() {
        vector<Edge> result;
        int e = 0;
        int i = 0;

        sort(edges.begin(), edges.end(), [](Edge a, Edge b) {
            return a.weight < b.weight;
        });

        int* parent = new int[V];
        int* rank = new int[V];

        for (int v = 0; v < V; v++) {
            parent[v] = v;
            rank[v] = 0;
        }

        while (e < V - 1 && i < E) {
            Edge next_edge = edges[i++];

            int x = find(parent, next_edge.src);
            int y = find(parent, next_edge.dest);

            if (x != y) {
                result.push_back(next_edge);
                e++;
                Union(parent, rank, x, y);
            }
        }

        cout << "Edges in the constructed MST\n";
        for (auto& edge : result)
            cout << edge.src << " -- " << edge.dest << " == " << edge.weight << endl;

        delete[] parent;
        delete[] rank;
    }
};

int main

() {
    int V = 4, E = 5;
    Graph g(V, E);

    g.addEdge(0, 1, 10);
    g.addEdge(0, 2, 6);
    g.addEdge(0, 3, 5);
    g.addEdge(1, 3, 15);
    g.addEdge(2, 3, 4);

    g.KruskalMST();

    return 0;
}
```
- **설명**:
  - `Graph` 클래스를 사용하여 그래프를 구현합니다.
  - 크루스칼 알고리즘을 사용하여 최소 신장 트리를 찾습니다.

##### 과제: 다익스트라 알고리즘을 사용하여 단일 출발점에서 모든 정점까지의 최단 경로를 찾는 프로그램 작성
- **문제**: 주어진 그래프에서 다익스트라 알고리즘을 사용하여 단일 출발점에서 모든 정점까지의 최단 경로를 찾는 프로그램 작성
- **해설**:
  - 다익스트라 알고리즘을 사용하여 최단 경로를 찾습니다.

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <limits.h>
using namespace std;

typedef pair<int, int> iPair;

class Graph {
    int V;
    vector<vector<iPair>> adj;

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v, int w) {
        adj[u].push_back(make_pair(v, w));
        adj[v].push_back(make_pair(u, w));
    }

    void dijkstra(int src) {
        priority_queue<iPair, vector<iPair>, greater<iPair>> pq;
        vector<int> dist(V, INT_MAX);

        pq.push(make_pair(0, src));
        dist[src] = 0;

        while (!pq.empty()) {
            int u = pq.top().second;
            pq.pop();

            for (auto x : adj[u]) {
                int v = x.first;
                int weight = x.second;

                if (dist[v] > dist[u] + weight) {
                    dist[v] = dist[u] + weight;
                    pq.push(make_pair(dist[v], v));
                }
            }
        }

        cout << "Vertex Distance from Source\n";
        for (int i = 0; i < V; ++i)
            cout << i << "\t\t" << dist[i] << endl;
    }
};

int main() {
    int V = 9;
    Graph g(V);

    g.addEdge(0, 1, 4);
    g.addEdge(0, 7, 8);
    g.addEdge(1, 2, 8);
    g.addEdge(1, 7, 11);
    g.addEdge(2, 3, 7);
    g.addEdge(2, 8, 2);
    g.addEdge(2, 5, 4);
    g.addEdge(3, 4, 9);
    g.addEdge(3, 5, 14);
    g.addEdge(4, 5, 10);
    g.addEdge(5, 6, 2);
    g.addEdge(6, 7, 1);
    g.addEdge(6, 8, 6);
    g.addEdge(7, 8, 7);

    g.dijkstra(0);

    return 0;
}
```
- **설명**:
  - `Graph` 클래스를 사용하여 그래프를 구현합니다.
  - 다익스트라 알고리즘을 사용하여 단일 출발점에서 모든 정점까지의 최단 경로를 찾습니다.

이로써 12주차 강의가 마무리됩니다. 학생들은 그래프의 기본 개념과 탐색 알고리즘, 최소 신장 트리 및 최단 경로 알고리즘의 구현 방법을 이해하고, 이를 활용하여 다양한 프로그램을 작성하는 능력을 기르게 됩니다.