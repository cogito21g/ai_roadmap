### 13주차 강의 계획안

#### 강의 주제: 고급 그래프 알고리즘
- 최소 스패닝 트리 (크루스칼 알고리즘, 프림 알고리즘)
- 위상 정렬 (Topological Sort)

---

### 강의 내용

#### 1. 최소 스패닝 트리 (Minimum Spanning Tree)
- **개념**: 그래프의 모든 정점을 포함하며, 간선의 가중치 합이 최소가 되는 트리
- **응용**: 네트워크 설계, 회로 설계 등

**크루스칼 알고리즘 (Kruskal's Algorithm)**
- **개념**: 간선을 하나씩 선택하여 MST를 형성하는 알고리즘
- **시간 복잡도**: O(E log E) (간선의 정렬 포함)

**예제**: 크루스칼 알고리즘
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Edge {
    int src, dest, weight;
};

bool compare(Edge a, Edge b) {
    return a.weight < b.weight;
}

class DisjointSet {
    vector<int> parent, rank;

public:
    DisjointSet(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; i++)
            parent[i] = i;
    }

    int find(int u) {
        if (u != parent[u])
            parent[u] = find(parent[u]);
        return parent[u];
    }

    void unite(int u, int v) {
        int root_u = find(u);
        int root_v = find(v);

        if (root_u != root_v) {
            if (rank[root_u] < rank[root_v])
                parent[root_u] = root_v;
            else if (rank[root_u] > rank[root_v])
                parent[root_v] = root_u;
            else {
                parent[root_v] = root_u;
                rank[root_u]++;
            }
        }
    }
};

void kruskalMST(vector<Edge>& edges, int V) {
    sort(edges.begin(), edges.end(), compare);
    DisjointSet ds(V);
    vector<Edge> result;

    for (Edge& edge : edges) {
        if (ds.find(edge.src) != ds.find(edge.dest)) {
            result.push_back(edge);
            ds.unite(edge.src, edge.dest);
        }
    }

    cout << "Edges in MST:\n";
    for (Edge& edge : result)
        cout << edge.src << " -- " << edge.dest << " == " << edge.weight << endl;
}

int main() {
    int V = 4;
    vector<Edge> edges = {{0, 1, 10}, {0, 2, 6}, {0, 3, 5}, {1, 3, 15}, {2, 3, 4}};
    kruskalMST(edges, V);
    return 0;
}
```

**프림 알고리즘 (Prim's Algorithm)**
- **개념**: 하나의 정점에서 시작하여 MST를 형성하는 알고리즘
- **시간 복잡도**: O(V^2), 힙을 사용하면 O(E log V)

**예제**: 프림 알고리즘
```cpp
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

void primMST(vector<vector<pair<int, int>>>& graph, int V) {
    vector<int> key(V, INT_MAX);
    vector<int> parent(V, -1);
    vector<bool> inMST(V, false);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

    key[0] = 0;
    pq.push({0, 0});

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        inMST[u] = true;

        for (auto& [v, weight] : graph[u]) {
            if (!inMST[v] && key[v] > weight) {
                key[v] = weight;
                pq.push({key[v], v});
                parent[v] = u;
            }
        }
    }

    cout << "Edge   Weight\n";
    for (int i = 1; i < V; i++)
        cout << parent[i] << " - " << i << "   " << key[i] << " \n";
}

int main() {
    int V = 5;
    vector<vector<pair<int, int>>> graph(V);
    graph[0].push_back({1, 2});
    graph[0].push_back({3, 6});
    graph[1].push_back({0, 2});
    graph[1].push_back({2, 3});
    graph[1].push_back({3, 8});
    graph[1].push_back({4, 5});
    graph[2].push_back({1, 3});
    graph[2].push_back({4, 7});
    graph[3].push_back({0, 6});
    graph[3].push_back({1, 8});
    graph[4].push_back({1, 5});
    graph[4].push_back({2, 7});

    primMST(graph, V);
    return 0;
}
```

#### 2. 위상 정렬 (Topological Sort)
- **개념**: 방향 그래프의 모든 정점을 선형으로 정렬하여, 모든 간선 (u, v)에 대해 u가 v보다 먼저 나오도록 하는 정렬
- **응용**: 작업 스케줄링, 강의 수강 순서 등
- **알고리즘**: Kahn's Algorithm, DFS 기반 알고리즘

**예제**: DFS를 이용한 위상 정렬
```cpp
#include <iostream>
#include <vector>
#include <stack>
using namespace std;

void topologicalSortUtil(int v, vector<bool>& visited, stack<int>& Stack, vector<vector<int>>& adj) {
    visited[v] = true;

    for (int i : adj[v])
        if (!visited[i])
            topologicalSortUtil(i, visited, Stack, adj);

    Stack.push(v);
}

void topologicalSort(vector<vector<int>>& adj, int V) {
    stack<int> Stack;
    vector<bool> visited(V, false);

    for (int i = 0; i < V; i++)
        if (!visited[i])
            topologicalSortUtil(i, visited, Stack, adj);

    while (!Stack.empty()) {
        cout << Stack.top() << " ";
        Stack.pop();
    }
    cout << endl;
}

int main() {
    int V = 6;
    vector<vector<int>> adj(V);
    adj[5].push_back(2);
    adj[5].push_back(0);
    adj[4].push_back(0);
    adj[4].push_back(1);
    adj[2].push_back(3);
    adj[3].push_back(1);

    cout << "Topological Sort of the given graph:\n";
    topologicalSort(adj, V);
    return 0;
}
```

**예제**: Kahn's Algorithm을 이용한 위상 정렬
```cpp
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

void topologicalSort(vector<vector<int>>& adj, int V) {
    vector<int> in_degree(V, 0);

    for (int u = 0; u < V; u++)
        for (int v : adj[u])
            in_degree[v]++;

    queue<int> q;
    for (int i = 0; i < V; i++)
        if (in_degree[i] == 0)
            q.push(i);

    int count = 0;
    vector<int> top_order;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        top_order.push_back(u);

        for (int v : adj[u])
            if (--in_degree[v] == 0)
                q.push(v);

        count++;
    }

    if (count != V) {
        cout << "There exists a cycle in the graph\n";
        return;
    }

    for (int i : top_order)
        cout << i << " ";
    cout << endl;
}

int main() {
    int V = 6;
    vector<vector<int>> adj(V);
    adj[5].push_back(2);
    adj[5].push_back(0);
    adj[4].push_back(0);
    adj[4].push_back(1);
    adj[2].push_back(3);
    adj[3].push_back(1);

    cout << "Topological Sort of the given graph:\n";
    topologicalSort(adj, V);
    return 0;
}
```

---

### 과제

#### 과제 1: 크루스칼 알고리즘을 이용한 최소 스패닝 트리 구현
사용자로부터 그래프의 정점과 간선을 입력받아 인접 리스트로 표현하고, 크루스칼 알고리즘을 사용해 최소 스패닝 트리를 출력하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Edge {
    int src, dest, weight;
};

bool compare(Edge a, Edge b) {
    return a.weight < b.weight;
}

class DisjointSet {
    vector<int> parent, rank;

public:
    Disjoint

Set(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; i++)
            parent[i] = i;
    }

    int find(int u) {
        if (u != parent[u])
            parent[u] = find(parent[u]);
        return parent[u];
    }

    void unite(int u, int v) {
        int root_u = find(u);
        int root_v = find(v);

        if (root_u != root_v) {
            if (rank[root_u] < rank[root_v])
                parent[root_u] = root_v;
            else if (rank[root_u] > rank[root_v])
                parent[root_v] = root_u;
            else {
                parent[root_v] = root_u;
                rank[root_u]++;
            }
        }
    }
};

void kruskalMST(vector<Edge>& edges, int V) {
    sort(edges.begin(), edges.end(), compare);
    DisjointSet ds(V);
    vector<Edge> result;

    for (Edge& edge : edges) {
        if (ds.find(edge.src) != ds.find(edge.dest)) {
            result.push_back(edge);
            ds.unite(edge.src, edge.dest);
        }
    }

    cout << "Edges in MST:\n";
    for (Edge& edge : result)
        cout << edge.src << " -- " << edge.dest << " == " << edge.weight << endl;
}

int main() {
    int V, E;
    cout << "Enter the number of vertices: ";
    cin >> V;
    cout << "Enter the number of edges: ";
    cin >> E;
    vector<Edge> edges(E);

    cout << "Enter the edges (format: src dest weight):\n";
    for (int i = 0; i < E; i++) {
        cin >> edges[i].src >> edges[i].dest >> edges[i].weight;
    }

    kruskalMST(edges, V);
    return 0;
}
```

**해설**:
1. 사용자로부터 정점과 간선을 입력받아 그래프를 생성합니다.
2. `kruskalMST` 함수를 사용해 최소 스패닝 트리를 계산하고, 결과를 출력합니다.

#### 과제 2: 위상 정렬 구현
사용자로부터 방향 그래프의 정점과 간선을 입력받아 인접 리스트로 표현하고, 위상 정렬 알고리즘을 사용해 정렬된 순서를 출력하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

void topologicalSort(vector<vector<int>>& adj, int V) {
    vector<int> in_degree(V, 0);

    for (int u = 0; u < V; u++)
        for (int v : adj[u])
            in_degree[v]++;

    queue<int> q;
    for (int i = 0; i < V; i++)
        if (in_degree[i] == 0)
            q.push(i);

    int count = 0;
    vector<int> top_order;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        top_order.push_back(u);

        for (int v : adj[u])
            if (--in_degree[v] == 0)
                q.push(v);

        count++;
    }

    if (count != V) {
        cout << "There exists a cycle in the graph\n";
        return;
    }

    for (int i : top_order)
        cout << i << " ";
    cout << endl;
}

int main() {
    int V, E;
    cout << "Enter the number of vertices: ";
    cin >> V;
    cout << "Enter the number of edges: ";
    cin >> E;
    vector<vector<int>> adj(V);

    cout << "Enter the edges (format: src dest):\n";
    for (int i = 0; i < E; i++) {
        int src, dest;
        cin >> src >> dest;
        adj[src].push_back(dest);
    }

    cout << "Topological Sort of the given graph:\n";
    topologicalSort(adj, V);
    return 0;
}
```

**해설**:
1. 사용자로부터 정점과 간선을 입력받아 그래프를 생성합니다.
2. `topologicalSort` 함수를 사용해 그래프를 위상 정렬하고, 결과를 출력합니다.

---

### 퀴즈

#### 퀴즈 1: 크루스칼 알고리즘에서 간선을 선택하는 기준은 무엇인가요?
1. 정점의 번호
2. 간선의 길이
3. 간선의 가중치
4. 정점의 차수

**정답**: 3. 간선의 가중치

#### 퀴즈 2: 프림 알고리즘에서 사용되는 자료 구조는 무엇인가요?
1. 스택
2. 큐
3. 힙
4. 트리

**정답**: 3. 힙

#### 퀴즈 3: 위상 정렬이 가능한 그래프는 어떤 그래프인가요?
1. 무방향 그래프
2. 방향 그래프
3. 순환 그래프
4. 비순환 방향 그래프

**정답**: 4. 비순환 방향 그래프

이 계획안은 13주차에 필요한 고급 그래프 알고리즘의 개념과 실습을 통해 학습할 수 있도록 구성되었습니다. 과제와 퀴즈를 통해 학습한 내용을 복습하고, 이해도를 높일 수 있습니다.