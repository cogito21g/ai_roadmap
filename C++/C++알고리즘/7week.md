### 7주차 강의 계획안

#### 강의 주제: 최단 경로 알고리즘
- 다익스트라 알고리즘
- 벨만-포드 알고리즘
- 플로이드-워셜 알고리즘

---

### 강의 내용

#### 1. 다익스트라 알고리즘
- **개념**: 가중치가 있는 그래프에서 한 정점에서 다른 모든 정점으로의 최단 경로를 찾는 알고리즘
- **제한**: 음수 가중치를 포함하지 않는 그래프에서만 작동
- **시간 복잡도**: O(V^2) (우선순위 큐를 사용하면 O((V + E) log V))

**예제**: 다익스트라 알고리즘
```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <climits>
using namespace std;

typedef pair<int, int> iPair;

void dijkstra(vector<pair<int, int>> adj[], int V, int src) {
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
        cout << i << "\t" << dist[i] << "\n";
}

int main() {
    int V = 5;
    vector<iPair> adj[V];

    adj[0].push_back(make_pair(1, 9));
    adj[0].push_back(make_pair(2, 6));
    adj[0].push_back(make_pair(3, 5));
    adj[0].push_back(make_pair(4, 3));

    adj[2].push_back(make_pair(1, 2));
    adj[2].push_back(make_pair(3, 4));

    dijkstra(adj, V, 0);

    return 0;
}
```

#### 2. 벨만-포드 알고리즘
- **개념**: 음수 가중치를 포함한 그래프에서 한 정점에서 다른 모든 정점으로의 최단 경로를 찾는 알고리즘
- **시간 복잡도**: O(VE)
- **특징**: 음수 사이클을 감지할 수 있음

**예제**: 벨만-포드 알고리즘
```cpp
#include <iostream>
#include <vector>
#include <climits>
using namespace std;

struct Edge {
    int src, dest, weight;
};

void bellmanFord(vector<Edge> &edges, int V, int E, int src) {
    vector<int> dist(V, INT_MAX);
    dist[src] = 0;

    for (int i = 1; i <= V - 1; i++) {
        for (int j = 0; j < E; j++) {
            int u = edges[j].src;
            int v = edges[j].dest;
            int weight = edges[j].weight;
            if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
                dist[v] = dist[u] + weight;
        }
    }

    for (int i = 0; i < E; i++) {
        int u = edges[i].src;
        int v = edges[i].dest;
        int weight = edges[i].weight;
        if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
            cout << "Graph contains negative weight cycle\n";
    }

    cout << "Vertex Distance from Source\n";
    for (int i = 0; i < V; ++i)
        cout << i << "\t" << dist[i] << "\n";
}

int main() {
    int V = 5, E = 8;
    vector<Edge> edges(E);

    edges[0] = {0, 1, -1};
    edges[1] = {0, 2, 4};
    edges[2] = {1, 2, 3};
    edges[3] = {1, 3, 2};
    edges[4] = {1, 4, 2};
    edges[5] = {3, 2, 5};
    edges[6] = {3, 1, 1};
    edges[7] = {4, 3, -3};

    bellmanFord(edges, V, E, 0);

    return 0;
}
```

#### 3. 플로이드-워셜 알고리즘
- **개념**: 모든 정점에서 모든 정점으로의 최단 경로를 찾는 알고리즘
- **시간 복잡도**: O(V^3)
- **특징**: 동적 계획법을 사용

**예제**: 플로이드-워셜 알고리즘
```cpp
#include <iostream>
using namespace std;

#define INF 99999
#define V 4

void printSolution(int dist[][V]);

void floydWarshall(int graph[][V]) {
    int dist[V][V], i, j, k;

    for (i = 0; i < V; i++)
        for (j = 0; j < V; j++)
            dist[i][j] = graph[i][j];

    for (k = 0; k < V; k++) {
        for (i = 0; i < V; i++) {
            for (j = 0; j < V; j++) {
                if (dist[i][j] > dist[i][k] + dist[k][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
            }
        }
    }

    printSolution(dist);
}

void printSolution(int dist[][V]) {
    cout << "The following matrix shows the shortest distances"
            " between every pair of vertices \n";
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (dist[i][j] == INF)
                cout << "INF"
                     << "     ";
            else
                cout << dist[i][j] << "     ";
        }
        cout << endl;
    }
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

---

### 과제

#### 과제 1: 다익스트라 알고리즘 구현
사용자로부터 그래프의 정점과 간선을 입력받아 인접 리스트로 표현하고, 다익스트라 알고리즘을 사용해 특정 정점에서 다른 모든 정점으로의 최단 경로를 출력하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <climits>
using namespace std;

typedef pair<int, int> iPair;

void dijkstra(vector<pair<int, int>> adj[], int V, int src) {
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
        cout << i << "\t" << dist[i] << "\n";
}

int main() {
    int V, E;
    cout << "Enter the number of vertices: ";
    cin >> V;
    vector<iPair> adj[V];

    cout << "Enter the number of edges: ";
    cin >> E;
    cout << "Enter the edges (format: v w weight):\n";
    for (int i = 0; i < E; i++) {
        int v, w, weight;
        cin >> v >> w >> weight;
        adj[v].push_back(make_pair(w, weight));
    }

    int startVertex;
    cout << "Enter the starting vertex: ";
    cin >> startVertex;

    dijkstra(adj, V, startVertex);

    return 0;
}
```

**해설**:
1. 사용자로부터 정점과 간선을 입력받아 그래프를 생성합니다.
2. `dijkstra` 함수를 사용해 시작 정점부터 다른 모든 정점으로의 최단 경로를 계산하고, 결과를 출력합니다.

#### 과제 2: 벨만-포드 알고

리즘 구현
사용자로부터 그래프의 정점과 간선을 입력받아 인접 리스트로 표현하고, 벨만-포드 알고리즘을 사용해 특정 정점에서 다른 모든 정점으로의 최단 경로를 출력하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
#include <climits>
using namespace std;

struct Edge {
    int src, dest, weight;
};

void bellmanFord(vector<Edge> &edges, int V, int E, int src) {
    vector<int> dist(V, INT_MAX);
    dist[src] = 0;

    for (int i = 1; i <= V - 1; i++) {
        for (int j = 0; j < E; j++) {
            int u = edges[j].src;
            int v = edges[j].dest;
            int weight = edges[j].weight;
            if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
                dist[v] = dist[u] + weight;
        }
    }

    for (int i = 0; i < E; i++) {
        int u = edges[i].src;
        int v = edges[i].dest;
        int weight = edges[i].weight;
        if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
            cout << "Graph contains negative weight cycle\n";
    }

    cout << "Vertex Distance from Source\n";
    for (int i = 0; i < V; ++i)
        cout << i << "\t" << dist[i] << "\n";
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
        int src, dest, weight;
        cin >> src >> dest >> weight;
        edges[i] = {src, dest, weight};
    }

    int startVertex;
    cout << "Enter the starting vertex: ";
    cin >> startVertex;

    bellmanFord(edges, V, E, startVertex);

    return 0;
}
```

**해설**:
1. 사용자로부터 정점과 간선을 입력받아 그래프를 생성합니다.
2. `bellmanFord` 함수를 사용해 시작 정점부터 다른 모든 정점으로의 최단 경로를 계산하고, 결과를 출력합니다.

---

### 퀴즈

#### 퀴즈 1: 다음 중 다익스트라 알고리즘에 대한 설명으로 옳지 않은 것은 무엇인가요?
1. 가중치가 있는 그래프에서 최단 경로를 찾는 알고리즘이다.
2. 음수 가중치를 포함하지 않는 그래프에서만 작동한다.
3. 동적 계획법을 사용하여 구현된다.
4. 시간 복잡도는 O(V^2)이다.

**정답**: 3. 동적 계획법을 사용하여 구현된다.

#### 퀴즈 2: 벨만-포드 알고리즘의 시간 복잡도는 무엇인가요?
1. O(V)
2. O(E)
3. O(VE)
4. O(V^2)

**정답**: 3. O(VE)

#### 퀴즈 3: 플로이드-워셜 알고리즘의 시간 복잡도는 무엇인가요?
1. O(V)
2. O(VE)
3. O(V^2)
4. O(V^3)

**정답**: 4. O(V^3)

이 계획안은 7주차에 필요한 최단 경로 알고리즘의 개념과 실습을 통해 학습할 수 있도록 구성되었습니다. 과제와 퀴즈를 통해 학습한 내용을 복습하고, 이해도를 높일 수 있습니다.