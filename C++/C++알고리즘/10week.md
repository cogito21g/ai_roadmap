### 10주차 강의 계획안

#### 강의 주제: 탐욕 알고리즘
- 탐욕 알고리즘의 개념
- 대표 문제 (활동 선택 문제, 최소 스패닝 트리 등)

---

### 강의 내용

#### 1. 탐욕 알고리즘의 개념
- **개념**: 매 단계에서 가장 좋다고 생각되는 것을 선택하여 최종 해답에 도달하는 알고리즘
- **특징**: 지역적으로 최적이지만, 항상 전역적으로 최적인 것은 아님
- **조건**: 탐욕 선택 속성과 최적 부분 구조를 가져야 함

#### 2. 활동 선택 문제 (Activity Selection Problem)
- **개념**: 주어진 활동들 중에서 최대한 많은 활동을 선택하는 문제
- **시간 복잡도**: O(n log n) (활동 정렬 포함)

**예제**: 활동 선택 문제
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Activity {
    int start, finish;
};

bool activityCompare(Activity s1, Activity s2) {
    return (s1.finish < s2.finish);
}

void printMaxActivities(vector<Activity>& activities) {
    sort(activities.begin(), activities.end(), activityCompare);

    cout << "Selected activities: ";

    int i = 0;
    cout << "(" << activities[i].start << ", " << activities[i].finish << "), ";

    for (int j = 1; j < activities.size(); j++) {
        if (activities[j].start >= activities[i].finish) {
            cout << "(" << activities[j].start << ", " << activities[j].finish << "), ";
            i = j;
        }
    }
    cout << endl;
}

int main() {
    vector<Activity> activities = {{5, 9}, {1, 2}, {3, 4}, {0, 6}, {5, 7}, {8, 9}};
    printMaxActivities(activities);
    return 0;
}
```

#### 3. 최소 스패닝 트리 (Minimum Spanning Tree)
- **개념**: 그래프의 모든 정점을 포함하는 트리 중에서 가중치의 합이 최소가 되는 트리
- **알고리즘**:
  - 크루스칼 알고리즘 (Kruskal's Algorithm): 간선 중심, O(E log E)
  - 프림 알고리즘 (Prim's Algorithm): 정점 중심, O(V^2) (힙 사용 시 O(E log V))

**예제**: 크루스칼 알고리즘을 이용한 최소 스패닝 트리
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Edge {
    int src, dest, weight;
};

bool edgeCompare(Edge e1, Edge e2) {
    return e1.weight < e2.weight;
}

struct DisjointSets {
    int *parent, *rank;
    int n;

    DisjointSets(int n) {
        this->n = n;
        parent = new int[n];
        rank = new int[n];

        for (int i = 0; i < n; i++) {
            parent[i] = i;
            rank[i] = 0;
        }
    }

    int find(int u) {
        if (u != parent[u])
            parent[u] = find(parent[u]);
        return parent[u];
    }

    void merge(int x, int y) {
        int xroot = find(x);
        int yroot = find(y);

        if (rank[xroot] < rank[yroot])
            parent[xroot] = yroot;
        else if (rank[xroot] > rank[yroot])
            parent[yroot] = xroot;
        else {
            parent[yroot] = xroot;
            rank[xroot]++;
        }
    }
};

void KruskalMST(vector<Edge>& edges, int V) {
    sort(edges.begin(), edges.end(), edgeCompare);

    DisjointSets ds(V);

    vector<Edge> result;
    for (Edge e : edges) {
        int u = ds.find(e.src);
        int v = ds.find(e.dest);

        if (u != v) {
            result.push_back(e);
            ds.merge(u, v);
        }
    }

    cout << "Edges in MST:" << endl;
    for (Edge e : result)
        cout << e.src << " -- " << e.dest << " == " << e.weight << endl;
}

int main() {
    int V = 4;
    vector<Edge> edges = {{0, 1, 10}, {0, 2, 6}, {0, 3, 5}, {1, 3, 15}, {2, 3, 4}};
    KruskalMST(edges, V);
    return 0;
}
```

**예제**: 프림 알고리즘을 이용한 최소 스패닝 트리
```cpp
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

#define INF 9999999

void PrimMST(vector<vector<int>>& graph, int V) {
    vector<int> parent(V, -1);
    vector<int> key(V, INF);
    vector<bool> inMST(V, false);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

    key[0] = 0;
    pq.push({0, 0});

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        inMST[u] = true;

        for (int v = 0; v < V; v++) {
            if (graph[u][v] && !inMST[v] && graph[u][v] < key[v]) {
                key[v] = graph[u][v];
                pq.push({key[v], v});
                parent[v] = u;
            }
        }
    }

    cout << "Edge   Weight\n";
    for (int i = 1; i < V; i++)
        cout << parent[i] << " - " << i << "   " << graph[i][parent[i]] << " \n";
}

int main() {
    int V = 5;
    vector<vector<int>> graph = {{0, 2, 0, 6, 0},
                                 {2, 0, 3, 8, 5},
                                 {0, 3, 0, 0, 7},
                                 {6, 8, 0, 0, 9},
                                 {0, 5, 7, 9, 0}};
    PrimMST(graph, V);
    return 0;
}
```

---

### 과제

#### 과제 1: 탐욕 알고리즘을 이용한 최소 동전 교환 문제
동전의 종류가 {1, 3, 4}일 때, 주어진 금액을 최소한의 동전으로 교환하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
using namespace std;

void findMinCoins(vector<int>& coins, int amount) {
    vector<int> result;
    int n = coins.size();

    for (int i = n - 1; i >= 0; i--) {
        while (amount >= coins[i]) {
            amount -= coins[i];
            result.push_back(coins[i]);
        }
    }

    cout << "Minimum coins required: ";
    for (int coin : result)
        cout << coin << " ";
    cout << endl;
}

int main() {
    vector<int> coins = {1, 3, 4};
    int amount;
    cout << "Enter the amount: ";
    cin >> amount;
    findMinCoins(coins, amount);
    return 0;
}
```

**해설**:
1. 탐욕 알고리즘을 사용하여 주어진 금액을 최소 동전으로 교환합니다.
2. 큰 동전부터 차례로 선택하여 금액을 줄여나갑니다.

#### 과제 2: 프림 알고리즘을 이용한 최소 스패닝 트리 구현
사용자로부터 그래프의 정점과 간선을 입력받아 인접 행렬로 표현하고, 프림 알고리즘을 사용해 최소 스패닝 트리를 출력하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

#define INF 9999999

void PrimMST(vector<vector<int>>& graph, int V) {
    vector<int> parent(V, -1);
    vector<int> key(V, INF);
    vector<bool> inMST(V, false);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

    key[0] = 0;
    pq.push({0, 0});

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        inMST[u] = true;

        for (int v = 0; v < V; v++) {
            if (graph[u][v] && !inMST[v] && graph[u][v] < key[v]) {
                key[v] = graph[u][v];
                pq.push({key[v], v});
                parent[v] = u;
            }
        }
    }

    cout << "Edge   Weight\n";
    for (int i = 1; i < V; i++)
        cout << parent[i] << " - "

 << i << "   " << graph[i][parent[i]] << " \n";
}

int main() {
    int V, E;
    cout << "Enter the number of vertices: ";
    cin >> V;
    vector<vector<int>> graph(V, vector<int>(V, 0));

    cout << "Enter the number of edges: ";
    cin >> E;
    cout << "Enter the edges (format: src dest weight):\n";
    for (int i = 0; i < E; i++) {
        int src, dest, weight;
        cin >> src >> dest >> weight;
        graph[src][dest] = weight;
        graph[dest][src] = weight; // For undirected graph
    }

    PrimMST(graph, V);
    return 0;
}
```

**해설**:
1. 사용자로부터 정점과 간선을 입력받아 그래프를 생성합니다.
2. `PrimMST` 함수를 사용해 최소 스패닝 트리를 계산하고, 결과를 출력합니다.

---

### 퀴즈

#### 퀴즈 1: 탐욕 알고리즘의 특징으로 옳지 않은 것은 무엇인가요?
1. 항상 최적의 해를 보장한다.
2. 매 단계에서 가장 좋다고 생각되는 것을 선택한다.
3. 탐욕 선택 속성과 최적 부분 구조를 가져야 한다.
4. 지역적으로 최적이지만 항상 전역적으로 최적인 것은 아니다.

**정답**: 1. 항상 최적의 해를 보장한다.

#### 퀴즈 2: 크루스칼 알고리즘의 시간 복잡도는 무엇인가요?
1. O(V^2)
2. O(E log E)
3. O(VE)
4. O(V log V)

**정답**: 2. O(E log E)

#### 퀴즈 3: 프림 알고리즘에 사용되는 자료 구조는 무엇인가요?
1. 큐
2. 스택
3. 힙
4. 트리

**정답**: 3. 힙

이 계획안은 10주차에 필요한 탐욕 알고리즘의 개념과 실습을 통해 학습할 수 있도록 구성되었습니다. 과제와 퀴즈를 통해 학습한 내용을 복습하고, 이해도를 높일 수 있습니다.