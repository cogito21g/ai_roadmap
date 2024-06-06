
### 15주차 강의 계획: 고급 그래프 알고리즘 (위상 정렬, 강한 연결 요소 등)

#### 강의 목표
- 위상 정렬의 개념과 응용 학습
- 강한 연결 요소(SCC) 알고리즘 이해 및 구현
- A* 알고리즘 이해 및 구현

#### 강의 구성
- **강의 시간**: 2시간
- **구성**: 위상 정렬(30분), 강한 연결 요소(30분), A* 알고리즘(30분), 실습 및 과제 안내(30분)

### 강의 내용

##### 1. 위상 정렬 (30분)

###### 1.1 위상 정렬의 기본 개념
- **위상 정렬 개요**:
  - 방향 비순환 그래프(DAG)에서 정점들의 선형 순서를 결정
  - 선행 조건이 있는 작업의 순서를 결정할 때 사용

###### 1.2 위상 정렬 알고리즘
- **위상 정렬 알고리즘**:
  - Kahn's 알고리즘
  - DFS 기반의 위상 정렬

###### 1.3 위상 정렬 구현
- **예제(Kahn's 알고리즘)**:
```cpp
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
    }

    void topologicalSort() {
        vector<int> in_degree(V, 0);

        for (int u = 0; u < V; u++) {
            for (int v : adj[u]) {
                in_degree[v]++;
            }
        }

        queue<int> q;
        for (int i = 0; i < V; i++) {
            if (in_degree[i] == 0) {
                q.push(i);
            }
        }

        int count = 0;
        vector<int> top_order;

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            top_order.push_back(u);

            for (int v : adj[u]) {
                if (--in_degree[v] == 0) {
                    q.push(v);
                }
            }

            count++;
        }

        if (count != V) {
            cout << "There exists a cycle in the graph\n";
            return;
        }

        for (int i : top_order) {
            cout << i << " ";
        }
        cout << endl;
    }
};

int main() {
    Graph g(6);
    g.addEdge(5, 2);
    g.addEdge(5, 0);
    g.addEdge(4, 0);
    g.addEdge(4, 1);
    g.addEdge(2, 3);
    g.addEdge(3, 1);

    cout << "Topological Sort of the given graph:\n";
    g.topologicalSort();

    return 0;
}
```

##### 2. 강한 연결 요소 (30분)

###### 2.1 강한 연결 요소의 기본 개념
- **강한 연결 요소(SCC) 개요**:
  - 방향 그래프에서 서로 강하게 연결된 정점들의 최대 부분 그래프

###### 2.2 강한 연결 요소 알고리즘
- **타잔의 알고리즘(Tarjan's Algorithm)**:
  - DFS 기반으로 SCC를 찾는 알고리즘

###### 2.3 타잔의 알고리즘 구현
- **예제**:
```cpp
#include <iostream>
#include <vector>
#include <stack>
#include <list>
using namespace std;

class Graph {
    int V;
    list<int>* adj;
    void SCCUtil(int u, int disc[], int low[], stack<int>* st, bool stackMember[]);

public:
    Graph(int V);
    void addEdge(int v, int w);
    void SCC();
};

Graph::Graph(int V) {
    this->V = V;
    adj = new list<int>[V];
}

void Graph::addEdge(int v, int w) {
    adj[v].push_back(w);
}

void Graph::SCCUtil(int u, int disc[], int low[], stack<int>* st, bool stackMember[]) {
    static int time = 0;

    disc[u] = low[u] = ++time;
    st->push(u);
    stackMember[u] = true;

    for (auto v : adj[u]) {
        if (disc[v] == -1) {
            SCCUtil(v, disc, low, st, stackMember);
            low[u] = min(low[u], low[v]);
        } else if (stackMember[v]) {
            low[u] = min(low[u], disc[v]);
        }
    }

    int w = 0;
    if (low[u] == disc[u]) {
        while (st->top() != u) {
            w = st->top();
            cout << w << " ";
            stackMember[w] = false;
            st->pop();
        }
        w = st->top();
        cout << w << "\n";
        stackMember[w] = false;
        st->pop();
    }
}

void Graph::SCC() {
    int* disc = new int[V];
    int* low = new int[V];
    bool* stackMember = new bool[V];
    stack<int>* st = new stack<int>();

    for (int i = 0; i < V; i++) {
        disc[i] = -1;
        low[i] = -1;
        stackMember[i] = false;
    }

    for (int i = 0; i < V; i++) {
        if (disc[i] == -1) {
            SCCUtil(i, disc, low, st, stackMember);
        }
    }
}

int main() {
    Graph g(5);
    g.addEdge(1, 0);
    g.addEdge(0, 2);
    g.addEdge(2, 1);
    g.addEdge(0, 3);
    g.addEdge(3, 4);

    cout << "Strongly Connected Components in the given graph:\n";
    g.SCC();

    return 0;
}
```

##### 3. A* 알고리즘 (30분)

###### 3.1 A* 알고리즘의 기본 개념
- **A* 알고리즘 개요**:
  - 그래프 탐색 알고리즘으로, 경로 탐색에서 최적의 경로를 찾는 데 사용
  - 휴리스틱 함수를 사용하여 최단 경로를 효율적으로 탐색

###### 3.2 A* 알고리즘의 동작 방식
- **동작 방식**:
  - 시작 노드에서 목표 노드까지 최단 경로를 찾기 위해 우선순위 큐를 사용
  - 각 노드에서 비용 함수(f = g + h)를 계산하여 탐색

###### 3.3 A* 알고리즘 구현
- **예제**:
```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
using namespace std;

class Node {
public:
    int x, y;
    double g, h;

    Node(int x, int y, double g, double h) : x(x), y(y), g(g), h(h) {}

    double f() const {
        return g + h;
    }

    bool operator<(const Node& other) const {
        return f() > other.f();
    }
};

double heuristic(int x1, int y1, int x2, int y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

vector<pair<int, int>> AStar(vector<vector<int>>& grid, pair<int, int> start, pair<int, int> goal) {
    priority_queue<Node> openList;
    vector<vector<bool>> closedList(grid.size(), vector<bool>(grid[0].size(), false));
    vector<vector<pair<int, int>>> parent(grid.size(), vector<pair<int, int>>(grid[0].size(), {-1, -1}));

    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    openList.emplace(start.first, start.second, 0.0, heuristic(start.first, start.second, goal.first, goal.second));

    while (!openList.empty()) {
        Node current = openList.top();
        openList.pop();

        if (closedList[current.x][current.y])
            continue;

        closedList[current.x][current.y] = true;

        if (current.x == goal.first && current.y == goal.second) {
            vector<pair<int, int>> path;
            while (current.x != start.first || current.y != start.second) {
                path.emplace_back(current.x, current.y);
                auto [px, py] = parent[current.x][current.y];
                current.x = px;
                current.y = py;
            }
            path.emplace_back(start.first, start.second);
            reverse(path.begin(), path.end());
            return path;
        }

        for (int i = 0; i < 4; ++i) {
            int nx = current.x + dx[i], ny = current.y + dy[i];
            if (nx >= 0 && nx < grid.size() && ny >= 0 && ny < grid[

0].size() && grid[nx][ny] == 0) {
                double gNew = current.g + 1.0;
                double hNew = heuristic(nx, ny, goal.first, goal.second);
                openList.emplace(nx, ny, gNew, hNew);
                parent[nx][ny] = {current.x, current.y};
            }
        }
    }

    return {};
}

int main() {
    vector<vector<int>> grid = {
        {0, 0, 0, 0, 0},
        {1, 1, 1, 1, 0},
        {0, 0, 0, 0, 0},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 0}
    };
    pair<int, int> start = {0, 0};
    pair<int, int> goal = {4, 4};

    vector<pair<int, int>> path = AStar(grid, start, goal);

    if (!path.empty()) {
        cout << "Path found:\n";
        for (auto [x, y] : path) {
            cout << "(" << x << ", " << y << ")\n";
        }
    } else {
        cout << "No path found\n";
    }

    return 0;
}
```

##### 4. 실습 및 과제 안내 (30분)

###### 4.1 실습
- **실습 목표**:
  - 고급 그래프 알고리즘을 사용한 프로그램 작성
- **실습 문제**:
  - 위상 정렬을 구현하고, 주어진 방향 그래프에서 정점을 정렬하는 프로그램 작성
  - 타잔의 알고리즘을 사용하여 강한 연결 요소를 찾는 프로그램 작성
  - A* 알고리즘을 사용하여 최단 경로를 찾는 프로그램 작성

###### 4.2 과제 안내
- **과제 내용**:
  - 위상 정렬을 사용하여 작업 순서를 결정하는 프로그램 작성
  - 타잔의 알고리즘을 사용하여 주어진 그래프의 강한 연결 요소를 찾는 프로그램 작성
  - A* 알고리즘을 사용하여 주어진 격자(grid)에서 최단 경로를 찾는 프로그램 작성
- **과제 제출 방법**:
  - 작성한 코드를 주차별 과제 제출 폴더에 업로드
  - 제출 기한: 다음 주 강의 전까지

##### 5. 퀴즈 및 해설 (30분)

###### 퀴즈 문제:
1. 위상 정렬이 가능한 그래프의 조건은?
   - a) 무방향 그래프
   - b) 방향 비순환 그래프(DAG)
   - c) 가중치 그래프
   - d) 트리
2. 타잔의 알고리즘은 어떤 방법을 사용하여 강한 연결 요소를 찾는가?
   - a) BFS
   - b) DFS
   - c) 동적 계획법
   - d) 분할 정복
3. A* 알고리즘에서 휴리스틱 함수의 역할은?
   - a) 최단 경로를 계산하는 것
   - b) 탐색 공간을 줄이는 것
   - c) 그래프의 모든 노드를 방문하는 것
   - d) 그래프의 모든 경로를 평가하는 것

###### 퀴즈 해설:
1. **정답: b) 방향 비순환 그래프(DAG)**
   - 위상 정렬은 방향 비순환 그래프(DAG)에서만 가능합니다.
2. **정답: b) DFS**
   - 타잔의 알고리즘은 DFS를 사용하여 강한 연결 요소를 찾습니다.
3. **정답: b) 탐색 공간을 줄이는 것**
   - A* 알고리즘에서 휴리스틱 함수는 목표 노드까지의 예상 거리를 평가하여 탐색 공간을 줄이는 역할을 합니다.

### 준비 자료
- 강의 슬라이드
- 실습 코드 예제
- 과제 안내서

### 과제 해설

##### 과제: 위상 정렬을 사용하여 작업 순서를 결정하는 프로그램 작성
- **문제**: 주어진 방향 그래프에서 위상 정렬을 사용하여 작업 순서를 결정하는 프로그램 작성
- **해설**:
  - Kahn's 알고리즘을 사용하여 위상 정렬을 구현합니다.

```cpp
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
    }

    void topologicalSort() {
        vector<int> in_degree(V, 0);

        for (int u = 0; u < V; u++) {
            for (int v : adj[u]) {
                in_degree[v]++;
            }
        }

        queue<int> q;
        for (int i = 0; i < V; i++) {
            if (in_degree[i] == 0) {
                q.push(i);
            }
        }

        int count = 0;
        vector<int> top_order;

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            top_order.push_back(u);

            for (int v : adj[u]) {
                if (--in_degree[v] == 0) {
                    q.push(v);
                }
            }

            count++;
        }

        if (count != V) {
            cout << "There exists a cycle in the graph\n";
            return;
        }

        for (int i : top_order) {
            cout << i << " ";
        }
        cout << endl;
    }
};

int main() {
    Graph g(6);
    g.addEdge(5, 2);
    g.addEdge(5, 0);
    g.addEdge(4, 0);
    g.addEdge(4, 1);
    g.addEdge(2, 3);
    g.addEdge(3, 1);

    cout << "Topological Sort of the given graph:\n";
    g.topologicalSort();

    return 0;
}
```
- **설명**:
  - `Graph` 클래스를 사용하여 방향 그래프를 구현합니다.
  - Kahn의 알고리즘을 사용하여 위상 정렬을 수행합니다.

##### 과제: 타잔의 알고리즘을 사용하여 주어진 그래프의 강한 연결 요소를 찾는 프로그램 작성
- **문제**: 타잔의 알고리즘을 사용하여 주어진 그래프의 강한 연결 요소를 찾는 프로그램 작성
- **해설**:
  - 타잔의 알고리즘을 사용하여 강한 연결 요소를 찾습니다.

```cpp
#include <iostream>
#include <vector>
#include <stack>
#include <list>
using namespace std;

class Graph {
    int V;
    list<int>* adj;
    void SCCUtil(int u, int disc[], int low[], stack<int>* st, bool stackMember[]);

public:
    Graph(int V);
    void addEdge(int v, int w);
    void SCC();
};

Graph::Graph(int V) {
    this->V = V;
    adj = new list<int>[V];
}

void Graph::addEdge(int v, int w) {
    adj[v].push_back(w);
}

void Graph::SCCUtil(int u, int disc[], int low[], stack<int>* st, bool stackMember[]) {
    static int time = 0;

    disc[u] = low[u] = ++time;
    st->push(u);
    stackMember[u] = true;

    for (auto v : adj[u]) {
        if (disc[v] == -1) {
            SCCUtil(v, disc, low, st, stackMember);
            low[u] = min(low[u], low[v]);
        } else if (stackMember[v]) {
            low[u] = min(low[u], disc[v]);
        }
    }

    int w = 0;
    if (low[u] == disc[u]) {
        while (st->top() != u) {
            w = st->top();
            cout << w << " ";
            stackMember[w] = false;
            st->pop();
        }
        w = st->top();
        cout << w << "\n";
        stackMember[w] = false;
        st->pop();
    }
}

void Graph::SCC() {
    int* disc = new int[V];
    int* low = new int[V];
    bool* stackMember = new bool[V];
    stack<int>* st = new stack<int>();

    for (int i = 0; i < V; i++) {
        disc[i] = -1;
        low[i] = -1;
        stackMember[i] = false;
    }

    for (int i = 0; i < V; i++) {
        if (disc[i] == -1) {
            SCCUtil(i, disc, low, st, stackMember);
        }
    }
}

int main() {
    Graph g(5);
    g.addEdge(1, 0);
    g.addEdge(0, 2);
    g.addEdge(2, 1);
    g.addEdge(0, 3);
    g.addEdge(3, 4);

    cout << "Strongly Connected Components in the given graph:\n";
    g.SCC();

    return 0

;
}
```
- **설명**:
  - `Graph` 클래스를 사용하여 방향 그래프를 구현합니다.
  - 타잔의 알고리즘을 사용하여 강한 연결 요소를 찾습니다.

##### 과제: A* 알고리즘을 사용하여 주어진 격자(grid)에서 최단 경로를 찾는 프로그램 작성
- **문제**: A* 알고리즘을 사용하여 주어진 격자(grid)에서 최단 경로를 찾는 프로그램 작성
- **해설**:
  - A* 알고리즘을 사용하여 최단 경로를 찾습니다.

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
using namespace std;

class Node {
public:
    int x, y;
    double g, h;

    Node(int x, int y, double g, double h) : x(x), y(y), g(g), h(h) {}

    double f() const {
        return g + h;
    }

    bool operator<(const Node& other) const {
        return f() > other.f();
    }
};

double heuristic(int x1, int y1, int x2, int y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

vector<pair<int, int>> AStar(vector<vector<int>>& grid, pair<int, int> start, pair<int, int> goal) {
    priority_queue<Node> openList;
    vector<vector<bool>> closedList(grid.size(), vector<bool>(grid[0].size(), false));
    vector<vector<pair<int, int>>> parent(grid.size(), vector<pair<int, int>>(grid[0].size(), {-1, -1}));

    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    openList.emplace(start.first, start.second, 0.0, heuristic(start.first, start.second, goal.first, goal.second));

    while (!openList.empty()) {
        Node current = openList.top();
        openList.pop();

        if (closedList[current.x][current.y])
            continue;

        closedList[current.x][current.y] = true;

        if (current.x == goal.first && current.y == goal.second) {
            vector<pair<int, int>> path;
            while (current.x != start.first || current.y != start.second) {
                path.emplace_back(current.x, current.y);
                auto [px, py] = parent[current.x][current.y];
                current.x = px;
                current.y = py;
            }
            path.emplace_back(start.first, start.second);
            reverse(path.begin(), path.end());
            return path;
        }

        for (int i = 0; i < 4; ++i) {
            int nx = current.x + dx[i], ny = current.y + dy[i];
            if (nx >= 0 && nx < grid.size() && ny >= 0 && ny < grid[0].size() && grid[nx][ny] == 0) {
                double gNew = current.g + 1.0;
                double hNew = heuristic(nx, ny, goal.first, goal.second);
                openList.emplace(nx, ny, gNew, hNew);
                parent[nx][ny] = {current.x, current.y};
            }
        }
    }

    return {};
}

int main() {
    vector<vector<int>> grid = {
        {0, 0, 0, 0, 0},
        {1, 1, 1, 1, 0},
        {0, 0, 0, 0, 0},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 0}
    };
    pair<int, int> start = {0, 0};
    pair<int, int> goal = {4, 4};

    vector<pair<int, int>> path = AStar(grid, start, goal);

    if (!path.empty()) {
        cout << "Path found:\n";
        for (auto [x, y] : path) {
            cout << "(" << x << ", " << y << ")\n";
        }
    } else {
        cout << "No path found\n";
    }

    return 0;
}
```
- **설명**:
  - `AStar` 함수를 사용하여 주어진 격자에서 최단 경로를 찾습니다.
  - 휴리스틱 함수를 사용하여 탐색을 효율적으로 수행합니다.

이로써 15주차 강의가 마무리됩니다. 학생들은 고급 그래프 알고리즘의 기본 개념과 구현 방법을 이해하고, 이를 활용하여 다양한 프로그램을 작성하는 능력을 기르게 됩니다.