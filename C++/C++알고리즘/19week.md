### 19주차 강의 계획안

#### 강의 주제: 알고리즘 설계 기법
- 문제 분할 및 정복
- 문제 변형 및 축소
- 근사 알고리즘

---

### 강의 내용

#### 1. 문제 분할 및 정복 (Divide and Conquer)
- **개념**: 문제를 작은 하위 문제로 나누어 해결한 후, 결과를 합쳐서 전체 문제를 해결하는 방법
- **적용 분야**: 정렬 알고리즘 (퀵 정렬, 병합 정렬), 이진 탐색 등

**예제**: 병합 정렬
```cpp
#include <iostream>
#include <vector>
using namespace std;

void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }

    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void mergeSort(vector<int>& arr, int left, int right) {
    if (left >= right) return;

    int mid = left + (right - left) / 2;
    mergeSort(arr, left, mid);
    mergeSort(arr, mid + 1, right);
    merge(arr, left, mid, right);
}

int main() {
    vector<int> arr = {12, 11, 13, 5, 6, 7};
    mergeSort(arr, 0, arr.size() - 1);

    cout << "Sorted array: ";
    for (int i : arr) cout << i << " ";
    cout << endl;

    return 0;
}
```

#### 2. 문제 변형 및 축소 (Problem Transformation and Reduction)
- **개념**: 문제를 해결하기 위해 문제를 다른 문제로 변형하거나 축소하는 방법
- **적용 분야**: NP 문제, 다양한 최적화 문제 등

**예제**: 최대 이분 매칭 문제를 최대 유량 문제로 변환
```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <climits>
using namespace std;

bool bfs(vector<vector<int>>& residualGraph, int s, int t, vector<int>& parent) {
    int V = residualGraph.size();
    vector<bool> visited(V, false);
    queue<int> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v = 0; v < V; v++) {
            if (!visited[v] && residualGraph[u][v] > 0) {
                if (v == t) {
                    parent[v] = u;
                    return true;
                }
                q.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
    }
    return false;
}

int fordFulkerson(vector<vector<int>>& graph, int s, int t) {
    int u, v;
    vector<vector<int>> residualGraph = graph;
    int V = graph.size();
    vector<int> parent(V);
    int maxFlow = 0;

    while (bfs(residualGraph, s, t, parent)) {
        int pathFlow = INT_MAX;
        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            pathFlow = min(pathFlow, residualGraph[u][v]);
        }

        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            residualGraph[u][v] -= pathFlow;
            residualGraph[v][u] += pathFlow;
        }
        maxFlow += pathFlow;
    }

    return maxFlow;
}

int main() {
    vector<vector<int>> graph = {
        {0, 16, 13, 0, 0, 0},
        {0, 0, 10, 12, 0, 0},
        {0, 4, 0, 0, 14, 0},
        {0, 0, 9, 0, 0, 20},
        {0, 0, 0, 7, 0, 4},
        {0, 0, 0, 0, 0, 0}
    };

    cout << "The maximum possible flow is " << fordFulkerson(graph, 0, 5) << endl;

    return 0;
}
```

#### 3. 근사 알고리즘 (Approximation Algorithms)
- **개념**: 최적의 해를 구할 수 없는 경우, 근사 해를 구하는 알고리즘
- **적용 분야**: NP-완전 문제, 여행하는 세일즈맨 문제 등

**예제**: 외판원 문제 근사 알고리즘
```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
using namespace std;

double distance(pair<int, int> a, pair<int, int> b) {
    return sqrt(pow(a.first - b.first, 2) + pow(a.second - b.second, 2));
}

void tspNearestNeighbor(vector<pair<int, int>>& points) {
    int n = points.size();
    vector<bool> visited(n, false);
    vector<int> path;

    int current = 0;
    visited[current] = true;
    path.push_back(current);

    for (int i = 1; i < n; i++) {
        double minDist = numeric_limits<double>::max();
        int next = -1;

        for (int j = 0; j < n; j++) {
            if (!visited[j] && distance(points[current], points[j]) < minDist) {
                minDist = distance(points[current], points[j]);
                next = j;
            }
        }

        path.push_back(next);
        visited[next] = true;
        current = next;
    }

    path.push_back(0); // Return to the starting point

    cout << "Approximate TSP path: ";
    for (int i : path) cout << i << " ";
    cout << endl;
}

int main() {
    vector<pair<int, int>> points = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
    tspNearestNeighbor(points);
    return 0;
}
```

---

### 과제

#### 과제 1: 분할 정복 알고리즘 구현
병합 정렬 알고리즘을 구현하고, 주어진 배열을 정렬하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
using namespace std;

void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }

    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void mergeSort(vector<int>& arr, int left, int right) {
    if (left >= right) return;

    int mid = left + (right - left) / 2;
    mergeSort(arr, left, mid);
    mergeSort(arr, mid + 1, right);
    merge(arr, left, mid, right);
}

int main() {
    vector<int> arr = {12, 11, 13, 5, 6, 7};
    mergeSort(arr, 0, arr.size() - 1);

    cout << "Sorted array: ";
    for (int i : arr) cout << i << " ";
    cout << endl;

    return 0;
}
```

#### 과제 2: 문제 변형 및 축소 예제 구현
최대 이분 매칭 문제를 최대 유량 문제로 변환하고 해결하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <climits>
using namespace std;

bool bfs(vector<vector<int>>& residualGraph, int s, int t, vector<int>& parent) {
    int V = residualGraph.size();
    vector<bool> visited(V, false);
    queue<int> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;

   

 while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v = 0; v < V; v++) {
            if (!visited[v] && residualGraph[u][v] > 0) {
                if (v == t) {
                    parent[v] = u;
                    return true;
                }
                q.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
    }
    return false;
}

int fordFulkerson(vector<vector<int>>& graph, int s, int t) {
    int u, v;
    vector<vector<int>> residualGraph = graph;
    int V = graph.size();
    vector<int> parent(V);
    int maxFlow = 0;

    while (bfs(residualGraph, s, t, parent)) {
        int pathFlow = INT_MAX;
        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            pathFlow = min(pathFlow, residualGraph[u][v]);
        }

        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            residualGraph[u][v] -= pathFlow;
            residualGraph[v][u] += pathFlow;
        }
        maxFlow += pathFlow;
    }

    return maxFlow;
}

int main() {
    vector<vector<int>> graph = {
        {0, 16, 13, 0, 0, 0},
        {0, 0, 10, 12, 0, 0},
        {0, 4, 0, 0, 14, 0},
        {0, 0, 9, 0, 0, 20},
        {0, 0, 0, 7, 0, 4},
        {0, 0, 0, 0, 0, 0}
    };

    cout << "The maximum possible flow is " << fordFulkerson(graph, 0, 5) << endl;

    return 0;
}
```

#### 과제 3: 근사 알고리즘 구현
외판원 문제에 대한 근사 알고리즘을 구현하고, 주어진 도시 좌표를 기반으로 근사 경로를 계산하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
using namespace std;

double distance(pair<int, int> a, pair<int, int> b) {
    return sqrt(pow(a.first - b.first, 2) + pow(a.second - b.second, 2));
}

void tspNearestNeighbor(vector<pair<int, int>>& points) {
    int n = points.size();
    vector<bool> visited(n, false);
    vector<int> path;

    int current = 0;
    visited[current] = true;
    path.push_back(current);

    for (int i = 1; i < n; i++) {
        double minDist = numeric_limits<double>::max();
        int next = -1;

        for (int j = 0; j < n; j++) {
            if (!visited[j] && distance(points[current], points[j]) < minDist) {
                minDist = distance(points[current], points[j]);
                next = j;
            }
        }

        path.push_back(next);
        visited[next] = true;
        current = next;
    }

    path.push_back(0); // Return to the starting point

    cout << "Approximate TSP path: ";
    for (int i : path) cout << i << " ";
    cout << endl;
}

int main() {
    vector<pair<int, int>> points = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
    tspNearestNeighbor(points);
    return 0;
}
```

---

### 퀴즈

#### 퀴즈 1: 문제 분할 및 정복 기법의 주요 단계는 무엇인가요?
1. 정의, 나누기, 합병
2. 나누기, 정복, 합병
3. 나누기, 해결, 결합
4. 분할, 해결, 병합

**정답**: 2. 나누기, 정복, 합병

#### 퀴즈 2: 문제 변형 및 축소 기법의 주요 목적은 무엇인가요?
1. 문제를 다른 문제로 변형하거나 축소하여 해결하는 것
2. 문제를 분할하여 해결하는 것
3. 문제를 무작위로 해결하는 것
4. 문제를 반복해서 해결하는 것

**정답**: 1. 문제를 다른 문제로 변형하거나 축소하여 해결하는 것

#### 퀴즈 3: 근사 알고리즘이 주로 사용되는 문제 유형은 무엇인가요?
1. 간단한 정렬 문제
2. NP-완전 문제
3. 그래프 탐색 문제
4. 기본 수학 문제

**정답**: 2. NP-완전 문제

이 계획안은 19주차에 필요한 알고리즘 설계 기법의 개념과 실습을 통해 학습할 수 있도록 구성되었습니다. 과제와 퀴즈를 통해 학습한 내용을 복습하고, 이해도를 높일 수 있습니다.