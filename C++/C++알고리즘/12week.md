### 12주차 강의 계획안

#### 강의 주제: 백트래킹 알고리즘
- 백트래킹의 개념
- 대표 문제 (N-Queens 문제, 순열 생성 등)

---

### 강의 내용

#### 1. 백트래킹의 개념
- **개념**: 문제를 해결하는 과정에서 해가 될 가능성이 없는 경로를 포기하고, 가능한 해를 찾는 방법
- **특징**: 모든 가능한 경우의 수를 체계적으로 탐색
- **적용 분야**: 퍼즐, 조합론적 문제 등

#### 2. N-Queens 문제
- **개념**: N x N 크기의 체스판에 N개의 퀸을 서로 공격할 수 없도록 배치하는 문제
- **시간 복잡도**: O(N!)

**예제**: N-Queens 문제
```cpp
#include <iostream>
#include <vector>
using namespace std;

bool isSafe(vector<vector<int>>& board, int row, int col, int N) {
    for (int i = 0; i < col; i++)
        if (board[row][i])
            return false;

    for (int i = row, j = col; i >= 0 && j >= 0; i--, j--)
        if (board[i][j])
            return false;

    for (int i = row, j = col; j >= 0 && i < N; i++, j--)
        if (board[i][j])
            return false;

    return true;
}

bool solveNQUtil(vector<vector<int>>& board, int col, int N) {
    if (col >= N)
        return true;

    for (int i = 0; i < N; i++) {
        if (isSafe(board, i, col, N)) {
            board[i][col] = 1;

            if (solveNQUtil(board, col + 1, N))
                return true;

            board[i][col] = 0;
        }
    }
    return false;
}

void solveNQ(int N) {
    vector<vector<int>> board(N, vector<int>(N, 0));

    if (!solveNQUtil(board, 0, N)) {
        cout << "Solution does not exist" << endl;
        return;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            cout << board[i][j] << " ";
        cout << endl;
    }
}

int main() {
    int N = 4;
    solveNQ(N);
    return 0;
}
```

#### 3. 순열 생성
- **개념**: 주어진 집합의 모든 순열을 생성하는 문제
- **시간 복잡도**: O(n!)

**예제**: 순열 생성
```cpp
#include <iostream>
#include <vector>
using namespace std;

void permute(vector<int>& nums, int l, int r) {
    if (l == r) {
        for (int i = 0; i <= r; i++)
            cout << nums[i] << " ";
        cout << endl;
    } else {
        for (int i = l; i <= r; i++) {
            swap(nums[l], nums[i]);
            permute(nums, l + 1, r);
            swap(nums[l], nums[i]);
        }
    }
}

int main() {
    vector<int> nums = {1, 2, 3};
    permute(nums, 0, nums.size() - 1);
    return 0;
}
```

---

### 과제

#### 과제 1: 사전 순열 생성
주어진 문자열의 모든 사전 순서 순열을 백트래킹을 사용하여 생성하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

void permute(string str, int l, int r) {
    if (l == r) {
        cout << str << endl;
    } else {
        for (int i = l; i <= r; i++) {
            swap(str[l], str[i]);
            permute(str, l + 1, r);
            swap(str[l], str[i]);
        }
    }
}

int main() {
    string str;
    cout << "Enter a string: ";
    cin >> str;
    sort(str.begin(), str.end());
    permute(str, 0, str.size() - 1);
    return 0;
}
```

**해설**:
1. 사용자로부터 문자열을 입력받습니다.
2. `permute` 함수를 사용해 모든 사전 순서 순열을 생성하고 출력합니다.
3. `sort` 함수를 사용해 입력된 문자열을 사전 순서로 정렬합니다.

#### 과제 2: 해밀턴 경로 문제
주어진 그래프에서 모든 정점을 한 번씩만 방문하는 경로(해밀턴 경로)를 찾는 프로그램을 백트래킹을 사용하여 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
using namespace std;

#define V 5

bool isSafe(int v, vector<vector<int>>& graph, vector<int>& path, int pos) {
    if (graph[path[pos - 1]][v] == 0)
        return false;

    for (int i = 0; i < pos; i++)
        if (path[i] == v)
            return false;

    return true;
}

bool hamCycleUtil(vector<vector<int>>& graph, vector<int>& path, int pos) {
    if (pos == V) {
        if (graph[path[pos - 1]][path[0]] == 1)
            return true;
        else
            return false;
    }

    for (int v = 1; v < V; v++) {
        if (isSafe(v, graph, path, pos)) {
            path[pos] = v;

            if (hamCycleUtil(graph, path, pos + 1))
                return true;

            path[pos] = -1;
        }
    }

    return false;
}

bool hamCycle(vector<vector<int>>& graph) {
    vector<int> path(V, -1);

    path[0] = 0;
    if (!hamCycleUtil(graph, path, 1)) {
        cout << "Solution does not exist" << endl;
        return false;
    }

    cout << "Solution exists: ";
    for (int i = 0; i < V; i++)
        cout << path[i] << " ";
    cout << path[0] << endl;
    return true;
}

int main() {
    vector<vector<int>> graph = {{0, 1, 0, 1, 0},
                                 {1, 0, 1, 1, 1},
                                 {0, 1, 0, 0, 1},
                                 {1, 1, 0, 0, 1},
                                 {0, 1, 1, 1, 0}};

    hamCycle(graph);
    return 0;
}
```

**해설**:
1. 주어진 그래프에서 해밀턴 경로를 찾습니다.
2. `hamCycleUtil` 함수를 사용하여 백트래킹을 통해 모든 정점을 한 번씩만 방문하는 경로를 찾습니다.
3. `isSafe` 함수를 사용하여 현재 정점이 안전한지 확인합니다.

---

### 퀴즈

#### 퀴즈 1: 다음 중 백트래킹 알고리즘의 특징으로 옳지 않은 것은 무엇인가요?
1. 모든 가능한 경우의 수를 탐색한다.
2. 최적의 해를 보장한다.
3. 해가 될 가능성이 없는 경로를 포기한다.
4. 재귀적 접근 방식을 사용한다.

**정답**: 2. 최적의 해를 보장한다.

#### 퀴즈 2: N-Queens 문제에서 백트래킹의 시간 복잡도는 무엇인가요?
1. O(n)
2. O(n log n)
3. O(n^2)
4. O(n!)

**정답**: 4. O(n!)

#### 퀴즈 3: 백트래킹은 다음 중 어떤 유형의 문제에 가장 적합한가요?
1. 정렬 문제
2. 최단 경로 문제
3. 조합론적 문제
4. 검색 문제

**정답**: 3. 조합론적 문제

이 계획안은 12주차에 필요한 백트래킹 알고리즘의 개념과 실습을 통해 학습할 수 있도록 구성되었습니다. 과제와 퀴즈를 통해 학습한 내용을 복습하고, 이해도를 높일 수 있습니다.