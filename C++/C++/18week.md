### 18주차: 고급 알고리즘과 자료구조

#### 강의 목표
- 트리와 그래프의 개념 이해 및 구현
- 정렬 알고리즘 심화 (퀵 정렬, 병합 정렬 등)
- 동적 프로그래밍의 이해 및 사용

#### 강의 내용

##### 1. 트리와 그래프
- **이진 트리 구현**

```cpp
#include <iostream>
using namespace std;

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

void inorder(TreeNode* root) {
    if (root != nullptr) {
        inorder(root->left);
        cout << root->val << " ";
        inorder(root->right);
    }
}

int main() {
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);

    cout << "Inorder traversal: ";
    inorder(root);
    cout << endl;

    // 메모리 해제
    delete root->left->left;
    delete root->left->right;
    delete root->left;
    delete root->right;
    delete root;

    return 0;
}
```

- **그래프 구현 (인접 리스트 사용)**

```cpp
#include <iostream>
#include <vector>
using namespace std;

class Graph {
private:
    int V;  // 정점의 개수
    vector<vector<int>> adjList;  // 인접 리스트

public:
    Graph(int V) : V(V) {
        adjList.resize(V);
    }

    void addEdge(int v, int w) {
        adjList[v].push_back(w);
    }

    void printGraph() {
        for (int v = 0; v < V; ++v) {
            cout << "Vertex " << v << ":";
            for (int w : adjList[v]) {
                cout << " " << w;
            }
            cout << endl;
        }
    }
};

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

##### 2. 정렬 알고리즘 심화
- **퀵 정렬 구현**

```cpp
#include <iostream>
#include <vector>
using namespace std;

int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; ++j) {
        if (arr[j] < pivot) {
            ++i;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main() {
    vector<int> arr = {10, 7, 8, 9, 1, 5};
    int n = arr.size();

    quickSort(arr, 0, n - 1);

    cout << "Sorted array: ";
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;

    return 0;
}
```

- **병합 정렬 구현**

```cpp
#include <iostream>
#include <vector>
using namespace std;

void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> L(n1);
    vector<int> R(n2);

    for (int i = 0; i < n1; ++i) {
        L[i] = arr[left + i];
    }
    for (int i = 0; i < n2; ++i) {
        R[i] = arr[mid + 1 + i];
    }

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            ++i;
        } else {
            arr[k] = R[j];
            ++j;
        }
        ++k;
    }

    while (i < n1) {
        arr[k] = L[i];
        ++i;
        ++k;
    }

    while (j < n2) {
        arr[k] = R[j];
        ++j;
        ++k;
    }
}

void mergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

int main() {
    vector<int> arr = {12, 11, 13, 5, 6, 7};
    int arr_size = arr.size();

    mergeSort(arr, 0, arr_size - 1);

    cout << "Sorted array: ";
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;

    return 0;
}
```

##### 3. 동적 프로그래밍
- **피보나치 수열 (동적 프로그래밍)**

```cpp
#include <iostream>
#include <vector>
using namespace std;

int fibonacci(int n) {
    if (n <= 1) return n;

    vector<int> fib(n + 1);
    fib[0] = 0;
    fib[1] = 1;

    for (int i = 2; i <= n; ++i) {
        fib[i] = fib[i - 1] + fib[i - 2];
    }

    return fib[n];
}

int main() {
    int n = 10;
    cout << "Fibonacci number " << n << " is " << fibonacci(n) << endl;
    return 0;
}
```

#### 과제

1. **이진 트리 구현 및 순회**
   - 이진 트리를 구현하고, 전위(preorder), 중위(inorder), 후위(postorder) 순회를 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

void preorder(TreeNode* root) {
    if (root != nullptr) {
        cout << root->val << " ";
        preorder(root->left);
        preorder(root->right);
    }
}

void inorder(TreeNode* root) {
    if (root != nullptr) {
        inorder(root->left);
        cout << root->val << " ";
        inorder(root->right);
    }
}

void postorder(TreeNode* root) {
    if (root != nullptr) {
        postorder(root->left);
        postorder(root->right);
        cout << root->val << " ";
    }
}

int main() {
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);

    cout << "Preorder traversal: ";
    preorder(root);
    cout << endl;

    cout << "Inorder traversal: ";
    inorder(root);
    cout << endl;

    cout << "Postorder traversal: ";
    postorder(root);
    cout << endl;

    // 메모리 해제
    delete root->left->left;
    delete root->left->right;
    delete root->left;
    delete root->right;
    delete root;

    return 0;
}
```

2. **그래프 구현 및 탐색**
   - 인접 리스트를 사용하여 그래프를 구현하고, 깊이 우선 탐색(DFS)와 너비 우선 탐색(BFS)을 구현하여 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <vector>
#include <stack>
#include <queue>
using namespace std;

class Graph {
private:
    int V;
    vector<vector<int>> adjList;

    void DFSUtil(int v, vector<bool>& visited) {
        visited[v] = true;
        cout << v << " ";

        for (int i : adjList[v]) {
            if (!visited[i]) {
                DFSUtil(i, visited);
            }
        }
    }

public:
    Graph(int V) : V(V) {
        adjList.resize(V);
    }

    void addEdge(int v, int w) {
        adjList[v].push_back(w);
    }

    void DFS(int v) {
        vector<bool> visited(V, false);
        DFSUtil(v, visited);
   

 }

    void BFS(int s) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[s] = true;
        q.push(s);

        while (!q.empty()) {
            int v = q.front();
            cout << v << " ";
            q.pop();

            for (int i : adjList[v]) {
                if (!visited[i]) {
                    visited[i] = true;
                    q.push(i);
                }
            }
        }
    }
};

int main() {
    Graph g(5);

    g.addEdge(0, 1);
    g.addEdge(0, 4);
    g.addEdge(1, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 3);
    g.addEdge(3, 4);

    cout << "DFS starting from vertex 0: ";
    g.DFS(0);
    cout << endl;

    cout << "BFS starting from vertex 0: ";
    g.BFS(0);
    cout << endl;

    return 0;
}
```

3. **퀵 정렬 및 병합 정렬 구현**
   - 정수 배열을 입력받아 퀵 정렬과 병합 정렬을 사용하여 정렬하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <vector>
using namespace std;

int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; ++j) {
        if (arr[j] < pivot) {
            ++i;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> L(n1);
    vector<int> R(n2);

    for (int i = 0; i < n1; ++i) {
        L[i] = arr[left + i];
    }
    for (int i = 0; i < n2; ++i) {
        R[i] = arr[mid + 1 + i];
    }

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            ++i;
        } else {
            arr[k] = R[j];
            ++j;
        }
        ++k;
    }

    while (i < n1) {
        arr[k] = L[i];
        ++i;
        ++k;
    }

    while (j < n2) {
        arr[k] = R[j];
        ++j;
        ++k;
    }
}

void mergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

int main() {
    vector<int> arr1 = {10, 7, 8, 9, 1, 5};
    vector<int> arr2 = {12, 11, 13, 5, 6, 7};

    quickSort(arr1, 0, arr1.size() - 1);
    mergeSort(arr2, 0, arr2.size() - 1);

    cout << "QuickSorted array: ";
    for (int x : arr1) {
        cout << x << " ";
    }
    cout << endl;

    cout << "MergeSorted array: ";
    for (int x : arr2) {
        cout << x << " ";
    }
    cout << endl;

    return 0;
}
```

4. **동적 프로그래밍 (피보나치 수열)**
   - 피보나치 수열을 동적 프로그래밍을 사용하여 계산하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <vector>
using namespace std;

int fibonacci(int n) {
    if (n <= 1) return n;

    vector<int> fib(n + 1);
    fib[0] = 0;
    fib[1] = 1;

    for (int i = 2; i <= n; ++i) {
        fib[i] = fib[i - 1] + fib[i - 2];
    }

    return fib[n];
}

int main() {
    int n = 10;
    cout << "Fibonacci number " << n << " is " << fibonacci(n) << endl;
    return 0;
}
```

#### 퀴즈

1. **트리에 대한 설명 중 맞는 것은?**
   - A) 이진 트리는 최대 두 개의 자식을 가질 수 있다.
   - B) 트리는 항상 순환 구조를 가진다.
   - C) 트리는 루트 노드가 없다.
   - D) 트리는 정렬된 데이터 구조이다.

2. **그래프에 대한 설명 중 맞는 것은?**
   - A) 그래프는 항상 연결되어 있어야 한다.
   - B) 그래프는 정점과 간선으로 구성된다.
   - C) 그래프는 순환 구조를 가질 수 없다.
   - D) 그래프는 트리와 동일한 데이터 구조이다.

3. **퀵 정렬에 대한 설명 중 맞는 것은?**
   - A) 퀵 정렬은 항상 O(n^2)의 시간 복잡도를 가진다.
   - B) 퀵 정렬은 불안정 정렬 알고리즘이다.
   - C) 퀵 정렬은 항상 안정 정렬 알고리즘이다.
   - D) 퀵 정렬은 데이터의 최악의 경우에도 빠르게 동작한다.

4. **동적 프로그래밍에 대한 설명 중 맞는 것은?**
   - A) 동적 프로그래밍은 항상 재귀적으로 구현된다.
   - B) 동적 프로그래밍은 동일한 문제를 여러 번 해결한다.
   - C) 동적 프로그래밍은 문제를 부분 문제로 나누어 해결한다.
   - D) 동적 프로그래밍은 항상 그리디 알고리즘보다 효율적이다.

#### 퀴즈 해설

1. **트리에 대한 설명 중 맞는 것은?**
   - **정답: A) 이진 트리는 최대 두 개의 자식을 가질 수 있다.**
     - 해설: 이진 트리는 각 노드가 최대 두 개의 자식을 가지는 트리 구조입니다.

2. **그래프에 대한 설명 중 맞는 것은?**
   - **정답: B) 그래프는 정점과 간선으로 구성된다.**
     - 해설: 그래프는 정점과 정점을 연결하는 간선으로 구성된 데이터 구조입니다.

3. **퀵 정렬에 대한 설명 중 맞는 것은?**
   - **정답: B) 퀵 정렬은 불안정 정렬 알고리즘이다.**
     - 해설: 퀵 정렬은 동일한 값의 상대적 순서를 보장하지 않기 때문에 불안정 정렬 알고리즘입니다.

4. **동적 프로그래밍에 대한 설명 중 맞는 것은?**
   - **정답: C) 동적 프로그래밍은 문제를 부분 문제로 나누어 해결한다.**
     - 해설: 동적 프로그래밍은 문제를 작은 부분 문제로 나누어 해결하고, 이미 해결한 부분 문제의 결과를 재사용하여 효율성을 높입니다.

다음 주차 강의 내용을 요청하시면, 19주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.