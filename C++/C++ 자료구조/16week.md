### 16주차 강의 계획: 종합 복습 및 프로젝트 발표

#### 강의 목표
- 전체 강의 내용의 종합 복습
- 학생 프로젝트 발표 및 코드 리뷰
- 추가적인 질의응답 시간

#### 강의 구성
- **강의 시간**: 2시간
- **구성**: 종합 복습(30분), 학생 프로젝트 발표(1시간), 질의응답(30분)

### 강의 내용

##### 1. 종합 복습 (30분)

###### 1.1 전체 강의 내용 복습
- **데이터 구조**:
  - 배열과 연결 리스트
  - 스택과 큐
  - 해시 테이블
  - 트리와 이진 탐색 트리
  - 균형 트리(AVL, Red-Black Tree)
  - 힙과 힙 정렬
  - B-트리와 B+트리

- **그래프 알고리즘**:
  - 그래프의 기본 개념과 탐색 (DFS, BFS)
  - 최소 신장 트리 (크루스칼, 프림 알고리즘)
  - 최단 경로 알고리즘 (다익스트라, 벨만-포드, 플로이드-워셜)
  - 고급 그래프 알고리즘 (위상 정렬, 강한 연결 요소, A* 알고리즘)

- **문자열 알고리즘**:
  - 트라이와 접미사 트리

###### 1.2 주요 개념 및 알고리즘 정리
- **주요 개념 복습**:
  - 각 자료 구조와 알고리즘의 시간 복잡도
  - 각 알고리즘의 사용 사례 및 응용 분야

##### 2. 학생 프로젝트 발표 (1시간)

###### 2.1 프로젝트 발표 준비
- **발표 준비**:
  - 각 학생 또는 팀이 준비한 프로젝트 발표
  - 프로젝트 코드와 결과 시연
  - 구현된 알고리즘 및 자료 구조 설명

###### 2.2 프로젝트 발표 및 리뷰
- **발표 및 리뷰**:
  - 학생들의 발표 진행
  - 다른 학생들과 교수의 피드백
  - 코드의 장단점 분석 및 개선 사항 제안

##### 3. 질의응답 (30분)

###### 3.1 추가 질문 및 답변
- **질의응답 시간**:
  - 강의 내용과 관련된 추가 질문
  - 프로젝트 구현 과정에서의 어려움 공유 및 해결 방안 논의

###### 3.2 종합 리뷰 및 마무리
- **종합 리뷰**:
  - 전체 강의에 대한 종합 리뷰
  - 앞으로의 학습 방향 제시

### 준비 자료
- 강의 슬라이드
- 학생 프로젝트 발표 자료
- 종합 복습 자료

### 과제 해설

##### 종합 복습 과제
- **문제**: 종합 복습을 위한 다양한 문제 풀이 및 구현
- **해설**:
  - 주요 개념과 알고리즘에 대한 문제 해결 및 코드 작성

**예제 문제 1**: 이진 탐색 트리에서 특정 값을 삽입, 삭제, 검색하는 함수 작성
```cpp
#include <iostream>
using namespace std;

class TreeNode {
public:
    int key;
    TreeNode* left, * right;

    TreeNode(int item) {
        key = item;
        left = right = nullptr;
    }
};

class BST {
public:
    TreeNode* root;

    BST() {
        root = nullptr;
    }

    TreeNode* insert(TreeNode* node, int key) {
        if (node == nullptr) return new TreeNode(key);

        if (key < node->key)
            node->left = insert(node->left, key);
        else if (key > node->key)
            node->right = insert(node->right, key);

        return node;
    }

    TreeNode* search(TreeNode* root, int key) {
        if (root == nullptr || root->key == key)
            return root;

        if (root->key < key)
            return search(root->right, key);

        return search(root->left, key);
    }

    TreeNode* minValueNode(TreeNode* node) {
        TreeNode* current = node;
        while (current && current->left != nullptr)
            current = current->left;
        return current;
    }

    TreeNode* deleteNode(TreeNode* root, int key) {
        if (root == nullptr) return root;

        if (key < root->key)
            root->left = deleteNode(root->left, key);
        else if (key > root->key)
            root->right = deleteNode(root->right, key);
        else {
            if (root->left == nullptr) {
                TreeNode* temp = root->right;
                delete root;
                return temp;
            } else if (root->right == nullptr) {
                TreeNode* temp = root->left;
                delete root;
                return temp;
            }

            TreeNode* temp = minValueNode(root->right);
            root->key = temp->key;
            root->right = deleteNode(root->right, temp->key);
        }
        return root;
    }

    void inorder(TreeNode* root) {
        if (root != nullptr) {
            inorder(root->left);
            cout << root->key << " ";
            inorder(root->right);
        }
    }
};

int main() {
    BST bst;
    bst.root = bst.insert(bst.root, 50);
    bst.insert(bst.root, 30);
    bst.insert(bst.root, 20);
    bst.insert(bst.root, 40);
    bst.insert(bst.root, 70);
    bst.insert(bst.root, 60);
    bst.insert(bst.root, 80);

    cout << "Inorder traversal: ";
    bst.inorder(bst.root);
    cout << endl;

    cout << "Deleting 20\n";
    bst.deleteNode(bst.root, 20);
    cout << "Inorder traversal: ";
    bst.inorder(bst.root);
    cout << endl;

    cout << "Deleting 30\n";
    bst.deleteNode(bst.root, 30);
    cout << "Inorder traversal: ";
    bst.inorder(bst.root);
    cout << endl;

    cout << "Deleting 50\n";
    bst.deleteNode(bst.root, 50);
    cout << "Inorder traversal: ";
    bst.inorder(bst.root);
    cout << endl;

    TreeNode* searchResult = bst.search(bst.root, 60);
    if (searchResult != nullptr) {
        cout << "Element found: " << searchResult->key << endl;
    } else {
        cout << "Element not found" << endl;
    }

    return 0;
}
```

**예제 문제 2**: 다익스트라 알고리즘을 사용하여 그래프의 최단 경로를 찾는 함수 작성
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

이로써 16주차 강의가 마무리됩니다. 학생들은 전체 강의 내용을 복습하고, 프로젝트 발표를 통해 학습한 내용을 적용하고 공유하는 시간을 가지며, 추가적인 질문과

 답변을 통해 궁금증을 해소하게 됩니다.