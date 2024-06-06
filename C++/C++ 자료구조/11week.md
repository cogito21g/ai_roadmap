### 11주차 강의 계획: 고급 트리 구조

#### 강의 목표
- 고급 트리 구조 (AVL 트리, 힙, B-트리 등) 이해
- AVL 트리의 균형 유지 및 회전 연산 이해
- 힙의 구조와 힙 정렬 구현
- B-트리의 개념과 데이터베이스에서의 활용 이해

#### 강의 구성
- **강의 시간**: 2시간
- **구성**: AVL 트리 (30분), 힙 및 힙 정렬 (30분), B-트리 (30분), 실습 및 과제 안내 (30분)

#### 강의 내용

##### 1. AVL 트리 (30분)

###### 1.1 AVL 트리의 기본 개념
- **AVL 트리 개요**:
  - 자가 균형 이진 탐색 트리
  - 각 노드의 왼쪽과 오른쪽 서브트리 높이 차이가 최대 1

###### 1.2 AVL 트리의 회전 연산
- **회전 연산**:
  - LL 회전, RR 회전, LR 회전, RL 회전
  - 균형을 맞추기 위한 회전 연산의 필요성

###### 1.3 AVL 트리의 구현
- **노드 정의 및 주요 연산 구현**:
  - 삽입, 삭제, 회전 연산
- **예제**:
```cpp
#include <iostream>
using namespace std;

class AVLNode {
public:
    int key;
    AVLNode* left;
    AVLNode* right;
    int height;

    AVLNode(int key) {
        this->key = key;
        this->left = this->right = nullptr;
        this->height = 1;
    }
};

class AVLTree {
public:
    AVLNode* root;

    AVLTree() {
        root = nullptr;
    }

    int height(AVLNode* node) {
        if (node == nullptr) return 0;
        return node->height;
    }

    int getBalance(AVLNode* node) {
        if (node == nullptr) return 0;
        return height(node->left) - height(node->right);
    }

    AVLNode* rightRotate(AVLNode* y) {
        AVLNode* x = y->left;
        AVLNode* T2 = x->right;

        x->right = y;
        y->left = T2;

        y->height = max(height(y->left), height(y->right)) + 1;
        x->height = max(height(x->left), height(x->right)) + 1;

        return x;
    }

    AVLNode* leftRotate(AVLNode* x) {
        AVLNode* y = x->right;
        AVLNode* T2 = y->left;

        y->left = x;
        x->right = T2;

        x->height = max(height(x->left), height(x->right)) + 1;
        y->height = max(height(y->left), height(y->right)) + 1;

        return y;
    }

    AVLNode* insert(AVLNode* node, int key) {
        if (node == nullptr) return new AVLNode(key);

        if (key < node->key) {
            node->left = insert(node->left, key);
        } else if (key > node->key) {
            node->right = insert(node->right, key);
        } else {
            return node;
        }

        node->height = 1 + max(height(node->left), height(node->right));
        int balance = getBalance(node);

        if (balance > 1 && key < node->left->key) {
            return rightRotate(node);
        }

        if (balance < -1 && key > node->right->key) {
            return leftRotate(node);
        }

        if (balance > 1 && key > node->left->key) {
            node->left = leftRotate(node->left);
            return rightRotate(node);
        }

        if (balance < -1 && key < node->right->key) {
            node->right = rightRotate(node->right);
            return leftRotate(node);
        }

        return node;
    }

    void inorder(AVLNode* root) {
        if (root != nullptr) {
            inorder(root->left);
            cout << root->key << " ";
            inorder(root->right);
        }
    }
};

int main() {
    AVLTree tree;
    tree.root = tree.insert(tree.root, 10);
    tree.root = tree.insert(tree.root, 20);
    tree.root = tree.insert(tree.root, 30);
    tree.root = tree.insert(tree.root, 40);
    tree.root = tree.insert(tree.root, 50);
    tree.root = tree.insert(tree.root, 25);

    cout << "Inorder traversal of the constructed AVL tree is: ";
    tree.inorder(tree.root);
    cout << endl;

    return 0;
}
```

##### 2. 힙 및 힙 정렬 (30분)

###### 2.1 힙의 기본 개념
- **힙 개요**:
  - 완전 이진 트리
  - 최대 힙과 최소 힙

###### 2.2 힙 연산
- **힙 연산**:
  - 삽입, 삭제, 힙 생성
  - 힙 정렬의 개념과 필요성

###### 2.3 힙 정렬 구현
- **힙 정렬 구현**:
  - 힙 생성 및 정렬 과정 구현
- **예제**:
```cpp
#include <iostream>
using namespace std;

void heapify(int arr[], int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;

    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

void heapSort(int arr[], int n) {
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    for (int i = n - 1; i >= 0; i--) {
        swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}

void printArray(int arr[], int n) {
    for (int i = 0; i < n; ++i)
        cout << arr[i] << " ";
    cout << endl;
}

int main() {
    int arr[] = {12, 11, 13, 5, 6, 7};
    int n = sizeof(arr) / sizeof(arr[0]);

    heapSort(arr, n);

    cout << "Sorted array is: ";
    printArray(arr, n);
    return 0;
}
```

##### 3. B-트리 (30분)

###### 3.1 B-트리의 기본 개념
- **B-트리 개요**:
  - 자가 균형 트리 구조
  - 노드가 여러 자식을 가질 수 있음
  - 데이터베이스 및 파일 시스템에서의 활용

###### 3.2 B-트리의 연산
- **B-트리 연산**:
  - 삽입, 삭제, 검색

###### 3.3 B-트리의 구현
- **B-트리 노드 및 연산 구현**:
  - 삽입 및 검색 연산 구현
- **예제**:
```cpp
#include <iostream>
using namespace std;

class BTreeNode {
    int* keys;
    int t;
    BTreeNode** C;
    int n;
    bool leaf;

public:
    BTreeNode(int _t, bool _leaf);

    void insertNonFull(int k);
    void splitChild(int i, BTreeNode* y);
    void traverse();

    BTreeNode* search(int k);

    friend class BTree;
};

class BTree {
    BTreeNode* root;
    int t;
public:
    BTree(int _t) {
        root = nullptr;
        t = _t;
    }

    void traverse() {
        if (root != nullptr) root->traverse();
    }

    BTreeNode* search(int k) {
        return (root == nullptr) ? nullptr : root->search(k);
    }

    void insert(int k);
};

BTreeNode::BTreeNode(int t1, bool leaf1) {
    t = t1;
    leaf = leaf1;
    keys = new int[2 * t - 1];
    C = new BTreeNode * [2 * t];
    n = 0;
}

void BTreeNode::traverse() {
    int i;
    for (i = 0; i < n; i++) {
        if (leaf == false)
            C[i]->traverse();
        cout << " " << keys[i];
    }
    if (leaf == false)
        C[i]->traverse();
}

BTreeNode* BTreeNode::search(int k) {
    int i = 0;
    while (i < n && k > keys[i])
        i++;

    if (keys[i] == k)
        return this;

    if (leaf == true)
        return nullptr;

    return

 C[i]->search(k);
}

void BTree::insert(int k) {
    if (root == nullptr) {
        root = new BTreeNode(t, true);
        root->keys[0] = k;
        root->n = 1;
    } else {
        if (root->n == 2 * t - 1) {
            BTreeNode* s = new BTreeNode(t, false);

            s->C[0] = root;

            s->splitChild(0, root);

            int i = 0;
            if (s->keys[0] < k)
                i++;
            s->C[i]->insertNonFull(k);

            root = s;
        } else
            root->insertNonFull(k);
    }
}

void BTreeNode::insertNonFull(int k) {
    int i = n - 1;

    if (leaf == true) {
        while (i >= 0 && keys[i] > k) {
            keys[i + 1] = keys[i];
            i--;
        }
        keys[i + 1] = k;
        n = n + 1;
    } else {
        while (i >= 0 && keys[i] > k)
            i--;

        if (C[i + 1]->n == 2 * t - 1) {
            splitChild(i + 1, C[i + 1]);

            if (keys[i + 1] < k)
                i++;
        }
        C[i + 1]->insertNonFull(k);
    }
}

void BTreeNode::splitChild(int i, BTreeNode* y) {
    BTreeNode* z = new BTreeNode(y->t, y->leaf);
    z->n = t - 1;

    for (int j = 0; j < t - 1; j++)
        z->keys[j] = y->keys[j + t];

    if (y->leaf == false) {
        for (int j = 0; j < t; j++)
            z->C[j] = y->C[j + t];
    }

    y->n = t - 1;

    for (int j = n; j >= i + 1; j--)
        C[j + 1] = C[j];

    C[i + 1] = z;

    for (int j = n - 1; j >= i; j--)
        keys[j + 1] = keys[j];

    keys[i] = y->keys[t - 1];

    n = n + 1;
}

int main() {
    BTree t(3);
    t.insert(10);
    t.insert(20);
    t.insert(5);
    t.insert(6);
    t.insert(12);
    t.insert(30);
    t.insert(7);
    t.insert(17);

    cout << "Traversal of the constructed tree is ";
    t.traverse();
    cout << endl;

    int k = 6;
    (t.search(k) != nullptr) ? cout << "\nPresent" : cout << "\nNot Present";

    k = 15;
    (t.search(k) != nullptr) ? cout << "\nPresent" : cout << "\nNot Present";

    return 0;
}
```

##### 4. 실습 및 과제 안내 (30분)

###### 4.1 실습
- **실습 목표**:
  - 고급 트리 구조를 사용한 프로그램 작성
- **실습 문제**:
  - AVL 트리를 구현하고, 데이터를 삽입, 삭제, 검색하는 프로그램 작성
  - 힙 정렬을 구현하고, 주어진 데이터를 정렬하는 프로그램 작성
  - B-트리를 구현하고, 데이터를 삽입, 검색하는 프로그램 작성

###### 4.2 과제 안내
- **과제 내용**:
  - AVL 트리를 사용하여 이진 탐색 트리보다 효율적인 학생 성적 관리 프로그램 작성
  - 힙을 사용하여 최대값과 최소값을 빠르게 찾는 우선순위 큐 프로그램 작성
  - B-트리를 사용하여 대용량 데이터베이스 관리 프로그램 작성
- **과제 제출 방법**:
  - 작성한 코드를 주차별 과제 제출 폴더에 업로드
  - 제출 기한: 다음 주 강의 전까지

##### 5. 퀴즈 및 해설 (30분)

###### 퀴즈 문제:
1. AVL 트리에서 균형을 맞추기 위해 사용되는 연산은?
   - a) 삽입
   - b) 삭제
   - c) 회전
   - d) 검색
2. 힙 정렬의 시간 복잡도는?
   - a) O(n)
   - b) O(n log n)
   - c) O(log n)
   - d) O(n^2)
3. B-트리의 특징이 아닌 것은?
   - a) 자가 균형 트리 구조
   - b) 모든 노드가 최대 두 개의 자식을 가짐
   - c) 데이터베이스 및 파일 시스템에서 활용
   - d) 노드가 여러 자식을 가질 수 있음

###### 퀴즈 해설:
1. **정답: c) 회전**
   - AVL 트리에서 균형을 맞추기 위해 회전 연산이 사용됩니다.
2. **정답: b) O(n log n)**
   - 힙 정렬의 시간 복잡도는 O(n log n)입니다.
3. **정답: b) 모든 노드가 최대 두 개의 자식을 가짐**
   - B-트리는 노드가 여러 자식을 가질 수 있는 자가 균형 트리 구조입니다.

### 준비 자료
- 강의 슬라이드
- 실습 코드 예제
- 과제 안내서

### 과제 해설

##### 과제: AVL 트리를 사용하여 학생 성적 관리 프로그램 작성
- **문제**: AVL 트리를 사용하여 학생의 이름과 성적을 관리하는 프로그램 작성
- **해설**:
  - AVL 트리를 사용하여 학생의 이름을 키로, 성적을 값으로 저장합니다.

```cpp
#include <iostream>
#include <string>
using namespace std;

class StudentNode {
public:
    string name;
    int grade;
    StudentNode* left, * right;
    int height;

    StudentNode(string name, int grade) {
        this->name = name;
        this->grade = grade;
        this->left = this->right = nullptr;
        this->height = 1;
    }
};

class StudentAVL {
public:
    StudentNode* root;

    StudentAVL() {
        root = nullptr;
    }

    int height(StudentNode* node) {
        if (node == nullptr) return 0;
        return node->height;
    }

    int getBalance(StudentNode* node) {
        if (node == nullptr) return 0;
        return height(node->left) - height(node->right);
    }

    StudentNode* rightRotate(StudentNode* y) {
        StudentNode* x = y->left;
        StudentNode* T2 = x->right;

        x->right = y;
        y->left = T2;

        y->height = max(height(y->left), height(y->right)) + 1;
        x->height = max(height(x->left), height(x->right)) + 1;

        return x;
    }

    StudentNode* leftRotate(StudentNode* x) {
        StudentNode* y = x->right;
        StudentNode* T2 = y->left;

        y->left = x;
        x->right = T2;

        x->height = max(height(x->left), height(x->right)) + 1;
        y->height = max(height(y->left), height(y->right)) + 1;

        return y;
    }

    StudentNode* insert(StudentNode* node, string name, int grade) {
        if (node == nullptr) return new StudentNode(name, grade);

        if (name < node->name) {
            node->left = insert(node->left, name, grade);
        } else if (name > node->name) {
            node->right = insert(node->right, name, grade);
        } else {
            return node;
        }

        node->height = 1 + max(height(node->left), height(node->right));
        int balance = getBalance(node);

        if (balance > 1 && name < node->left->name) {
            return rightRotate(node);
        }

        if (balance < -1 && name > node->right->name) {
            return leftRotate(node);
        }

        if (balance > 1 && name > node->left->name) {
            node->left = leftRotate(node->left);
            return rightRotate(node);
        }

        if (balance < -1 && name < node->right->name) {
            node->right = rightRotate(node->right);
            return leftRotate(node);
        }

        return node;
    }

    StudentNode* search(StudentNode* node, string name) {
        if (node == nullptr || node->name == name) {
            return node;
        }
        if (name < node->name) {
            return search(node->left, name);
        } else {
            return search(node->right, name);
        }
    }

    void inorder(StudentNode* root) {
        if (root != nullptr) {
            inorder(root->left);
            cout << root->name << ": " << root->grade << endl;
            inorder

(root->right);
        }
    }
};

int main() {
    StudentAVL avl;
    avl.root = avl.insert(avl.root, "Alice", 90);
    avl.insert(avl.root, "Bob", 85);
    avl.insert(avl.root, "Charlie", 92);

    cout << "Inorder traversal: " << endl;
    avl.inorder(avl.root);

    string name = "Bob";
    StudentNode* foundNode = avl.search(avl.root, name);
    if (foundNode) {
        cout << "Found " << foundNode->name << " with grade " << foundNode->grade << endl;
    } else {
        cout << "Not Found" << endl;
    }

    return 0;
}
```
- **설명**:
  - `StudentAVL` 클래스를 사용하여 학생의 이름과 성적을 관리합니다.
  - 학생의 성적을 삽입, 검색, 출력하는 기능을 구현합니다.

##### 과제: 힙을 사용하여 최대값과 최소값을 빠르게 찾는 우선순위 큐 프로그램 작성
- **문제**: 힙을 사용하여 우선순위 큐를 구현하고, 최대값과 최소값을 빠르게 찾는 프로그램 작성
- **해설**:
  - 최대 힙과 최소 힙을 사용하여 우선순위 큐를 구현합니다.

```cpp
#include <iostream>
#include <queue>
using namespace std;

int main() {
    priority_queue<int> maxHeap;
    priority_queue<int, vector<int>, greater<int>> minHeap;

    maxHeap.push(10);
    maxHeap.push(20);
    maxHeap.push(5);

    minHeap.push(10);
    minHeap.push(20);
    minHeap.push(5);

    cout << "Max Heap: ";
    while (!maxHeap.empty()) {
        cout << maxHeap.top() << " ";
        maxHeap.pop();
    }
    cout << endl;

    cout << "Min Heap: ";
    while (!minHeap.empty()) {
        cout << minHeap.top() << " ";
        minHeap.pop();
    }
    cout << endl;

    return 0;
}
```
- **설명**:
  - `priority_queue`를 사용하여 최대 힙과 최소 힙을 구현합니다.
  - 최대값과 최소값을 빠르게 찾는 기능을 구현합니다.

##### 과제: B-트리를 사용하여 대용량 데이터베이스 관리 프로그램 작성
- **문제**: B-트리를 사용하여 대용량 데이터를 관리하는 프로그램 작성
- **해설**:
  - B-트리를 사용하여 데이터를 삽입, 검색하는 기능을 구현합니다.

```cpp
#include <iostream>
using namespace std;

class BTreeNode {
    int* keys;
    int t;
    BTreeNode** C;
    int n;
    bool leaf;

public:
    BTreeNode(int _t, bool _leaf);

    void insertNonFull(int k);
    void splitChild(int i, BTreeNode* y);
    void traverse();

    BTreeNode* search(int k);

    friend class BTree;
};

class BTree {
    BTreeNode* root;
    int t;
public:
    BTree(int _t) {
        root = nullptr;
        t = _t;
    }

    void traverse() {
        if (root != nullptr) root->traverse();
    }

    BTreeNode* search(int k) {
        return (root == nullptr) ? nullptr : root->search(k);
    }

    void insert(int k);
};

BTreeNode::BTreeNode(int t1, bool leaf1) {
    t = t1;
    leaf = leaf1;
    keys = new int[2 * t - 1];
    C = new BTreeNode * [2 * t];
    n = 0;
}

void BTreeNode::traverse() {
    int i;
    for (i = 0; i < n; i++) {
        if (leaf == false)
            C[i]->traverse();
        cout << " " << keys[i];
    }
    if (leaf == false)
        C[i]->traverse();
}

BTreeNode* BTreeNode::search(int k) {
    int i = 0;
    while (i < n && k > keys[i])
        i++;

    if (keys[i] == k)
        return this;

    if (leaf == true)
        return nullptr;

    return C[i]->search(k);
}

void BTree::insert(int k) {
    if (root == nullptr) {
        root = new BTreeNode(t, true);
        root->keys[0] = k;
        root->n = 1;
    } else {
        if (root->n == 2 * t - 1) {
            BTreeNode* s = new BTreeNode(t, false);

            s->C[0] = root;

            s->splitChild(0, root);

            int i = 0;
            if (s->keys[0] < k)
                i++;
            s->C[i]->insertNonFull(k);

            root = s;
        } else
            root->insertNonFull(k);
    }
}

void BTreeNode::insertNonFull(int k) {
    int i = n - 1;

    if (leaf == true) {
        while (i >= 0 && keys[i] > k) {
            keys[i + 1] = keys[i];
            i--;
        }
        keys[i + 1] = k;
        n = n + 1;
    } else {
        while (i >= 0 && keys[i] > k)
            i--;

        if (C[i + 1]->n == 2 * t - 1) {
            splitChild(i + 1, C[i + 1]);

            if (keys[i + 1] < k)
                i++;
        }
        C[i + 1]->insertNonFull(k);
    }
}

void BTreeNode::splitChild(int i, BTreeNode* y) {
    BTreeNode* z = new BTreeNode(y->t, y->leaf);
    z->n = t - 1;

    for (int j = 0; j < t - 1; j++)
        z->keys[j] = y->keys[j + t];

    if (y->leaf == false) {
        for (int j = 0; j < t; j++)
            z->C[j] = y->C[j + t];
    }

    y->n = t - 1;

    for (int j = n; j >= i + 1; j--)
        C[j + 1] = C[j];

    C[i + 1] = z;

    for (int j = n - 1; j >= i; j--)
        keys[j + 1] = keys[j];

    keys[i] = y->keys[t - 1];

    n = n + 1;
}

int main() {
    BTree t(3);
    t.insert(10);
    t.insert(20);
    t.insert(5);
    t.insert(6);
    t.insert(12);
    t.insert(30);
    t.insert(7);
    t.insert(17);

    cout << "Traversal of the constructed tree is ";
    t.traverse();
    cout << endl;

    int k = 6;
    (t.search(k) != nullptr) ? cout << "\nPresent" : cout << "\nNot Present";

    k = 15;
    (t.search(k) != nullptr) ? cout << "\nPresent" : cout << "\nNot Present";

    return 0;
}
```
- **설명**:
  - `BTree` 클래스를 사용하여 대용량 데이터를 관리합니다.
  - 데이터를 삽입, 검색, 출력하는 기능을 구현합니다.

이로써 11주차 강의가 마무리됩니다. 학생들은 고급 트리 구조의 기본 개념과 구현 방법을 이해하고, 이를 활용하여 다양한 프로그램을 작성하는 능력을 기르게 됩니다.