### 10주차 강의 계획: 트리 (Tree)

#### 강의 목표
- 트리의 기본 개념과 구조 이해
- 이진 트리와 이진 탐색 트리(BST) 이해 및 구현
- 트리 순회 방법(전위, 중위, 후위) 학습

#### 강의 구성
- **강의 시간**: 2시간
- **구성**: 트리 이론 (30분), 이진 트리와 이진 탐색 트리 (30분), 트리 순회 방법 (30분), 실습 및 과제 안내 (30분)

#### 강의 내용

##### 1. 트리 이론 (30분)

###### 1.1 트리의 기본 개념
- **트리 개요**:
  - 트리의 정의 및 구조
  - 노드, 간선, 루트, 리프 노드의 개념
  - 트리의 종류 (이진 트리, AVL 트리, B-트리 등)

###### 1.2 트리의 활용 사례
- **활용 사례**:
  - 계층적 데이터 구조 표현 (파일 시스템, 조직도 등)
  - 검색 알고리즘 (이진 탐색 트리)
  - 네트워크 라우팅

##### 2. 이진 트리와 이진 탐색 트리 (30분)

###### 2.1 이진 트리의 구조
- **이진 트리**:
  - 각 노드가 최대 두 개의 자식 노드를 가지는 트리
  - 완전 이진 트리와 완전 이진 트리의 개념

###### 2.2 이진 탐색 트리(BST)의 구조와 성질
- **이진 탐색 트리**:
  - 왼쪽 자식 노드는 부모 노드보다 작고, 오른쪽 자식 노드는 부모 노드보다 큰 특성
  - 삽입, 삭제, 검색 연산의 평균 시간 복잡도는 O(log n)

###### 2.3 이진 탐색 트리의 구현
- **노드 정의 및 주요 연산 구현**:
  - 삽입, 삭제, 검색
- **예제**:
```cpp
#include <iostream>
using namespace std;

class TreeNode {
public:
    int key;
    TreeNode* left, * right;
    TreeNode(int key) {
        this->key = key;
        this->left = this->right = nullptr;
    }
};

class BinarySearchTree {
public:
    TreeNode* root;
    BinarySearchTree() {
        root = nullptr;
    }

    TreeNode* insert(TreeNode* node, int key) {
        if (node == nullptr) {
            return new TreeNode(key);
        }
        if (key < node->key) {
            node->left = insert(node->left, key);
        } else {
            node->right = insert(node->right, key);
        }
        return node;
    }

    TreeNode* search(TreeNode* node, int key) {
        if (node == nullptr || node->key == key) {
            return node;
        }
        if (key < node->key) {
            return search(node->left, key);
        } else {
            return search(node->right, key);
        }
    }

    TreeNode* minValueNode(TreeNode* node) {
        TreeNode* current = node;
        while (current && current->left != nullptr) {
            current = current->left;
        }
        return current;
    }

    TreeNode* deleteNode(TreeNode* root, int key) {
        if (root == nullptr) return root;
        if (key < root->key) {
            root->left = deleteNode(root->left, key);
        } else if (key > root->key) {
            root->right = deleteNode(root->right, key);
        } else {
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
    BinarySearchTree bst;
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

    cout << "Delete 20\n";
    bst.root = bst.deleteNode(bst.root, 20);
    cout << "Inorder traversal: ";
    bst.inorder(bst.root);
    cout << endl;

    cout << "Search for 70: ";
    TreeNode* foundNode = bst.search(bst.root, 70);
    if (foundNode) {
        cout << "Found " << foundNode->key << endl;
    } else {
        cout << "Not Found" << endl;
    }

    return 0;
}
```

##### 3. 트리 순회 방법 (30분)

###### 3.1 트리 순회의 종류
- **전위 순회 (Preorder Traversal)**:
  - 노드 방문 -> 왼쪽 서브트리 방문 -> 오른쪽 서브트리 방문
- **중위 순회 (Inorder Traversal)**:
  - 왼쪽 서브트리 방문 -> 노드 방문 -> 오른쪽 서브트리 방문
- **후위 순회 (Postorder Traversal)**:
  - 왼쪽 서브트리 방문 -> 오른쪽 서브트리 방문 -> 노드 방문

###### 3.2 순회 방법 구현
- **순회 방법 구현**:
  - 전위, 중위, 후위 순회
- **예제**:
```cpp
#include <iostream>
using namespace std;

class TreeNode {
public:
    int key;
    TreeNode* left, * right;
    TreeNode(int key) {
        this->key = key;
        this->left = this->right = nullptr;
    }
};

class BinaryTree {
public:
    TreeNode* root;
    BinaryTree() {
        root = nullptr;
    }

    void preorder(TreeNode* node) {
        if (node == nullptr) return;
        cout << node->key << " ";
        preorder(node->left);
        preorder(node->right);
    }

    void inorder(TreeNode* node) {
        if (node == nullptr) return;
        inorder(node->left);
        cout << node->key << " ";
        inorder(node->right);
    }

    void postorder(TreeNode* node) {
        if (node == nullptr) return;
        postorder(node->left);
        postorder(node->right);
        cout << node->key << " ";
    }
};

int main() {
    BinaryTree tree;
    tree.root = new TreeNode(1);
    tree.root->left = new TreeNode(2);
    tree.root->right = new TreeNode(3);
    tree.root->left->left = new TreeNode(4);
    tree.root->left->right = new TreeNode(5);

    cout << "Preorder traversal: ";
    tree.preorder(tree.root);
    cout << endl;

    cout << "Inorder traversal: ";
    tree.inorder(tree.root);
    cout << endl;

    cout << "Postorder traversal: ";
    tree.postorder(tree.root);
    cout << endl;

    return 0;
}
```

##### 4. 실습 및 과제 안내 (30분)

###### 4.1 실습
- **실습 목표**:
  - 트리를 사용한 프로그램 작성
- **실습 문제**:
  - 이진 탐색 트리를 구현하고, 데이터를 삽입, 삭제, 검색하는 프로그램 작성
  - 트리 순회 방법(전위, 중위, 후위)을 구현하고 각 순회 방법을 통해 트리를 출력하는 프로그램 작성

###### 4.2 과제 안내
- **과제 내용**:
  - 이진 탐색 트리를 사용하여 학생 성적 관리 프로그램 작성
  - 트리를 사용하여 표현식(수식) 트리(Expression Tree)를 구현하고, 전위, 중위, 후위 순회를 통해 수식을 계산하는 프로그램 작성
- **과제 제출 방법**:
  - 작성한 코드를 주차별 과제 제출 폴더에 업로드
  - 제출 기한: 다음 주 강의 전까지

##### 5. 퀴즈 및 해설 (30분)

###### 퀴즈 문제:
1. 이진 탐색 트리에서 검색 연산의 평균 시간 복잡도는?
   - a) O(n)
   - b) O(log n)
   - c) O(n^2)
   - d) O(1)
2. 중위 순회(Inorder Traversal)에서 트리의 노드가 방문되는 순서는?
   - a) 노드 -> 왼쪽 서브트리 -> 오른쪽 서

브트리
   - b) 왼쪽 서브트리 -> 노드 -> 오른쪽 서브트리
   - c) 왼쪽 서브트리 -> 오른쪽 서브트리 -> 노드
   - d) 오른쪽 서브트리 -> 노드 -> 왼쪽 서브트리
3. 이진 탐색 트리에서 최소 값을 찾기 위해 탐색해야 하는 서브트리는?
   - a) 오른쪽 서브트리
   - b) 왼쪽 서브트리
   - c) 루트 노드
   - d) 아무 서브트리나 상관없음

###### 퀴즈 해설:
1. **정답: b) O(log n)**
   - 이진 탐색 트리에서 검색 연산의 평균 시간 복잡도는 O(log n)입니다.
2. **정답: b) 왼쪽 서브트리 -> 노드 -> 오른쪽 서브트리**
   - 중위 순회에서는 왼쪽 서브트리, 노드, 오른쪽 서브트리 순으로 방문합니다.
3. **정답: b) 왼쪽 서브트리**
   - 이진 탐색 트리에서 최소 값은 항상 왼쪽 서브트리에 있습니다.

### 준비 자료
- 강의 슬라이드
- 실습 코드 예제
- 과제 안내서

### 과제 해설

##### 과제: 이진 탐색 트리를 사용하여 학생 성적 관리 프로그램 작성
- **문제**: 이진 탐색 트리를 사용하여 학생의 이름과 성적을 관리하는 프로그램 작성
- **해설**:
  - 이진 탐색 트리를 사용하여 학생의 이름을 키로, 성적을 값으로 저장합니다.

```cpp
#include <iostream>
#include <string>
using namespace std;

class StudentNode {
public:
    string name;
    int grade;
    StudentNode* left, * right;
    StudentNode(string name, int grade) {
        this->name = name;
        this->grade = grade;
        this->left = this->right = nullptr;
    }
};

class StudentBST {
public:
    StudentNode* root;
    StudentBST() {
        root = nullptr;
    }

    StudentNode* insert(StudentNode* node, string name, int grade) {
        if (node == nullptr) {
            return new StudentNode(name, grade);
        }
        if (name < node->name) {
            node->left = insert(node->left, name, grade);
        } else {
            node->right = insert(node->right, name, grade);
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
            inorder(root->right);
        }
    }
};

int main() {
    StudentBST bst;
    bst.root = bst.insert(bst.root, "Alice", 90);
    bst.insert(bst.root, "Bob", 85);
    bst.insert(bst.root, "Charlie", 92);

    cout << "Inorder traversal: " << endl;
    bst.inorder(bst.root);

    string name = "Bob";
    StudentNode* foundNode = bst.search(bst.root, name);
    if (foundNode) {
        cout << "Found " << foundNode->name << " with grade " << foundNode->grade << endl;
    } else {
        cout << "Not Found" << endl;
    }

    return 0;
}
```
- **설명**:
  - `StudentBST` 클래스를 사용하여 학생의 이름과 성적을 관리합니다.
  - 학생의 성적을 삽입, 검색, 출력하는 기능을 구현합니다.

##### 과제: 트리를 사용하여 표현식(수식) 트리(Expression Tree)를 구현하고, 전위, 중위, 후위 순회를 통해 수식을 계산하는 프로그램 작성
- **문제**: 트리를 사용하여 수식 트리를 구현하고, 전위, 중위, 후위 순회를 통해 수식을 계산하는 프로그램 작성
- **해설**:
  - 수식 트리를 구현하고, 각 순회 방법을 사용하여 수식을 계산합니다.

```cpp
#include <iostream>
#include <stack>
using namespace std;

class ExprNode {
public:
    char value;
    ExprNode* left, * right;
    ExprNode(char value) {
        this->value = value;
        this->left = this->right = nullptr;
    }
};

class ExpressionTree {
public:
    ExprNode* root;
    ExpressionTree() {
        root = nullptr;
    }

    bool isOperator(char c) {
        return (c == '+' || c == '-' || c == '*' || c == '/');
    }

    void inorder(ExprNode* node) {
        if (node != nullptr) {
            inorder(node->left);
            cout << node->value << " ";
            inorder(node->right);
        }
    }

    void preorder(ExprNode* node) {
        if (node != nullptr) {
            cout << node->value << " ";
            preorder(node->left);
            preorder(node->right);
        }
    }

    void postorder(ExprNode* node) {
        if (node != nullptr) {
            postorder(node->left);
            postorder(node->right);
            cout << node->value << " ";
        }
    }

    ExprNode* constructTree(string postfix) {
        stack<ExprNode*> st;
        ExprNode* t, * t1, * t2;

        for (int i = 0; i < postfix.length(); i++) {
            if (!isOperator(postfix[i])) {
                t = new ExprNode(postfix[i]);
                st.push(t);
            } else {
                t = new ExprNode(postfix[i]);

                t1 = st.top();
                st.pop();
                t2 = st.top();
                st.pop();

                t->right = t1;
                t->left = t2;

                st.push(t);
            }
        }

        t = st.top();
        st.pop();

        return t;
    }
};

int main() {
    ExpressionTree et;
    string postfix = "ab+ef*g*-";
    et.root = et.constructTree(postfix);

    cout << "Inorder traversal: ";
    et.inorder(et.root);
    cout << endl;

    cout << "Preorder traversal: ";
    et.preorder(et.root);
    cout << endl;

    cout << "Postorder traversal: ";
    et.postorder(et.root);
    cout << endl;

    return 0;
}
```
- **설명**:
  - `ExpressionTree` 클래스를 사용하여 수식 트리를 구현합니다.
  - 후위 표기법(Postfix)을 사용하여 트리를 구성하고, 전위, 중위, 후위 순회를 통해 수식을 출력합니다.

이로써 10주차 강의가 마무리됩니다. 학생들은 트리의 기본 개념과 구현 방법을 이해하고, 이를 활용하여 다양한 프로그램을 작성하는 능력을 기르게 됩니다.