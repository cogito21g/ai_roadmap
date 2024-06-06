### 자료구조 교육과정 - 5주차: AVL 트리

**강의 목표:**
AVL 트리의 개념과 필요성을 이해하고, AVL 트리의 삽입, 삭제, 탐색 연산을 학습합니다.

**강의 구성:**

#### 5. AVL 트리

**강의 내용:**
- AVL 트리의 개념
  - AVL 트리란 무엇인가?
  - AVL 트리의 특징과 필요성
- AVL 트리의 균형 인수와 회전
  - 균형 인수의 정의
  - 회전의 종류 (LL, RR, LR, RL)
- AVL 트리의 연산
  - 삽입 연산과 균형 유지
  - 삭제 연산과 균형 유지
  - 탐색 연산

**실습:**
- AVL 트리 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>

  struct Node {
      int key;
      struct Node *left;
      struct Node *right;
      int height;
  };

  int max(int a, int b) {
      return (a > b) ? a : b;
  }

  int height(struct Node *N) {
      if (N == NULL)
          return 0;
      return N->height;
  }

  struct Node* newNode(int key) {
      struct Node* node = (struct Node*) malloc(sizeof(struct Node));
      node->key = key;
      node->left = NULL;
      node->right = NULL;
      node->height = 1;
      return(node);
  }

  struct Node *rightRotate(struct Node *y) {
      struct Node *x = y->left;
      struct Node *T2 = x->right;

      x->right = y;
      y->left = T2;

      y->height = max(height(y->left), height(y->right)) + 1;
      x->height = max(height(x->left), height(x->right)) + 1;

      return x;
  }

  struct Node *leftRotate(struct Node *x) {
      struct Node *y = x->right;
      struct Node *T2 = y->left;

      y->left = x;
      x->right = T2;

      x->height = max(height(x->left), height(x->right)) + 1;
      y->height = max(height(y->left), height(y->right)) + 1;

      return y;
  }

  int getBalance(struct Node *N) {
      if (N == NULL)
          return 0;
      return height(N->left) - height(N->right);
  }

  struct Node* insert(struct Node* node, int key) {
      if (node == NULL)
          return(newNode(key));

      if (key < node->key)
          node->left = insert(node->left, key);
      else if (key > node->key)
          node->right = insert(node->right, key);
      else
          return node;

      node->height = 1 + max(height(node->left), height(node->right));

      int balance = getBalance(node);

      if (balance > 1 && key < node->left->key)
          return rightRotate(node);

      if (balance < -1 && key > node->right->key)
          return leftRotate(node);

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

  struct Node* minValueNode(struct Node* node) {
      struct Node* current = node;

      while (current->left != NULL)
          current = current->left;

      return current;
  }

  struct Node* deleteNode(struct Node* root, int key) {
      if (root == NULL)
          return root;

      if (key < root->key)
          root->left = deleteNode(root->left, key);
      else if (key > root->key)
          root->right = deleteNode(root->right, key);
      else {
          if ((root->left == NULL) || (root->right == NULL)) {
              struct Node *temp = root->left ? root->left : root->right;

              if (temp == NULL) {
                  temp = root;
                  root = NULL;
              } else
                  *root = *temp;
              free(temp);
          } else {
              struct Node* temp = minValueNode(root->right);
              root->key = temp->key;
              root->right = deleteNode(root->right, temp->key);
          }
      }

      if (root == NULL)
          return root;

      root->height = 1 + max(height(root->left), height(root->right));

      int balance = getBalance(root);

      if (balance > 1 && getBalance(root->left) >= 0)
          return rightRotate(root);

      if (balance > 1 && getBalance(root->left) < 0) {
          root->left = leftRotate(root->left);
          return rightRotate(root);
      }

      if (balance < -1 && getBalance(root->right) <= 0)
          return leftRotate(root);

      if (balance < -1 && getBalance(root->right) > 0) {
          root->right = rightRotate(root->right);
          return leftRotate(root);
      }

      return root;
  }

  void preOrder(struct Node *root) {
      if (root != NULL) {
          printf("%d ", root->key);
          preOrder(root->left);
          preOrder(root->right);
      }
  }

  int main() {
      struct Node *root = NULL;

      root = insert(root, 10);
      root = insert(root, 20);
      root = insert(root, 30);
      root = insert(root, 40);
      root = insert(root, 50);
      root = insert(root, 25);

      printf("Preorder traversal of the constructed AVL tree is \n");
      preOrder(root);

      root = deleteNode(root, 10);
      printf("\nPreorder traversal after deletion of 10 \n");
      preOrder(root);

      return 0;
  }
  ```

**과제:**
- AVL 트리의 삽입 연산을 구현하고, 여러 데이터를 삽입하여 균형이 잘 유지되는지 확인합니다.
- AVL 트리에서 특정 노드를 삭제하는 연산을 구현하고, 삭제 후에도 균형이 유지되는지 확인합니다.

**퀴즈 및 해설:**

1. **AVL 트리란 무엇인가요?**
   - AVL 트리는 모든 노드의 왼쪽 서브트리와 오른쪽 서브트리의 높이 차이가 최대 1인 이진 탐색 트리입니다. 이를 통해 항상 균형이 유지되어 최악의 경우에도 O(log n)의 시간 복잡도를 보장합니다.

2. **AVL 트리의 균형 인수란 무엇인가요?**
   - 균형 인수는 특정 노드의 왼쪽 서브트리와 오른쪽 서브트리의 높이 차이를 나타냅니다. AVL 트리에서는 이 값이 -1, 0, 1 중 하나여야 합니다.

3. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    struct Node *root = NULL;
    root = insert(root, 10);
    root = insert(root, 20);
    root = insert(root, 30);
    root = insert(root, 40);
    root = insert(root, 50);
    root = insert(root, 25);

    printf("Preorder traversal of the constructed AVL tree is \n");
    preOrder(root);
    ```
   - 출력 결과:
     ```
     Preorder traversal of the constructed AVL tree is 
     30 20 10 25 40 50 
     ```

**해설:**
1. **AVL 트리의 정의**는 모든 노드의 왼쪽 서브트리와 오른쪽 서브트리의 높이 차이가 최대 1인 이진 탐색 트리입니다. 이를 통해 항상 균형이 유지되어 최악의 경우에도 O(log n)의 시간 복잡도를 보장합니다.
2. **AVL 트리의 균형 인수**는 특정 노드의 왼쪽 서브트리와 오른쪽 서브트리의 높이 차이를 나타냅니다. AVL 트리에서는 이 값이 -1, 0, 1 중 하나여야 합니다.
3. **코드 출력 결과**는 AVL 트리의 삽입 연산 후 트리의 전위 순회를 통해 각 노드의 값을 출력합니다. AVL 트리의 삽입 연산은 삽입 후 균형을 유지하기 위해 회전을 수행할 수 있습니다.

---

다음 주차 내용을 원하시면 알려주세요. 6주차는 B-트리와 해시 테이블을 다룰 예정입니다.