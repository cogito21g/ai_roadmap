### 자료구조 교육과정 - 4주차: 이진 탐색 트리 (BST)

**강의 목표:**
이진 탐색 트리의 개념과 구현 방법을 학습하고, 삽입, 삭제, 탐색 연산을 구현합니다.

**강의 구성:**

#### 4. 이진 탐색 트리 (BST)

**강의 내용:**
- 이진 탐색 트리의 개념
  - 이진 탐색 트리란?
  - 이진 탐색 트리의 특징
- 이진 탐색 트리의 연산
  - 삽입 연산
  - 삭제 연산
  - 탐색 연산

**실습:**
- 이진 탐색 트리 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>

  struct Node {
      int key;
      struct Node* left;
      struct Node* right;
  };

  struct Node* newNode(int item) {
      struct Node* temp = (struct Node*) malloc(sizeof(struct Node));
      temp->key = item;
      temp->left = temp->right = NULL;
      return temp;
  }

  void inorder(struct Node* root) {
      if (root != NULL) {
          inorder(root->left);
          printf("%d ", root->key);
          inorder(root->right);
      }
  }

  struct Node* insert(struct Node* node, int key) {
      if (node == NULL) return newNode(key);

      if (key < node->key)
          node->left = insert(node->left, key);
      else if (key > node->key)
          node->right = insert(node->right, key);

      return node;
  }

  struct Node* minValueNode(struct Node* node) {
      struct Node* current = node;

      while (current && current->left != NULL)
          current = current->left;

      return current;
  }

  struct Node* deleteNode(struct Node* root, int key) {
      if (root == NULL) return root;

      if (key < root->key)
          root->left = deleteNode(root->left, key);
      else if (key > root->key)
          root->right = deleteNode(root->right, key);
      else {
          if (root->left == NULL) {
              struct Node* temp = root->right;
              free(root);
              return temp;
          } else if (root->right == NULL) {
              struct Node* temp = root->left;
              free(root);
              return temp;
          }

          struct Node* temp = minValueNode(root->right);
          root->key = temp->key;
          root->right = deleteNode(root->right, temp->key);
      }
      return root;
  }

  struct Node* search(struct Node* root, int key) {
      if (root == NULL || root->key == key)
          return root;

      if (root->key < key)
          return search(root->right, key);

      return search(root->left, key);
  }

  int main() {
      struct Node* root = NULL;
      root = insert(root, 50);
      root = insert(root, 30);
      root = insert(root, 20);
      root = insert(root, 40);
      root = insert(root, 70);
      root = insert(root, 60);
      root = insert(root, 80);

      printf("Inorder traversal: ");
      inorder(root);
      printf("\n");

      printf("Delete 20\n");
      root = deleteNode(root, 20);
      printf("Inorder traversal: ");
      inorder(root);
      printf("\n");

      printf("Delete 30\n");
      root = deleteNode(root, 30);
      printf("Inorder traversal: ");
      inorder(root);
      printf("\n");

      printf("Delete 50\n");
      root = deleteNode(root, 50);
      printf("Inorder traversal: ");
      inorder(root);
      printf("\n");

      struct Node* result = search(root, 40);
      if (result != NULL) {
          printf("Node found: %d\n", result->key);
      } else {
          printf("Node not found\n");
      }

      return 0;
  }
  ```

**과제:**
- 이진 탐색 트리에서 특정 값을 삽입하고 삭제하는 함수를 각각 구현
- 이진 탐색 트리에서 특정 값을 탐색하는 함수를 구현하고, 해당 값이 존재하는지 확인

**퀴즈 및 해설:**

1. **이진 탐색 트리란 무엇인가요?**
   - 이진 탐색 트리는 각 노드가 최대 두 개의 자식 노드를 가지며, 왼쪽 서브트리의 모든 노드 값이 루트 노드 값보다 작고, 오른쪽 서브트리의 모든 노드 값이 루트 노드 값보다 큰 특성을 가지는 트리입니다.

2. **이진 탐색 트리의 삽입 연산의 시간 복잡도는 무엇인가요?**
   - 평균적으로 O(log n)의 시간 복잡도를 가지지만, 최악의 경우(편향 트리)에는 O(n)의 시간 복잡도를 가집니다.

3. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    struct Node* root = NULL;
    root = insert(root, 50);
    root = insert(root, 30);
    root = insert(root, 20);
    root = insert(root, 40);
    root = insert(root, 70);
    root = insert(root, 60);
    root = insert(root, 80);

    printf("Inorder traversal: ");
    inorder(root);
    printf("\n");

    root = deleteNode(root, 20);
    printf("Inorder traversal: ");
    inorder(root);
    printf("\n");

    struct Node* result = search(root, 40);
    if (result != NULL) {
        printf("Node found: %d\n", result->key);
    } else {
        printf("Node not found\n");
    }
    ```
   - 출력 결과:
     ```
     Inorder traversal: 20 30 40 50 60 70 80 
     Inorder traversal: 30 40 50 60 70 80 
     Node found: 40
     ```

**해설:**
1. **이진 탐색 트리의 정의**는 각 노드가 최대 두 개의 자식 노드를 가지며, 왼쪽 서브트리의 모든 노드 값이 루트 노드 값보다 작고, 오른쪽 서브트리의 모든 노드 값이 루트 노드 값보다 큰 특성을 가지는 트리입니다.
2. **이진 탐색 트리의 삽입 연산의 시간 복잡도**는 평균적으로 O(log n)이지만, 최악의 경우(편향 트리)에는 O(n)의 시간 복잡도를 가집니다.
3. **코드 출력 결과**는 이진 탐색 트리의 중위 순회, 삭제 연산, 탐색 연산의 결과를 보여줍니다. 중위 순회는 노드를 오름차순으로 방문하며, 삭제 연산은 특정 값을 가진 노드를 제거하고, 탐색 연산은 특정 값을 가진 노드를 찾습니다.

---
