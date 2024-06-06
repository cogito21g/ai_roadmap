### 자료구조 교육과정 - 3주차: 트리 기초

**강의 목표:**
트리의 기본 개념과 이진 트리의 구현 및 탐색 방법을 학습합니다.

**강의 구성:**

#### 3. 트리 기초

**강의 내용:**
- 트리의 개념
  - 트리의 정의 및 특징
  - 트리의 용어: 루트, 노드, 리프, 서브트리
- 이진 트리
  - 이진 트리의 정의 및 특징
  - 이진 트리의 구현
  - 이진 트리의 순회 방법 (전위, 중위, 후위)

**실습:**
- 이진 트리 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>

  struct Node {
      int data;
      struct Node* left;
      struct Node* right;
  };

  struct Node* newNode(int data) {
      struct Node* node = (struct Node*) malloc(sizeof(struct Node));
      node->data = data;
      node->left = NULL;
      node->right = NULL;
      return node;
  }

  void printPreorder(struct Node* node) {
      if (node == NULL)
          return;
      printf("%d ", node->data);
      printPreorder(node->left);
      printPreorder(node->right);
  }

  void printInorder(struct Node* node) {
      if (node == NULL)
          return;
      printInorder(node->left);
      printf("%d ", node->data);
      printInorder(node->right);
  }

  void printPostorder(struct Node* node) {
      if (node == NULL)
          return;
      printPostorder(node->left);
      printPostorder(node->right);
      printf("%d ", node->data);
  }

  int main() {
      struct Node* root = newNode(1);
      root->left = newNode(2);
      root->right = newNode(3);
      root->left->left = newNode(4);
      root->left->right = newNode(5);

      printf("Preorder traversal: ");
      printPreorder(root);
      printf("\n");

      printf("Inorder traversal: ");
      printInorder(root);
      printf("\n");

      printf("Postorder traversal: ");
      printPostorder(root);
      printf("\n");

      return 0;
  }
  ```

**과제:**
- 주어진 이진 트리를 전위, 중위, 후위 순회하는 함수를 각각 구현
- 이진 트리에서 특정 값을 탐색하는 함수 구현

**퀴즈 및 해설:**

1. **트리의 정의는 무엇인가요?**
   - 트리는 계층적 데이터 구조로, 노드들이 하나의 루트 노드에서 시작하여 여러 개의 자식 노드를 가질 수 있습니다. 각 노드는 0개 이상의 자식 노드를 가질 수 있으며, 루트 노드는 부모가 없는 유일한 노드입니다.

2. **이진 트리란 무엇인가요?**
   - 이진 트리는 각 노드가 최대 두 개의 자식 노드를 가지는 트리입니다. 이진 트리의 각 노드는 왼쪽 자식과 오른쪽 자식을 가지며, 트리의 깊이는 루트 노드에서 리프 노드까지의 최대 경로 길이로 정의됩니다.

3. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    struct Node* root = newNode(1);
    root->left = newNode(2);
    root->right = newNode(3);
    root->left->left = newNode(4);
    root->left->right = newNode(5);

    printf("Preorder traversal: ");
    printPreorder(root);
    printf("\n");

    printf("Inorder traversal: ");
    printInorder(root);
    printf("\n");

    printf("Postorder traversal: ");
    printPostorder(root);
    printf("\n");
    ```
   - 출력 결과:
     ```
     Preorder traversal: 1 2 4 5 3 
     Inorder traversal: 4 2 5 1 3 
     Postorder traversal: 4 5 2 3 1 
     ```

**해설:**
1. **트리의 정의**는 계층적 데이터 구조로, 루트 노드에서 시작하여 여러 개의 자식 노드를 가질 수 있습니다. 각 노드는 0개 이상의 자식 노드를 가지며, 루트 노드는 부모가 없는 유일한 노드입니다.
2. **이진 트리**는 각 노드가 최대 두 개의 자식 노드를 가지는 트리입니다. 이진 트리의 각 노드는 왼쪽 자식과 오른쪽 자식을 가지며, 트리의 깊이는 루트 노드에서 리프 노드까지의 최대 경로 길이로 정의됩니다.
3. **코드 출력 결과**는 이진 트리의 전위, 중위, 후위 순회를 통해 각 노드의 값을 출력합니다. 전위 순회는 루트, 왼쪽, 오른쪽 순으로, 중위 순회는 왼쪽, 루트, 오른쪽 순으로, 후위 순회는 왼쪽, 오른쪽, 루트 순으로 노드를 방문합니다.

---

다음 주차 내용을 원하시면 알려주세요. 4주차는 이진 탐색 트리 (BST)를 다룰 예정입니다.