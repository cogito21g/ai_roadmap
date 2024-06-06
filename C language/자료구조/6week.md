### 자료구조 교육과정 - 6주차: B-트리와 해시 테이블

**강의 목표:**
B-트리와 해시 테이블의 개념과 구현 방법을 학습하고, 각각의 연산을 구현합니다.

**강의 구성:**

#### 6. B-트리와 해시 테이블

### B-트리

**강의 내용:**
- B-트리의 개념
  - B-트리란 무엇인가?
  - B-트리의 특징과 필요성
- B-트리의 구조
  - 노드 구조
  - 최소 차수 (t)
- B-트리의 연산
  - 삽입 연산
  - 삭제 연산
  - 탐색 연산

**실습:**
- B-트리 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>

  // B-트리 노드의 최소 차수 정의
  #define T 3

  struct BTreeNode {
      int keys[2*T-1];
      struct BTreeNode *C[2*T];
      int n;
      int leaf;
  };

  struct BTreeNode *createNode(int leaf) {
      struct BTreeNode *newNode = (struct BTreeNode *)malloc(sizeof(struct BTreeNode));
      newNode->leaf = leaf;
      newNode->n = 0;
      return newNode;
  }

  void traverse(struct BTreeNode *root) {
      int i;
      for (i = 0; i < root->n; i++) {
          if (root->leaf == 0)
              traverse(root->C[i]);
          printf("%d ", root->keys[i]);
      }

      if (root->leaf == 0)
          traverse(root->C[i]);
  }

  struct BTreeNode *search(struct BTreeNode *root, int k) {
      int i = 0;
      while (i < root->n && k > root->keys[i])
          i++;

      if (root->keys[i] == k)
          return root;

      if (root->leaf == 1)
          return NULL;

      return search(root->C[i], k);
  }

  void insert(struct BTreeNode **root, int k);
  void insertNonFull(struct BTreeNode *x, int k);
  void splitChild(struct BTreeNode *x, int i, struct BTreeNode *y);

  void insert(struct BTreeNode **root, int k) {
      if ((*root)->n == 2*T-1) {
          struct BTreeNode *s = createNode(0);
          s->C[0] = *root;
          splitChild(s, 0, *root);
          int i = 0;
          if (s->keys[0] < k)
              i++;
          insertNonFull(s->C[i], k);
          *root = s;
      } else
          insertNonFull(*root, k);
  }

  void insertNonFull(struct BTreeNode *x, int k) {
      int i = x->n-1;

      if (x->leaf == 1) {
          while (i >= 0 && x->keys[i] > k) {
              x->keys[i+1] = x->keys[i];
              i--;
          }
          x->keys[i+1] = k;
          x->n = x->n + 1;
      } else {
          while (i >= 0 && x->keys[i] > k)
              i--;
          if (x->C[i+1]->n == 2*T-1) {
              splitChild(x, i+1, x->C[i+1]);
              if (x->keys[i+1] < k)
                  i++;
          }
          insertNonFull(x->C[i+1], k);
      }
  }

  void splitChild(struct BTreeNode *x, int i, struct BTreeNode *y) {
      struct BTreeNode *z = createNode(y->leaf);
      z->n = T - 1;
      for (int j = 0; j < T-1; j++)
          z->keys[j] = y->keys[j+T];
      if (y->leaf == 0) {
          for (int j = 0; j < T; j++)
              z->C[j] = y->C[j+T];
      }
      y->n = T - 1;
      for (int j = x->n; j >= i+1; j--)
          x->C[j+1] = x->C[j];
      x->C[i+1] = z;
      for (int j = x->n-1; j >= i; j--)
          x->keys[j+1] = x->keys[j];
      x->keys[i] = y->keys[T-1];
      x->n = x->n + 1;
  }

  int main() {
      struct BTreeNode *root = createNode(1);

      insert(&root, 10);
      insert(&root, 20);
      insert(&root, 5);
      insert(&root, 6);
      insert(&root, 12);
      insert(&root, 30);
      insert(&root, 7);
      insert(&root, 17);

      printf("Traversal of the constructed B-tree is ");
      traverse(root);
      printf("\n");

      struct BTreeNode *result = search(root, 6);
      if (result != NULL) {
          printf("Node found\n");
      } else {
          printf("Node not found\n");
      }

      return 0;
  }
  ```

**과제:**
- B-트리에서 특정 값을 삽입하고 삭제하는 함수를 각각 구현
- B-트리에서 특정 값을 탐색하는 함수를 구현하고, 해당 값이 존재하는지 확인

**퀴즈 및 해설:**

1. **B-트리란 무엇인가요?**
   - B-트리는 자식 노드의 수가 가변적인 균형 트리로, 데이터베이스와 파일 시스템에서 대량의 데이터를 관리하는 데 사용됩니다. 모든 노드가 같은 레벨에 존재하며, 트리의 높이가 작아서 데이터 접근이 빠릅니다.

2. **B-트리의 최소 차수(T)란 무엇인가요?**
   - 최소 차수(T)는 B-트리의 각 노드가 가질 수 있는 최소 자식 노드의 수를 의미합니다. 각 노드는 최소 T-1개의 키를 가지며, 최대 2T-1개의 키를 가질 수 있습니다.

3. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    struct BTreeNode *root = createNode(1);

    insert(&root, 10);
    insert(&root, 20);
    insert(&root, 5);
    insert(&root, 6);
    insert(&root, 12);
    insert(&root, 30);
    insert(&root, 7);
    insert(&root, 17);

    printf("Traversal of the constructed B-tree is ");
    traverse(root);
    printf("\n");

    struct BTreeNode *result = search(root, 6);
    if (result != NULL) {
        printf("Node found\n");
    } else {
        printf("Node not found\n");
    }
    ```
   - 출력 결과:
     ```
     Traversal of the constructed B-tree is 5 6 7 10 12 17 20 30 
     Node found
     ```

**해설:**
1. **B-트리의 정의**는 자식 노드의 수가 가변적인 균형 트리로, 데이터베이스와 파일 시스템에서 대량의 데이터를 관리하는 데 사용됩니다. 모든 노드가 같은 레벨에 존재하며, 트리의 높이가 작아서 데이터 접근이 빠릅니다.
2. **B-트리의 최소 차수(T)**는 각 노드가 가질 수 있는 최소 자식 노드의 수를 의미합니다. 각 노드는 최소 T-1개의 키를 가지며, 최대 2T-1개의 키를 가질 수 있습니다.
3. **코드 출력 결과**는 B-트리의 삽입 연산 후 트리의 중위 순회를 통해 각 노드의 값을 출력하고, 탐색 연산을 통해 특정 값을 가진 노드를 찾습니다.

---

### 해시 테이블

**강의 내용:**
- 해시 테이블의 개념
  - 해시 테이블이란?
  - 해시 함수와 충돌 해결 방법
- 해시 테이블의 구조
  - 해시 함수 설계
  - 충돌 해결 방법 (체이닝, 개방 주소법)
- 해시 테이블의 연산
  - 삽입 연산
  - 삭제 연산
  - 탐색 연산

**실습:**
- 해시 테이블 구현 예제 (체이닝 방법)
  ```c
  #include <stdio.h>
  #include <stdlib.h>

  struct Node {
      int key;
      int value;
      struct Node* next;
  };

  struct HashTable {
      int size;
      struct Node** table;
  };

  struct HashTable* createTable(int size) {
      struct HashTable* newTable = (struct HashTable*) malloc(sizeof(struct HashTable));
      newTable->size = size;
      newTable->table = (

struct Node**) malloc(size * sizeof(struct Node*));
      for (int i = 0; i < size; i++)
          newTable->table[i] = NULL;
      return newTable;
  }

  int hashFunction(struct HashTable* hashtable, int key) {
      return key % hashtable->size;
  }

  void insert(struct HashTable* hashtable, int key, int value) {
      int hashIndex = hashFunction(hashtable, key);
      struct Node* newNode = (struct Node*) malloc(sizeof(struct Node));
      newNode->key = key;
      newNode->value = value;
      newNode->next = hashtable->table[hashIndex];
      hashtable->table[hashIndex] = newNode;
  }

  int search(struct HashTable* hashtable, int key) {
      int hashIndex = hashFunction(hashtable, key);
      struct Node* node = hashtable->table[hashIndex];
      while (node != NULL) {
          if (node->key == key)
              return node->value;
          node = node->next;
      }
      return -1;  // key not found
  }

  void delete(struct HashTable* hashtable, int key) {
      int hashIndex = hashFunction(hashtable, key);
      struct Node* node = hashtable->table[hashIndex];
      struct Node* prev = NULL;
      while (node != NULL && node->key != key) {
          prev = node;
          node = node->next;
      }
      if (node == NULL) return;
      if (prev == NULL)
          hashtable->table[hashIndex] = node->next;
      else
          prev->next = node->next;
      free(node);
  }

  void display(struct HashTable* hashtable) {
      for (int i = 0; i < hashtable->size; i++) {
          struct Node* node = hashtable->table[i];
          printf("Bucket %d: ", i);
          while (node != NULL) {
              printf("(%d, %d) -> ", node->key, node->value);
              node = node->next;
          }
          printf("NULL\n");
      }
  }

  int main() {
      struct HashTable* table = createTable(10);

      insert(table, 1, 10);
      insert(table, 2, 20);
      insert(table, 42, 30);
      insert(table, 4, 40);
      insert(table, 12, 50);

      printf("Hash table:\n");
      display(table);

      printf("Search key 2: %d\n", search(table, 2));
      printf("Search key 12: %d\n", search(table, 12));

      delete(table, 2);
      printf("Hash table after deletion:\n");
      display(table);

      return 0;
  }
  ```

**과제:**
- 해시 테이블에서 특정 값을 삽입하고 삭제하는 함수를 각각 구현
- 해시 테이블에서 특정 값을 탐색하는 함수를 구현하고, 해당 값이 존재하는지 확인

**퀴즈 및 해설:**

1. **해시 테이블이란 무엇인가요?**
   - 해시 테이블은 키-값 쌍을 저장하는 데이터 구조로, 해시 함수를 사용하여 데이터를 저장하고 검색하는 시간을 줄입니다. 각 키는 해시 함수를 통해 해시 값으로 변환되어 해시 테이블의 인덱스로 사용됩니다.

2. **해시 함수란 무엇인가요?**
   - 해시 함수는 임의의 데이터를 고정된 크기의 고유한 값(해시 값)으로 변환하는 함수입니다. 해시 함수는 해시 테이블에서 데이터를 효율적으로 저장하고 검색하는 데 사용됩니다.

3. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    struct HashTable* table = createTable(10);

    insert(table, 1, 10);
    insert(table, 2, 20);
    insert(table, 42, 30);
    insert(table, 4, 40);
    insert(table, 12, 50);

    printf("Hash table:\n");
    display(table);

    printf("Search key 2: %d\n", search(table, 2));
    printf("Search key 12: %d\n", search(table, 12));

    delete(table, 2);
    printf("Hash table after deletion:\n");
    display(table);
    ```
   - 출력 결과:
     ```
     Hash table:
     Bucket 0: NULL
     Bucket 1: (1, 10) -> NULL
     Bucket 2: (2, 20) -> (12, 50) -> NULL
     Bucket 3: NULL
     Bucket 4: (4, 40) -> NULL
     Bucket 5: NULL
     Bucket 6: NULL
     Bucket 7: NULL
     Bucket 8: NULL
     Bucket 9: (42, 30) -> NULL
     Search key 2: 20
     Search key 12: 50
     Hash table after deletion:
     Bucket 0: NULL
     Bucket 1: (1, 10) -> NULL
     Bucket 2: (12, 50) -> NULL
     Bucket 3: NULL
     Bucket 4: (4, 40) -> NULL
     Bucket 5: NULL
     Bucket 6: NULL
     Bucket 7: NULL
     Bucket 8: NULL
     Bucket 9: (42, 30) -> NULL
     ```

**해설:**
1. **해시 테이블의 정의**는 키-값 쌍을 저장하는 데이터 구조로, 해시 함수를 사용하여 데이터를 저장하고 검색하는 시간을 줄입니다. 각 키는 해시 함수를 통해 해시 값으로 변환되어 해시 테이블의 인덱스로 사용됩니다.
2. **해시 함수**는 임의의 데이터를 고정된 크기의 고유한 값(해시 값)으로 변환하는 함수입니다. 해시 함수는 해시 테이블에서 데이터를 효율적으로 저장하고 검색하는 데 사용됩니다.
3. **코드 출력 결과**는 해시 테이블의 삽입, 삭제, 탐색 연산 후의 상태를 보여줍니다. 해시 테이블의 각 버킷에는 체이닝 방식을 통해 충돌이 해결된 키-값 쌍이 저장됩니다.

---

