### 자료구조 교육과정 - 2주차: 연결 리스트

**강의 목표:**
연결 리스트의 기본 개념을 이해하고, 연결 리스트의 구현 및 기본 연산을 학습합니다.

**강의 구성:**

#### 2. 연결 리스트

**강의 내용:**
- 연결 리스트의 개념
  - 연결 리스트의 정의 및 특징
  - 연결 리스트의 장단점
- 연결 리스트의 기본 연산
  - 노드 삽입, 삭제, 검색

**실습:**
- 연결 리스트 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>

  struct Node {
      int data;
      struct Node* next;
  };

  void printList(struct Node* n) {
      while (n != NULL) {
          printf("%d ", n->data);
          n = n->next;
      }
      printf("\n");
  }

  void push(struct Node** head_ref, int new_data) {
      struct Node* new_node = (struct Node*) malloc(sizeof(struct Node));
      new_node->data = new_data;
      new_node->next = (*head_ref);
      (*head_ref) = new_node;
  }

  void deleteNode(struct Node** head_ref, int key) {
      struct Node* temp = *head_ref, *prev;

      if (temp != NULL && temp->data == key) {
          *head_ref = temp->next;
          free(temp);
          return;
      }

      while (temp != NULL && temp->data != key) {
          prev = temp;
          temp = temp->next;
      }

      if (temp == NULL) return;

      prev->next = temp->next;
      free(temp);
  }

  int main() {
      struct Node* head = NULL;

      push(&head, 1);
      push(&head, 2);
      push(&head, 3);

      printf("Created Linked list: ");
      printList(head);

      deleteNode(&head, 2);
      printf("Linked list after deletion: ");
      printList(head);

      return 0;
  }
  ```

**과제:**
- 연결 리스트에서 특정 위치에 노드를 삽입하는 함수 구현
- 연결 리스트에서 특정 값을 가지는 노드를 삭제하는 함수 구현

**퀴즈 및 해설:**

1. **연결 리스트의 장점과 단점은 무엇인가요?**
   - **장점:**
     - 동적 크기 조정이 가능
     - 메모리 효율적 사용 가능 (필요한 만큼만 할당)
   - **단점:**
     - 인덱스 접근이 불가능 (O(n))
     - 추가적인 메모리 사용 (포인터 저장 공간)

2. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    struct Node* head = NULL;
    push(&head, 1);
    push(&head, 2);
    push(&head, 3);
    printList(head);
    deleteNode(&head, 2);
    printList(head);
    ```
   - 출력 결과:
     ```
     Created Linked list: 3 2 1
     Linked list after deletion: 3 1
     ```

**해설:**
1. **연결 리스트의 장점과 단점**은 동적 메모리 할당과 관련이 있습니다. 연결 리스트는 필요한 만큼 메모리를 동적으로 할당할 수 있어 메모리 사용이 효율적이지만, 인덱스 접근이 불가능하고 각 노드가 추가적인 포인터 공간을 차지합니다.
2. **코드 출력 결과**는 함수 `push`와 `deleteNode`의 동작을 통해 연결 리스트의 변화를 보여줍니다. `push` 함수는 리스트의 맨 앞에 새로운 노드를 추가하고, `deleteNode` 함수는 리스트에서 특정 값을 가지는 노드를 삭제합니다.

---
