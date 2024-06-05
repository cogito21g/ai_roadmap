### C 언어 20주차 심화 교육과정 - 11주차: 데이터 구조 기초

#### 11주차: 데이터 구조 기초

**강의 목표:**
11주차의 목표는 기본적인 데이터 구조인 연결 리스트, 스택, 큐의 개념과 구현 방법을 이해하고, 이들을 활용하여 다양한 문제를 해결하는 능력을 기르는 것입니다.

**강의 구성:**

##### 1. 연결 리스트 (Linked List)
- **강의 내용:**
  - 연결 리스트의 개념
    - 노드와 포인터로 구성된 데이터 구조
    - 각 노드는 데이터와 다음 노드의 주소를 가짐
  - 단일 연결 리스트의 구현
    - 노드 구조체 정의
    - 노드 삽입, 삭제, 탐색 방법
  - 단일 연결 리스트와 배열의 비교
    - 메모리 사용, 삽입/삭제 성능 비교
- **실습:**
  - 단일 연결 리스트 구현 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>

    struct Node {
        int data;
        struct Node* next;
    };

    void printList(struct Node* n) {
        while (n != NULL) {
            printf("%d -> ", n->data);
            n = n->next;
        }
        printf("NULL\n");
    }

    void insertAtBeginning(struct Node** head_ref, int new_data) {
        struct Node* new_node = (struct Node*)malloc(sizeof(struct Node));
        new_node->data = new_data;
        new_node->next = (*head_ref);
        (*head_ref) = new_node;
    }

    int main() {
        struct Node* head = NULL;
        insertAtBeginning(&head, 1);
        insertAtBeginning(&head, 2);
        insertAtBeginning(&head, 3);

        printList(head);

        return 0;
    }
    ```

##### 2. 스택 (Stack)
- **강의 내용:**
  - 스택의 개념
    - LIFO (Last In, First Out) 구조
  - 스택의 주요 연산
    - `push`, `pop`, `peek`, `isEmpty`
  - 배열을 이용한 스택 구현
  - 연결 리스트를 이용한 스택 구현
- **실습:**
  - 배열을 이용한 스택 구현 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>

    #define MAX 1000

    struct Stack {
        int top;
        int items[MAX];
    };

    void initStack(struct Stack* s) {
        s->top = -1;
    }

    int isFull(struct Stack* s) {
        return s->top == MAX - 1;
    }

    int isEmpty(struct Stack* s) {
        return s->top == -1;
    }

    void push(struct Stack* s, int item) {
        if (isFull(s)) {
            printf("Stack overflow\n");
            return;
        }
        s->items[++s->top] = item;
    }

    int pop(struct Stack* s) {
        if (isEmpty(s)) {
            printf("Stack underflow\n");
            return -1;
        }
        return s->items[s->top--];
    }

    int peek(struct Stack* s) {
        if (isEmpty(s)) {
            printf("Stack is empty\n");
            return -1;
        }
        return s->items[s->top];
    }

    int main() {
        struct Stack s;
        initStack(&s);

        push(&s, 10);
        push(&s, 20);
        push(&s, 30);

        printf("Top element is %d\n", peek(&s));
        printf("Stack elements:\n");
        while (!isEmpty(&s)) {
            printf("%d\n", pop(&s));
        }

        return 0;
    }
    ```

  - 연결 리스트를 이용한 스택 구현 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>

    struct Node {
        int data;
        struct Node* next;
    };

    void push(struct Node** top_ref, int new_data) {
        struct Node* new_node = (struct Node*)malloc(sizeof(struct Node));
        if (new_node == NULL) {
            printf("Stack overflow\n");
            return;
        }
        new_node->data = new_data;
        new_node->next = (*top_ref);
        (*top_ref) = new_node;
    }

    int pop(struct Node** top_ref) {
        if (*top_ref == NULL) {
            printf("Stack underflow\n");
            return -1;
        }
        struct Node* temp = *top_ref;
        *top_ref = (*top_ref)->next;
        int popped = temp->data;
        free(temp);
        return popped;
    }

    int peek(struct Node* top) {
        if (top == NULL) {
            printf("Stack is empty\n");
            return -1;
        }
        return top->data;
    }

    int main() {
        struct Node* stack = NULL;

        push(&stack, 10);
        push(&stack, 20);
        push(&stack, 30);

        printf("Top element is %d\n", peek(stack));
        printf("Stack elements:\n");
        while (stack != NULL) {
            printf("%d\n", pop(&stack));
        }

        return 0;
    }
    ```

##### 3. 큐 (Queue)
- **강의 내용:**
  - 큐의 개념
    - FIFO (First In, First Out) 구조
  - 큐의 주요 연산
    - `enqueue`, `dequeue`, `front`, `rear`, `isEmpty`
  - 배열을 이용한 큐 구현
  - 연결 리스트를 이용한 큐 구현
- **실습:**
  - 배열을 이용한 큐 구현 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>

    #define MAX 1000

    struct Queue {
        int front, rear, size;
        unsigned capacity;
        int* array;
    };

    struct Queue* createQueue(unsigned capacity) {
        struct Queue* queue = (struct Queue*)malloc(sizeof(struct Queue));
        queue->capacity = capacity;
        queue->front = queue->size = 0;
        queue->rear = capacity - 1;
        queue->array = (int*)malloc(queue->capacity * sizeof(int));
        return queue;
    }

    int isFull(struct Queue* queue) {
        return (queue->size == queue->capacity);
    }

    int isEmpty(struct Queue* queue) {
        return (queue->size == 0);
    }

    void enqueue(struct Queue* queue, int item) {
        if (isFull(queue))
            return;
        queue->rear = (queue->rear + 1) % queue->capacity;
        queue->array[queue->rear] = item;
        queue->size = queue->size + 1;
    }

    int dequeue(struct Queue* queue) {
        if (isEmpty(queue))
            return -1;
        int item = queue->array[queue->front];
        queue->front = (queue->front + 1) % queue->capacity;
        queue->size = queue->size - 1;
        return item;
    }

    int front(struct Queue* queue) {
        if (isEmpty(queue))
            return -1;
        return queue->array[queue->front];
    }

    int rear(struct Queue* queue) {
        if (isEmpty(queue))
            return -1;
        return queue->array[queue->rear];
    }

    int main() {
        struct Queue* queue = createQueue(MAX);

        enqueue(queue, 10);
        enqueue(queue, 20);
        enqueue(queue, 30);

        printf("Front element is %d\n", front(queue));
        printf("Queue elements:\n");
        while (!isEmpty(queue)) {
            printf("%d\n", dequeue(queue));
        }

        return 0;
    }
    ```

  - 연결 리스트를 이용한 큐 구현 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>

    struct Node {
        int data;
        struct Node* next;
    };

    struct Queue {
        struct Node *front, *rear;
    };

    struct Queue* createQueue() {
        struct Queue* queue = (struct Queue*)malloc(sizeof(struct Queue));
        queue->front = queue->rear = NULL;
        return queue;
    }

    void enqueue(struct Queue* queue, int item) {
        struct Node* temp = (struct Node*)malloc(sizeof(struct Node));
        temp->data = item;
        temp->next = NULL;
        if (queue->rear == NULL) {
            queue->front = queue->rear = temp;
            return;
        }
        queue->rear->next = temp;
        queue->rear = temp;
    }

    int dequeue(struct Queue* queue) {
        if (queue->front == NULL)
            return -1;
        struct Node* temp = queue->front;
        queue->front = queue->front->next;
        if (queue->front == NULL)
            queue->rear = NULL;
        int item = temp->data;
        free(temp);
        return item;
    }

    int front(struct Queue* queue) {
        if (queue->front == NULL)
            return -1;
        return queue->front->data;
    }

    int rear(struct Queue*

 queue) {
        if (queue->rear == NULL)
            return -1;
        return queue->rear->data;
    }

    int main() {
        struct Queue* queue = createQueue();

        enqueue(queue, 10);
        enqueue(queue, 20);
        enqueue(queue, 30);

        printf("Front element is %d\n", front(queue));
        printf("Queue elements:\n");
        while (queue->front != NULL) {
            printf("%d\n", dequeue(queue));
        }

        return 0;
    }
    ```

**과제:**
11주차 과제는 다음과 같습니다.
- 단일 연결 리스트를 구현하여, 사용자가 입력한 데이터를 삽입, 삭제, 탐색하는 프로그램 작성
- 배열을 이용하여 스택을 구현하고, 사용자가 입력한 정수를 스택에 `push`하고 `pop`하는 프로그램 작성
- 연결 리스트를 이용하여 큐를 구현하고, 사용자가 입력한 정수를 큐에 `enqueue`하고 `dequeue`하는 프로그램 작성

**퀴즈 및 해설:**

1. **단일 연결 리스트와 배열의 차이점은 무엇인가요?**
   - 단일 연결 리스트는 노드와 포인터로 구성된 데이터 구조로, 삽입과 삭제가 빠르지만, 임의 접근이 느립니다. 배열은 고정된 크기의 연속된 메모리 공간으로, 임의 접근이 빠르지만, 삽입과 삭제가 느립니다.

2. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    struct Node {
        int data;
        struct Node* next;
    };

    int main() {
        struct Node* head = NULL;
        struct Node* second = NULL;
        struct Node* third = NULL;

        head = (struct Node*)malloc(sizeof(struct Node));
        second = (struct Node*)malloc(sizeof(struct Node));
        third = (struct Node*)malloc(sizeof(struct Node));

        head->data = 1;
        head->next = second;
        second->data = 2;
        second->next = third;
        third->data = 3;
        third->next = NULL;

        struct Node* temp = head;
        while (temp != NULL) {
            printf("%d ", temp->data);
            temp = temp->next;
        }

        return 0;
    }
    ```
   - 출력 결과는 `1 2 3`입니다. 연결 리스트의 각 노드를 순서대로 출력합니다.

3. **스택과 큐의 차이점은 무엇인가요?**
   - 스택은 LIFO(Last In, First Out) 구조로, 가장 마지막에 삽입된 요소가 가장 먼저 삭제됩니다. 큐는 FIFO(First In, First Out) 구조로, 가장 먼저 삽입된 요소가 가장 먼저 삭제됩니다.

4. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    struct Stack {
        int top;
        int items[100];
    };

    void push(struct Stack* s, int item) {
        s->items[++(s->top)] = item;
    }

    int pop(struct Stack* s) {
        return s->items[(s->top)--];
    }

    int main() {
        struct Stack s;
        s.top = -1;

        push(&s, 1);
        push(&s, 2);
        push(&s, 3);

        printf("%d ", pop(&s));
        printf("%d ", pop(&s));
        printf("%d ", pop(&s));

        return 0;
    }
    ```
   - 출력 결과는 `3 2 1`입니다. 스택에 1, 2, 3을 차례대로 `push`하고, `pop`하면 마지막에 삽입된 3부터 먼저 출력됩니다.

5. **큐의 주요 연산은 무엇인가요?**
   - 큐의 주요 연산은 `enqueue`(삽입), `dequeue`(삭제), `front`(첫 번째 요소 확인), `rear`(마지막 요소 확인), `isEmpty`(큐가 비어 있는지 확인)입니다.

**해설:**
1. 단일 연결 리스트는 삽입과 삭제가 빠르지만, 임의 접근이 느립니다. 배열은 임의 접근이 빠르지만, 삽입과 삭제가 느립니다.
2. 연결 리스트의 각 노드를 순서대로 출력하므로, 출력 결과는 `1 2 3`입니다.
3. 스택은 LIFO 구조로, 큐는 FIFO 구조로 동작합니다.
4. 스택에 1, 2, 3을 차례대로 `push`하고, `pop`하면 마지막에 삽입된 3부터 먼저 출력되므로, 출력 결과는 `3 2 1`입니다.
5. 큐의 주요 연산은 `enqueue`(삽입), `dequeue`(삭제), `front`(첫 번째 요소 확인), `rear`(마지막 요소 확인), `isEmpty`(큐가 비어 있는지 확인)입니다.

이 11주차 강의는 학생들이 기본적인 데이터 구조인 연결 리스트, 스택, 큐의 개념과 구현 방법을 이해하고, 이들을 활용하여 다양한 문제를 해결하는 능력을 기를 수 있도록 도와줍니다.