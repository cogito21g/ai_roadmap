### 6주차 강의 계획: 연결 리스트 (Linked List)

#### 강의 목표
- 연결 리스트의 개념과 필요성 이해
- 단일 연결 리스트, 이중 연결 리스트, 원형 연결 리스트 구현
- 연결 리스트의 장단점 및 활용 이해

#### 강의 구성
- **강의 시간**: 2시간
- **구성**: 연결 리스트 이론 (30분), 단일 연결 리스트 (30분), 이중 및 원형 연결 리스트 (30분), 실습 및 과제 안내 (30분)

#### 강의 내용

##### 1. 연결 리스트 이론 (30분)

###### 1.1 연결 리스트의 기본 개념
- **연결 리스트 개요**:
  - 연결 리스트의 정의
  - 배열과의 차이점
  - 노드 구조 (데이터와 포인터)

###### 1.2 연결 리스트의 장단점
- **장점**:
  - 동적 크기 조정
  - 빠른 삽입 및 삭제
- **단점**:
  - 인덱스를 통한 접근이 불가능
  - 추가적인 메모리 사용 (포인터)

##### 2. 단일 연결 리스트 (30분)

###### 2.1 단일 연결 리스트의 구조
- **단일 연결 리스트**:
  - 노드 구조 (데이터와 다음 노드에 대한 포인터)
  - 리스트의 처음과 끝

###### 2.2 단일 연결 리스트의 구현
- **노드 정의 및 기본 연산**:
  - 노드 삽입, 삭제, 탐색
- **예제**:
```cpp
#include <iostream>
using namespace std;

class Node {
public:
    int data;
    Node* next;
    Node(int data) {
        this->data = data;
        this->next = nullptr;
    }
};

class LinkedList {
public:
    Node* head;
    LinkedList() {
        head = nullptr;
    }

    void insert(int data) {
        Node* newNode = new Node(data);
        newNode->next = head;
        head = newNode;
    }

    void deleteNode(int key) {
        Node* temp = head;
        Node* prev = nullptr;

        if (temp != nullptr && temp->data == key) {
            head = temp->next;
            delete temp;
            return;
        }

        while (temp != nullptr && temp->data != key) {
            prev = temp;
            temp = temp->next;
        }

        if (temp == nullptr) return;

        prev->next = temp->next;
        delete temp;
    }

    void display() {
        Node* temp = head;
        while (temp != nullptr) {
            cout << temp->data << " ";
            temp = temp->next;
        }
        cout << endl;
    }
};

int main() {
    LinkedList list;
    list.insert(10);
    list.insert(20);
    list.insert(30);
    list.display();
    list.deleteNode(20);
    list.display();
    return 0;
}
```

##### 3. 이중 및 원형 연결 리스트 (30분)

###### 3.1 이중 연결 리스트
- **이중 연결 리스트 구조**:
  - 노드 구조 (데이터, 이전 노드 포인터, 다음 노드 포인터)
- **이중 연결 리스트 구현**:
  - 노드 삽입, 삭제, 탐색
- **예제**:
```cpp
#include <iostream>
using namespace std;

class DNode {
public:
    int data;
    DNode* next;
    DNode* prev;
    DNode(int data) {
        this->data = data;
        this->next = nullptr;
        this->prev = nullptr;
    }
};

class DoublyLinkedList {
public:
    DNode* head;
    DoublyLinkedList() {
        head = nullptr;
    }

    void insert(int data) {
        DNode* newNode = new DNode(data);
        newNode->next = head;
        if (head != nullptr) {
            head->prev = newNode;
        }
        head = newNode;
    }

    void deleteNode(int key) {
        DNode* temp = head;
        while (temp != nullptr && temp->data != key) {
            temp = temp->next;
        }
        if (temp == nullptr) return;

        if (temp->prev != nullptr) {
            temp->prev->next = temp->next;
        } else {
            head = temp->next;
        }

        if (temp->next != nullptr) {
            temp->next->prev = temp->prev;
        }

        delete temp;
    }

    void display() {
        DNode* temp = head;
        while (temp != nullptr) {
            cout << temp->data << " ";
            temp = temp->next;
        }
        cout << endl;
    }
};

int main() {
    DoublyLinkedList list;
    list.insert(10);
    list.insert(20);
    list.insert(30);
    list.display();
    list.deleteNode(20);
    list.display();
    return 0;
}
```

###### 3.2 원형 연결 리스트
- **원형 연결 리스트 구조**:
  - 마지막 노드가 첫 번째 노드를 가리킴
- **원형 연결 리스트 구현**:
  - 노드 삽입, 삭제, 탐색
- **예제**:
```cpp
#include <iostream>
using namespace std;

class CNode {
public:
    int data;
    CNode* next;
    CNode(int data) {
        this->data = data;
        this->next = this;
    }
};

class CircularLinkedList {
public:
    CNode* head;
    CircularLinkedList() {
        head = nullptr;
    }

    void insert(int data) {
        CNode* newNode = new CNode(data);
        if (head == nullptr) {
            head = newNode;
        } else {
            CNode* temp = head;
            while (temp->next != head) {
                temp = temp->next;
            }
            temp->next = newNode;
            newNode->next = head;
        }
    }

    void deleteNode(int key) {
        if (head == nullptr) return;

        CNode* temp = head;
        CNode* prev = nullptr;

        if (head->data == key) {
            while (temp->next != head) {
                temp = temp->next;
            }
            temp->next = head->next;
            delete head;
            head = temp->next;
            return;
        }

        while (temp->next != head && temp->data != key) {
            prev = temp;
            temp = temp->next;
        }

        if (temp->data == key) {
            prev->next = temp->next;
            delete temp;
        }
    }

    void display() {
        if (head == nullptr) return;

        CNode* temp = head;
        do {
            cout << temp->data << " ";
            temp = temp->next;
        } while (temp != head);
        cout << endl;
    }
};

int main() {
    CircularLinkedList list;
    list.insert(10);
    list.insert(20);
    list.insert(30);
    list.display();
    list.deleteNode(20);
    list.display();
    return 0;
}
```

##### 4. 실습 및 과제 안내 (30분)

###### 4.1 실습
- **실습 목표**:
  - 연결 리스트를 사용한 프로그램 작성
- **실습 문제**:
  - 단일 연결 리스트, 이중 연결 리스트, 원형 연결 리스트를 구현하고, 각 리스트에 데이터를 삽입, 삭제, 출력하는 프로그램 작성

###### 4.2 과제 안내
- **과제 내용**:
  - 단일 연결 리스트를 사용하여 역순으로 데이터를 출력하는 프로그램 작성
  - 이중 연결 리스트를 사용하여 데이터를 오름차순으로 정렬하는 프로그램 작성
  - 원형 연결 리스트를 사용하여 큐를 구현하고, 큐의 기본 연산 (삽입, 삭제, 출력)을 수행하는 프로그램 작성
- **과제 제출 방법**:
  - 작성한 코드를 주차별 과제 제출 폴더에 업로드
  - 제출 기한: 다음 주 강의 전까지

##### 5. 퀴즈 및 해설 (30분)

###### 퀴즈 문제:
1. 단일 연결 리스트에서 노드 삽입 시 가장 빠른 위치는?
   - a) 리스트의 끝
   - b) 리스트의 중간
   - c) 리스트의 시작
   - d) 리스트의 임의 위치
2. 이중 연결 리스트의 각 노드는 몇 개의 포인터를 가지고 있는가?
   - a) 1
   - b) 2
   - c) 3
   - d) 4
3. 원형 연결 리스트의 특징은?
   - a) 마지막 노드가 첫 번째 노드를 가리킨다
   - b) 첫 번째 노드가 마지막 노드를 가리킨다
   - c) 모든 노드가 자기 자신을 가리킨다
   - d) 노드가 하나도 없다

###### 퀴즈 해설:
1. **정답: c) 리스트의 시작**
   - 단일 연결 리스트에서 새로운 노드를 삽입할 때 가장 빠른 위치는 리스트의 시작 부분입니다.
2. **정답: b) 2**
   - 이중 연결 리스트의 각 노드는 두 개의 포인터를 가지고

 있습니다. 하나는 다음 노드를 가리키고, 다른 하나는 이전 노드를 가리킵니다.
3. **정답: a) 마지막 노드가 첫 번째 노드를 가리킨다**
   - 원형 연결 리스트는 마지막 노드가 첫 번째 노드를 가리키는 구조를 가지고 있습니다.

### 준비 자료
- 강의 슬라이드
- 실습 코드 예제
- 과제 안내서

### 과제 해설

##### 과제: 단일 연결 리스트를 사용하여 역순으로 데이터를 출력하는 프로그램 작성
- **문제**: 단일 연결 리스트를 사용하여 데이터를 역순으로 출력하는 프로그램 작성
- **해설**:
  - 단일 연결 리스트에서 역순으로 데이터를 출력하기 위해 재귀함수를 사용하거나 스택을 활용합니다.

```cpp
#include <iostream>
#include <stack>
using namespace std;

class Node {
public:
    int data;
    Node* next;
    Node(int data) {
        this->data = data;
        this->next = nullptr;
    }
};

class LinkedList {
public:
    Node* head;
    LinkedList() {
        head = nullptr;
    }

    void insert(int data) {
        Node* newNode = new Node(data);
        newNode->next = head;
        head = newNode;
    }

    void displayReverse(Node* node) {
        if (node == nullptr) return;
        displayReverse(node->next);
        cout << node->data << " ";
    }

    void displayReverse() {
        displayReverse(head);
        cout << endl;
    }
};

int main() {
    LinkedList list;
    list.insert(10);
    list.insert(20);
    list.insert(30);
    list.displayReverse();
    return 0;
}
```
- **설명**:
  - `displayReverse` 함수를 재귀적으로 호출하여 역순으로 데이터를 출력합니다.

##### 과제: 이중 연결 리스트를 사용하여 데이터를 오름차순으로 정렬하는 프로그램 작성
- **문제**: 이중 연결 리스트를 사용하여 데이터를 오름차순으로 정렬하는 프로그램 작성
- **해설**:
  - 이중 연결 리스트의 삽입 정렬을 사용하여 데이터를 오름차순으로 정렬합니다.

```cpp
#include <iostream>
using namespace std;

class DNode {
public:
    int data;
    DNode* next;
    DNode* prev;
    DNode(int data) {
        this->data = data;
        this->next = nullptr;
        this->prev = nullptr;
    }
};

class DoublyLinkedList {
public:
    DNode* head;
    DoublyLinkedList() {
        head = nullptr;
    }

    void insertSorted(int data) {
        DNode* newNode = new DNode(data);
        if (head == nullptr || head->data >= data) {
            newNode->next = head;
            if (head != nullptr) {
                head->prev = newNode;
            }
            head = newNode;
        } else {
            DNode* temp = head;
            while (temp->next != nullptr && temp->next->data < data) {
                temp = temp->next;
            }
            newNode->next = temp->next;
            if (temp->next != nullptr) {
                temp->next->prev = newNode;
            }
            temp->next = newNode;
            newNode->prev = temp;
        }
    }

    void display() {
        DNode* temp = head;
        while (temp != nullptr) {
            cout << temp->data << " ";
            temp = temp->next;
        }
        cout << endl;
    }
};

int main() {
    DoublyLinkedList list;
    list.insertSorted(30);
    list.insertSorted(20);
    list.insertSorted(40);
    list.insertSorted(10);
    list.display();
    return 0;
}
```
- **설명**:
  - `insertSorted` 함수는 새로운 노드를 삽입할 때 정렬된 위치에 삽입합니다.

##### 과제: 원형 연결 리스트를 사용하여 큐를 구현하고, 큐의 기본 연산 (삽입, 삭제, 출력)을 수행하는 프로그램 작성
- **문제**: 원형 연결 리스트를 사용하여 큐를 구현하고, 큐의 삽입, 삭제, 출력 기능을 구현
- **해설**:
  - 원형 연결 리스트를 사용하여 큐를 구현하고, `enqueue`, `dequeue`, `display` 함수를 작성합니다.

```cpp
#include <iostream>
using namespace std;

class CNode {
public:
    int data;
    CNode* next;
    CNode(int data) {
        this->data = data;
        this->next = nullptr;
    }
};

class CircularQueue {
public:
    CNode* front;
    CNode* rear;
    CircularQueue() {
        front = rear = nullptr;
    }

    void enqueue(int data) {
        CNode* newNode = new CNode(data);
        if (front == nullptr) {
            front = rear = newNode;
            rear->next = front;
        } else {
            rear->next = newNode;
            rear = newNode;
            rear->next = front;
        }
    }

    void dequeue() {
        if (front == nullptr) {
            cout << "Queue is empty" << endl;
            return;
        }
        if (front == rear) {
            delete front;
            front = rear = nullptr;
        } else {
            CNode* temp = front;
            front = front->next;
            rear->next = front;
            delete temp;
        }
    }

    void display() {
        if (front == nullptr) {
            cout << "Queue is empty" << endl;
            return;
        }
        CNode* temp = front;
        do {
            cout << temp->data << " ";
            temp = temp->next;
        } while (temp != front);
        cout << endl;
    }
};

int main() {
    CircularQueue queue;
    queue.enqueue(10);
    queue.enqueue(20);
    queue.enqueue(30);
    queue.display();
    queue.dequeue();
    queue.display();
    return 0;
}
```
- **설명**:
  - `enqueue` 함수는 큐에 요소를 삽입하며, 원형 연결 리스트의 구조를 유지합니다.
  - `dequeue` 함수는 큐에서 요소를 제거하며, 원형 연결 리스트의 구조를 유지합니다.
  - `display` 함수는 큐의 요소를 출력합니다.

이로써 6주차 강의가 마무리됩니다. 학생들은 연결 리스트의 다양한 형태와 구현 방법을 이해하고, 이를 활용하여 다양한 프로그램을 작성하는 능력을 기르게 됩니다.