### 8주차 강의 계획: 큐 (Queue)

#### 강의 목표
- 큐의 기본 개념과 활용 이해
- 배열 기반 큐와 연결 리스트 기반 큐 구현
- 원형 큐 및 우선순위 큐 이해 및 구현

#### 강의 구성
- **강의 시간**: 2시간
- **구성**: 큐 이론 (30분), 배열 기반 큐 (30분), 연결 리스트 기반 큐 및 원형 큐 (30분), 실습 및 과제 안내 (30분)

#### 강의 내용

##### 1. 큐 이론 (30분)

###### 1.1 큐의 기본 개념
- **큐 개요**:
  - FIFO (First In, First Out) 구조
  - 주요 연산: `enqueue`, `dequeue`, `peek`, `isEmpty`
- **큐의 활용 사례**:
  - 프로세스 관리
  - 데이터 버퍼
  - 너비 우선 탐색 (BFS) 알고리즘

###### 1.2 큐의 장단점
- **장점**:
  - 간단한 구현
  - FIFO 구조로 특정 응용에 유용
- **단점**:
  - 고정된 크기를 가지면 메모리 낭비 가능 (배열 기반)

##### 2. 배열 기반 큐 (30분)

###### 2.1 배열 기반 큐의 구조
- **배열 기반 큐 구현**:
  - 고정 크기의 배열 사용
  - 두 개의 포인터 (front, rear)로 큐의 앞과 뒤를 관리

###### 2.2 배열 기반 큐의 구현
- **큐의 주요 연산 구현**:
  - `enqueue`, `dequeue`, `peek`, `isEmpty`, `isFull`
- **예제**:
```cpp
#include <iostream>
using namespace std;

class Queue {
private:
    int* arr;
    int front;
    int rear;
    int capacity;
    int count;

public:
    Queue(int size) {
        arr = new int[size];
        capacity = size;
        front = 0;
        rear = -1;
        count = 0;
    }

    ~Queue() {
        delete[] arr;
    }

    void enqueue(int x) {
        if (isFull()) {
            cout << "Queue Overflow\n";
            return;
        }
        rear = (rear + 1) % capacity;
        arr[rear] = x;
        count++;
    }

    int dequeue() {
        if (isEmpty()) {
            cout << "Queue Underflow\n";
            return -1;
        }
        int x = arr[front];
        front = (front + 1) % capacity;
        count--;
        return x;
    }

    int peek() {
        if (isEmpty()) {
            cout << "Queue is empty\n";
            return -1;
        }
        return arr[front];
    }

    bool isEmpty() {
        return count == 0;
    }

    bool isFull() {
        return count == capacity;
    }

    int size() {
        return count;
    }

    void display() {
        if (isEmpty()) {
            cout << "Queue is empty\n";
            return;
        }
        for (int i = 0; i < count; ++i) {
            cout << arr[(front + i) % capacity] << " ";
        }
        cout << endl;
    }
};

int main() {
    Queue q(5);
    q.enqueue(10);
    q.enqueue(20);
    q.enqueue(30);
    q.display();
    cout << "Front element: " << q.peek() << endl;
    q.dequeue();
    q.display();
    return 0;
}
```

##### 3. 연결 리스트 기반 큐 및 원형 큐 (30분)

###### 3.1 연결 리스트 기반 큐의 구조
- **연결 리스트 기반 큐 구현**:
  - 동적 크기의 연결 리스트 사용
  - 노드를 사용하여 큐의 요소 관리

###### 3.2 연결 리스트 기반 큐의 구현
- **큐의 주요 연산 구현**:
  - `enqueue`, `dequeue`, `peek`, `isEmpty`
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

class Queue {
private:
    Node* front;
    Node* rear;

public:
    Queue() {
        front = rear = nullptr;
    }

    ~Queue() {
        while (!isEmpty()) {
            dequeue();
        }
    }

    void enqueue(int x) {
        Node* newNode = new Node(x);
        if (isEmpty()) {
            front = rear = newNode;
        } else {
            rear->next = newNode;
            rear = newNode;
        }
    }

    int dequeue() {
        if (isEmpty()) {
            cout << "Queue Underflow\n";
            return -1;
        }
        Node* temp = front;
        int data = front->data;
        front = front->next;
        delete temp;
        if (front == nullptr) {
            rear = nullptr;
        }
        return data;
    }

    int peek() {
        if (isEmpty()) {
            cout << "Queue is empty\n";
            return -1;
        }
        return front->data;
    }

    bool isEmpty() {
        return front == nullptr;
    }

    void display() {
        if (isEmpty()) {
            cout << "Queue is empty\n";
            return;
        }
        Node* temp = front;
        while (temp != nullptr) {
            cout << temp->data << " ";
            temp = temp->next;
        }
        cout << endl;
    }
};

int main() {
    Queue q;
    q.enqueue(10);
    q.enqueue(20);
    q.enqueue(30);
    q.display();
    cout << "Front element: " << q.peek() << endl;
    q.dequeue();
    q.display();
    return 0;
}
```

###### 3.3 원형 큐의 구조 및 구현
- **원형 큐**:
  - 배열 기반 큐의 한계를 극복
  - 큐의 끝과 시작이 연결된 구조

- **원형 큐 구현**:
  - 배열 기반 큐와 유사하지만 포인터 연산이 원형으로 연결
  - `enqueue`, `dequeue`, `peek`, `isEmpty`, `isFull` 구현
- **예제**:
```cpp
#include <iostream>
using namespace std;

class CircularQueue {
private:
    int* arr;
    int front;
    int rear;
    int capacity;
    int count;

public:
    CircularQueue(int size) {
        arr = new int[size];
        capacity = size;
        front = 0;
        rear = -1;
        count = 0;
    }

    ~CircularQueue() {
        delete[] arr;
    }

    void enqueue(int x) {
        if (isFull()) {
            cout << "Queue Overflow\n";
            return;
        }
        rear = (rear + 1) % capacity;
        arr[rear] = x;
        count++;
    }

    int dequeue() {
        if (isEmpty()) {
            cout << "Queue Underflow\n";
            return -1;
        }
        int x = arr[front];
        front = (front + 1) % capacity;
        count--;
        return x;
    }

    int peek() {
        if (isEmpty()) {
            cout << "Queue is empty\n";
            return -1;
        }
        return arr[front];
    }

    bool isEmpty() {
        return count == 0;
    }

    bool isFull() {
        return count == capacity;
    }

    int size() {
        return count;
    }

    void display() {
        if (isEmpty()) {
            cout << "Queue is empty\n";
            return;
        }
        for (int i = 0; i < count; ++i) {
            cout << arr[(front + i) % capacity] << " ";
        }
        cout << endl;
    }
};

int main() {
    CircularQueue q(5);
    q.enqueue(10);
    q.enqueue(20);
    q.enqueue(30);
    q.display();
    cout << "Front element: " << q.peek() << endl;
    q.dequeue();
    q.display();
    return 0;
}
```

##### 4. 실습 및 과제 안내 (30분)

###### 4.1 실습
- **실습 목표**:
  - 큐를 사용한 프로그램 작성
- **실습 문제**:
  - 배열 기반 큐와 연결 리스트 기반 큐, 원형 큐를 구현하고, 각 큐에 데이터를 삽입, 삭제, 출력하는 프로그램 작성

###### 4.2 과제 안내
- **과제 내용**:
  - 큐를 사용하여 은행 대기열 시뮬레이션 프로그램 작성
  - 우선순위 큐를 사용하여 작업 스케줄러 프로그램 작성
- **과제 제출 방법**:
  - 작성한 코드를 주차별 과제 제출 폴더에 업로드
  - 제출 기한: 다음 주 강의 전까지

##### 5. 퀴즈 및 해설 (30분)

###### 퀴즈 문제:
1. 큐의 기본 연산이 아닌 것은?


   - a) enqueue
   - b) dequeue
   - c) peek
   - d) search
2. 큐의 구조는 어떤 방식으로 작동하는가?
   - a) FIFO
   - b) LIFO
   - c) LILO
   - d) FILO
3. 연결 리스트 기반 큐에서 노드를 추가할 때 추가되는 위치는?
   - a) 리스트의 시작
   - b) 리스트의 중간
   - c) 리스트의 끝
   - d) 리스트의 임의 위치

###### 퀴즈 해설:
1. **정답: d) search**
   - 큐의 기본 연산은 `enqueue`, `dequeue`, `peek`, `isEmpty`입니다. `search`는 기본 연산이 아닙니다.
2. **정답: a) FIFO**
   - 큐는 FIFO (First In, First Out) 구조로 작동합니다.
3. **정답: c) 리스트의 끝**
   - 연결 리스트 기반 큐에서 노드는 리스트의 끝 부분에 추가됩니다.

### 준비 자료
- 강의 슬라이드
- 실습 코드 예제
- 과제 안내서

### 과제 해설

##### 과제: 큐를 사용하여 은행 대기열 시뮬레이션 프로그램 작성
- **문제**: 큐를 사용하여 은행 대기열 시뮬레이션 프로그램 작성
- **해설**:
  - 큐를 사용하여 은행 대기열을 관리하고, 고객이 도착하고 처리되는 순서를 시뮬레이션합니다.

```cpp
#include <iostream>
#include <queue>
using namespace std;

class Customer {
public:
    int id;
    string name;
    Customer(int id, string name) : id(id), name(name) {}
};

int main() {
    queue<Customer> q;
    q.push(Customer(1, "Alice"));
    q.push(Customer(2, "Bob"));
    q.push(Customer(3, "Charlie"));

    while (!q.empty()) {
        Customer c = q.front();
        cout << "Processing customer ID: " << c.id << ", Name: " << c.name << endl;
        q.pop();
    }
    return 0;
}
```
- **설명**:
  - `queue<Customer>`를 사용하여 은행 대기열을 관리합니다.
  - 고객이 도착할 때 `enqueue` 연산을 사용하고, 처리될 때 `dequeue` 연산을 사용합니다.

##### 과제: 우선순위 큐를 사용하여 작업 스케줄러 프로그램 작성
- **문제**: 우선순위 큐를 사용하여 작업 스케줄러 프로그램 작성
- **해설**:
  - 우선순위 큐를 사용하여 작업의 우선순위에 따라 처리 순서를 결정합니다.

```cpp
#include <iostream>
#include <queue>
#include <vector>
using namespace std;

class Task {
public:
    int priority;
    string description;
    Task(int priority, string description) : priority(priority), description(description) {}
};

struct CompareTask {
    bool operator()(Task const& t1, Task const& t2) {
        return t1.priority > t2.priority;
    }
};

int main() {
    priority_queue<Task, vector<Task>, CompareTask> pq;
    pq.push(Task(3, "Task 1"));
    pq.push(Task(1, "Task 2"));
    pq.push(Task(2, "Task 3"));

    while (!pq.empty()) {
        Task t = pq.top();
        cout << "Processing task: " << t.description << " with priority " << t.priority << endl;
        pq.pop();
    }
    return 0;
}
```
- **설명**:
  - `priority_queue<Task, vector<Task>, CompareTask>`를 사용하여 우선순위 큐를 구현합니다.
  - 작업의 우선순위에 따라 처리 순서를 결정합니다.

이로써 8주차 강의가 마무리됩니다. 학생들은 큐의 기본 개념과 구현 방법을 이해하고, 이를 활용하여 다양한 프로그램을 작성하는 능력을 기르게 됩니다.