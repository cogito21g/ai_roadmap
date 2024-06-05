### 12주차: STL - 기타 컨테이너와 어댑터

#### 강의 목표
- 기타 STL 컨테이너 (stack, queue, priority_queue)의 개념 이해 및 사용
- 컨테이너 어댑터의 개념 이해 및 사용
- 주요 메서드 및 활용법 익히기

#### 강의 내용

##### 1. stack
- **stack 선언 및 사용**

```cpp
#include <iostream>
#include <stack>
using namespace std;

int main() {
    stack<int> s;

    s.push(1);
    s.push(2);
    s.push(3);

    while (!s.empty()) {
        cout << s.top() << " ";
        s.pop();
    }
    cout << endl;

    return 0;
}
```

- **stack의 주요 메서드**

```cpp
#include <iostream>
#include <stack>
using namespace std;

int main() {
    stack<int> s;

    s.push(1);
    s.push(2);
    s.push(3);

    cout << "Top: " << s.top() << endl;
    cout << "Size: " << s.size() << endl;

    s.pop();
    cout << "After pop, Top: " << s.top() << endl;
    cout << "After pop, Size: " << s.size() << endl;

    return 0;
}
```

##### 2. queue
- **queue 선언 및 사용**

```cpp
#include <iostream>
#include <queue>
using namespace std;

int main() {
    queue<int> q;

    q.push(1);
    q.push(2);
    q.push(3);

    while (!q.empty()) {
        cout << q.front() << " ";
        q.pop();
    }
    cout << endl;

    return 0;
}
```

- **queue의 주요 메서드**

```cpp
#include <iostream>
#include <queue>
using namespace std;

int main() {
    queue<int> q;

    q.push(1);
    q.push(2);
    q.push(3);

    cout << "Front: " << q.front() << endl;
    cout << "Back: " << q.back() << endl;
    cout << "Size: " << q.size() << endl;

    q.pop();
    cout << "After pop, Front: " << q.front() << endl;
    cout << "After pop, Size: " << q.size() << endl;

    return 0;
}
```

##### 3. priority_queue
- **priority_queue 선언 및 사용**

```cpp
#include <iostream>
#include <queue>
using namespace std;

int main() {
    priority_queue<int> pq;

    pq.push(10);
    pq.push(30);
    pq.push(20);
    pq.push(5);

    while (!pq.empty()) {
        cout << pq.top() << " ";
        pq.pop();
    }
    cout << endl;

    return 0;
}
```

- **priority_queue의 주요 메서드**

```cpp
#include <iostream>
#include <queue>
using namespace std;

int main() {
    priority_queue<int> pq;

    pq.push(10);
    pq.push(30);
    pq.push(20);
    pq.push(5);

    cout << "Top: " << pq.top() << endl;
    cout << "Size: " << pq.size() << endl;

    pq.pop();
    cout << "After pop, Top: " << pq.top() << endl;
    cout << "After pop, Size: " << pq.size() << endl;

    return 0;
}
```

#### 과제

1. **stack 사용**
   - 정수를 입력받아 stack에 저장하고, 모든 요소를 거꾸로 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <stack>
using namespace std;

int main() {
    stack<int> s;
    int num;

    cout << "Enter numbers (enter -1 to stop): ";
    while (true) {
        cin >> num;
        if (num == -1) {
            break;
        }
        s.push(num);
    }

    cout << "Reversed order: ";
    while (!s.empty()) {
        cout << s.top() << " ";
        s.pop();
    }
    cout << endl;

    return 0;
}
```

2. **queue 사용**
   - 정수를 입력받아 queue에 저장하고, 모든 요소를 입력 순서대로 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <queue>
using namespace std;

int main() {
    queue<int> q;
    int num;

    cout << "Enter numbers (enter -1 to stop): ";
    while (true) {
        cin >> num;
        if (num == -1) {
            break;
        }
        q.push(num);
    }

    cout << "Original order: ";
    while (!q.empty()) {
        cout << q.front() << " ";
        q.pop();
    }
    cout << endl;

    return 0;
}
```

3. **priority_queue 사용**
   - 정수를 입력받아 priority_queue에 저장하고, 모든 요소를 우선순위에 따라 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <queue>
using namespace std;

int main() {
    priority_queue<int> pq;
    int num;

    cout << "Enter numbers (enter -1 to stop): ";
    while (true) {
        cin >> num;
        if (num == -1) {
            break;
        }
        pq.push(num);
    }

    cout << "Priority order: ";
    while (!pq.empty()) {
        cout << pq.top() << " ";
        pq.pop();
    }
    cout << endl;

    return 0;
}
```

#### 퀴즈

1. **stack에 대한 설명 중 맞는 것은?**
   - A) stack은 FIFO(First In First Out) 구조이다.
   - B) stack은 요소를 뒤에서만 삽입하고 제거할 수 있다.
   - C) stack은 요소를 앞에서만 삽입하고 제거할 수 있다.
   - D) stack은 요소를 임의의 위치에서 삽입하고 제거할 수 있다.

2. **queue에 대한 설명 중 맞는 것은?**
   - A) queue는 LIFO(Last In First Out) 구조이다.
   - B) queue는 요소를 뒤에서 삽입하고 앞에서 제거할 수 있다.
   - C) queue는 요소를 앞에서 삽입하고 뒤에서 제거할 수 있다.
   - D) queue는 요소를 임의의 위치에서 삽입하고 제거할 수 있다.

3. **priority_queue에 대한 설명 중 맞는 것은?**
   - A) priority_queue는 요소를 임의의 순서로 저장한다.
   - B) priority_queue는 요소를 FIFO 순서로 제거한다.
   - C) priority_queue는 요소를 우선순위에 따라 저장하고 제거한다.
   - D) priority_queue는 요소를 임의의 위치에서 삽입하고 제거할 수 있다.

4. **stack, queue, priority_queue의 공통점은?**
   - A) 모두 FIFO 구조이다.
   - B) 모두 LIFO 구조이다.
   - C) 모두 컨테이너 어댑터이다.
   - D) 모두 요소를 정렬된 순서로 저장한다.

#### 퀴즈 해설

1. **stack에 대한 설명 중 맞는 것은?**
   - **정답: B) stack은 요소를 뒤에서만 삽입하고 제거할 수 있다.**
     - 해설: stack은 LIFO(Last In First Out) 구조로, 요소를 뒤에서만 삽입하고 제거할 수 있습니다.

2. **queue에 대한 설명 중 맞는 것은?**
   - **정답: B) queue는 요소를 뒤에서 삽입하고 앞에서 제거할 수 있다.**
     - 해설: queue는 FIFO(First In First Out) 구조로, 요소를 뒤에서 삽입하고 앞에서 제거할 수 있습니다.

3. **priority_queue에 대한 설명 중 맞는 것은?**
   - **정답: C) priority_queue는 요소를 우선순위에 따라 저장하고 제거한다.**
     - 해설: priority_queue는 요소를 우선순위에 따라 저장하고 제거하며, 가장 높은 우선순위의 요소가 먼저 제거됩니다.

4. **stack, queue, priority_queue의 공통점은?**
   - **정답: C) 모두 컨테이너 어댑터이다.**
     - 해설: stack, queue, priority_queue는 모두 컨테이너 어댑터로, 기본 컨테이너의 인터페이스를 어댑트하여 특정 데이터 구조로 작동하게 합니다.

다음 주차 강의 내용을 요청하시면, 13주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.