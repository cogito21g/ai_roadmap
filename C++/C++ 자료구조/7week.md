### 7주차 강의 계획: 스택 (Stack)

#### 강의 목표
- 스택의 기본 개념과 활용 이해
- 배열 기반 스택과 연결 리스트 기반 스택 구현
- 스택의 다양한 응용 사례 학습

#### 강의 구성
- **강의 시간**: 2시간
- **구성**: 스택 이론 (30분), 배열 기반 스택 (30분), 연결 리스트 기반 스택 (30분), 실습 및 과제 안내 (30분)

#### 강의 내용

##### 1. 스택 이론 (30분)

###### 1.1 스택의 기본 개념
- **스택 개요**:
  - LIFO (Last In, First Out) 구조
  - 주요 연산: `push`, `pop`, `peek`, `isEmpty`
- **스택의 활용 사례**:
  - 함수 호출의 관리 (재귀)
  - 수식의 괄호 검사
  - 문자열 뒤집기

###### 1.2 스택의 장단점
- **장점**:
  - 구현이 간단
  - LIFO 구조로 특정 응용에 유용
- **단점**:
  - 고정된 크기를 가지면 메모리 낭비 가능 (배열 기반)
  - 탐색이 비효율적

##### 2. 배열 기반 스택 (30분)

###### 2.1 배열 기반 스택의 구조
- **배열 기반 스택 구현**:
  - 고정 크기의 배열을 사용
  - 스택 포인터를 사용하여 스택의 최상위 요소 추적

###### 2.2 배열 기반 스택의 구현
- **스택의 주요 연산 구현**:
  - `push`, `pop`, `peek`, `isEmpty`, `isFull`
- **예제**:
```cpp
#include <iostream>
using namespace std;

class Stack {
private:
    int* arr;
    int top;
    int capacity;
public:
    Stack(int size) {
        arr = new int[size];
        capacity = size;
        top = -1;
    }

    ~Stack() {
        delete[] arr;
    }

    void push(int x) {
        if (isFull()) {
            cout << "Stack Overflow\n";
            return;
        }
        arr[++top] = x;
    }

    int pop() {
        if (isEmpty()) {
            cout << "Stack Underflow\n";
            return -1;
        }
        return arr[top--];
    }

    int peek() {
        if (isEmpty()) {
            cout << "Stack is empty\n";
            return -1;
        }
        return arr[top];
    }

    bool isEmpty() {
        return top == -1;
    }

    bool isFull() {
        return top == capacity - 1;
    }

    int size() {
        return top + 1;
    }

    void display() {
        if (isEmpty()) {
            cout << "Stack is empty\n";
            return;
        }
        for (int i = 0; i <= top; ++i) {
            cout << arr[i] << " ";
        }
        cout << endl;
    }
};

int main() {
    Stack stack(5);
    stack.push(10);
    stack.push(20);
    stack.push(30);
    stack.display();
    cout << "Top element: " << stack.peek() << endl;
    stack.pop();
    stack.display();
    return 0;
}
```

##### 3. 연결 리스트 기반 스택 (30분)

###### 3.1 연결 리스트 기반 스택의 구조
- **연결 리스트 기반 스택 구현**:
  - 동적 크기의 연결 리스트 사용
  - 노드를 사용하여 스택의 요소 관리

###### 3.2 연결 리스트 기반 스택의 구현
- **스택의 주요 연산 구현**:
  - `push`, `pop`, `peek`, `isEmpty`
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

class Stack {
private:
    Node* top;
public:
    Stack() {
        top = nullptr;
    }

    ~Stack() {
        while (!isEmpty()) {
            pop();
        }
    }

    void push(int x) {
        Node* newNode = new Node(x);
        newNode->next = top;
        top = newNode;
    }

    int pop() {
        if (isEmpty()) {
            cout << "Stack Underflow\n";
            return -1;
        }
        Node* temp = top;
        top = top->next;
        int popped = temp->data;
        delete temp;
        return popped;
    }

    int peek() {
        if (isEmpty()) {
            cout << "Stack is empty\n";
            return -1;
        }
        return top->data;
    }

    bool isEmpty() {
        return top == nullptr;
    }

    void display() {
        if (isEmpty()) {
            cout << "Stack is empty\n";
            return;
        }
        Node* temp = top;
        while (temp != nullptr) {
            cout << temp->data << " ";
            temp = temp->next;
        }
        cout << endl;
    }
};

int main() {
    Stack stack;
    stack.push(10);
    stack.push(20);
    stack.push(30);
    stack.display();
    cout << "Top element: " << stack.peek() << endl;
    stack.pop();
    stack.display();
    return 0;
}
```

##### 4. 실습 및 과제 안내 (30분)

###### 4.1 실습
- **실습 목표**:
  - 스택을 사용한 프로그램 작성
- **실습 문제**:
  - 배열 기반 스택과 연결 리스트 기반 스택을 구현하고, 각 스택에 데이터를 삽입, 삭제, 출력하는 프로그램 작성

###### 4.2 과제 안내
- **과제 내용**:
  - 스택을 사용하여 문자열의 괄호 유효성을 검사하는 프로그램 작성
  - 스택을 사용하여 후위 표기법(Postfix notation)의 수식을 계산하는 프로그램 작성
- **과제 제출 방법**:
  - 작성한 코드를 주차별 과제 제출 폴더에 업로드
  - 제출 기한: 다음 주 강의 전까지

##### 5. 퀴즈 및 해설 (30분)

###### 퀴즈 문제:
1. 스택의 기본 연산이 아닌 것은?
   - a) push
   - b) pop
   - c) peek
   - d) search
2. 스택의 구조는 어떤 방식으로 작동하는가?
   - a) FIFO
   - b) LIFO
   - c) LILO
   - d) FILO
3. 연결 리스트 기반 스택에서 노드를 추가할 때 추가되는 위치는?
   - a) 리스트의 시작
   - b) 리스트의 중간
   - c) 리스트의 끝
   - d) 리스트의 임의 위치

###### 퀴즈 해설:
1. **정답: d) search**
   - 스택의 기본 연산은 `push`, `pop`, `peek`, `isEmpty`입니다. `search`는 기본 연산이 아닙니다.
2. **정답: b) LIFO**
   - 스택은 LIFO (Last In, First Out) 구조로 작동합니다.
3. **정답: a) 리스트의 시작**
   - 연결 리스트 기반 스택에서 노드는 리스트의 시작 부분에 추가됩니다.

### 준비 자료
- 강의 슬라이드
- 실습 코드 예제
- 과제 안내서

### 과제 해설

##### 과제: 스택을 사용하여 문자열의 괄호 유효성을 검사하는 프로그램 작성
- **문제**: 주어진 문자열에서 괄호의 유효성을 검사하는 프로그램 작성
- **해설**:
  - 스택을 사용하여 여는 괄호를 저장하고, 닫는 괄호를 만나면 스택에서 여는 괄호를 꺼내어 매칭합니다.

```cpp
#include <iostream>
#include <stack>
using namespace std;

bool isValid(string s) {
    stack<char> st;
    for (char ch : s) {
        if (ch == '(' || ch == '{' || ch == '[') {
            st.push(ch);
        } else {
            if (st.empty()) return false;
            char top = st.top();
            st.pop();
            if ((ch == ')' && top != '(') ||
                (ch == '}' && top != '{') ||
                (ch == ']' && top != '[')) {
                return false;
            }
        }
    }
    return st.empty();
}

int main() {
    string s;
    cout << "Enter a string with parentheses: ";
    cin >> s;
    if (isValid(s)) {
        cout << "The parentheses are valid." << endl;
    } else {
        cout << "The parentheses are not valid." << endl;
    }
    return 0;
}
```
- **설명**:
  - `isValid` 함수는 문자열의 괄호가 유효한지 검사합니다.
  - 여는 괄호를 스택에 저장하고, 닫는 괄호를 만나면 스택에서 꺼내어 매칭

합니다.
  - 모든 괄호가 매칭되면 유효한 문자열로 판단합니다.

##### 과제: 스택을 사용하여 후위 표기법(Postfix notation)의 수식을 계산하는 프로그램 작성
- **문제**: 주어진 후위 표기법의 수식을 계산하는 프로그램 작성
- **해설**:
  - 스택을 사용하여 피연산자를 저장하고, 연산자를 만나면 스택에서 피연산자를 꺼내어 계산합니다.

```cpp
#include <iostream>
#include <stack>
using namespace std;

int evaluatePostfix(string exp) {
    stack<int> st;
    for (char ch : exp) {
        if (isdigit(ch)) {
            st.push(ch - '0');
        } else {
            int val1 = st.top(); st.pop();
            int val2 = st.top(); st.pop();
            switch (ch) {
                case '+': st.push(val2 + val1); break;
                case '-': st.push(val2 - val1); break;
                case '*': st.push(val2 * val1); break;
                case '/': st.push(val2 / val1); break;
            }
        }
    }
    return st.top();
}

int main() {
    string exp;
    cout << "Enter a postfix expression: ";
    cin >> exp;
    cout << "The result is: " << evaluatePostfix(exp) << endl;
    return 0;
}
```
- **설명**:
  - `evaluatePostfix` 함수는 후위 표기법의 수식을 계산합니다.
  - 피연산자를 스택에 저장하고, 연산자를 만나면 스택에서 피연산자를 꺼내어 계산합니다.
  - 계산 결과를 다시 스택에 저장하여 최종 결과를 반환합니다.

이로써 7주차 강의가 마무리됩니다. 학생들은 스택의 기본 개념과 구현 방법을 이해하고, 이를 활용하여 다양한 프로그램을 작성하는 능력을 기르게 됩니다.