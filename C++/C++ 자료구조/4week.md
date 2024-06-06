### 4주차 강의 계획: 템플릿 및 예외처리

#### 강의 목표
- 템플릿의 개념과 사용법 이해
- 함수 템플릿과 클래스 템플릿 작성 및 활용
- 예외처리의 개념 이해 및 활용 방법 학습
- 예외처리를 통해 프로그램의 안정성 향상

#### 강의 구성
- **강의 시간**: 2시간
- **구성**: 템플릿 이론 (30분), 예외처리 이론 (30분), 실습 및 과제 안내 (30분), 퀴즈 및 해설 (30분)

#### 강의 내용

##### 1. 템플릿 이론 (30분)

###### 1.1 템플릿의 개념
- **템플릿 개요**:
  - 템플릿의 필요성
  - 코드 재사용성 향상

###### 1.2 함수 템플릿
- **함수 템플릿 정의**:
  - 함수 템플릿의 기본 구조 및 사용법
- **예제**:
```cpp
#include <iostream>
using namespace std;

template <typename T>
T add(T a, T b) {
    return a + b;
}

int main() {
    cout << "Int: " << add(3, 4) << endl;
    cout << "Double: " << add(3.5, 4.5) << endl;
    return 0;
}
```

###### 1.3 클래스 템플릿
- **클래스 템플릿 정의**:
  - 클래스 템플릿의 기본 구조 및 사용법
- **예제**:
```cpp
#include <iostream>
using namespace std;

template <typename T>
class Box {
private:
    T value;
public:
    Box(T v) : value(v) {}
    T getValue() { return value; }
};

int main() {
    Box<int> intBox(123);
    Box<double> doubleBox(456.78);

    cout << "Int Box: " << intBox.getValue() << endl;
    cout << "Double Box: " << doubleBox.getValue() << endl;
    return 0;
}
```

##### 2. 예외처리 이론 (30분)

###### 2.1 예외처리의 개념
- **예외처리 개요**:
  - 예외의 정의 및 필요성
  - 예외 발생 시 프로그램의 흐름 제어

###### 2.2 예외처리 구문
- **try-catch 구문**:
  - 예외를 발생시키는 코드와 예외를 처리하는 코드 분리
- **예제**:
```cpp
#include <iostream>
using namespace std;

int main() {
    try {
        int a = 10;
        int b = 0;
        if (b == 0) {
            throw "Division by zero error!";
        }
        cout << "Result: " << a / b << endl;
    } catch (const char* msg) {
        cerr << "Error: " << msg << endl;
    }
    return 0;
}
```

###### 2.3 예외의 종류 및 처리
- **표준 예외**:
  - std::exception 및 표준 예외 클래스들 (std::out_of_range, std::invalid_argument 등)
- **사용자 정의 예외**:
  - 사용자 정의 예외 클래스 작성 및 사용
- **예제**:
```cpp
#include <iostream>
#include <stdexcept>
using namespace std;

class MyException : public exception {
public:
    const char* what() const noexcept override {
        return "My custom exception occurred!";
    }
};

int main() {
    try {
        throw MyException();
    } catch (MyException& e) {
        cerr << e.what() << endl;
    }
    return 0;
}
```

##### 3. 실습 및 과제 안내 (30분)

###### 3.1 실습
- **실습 목표**:
  - 강의에서 다룬 내용을 직접 코드로 작성해보기
- **실습 문제**:
  - 함수 템플릿과 클래스 템플릿을 사용한 간단한 프로그램 작성
  - 예외처리를 활용한 안정적인 프로그램 작성

###### 3.2 과제 안내
- **과제 내용**:
  - 템플릿을 사용하여 다양한 데이터 타입을 처리하는 Stack 클래스 작성
  - Stack 클래스는 push, pop, top, isEmpty 메서드를 포함
  - Stack 클래스에서 예외처리를 활용하여 스택 언더플로우 및 오버플로우 처리
- **과제 제출 방법**:
  - 작성한 코드를 주차별 과제 제출 폴더에 업로드
  - 제출 기한: 다음 주 강의 전까지

##### 4. 퀴즈 및 해설 (30분)

###### 퀴즈 문제:
1. 템플릿을 사용하여 정의된 함수는 어떤 데이터 타입을 처리할 수 있는가?
   - a) 정수형 데이터만
   - b) 실수형 데이터만
   - c) 문자형 데이터만
   - d) 모든 데이터 타입
2. 예외를 발생시키기 위해 사용하는 키워드는?
   - a) try
   - b) catch
   - c) throw
   - d) exception
3. 표준 예외 클래스가 아닌 것은?
   - a) std::out_of_range
   - b) std::invalid_argument
   - c) std::exception
   - d) std::invalid_type

###### 퀴즈 해설:
1. **정답: d) 모든 데이터 타입**
   - 템플릿을 사용하면 함수나 클래스를 정의할 때 데이터 타입을 미리 지정하지 않고, 필요할 때 지정할 수 있습니다.
2. **정답: c) throw**
   - `throw` 키워드는 예외를 발생시킬 때 사용합니다.
3. **정답: d) std::invalid_type**
   - `std::invalid_type`는 존재하지 않는 표준 예외 클래스입니다.

### 준비 자료
- 강의 슬라이드
- 실습 코드 예제
- 과제 안내서

### 과제 해설

##### 과제: 템플릿을 사용하여 다양한 데이터 타입을 처리하는 Stack 클래스 작성
- **문제**: 템플릿을 사용하여 다양한 데이터 타입을 처리하는 Stack 클래스를 작성하고, 예외처리를 통해 안정성을 향상시킵니다.
- **해설**:
  - `Stack` 클래스는 템플릿을 사용하여 다양한 데이터 타입을 처리할 수 있어야 합니다.
  - `push`, `pop`, `top`, `isEmpty` 메서드를 포함하며, 스택 언더플로우 및 오버플로우를 처리합니다.

```cpp
#include <iostream>
#include <stdexcept>
using namespace std;

template <typename T>
class Stack {
private:
    T* arr;
    int topIndex;
    int capacity;
public:
    Stack(int size) : capacity(size), topIndex(-1) {
        arr = new T[size];
    }

    ~Stack() {
        delete[] arr;
    }

    void push(T value) {
        if (topIndex >= capacity - 1) {
            throw overflow_error("Stack Overflow");
        }
        arr[++topIndex] = value;
    }

    void pop() {
        if (topIndex < 0) {
            throw underflow_error("Stack Underflow");
        }
        --topIndex;
    }

    T top() {
        if (topIndex < 0) {
            throw underflow_error("Stack is Empty");
        }
        return arr[topIndex];
    }

    bool isEmpty() {
        return topIndex < 0;
    }
};

int main() {
    try {
        Stack<int> intStack(5);

        intStack.push(1);
        intStack.push(2);
        intStack.push(3);

        cout << "Top element: " << intStack.top() << endl;

        intStack.pop();
        cout << "Top element after pop: " << intStack.top() << endl;

        intStack.pop();
        intStack.pop();
        intStack.pop();  // This will cause an underflow error

    } catch (const exception& e) {
        cerr << "Exception: " << e.what() << endl;
    }

    return 0;
}
```
- **설명**:
  - `Stack` 클래스는 템플릿을 사용하여 다양한 데이터 타입을 처리할 수 있도록 설계되었습니다.
  - `push` 메서드는 스택에 요소를 추가하며, 오버플로우 발생 시 예외를 던집니다.
  - `pop` 메서드는 스택에서 요소를 제거하며, 언더플로우 발생 시 예외를 던집니다.
  - `top` 메서드는 스택의 최상위 요소를 반환하며, 스택이 비어 있을 경우 예외를 던집니다.
  - `isEmpty` 메서드는 스택이 비어 있는지 여부를 확인합니다.

이로써 4주차 강의가 마무리됩니다. 학생들은 템플릿과 예외처리의 기본 개념을 이해하고, 이를 활용하여 다양한 프로그램을 작성하는 능력을 기르게 됩니다.