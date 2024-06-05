### 9주차: 템플릿과 예외 처리

#### 강의 목표
- 템플릿의 개념 이해 및 사용
- 함수 템플릿과 클래스 템플릿의 사용법 이해
- 예외 처리의 개념 이해 및 사용
- try, catch, throw 구문의 사용법 이해

#### 강의 내용

##### 1. 함수 템플릿
- **함수 템플릿 선언 및 사용**

```cpp
#include <iostream>
using namespace std;

template <typename T>
T add(T a, T b) {
    return a + b;
}

int main() {
    cout << "Int: " << add<int>(3, 4) << endl;
    cout << "Double: " << add<double>(2.5, 3.7) << endl;

    return 0;
}
```

##### 2. 클래스 템플릿
- **클래스 템플릿 선언 및 사용**

```cpp
#include <iostream>
using namespace std;

template <typename T>
class Calculator {
public:
    T add(T a, T b) {
        return a + b;
    }

    T subtract(T a, T b) {
        return a - b;
    }
};

int main() {
    Calculator<int> intCalc;
    Calculator<double> doubleCalc;

    cout << "Int Add: " << intCalc.add(3, 4) << endl;
    cout << "Double Subtract: " << doubleCalc.subtract(5.5, 2.3) << endl;

    return 0;
}
```

##### 3. 템플릿 특수화
- **템플릿 특수화 사용**

```cpp
#include <iostream>
using namespace std;

template <typename T>
class Printer {
public:
    void print(T value) {
        cout << "Value: " << value << endl;
    }
};

// 템플릿 특수화
template <>
class Printer<char> {
public:
    void print(char value) {
        cout << "Char: " << value << endl;
    }
};

int main() {
    Printer<int> intPrinter;
    Printer<char> charPrinter;

    intPrinter.print(100);
    charPrinter.print('A');

    return 0;
}
```

##### 4. 예외 처리 (try, catch, throw)
- **예외 처리 구문 사용**

```cpp
#include <iostream>
using namespace std;

double divide(double numerator, double denominator) {
    if (denominator == 0) {
        throw runtime_error("Division by zero!");
    }
    return numerator / denominator;
}

int main() {
    double num, den;
    cout << "Enter numerator: ";
    cin >> num;
    cout << "Enter denominator: ";
    cin >> den;

    try {
        double result = divide(num, den);
        cout << "Result: " << result << endl;
    } catch (const runtime_error& e) {
        cout << "Error: " << e.what() << endl;
    }

    return 0;
}
```

#### 과제

1. **함수 템플릿 작성**
   - 두 개의 값을 비교하여 더 큰 값을 반환하는 함수 템플릿 `maxValue`를 작성하세요.

```cpp
#include <iostream>
using namespace std;

template <typename T>
T maxValue(T a, T b) {
    return (a > b) ? a : b;
}

int main() {
    cout << "Max of 3 and 7: " << maxValue(3, 7) << endl;
    cout << "Max of 2.5 and 3.7: " << maxValue(2.5, 3.7) << endl;

    return 0;
}
```

2. **클래스 템플릿 작성**
   - Stack 클래스 템플릿을 작성하여, 정수형과 실수형 데이터를 저장할 수 있도록 구현하세요.

```cpp
#include <iostream>
using namespace std;

template <typename T>
class Stack {
private:
    T arr[100];
    int top;
public:
    Stack() : top(-1) {}

    void push(T value) {
        if (top >= 99) {
            cout << "Stack overflow" << endl;
            return;
        }
        arr[++top] = value;
    }

    T pop() {
        if (top < 0) {
            throw runtime_error("Stack underflow");
        }
        return arr[top--];
    }

    bool isEmpty() {
        return top == -1;
    }
};

int main() {
    Stack<int> intStack;
    Stack<double> doubleStack;

    intStack.push(1);
    intStack.push(2);
    cout << "Popped from intStack: " << intStack.pop() << endl;

    doubleStack.push(3.5);
    doubleStack.push(4.5);
    cout << "Popped from doubleStack: " << doubleStack.pop() << endl;

    return 0;
}
```

3. **예외 처리 연습**
   - 두 개의 정수를 입력받아 나누기 연산을 수행하는 함수 `divide`를 작성하세요. 분모가 0일 경우 예외를 발생시키고, 이를 처리하는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

int divide(int numerator, int denominator) {
    if (denominator == 0) {
        throw runtime_error("Division by zero!");
    }
    return numerator / denominator;
}

int main() {
    int num, den;
    cout << "Enter numerator: ";
    cin >> num;
    cout << "Enter denominator: ";
    cin >> den;

    try {
        int result = divide(num, den);
        cout << "Result: " << result << endl;
    } catch (const runtime_error& e) {
        cout << "Error: " << e.what() << endl;
    }

    return 0;
}
```

#### 퀴즈

1. **템플릿에 대한 설명 중 틀린 것은?**
   - A) 템플릿은 데이터 타입에 의존하지 않는 코드를 작성할 수 있게 한다.
   - B) 함수 템플릿과 클래스 템플릿이 있다.
   - C) 템플릿 특수화는 특정 데이터 타입에 대해 템플릿을 재정의하는 것이다.
   - D) 템플릿은 상속과 다형성을 지원하지 않는다.

2. **예외 처리에 대한 설명 중 맞는 것은?**
   - A) try 블록 내에서 발생한 예외는 catch 블록에서 처리된다.
   - B) 예외는 함수 내부에서만 처리할 수 있다.
   - C) throw 키워드는 예외를 발생시키는 데 사용된다.
   - D) catch 블록은 항상 하나만 있어야 한다.

3. **클래스 템플릿에 대한 설명 중 맞는 것은?**
   - A) 클래스 템플릿은 함수 템플릿과 동일한 방식으로 사용된다.
   - B) 클래스 템플릿은 특정 데이터 타입에 대해서만 인스턴스화할 수 있다.
   - C) 클래스 템플릿은 템플릿 매개변수를 사용하여 여러 데이터 타입을 지원한다.
   - D) 클래스 템플릿은 연산자 오버로딩을 지원하지 않는다.

#### 퀴즈 해설

1. **템플릿에 대한 설명 중 틀린 것은?**
   - **정답: D) 템플릿은 상속과 다형성을 지원하지 않는다.**
     - 해설: 템플릿은 상속과 다형성을 지원합니다.

2. **예외 처리에 대한 설명 중 맞는 것은?**
   - **정답: A) try 블록 내에서 발생한 예외는 catch 블록에서 처리된다.**
     - 해설: try 블록 내에서 발생한 예외는 해당 예외를 처리할 수 있는 catch 블록에서 처리됩니다.

3. **클래스 템플릿에 대한 설명 중 맞는 것은?**
   - **정답: C) 클래스 템플릿은 템플릿 매개변수를 사용하여 여러 데이터 타입을 지원한다.**
     - 해설: 클래스 템플릿은 템플릿 매개변수를 사용하여 다양한 데이터 타입을 지원합니다.

다음 주차 강의 내용을 요청하시면, 10주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.