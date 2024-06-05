### 3주차: 함수

#### 강의 목표
- 함수 정의와 호출 이해
- 매개변수와 반환값 처리
- 함수 오버로딩 개념 이해
- 재귀 함수 이해 및 사용

#### 강의 내용

##### 1. 함수 정의와 호출
- **함수 정의**

```cpp
#include <iostream>
using namespace std;

// 함수 정의
void sayHello() {
    cout << "Hello, World!" << endl;
}

int main() {
    // 함수 호출
    sayHello();
    return 0;
}
```

- **매개변수와 반환값**

```cpp
#include <iostream>
using namespace std;

// 함수 정의: 매개변수와 반환값
int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(3, 4); // 함수 호출
    cout << "Sum: " << result << endl;
    return 0;
}
```

##### 2. 함수 오버로딩
- **함수 오버로딩**

```cpp
#include <iostream>
using namespace std;

// 함수 오버로딩: 같은 이름의 함수, 다른 매개변수
int add(int a, int b) {
    return a + b;
}

double add(double a, double b) {
    return a + b;
}

int main() {
    int intResult = add(3, 4);       // 정수 덧셈
    double doubleResult = add(2.5, 3.7); // 실수 덧셈
    cout << "Sum (int): " << intResult << endl;
    cout << "Sum (double): " << doubleResult << endl;
    return 0;
}
```

##### 3. 재귀 함수
- **재귀 함수**

```cpp
#include <iostream>
using namespace std;

// 재귀 함수: 팩토리얼 계산
int factorial(int n) {
    if (n <= 1) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

int main() {
    int num;
    cout << "Enter a positive integer: ";
    cin >> num;
    cout << "Factorial of " << num << " is " << factorial(num) << endl;
    return 0;
}
```

#### 과제

1. **간단한 함수 작성**
   - 두 개의 정수를 매개변수로 받아서 더한 값을 반환하는 함수 `add`를 작성하세요.

```cpp
#include <iostream>
using namespace std;

int add(int a, int b) {
    return a + b;
}

int main() {
    int num1, num2;
    cout << "Enter two integers: ";
    cin >> num1 >> num2;
    cout << "Sum: " << add(num1, num2) << endl;
    return 0;
}
```

2. **함수 오버로딩 연습**
   - 두 개의 실수를 매개변수로 받아서 더한 값을 반환하는 `add` 함수를 추가로 작성하세요.

```cpp
#include <iostream>
using namespace std;

int add(int a, int b) {
    return a + b;
}

double add(double a, double b) {
    return a + b;
}

int main() {
    int num1, num2;
    double num3, num4;
    cout << "Enter two integers: ";
    cin >> num1 >> num2;
    cout << "Sum (int): " << add(num1, num2) << endl;
    
    cout << "Enter two floating point numbers: ";
    cin >> num3 >> num4;
    cout << "Sum (double): " << add(num3, num4) << endl;
    return 0;
}
```

3. **재귀 함수 작성**
   - 재귀 함수를 사용하여 피보나치 수열의 n번째 숫자를 계산하는 함수를 작성하세요.

```cpp
#include <iostream>
using namespace std;

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    } else {
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}

int main() {
    int num;
    cout << "Enter an integer: ";
    cin >> num;
    cout << "Fibonacci number: " << fibonacci(num) << endl;
    return 0;
}
```

#### 퀴즈

1. **함수에 대한 설명 중 틀린 것은?**
   - A) 함수는 프로그램을 구조화하는 데 사용된다.
   - B) 함수는 오직 하나의 반환값만 가질 수 있다.
   - C) 함수는 이름을 통해 호출할 수 있다.
   - D) 함수는 반드시 매개변수를 가져야 한다.

2. **다음 중 함수 오버로딩의 예는?**
   - A) 두 함수가 같은 이름과 같은 매개변수를 가진다.
   - B) 두 함수가 같은 이름과 다른 매개변수를 가진다.
   - C) 두 함수가 다른 이름과 같은 매개변수를 가진다.
   - D) 두 함수가 다른 이름과 다른 매개변수를 가진다.

3. **재귀 함수에 대한 설명 중 맞는 것은?**
   - A) 재귀 함수는 항상 빠르게 실행된다.
   - B) 재귀 함수는 종료 조건이 필요 없다.
   - C) 재귀 함수는 스택 오버플로우를 일으킬 수 있다.
   - D) 재귀 함수는 반복문을 대체할 수 없다.

#### 퀴즈 해설

1. **함수에 대한 설명 중 틀린 것은?**
   - **정답: D) 함수는 반드시 매개변수를 가져야 한다.**
     - 해설: 함수는 매개변수가 없어도 정의될 수 있습니다.

2. **다음 중 함수 오버로딩의 예는?**
   - **정답: B) 두 함수가 같은 이름과 다른 매개변수를 가진다.**
     - 해설: 함수 오버로딩은 같은 이름의 함수가 서로 다른 매개변수를 가질 때 발생합니다.

3. **재귀 함수에 대한 설명 중 맞는 것은?**
   - **정답: C) 재귀 함수는 스택 오버플로우를 일으킬 수 있다.**
     - 해설: 재귀 함수는 종료 조건이 없거나 깊이가 너무 깊을 경우 스택 오버플로우를 일으킬 수 있습니다.

다음 주차 강의 내용을 요청하시면, 4주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.