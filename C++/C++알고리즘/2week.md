### 2주차 강의 계획안

#### 강의 주제: 재귀 알고리즘
- 재귀의 개념 및 기초 예제
- 재귀를 이용한 문제 해결 (피보나치 수열, 팩토리얼 등)

---

### 강의 내용

#### 1. 재귀의 개념 및 기초 예제
- **재귀 함수**: 자기 자신을 호출하는 함수
- **기본 구성 요소**:
  - 기본 사례(Base Case): 재귀 호출을 멈추는 조건
  - 재귀 사례(Recursive Case): 자기 자신을 호출하는 부분

**예제**: 간단한 재귀 함수
```cpp
#include <iostream>
using namespace std;

void printNumbers(int n) {
    if (n == 0) {
        return;
    }
    cout << n << " ";
    printNumbers(n - 1);
}

int main() {
    printNumbers(5);
    return 0;
}
```

#### 2. 재귀를 이용한 문제 해결
- **피보나치 수열**: `F(n) = F(n-1) + F(n-2)`
- **팩토리얼**: `n! = n * (n-1)!`

**예제**: 피보나치 수열
```cpp
#include <iostream>
using namespace std;

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    int n = 10;
    for (int i = 0; i < n; i++) {
        cout << fibonacci(i) << " ";
    }
    return 0;
}
```

**예제**: 팩토리얼
```cpp
#include <iostream>
using namespace std;

int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

int main() {
    int n = 5;
    cout << n << "! = " << factorial(n) << endl;
    return 0;
}
```

---

### 과제

#### 과제 1: 재귀를 이용한 문자열 역순 출력
사용자로부터 문자열을 입력받아 재귀를 이용해 역순으로 출력하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <string>
using namespace std;

void reverseString(string str, int index) {
    if (index < 0) {
        return;
    }
    cout << str[index];
    reverseString(str, index - 1);
}

int main() {
    string str;
    cout << "Enter a string: ";
    cin >> str;
    reverseString(str, str.length() - 1);
    return 0;
}
```

**해설**:
1. `reverseString` 함수는 문자열의 인덱스를 인자로 받아 역순으로 문자를 출력합니다.
2. 기본 사례는 인덱스가 0보다 작아지면 함수가 종료됩니다.
3. 재귀 사례는 현재 인덱스의 문자를 출력하고, 인덱스를 감소시켜 재귀 호출합니다.

#### 과제 2: 재귀를 이용한 최대 공약수(GCD) 계산
두 정수의 최대 공약수를 재귀를 이용해 계산하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
using namespace std;

int gcd(int a, int b) {
    if (b == 0) {
        return a;
    }
    return gcd(b, a % b);
}

int main() {
    int a, b;
    cout << "Enter two integers: ";
    cin >> a >> b;
    cout << "GCD of " << a << " and " << b << " is " << gcd(a, b) << endl;
    return 0;
}
```

**해설**:
1. `gcd` 함수는 두 정수를 인자로 받아 최대 공약수를 계산합니다.
2. 기본 사례는 두 번째 정수가 0이 되면 첫 번째 정수를 반환합니다.
3. 재귀 사례는 두 번째 정수와 첫 번째 정수를 두 번째 정수로 나눈 나머지로 재귀 호출합니다.

---

### 퀴즈

#### 퀴즈 1: 다음 중 재귀 함수의 기본 사례(Base Case)에 대한 설명으로 옳은 것은 무엇인가요?
1. 재귀 호출을 계속하는 조건
2. 재귀 호출을 멈추는 조건
3. 함수의 매개변수를 초기화하는 조건
4. 함수의 반환 값을 출력하는 조건

**정답**: 2. 재귀 호출을 멈추는 조건

#### 퀴즈 2: 다음 코드의 출력 결과는 무엇인가요?
```cpp
#include <iostream>
using namespace std;

void countDown(int n) {
    if (n == 0) {
        cout << "Lift off!" << endl;
        return;
    }
    cout << n << " ";
    countDown(n - 1);
}

int main() {
    countDown(3);
    return 0;
}
```

**정답**:
```
3 2 1 Lift off!
```

**해설**:
- `countDown` 함수는 인자가 0이 될 때까지 감소시키며 출력합니다.
- 인자가 0이 되면 "Lift off!"를 출력하고 함수를 종료합니다.

이 계획안은 2주차에 필요한 재귀 알고리즘의 기본 개념과 실습을 통해 학습할 수 있도록 구성되었습니다. 과제와 퀴즈를 통해 학습한 내용을 복습하고, 이해도를 높일 수 있습니다.