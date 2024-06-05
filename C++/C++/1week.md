### 1주차: C++ 기초

#### 강의 목표
- C++ 언어의 기본 개념 이해
- 개발 환경 설정
- 기본 문법 학습 (변수, 데이터 타입, 연산자)
- 기본 입출력 방법 익히기 (cin, cout)

#### 강의 내용

##### 1. C++ 언어 소개
- **역사 및 특징**: C++는 Bjarne Stroustrup가 1979년에 개발한 C 언어의 확장판으로, 객체지향 프로그래밍을 지원합니다. C++의 주요 특징은 고성능, 유연성, 저수준 메모리 조작 능력입니다.

##### 2. 개발 환경 설정
- **IDE 설치**: Visual Studio, Code::Blocks, CLion 등
- **컴파일러 설치**: GCC, Clang 등
- **기본 설정**: 프로젝트 생성, 첫 번째 C++ 프로그램 컴파일 및 실행

1. **Visual Studio 설치 및 설정**
   - Visual Studio 설치
   - C++ 개발 환경 설정
   - 새 프로젝트 생성

2. **Hello, World! 프로그램 작성**
   - 새 프로젝트에서 첫 번째 C++ 파일 생성
   - Hello, World! 프로그램 작성 및 실행

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    return 0;
}
```

##### 3. 기본 문법
- **변수 선언**: `int`, `float`, `char`, `double` 등
- **데이터 타입**: 기본 데이터 타입 및 사용법
- **연산자**: 산술 연산자, 관계 연산자, 논리 연산자

```cpp
#include <iostream>
using namespace std;

int main() {
    int a = 5;
    int b = 3;
    cout << "a + b = " << a + b << endl;
    cout << "a - b = " << a - b << endl;
    cout << "a * b = " << a * b << endl;
    cout << "a / b = " << a / b << endl;
    return 0;
}
```

##### 4. 기본 입출력
- **기본 입출력**: `cin`, `cout` 사용법

```cpp
#include <iostream>
using namespace std;

int main() {
    int number;
    cout << "Enter an integer: ";
    cin >> number;
    cout << "You entered: " << number << endl;
    return 0;
}
```

#### 과제

1. **개발 환경 설정 및 Hello World 프로그램**
   - IDE를 설치하고, 첫 번째 프로그램을 작성해보세요.
   - "Hello, World!"를 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    return 0;
}
```

2. **기본 연산 프로그램**
   - 두 개의 정수를 입력받고, 이들의 합, 차, 곱, 나눗셈을 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

int main() {
    int num1, num2;
    cout << "Enter two integers: ";
    cin >> num1 >> num2;
    cout << "Sum: " << num1 + num2 << endl;
    cout << "Difference: " << num1 - num2 << endl;
    cout << "Product: " << num1 * num2 << endl;
    cout << "Quotient: " << num1 / num2 << endl;
    return 0;
}
```

#### 퀴즈

1. **C++의 주요 특징 중 하나가 아닌 것은?**
   - A) 객체지향 프로그래밍 지원
   - B) 고성능
   - C) 인터프리터 언어
   - D) 저수준 메모리 조작 능력

2. **다음 중 변수 선언이 올바른 것은?**
   - A) int 1num;
   - B) double number = 3.14;
   - C) char 'a' = 'A';
   - D) float num = "3.14";

#### 퀴즈 해설

1. **C++의 주요 특징 중 하나가 아닌 것은?**
   - **정답: C) 인터프리터 언어**
     - 해설: C++는 컴파일러 기반 언어입니다.

2. **다음 중 변수 선언이 올바른 것은?**
   - **정답: B) double number = 3.14;**
     - 해설: 변수 이름은 숫자로 시작할 수 없고, char 타입 변수는 작은따옴표로 값을 정의해야 하며, float 타입 변수는 문자열을 값으로 가질 수 없습니다.

다음 주차 강의 내용을 요청하시면, 2주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.