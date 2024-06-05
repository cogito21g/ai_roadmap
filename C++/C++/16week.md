### 16주차: 네임스페이스와 모듈

#### 강의 목표
- 네임스페이스의 개념 이해 및 사용
- 모듈화와 코드 관리의 이해
- 전처리기 지시문 (#define, #include 등) 이해 및 사용

#### 강의 내용

##### 1. 네임스페이스 사용법
- **기본 네임스페이스**

```cpp
#include <iostream>

namespace MyNamespace {
    void printMessage() {
        std::cout << "Hello from MyNamespace!" << std::endl;
    }
}

int main() {
    MyNamespace::printMessage();  // 네임스페이스를 사용하여 함수 호출
    return 0;
}
```

- **using 선언**

```cpp
#include <iostream>

namespace MyNamespace {
    void printMessage() {
        std::cout << "Hello from MyNamespace!" << std::endl;
    }
}

int main() {
    using namespace MyNamespace;
    printMessage();  // 네임스페이스를 사용하지 않고 함수 호출
    return 0;
}
```

##### 2. 모듈화와 코드 관리
- **헤더 파일과 소스 파일 분리**

**MyClass.h**

```cpp
#ifndef MYCLASS_H
#define MYCLASS_H

class MyClass {
public:
    void printMessage();
};

#endif
```

**MyClass.cpp**

```cpp
#include "MyClass.h"
#include <iostream>

void MyClass::printMessage() {
    std::cout << "Hello from MyClass!" << std::endl;
}
```

**main.cpp**

```cpp
#include "MyClass.h"

int main() {
    MyClass obj;
    obj.printMessage();
    return 0;
}
```

##### 3. 전처리기 지시문
- **#define 사용**

```cpp
#include <iostream>

#define PI 3.14159
#define AREA(radius) (PI * (radius) * (radius))

int main() {
    double radius = 5.0;
    std::cout << "Area of circle: " << AREA(radius) << std::endl;
    return 0;
}
```

- **#include 사용**

**constants.h**

```cpp
#ifndef CONSTANTS_H
#define CONSTANTS_H

const double PI = 3.14159;

#endif
```

**main.cpp**

```cpp
#include <iostream>
#include "constants.h"

int main() {
    double radius = 5.0;
    std::cout << "Area of circle: " << PI * radius * radius << std::endl;
    return 0;
}
```

#### 과제

1. **네임스페이스 사용**
   - 네임스페이스를 사용하여 두 개의 함수를 포함하는 프로그램을 작성하세요. 하나는 네임스페이스 내에서, 다른 하나는 전역 네임스페이스에서 정의합니다.

```cpp
#include <iostream>

namespace MyNamespace {
    void printMessage() {
        std::cout << "Hello from MyNamespace!" << std::endl;
    }
}

void printMessage() {
    std::cout << "Hello from the global namespace!" << std::endl;
}

int main() {
    MyNamespace::printMessage();
    ::printMessage();
    return 0;
}
```

2. **모듈화 사용**
   - 간단한 계산기 클래스를 헤더 파일과 소스 파일로 분리하여 구현하고, 이를 사용하는 프로그램을 작성하세요.

**Calculator.h**

```cpp
#ifndef CALCULATOR_H
#define CALCULATOR_H

class Calculator {
public:
    int add(int a, int b);
    int subtract(int a, int b);
};

#endif
```

**Calculator.cpp**

```cpp
#include "Calculator.h"

int Calculator::add(int a, int b) {
    return a + b;
}

int Calculator::subtract(int a, int b) {
    return a - b;
}
```

**main.cpp**

```cpp
#include <iostream>
#include "Calculator.h"

int main() {
    Calculator calc;
    std::cout << "5 + 3 = " << calc.add(5, 3) << std::endl;
    std::cout << "5 - 3 = " << calc.subtract(5, 3) << std::endl;
    return 0;
}
```

3. **전처리기 지시문 사용**
   - #define을 사용하여 상수를 정의하고, 이를 사용하여 원의 둘레를 계산하는 프로그램을 작성하세요.

```cpp
#include <iostream>

#define PI 3.14159
#define CIRCUMFERENCE(radius) (2 * PI * (radius))

int main() {
    double radius = 5.0;
    std::cout << "Circumference of circle: " << CIRCUMFERENCE(radius) << std::endl;
    return 0;
}
```

#### 퀴즈

1. **네임스페이스에 대한 설명 중 맞는 것은?**
   - A) 네임스페이스는 함수만 포함할 수 있다.
   - B) 네임스페이스는 클래스와 함수를 포함할 수 있다.
   - C) 네임스페이스는 전역 네임스페이스 내에 정의될 수 없다.
   - D) 네임스페이스는 다른 네임스페이스와 동일한 이름을 가질 수 없다.

2. **모듈화의 장점은?**
   - A) 모든 코드를 한 파일에 작성할 수 있다.
   - B) 코드의 재사용성을 높이고 유지보수를 용이하게 한다.
   - C) 컴파일 시간을 증가시킨다.
   - D) 전처리기 지시문을 사용하지 않는다.

3. **#define의 사용 중 틀린 것은?**
   - A) #define은 상수를 정의하는 데 사용될 수 있다.
   - B) #define은 매크로 함수를 정의하는 데 사용될 수 있다.
   - C) #define은 변수의 값을 변경하는 데 사용될 수 있다.
   - D) #define은 전처리기 지시문이다.

4. **#include에 대한 설명 중 맞는 것은?**
   - A) #include는 소스 파일의 내용을 주석 처리한다.
   - B) #include는 다른 파일의 내용을 현재 파일에 삽입한다.
   - C) #include는 컴파일 시간을 줄인다.
   - D) #include는 함수의 동작을 변경한다.

#### 퀴즈 해설

1. **네임스페이스에 대한 설명 중 맞는 것은?**
   - **정답: B) 네임스페이스는 클래스와 함수를 포함할 수 있다.**
     - 해설: 네임스페이스는 클래스, 함수, 변수 등을 포함할 수 있습니다.

2. **모듈화의 장점은?**
   - **정답: B) 코드의 재사용성을 높이고 유지보수를 용이하게 한다.**
     - 해설: 모듈화는 코드를 여러 파일로 분리하여 재사용성을 높이고 유지보수를 용이하게 합니다.

3. **#define의 사용 중 틀린 것은?**
   - **정답: C) #define은 변수의 값을 변경하는 데 사용될 수 있다.**
     - 해설: #define은 상수나 매크로 함수를 정의하는 데 사용되며, 변수의 값을 변경할 수 없습니다.

4. **#include에 대한 설명 중 맞는 것은?**
   - **정답: B) #include는 다른 파일의 내용을 현재 파일에 삽입한다.**
     - 해설: #include는 지정된 파일의 내용을 현재 파일에 삽입하여 컴파일러가 해당 파일을 포함하도록 합니다.

다음 주차 강의 내용을 요청하시면, 17주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.