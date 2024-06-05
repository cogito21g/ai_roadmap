### 2주차: 제어문

#### 강의 목표
- 조건문과 반복문의 이해 및 사용법 익히기
- break와 continue의 사용법 이해

#### 강의 내용

##### 1. 조건문
- **if, else, else if**

```cpp
#include <iostream>
using namespace std;

int main() {
    int number;
    cout << "Enter an integer: ";
    cin >> number;

    if (number > 0) {
        cout << "The number is positive." << endl;
    } else if (number < 0) {
        cout << "The number is negative." << endl;
    } else {
        cout << "The number is zero." << endl;
    }

    return 0;
}
```

- **switch 문**

```cpp
#include <iostream>
using namespace std;

int main() {
    int number;
    cout << "Enter an integer: ";
    cin >> number;

    switch (number) {
        case 0:
            cout << "The number is zero." << endl;
            break;
        case 1:
            cout << "The number is one." << endl;
            break;
        default:
            cout << "The number is neither zero nor one." << endl;
            break;
    }

    return 0;
}
```

##### 2. 반복문
- **for 루프**

```cpp
#include <iostream>
using namespace std;

int main() {
    for (int i = 0; i < 5; i++) {
        cout << "i = " << i << endl;
    }

    return 0;
}
```

- **while 루프**

```cpp
#include <iostream>
using namespace std;

int main() {
    int i = 0;
    while (i < 5) {
        cout << "i = " << i << endl;
        i++;
    }

    return 0;
}
```

- **do-while 루프**

```cpp
#include <iostream>
using namespace std;

int main() {
    int i = 0;
    do {
        cout << "i = " << i << endl;
        i++;
    } while (i < 5);

    return 0;
}
```

##### 3. break와 continue
- **break 사용**

```cpp
#include <iostream>
using namespace std;

int main() {
    for (int i = 0; i < 10; i++) {
        if (i == 5) {
            break;
        }
        cout << "i = " << i << endl;
    }

    return 0;
}
```

- **continue 사용**

```cpp
#include <iostream>
using namespace std;

int main() {
    for (int i = 0; i < 10; i++) {
        if (i == 5) {
            continue;
        }
        cout << "i = " << i << endl;
    }

    return 0;
}
```

#### 과제

1. **숫자 양수/음수 판별 프로그램**
   - 사용자로부터 정수를 입력받아 그 숫자가 양수인지 음수인지 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

int main() {
    int number;
    cout << "Enter an integer: ";
    cin >> number;

    if (number > 0) {
        cout << "The number is positive." << endl;
    } else if (number < 0) {
        cout << "The number is negative." << endl;
    } else {
        cout << "The number is zero." << endl;
    }

    return 0;
}
```

2. **1부터 10까지 출력하는 프로그램**
   - for 루프를 사용하여 1부터 10까지의 숫자를 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

int main() {
    for (int i = 1; i <= 10; i++) {
        cout << i << endl;
    }

    return 0;
}
```

3. **짝수만 출력하는 프로그램**
   - 1부터 20까지의 숫자 중 짝수만 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

int main() {
    for (int i = 1; i <= 20; i++) {
        if (i % 2 == 0) {
            cout << i << endl;
        }
    }

    return 0;
}
```

#### 퀴즈

1. **조건문에 대한 설명 중 틀린 것은?**
   - A) if 문은 조건이 참일 때 코드를 실행한다.
   - B) else 문은 항상 if 문과 함께 사용된다.
   - C) else if 문은 여러 조건을 순차적으로 검사할 때 사용된다.
   - D) switch 문은 정수 값에 대한 조건문이다.

2. **다음 중 반복문에 대한 설명 중 틀린 것은?**
   - A) for 루프는 반복 횟수가 정해져 있을 때 사용된다.
   - B) while 루프는 조건이 참인 동안 반복된다.
   - C) do-while 루프는 조건이 참일 때만 반복된다.
   - D) break 문은 루프를 종료시킨다.

#### 퀴즈 해설

1. **조건문에 대한 설명 중 틀린 것은?**
   - **정답: B) else 문은 항상 if 문과 함께 사용된다.**
     - 해설: else 문은 if 문과 함께 사용되지만, else 문 없이 단독으로 if 문을 사용할 수 있습니다.

2. **다음 중 반복문에 대한 설명 중 틀린 것은?**
   - **정답: C) do-while 루프는 조건이 참일 때만 반복된다.**
     - 해설: do-while 루프는 조건이 참이든 거짓이든 적어도 한 번은 실행됩니다.

다음 주차 강의 내용을 요청하시면, 3주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.