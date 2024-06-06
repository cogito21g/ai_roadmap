### 2주차 강의 계획: 함수 및 포인터

#### 강의 목표
- 함수의 정의와 호출 방식 이해
- 함수의 매개변수 전달 방식 (값에 의한 전달, 참조에 의한 전달) 이해
- 포인터의 개념 및 사용법 익히기
- 동적 메모리 할당의 기초 이해

#### 강의 구성
- **강의 시간**: 2시간
- **구성**: 함수 이론 (30분), 포인터 이론 (30분), 실습 및 과제 안내 (30분), 퀴즈 및 해설 (30분)

#### 강의 내용

##### 1. 함수 이론 (30분)

###### 1.1 함수의 정의와 호출
- **함수 정의**:
  - 함수의 기본 구조 (반환형, 함수명, 매개변수, 본문)
- **예제**:
```cpp
#include <iostream>
using namespace std;

int add(int a, int b) {
    return a + b;
}

int main() {
    int x = 5, y = 10;
    cout << "Sum: " << add(x, y) << endl;
    return 0;
}
```

###### 1.2 함수의 매개변수 전달 방식
- **값에 의한 전달**:
  - 매개변수로 값을 복사하여 전달
- **참조에 의한 전달**:
  - 매개변수로 변수의 주소를 전달 (참조자 사용)
- **예제**:
```cpp
#include <iostream>
using namespace std;

void swapByValue(int a, int b) {
    int temp = a;
    a = b;
    b = temp;
}

void swapByReference(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

int main() {
    int x = 5, y = 10;
    swapByValue(x, y);
    cout << "After swapByValue: x = " << x << ", y = " << y << endl;
    swapByReference(x, y);
    cout << "After swapByReference: x = " << x << ", y = " << y << endl;
    return 0;
}
```

##### 2. 포인터 이론 (30분)

###### 2.1 포인터의 기본 개념
- **포인터 정의 및 사용법**:
  - 포인터 선언 및 초기화
  - 포인터 연산 (주소 연산자 &, 간접 연산자 *)
- **예제**:
```cpp
#include <iostream>
using namespace std;

int main() {
    int var = 10;
    int *ptr = &var;
    cout << "Value of var: " << var << endl;
    cout << "Address of var: " << &var << endl;
    cout << "Pointer ptr points to address: " << ptr << endl;
    cout << "Value pointed to by ptr: " << *ptr << endl;
    return 0;
}
```

###### 2.2 동적 메모리 할당
- **동적 메모리 할당**:
  - `new` 연산자를 사용한 메모리 할당
  - `delete` 연산자를 사용한 메모리 해제
- **예제**:
```cpp
#include <iostream>
using namespace std;

int main() {
    int *ptr = new int;
    *ptr = 10;
    cout << "Value of dynamically allocated memory: " << *ptr << endl;
    delete ptr;
    return 0;
}
```

##### 3. 실습 및 과제 안내 (30분)

###### 3.1 실습
- **실습 목표**:
  - 강의에서 다룬 내용을 직접 코드로 작성해보기
- **실습 문제**:
  - 함수와 포인터를 사용한 간단한 프로그램 작성

###### 3.2 과제 안내
- **과제 내용**:
  - 사용자로부터 두 정수를 입력받아, 두 수를 교환하는 프로그램 작성
  - 함수로 구현 (값에 의한 전달과 참조에 의한 전달 각각 구현)
  - 포인터를 사용하여 동적 메모리 할당을 활용한 프로그램 작성
- **과제 제출 방법**:
  - 작성한 코드를 주차별 과제 제출 폴더에 업로드
  - 제출 기한: 다음 주 강의 전까지

##### 4. 퀴즈 및 해설 (30분)

###### 퀴즈 문제:
1. 함수의 반환형이 없는 경우 사용되는 키워드는?
   - a) void
   - b) int
   - c) null
   - d) none
2. 포인터를 선언할 때 사용되는 연산자는?
   - a) &
   - b) *
   - c) %
   - d) @
3. 동적 메모리 할당 시 사용하는 연산자는?
   - a) new
   - b) malloc
   - c) allocate
   - d) dynamic

###### 퀴즈 해설:
1. **정답: a) void**
   - 반환형이 없는 함수는 `void` 키워드를 사용하여 정의합니다.
2. **정답: b) ***
   - 포인터를 선언할 때는 `*` 연산자를 사용합니다.
3. **정답: a) new**
   - C++에서는 동적 메모리 할당 시 `new` 연산자를 사용합니다.

### 준비 자료
- 강의 슬라이드
- 실습 코드 예제
- 과제 안내서

### 과제 해설

##### 과제: 두 정수를 교환하는 프로그램 작성
- **문제**: 사용자로부터 두 개의 정수를 입력받아, 두 수를 교환하는 프로그램 작성
- **해설**:
  - 값을 교환하는 두 가지 방법을 함수로 구현
  - 값을 교환하는 과정을 확인하는 프로그램 작성

```cpp
#include <iostream>
using namespace std;

void swapByValue(int a, int b) {
    int temp = a;
    a = b;
    b = temp;
    cout << "Inside swapByValue: a = " << a << ", b = " << b << endl;
}

void swapByReference(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

int main() {
    int x, y;
    cout << "Enter two numbers: ";
    cin >> x >> y;

    cout << "Before swap: x = " << x << ", y = " << y << endl;
    swapByValue(x, y);
    cout << "After swapByValue: x = " << x << ", y = " << y << endl;
    
    swapByReference(x, y);
    cout << "After swapByReference: x = " << x << ", y = " << y << endl;

    return 0;
}
```
- **설명**:
  - `swapByValue` 함수는 값에 의한 전달로 교환을 시도하므로 원본 값은 변경되지 않습니다.
  - `swapByReference` 함수는 참조에 의한 전달로 교환을 시도하므로 원본 값이 변경됩니다.
  - 결과 출력으로 두 함수의 차이를 확인할 수 있습니다.

이로써 2주차 강의가 마무리됩니다. 학생들은 함수와 포인터의 기본 개념을 이해하고, 이를 활용하여 간단한 프로그램을 작성하는 능력을 기르게 됩니다.