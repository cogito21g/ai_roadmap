### 5주차: 포인터와 참조

#### 강의 목표
- 포인터의 개념과 사용법 이해
- 참조자의 개념과 사용법 이해
- 포인터와 배열의 관계 이해
- 동적 메모리 할당 이해 및 사용

#### 강의 내용

##### 1. 포인터 기초
- **포인터 선언 및 사용**

```cpp
#include <iostream>
using namespace std;

int main() {
    int var = 20;
    int* ptr = &var;  // 포인터 선언 및 초기화

    cout << "Value of var: " << var << endl;
    cout << "Address of var: " << &var << endl;
    cout << "Value of ptr: " << ptr << endl;
    cout << "Value pointed to by ptr: " << *ptr << endl;

    return 0;
}
```

- **포인터 연산**

```cpp
#include <iostream>
using namespace std;

int main() {
    int arr[5] = {10, 20, 30, 40, 50};
    int* ptr = arr;

    for (int i = 0; i < 5; i++) {
        cout << "Value of arr[" << i << "]: " << *(ptr + i) << endl;
    }

    return 0;
}
```

##### 2. 참조자 (References)
- **참조자의 선언 및 사용**

```cpp
#include <iostream>
using namespace std;

int main() {
    int var = 10;
    int& ref = var;  // 참조자 선언

    cout << "Value of var: " << var << endl;
    cout << "Value of ref: " << ref << endl;

    ref = 20;  // 참조자를 통해 변수 값 변경
    cout << "New value of var: " << var << endl;
    cout << "New value of ref: " << ref << endl;

    return 0;
}
```

##### 3. 포인터와 배열
- **배열과 포인터의 관계**

```cpp
#include <iostream>
using namespace std;

int main() {
    int arr[5] = {10, 20, 30, 40, 50};
    int* ptr = arr;  // 배열 이름은 배열의 첫 번째 요소의 주소를 가리키는 포인터

    for (int i = 0; i < 5; i++) {
        cout << "arr[" << i << "] = " << *(ptr + i) << endl;
    }

    return 0;
}
```

##### 4. 동적 메모리 할당 (new, delete)
- **동적 메모리 할당 및 해제**

```cpp
#include <iostream>
using namespace std;

int main() {
    int* ptr = new int;  // 정수형 메모리 할당
    *ptr = 10;
    cout << "Value at allocated memory: " << *ptr << endl;

    delete ptr;  // 메모리 해제

    int* arr = new int[5];  // 정수형 배열 메모리 할당
    for (int i = 0; i < 5; i++) {
        arr[i] = i + 1;
    }

    cout << "Values in allocated array: ";
    for (int i = 0; i < 5; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    delete[] arr;  // 배열 메모리 해제

    return 0;
}
```

#### 과제

1. **포인터와 변수**
   - 변수를 선언하고, 해당 변수의 주소를 가리키는 포인터를 사용하여 변수의 값을 변경하는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

int main() {
    int var = 10;
    int* ptr = &var;

    cout << "Original value of var: " << var << endl;
    *ptr = 20;  // 포인터를 통해 변수 값 변경
    cout << "New value of var: " << var << endl;

    return 0;
}
```

2. **참조자 사용**
   - 변수를 선언하고, 참조자를 사용하여 변수의 값을 변경하는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

int main() {
    int var = 10;
    int& ref = var;

    cout << "Original value of var: " << var << endl;
    ref = 20;  // 참조자를 통해 변수 값 변경
    cout << "New value of var: " << var << endl;

    return 0;
}
```

3. **동적 배열 할당**
   - 사용자로부터 배열의 크기를 입력받아 동적으로 배열을 할당하고, 값을 입력받아 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

int main() {
    int size;
    cout << "Enter the size of the array: ";
    cin >> size;

    int* arr = new int[size];  // 동적 배열 할당

    cout << "Enter " << size << " numbers: ";
    for (int i = 0; i < size; i++) {
        cin >> arr[i];
    }

    cout << "You entered: ";
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    delete[] arr;  // 메모리 해제

    return 0;
}
```

#### 퀴즈

1. **포인터에 대한 설명 중 틀린 것은?**
   - A) 포인터는 메모리 주소를 저장하는 변수이다.
   - B) 포인터는 변수의 주소를 가리킬 수 있다.
   - C) 포인터 연산은 주소를 기반으로 한다.
   - D) 포인터는 항상 정적 메모리만 가리킬 수 있다.

2. **참조자에 대한 설명 중 틀린 것은?**
   - A) 참조자는 변수의 별명이다.
   - B) 참조자는 선언 시 반드시 초기화해야 한다.
   - C) 참조자는 NULL 값을 가질 수 있다.
   - D) 참조자를 통해 변수의 값을 변경할 수 있다.

3. **동적 메모리 할당에 대한 설명 중 맞는 것은?**
   - A) new 연산자는 정적 메모리를 할당한다.
   - B) 동적으로 할당된 메모리는 delete 연산자를 사용하여 해제해야 한다.
   - C) 동적으로 할당된 메모리는 프로그램 종료 시 자동으로 해제된다.
   - D) 동적 메모리 할당은 항상 실패하지 않는다.

#### 퀴즈 해설

1. **포인터에 대한 설명 중 틀린 것은?**
   - **정답: D) 포인터는 항상 정적 메모리만 가리킬 수 있다.**
     - 해설: 포인터는 정적 메모리뿐만 아니라 동적 메모리도 가리킬 수 있습니다.

2. **참조자에 대한 설명 중 틀린 것은?**
   - **정답: C) 참조자는 NULL 값을 가질 수 있다.**
     - 해설: 참조자는 NULL 값을 가질 수 없으며, 선언 시 반드시 초기화해야 합니다.

3. **동적 메모리 할당에 대한 설명 중 맞는 것은?**
   - **정답: B) 동적으로 할당된 메모리는 delete 연산자를 사용하여 해제해야 한다.**
     - 해설: 동적 메모리는 사용 후 반드시 delete 연산자를 사용하여 해제해야 메모리 누수를 방지할 수 있습니다.

다음 주차 강의 내용을 요청하시면, 6주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.