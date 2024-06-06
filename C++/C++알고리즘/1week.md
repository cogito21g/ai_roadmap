### 1주차 강의 계획안

#### 강의 주제: 오리엔테이션 및 환경 설정
- 강의 소개 및 학습 목표 설정
- C++ 개발 환경 설정 (IDE 설치, 컴파일러 설정 등)
- 기본 C++ 문법 리뷰 (함수, 포인터, 메모리 관리 등)

---

#### 1. 강의 소개 및 학습 목표 설정
- 강의 목표: 알고리즘의 기본 개념을 이해하고, C++를 사용하여 다양한 알고리즘을 구현할 수 있는 능력을 기릅니다.
- 학습 목표: 기초 문법을 확실히 이해하고, 앞으로 진행될 알고리즘 강의에 필요한 C++ 개발 환경을 설정합니다.

#### 2. C++ 개발 환경 설정
- **IDE 설치**: Visual Studio, Code::Blocks, CLion 중 하나 선택
- **컴파일러 설정**: GCC (MinGW) 설치 및 환경 변수 설정

#### 3. 기본 C++ 문법 리뷰
- **변수 및 자료형**: int, float, double, char, bool
- **조건문**: if, else if, else
- **반복문**: for, while, do-while
- **함수**: 함수 선언과 정의, 매개변수 전달, 반환값
- **포인터**: 포인터 선언과 사용, 주소 연산자 (&), 포인터 연산자 (*)
- **메모리 관리**: 동적 메모리 할당 (new, delete)

---

### 과제

#### 과제 1: 간단한 계산기 프로그램 작성
간단한 사칙연산(덧셈, 뺄셈, 곱셈, 나눗셈)을 수행하는 프로그램을 작성하세요. 사용자로부터 두 개의 숫자와 연산자를 입력받아 결과를 출력합니다.

**코드 예시**:
```cpp
#include <iostream>
using namespace std;

int main() {
    char op;
    double num1, num2;
    
    cout << "Enter operator (+, -, *, /): ";
    cin >> op;
    cout << "Enter two operands: ";
    cin >> num1 >> num2;
    
    switch(op) {
        case '+':
            cout << num1 << " + " << num2 << " = " << num1 + num2 << endl;
            break;
        case '-':
            cout << num1 << " - " << num2 << " = " << num1 - num2 << endl;
            break;
        case '*':
            cout << num1 << " * " << num2 << " = " << num1 * num2 << endl;
            break;
        case '/':
            if(num2 != 0)
                cout << num1 << " / " << num2 << " = " << num1 / num2 << endl;
            else
                cout << "Division by zero error!" << endl;
            break;
        default:
            cout << "Invalid operator!" << endl;
            break;
    }

    return 0;
}
```

**해설**:
1. `char op;` : 연산자를 저장할 변수입니다.
2. `double num1, num2;` : 두 개의 피연산자를 저장할 변수입니다.
3. `cin` : 사용자로부터 입력을 받습니다.
4. `switch` : 입력된 연산자에 따라 알맞은 연산을 수행합니다.
5. 나누기 연산에서는 0으로 나누는 경우를 처리하여 오류를 방지합니다.

#### 과제 2: 동적 배열 할당과 사용
사용자로부터 배열의 크기를 입력받고, 그 크기의 동적 배열을 생성한 후, 배열의 각 요소를 사용자로부터 입력받아 출력하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
using namespace std;

int main() {
    int size;
    cout << "Enter the size of the array: ";
    cin >> size;
    
    int* arr = new int[size];
    
    cout << "Enter " << size << " elements:" << endl;
    for(int i = 0; i < size; i++) {
        cin >> arr[i];
    }
    
    cout << "You entered: ";
    for(int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
    
    delete[] arr;
    return 0;
}
```

**해설**:
1. `int size;` : 배열의 크기를 저장할 변수입니다.
2. `new int[size];` : 입력받은 크기만큼의 동적 배열을 할당합니다.
3. `cin` : 배열의 각 요소를 사용자로부터 입력받습니다.
4. `delete[] arr;` : 할당한 동적 배열을 해제합니다.

---

### 퀴즈

#### 퀴즈 1: 다음 중 포인터에 대한 설명으로 옳은 것은 무엇인가요?
1. 포인터는 변수의 주소를 저장합니다.
2. 포인터는 변수의 값을 저장합니다.
3. 포인터는 배열의 크기를 저장합니다.
4. 포인터는 문자열의 길이를 저장합니다.

**정답**: 1. 포인터는 변수의 주소를 저장합니다.

#### 퀴즈 2: 다음 코드의 출력 결과는 무엇인가요?
```cpp
#include <iostream>
using namespace std;

void swap(int *xp, int *yp) {
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}

int main() {
    int x = 10, y = 20;
    swap(&x, &y);
    cout << "x = " << x << ", y = " << y << endl;
    return 0;
}
```

**정답**: `x = 20, y = 10`

**해설**:
- `swap` 함수는 두 변수의 값을 서로 교환합니다.
- `&x`와 `&y`는 변수 `x`와 `y`의 주소를 전달합니다.
- 함수 내에서 포인터를 통해 변수의 값을 교환합니다.

이 계획안은 1주차에 필요한 기본적인 개념을 잘 이해하고, 과제를 통해 실습을 할 수 있도록 구성되었습니다. 과제와 퀴즈를 통해 학습한 내용을 복습하고, 이해도를 높일 수 있습니다.