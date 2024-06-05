### 4주차: 배열과 문자열

#### 강의 목표
- 배열의 기초 개념과 사용법 이해
- 다차원 배열의 사용법 익히기
- 문자열 처리 방법 이해 (C 스타일 문자열과 string 클래스)

#### 강의 내용

##### 1. 1차원 배열
- **배열 선언 및 초기화**

```cpp
#include <iostream>
using namespace std;

int main() {
    int numbers[5] = {1, 2, 3, 4, 5};

    for (int i = 0; i < 5; i++) {
        cout << "numbers[" << i << "] = " << numbers[i] << endl;
    }

    return 0;
}
```

- **배열의 크기와 범위**

```cpp
#include <iostream>
using namespace std;

int main() {
    int numbers[5];
    cout << "Enter 5 numbers: ";

    for (int i = 0; i < 5; i++) {
        cin >> numbers[i];
    }

    cout << "You entered: ";
    for (int i = 0; i < 5; i++) {
        cout << numbers[i] << " ";
    }

    cout << endl;
    return 0;
}
```

##### 2. 다차원 배열
- **2차원 배열 선언 및 초기화**

```cpp
#include <iostream>
using namespace std;

int main() {
    int matrix[2][3] = {
        {1, 2, 3},
        {4, 5, 6}
    };

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            cout << "matrix[" << i << "][" << j << "] = " << matrix[i][j] << endl;
        }
    }

    return 0;
}
```

- **다차원 배열의 활용**

```cpp
#include <iostream>
using namespace std;

int main() {
    int matrix[2][3];
    cout << "Enter 6 numbers: ";

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            cin >> matrix[i][j];
        }
    }

    cout << "You entered: " << endl;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}
```

##### 3. 문자열 처리
- **C 스타일 문자열**

```cpp
#include <iostream>
#include <cstring>  // 문자열 처리를 위한 헤더 파일
using namespace std;

int main() {
    char str1[20] = "Hello";
    char str2[20];

    strcpy(str2, str1);  // 문자열 복사
    cout << "str2: " << str2 << endl;

    strcat(str2, ", World!");  // 문자열 이어붙이기
    cout << "str2: " << str2 << endl;

    int len = strlen(str2);  // 문자열 길이
    cout << "Length of str2: " << len << endl;

    return 0;
}
```

- **string 클래스**

```cpp
#include <iostream>
#include <string>  // string 클래스를 사용하기 위한 헤더 파일
using namespace std;

int main() {
    string str1 = "Hello";
    string str2;

    str2 = str1;  // 문자열 복사
    cout << "str2: " << str2 << endl;

    str2 += ", World!";  // 문자열 이어붙이기
    cout << "str2: " << str2 << endl;

    int len = str2.length();  // 문자열 길이
    cout << "Length of str2: " << len << endl;

    return 0;
}
```

#### 과제

1. **배열의 합 구하기**
   - 10개의 정수를 입력받아 배열에 저장한 후, 배열의 합을 계산하는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

int main() {
    int numbers[10], sum = 0;
    cout << "Enter 10 numbers: ";

    for (int i = 0; i < 10; i++) {
        cin >> numbers[i];
        sum += numbers[i];
    }

    cout << "Sum of the numbers: " << sum << endl;
    return 0;
}
```

2. **2차원 배열 평균 구하기**
   - 3x3 행렬의 원소를 입력받아 배열에 저장한 후, 각 행의 평균을 계산하여 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

int main() {
    int matrix[3][3];
    cout << "Enter 9 numbers for the 3x3 matrix: ";

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cin >> matrix[i][j];
        }
    }

    for (int i = 0; i < 3; i++) {
        int sum = 0;
        for (int j = 0; j < 3; j++) {
            sum += matrix[i][j];
        }
        cout << "Average of row " << i + 1 << ": " << sum / 3.0 << endl;
    }

    return 0;
}
```

3. **문자열 길이 비교**
   - 두 문자열을 입력받아 각 문자열의 길이를 비교하고, 어느 문자열이 더 긴지 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    string str1, str2;
    cout << "Enter first string: ";
    cin >> str1;
    cout << "Enter second string: ";
    cin >> str2;

    if (str1.length() > str2.length()) {
        cout << "First string is longer." << endl;
    } else if (str1.length() < str2.length()) {
        cout << "Second string is longer." << endl;
    } else {
        cout << "Both strings are of equal length." << endl;
    }

    return 0;
}
```

#### 퀴즈

1. **배열에 대한 설명 중 틀린 것은?**
   - A) 배열은 같은 데이터 타입의 집합을 저장하는 데 사용된다.
   - B) 배열의 크기는 선언 시 결정되며, 이후 변경할 수 없다.
   - C) 배열 요소는 0부터 시작하는 인덱스로 접근할 수 있다.
   - D) 배열의 크기는 실행 중에 동적으로 변경될 수 있다.

2. **다음 중 2차원 배열의 선언이 올바른 것은?**
   - A) int matrix[2, 3];
   - B) int matrix[2][3];
   - C) int matrix{2}{3};
   - D) int matrix[2x3];

3. **C 스타일 문자열에 대한 설명 중 맞는 것은?**
   - A) 문자열의 끝은 NULL 문자로 끝난다.
   - B) 문자열은 항상 동적 할당되어야 한다.
   - C) 문자열의 길이는 strlen 함수로 구할 수 없다.
   - D) 문자열 복사에는 strcpy 함수를 사용할 수 없다.

#### 퀴즈 해설

1. **배열에 대한 설명 중 틀린 것은?**
   - **정답: D) 배열의 크기는 실행 중에 동적으로 변경될 수 있다.**
     - 해설: 배열의 크기는 선언 시 결정되며, 실행 중에 변경할 수 없습니다. 크기를 변경하려면 동적 배열이나 벡터를 사용해야 합니다.

2. **다음 중 2차원 배열의 선언이 올바른 것은?**
   - **정답: B) int matrix[2][3];**
     - 해설: 2차원 배열은 각 차원의 크기를 대괄호로 지정하여 선언합니다.

3. **C 스타일 문자열에 대한 설명 중 맞는 것은?**
   - **정답: A) 문자열의 끝은 NULL 문자로 끝난다.**
     - 해설: C 스타일 문자열은 NULL 문자 (`'\0'`)로 끝나야 합니다. `strlen` 함수는 문자열의 길이를 구할 수 있으며, `strcpy` 함수는 문자열 복사에 사용됩니다.

다음 주차 강의 내용을 요청하시면, 5주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.