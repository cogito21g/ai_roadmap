### 5주차 강의 계획: 배열과 벡터

#### 강의 목표
- 배열의 기본 개념과 사용법 이해
- 배열의 장단점 파악
- 벡터의 개념과 STL 벡터 사용법 학습
- 배열과 벡터의 차이점 이해

#### 강의 구성
- **강의 시간**: 2시간
- **구성**: 배열 이론 (30분), 벡터 이론 (30분), 실습 및 과제 안내 (30분), 퀴즈 및 해설 (30분)

#### 강의 내용

##### 1. 배열 이론 (30분)

###### 1.1 배열의 기본 개념
- **배열의 정의**:
  - 배열의 구조 및 특징
  - 고정 크기의 연속된 메모리 블록
- **예제**:
```cpp
#include <iostream>
using namespace std;

int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    for (int i = 0; i < 5; ++i) {
        cout << arr[i] << " ";
    }
    cout << endl;
    return 0;
}
```

###### 1.2 배열의 장단점
- **장점**:
  - 인덱스를 통한 빠른 접근
  - 메모리 연속성
- **단점**:
  - 크기가 고정되어 있음
  - 삽입 및 삭제가 비효율적

##### 2. 벡터 이론 (30분)

###### 2.1 벡터의 기본 개념
- **벡터의 정의**:
  - 동적 배열
  - 자동 크기 조정 기능
- **예제**:
```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> vec = {1, 2, 3, 4, 5};
    vec.push_back(6);
    for (int i = 0; i < vec.size(); ++i) {
        cout << vec[i] << " ";
    }
    cout << endl;
    return 0;
}
```

###### 2.2 STL 벡터 사용법
- **벡터 초기화 및 접근**:
  - 벡터 선언 및 초기화
  - 벡터 요소 접근
- **벡터의 주요 메서드**:
  - `push_back()`, `pop_back()`, `size()`, `empty()`, `clear()`
- **예제**:
```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);
    cout << "Vector size: " << vec.size() << endl;
    cout << "Vector elements: ";
    for (int i = 0; i < vec.size(); ++i) {
        cout << vec[i] << " ";
    }
    cout << endl;
    vec.pop_back();
    cout << "Vector after pop_back: ";
    for (int i = 0; i < vec.size(); ++i) {
        cout << vec[i] << " ";
    }
    cout << endl;
    return 0;
}
```

###### 2.3 배열과 벡터의 차이점
- **배열**:
  - 고정 크기
  - 메모리 연속성
  - 빠른 접근 속도
- **벡터**:
  - 동적 크기 조정
  - 삽입 및 삭제 용이
  - STL의 다양한 기능 제공

##### 3. 실습 및 과제 안내 (30분)

###### 3.1 실습
- **실습 목표**:
  - 배열과 벡터를 사용한 프로그램 작성
- **실습 문제**:
  - 정수 배열을 입력받아 합계와 평균을 구하는 프로그램 작성
  - 벡터를 사용하여 사용자 입력을 저장하고 역순으로 출력하는 프로그램 작성

###### 3.2 과제 안내
- **과제 내용**:
  - 배열을 사용하여 10개의 정수를 입력받고, 최대값과 최소값을 구하는 프로그램 작성
  - 벡터를 사용하여 문자열을 저장하고, 알파벳 순서로 정렬한 후 출력하는 프로그램 작성
- **과제 제출 방법**:
  - 작성한 코드를 주차별 과제 제출 폴더에 업로드
  - 제출 기한: 다음 주 강의 전까지

##### 4. 퀴즈 및 해설 (30분)

###### 퀴즈 문제:
1. 배열의 크기를 동적으로 조정할 수 있는 것은?
   - a) int array[10]
   - b) vector<int>
   - c) float array[20]
   - d) char array[5]
2. 벡터의 마지막 요소를 제거하는 메서드는?
   - a) remove()
   - b) delete()
   - c) pop_back()
   - d) erase()
3. 배열의 단점이 아닌 것은?
   - a) 크기가 고정되어 있다
   - b) 삽입 및 삭제가 비효율적이다
   - c) 메모리 연속성을 가진다
   - d) 인덱스를 통한 접근이 가능하다

###### 퀴즈 해설:
1. **정답: b) vector<int>**
   - 벡터는 동적으로 크기를 조정할 수 있는 동적 배열입니다.
2. **정답: c) pop_back()**
   - `pop_back()` 메서드는 벡터의 마지막 요소를 제거합니다.
3. **정답: d) 인덱스를 통한 접근이 가능하다**
   - 배열은 인덱스를 통한 접근이 가능하다는 점에서 단점이 아닙니다.

### 준비 자료
- 강의 슬라이드
- 실습 코드 예제
- 과제 안내서

### 과제 해설

##### 과제: 배열을 사용하여 10개의 정수를 입력받고, 최대값과 최소값을 구하는 프로그램 작성
- **문제**: 사용자로부터 10개의 정수를 입력받아 배열에 저장하고, 최대값과 최소값을 찾아 출력하는 프로그램 작성
- **해설**:
  - 배열을 사용하여 10개의 정수를 저장합니다.
  - 반복문을 사용하여 최대값과 최소값을 찾습니다.

```cpp
#include <iostream>
using namespace std;

int main() {
    int arr[10];
    cout << "Enter 10 integers: ";
    for (int i = 0; i < 10; ++i) {
        cin >> arr[i];
    }

    int max = arr[0];
    int min = arr[0];
    for (int i = 1; i < 10; ++i) {
        if (arr[i] > max) {
            max = arr[i];
        }
        if (arr[i] < min) {
            min = arr[i];
        }
    }

    cout << "Maximum value: " << max << endl;
    cout << "Minimum value: " << min << endl;

    return 0;
}
```
- **설명**:
  - 배열 `arr`를 선언하고 10개의 정수를 입력받습니다.
  - 초기 최대값과 최소값을 배열의 첫 번째 요소로 설정합니다.
  - 반복문을 통해 배열의 나머지 요소와 비교하여 최대값과 최소값을 업데이트합니다.
  - 결과를 출력합니다.

##### 과제: 벡터를 사용하여 문자열을 저장하고, 알파벳 순서로 정렬한 후 출력하는 프로그램 작성
- **문제**: 사용자로부터 문자열을 입력받아 벡터에 저장하고, 알파벳 순서로 정렬한 후 출력하는 프로그램 작성
- **해설**:
  - 벡터를 사용하여 문자열을 저장합니다.
  - `sort` 함수로 벡터를 정렬합니다.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    vector<string> vec;
    string input;
    cout << "Enter strings (type 'end' to stop): ";
    while (cin >> input && input != "end") {
        vec.push_back(input);
    }

    sort(vec.begin(), vec.end());

    cout << "Sorted strings: ";
    for (const auto& str : vec) {
        cout << str << " ";
    }
    cout << endl;

    return 0;
}
```
- **설명**:
  - 벡터 `vec`를 선언하고 사용자로부터 문자열을 입력받습니다.
  - `sort` 함수를 사용하여 벡터를 알파벳 순서로 정렬합니다.
  - 정렬된 문자열을 출력합니다.

이로써 5주차 강의가 마무리됩니다. 학생들은 배열과 벡터의 기본 개념을 이해하고, 이를 활용하여 다양한 프로그램을 작성하는 능력을 기르게 됩니다.