### 10주차: 표준 템플릿 라이브러리 (STL) - 시퀀스 컨테이너

#### 강의 목표
- STL의 개념 이해 및 사용
- 시퀀스 컨테이너 (vector, list, deque)의 사용법 이해
- 반복자의 개념 및 사용법 이해
- STL 알고리즘 개요

#### 강의 내용

##### 1. STL 개요
- **STL의 구성요소**: 컨테이너, 반복자, 알고리즘
- **STL의 장점**: 코드 재사용성, 효율성, 표준화

##### 2. vector
- **vector 선언 및 사용**

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> vec;  // 벡터 선언

    // 값 추가
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);

    // 값 출력
    for (int i = 0; i < vec.size(); i++) {
        cout << "vec[" << i << "] = " << vec[i] << endl;
    }

    return 0;
}
```

- **vector의 주요 메서드**

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> vec = {1, 2, 3, 4, 5};

    cout << "Size: " << vec.size() << endl;
    cout << "Front: " << vec.front() << endl;
    cout << "Back: " << vec.back() << endl;

    vec.pop_back();
    cout << "After pop_back, Size: " << vec.size() << endl;

    return 0;
}
```

##### 3. list
- **list 선언 및 사용**

```cpp
#include <iostream>
#include <list>
using namespace std;

int main() {
    list<int> lst;

    // 값 추가
    lst.push_back(1);
    lst.push_back(2);
    lst.push_back(3);

    // 값 출력
    for (auto it = lst.begin(); it != lst.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    return 0;
}
```

- **list의 주요 메서드**

```cpp
#include <iostream>
#include <list>
using namespace std;

int main() {
    list<int> lst = {1, 2, 3, 4, 5};

    cout << "Size: " << lst.size() << endl;
    cout << "Front: " << lst.front() << endl;
    cout << "Back: " << lst.back() << endl;

    lst.pop_back();
    cout << "After pop_back, Size: " << lst.size() << endl;

    return 0;
}
```

##### 4. deque
- **deque 선언 및 사용**

```cpp
#include <iostream>
#include <deque>
using namespace std;

int main() {
    deque<int> deq;

    // 값 추가
    deq.push_back(1);
    deq.push_back(2);
    deq.push_back(3);
    deq.push_front(0);

    // 값 출력
    for (int i = 0; i < deq.size(); i++) {
        cout << "deq[" << i << "] = " << deq[i] << endl;
    }

    return 0;
}
```

- **deque의 주요 메서드**

```cpp
#include <iostream>
#include <deque>
using namespace std;

int main() {
    deque<int> deq = {1, 2, 3, 4, 5};

    cout << "Size: " << deq.size() << endl;
    cout << "Front: " << deq.front() << endl;
    cout << "Back: " << deq.back() << endl;

    deq.pop_back();
    cout << "After pop_back, Size: " << deq.size() << endl;

    deq.pop_front();
    cout << "After pop_front, Size: " << deq.size() << endl;

    return 0;
}
```

##### 5. 반복자 (Iterator)
- **반복자의 사용**

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> vec = {1, 2, 3, 4, 5};

    for (vector<int>::iterator it = vec.begin(); it != vec.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    return 0;
}
```

##### 6. STL 알고리즘 개요
- **sort, find 등의 기본 알고리즘 사용**

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    vector<int> vec = {5, 3, 1, 4, 2};

    // 정렬
    sort(vec.begin(), vec.end());
    cout << "Sorted vector: ";
    for (int v : vec) {
        cout << v << " ";
    }
    cout << endl;

    // 요소 찾기
    auto it = find(vec.begin(), vec.end(), 3);
    if (it != vec.end()) {
        cout << "Found 3 at position: " << distance(vec.begin(), it) << endl;
    } else {
        cout << "3 not found" << endl;
    }

    return 0;
}
```

#### 과제

1. **vector 사용**
   - 벡터에 정수를 입력받아 저장하고, 모든 요소를 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> vec;
    int num;

    cout << "Enter numbers (enter -1 to stop): ";
    while (true) {
        cin >> num;
        if (num == -1) {
            break;
        }
        vec.push_back(num);
    }

    cout << "You entered: ";
    for (int v : vec) {
        cout << v << " ";
    }
    cout << endl;

    return 0;
}
```

2. **list 사용**
   - 리스트에 정수를 입력받아 저장하고, 모든 요소를 거꾸로 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <list>
using namespace std;

int main() {
    list<int> lst;
    int num;

    cout << "Enter numbers (enter -1 to stop): ";
    while (true) {
        cin >> num;
        if (num == -1) {
            break;
        }
        lst.push_back(num);
    }

    cout << "You entered (in reverse): ";
    for (auto it = lst.rbegin(); it != lst.rend(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    return 0;
}
```

3. **deque 사용**
   - 덱에 정수를 입력받아 저장하고, 앞과 뒤에서 요소를 제거한 후 남은 요소를 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <deque>
using namespace std;

int main() {
    deque<int> deq;
    int num;

    cout << "Enter numbers (enter -1 to stop): ";
    while (true) {
        cin >> num;
        if (num == -1) {
            break;
        }
        deq.push_back(num);
    }

    if (!deq.empty()) {
        deq.pop_front();
    }
    if (!deq.empty()) {
        deq.pop_back();
    }

    cout << "Remaining elements: ";
    for (int d : deq) {
        cout << d << " ";
    }
    cout << endl;

    return 0;
}
```

4. **STL 알고리즘 사용**
   - 벡터에 정수를 입력받아 저장한 후, 벡터를 정렬하고 특정 값을 찾는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    vector<int> vec;
    int num;

    cout << "Enter numbers (enter -1 to stop): ";
    while (true) {
        cin >> num;
        if (num == -1) {
            break;
        }
        vec.push_back(num);
    }

    // 벡터 정렬
    sort(vec.begin(), vec.end());

    cout << "Sorted vector: ";
    for (int v : vec) {
        cout << v << " ";
    }
    cout << endl;

    // 요소 찾기
    cout << "Enter a number to find: ";
    cin >> num;
    auto it = find(vec.begin(), vec.end(), num);
    if (it != vec.end()) {
        cout << "Found " << num << " at position: " << distance(vec.begin(), it) << endl;
    } else {
        cout << num << " not found" << endl;
    }

    return 0;
}
```

#### 퀴즈

1. **STL의 구성요소가 아닌 것은?**
   - A) 컨테이너
   - B) 반복자
   - C) 알고리즘
   - D) 모듈

2. **vector에 대한 설명 중 맞는 것은?**
   - A) vector는 고정된 크기의 배열이다.
   - B) vector는 동적 배열로 크기가 자동으로 조정된다.
   - C) vector

는 요소를 앞에 삽입할 수 있다.
   - D) vector는 중복된 값을 허용하지 않는다.

3. **deque에 대한 설명 중 틀린 것은?**
   - A) deque는 양 끝에서 요소를 삽입하고 제거할 수 있다.
   - B) deque는 크기가 고정되어 있다.
   - C) deque는 vector보다 더 효율적인 요소 접근을 제공한다.
   - D) deque는 double-ended queue의 줄임말이다.

4. **STL 알고리즘에 대한 설명 중 맞는 것은?**
   - A) STL 알고리즘은 특정 컨테이너에만 적용된다.
   - B) STL 알고리즘은 컨테이너의 요소를 조작할 수 있다.
   - C) STL 알고리즘은 반복자를 사용하지 않는다.
   - D) STL 알고리즘은 사용자 정의 함수와 함께 사용할 수 없다.

#### 퀴즈 해설

1. **STL의 구성요소가 아닌 것은?**
   - **정답: D) 모듈**
     - 해설: STL은 컨테이너, 반복자, 알고리즘으로 구성됩니다.

2. **vector에 대한 설명 중 맞는 것은?**
   - **정답: B) vector는 동적 배열로 크기가 자동으로 조정된다.**
     - 해설: vector는 동적 배열로, 요소를 추가하거나 제거할 때 크기가 자동으로 조정됩니다.

3. **deque에 대한 설명 중 틀린 것은?**
   - **정답: B) deque는 크기가 고정되어 있다.**
     - 해설: deque는 크기가 동적으로 조정되며, 양 끝에서 요소를 삽입하고 제거할 수 있습니다.

4. **STL 알고리즘에 대한 설명 중 맞는 것은?**
   - **정답: B) STL 알고리즘은 컨테이너의 요소를 조작할 수 있다.**
     - 해설: STL 알고리즘은 반복자를 사용하여 컨테이너의 요소를 조작할 수 있습니다.

다음 주차 강의 내용을 요청하시면, 11주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.