### 21주차: 고급 C++ 기법

#### 강의 목표
- 템플릿 메타프로그래밍의 이해 및 사용
- 표준 라이브러리 심화 학습
- 고급 STL 사용법 익히기

#### 강의 내용

##### 1. 템플릿 메타프로그래밍
- **템플릿 재귀와 조건**

```cpp
#include <iostream>
using namespace std;

template<int N>
struct Factorial {
    static const int value = N * Factorial<N - 1>::value;
};

template<>
struct Factorial<0> {
    static const int value = 1;
};

int main() {
    cout << "Factorial of 5: " << Factorial<5>::value << endl;
    return 0;
}
```

- **SFINAE (Substitution Failure Is Not An Error)**

```cpp
#include <iostream>
#include <type_traits>
using namespace std;

template<typename T>
typename enable_if<is_integral<T>::value, T>::type
foo(T t) {
    cout << "Integral type" << endl;
    return t;
}

template<typename T>
typename enable_if<!is_integral<T>::value, T>::type
foo(T t) {
    cout << "Non-integral type" << endl;
    return t;
}

int main() {
    foo(10);  // Integral type
    foo(10.5);  // Non-integral type
    return 0;
}
```

- **템플릿 특수화**

```cpp
#include <iostream>
using namespace std;

template<typename T>
class TypeInfo {
public:
    static void print() {
        cout << "Unknown type" << endl;
    }
};

template<>
class TypeInfo<int> {
public:
    static void print() {
        cout << "Type is int" << endl;
    }
};

template<>
class TypeInfo<double> {
public:
    static void print() {
        cout << "Type is double" << endl;
    }
};

int main() {
    TypeInfo<int>::print();
    TypeInfo<double>::print();
    TypeInfo<char>::print();
    return 0;
}
```

##### 2. 표준 라이브러리 심화 학습
- **std::function 및 std::bind**

```cpp
#include <iostream>
#include <functional>
using namespace std;

void printMessage(const string& message) {
    cout << message << endl;
}

int main() {
    function<void(const string&)> func = printMessage;
    func("Hello, std::function!");

    auto boundFunc = bind(printMessage, "Hello, std::bind!");
    boundFunc();

    return 0;
}
```

- **std::any, std::variant 사용법**

```cpp
#include <iostream>
#include <any>
#include <variant>
using namespace std;

int main() {
    any a = 10;
    cout << any_cast<int>(a) << endl;

    a = string("Hello, any!");
    cout << any_cast<string>(a) << endl;

    variant<int, float, string> v;
    v = 10;
    cout << get<int>(v) << endl;

    v = 3.14f;
    cout << get<float>(v) << endl;

    v = "Hello, variant!";
    cout << get<string>(v) << endl;

    return 0;
}
```

- **std::filesystem 사용법**

```cpp
#include <iostream>
#include <filesystem>
using namespace std;
namespace fs = std::filesystem;

int main() {
    string path = ".";
    for (const auto& entry : fs::directory_iterator(path)) {
        cout << entry.path() << endl;
    }
    return 0;
}
```

##### 3. 고급 STL 사용법
- **STL 알고리즘 최적화**

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    vector<int> vec = {4, 2, 5, 2, 3, 1, 4, 5};

    // 정렬
    sort(vec.begin(), vec.end());

    // 중복 제거
    auto last = unique(vec.begin(), vec.end());
    vec.erase(last, vec.end());

    cout << "Sorted and unique elements: ";
    for (int val : vec) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}
```

- **커스텀 컨테이너 구현**

```cpp
#include <iostream>
#include <vector>
using namespace std;

template<typename T>
class CustomContainer {
private:
    vector<T> data;

public:
    void add(const T& value) {
        data.push_back(value);
    }

    void remove() {
        if (!data.empty()) {
            data.pop_back();
        }
    }

    void print() const {
        for (const auto& val : data) {
            cout << val << " ";
        }
        cout << endl;
    }
};

int main() {
    CustomContainer<int> container;
    container.add(1);
    container.add(2);
    container.add(3);
    container.print();

    container.remove();
    container.print();

    return 0;
}
```

- **고급 반복자 및 어댑터**

```cpp
#include <iostream>
#include <vector>
#include <iterator>
using namespace std;

int main() {
    vector<int> vec = {1, 2, 3, 4, 5};

    cout << "Original vector: ";
    copy(vec.begin(), vec.end(), ostream_iterator<int>(cout, " "));
    cout << endl;

    cout << "Reversed vector: ";
    copy(vec.rbegin(), vec.rend(), ostream_iterator<int>(cout, " "));
    cout << endl;

    return 0;
}
```

#### 과제

1. **템플릿 메타프로그래밍을 사용하여 피보나치 수열을 계산하는 프로그램을 작성하세요.**

```cpp
#include <iostream>
using namespace std;

template<int N>
struct Fibonacci {
    static const int value = Fibonacci<N - 1>::value + Fibonacci<N - 2>::value;
};

template<>
struct Fibonacci<0> {
    static const int value = 0;
};

template<>
struct Fibonacci<1> {
    static const int value = 1;
};

int main() {
    cout << "Fibonacci of 10: " << Fibonacci<10>::value << endl;
    return 0;
}
```

2. **std::variant를 사용하여 여러 데이터 타입을 저장하고 출력하는 프로그램을 작성하세요.**

```cpp
#include <iostream>
#include <variant>
#include <string>
using namespace std;

int main() {
    variant<int, float, string> var;

    var = 42;
    cout << "Integer: " << get<int>(var) << endl;

    var = 3.14f;
    cout << "Float: " << get<float>(var) << endl;

    var = "Hello, World!";
    cout << "String: " << get<string>(var) << endl;

    return 0;
}
```

3. **std::filesystem을 사용하여 디렉토리 내의 파일 목록을 출력하는 프로그램을 작성하세요.**

```cpp
#include <iostream>
#include <filesystem>
using namespace std;
namespace fs = std::filesystem;

int main() {
    string path = ".";
    for (const auto& entry : fs::directory_iterator(path)) {
        cout << entry.path() << endl;
    }
    return 0;
}
```

4. **고급 STL 알고리즘을 사용하여 벡터를 정렬하고 중복된 요소를 제거하는 프로그램을 작성하세요.**

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    vector<int> vec = {4, 2, 5, 2, 3, 1, 4, 5};

    // 정렬
    sort(vec.begin(), vec.end());

    // 중복 제거
    auto last = unique(vec.begin(), vec.end());
    vec.erase(last, vec.end());

    cout << "Sorted and unique elements: ";
    for (int val : vec) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}
```

#### 퀴즈

1. **템플릿 메타프로그래밍에 대한 설명 중 맞는 것은?**
   - A) 템플릿 메타프로그래밍은 컴파일 시간에 실행되는 프로그래밍 기법이다.
   - B) 템플릿 메타프로그래밍은 런타임에만 적용된다.
   - C) 템플릿 메타프로그래밍은 변수의 값을 변경할 수 없다.
   - D) 템플릿 메타프로그래밍은 주로 입출력 작업에 사용된다.

2. **std::variant에 대한 설명 중 맞는 것은?**
   - A) std::variant는 여러 데이터 타입을 저장할 수 있는 컨테이너이다.
   - B) std::variant는 하나의 데이터 타입만 저장할 수 있다.
   - C) std::variant는 항상 크기가 고정되어 있다.
   - D) std::variant는 std::vector와 동일한 방식으로 사용된다.

3. **std::filesystem에 대한 설명 중 맞는 것은?**
   - A) std::filesystem은 파일 시스템 작업을 위한 라이브러리이다.
   - B) std::filesystem은 네트워크 통신을 위한 라이브러리이다.


   - C) std::filesystem은 메모리 관리 작업에 사용된다.
   - D) std::filesystem은 그래픽 렌더링을 위한 라이브러리이다.

4. **STL 알고리즘에 대한 설명 중 맞는 것은?**
   - A) STL 알고리즘은 컨테이너의 요소를 조작하기 위한 함수 집합이다.
   - B) STL 알고리즘은 항상 정렬된 데이터에만 사용된다.
   - C) STL 알고리즘은 STL 컨테이너와 함께 사용할 수 없다.
   - D) STL 알고리즘은 입출력 작업에만 사용된다.

#### 퀴즈 해설

1. **템플릿 메타프로그래밍에 대한 설명 중 맞는 것은?**
   - **정답: A) 템플릿 메타프로그래밍은 컴파일 시간에 실행되는 프로그래밍 기법이다.**
     - 해설: 템플릿 메타프로그래밍은 컴파일 시간에 실행되는 프로그래밍 기법으로, 주로 컴파일 시간 계산 및 코드 최적화에 사용됩니다.

2. **std::variant에 대한 설명 중 맞는 것은?**
   - **정답: A) std::variant는 여러 데이터 타입을 저장할 수 있는 컨테이너이다.**
     - 해설: std::variant는 여러 데이터 타입을 저장할 수 있는 컨테이너로, 런타임에 타입을 안전하게 다룰 수 있게 해줍니다.

3. **std::filesystem에 대한 설명 중 맞는 것은?**
   - **정답: A) std::filesystem은 파일 시스템 작업을 위한 라이브러리이다.**
     - 해설: std::filesystem은 파일 및 디렉토리의 생성, 삭제, 탐색 등 파일 시스템 작업을 수행할 수 있는 라이브러리입니다.

4. **STL 알고리즘에 대한 설명 중 맞는 것은?**
   - **정답: A) STL 알고리즘은 컨테이너의 요소를 조작하기 위한 함수 집합이다.**
     - 해설: STL 알고리즘은 정렬, 탐색, 변환 등의 작업을 수행하는 함수 집합으로, STL 컨테이너와 함께 사용됩니다.

다음 주차 강의 내용을 요청하시면, 22주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.