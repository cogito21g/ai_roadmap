### 13주차: 고급 포인터 개념

#### 강의 목표
- 스마트 포인터의 개념 이해 및 사용 (unique_ptr, shared_ptr, weak_ptr)
- RAII (Resource Acquisition Is Initialization)의 이해 및 사용
- 메모리 누수와 관리 이해 및 예방

#### 강의 내용

##### 1. 스마트 포인터 (unique_ptr)
- **unique_ptr 선언 및 사용**

```cpp
#include <iostream>
#include <memory>  // 스마트 포인터를 사용하기 위한 헤더 파일
using namespace std;

int main() {
    unique_ptr<int> ptr1(new int(5));
    cout << "Value: " << *ptr1 << endl;

    unique_ptr<int> ptr2 = move(ptr1);  // ptr1의 소유권을 ptr2로 이동
    if (ptr1 == nullptr) {
        cout << "ptr1 is now nullptr" << endl;
    }
    cout << "Value: " << *ptr2 << endl;

    return 0;
}
```

##### 2. 스마트 포인터 (shared_ptr)
- **shared_ptr 선언 및 사용**

```cpp
#include <iostream>
#include <memory>
using namespace std;

int main() {
    shared_ptr<int> ptr1(new int(10));
    {
        shared_ptr<int> ptr2 = ptr1;
        cout << "Value: " << *ptr2 << endl;
        cout << "Use count: " << ptr1.use_count() << endl;
    }  // ptr2가 범위를 벗어나면 참조 카운트가 감소

    cout << "Use count after ptr2 is out of scope: " << ptr1.use_count() << endl;
    return 0;
}
```

##### 3. 스마트 포인터 (weak_ptr)
- **weak_ptr 선언 및 사용**

```cpp
#include <iostream>
#include <memory>
using namespace std;

int main() {
    shared_ptr<int> sptr = make_shared<int>(20);
    weak_ptr<int> wptr = sptr;

    cout << "Shared pointer value: " << *sptr << endl;
    cout << "Use count: " << sptr.use_count() << endl;

    if (auto temp = wptr.lock()) {  // weak_ptr을 shared_ptr로 잠금
        cout << "Weak pointer value: " << *temp << endl;
        cout << "Use count after locking: " << sptr.use_count() << endl;
    } else {
        cout << "Weak pointer is expired" << endl;
    }

    return 0;
}
```

##### 4. RAII (Resource Acquisition Is Initialization)
- **RAII 개념 및 사용**

```cpp
#include <iostream>
#include <fstream>
using namespace std;

class FileHandler {
public:
    FileHandler(const string& filename) : file(filename) {
        if (!file.is_open()) {
            throw runtime_error("Unable to open file");
        }
    }

    ~FileHandler() {
        file.close();
    }

    void write(const string& text) {
        file << text;
    }

private:
    ofstream file;
};

int main() {
    try {
        FileHandler fileHandler("example.txt");
        fileHandler.write("Hello, RAII!");
    } catch (const runtime_error& e) {
        cout << "Error: " << e.what() << endl;
    }

    return 0;
}
```

#### 과제

1. **unique_ptr 사용**
   - unique_ptr을 사용하여 동적 메모리를 관리하고, 소유권을 다른 unique_ptr로 이동하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <memory>
using namespace std;

int main() {
    unique_ptr<int> ptr1(new int(100));
    cout << "Value: " << *ptr1 << endl;

    unique_ptr<int> ptr2 = move(ptr1);
    if (ptr1 == nullptr) {
        cout << "ptr1 is now nullptr" << endl;
    }
    cout << "Value: " << *ptr2 << endl;

    return 0;
}
```

2. **shared_ptr 사용**
   - shared_ptr을 사용하여 참조 카운트가 어떻게 변하는지 확인하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <memory>
using namespace std;

int main() {
    shared_ptr<int> ptr1(new int(200));
    {
        shared_ptr<int> ptr2 = ptr1;
        cout << "Value: " << *ptr2 << endl;
        cout << "Use count: " << ptr1.use_count() << endl;
    }

    cout << "Use count after ptr2 is out of scope: " << ptr1.use_count() << endl;
    return 0;
}
```

3. **weak_ptr 사용**
   - weak_ptr을 사용하여 shared_ptr의 참조를 유지하고, lock을 사용하여 값을 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <memory>
using namespace std;

int main() {
    shared_ptr<int> sptr = make_shared<int>(300);
    weak_ptr<int> wptr = sptr;

    cout << "Shared pointer value: " << *sptr << endl;
    cout << "Use count: " << sptr.use_count() << endl;

    if (auto temp = wptr.lock()) {
        cout << "Weak pointer value: " << *temp << endl;
        cout << "Use count after locking: " << sptr.use_count() << endl;
    } else {
        cout << "Weak pointer is expired" << endl;
    }

    return 0;
}
```

4. **RAII 사용**
   - RAII를 사용하여 파일을 자동으로 열고 닫는 클래스를 작성하고, 파일에 데이터를 쓰는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <fstream>
using namespace std;

class FileHandler {
public:
    FileHandler(const string& filename) : file(filename) {
        if (!file.is_open()) {
            throw runtime_error("Unable to open file");
        }
    }

    ~FileHandler() {
        file.close();
    }

    void write(const string& text) {
        file << text;
    }

private:
    ofstream file;
};

int main() {
    try {
        FileHandler fileHandler("example.txt");
        fileHandler.write("Hello, RAII!");
    } catch (const runtime_error& e) {
        cout << "Error: " << e.what() << endl;
    }

    return 0;
}
```

#### 퀴즈

1. **unique_ptr에 대한 설명 중 맞는 것은?**
   - A) unique_ptr은 여러 개의 소유자가 있을 수 있다.
   - B) unique_ptr은 복사할 수 있다.
   - C) unique_ptr은 소유권을 다른 unique_ptr로 이동할 수 있다.
   - D) unique_ptr은 메모리 누수를 방지할 수 없다.

2. **shared_ptr에 대한 설명 중 맞는 것은?**
   - A) shared_ptr은 참조 카운트를 사용하지 않는다.
   - B) shared_ptr은 여러 개의 소유자를 가질 수 없다.
   - C) shared_ptr은 자동으로 메모리를 해제하지 않는다.
   - D) shared_ptr은 참조 카운트를 사용하여 메모리를 관리한다.

3. **weak_ptr에 대한 설명 중 맞는 것은?**
   - A) weak_ptr은 참조 카운트를 증가시킨다.
   - B) weak_ptr은 항상 유효한 포인터를 가리킨다.
   - C) weak_ptr은 shared_ptr의 소유권을 갖지 않는다.
   - D) weak_ptr은 직접 메모리를 해제할 수 있다.

4. **RAII에 대한 설명 중 맞는 것은?**
   - A) RAII는 리소스를 명시적으로 해제해야 한다.
   - B) RAII는 리소스를 자동으로 관리하는 기법이다.
   - C) RAII는 예외 안전성을 제공하지 않는다.
   - D) RAII는 스마트 포인터와 사용할 수 없다.

#### 퀴즈 해설

1. **unique_ptr에 대한 설명 중 맞는 것은?**
   - **정답: C) unique_ptr은 소유권을 다른 unique_ptr로 이동할 수 있다.**
     - 해설: unique_ptr은 유일한 소유권을 가지며, 소유권을 다른 unique_ptr로 이동할 수 있습니다.

2. **shared_ptr에 대한 설명 중 맞는 것은?**
   - **정답: D) shared_ptr은 참조 카운트를 사용하여 메모리를 관리한다.**
     - 해설: shared_ptr은 참조 카운트를 사용하여 여러 소유자가 동일한 객체를 공유할 수 있게 하고, 참조 카운트가 0이 되면 메모리를 자동으로 해제합니다.

3. **weak_ptr에 대한 설명 중 맞는 것은?**
   - **정답: C) weak_ptr은 shared_ptr의 소유권을 갖지 않는다.**
     - 해설: weak_ptr은 shared_ptr의 소유권을 갖지 않으며, 참조 카운트를 증가시키지 않고 객체가 존재하는지 확인하는 데 사용됩니다.

4. **RAII에 대한 설명 중 맞는 것은?**
   - **정답: B) RAII는 리소스를 자동으로 관리하는 기법이다.**
     - 해설: RAII는 리소스를 객체의 생명 주기와 함께 자동으로 관리하여 예외 안전성을 제공합니다.

다음 주차 강의 내용을 요청하시면, 14주차 강의 내용, 과제 및 퀴즈