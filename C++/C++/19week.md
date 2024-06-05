### 19주차: 성능 최적화

#### 강의 목표
- 코드 최적화 기법 이해 및 사용
- 메모리 최적화 이해 및 사용
- 컴파일러 최적화 옵션 이해 및 사용

#### 강의 내용

##### 1. 코드 최적화 기법
- **루프 최적화**

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> vec(1000, 1);

    // 비최적화 루프
    int sum = 0;
    for (int i = 0; i < vec.size(); ++i) {
        sum += vec[i];
    }
    cout << "Sum: " << sum << endl;

    // 최적화된 루프 (루프 언롤링)
    sum = 0;
    for (int i = 0; i < vec.size(); i += 4) {
        sum += vec[i] + vec[i + 1] + vec[i + 2] + vec[i + 3];
    }
    cout << "Optimized Sum: " << sum << endl;

    return 0;
}
```

- **조건문 최적화**

```cpp
#include <iostream>
using namespace std;

int main() {
    int x = 5;

    // 비최적화 조건문
    if (x == 1) {
        cout << "x is 1" << endl;
    } else if (x == 2) {
        cout << "x is 2" << endl;
    } else if (x == 3) {
        cout << "x is 3" << endl;
    } else if (x == 4) {
        cout << "x is 4" << endl;
    } else {
        cout << "x is 5" << endl;
    }

    // 최적화된 조건문 (switch 문 사용)
    switch (x) {
        case 1: cout << "x is 1" << endl; break;
        case 2: cout << "x is 2" << endl; break;
        case 3: cout << "x is 3" << endl; break;
        case 4: cout << "x is 4" << endl; break;
        default: cout << "x is 5" << endl; break;
    }

    return 0;
}
```

##### 2. 메모리 최적화
- **메모리 풀 사용**

```cpp
#include <iostream>
#include <vector>
using namespace std;

class MemoryPool {
private:
    vector<void*> pool;
    size_t blockSize;

public:
    MemoryPool(size_t size, size_t count) : blockSize(size) {
        pool.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            pool.push_back(::operator new(blockSize));
        }
    }

    ~MemoryPool() {
        for (void* ptr : pool) {
            ::operator delete(ptr);
        }
    }

    void* allocate() {
        if (pool.empty()) {
            return ::operator new(blockSize);
        }
        void* ptr = pool.back();
        pool.pop_back();
        return ptr;
    }

    void deallocate(void* ptr) {
        pool.push_back(ptr);
    }
};

int main() {
    const size_t blockSize = 32;
    const size_t poolSize = 10;
    MemoryPool pool(blockSize, poolSize);

    void* ptr1 = pool.allocate();
    void* ptr2 = pool.allocate();

    pool.deallocate(ptr1);
    pool.deallocate(ptr2);

    return 0;
}
```

- **캐시 친화적 데이터 구조**

```cpp
#include <iostream>
#include <vector>
using namespace std;

struct Point {
    float x, y, z;
};

int main() {
    const int size = 1000000;
    vector<Point> points(size);

    // 비최적화 접근
    for (int i = 0; i < size; ++i) {
        points[i].x += 1.0f;
    }
    for (int i = 0; i < size; ++i) {
        points[i].y += 1.0f;
    }
    for (int i = 0; i < size; ++i) {
        points[i].z += 1.0f;
    }

    // 최적화된 접근 (캐시 친화적)
    for (int i = 0; i < size; ++i) {
        points[i].x += 1.0f;
        points[i].y += 1.0f;
        points[i].z += 1.0f;
    }

    return 0;
}
```

##### 3. 컴파일러 최적화 옵션
- **컴파일러 최적화 옵션 사용**

```bash
# 최적화 레벨 1 (O1)
g++ -O1 -o optimized_program program.cpp

# 최적화 레벨 2 (O2)
g++ -O2 -o optimized_program program.cpp

# 최적화 레벨 3 (O3)
g++ -O3 -o optimized_program program.cpp

# 최적화 레벨 s (Os, 크기 최적화)
g++ -Os -o optimized_program program.cpp
```

#### 과제

1. **루프 최적화**
   - 1부터 100까지의 합을 계산하는 프로그램을 작성하고, 루프 언롤링을 사용하여 최적화하세요.

```cpp
#include <iostream>
using namespace std;

int main() {
    // 비최적화 루프
    int sum = 0;
    for (int i = 1; i <= 100; ++i) {
        sum += i;
    }
    cout << "Sum: " << sum << endl;

    // 최적화된 루프 (루프 언롤링)
    sum = 0;
    for (int i = 1; i <= 100; i += 4) {
        sum += i + (i + 1) + (i + 2) + (i + 3);
    }
    cout << "Optimized Sum: " << sum << endl;

    return 0;
}
```

2. **조건문 최적화**
   - 사용자로부터 입력받은 숫자가 1에서 5 사이의 값인 경우 해당 숫자를 출력하는 프로그램을 작성하고, if-else 문을 switch 문으로 최적화하세요.

```cpp
#include <iostream>
using namespace std;

int main() {
    int x;
    cout << "Enter a number (1-5): ";
    cin >> x;

    // 비최적화 조건문
    if (x == 1) {
        cout << "x is 1" << endl;
    } else if (x == 2) {
        cout << "x is 2" << endl;
    } else if (x == 3) {
        cout << "x is 3" << endl;
    } else if (x == 4) {
        cout << "x is 4" << endl;
    } else if (x == 5) {
        cout << "x is 5" << endl;
    } else {
        cout << "Invalid number" << endl;
    }

    // 최적화된 조건문 (switch 문 사용)
    switch (x) {
        case 1: cout << "x is 1" << endl; break;
        case 2: cout << "x is 2" << endl; break;
        case 3: cout << "x is 3" << endl; break;
        case 4: cout << "x is 4" << endl; break;
        case 5: cout << "x is 5" << endl; break;
        default: cout << "Invalid number" << endl; break;
    }

    return 0;
}
```

3. **메모리 최적화**
   - 메모리 풀을 사용하여 동적 메모리를 관리하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <vector>
using namespace std;

class MemoryPool {
private:
    vector<void*> pool;
    size_t blockSize;

public:
    MemoryPool(size_t size, size_t count) : blockSize(size) {
        pool.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            pool.push_back(::operator new(blockSize));
        }
    }

    ~MemoryPool() {
        for (void* ptr : pool) {
            ::operator delete(ptr);
        }
    }

    void* allocate() {
        if (pool.empty()) {
            return ::operator new(blockSize);
        }
        void* ptr = pool.back();
        pool.pop_back();
        return ptr;
    }

    void deallocate(void* ptr) {
        pool.push_back(ptr);
    }
};

int main() {
    const size_t blockSize = 32;
    const size_t poolSize = 10;
    MemoryPool pool(blockSize, poolSize);

    void* ptr1 = pool.allocate();
    void* ptr2 = pool.allocate();

    pool.deallocate(ptr1);
    pool.deallocate(ptr2);

    return 0;
}
```

4. **컴파일러 최적화 옵션 사용**
   - 프로그램을 작성하고, 다양한 컴파일러 최적화 옵션을 사용하여 컴파일한 후 성능 차이를 비교하세요.

```cpp
#include <iostream>
#include <chrono>
using namespace std;
using namespace

 chrono;

void compute() {
    int sum = 0;
    for (int i = 0; i < 1000000; ++i) {
        sum += i;
    }
    cout << "Sum: " << sum << endl;
}

int main() {
    auto start = high_resolution_clock::now();
    compute();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout << "Execution time: " << duration.count() << " ms" << endl;
    return 0;
}
```

컴파일 명령:
```bash
# 기본 컴파일
g++ -o program program.cpp

# 최적화 레벨 1 (O1)
g++ -O1 -o program_O1 program.cpp

# 최적화 레벨 2 (O2)
g++ -O2 -o program_O2 program.cpp

# 최적화 레벨 3 (O3)
g++ -O3 -o program_O3 program.cpp

# 최적화 레벨 s (Os, 크기 최적화)
g++ -Os -o program_Os program.cpp
```

#### 퀴즈

1. **루프 최적화에 대한 설명 중 맞는 것은?**
   - A) 루프 최적화는 항상 코드의 크기를 증가시킨다.
   - B) 루프 최적화는 루프의 실행 시간을 줄이기 위해 사용된다.
   - C) 루프 최적화는 루프의 논리를 변경한다.
   - D) 루프 최적화는 컴파일러에 의해 자동으로 수행된다.

2. **조건문 최적화에 대한 설명 중 맞는 것은?**
   - A) 조건문 최적화는 코드의 가독성을 높인다.
   - B) 조건문 최적화는 항상 if-else 문을 switch 문으로 변경한다.
   - C) 조건문 최적화는 조건문의 실행 시간을 줄이기 위해 사용된다.
   - D) 조건문 최적화는 코드의 크기를 줄인다.

3. **메모리 풀에 대한 설명 중 맞는 것은?**
   - A) 메모리 풀은 메모리 할당과 해제를 빠르게 하기 위해 사용된다.
   - B) 메모리 풀은 항상 메모리 누수를 발생시킨다.
   - C) 메모리 풀은 대규모 프로그램에서만 사용된다.
   - D) 메모리 풀은 메모리의 효율적인 관리를 방해한다.

4. **컴파일러 최적화 옵션에 대한 설명 중 맞는 것은?**
   - A) 컴파일러 최적화 옵션은 항상 프로그램의 크기를 줄인다.
   - B) 컴파일러 최적화 옵션은 프로그램의 실행 속도를 향상시킬 수 있다.
   - C) 컴파일러 최적화 옵션은 코드의 가독성을 높인다.
   - D) 컴파일러 최적화 옵션은 디버깅을 쉽게 한다.

#### 퀴즈 해설

1. **루프 최적화에 대한 설명 중 맞는 것은?**
   - **정답: B) 루프 최적화는 루프의 실행 시간을 줄이기 위해 사용된다.**
     - 해설: 루프 최적화는 반복문의 실행 시간을 줄이기 위해 사용되며, 코드의 성능을 향상시킵니다.

2. **조건문 최적화에 대한 설명 중 맞는 것은?**
   - **정답: C) 조건문 최적화는 조건문의 실행 시간을 줄이기 위해 사용된다.**
     - 해설: 조건문 최적화는 조건문의 실행 시간을 줄여 코드의 성능을 향상시킵니다.

3. **메모리 풀에 대한 설명 중 맞는 것은?**
   - **정답: A) 메모리 풀은 메모리 할당과 해제를 빠르게 하기 위해 사용된다.**
     - 해설: 메모리 풀은 메모리 할당과 해제 속도를 향상시켜 프로그램의 성능을 최적화합니다.

4. **컴파일러 최적화 옵션에 대한 설명 중 맞는 것은?**
   - **정답: B) 컴파일러 최적화 옵션은 프로그램의 실행 속도를 향상시킬 수 있다.**
     - 해설: 컴파일러 최적화 옵션은 프로그램의 실행 속도를 향상시켜 성능을 최적화할 수 있습니다.

다음 주차 강의 내용을 요청하시면, 20주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.