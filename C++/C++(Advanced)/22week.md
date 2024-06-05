### 22주차: 성능 최적화 심화

#### 강의 목표
- 성능 최적화를 위한 프로파일링 기법 이해 및 사용
- 고급 메모리 관리 기법 이해 및 사용
- 병렬 프로그래밍 최적화 기법 이해 및 사용

#### 강의 내용

##### 1. 프로파일링 기법
- **프로파일링 도구 사용**
  - **Valgrind**: 메모리 사용 분석 및 누수 탐지
  - **gprof**: CPU 사용 시간 분석

```bash
# Valgrind 사용 예시
valgrind --leak-check=full ./program

# gprof 사용 예시
g++ -pg -o program program.cpp
./program
gprof program gmon.out > analysis.txt
```

- **Code Example for Profiling**

```cpp
#include <iostream>
using namespace std;

void compute() {
    int sum = 0;
    for (int i = 0; i < 1000000; ++i) {
        sum += i;
    }
    cout << "Sum: " << sum << endl;
}

int main() {
    compute();
    return 0;
}
```

##### 2. 고급 메모리 관리
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

##### 3. 병렬 프로그래밍 최적화
- **병렬 프로그래밍 기초 (OpenMP 사용)**

```cpp
#include <iostream>
#include <omp.h>
using namespace std;

int main() {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < 1000000; ++i) {
        sum += i;
    }
    cout << "Sum: " << sum << endl;
    return 0;
}
```

- **병렬 퀵 정렬 구현**

```cpp
#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; ++j) {
        if (arr[j] < pivot) {
            ++i;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                quickSort(arr, low, pi - 1);
            }
            #pragma omp section
            {
                quickSort(arr, pi + 1, high);
            }
        }
    }
}

int main() {
    vector<int> arr = {10, 7, 8, 9, 1, 5};
    int n = arr.size();

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            quickSort(arr, 0, n - 1);
        }
    }

    cout << "Sorted array: ";
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;

    return 0;
}
```

#### 과제

1. **프로파일링 기법 사용**
   - 제공된 코드를 작성하고 Valgrind와 gprof를 사용하여 메모리 사용 및 CPU 사용 시간을 분석하세요.

```cpp
#include <iostream>
#include <vector>
using namespace std;

void compute() {
    vector<int> vec(1000000, 1);
    int sum = 0;
    for (int i : vec) {
        sum += i;
    }
    cout << "Sum: " << sum << endl;
}

int main() {
    compute();
    return 0;
}
```

2. **고급 메모리 관리 기법**
   - 메모리 풀을 사용하여 동적 메모리를 관리하고, 캐시 친화적인 데이터 구조를 사용하는 프로그램을 작성하세요.

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

struct Point {
    float x, y, z;
};

int main() {
    const size_t blockSize = sizeof(Point);
    const size_t poolSize = 1000;
    MemoryPool pool(blockSize, poolSize);

    Point* points = static_cast<Point*>(pool.allocate());
    points->x = 1.0f;
    points->y = 2.0f;
    points->z = 3.0f;

    cout << "Point: (" << points->x << ", " << points->y << ", " << points->z << ")" << endl;

    pool.deallocate(points);

    return 0;
}
```

3. **병렬 프로그래밍 최적화**
   - OpenMP를 사용하여 병렬 퀵 정렬을 구현하고, 성능을 최적화하세요.

```cpp
#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; ++j) {
        if (arr[j] < pivot) {
            ++i;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                quickSort(arr, low, pi - 1);
            }
            #pragma omp section
            {
                quickSort(arr, pi + 1, high);
            }
        }
    }
}

int main() {
    vector<int> arr = {10, 7, 8, 9, 1, 5};
    int n = arr.size();

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            quickSort(arr, 0, n - 1);
        }
    }

    cout << "Sorted array: ";
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;

    return 0;
}
```

#### 퀴즈

1. **프로파일링 도구의 사용

에 대한 설명 중 맞는 것은?**
   - A) Valgrind는 CPU 사용 시간을 분석하는 도구이다.
   - B) gprof는 메모리 사용을 분석하는 도구이다.
   - C) Valgrind는 메모리 사용을 분석하고 누수를 탐지하는 도구이다.
   - D) gprof는 네트워크 트래픽을 분석하는 도구이다.

2. **메모리 풀의 장점 중 맞는 것은?**
   - A) 메모리 풀은 항상 메모리 사용량을 줄인다.
   - B) 메모리 풀은 동적 메모리 할당 속도를 향상시킨다.
   - C) 메모리 풀은 코드의 가독성을 높인다.
   - D) 메모리 풀은 모든 프로그램에서 사용된다.

3. **OpenMP를 사용한 병렬 프로그래밍의 장점 중 맞는 것은?**
   - A) OpenMP는 단일 스레드 프로그램에서만 사용할 수 있다.
   - B) OpenMP는 병렬 프로그래밍을 쉽게 구현할 수 있게 한다.
   - C) OpenMP는 모든 병렬 프로그래밍 문제를 자동으로 해결한다.
   - D) OpenMP는 프로그램의 실행 속도를 항상 줄인다.

4. **캐시 친화적 데이터 구조에 대한 설명 중 맞는 것은?**
   - A) 캐시 친화적 데이터 구조는 메모리 사용을 늘린다.
   - B) 캐시 친화적 데이터 구조는 CPU 캐시를 효율적으로 사용한다.
   - C) 캐시 친화적 데이터 구조는 디스크 I/O를 최적화한다.
   - D) 캐시 친화적 데이터 구조는 네트워크 성능을 향상시킨다.

#### 퀴즈 해설

1. **프로파일링 도구의 사용에 대한 설명 중 맞는 것은?**
   - **정답: C) Valgrind는 메모리 사용을 분석하고 누수를 탐지하는 도구이다.**
     - 해설: Valgrind는 메모리 사용을 분석하고 메모리 누수를 탐지하는 데 사용되는 도구입니다.

2. **메모리 풀의 장점 중 맞는 것은?**
   - **정답: B) 메모리 풀은 동적 메모리 할당 속도를 향상시킨다.**
     - 해설: 메모리 풀은 동적 메모리 할당과 해제의 속도를 향상시키는 데 사용됩니다.

3. **OpenMP를 사용한 병렬 프로그래밍의 장점 중 맞는 것은?**
   - **정답: B) OpenMP는 병렬 프로그래밍을 쉽게 구현할 수 있게 한다.**
     - 해설: OpenMP는 병렬 프로그래밍을 쉽게 구현할 수 있도록 도와주는 도구입니다.

4. **캐시 친화적 데이터 구조에 대한 설명 중 맞는 것은?**
   - **정답: B) 캐시 친화적 데이터 구조는 CPU 캐시를 효율적으로 사용한다.**
     - 해설: 캐시 친화적 데이터 구조는 CPU 캐시를 효율적으로 사용하여 메모리 접근 속도를 향상시킵니다.

다음 주차 강의 내용을 요청하시면, 23주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.