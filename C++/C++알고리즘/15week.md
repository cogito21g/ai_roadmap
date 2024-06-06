### 15주차 강의 계획안

#### 강의 주제: 복잡도 분석 및 최적화
- 시간 복잡도와 공간 복잡도의 이해
- 빅오 표기법
- 알고리즘 최적화 기법

---

### 강의 내용

#### 1. 시간 복잡도와 공간 복잡도의 이해
- **개념**: 알고리즘의 효율성을 평가하는 기준
- **시간 복잡도**: 입력 크기와 실행 시간 간의 관계
- **공간 복잡도**: 입력 크기와 메모리 사용량 간의 관계

**예제**: 시간 복잡도
```cpp
#include <iostream>
using namespace std;

void constantTimeOperation(int n) {
    cout << "Constant time operation" << endl;
}

void linearTimeOperation(int n) {
    for (int i = 0; i < n; i++) {
        cout << "Linear time operation" << endl;
    }
}

void quadraticTimeOperation(int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << "Quadratic time operation" << endl;
        }
    }
}

int main() {
    int n = 5;
    constantTimeOperation(n);
    linearTimeOperation(n);
    quadraticTimeOperation(n);
    return 0;
}
```

#### 2. 빅오 표기법
- **개념**: 알고리즘의 시간 복잡도와 공간 복잡도를 표기하는 방법
- **표기법**: O(1), O(log n), O(n), O(n log n), O(n^2), O(2^n) 등

**예제**: 빅오 표기법 예제
```cpp
#include <iostream>
using namespace std;

void logTimeOperation(int n) {
    for (int i = 1; i < n; i *= 2) {
        cout << "Logarithmic time operation" << endl;
    }
}

void linearLogTimeOperation(int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 1; j < n; j *= 2) {
            cout << "Linearithmic time operation" << endl;
        }
    }
}

int main() {
    int n = 16;
    logTimeOperation(n);
    linearLogTimeOperation(n);
    return 0;
}
```

#### 3. 알고리즘 최적화 기법
- **개념**: 알고리즘의 성능을 향상시키는 방법
- **기법**:
  - 중복 계산 제거
  - 적절한 데이터 구조 사용
  - 알고리즘 개선 (예: 동적 계획법 사용)

**예제**: 중복 계산 제거
```cpp
#include <iostream>
#include <vector>
using namespace std;

int fibonacci(int n, vector<int>& memo) {
    if (n <= 1) return n;
    if (memo[n] != -1) return memo[n];
    memo[n] = fibonacci(n - 1) + fibonacci(n - 2);
    return memo[n];
}

int main() {
    int n = 10;
    vector<int> memo(n + 1, -1);
    cout << "Fibonacci of " << n << " is " << fibonacci(n, memo) << endl;
    return 0;
}
```

---

### 과제

#### 과제 1: 다양한 시간 복잡도의 알고리즘 구현
다양한 시간 복잡도를 가진 알고리즘을 구현하고, 각각의 실행 시간을 측정하여 비교하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

void constantTimeOperation() {
    cout << "Constant time operation" << endl;
}

void linearTimeOperation(int n) {
    for (int i = 0; i < n; i++) {
        cout << "Linear time operation" << endl;
    }
}

void quadraticTimeOperation(int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << "Quadratic time operation" << endl;
        }
    }
}

int main() {
    int n = 100;

    auto start = high_resolution_clock::now();
    constantTimeOperation();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by constant time operation: " << duration.count() << " microseconds" << endl;

    start = high_resolution_clock::now();
    linearTimeOperation(n);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by linear time operation: " << duration.count() << " microseconds" << endl;

    start = high_resolution_clock::now();
    quadraticTimeOperation(n);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by quadratic time operation: " << duration.count() << " microseconds" << endl;

    return 0;
}
```

**해설**:
1. 다양한 시간 복잡도를 가진 알고리즘을 구현합니다.
2. `chrono` 라이브러리를 사용해 각각의 실행 시간을 측정하고 비교합니다.

#### 과제 2: 동적 계획법을 이용한 피보나치 수열 최적화
동적 계획법을 사용해 피보나치 수열을 계산하는 프로그램을 작성하고, 재귀적 접근 방식과의 성능을 비교하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
using namespace std::chrono;

int fibonacciRecursive(int n) {
    if (n <= 1) return n;
    return fibonacciRecursive(n - 1) + fibonacciRecursive(n - 2);
}

int fibonacciDP(int n) {
    vector<int> dp(n + 1, 0);
    dp[1] = 1;
    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}

int main() {
    int n = 30;

    auto start = high_resolution_clock::now();
    cout << "Fibonacci (Recursive) of " << n << " is " << fibonacciRecursive(n) << endl;
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by recursive approach: " << duration.count() << " microseconds" << endl;

    start = high_resolution_clock::now();
    cout << "Fibonacci (DP) of " << n << " is " << fibonacciDP(n) << endl;
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by dynamic programming approach: " << duration.count() << " microseconds" << endl;

    return 0;
}
```

**해설**:
1. 재귀적 접근 방식과 동적 계획법을 사용한 피보나치 수열 계산 프로그램을 구현합니다.
2. 각각의 실행 시간을 측정하고 비교합니다.

---

### 퀴즈

#### 퀴즈 1: 다음 중 시간 복잡도가 가장 낮은 것은 무엇인가요?
1. O(n)
2. O(n log n)
3. O(n^2)
4. O(log n)

**정답**: 4. O(log n)

#### 퀴즈 2: 빅오 표기법에서 시간 복잡도가 O(n^2)인 알고리즘은 무엇을 의미하나요?
1. 입력 크기에 비례하여 실행 시간이 선형적으로 증가한다.
2. 입력 크기에 비례하여 실행 시간이 로그적으로 증가한다.
3. 입력 크기에 비례하여 실행 시간이 제곱으로 증가한다.
4. 입력 크기에 관계없이 실행 시간이 일정하다.

**정답**: 3. 입력 크기에 비례하여 실행 시간이 제곱으로 증가한다.

#### 퀴즈 3: 동적 계획법의 주요 특징은 무엇인가요?
1. 입력 데이터를 분할하여 병렬로 처리한다.
2. 중복 계산을 피하기 위해 이전에 계산한 값을 저장한다.
3. 탐욕적 선택을 통해 최적의 해를 찾는다.
4. 모든 가능한 경우의 수를 체계적으로 탐색한다.

**정답**: 2. 중복 계산을 피하기 위해 이전에 계산한 값을 저장한다.

이 계획안은 15주차에 필요한 복잡도 분석 및 최적화의 개념과 실습을 통해 학습할 수 있도록 구성되었습니다. 과제와 퀴즈를 통해 학습한 내용을 복습하고, 이해도를 높일 수 있습니다.