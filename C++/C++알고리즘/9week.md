### 9주차 강의 계획안

#### 강의 주제: 동적 계획법 심화
- 고급 문제 (최장 공통 부분 수열, 행렬 체인 곱셈 등)
- 메모이제이션 기법

---

### 강의 내용

#### 1. 최장 공통 부분 수열 (LCS)
- **개념**: 두 문자열에서 공통으로 나타나는 가장 긴 부분 수열을 찾는 문제
- **시간 복잡도**: O(mn) (m, n은 두 문자열의 길이)

**예제**: 최장 공통 부분 수열
```cpp
#include <iostream>
#include <vector>
#include <string>
using namespace std;

int lcs(string X, string Y) {
    int m = X.length();
    int n = Y.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));

    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            if (i == 0 || j == 0)
                dp[i][j] = 0;
            else if (X[i - 1] == Y[j - 1])
                dp[i][j] = dp[i - 1][j - 1] + 1;
            else
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
        }
    }
    return dp[m][n];
}

int main() {
    string X = "AGGTAB";
    string Y = "GXTXAYB";
    cout << "Length of LCS is " << lcs(X, Y) << endl;
    return 0;
}
```

#### 2. 행렬 체인 곱셈
- **개념**: 주어진 행렬들을 곱하는 순서를 최적화하여 곱셈 연산의 수를 최소화하는 문제
- **시간 복잡도**: O(n^3)

**예제**: 행렬 체인 곱셈
```cpp
#include <iostream>
#include <vector>
using namespace std;

int matrixChainOrder(vector<int>& p, int n) {
    vector<vector<int>> dp(n, vector<int>(n, 0));

    for (int l = 2; l < n; l++) {
        for (int i = 1; i < n - l + 1; i++) {
            int j = i + l - 1;
            dp[i][j] = INT_MAX;
            for (int k = i; k <= j - 1; k++) {
                int q = dp[i][k] + dp[k + 1][j] + p[i - 1] * p[k] * p[j];
                if (q < dp[i][j])
                    dp[i][j] = q;
            }
        }
    }
    return dp[1][n - 1];
}

int main() {
    vector<int> arr = {1, 2, 3, 4};
    int n = arr.size();
    cout << "Minimum number of multiplications is " << matrixChainOrder(arr, n) << endl;
    return 0;
}
```

#### 3. 메모이제이션 기법
- **개념**: 이전에 계산된 값을 저장하여 동일한 계산을 반복하지 않도록 하는 최적화 기법
- **특징**: 재귀적 접근 방식에서 많이 사용됨

**예제**: 메모이제이션을 이용한 피보나치 수열
```cpp
#include <iostream>
#include <vector>
using namespace std;

vector<int> memo;

int fib(int n) {
    if (n <= 1)
        return n;
    if (memo[n] != -1)
        return memo[n];
    memo[n] = fib(n - 1) + fib(n - 2);
    return memo[n];
}

int main() {
    int n;
    cout << "Enter the value of n: ";
    cin >> n;
    memo.resize(n + 1, -1);
    cout << "Fibonacci number " << n << " is " << fib(n) << endl;
    return 0;
}
```

---

### 과제

#### 과제 1: 최장 증가 부분 수열 (LIS)
동적 계획법을 사용하여 주어진 배열의 최장 증가 부분 수열(LIS)의 길이를 계산하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
using namespace std;

int lis(vector<int>& arr) {
    int n = arr.size();
    vector<int> lis(n, 1);

    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[i] > arr[j] && lis[i] < lis[j] + 1)
                lis[i] = lis[j] + 1;
        }
    }

    int max_lis = 0;
    for (int i = 0; i < n; i++) {
        if (max_lis < lis[i])
            max_lis = lis[i];
    }

    return max_lis;
}

int main() {
    vector<int> arr = {10, 22, 9, 33, 21, 50, 41, 60, 80};
    cout << "Length of LIS is " << lis(arr) << endl;
    return 0;
}
```

**해설**:
1. 동적 계획법을 사용하여 배열의 최장 증가 부분 수열(LIS)의 길이를 계산합니다.
2. `lis` 배열을 사용하여 각 원소를 포함한 LIS의 길이를 저장합니다.

#### 과제 2: 최소 편집 거리 (Edit Distance)
두 문자열 간의 최소 편집 거리를 계산하는 프로그램을 동적 계획법을 사용하여 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
#include <string>
using namespace std;

int minEditDistance(string str1, string str2) {
    int m = str1.length();
    int n = str2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));

    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            if (i == 0)
                dp[i][j] = j;
            else if (j == 0)
                dp[i][j] = i;
            else if (str1[i - 1] == str2[j - 1])
                dp[i][j] = dp[i - 1][j - 1];
            else
                dp[i][j] = 1 + min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});
        }
    }

    return dp[m][n];
}

int main() {
    string str1 = "sunday";
    string str2 = "saturday";
    cout << "Minimum edit distance is " << minEditDistance(str1, str2) << endl;
    return 0;
}
```

**해설**:
1. 동적 계획법을 사용하여 두 문자열 간의 최소 편집 거리를 계산합니다.
2. `dp` 배열을 사용하여 각 부분 문제의 해를 저장합니다.

---

### 퀴즈

#### 퀴즈 1: 최장 공통 부분 수열(LCS) 문제의 시간 복잡도는 무엇인가요?
1. O(n)
2. O(m + n)
3. O(mn)
4. O(m^2 + n^2)

**정답**: 3. O(mn)

#### 퀴즈 2: 행렬 체인 곱셈 문제의 시간 복잡도는 무엇인가요?
1. O(n)
2. O(n^2)
3. O(n^2 log n)
4. O(n^3)

**정답**: 4. O(n^3)

#### 퀴즈 3: 메모이제이션은 어떤 유형의 문제에 가장 적합한가요?
1. 문제가 독립적인 경우
2. 하위 문제가 중복되는 경우
3. 하위 문제가 독립적인 경우
4. 문제가 단순한 경우

**정답**: 2. 하위 문제가 중복되는 경우

이 계획안은 9주차에 필요한 동적 계획법 심화의 개념과 실습을 통해 학습할 수 있도록 구성되었습니다. 과제와 퀴즈를 통해 학습한 내용을 복습하고, 이해도를 높일 수 있습니다.