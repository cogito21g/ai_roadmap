### 8주차 강의 계획안

#### 강의 주제: 동적 계획법 기초
- 동적 계획법의 개념
- 기본 문제 (최대 부분 배열 합, 배낭 문제 등)

---

### 강의 내용

#### 1. 동적 계획법의 개념
- **개념**: 문제를 작은 하위 문제로 나누어 해결한 후, 그 해를 저장하여 동일한 하위 문제가 반복해서 계산되지 않도록 하는 방법
- **특징**: 최적 부분 구조, 중복된 하위 문제
- **방법**: 메모이제이션(Memoization)과 탑다운 방식, 테이블화(Tabulation)와 바텀업 방식

#### 2. 최대 부분 배열 합 (Kadane's Algorithm)
- **개념**: 연속된 부분 배열의 합이 최대가 되는 값을 찾는 문제
- **시간 복잡도**: O(n)

**예제**: 최대 부분 배열 합
```cpp
#include <iostream>
#include <vector>
using namespace std;

int maxSubArraySum(vector<int>& arr) {
    int max_so_far = arr[0];
    int current_max = arr[0];

    for (int i = 1; i < arr.size(); i++) {
        current_max = max(arr[i], current_max + arr[i]);
        max_so_far = max(max_so_far, current_max);
    }

    return max_so_far;
}

int main() {
    vector<int> arr = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    cout << "Maximum contiguous sum is " << maxSubArraySum(arr) << endl;
    return 0;
}
```

#### 3. 배낭 문제 (0/1 Knapsack)
- **개념**: 무게와 가치가 있는 아이템들이 주어졌을 때, 배낭의 최대 무게를 넘지 않으면서 가치를 최대화하는 문제
- **시간 복잡도**: O(nW) (n은 아이템의 수, W는 배낭의 용량)

**예제**: 배낭 문제
```cpp
#include <iostream>
#include <vector>
using namespace std;

int knapSack(int W, vector<int>& wt, vector<int>& val, int n) {
    vector<vector<int>> dp(n + 1, vector<int>(W + 1));

    for (int i = 0; i <= n; i++) {
        for (int w = 0; w <= W; w++) {
            if (i == 0 || w == 0)
                dp[i][w] = 0;
            else if (wt[i - 1] <= w)
                dp[i][w] = max(val[i - 1] + dp[i - 1][w - wt[i - 1]], dp[i - 1][w]);
            else
                dp[i][w] = dp[i - 1][w];
        }
    }
    return dp[n][W];
}

int main() {
    int W = 50;
    vector<int> wt = {10, 20, 30};
    vector<int> val = {60, 100, 120};
    int n = wt.size();
    cout << "Maximum value in Knapsack = " << knapSack(W, wt, val, n) << endl;
    return 0;
}
```

---

### 과제

#### 과제 1: 동적 계획법을 이용한 피보나치 수열
동적 계획법을 사용하여 피보나치 수열의 n번째 값을 계산하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
using namespace std;

int fib(int n) {
    vector<int> dp(n + 1, 0);
    dp[1] = 1;

    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }

    return dp[n];
}

int main() {
    int n;
    cout << "Enter the value of n: ";
    cin >> n;
    cout << "Fibonacci number " << n << " is " << fib(n) << endl;
    return 0;
}
```

**해설**:
1. 동적 계획법을 사용하여 피보나치 수열을 계산합니다.
2. `dp` 배열을 사용하여 이전 계산 값을 저장합니다.

#### 과제 2: 동적 계획법을 이용한 최소 동전 교환 문제
주어진 금액을 최소한의 동전으로 교환하는 문제를 동적 계획법을 이용하여 해결하는 프로그램을 작성하세요. 사용 가능한 동전의 종류는 {1, 3, 4}입니다.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
#include <climits>
using namespace std;

int minCoins(int coins[], int m, int V) {
    vector<int> dp(V + 1, INT_MAX);
    dp[0] = 0;

    for (int i = 1; i <= V; i++) {
        for (int j = 0; j < m; j++) {
            if (coins[j] <= i) {
                int sub_res = dp[i - coins[j]];
                if (sub_res != INT_MAX && sub_res + 1 < dp[i])
                    dp[i] = sub_res + 1;
            }
        }
    }

    return dp[V];
}

int main() {
    int coins[] = {1, 3, 4};
    int m = sizeof(coins) / sizeof(coins[0]);
    int V;
    cout << "Enter the amount: ";
    cin >> V;
    cout << "Minimum coins required is " << minCoins(coins, m, V) << endl;
    return 0;
}
```

**해설**:
1. 동적 계획법을 사용하여 최소 동전 교환 문제를 해결합니다.
2. `dp` 배열을 사용하여 주어진 금액을 만들기 위한 최소 동전 수를 저장합니다.

---

### 퀴즈

#### 퀴즈 1: 다음 중 동적 계획법의 두 가지 접근 방식은 무엇인가요?
1. 그리디 알고리즘, 분할 정복
2. 메모이제이션, 테이블화
3. 깊이 우선 탐색, 너비 우선 탐색
4. 삽입 정렬, 선택 정렬

**정답**: 2. 메모이제이션, 테이블화

#### 퀴즈 2: 최대 부분 배열 합 문제의 시간 복잡도는 무엇인가요?
1. O(n)
2. O(n^2)
3. O(n log n)
4. O(log n)

**정답**: 1. O(n)

#### 퀴즈 3: 배낭 문제의 시간 복잡도는 무엇인가요?
1. O(n)
2. O(n^2)
3. O(nW)
4. O(log n)

**정답**: 3. O(nW)

이 계획안은 8주차에 필요한 동적 계획법의 기초 개념과 실습을 통해 학습할 수 있도록 구성되었습니다. 과제와 퀴즈를 통해 학습한 내용을 복습하고, 이해도를 높일 수 있습니다.