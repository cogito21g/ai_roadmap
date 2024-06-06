### 20주차 강의 계획안

#### 강의 주제: 최종 복습 및 평가
- 전체 강의 내용 복습
- 최종 평가 및 실전 문제 풀이
- 질의응답 및 강의 마무리

---

### 강의 내용

#### 1. 전체 강의 내용 복습
- **목표**: 전체 강의에서 다룬 주요 개념과 알고리즘을 다시 확인하고 복습
- **방법**:
  1. 주차별 주요 내용 요약
  2. 핵심 개념과 알고리즘 정리
  3. 중요한 코드 예제와 문제 해결 방법 복습

**복습 항목**:
1. 기초 알고리즘 및 자료구조
2. 탐욕 알고리즘과 동적 계획법
3. 분할 정복 및 백트래킹 알고리즘
4. 그래프 알고리즘과 고급 자료구조
5. 문자열 알고리즘 및 최적화 기법
6. 실전 문제 해결 프로젝트

---

#### 2. 최종 평가 및 실전 문제 풀이
- **목표**: 학습한 내용을 바탕으로 실전 문제를 해결하고, 최종 평가를 통해 학습 성과를 측정
- **방법**:
  1. 다양한 난이도의 실전 문제 풀이
  2. 문제 해결 과정에서 배운 기법과 알고리즘 적용
  3. 최종 평가를 통해 이해도 및 문제 해결 능력 측정

**실전 문제 예제**:

**문제 1: 최대 연속 부분 합 (Kadane's Algorithm)**
```cpp
#include <iostream>
#include <vector>
using namespace std;

int maxSubArraySum(vector<int>& arr) {
    int max_so_far = arr[0];
    int curr_max = arr[0];

    for (int i = 1; i < arr.size(); i++) {
        curr_max = max(arr[i], curr_max + arr[i]);
        max_so_far = max(max_so_far, curr_max);
    }

    return max_so_far;
}

int main() {
    vector<int> arr = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    cout << "Maximum contiguous sum is " << maxSubArraySum(arr) << endl;
    return 0;
}
```

**문제 2: 최장 증가 부분 수열 (Longest Increasing Subsequence)**
```cpp
#include <iostream>
#include <vector>
using namespace std;

int lis(vector<int>& arr) {
    int n = arr.size();
    vector<int> lis(n, 1);

    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[i] > arr[j] && lis[i] < lis[j] + 1) {
                lis[i] = lis[j] + 1;
            }
        }
    }

    int max_lis = 0;
    for (int i = 0; i < n; i++) {
        if (max_lis < lis[i]) {
            max_lis = lis[i];
        }
    }

    return max_lis;
}

int main() {
    vector<int> arr = {10, 22, 9, 33, 21, 50, 41, 60, 80};
    cout << "Length of LIS is " << lis(arr) << endl;
    return 0;
}
```

**문제 3: 최소 스패닝 트리 (Prim's Algorithm)**
```cpp
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

void primMST(vector<vector<pair<int, int>>>& graph, int V) {
    vector<int> key(V, INT_MAX);
    vector<int> parent(V, -1);
    vector<bool> inMST(V, false);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

    key[0] = 0;
    pq.push({0, 0});

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        inMST[u] = true;

        for (auto& [v, weight] : graph[u]) {
            if (!inMST[v] && key[v] > weight) {
                key[v] = weight;
                pq.push({key[v], v});
                parent[v] = u;
            }
        }
    }

    cout << "Edge   Weight\n";
    for (int i = 1; i < V; i++)
        cout << parent[i] << " - " << i << "   " << key[i] << " \n";
}

int main() {
    int V = 5;
    vector<vector<pair<int, int>>> graph(V);
    graph[0].push_back({1, 2});
    graph[0].push_back({3, 6});
    graph[1].push_back({0, 2});
    graph[1].push_back({2, 3});
    graph[1].push_back({3, 8});
    graph[1].push_back({4, 5});
    graph[2].push_back({1, 3});
    graph[2].push_back({4, 7});
    graph[3].push_back({0, 6});
    graph[3].push_back({1, 8});
    graph[4].push_back({1, 5});
    graph[4].push_back({2, 7});

    primMST(graph, V);
    return 0;
}
```

---

#### 3. 질의응답 및 강의 마무리
- **목표**: 학습 과정에서 생긴 궁금증을 해결하고, 강의를 마무리
- **진행 방법**:
  1. 질의응답 시간: 학생들이 질문을 하고 강사가 답변
  2. 강의 마무리: 주요 내용 요약 및 향후 학습 방향 제시

**질의응답 세션**:
- 학습한 알고리즘과 자료구조에 대한 질문
- 실전 문제 해결 과정에서의 궁금증 해결
- 추가 학습 자료 및 방향에 대한 조언

**강의 마무리**:
- 전체 강의 내용 요약
- 향후 학습 방향 제시: 고급 알고리즘, 데이터 구조 심화 학습, 문제 해결 능력 향상
- 수료증 수여 및 격려

---

### 과제

#### 과제 1: 최종 복습 자료 작성
- 학습한 주요 개념과 알고리즘을 정리한 최종 복습 자료를 작성하세요.

#### 과제 2: 실전 문제 풀이
- 제공된 실전 문제를 해결하고, 문제 해결 과정을 설명하는 보고서를 작성하세요.

#### 과제 3: 피드백 및 개선
- 강의와 과제에 대한 피드백을 작성하고, 향후 학습 계획을 세우세요.

---

### 퀴즈

#### 퀴즈 1: 동적 계획법의 주요 특징은 무엇인가요?
1. 입력 데이터를 분할하여 병렬로 처리한다.
2. 중복 계산을 피하기 위해 이전에 계산한 값을 저장한다.
3. 탐욕적 선택을 통해 최적의 해를 찾는다.
4. 모든 가능한 경우의 수를 체계적으로 탐색한다.

**정답**: 2. 중복 계산을 피하기 위해 이전에 계산한 값을 저장한다.

#### 퀴즈 2: 그래프 알고리즘에서 최소 스패닝 트리를 찾는 방법은 무엇인가요?
1. 깊이 우선 탐색
2. 너비 우선 탐색
3. 프림 알고리즘
4. 플로이드-워셜 알고리즘

**정답**: 3. 프림 알고리즘

#### 퀴즈 3: 분할 정복 기법의 주요 단계는 무엇인가요?
1. 정의, 나누기, 합병
2. 나누기, 정복, 합병
3. 나누기, 해결, 결합
4. 분할, 해결, 병합

**정답**: 2. 나누기, 정복, 합병

이 계획안은 20주차에 학습한 내용을 복습하고, 최종 평가를 통해 학습 성과를 측정하며, 질의응답 시간을 통해 궁금증을 해결하고 강의를 마무리할 수 있도록 구성되었습니다.