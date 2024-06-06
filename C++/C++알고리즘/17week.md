### 17주차 강의 계획안

#### 강의 주제: 실전 문제 해결 프로젝트 1
- 실제 문제 분석 및 해결
- 팀별 프로젝트 진행

---

### 강의 내용

#### 1. 실제 문제 분석 및 해결
- **개념**: 실제 문제를 분석하고 해결하기 위한 과정
- **단계**:
  1. 문제 정의: 문제의 범위와 목표를 명확히 설정
  2. 요구사항 분석: 해결책을 위해 필요한 요구사항을 수집하고 분석
  3. 해결 방안 설계: 문제를 해결하기 위한 알고리즘과 자료구조 설계
  4. 구현: 설계한 알고리즘과 자료구조를 코드로 구현
  5. 테스트 및 검증: 구현한 해결 방안을 다양한 테스트 케이스로 검증
  6. 문서화 및 발표: 결과를 문서화하고 팀별로 발표

#### 2. 팀별 프로젝트 진행
- **목표**: 팀 단위로 실전 문제를 해결하며 협업 능력과 문제 해결 능력 향상
- **진행 방법**:
  1. 팀 구성: 3~4명으로 팀을 구성
  2. 주제 선정: 팀별로 해결할 문제를 선정
  3. 역할 분담: 팀원 간 역할을 분담 (문제 분석, 알고리즘 설계, 구현, 테스트, 문서화 등)
  4. 진행 상황 점검: 주차별로 진행 상황을 점검하고 피드백 제공
  5. 최종 발표: 프로젝트 결과를 발표하고 피드백

---

### 프로젝트 예제

#### 프로젝트 주제 예시 1: 교통 네트워크 최적화
- **문제 정의**: 도시의 교통 네트워크를 최적화하여 교통 혼잡을 줄이는 방안을 마련
- **요구사항 분석**:
  1. 교통량 데이터 수집
  2. 교통 네트워크 모델링
  3. 최적화 알고리즘 설계 (예: 다익스트라 알고리즘, 플로이드-워셜 알고리즘)
- **해결 방안 설계**:
  - 그래프를 이용한 교통 네트워크 모델링
  - 최적의 경로를 찾기 위한 알고리즘 적용
- **구현**:
  ```cpp
  #include <iostream>
  #include <vector>
  #include <climits>
  using namespace std;

  void floydWarshall(vector<vector<int>>& graph, int V) {
      vector<vector<int>> dist = graph;

      for (int k = 0; k < V; k++) {
          for (int i = 0; i < V; i++) {
              for (int j = 0; j < V; j++) {
                  if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX && dist[i][k] + dist[k][j] < dist[i][j]) {
                      dist[i][j] = dist[i][k] + dist[k][j];
                  }
              }
          }
      }

      cout << "Shortest distances between every pair of vertices:\n";
      for (int i = 0; i < V; i++) {
          for (int j = 0; j < V; j++) {
              if (dist[i][j] == INT_MAX)
                  cout << "INF ";
              else
                  cout << dist[i][j] << " ";
          }
          cout << endl;
      }
  }

  int main() {
      int V = 4;
      vector<vector<int>> graph = {{0, 5, INT_MAX, 10},
                                   {INT_MAX, 0, 3, INT_MAX},
                                   {INT_MAX, INT_MAX, 0, 1},
                                   {INT_MAX, INT_MAX, INT_MAX, 0}};
      floydWarshall(graph, V);
      return 0;
  }
  ```
- **테스트 및 검증**:
  - 다양한 교통 시나리오에 대한 테스트
  - 성능 및 정확성 검증
- **문서화 및 발표**:
  - 프로젝트 결과를 문서화하고 발표 자료 준비
  - 팀별 발표 및 피드백

#### 프로젝트 주제 예시 2: 전자 상거래 추천 시스템
- **문제 정의**: 사용자에게 맞춤형 상품 추천 시스템을 개발
- **요구사항 분석**:
  1. 사용자 행동 데이터 수집
  2. 추천 알고리즘 설계 (예: 협업 필터링, 콘텐츠 기반 필터링)
- **해결 방안 설계**:
  - 사용자-상품 매트릭스 생성
  - 추천 알고리즘 적용
- **구현**:
  ```cpp
  #include <iostream>
  #include <vector>
  #include <unordered_map>
  using namespace std;

  vector<int> recommendProducts(const unordered_map<int, vector<int>>& userProducts, int userId) {
      unordered_map<int, int> productCount;
      for (const auto& [user, products] : userProducts) {
          if (user == userId) continue;
          for (int product : products) {
              productCount[product]++;
          }
      }

      vector<int> recommendations;
      for (const auto& [product, count] : productCount) {
          if (count > 1) recommendations.push_back(product);
      }
      return recommendations;
  }

  int main() {
      unordered_map<int, vector<int>> userProducts = {
          {1, {1, 2, 3}},
          {2, {2, 3, 4}},
          {3, {1, 2, 4}},
          {4, {2, 3}}
      };

      int userId = 1;
      vector<int> recommendations = recommendProducts(userProducts, userId);

      cout << "Recommended products for user " << userId << ": ";
      for (int product : recommendations) {
          cout << product << " ";
      }
      cout << endl;

      return 0;
  }
  ```
- **테스트 및 검증**:
  - 다양한 사용자 데이터에 대한 테스트
  - 추천 정확성 검증
- **문서화 및 발표**:
  - 프로젝트 결과를 문서화하고 발표 자료 준비
  - 팀별 발표 및 피드백

---

### 과제

#### 과제 1: 팀별 프로젝트 계획서 작성
- 팀별로 프로젝트 주제를 선정하고, 문제 정의, 요구사항 분석, 해결 방안 설계 등을 포함한 프로젝트 계획서를 작성하세요.

#### 과제 2: 팀별 프로젝트 진행
- 팀별로 역할을 분담하여 프로젝트를 진행하고, 각 주차별로 진행 상황을 점검하여 피드백을 받으세요.

---

### 퀴즈

#### 퀴즈 1: 문제 정의 단계에서 중요한 요소는 무엇인가요?
1. 문제의 범위와 목표 설정
2. 코드 구현
3. 성능 테스트
4. 발표 자료 준비

**정답**: 1. 문제의 범위와 목표 설정

#### 퀴즈 2: 요구사항 분석 단계에서 수행할 작업은 무엇인가요?
1. 문제의 범위와 목표 설정
2. 코드 구현
3. 해결책을 위해 필요한 요구사항 수집 및 분석
4. 발표 자료 준비

**정답**: 3. 해결책을 위해 필요한 요구사항 수집 및 분석

#### 퀴즈 3: 팀별 프로젝트에서 중요한 협업 요소는 무엇인가요?
1. 코드 작성
2. 발표 준비
3. 역할 분담과 커뮤니케이션
4. 문서화

**정답**: 3. 역할 분담과 커뮤니케이션

이 계획안은 17주차에 실전 문제 해결 프로젝트를 통해 팀별로 문제를 분석하고 해결하는 능력을 기를 수 있도록 구성되었습니다. 팀별 프로젝트를 통해 실전 문제 해결 능력과 협업 능력을 향상시킬 수 있습니다.