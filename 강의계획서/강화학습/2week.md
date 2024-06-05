### 2주차 강의 상세 계획: MDP와 기본 원리

#### 강의 목표
- 마코프 결정 과정(MDP)의 개념과 원리 이해
- MDP를 사용하여 강화학습 문제를 모델링하는 방법 학습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 마코프 결정 과정(MDP) 개요 (20분)

##### MDP의 정의
- **정의**: 상태, 행동, 보상, 상태 전이 확률로 정의되는 수학적 모델.
- **주요 요소**:
  - **상태 집합(S)**: 가능한 모든 상태의 집합.
  - **행동 집합(A)**: 가능한 모든 행동의 집합.
  - **보상 함수(R)**: 상태와 행동에 따라 주어지는 보상.
  - **상태 전이 확률(P)**: 상태와 행동에 따른 다음 상태의 확률 분포.

#### 1.2 MDP의 기본 원리 (20분)
- **정책(Policy)**: 상태에서 행동을 선택하는 전략.
  - \(\pi(a|s)\): 상태 \(s\)에서 행동 \(a\)를 선택할 확률.
- **가치 함수(Value Function)**: 상태 또는 상태-행동 쌍의 가치를 평가.
  - **상태 가치 함수**: \(V(s)\)
  - **행동 가치 함수**: \(Q(s, a)\)
- **벨만 방정식(Bellman Equation)**: 가치 함수 간의 관계를 정의.
  - **상태 가치 함수 벨만 방정식**: \(V(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V(s')]\)
  - **행동 가치 함수 벨만 방정식**: \(Q(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s, a) \max_{a'} Q(s', a')\)

#### 1.3 MDP의 예제 (20분)
- **그리드 월드(Grid World)**: 강화학습 문제를 설명하기 위한 간단한 예제.
- **예제 문제**: 특정 목표 지점으로 이동하는 에이전트의 경로 최적화.
- **MDP 모델링**: 상태, 행동, 보상, 상태 전이 확률 정의.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 MDP 모델링 실습

##### 필요 라이브러리 설치
```bash
pip install numpy matplotlib
```

##### MDP 구현 코드 (Python)
```python
import numpy as np
import matplotlib.pyplot as plt

# 그리드 월드 환경 설정
class GridWorld:
    def __init__(self, size):
        self.size = size
        self.state = (0, 0)
        self.terminal_states = [(size-1, size-1)]
        self.actions = ['up', 'down', 'left', 'right']
    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        x, y = self.state
        if action == 'up':
            x = max(0, x - 1)
        elif action == 'down':
            x = min(self.size - 1, x + 1)
        elif action == 'left':
            y = max(0, y - 1)
        elif action == 'right':
            y = min(self.size - 1, y + 1)
        
        self.state = (x, y)
        if self.state in self.terminal_states:
            return self.state, 1, True
        return self.state, -0.1, False

# MDP 모델링
grid_world = GridWorld(size=5)

# 정책 정의 (무작위 정책)
def random_policy(state):
    return np.random.choice(grid_world.actions)

# 가치 함수 초기화
V = np.zeros((grid_world.size, grid_world.size))
gamma =

 0.9

# 가치 반복 알고리즘
for _ in range(100):
    for i in range(grid_world.size):
        for j in range(grid_world.size):
            state = (i, j)
            if state in grid_world.terminal_states:
                continue
            new_value = 0
            for action in grid_world.actions:
                next_state, reward, _ = grid_world.step(action)
                new_value += 1/len(grid_world.actions) * (reward + gamma * V[next_state])
            V[state] = new_value

# 가치 함수 시각화
plt.imshow(V, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('State Value Function')
plt.show()
```

### 준비 자료
- **강의 자료**: MDP와 기본 원리 슬라이드 (PDF)
- **참고 코드**: MDP 모델링 및 가치 함수 구현 예제 코드 (Python)

### 과제
- **이론 정리**: MDP의 개념과 벨만 방정식 정리.
- **코드 실습**: MDP를 사용하여 그리드 월드 문제 해결 및 가치 함수 시각화.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 강화학습의 기본 개념과 원리를 이해하고, MDP를 사용하여 실제 문제를 모델링하고 해결하는 경험을 쌓을 수 있도록 유도합니다.