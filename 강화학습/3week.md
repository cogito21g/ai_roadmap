### 3주차 강의 상세 계획: 가치 함수와 벨만 방정식

#### 강의 목표
- 가치 함수(Value Function)와 벨만 방정식(Bellman Equation) 이해
- 최적 정책을 찾기 위한 방법 학습
- 가치 함수와 벨만 방정식을 사용한 문제 해결 경험

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 가치 함수의 정의와 역할 (20분)

##### 가치 함수란?
- **정의**: 상태 또는 상태-행동 쌍의 가치를 평가하는 함수.
- **상태 가치 함수 \(V(s)\)**: 상태 \(s\)에서 얻을 수 있는 기대 보상.
- **행동 가치 함수 \(Q(s, a)\)**: 상태 \(s\)에서 행동 \(a\)를 수행할 때 얻을 수 있는 기대 보상.

##### 가치 함수의 역할
- **정책 평가**: 현재 정책의 성능을 평가.
- **정책 개선**: 가치 함수를 사용하여 더 나은 정책을 찾음.

#### 1.2 벨만 방정식 (20분)

##### 벨만 방정식의 정의
- **상태 가치 함수 벨만 방정식**:
  \[
  V(s) = \mathbb{E}_\pi [R_{t+1} + \gamma V(S_{t+1}) | S_t = s]
  \]
  - 현재 상태의 가치는 현재 보상과 다음 상태의 가치의 합으로 표현.
- **행동 가치 함수 벨만 방정식**:
  \[
  Q(s, a) = \mathbb{E}_\pi [R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') | S_t = s, A_t = a]
  \]
  - 현재 상태와 행동의 가치는 현재 보상과 다음 상태에서 최적 행동의 가치의 합으로 표현.

##### 벨만 최적 방정식
- **상태 가치 함수 벨만 최적 방정식**:
  \[
  V^*(s) = \max_\pi \mathbb{E}_\pi [R_{t+1} + \gamma V^*(S_{t+1}) | S_t = s]
  \]
  - 최적 상태 가치는 모든 가능한 정책 중 최대 기대 보상을 제공하는 정책에 해당.
- **행동 가치 함수 벨만 최적 방정식**:
  \[
  Q^*(s, a) = \mathbb{E} [R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') | S_t = s, A_t = a]
  \]
  - 최적 행동 가치는 다음 상태에서 최적 행동의 가치의 합으로 표현.

#### 1.3 최적 정책 (20분)

##### 정책 평가와 정책 개선
- **정책 평가**: 주어진 정책의 가치 함수를 계산.
- **정책 개선**: 현재 정책의 가치 함수를 사용하여 더 나은 정책으로 업데이트.
- **정책 반복**: 정책 평가와 정책 개선을 반복하여 최적 정책을 찾음.

##### 가치 반복(Value Iteration)
- **정의**: 벨만 최적 방정식을 반복적으로 적용하여 최적 가치 함수를 찾는 방법.
- **알고리즘**:
  1. 가치 함수 초기화.
  2. 벨만 최적 방정식을 사용하여 가치 함수 업데이트.
  3. 가치 함수가 수렴할 때까지 반복.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 벨만 방정식과 가치 반복 구현 실습

##### 필요 라이브러리 설치
```bash
pip install numpy matplotlib
```

##### 벨만 방정식과 가치 반복 구현 코드 (Python)
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
        self.action_effects = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, state, action):
        if state in self.terminal_states:
            return state, 0
        effect = self.action_effects[action]
        new_state = (max(0, min(self.size-1, state[0] + effect[0])),
                     max(0, min(self.size-1, state[1] + effect[1])))
        reward = 1 if new_state in self.terminal_states else -0.1
        return new_state, reward

# 가치 반복 알고리즘
def value_iteration(env, gamma=0.9, theta=1e-6):
    V = np.zeros((env.size, env.size))
    while True:
        delta = 0
        for i in range(env.size):
            for j in range(env.size):
                state = (i, j)
                if state in env.terminal_states:
                    continue
                v = V[state]
                new_values = []
                for action in env.actions:
                    next_state, reward = env.step(state, action)
                    new_values.append(reward + gamma * V[next_state])
                V[state] = max(new_values)
                delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V

# 그리드 월드 환경 생성
grid_world = GridWorld(size=5)

# 가치 반복 수행
V = value_iteration(grid_world)

# 가치 함수 시각화
plt.imshow(V, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('State Value Function')
plt.show()
```

### 준비 자료
- **강의 자료**: 가치 함수와 벨만 방정식 슬라이드 (PDF)
- **참고 코드**: 벨만 방정식과 가치 반복 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 가치 함수와 벨만 방정식, 최적 정책의 개념 정리.
- **코드 실습**: 벨만 방정식과 가치 반복을 사용하여 그리드 월드 문제 해결 및 가치 함수 시각화.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 가치 함수와 벨만 방정식의 개념을 이해하고, 이를 사용하여 최적 정책을 찾는 방법을 학습하며, 실제 문제를 해결하는 경험을 쌓을 수 있도록 유도합니다.