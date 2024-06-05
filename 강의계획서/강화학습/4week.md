### 4주차 강의 상세 계획: 동적 프로그래밍

#### 강의 목표
- 동적 프로그래밍의 개념과 원리 이해
- 정책 반복(Policy Iteration)과 가치 반복(Value Iteration) 알고리즘 학습
- 동적 프로그래밍 기법을 사용한 문제 해결 경험

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 동적 프로그래밍의 기본 개념 (15분)

##### 동적 프로그래밍이란?
- **정의**: 최적의 결정을 하기 위해 문제를 작은 부분 문제로 나누어 해결하는 방법.
- **주요 원리**:
  - **최적 부분 구조**: 문제의 최적 해가 부분 문제의 최적 해로 구성.
  - **중복 부분 문제**: 동일한 부분 문제가 여러 번 반복.

##### 동적 프로그래밍의 응용
- **강화학습**: MDP에서 최적 정책을 찾는 문제에 적용.

#### 1.2 정책 반복 (Policy Iteration) (20분)

##### 정책 반복이란?
- **정의**: 주어진 정책을 평가하고, 그 정책을 개선하는 과정을 반복하여 최적 정책을 찾는 방법.

##### 정책 반복의 단계
1. **정책 평가(Policy Evaluation)**:
  - 주어진 정책에 따라 상태 가치 함수 \( V^\pi(s) \)를 계산.
2. **정책 개선(Policy Improvement)**:
  - 현재 상태 가치 함수를 바탕으로 더 나은 정책 \( \pi' \)를 찾음.
3. **정책 반복**:
  - 정책 평가와 정책 개선을 반복하여 최적 정책 \( \pi^* \)에 수렴.

##### 정책 반복 알고리즘
- 초기 정책을 무작위로 설정.
- 정책 평가와 정책 개선을 번갈아 수행.
- 정책이 더 이상 개선되지 않을 때까지 반복.

#### 1.3 가치 반복 (Value Iteration) (25분)

##### 가치 반복이란?
- **정의**: 벨만 최적 방정식을 반복적으로 적용하여 최적 가치 함수를 찾는 방법.

##### 가치 반복의 단계
1. **가치 함수 초기화**: 상태 가치 함수를 무작위로 초기화.
2. **벨만 업데이트**: 벨만 최적 방정식을 사용하여 가치 함수를 업데이트.
3. **정책 추출**: 가치 함수로부터 최적 정책을 추출.

##### 가치 반복 알고리즘
- 초기 가치 함수를 무작위로 설정.
- 벨만 업데이트를 통해 가치 함수가 수렴할 때까지 반복.
- 최적 가치 함수로부터 최적 정책을 추출.

##### 정책 반복과 가치 반복의 비교
- **정책 반복**: 정책 평가와 정책 개선을 반복.
- **가치 반복**: 벨만 최적 방정식을 반복하여 가치 함수를 업데이트.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 정책 반복 구현 실습

##### 필요 라이브러리 설치
```bash
pip install numpy matplotlib
```

##### 정책 반복 구현 코드 (Python)
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

# 정책 반복 알고리즘
def policy_iteration(env, gamma=0.9, theta=1e-6):
    policy = np.random.choice(env.actions, size=(env.size, env.size))
    V = np.zeros((env.size, env.size))
    
    def policy_evaluation(policy):
        while True:
            delta = 0
            for i in range(env.size):
                for j in range(env.size):
                    state = (i, j)
                    if state in env.terminal_states:
                        continue
                    v = V[state]
                    action = policy[state]
                    next_state, reward = env.step(state, action)
                    V[state] = reward + gamma * V[next_state]
                    delta = max(delta, abs(v - V[state]))
            if delta < theta:
                break
    
    def policy_improvement():
        policy_stable = True
        for i in range(env.size):
            for j in range(env.size):
                state = (i, j)
                if state in env.terminal_states:
                    continue
                old_action = policy[state]
                action_values = []
                for action in env.actions:
                    next_state, reward = env.step(state, action)
                    action_values.append(reward + gamma * V[next_state])
                new_action = env.actions[np.argmax(action_values)]
                policy[state] = new_action
                if old_action != new_action:
                    policy_stable = False
        return policy_stable
    
    while True:
        policy_evaluation(policy)
        if policy_improvement():
            break
    
    return policy, V

# 그리드 월드 환경 생성
grid_world = GridWorld(size=5)

# 정책 반복 수행
policy, V = policy_iteration(grid_world)

# 정책 시각화
policy_grid = np.vectorize(lambda x: grid_world.actions.index(x))(policy)
plt.imshow(policy_grid, cmap='tab20', interpolation='nearest')
plt.colorbar()
plt.title('Optimal Policy')
plt.show()

# 가치 함수 시각화
plt.imshow(V, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('State Value Function')
plt.show()
```

### 2.2 가치 반복 구현 실습

##### 가치 반복 구현 코드 (Python)
```python
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
    
    policy = np.zeros((env.size, env.size), dtype=str)
    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            if state in env.terminal_states:
                continue
            action_values = []
            for action in env.actions:
                next_state, reward = env.step(state, action)
                action_values.append(reward + gamma * V[next_state])
            best_action = env.actions[np.argmax(action_values)]
            policy[state] = best_action
    
    return policy, V

# 가치 반복 수행
policy, V = value_iteration(grid_world)

# 정책 시각화
policy_grid = np.vectorize(lambda x: grid_world.actions.index(x))(policy)
plt.imshow(policy_grid, cmap='tab20', interpolation='nearest')
plt.colorbar()
plt.title('Optimal Policy')
plt.show()

# 가치 함수 시각화
plt.imshow(V, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('State Value Function')
plt.show()
```

### 준비 자료
- **강의 자료**: 동적 프로그래밍, 정책 반복 및 가치 반복 슬라이드 (PDF)
- **참고 코드**: 정책 반복 및 가치 반복 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 정책 반복과 가치 반복의 개념과 알고리즘 정리.
- **코드 실습**: 정책 반복 및 가치 반복을 사용하여 그리드 월드 문제 해결 및 가치 함수 시각화.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 동적 프로그래밍의 개념과 원리를 이해하고, 정책 반복 및 가치 반복 알고리즘을 학습하며, 실제 문제를 해결하는 경험을 쌓을 수 있도록 유도합니다.