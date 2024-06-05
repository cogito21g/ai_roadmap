### 6주차 강의 상세 계획: 시간차 학습

#### 강의 목표
- 시간차 학습(TD Learning)의 개념과 원리 이해
- 주요 TD 학습 알고리즘 (TD(0), Sarsa, Q-learning) 학습
- TD 학습을 사용한 문제 해결 경험

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 시간차 학습의 기본 개념 (20분)

##### 시간차 학습이란?
- **정의**: 현재 상태의 가치 추정과 다음 상태의 가치 추정 간의 차이를 이용해 학습하는 방법.
- **주요 특징**:
  - 모델 프리 방식 (model-free).
  - 에피소드가 끝나기 전에 학습 가능.
  - 몬테카를로 방법과 동적 프로그래밍의 중간 방식.

##### 시간차 학습의 응용
- **강화학습**: MDP에서 가치 함수와 정책을 학습하는 데 사용.

#### 1.2 TD(0) 알고리즘 (15분)

##### TD(0)의 정의
- **목적**: 현재 상태의 가치 \( V(s_t) \)를 다음 상태의 가치 \( V(s_{t+1}) \)를 이용해 업데이트.

##### TD(0) 알고리즘
1. **초기화**: 모든 상태의 가치 함수 \( V(s) \)를 임의의 값으로 초기화.
2. **반복**:
   - 현재 상태 \( s_t \)에서 행동 \( a_t \)를 선택하고 실행.
   - 보상 \( r_{t+1} \)와 다음 상태 \( s_{t+1} \)를 관찰.
   - 가치 함수 업데이트:
     \[
     V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]
     \]
   - \( s_t \leftarrow s_{t+1} \)로 상태 업데이트.
3. **종료 조건**: 상태가 터미널 상태에 도달할 때까지 반복.

#### 1.3 Sarsa 알고리즘 (15분)

##### Sarsa의 정의
- **목적**: 현재 상태-행동 쌍의 가치 \( Q(s_t, a_t) \)를 다음 상태-행동 쌍의 가치 \( Q(s_{t+1}, a_{t+1}) \)를 이용해 업데이트.

##### Sarsa 알고리즘
1. **초기화**: 모든 상태-행동 쌍의 가치 함수 \( Q(s, a) \)를 임의의 값으로 초기화.
2. **반복**:
   - 현재 상태 \( s_t \)에서 행동 \( a_t \)를 선택하고 실행.
   - 보상 \( r_{t+1} \)와 다음 상태 \( s_{t+1} \)를 관찰.
   - 다음 상태 \( s_{t+1} \)에서 다음 행동 \( a_{t+1} \)를 선택.
   - 가치 함수 업데이트:
     \[
     Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]
     \]
   - \( s_t \leftarrow s_{t+1} \), \( a_t \leftarrow a_{t+1} \)로 상태 및 행동 업데이트.
3. **종료 조건**: 상태가 터미널 상태에 도달할 때까지 반복.

#### 1.4 Q-learning 알고리즘 (15분)

##### Q-learning의 정의
- **목적**: 현재 상태-행동 쌍의 가치 \( Q(s_t, a_t) \)를 다음 상태에서의 최대 가치 \( \max_{a'} Q(s_{t+1}, a') \)를 이용해 업데이트.

##### Q-learning 알고리즘
1. **초기화**: 모든 상태-행동 쌍의 가치 함수 \( Q(s, a) \)를 임의의 값으로 초기화.
2. **반복**:
   - 현재 상태 \( s_t \)에서 행동 \( a_t \)를 선택하고 실행.
   - 보상 \( r_{t+1} \)와 다음 상태 \( s_{t+1} \)를 관찰.
   - 가치 함수 업데이트:
     \[
     Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]
     \]
   - \( s_t \leftarrow s_{t+1} \)로 상태 업데이트.
3. **종료 조건**: 상태가 터미널 상태에 도달할 때까지 반복.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 TD(0) 구현 실습

##### 필요 라이브러리 설치
```bash
pip install numpy matplotlib
```

##### TD(0) 구현 코드 (Python)
```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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
    
    def step(self, action):
        if self.state in self.terminal_states:
            return self.state, 0, True
        effect = self.action_effects[action]
        new_state = (max(0, min(self.size-1, self.state[0] + effect[0])),
                     max(0, min(self.size-1, self.state[1] + effect[1])))
        reward = 1 if new_state in self.terminal_states else -0.1
        self.state = new_state
        return new_state, reward, new_state in self.terminal_states
    
    def render(self):
        grid = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.state:
                    grid[i, j] = 1
                elif (i, j) in self.terminal_states:
                    grid[i, j] = 2
        plt.imshow(grid, cmap='cool')
        plt.show()

# TD(0) 알고리즘
def td_0(env, num_episodes, alpha=0.1, gamma=0.9):
    V = defaultdict(float)
    
    for _ in range(num_episodes):
        state = env.reset()
        while True:
            action = np.random.choice(env.actions)
            next_state, reward, done = env.step(action)
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state
            if done:
                break
    
    return V

# 그리드 월드 환경 생성
grid_world = GridWorld(size=5)

# TD(0) 수행
V = td_0(grid_world, num_episodes=1000)

# 가치 함수 시각화
grid = np.zeros((grid_world.size, grid_world.size))
for state, value in V.items():
    grid[state] = value
plt.imshow(grid, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('State Value Function')
plt.show()
```

#### 2.2 Sarsa 구현 실습

##### Sarsa 구현 코드 (Python)
```python
# Sarsa 알고리즘
def sarsa(env, num_episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(len(env.actions)))
    
    def epsilon_greedy(state):
        if np.random.rand() < epsilon:
            return np.random.choice(env.actions)
        else:
            return env.actions[np.argmax(Q[state])]
    
    for _ in range(num_episodes):
        state = env.reset()
        action = epsilon_greedy(state)
        while True:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy(next_state)
            Q[state][env.actions.index(action)] += alpha * (reward + gamma * Q[next_state][env.actions.index(next_action)] - Q[state][env.actions.index(action)])
            state, action = next_state, next_action
            if done:
                break
    
    return Q

# Sarsa 수행
Q = sarsa(grid_world, num_episodes=1000)

# 정책 시각화
policy_grid = np.zeros((grid_world.size, grid_world.size), dtype=str)
for state in Q.keys():
    policy_grid[state] = env.actions[np.argmax(Q[state])]
plt.imshow(np.vectorize(grid_world.actions.index)(policy_grid), cmap='tab20', interpolation='nearest')


plt.colorbar()
plt.title('Optimal Policy')
plt.show()

# 행동 가치 함수 시각화
grid = np.zeros((grid_world.size, grid_world.size))
for state, actions in Q.items():
    grid[state] = np.max(actions)
plt.imshow(grid, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('State-Action Value Function')
plt.show()
```

#### 2.3 Q-learning 구현 실습

##### Q-learning 구현 코드 (Python)
```python
# Q-learning 알고리즘
def q_learning(env, num_episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(len(env.actions)))
    
    def epsilon_greedy(state):
        if np.random.rand() < epsilon:
            return np.random.choice(env.actions)
        else:
            return env.actions[np.argmax(Q[state])]
    
    for _ in range(num_episodes):
        state = env.reset()
        while True:
            action = epsilon_greedy(state)
            next_state, reward, done = env.step(action)
            Q[state][env.actions.index(action)] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][env.actions.index(action)])
            state = next_state
            if done:
                break
    
    return Q

# Q-learning 수행
Q = q_learning(grid_world, num_episodes=1000)

# 정책 시각화
policy_grid = np.zeros((grid_world.size, grid_world.size), dtype=str)
for state in Q.keys():
    policy_grid[state] = env.actions[np.argmax(Q[state])]
plt.imshow(np.vectorize(grid_world.actions.index)(policy_grid), cmap='tab20', interpolation='nearest')
plt.colorbar()
plt.title('Optimal Policy')
plt.show()

# 행동 가치 함수 시각화
grid = np.zeros((grid_world.size, grid_world.size))
for state, actions in Q.items():
    grid[state] = np.max(actions)
plt.imshow(grid, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('State-Action Value Function')
plt.show()
```

### 준비 자료
- **강의 자료**: 시간차 학습, TD(0), Sarsa 및 Q-learning 슬라이드 (PDF)
- **참고 코드**: TD(0), Sarsa 및 Q-learning 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 시간차 학습, TD(0), Sarsa 및 Q-learning의 개념과 알고리즘 정리.
- **코드 실습**: TD(0), Sarsa 및 Q-learning을 사용하여 그리드 월드 문제 해결 및 결과 시각화.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 시간차 학습의 개념과 원리를 이해하고, 주요 TD 학습 알고리즘을 학습하며, 실제 문제를 해결하는 경험을 쌓을 수 있도록 유도합니다.