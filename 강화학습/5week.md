### 5주차 강의 상세 계획: 몬테카를로 방법

#### 강의 목표
- 몬테카를로 방법의 개념과 원리 이해
- 몬테카를로 예측 및 제어 방법 학습
- 몬테카를로 방법을 사용한 문제 해결 경험

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 몬테카를로 방법의 기본 개념 (20분)

##### 몬테카를로 방법이란?
- **정의**: 무작위 샘플링을 통해 확률적 문제를 해결하는 기법.
- **주요 특징**:
  - 에피소드 단위로 학습.
  - 모델 없이 데이터로부터 직접 학습.
  - 각 상태의 가치 추정에 전체 에피소드 사용.

##### 몬테카를로 방법의 응용
- **강화학습**: 환경 모델이 없을 때 가치 함수와 정책을 추정하는 데 사용.

#### 1.2 몬테카를로 예측 (20분)

##### 몬테카를로 예측의 정의
- **목적**: 주어진 정책을 따라 샘플링된 에피소드를 통해 상태 가치 함수 \( V(s) \)를 추정.

##### 몬테카를로 예측의 과정
1. **에피소드 생성**: 정책을 따라 에이전트가 환경과 상호작용하여 에피소드 생성.
2. **반환 계산**: 각 상태에서 시작하는 반환(Return)을 계산.
3. **상태 가치 함수 갱신**: 각 상태의 반환을 평균하여 상태 가치 함수 갱신.

##### 몬테카를로 예측 알고리즘
- 모든 상태의 반환을 저장하고, 반환의 평균으로 가치 함수 갱신.

#### 1.3 몬테카를로 제어 (20분)

##### 몬테카를로 제어의 정의
- **목적**: 최적 정책을 찾기 위해 정책을 개선하는 과정.

##### 몬테카를로 제어의 과정
1. **초기 정책 설정**: 무작위 정책 설정.
2. **정책 평가**: 몬테카를로 예측을 사용하여 현재 정책 평가.
3. **정책 개선**: \(\epsilon\)-탐욕 정책으로 정책 개선.
4. **반복**: 정책 평가와 정책 개선을 반복하여 최적 정책 수렴.

##### 몬테카를로 제어 알고리즘
- \(\epsilon\)-탐욕 정책을 사용하여 탐험과 활용 균형 유지.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 몬테카를로 예측 실습

##### 필요 라이브러리 설치
```bash
pip install numpy matplotlib
```

##### 몬테카를로 예측 구현 코드 (Python)
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

# 무작위 정책 정의
def random_policy(state):
    return np.random.choice(grid_world.actions)

# 몬테카를로 예측 알고리즘
def mc_prediction(env, policy, num_episodes, gamma=0.9):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)

    for _ in range(num_episodes):
        episode = []
        state = env.reset()
        while True:
            action = policy(state)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if state not in [x[0] for x in episode[:t]]:
                returns_sum[state] += G
                returns_count[state] += 1.0
                V[state] = returns_sum[state] / returns_count[state]

    return V

# 그리드 월드 환경 생성
grid_world = GridWorld(size=5)

# 몬테카를로 예측 수행
V = mc_prediction(grid_world, random_policy, num_episodes=1000)

# 가치 함수 시각화
grid = np.zeros((grid_world.size, grid_world.size))
for state, value in V.items():
    grid[state] = value
plt.imshow(grid, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('State Value Function')
plt.show()
```

#### 2.2 몬테카를로 제어 실습

##### 몬테카를로 제어 구현 코드 (Python)
```python
# 몬테카를로 제어 알고리즘
def mc_control(env, num_episodes, gamma=0.9, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(len(env.actions)))
    policy = defaultdict(lambda: np.random.choice(env.actions))
    
    def generate_episode(policy):
        episode = []
        state = env.reset()
        while True:
            action = np.random.choice(env.actions) if np.random.rand() < epsilon else policy[state]
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode
    
    for _ in range(num_episodes):
        episode = generate_episode(policy)
        G = 0
        returns_sum = defaultdict(float)
        returns_count = defaultdict(float)
        
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if (state, action) not in [(x[0], x[1]) for x in episode[:t]]:
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1.0
                Q[state][env.actions.index(action)] = returns_sum[(state, action)] / returns_count[(state, action)]
                policy[state] = env.actions[np.argmax(Q[state])]

    return policy, Q

# 몬테카를로 제어 수행
policy, Q = mc_control(grid_world, num_episodes=1000)

# 정책 시각화
policy_grid = np.zeros((grid_world.size, grid_world.size), dtype=str)
for state in policy.keys():
    policy_grid[state] = policy[state]
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
- **강의 자료**: 몬테카를로 방법, 예측 및 제어 슬라이드 (PDF)
- **참고 코드**: 몬테카를로 예측 및 제어 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 몬테카를로 예측 및 제어의 개념과 알고리즘 정리.
- **코드 실습**: 몬테카를로 예측 및 제어를 사용하여 그리드 월드 문제 해결 및 결과 시각화.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 몬테카를로 방법의 개념과 원리를 이해하고, 예측 및 제어 알고리즘을 학

습하며, 실제 문제를 해결하는 경험을 쌓을 수 있도록 유도합니다.