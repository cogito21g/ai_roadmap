### 7주차 강의 상세 계획: 정책 기반 방법

#### 강의 목표
- 정책 기반 방법의 개념과 원리 이해
- REINFORCE 알고리즘 및 액터-크리틱 방법 학습
- 정책 기반 방법을 사용한 문제 해결 경험

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 정책 기반 방법의 기본 개념 (15분)

##### 정책 기반 방법이란?
- **정의**: 정책 함수를 직접 학습하여 최적 정책을 찾는 강화학습 방법.
- **주요 특징**:
  - 정책을 명시적으로 파라미터화.
  - 고정된 정책이 아니라 확률적 정책을 사용.
  - 연속적 행동 공간에서도 사용 가능.

##### 정책 기반 방법의 응용
- **강화학습**: 정책을 직접 학습하여 복잡한 환경에서 최적의 행동을 학습.

#### 1.2 REINFORCE 알고리즘 (20분)

##### REINFORCE 알고리즘이란?
- **정의**: 경사 상승 방법을 사용하여 정책의 파라미터를 업데이트하는 방법.

##### REINFORCE 알고리즘의 과정
1. **에피소드 생성**: 현재 정책을 사용하여 에피소드 생성.
2. **반환 계산**: 각 시간 단계에서의 반환 \( G_t \) 계산.
3. **정책 업데이트**:
   \[
   \theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) G_t
   \]

##### REINFORCE 알고리즘의 장단점
- **장점**: 간단하고 직관적, 정책의 확률적 특성을 유지.
- **단점**: 고차원 환경에서의 느린 수렴, 높은 분산.

#### 1.3 액터-크리틱 방법 (25분)

##### 액터-크리틱 방법이란?
- **정의**: 정책 기반 방법과 가치 기반 방법을 결합하여 학습하는 강화학습 방법.

##### 액터-크리틱의 구성 요소
- **액터(Actor)**: 정책 함수를 학습.
- **크리틱(Critic)**: 가치 함수를 학습하여 액터를 평가.

##### 액터-크리틱 알고리즘의 과정
1. **초기화**: 정책 파라미터 \( \theta \)와 가치 함수 파라미터 \( w \) 초기화.
2. **반복**:
   - 현재 상태 \( s_t \)에서 행동 \( a_t \) 선택.
   - 다음 상태 \( s_{t+1} \)와 보상 \( r_{t+1} \) 관찰.
   - 가치 함수 업데이트:
     \[
     \delta_t = r_{t+1} + \gamma V_w(s_{t+1}) - V_w(s_t)
     \]
     \[
     w \leftarrow w + \beta \delta_t \nabla_w V_w(s_t)
     \]
   - 정책 함수 업데이트:
     \[
     \theta \leftarrow \theta + \alpha \delta_t \nabla_\theta \log \pi_\theta(a_t | s_t)
     \]

##### 액터-크리틱의 장단점
- **장점**: 낮은 분산, 빠른 수렴.
- **단점**: 구현의 복잡성, 하이퍼파라미터 조정 필요.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 REINFORCE 알고리즘 구현 실습

##### 필요 라이브러리 설치
```bash
pip install numpy matplotlib torch
```

##### REINFORCE 알고리즘 구현 코드 (Python)
```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 간단한 환경 설정
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

# 정책 신경망 정의
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = self.fc(x)
        return nn.Softmax(dim=-1)(x)

# REINFORCE 알고리즘
def reinforce(env, policy_net, optimizer, num_episodes, gamma=0.99):
    for _ in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy_net(state_tensor)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            next_state, reward, done = env.step(env.actions[action.item()])
            log_probs.append(action_dist.log_prob(action))
            rewards.append(reward)
            state = next_state
        
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        log_probs = torch.stack(log_probs)
        loss = -torch.sum(log_probs * returns)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 환경 및 정책 신경망 설정
env = GridWorld(size=5)
policy_net = PolicyNetwork(input_size=2, output_size=len(env.actions))
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

# REINFORCE 알고리즘 수행
reinforce(env, policy_net, optimizer, num_episodes=1000)

# 정책 시각화
policy_grid = np.zeros((env.size, env.size), dtype=str)
for i in range(env.size):
    for j in range(env.size):
        state = (i, j)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = policy_net(state_tensor)
        action = torch.argmax(action_probs).item()
        policy_grid[state] = env.actions[action]
plt.imshow(np.vectorize(env.actions.index)(policy_grid), cmap='tab20', interpolation='nearest')
plt.colorbar()
plt.title('Optimal Policy')
plt.show()
```

#### 2.2 액터-크리틱 구현 실습

##### 액터-크리틱 구현 코드 (Python)
```python
# 크리틱 신경망 정의
class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return self.fc(x)

# 액터-크리틱 알고리즘
def actor_critic(env, policy_net, value_net, policy_optimizer, value_optimizer, num_episodes, gamma=0.99):
    for _ in range(num_episodes):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy_net(state_tensor)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            next_state, reward, done = env.step(env.actions[action.item()])
            
            value = value_net(state_tensor)
            log_probs.append(action_dist.log_prob(action))
            values.append(value)
            rewards.append(reward)
            state = next_state
        
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        log_probs = torch.stack(log_probs)
        values = torch.cat(values)
        advantages = returns - values.detach()
        
        policy_loss = -torch.sum(log_probs * advantages)
        value_loss = torch.sum((returns - values) ** 2)
        
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

# 환경 및 네트워크 설정
env = GridWorld(size=

5)
policy_net = PolicyNetwork(input_size=2, output_size=len(env.actions))
value_net = ValueNetwork(input_size=2)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
value_optimizer = optim.Adam(value_net.parameters(), lr=0.01)

# 액터-크리틱 알고리즘 수행
actor_critic(env, policy_net, value_net, policy_optimizer, value_optimizer, num_episodes=1000)

# 정책 시각화
policy_grid = np.zeros((env.size, env.size), dtype=str)
for i in range(env.size):
    for j in range(env.size):
        state = (i, j)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = policy_net(state_tensor)
        action = torch.argmax(action_probs).item()
        policy_grid[state] = env.actions[action]
plt.imshow(np.vectorize(env.actions.index)(policy_grid), cmap='tab20', interpolation='nearest')
plt.colorbar()
plt.title('Optimal Policy')
plt.show()
```

### 준비 자료
- **강의 자료**: 정책 기반 방법, REINFORCE 및 액터-크리틱 슬라이드 (PDF)
- **참고 코드**: REINFORCE 및 액터-크리틱 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 정책 기반 방법, REINFORCE 및 액터-크리틱의 개념과 알고리즘 정리.
- **코드 실습**: REINFORCE 및 액터-크리틱을 사용하여 그리드 월드 문제 해결 및 결과 시각화.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 정책 기반 방법의 개념과 원리를 이해하고, REINFORCE 및 액터-크리틱 알고리즘을 학습하며, 실제 문제를 해결하는 경험을 쌓을 수 있도록 유도합니다.