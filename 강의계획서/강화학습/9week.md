### 9주차 강의 상세 계획: 심층 강화학습 심화

#### 강의 목표
- 심층 강화학습의 심화 개념과 최신 알고리즘 이해
- 이중 DQN (Double DQN), 우선순위 경험 재생 (Prioritized Experience Replay) 학습
- 심층 강화학습 심화 기법을 사용한 문제 해결 경험

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 이중 DQN (Double DQN) (30분)

##### 이중 DQN의 목적
- **정의**: DQN의 Q값 과대평가 문제를 해결하기 위해 두 개의 Q 네트워크 사용.
- **주요 특징**:
  - 하나의 네트워크로 행동을 선택하고, 다른 네트워크로 Q값을 평가.

##### 이중 DQN 알고리즘의 과정
1. **Q 네트워크**:
   - Q 네트워크와 타겟 네트워크 두 개 사용.
2. **손실 함수**:
   \[
   L(\theta) = \mathbb{E}_{(s, a, r, s') \sim D} \left[ \left( r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_\theta(s', a')) - Q_\theta(s, a) \right)^2 \right]
   \]

#### 1.2 우선순위 경험 재생 (30분)

##### 우선순위 경험 재생의 목적
- **정의**: 중요한 경험을 더 자주 재생하여 학습 속도와 효율성 향상.
- **주요 특징**:
  - TD 오류를 기반으로 샘플의 중요도 결정.
  - 중요도가 높은 샘플을 더 자주 학습.

##### 우선순위 경험 재생 알고리즘의 과정
1. **TD 오류 계산**:
   - 경험마다 TD 오류를 계산하고 우선순위 설정.
2. **경험 샘플링**:
   - 우선순위에 따라 경험을 샘플링하여 학습.
3. **손실 함수**:
   \[
   L(\theta) = \frac{1}{N} \sum_{i=1}^{N} p_i \left( r_i + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2
   \]

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 이중 DQN 구현 실습

##### 필요 라이브러리 설치
```bash
pip install numpy matplotlib torch gym
```

##### 이중 DQN 구현 코드 (Python)
```python
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# 신경망 정의
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 경험 재생 버퍼
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def __len__(self):
        return len(self.buffer)

# 이중 DQN 에이전트
class DoubleDQNAgent:
    def __init__(self, env, buffer_size, batch_size, gamma, lr, epsilon_start, epsilon_end, epsilon_decay):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.q_network = DQN(self.state_dim, self.action_dim)
        self.target_network = DQN(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def select_action(self, state):
        if random.random() > self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_dim)
        return action
    
    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatFloatTensor(next_state)
        done = torch.FloatTensor(done)
        
        q_values = self.q_network(state)
        next_q_values = self.q_network(next_state)
        next_q_state_values = self.target_network(next_state)
        
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(self, num_episodes):
        all_rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            while True:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                
                self.update()
                
                if done:
                    all_rewards.append(episode_reward)
                    break
            
            if episode % 10 == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {self.epsilon}")
        
        return all_rewards

# 환경 설정 및 이중 DQN 에이전트 학습
env = gym.make('CartPole-v1')
agent = DoubleDQNAgent(env, buffer_size=10000, batch_size=32, gamma=0.99, lr=0.001, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)
rewards = agent.train(num_episodes=1000)

# 학습 결과 시각화
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Double DQN Training Rewards')
plt.show()
```

---
