### 8주차 강의 상세 계획: 심층 강화학습 개요

#### 강의 목표
- 심층 강화학습(Deep Reinforcement Learning)의 기본 개념과 원리 이해
- 주요 심층 강화학습 알고리즘 (DQN, DDQN, DDPG) 학습
- 심층 강화학습을 사용한 문제 해결 경험

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 심층 강화학습의 기본 개념 (20분)

##### 심층 강화학습이란?
- **정의**: 딥러닝과 강화학습을 결합하여 복잡한 환경에서의 학습을 가능하게 하는 방법.
- **주요 특징**:
  - 딥러닝을 사용하여 상태나 행동을 표현.
  - 강화학습을 통해 정책이나 가치 함수를 학습.

##### 심층 강화학습의 응용
- **게임 플레이**: 알파고, 딥마인드의 아타리 게임.
- **로봇 제어**: 복잡한 로봇의 움직임 학습.
- **자율 주행**: 자율 주행 차량의 경로 계획 및 제어.

#### 1.2 심층 Q-네트워크(DQN) (20분)

##### DQN의 정의
- **목적**: Q-learning을 딥러닝으로 확장하여 복잡한 환경에서의 학습 가능하게 함.

##### DQN 알고리즘의 과정
1. **경험 재생(Experience Replay)**:
   - 에이전트의 경험을 저장하고 샘플링하여 학습.
2. **타겟 네트워크(Target Network)**:
   - Q값의 안정성을 위해 타겟 네트워크 사용.
3. **손실 함수**:
   \[
   L(\theta) = \mathbb{E}_{(s, a, r, s') \sim D} \left[ \left( r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2 \right]
   \]

##### DQN의 장단점
- **장점**: 높은 샘플 효율성, 복잡한 환경에서의 성공적인 학습.
- **단점**: 불안정한 학습, 느린 수렴.

#### 1.3 이중 DQN(Double DQN) 및 DDPG (20분)

##### 이중 DQN(Double DQN)
- **목적**: DQN의 Q값 과대평가 문제를 해결.
- **알고리즘**:
  \[
  L(\theta) = \mathbb{E}_{(s, a, r, s') \sim D} \left[ \left( r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_\theta(s', a')) - Q_\theta(s, a) \right)^2 \right]
  \]

##### 심층 결정적 정책 그래디언트(DDPG)
- **목적**: 연속적 행동 공간에서의 강화학습 문제 해결.
- **구성 요소**:
  - **정책 네트워크(Actor)**: 최적 정책 학습.
  - **Q-네트워크(Critic)**: 정책 평가.
- **알고리즘**:
  \[
  \nabla_{\theta^\mu} J \approx \mathbb{E}_{s \sim D} \left[ \nabla_a Q(s, a|\theta^Q) |_{a=\mu(s|\theta^\mu)} \nabla_{\theta^\mu} \mu(s|\theta^\mu) \right]
  \]

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 DQN 구현 실습

##### 필요 라이브러리 설치
```bash
pip install numpy matplotlib torch gym
```

##### DQN 구현 코드 (Python)
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

# DQN 에이전트
class DQNAgent:
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
        next_state = torch.FloatTensor(next_state)
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

# 환경 설정 및 DQN 에이전트 학습
env = gym.make('CartPole-v1')
agent = DQNAgent(env, buffer_size=10000, batch_size=32, gamma=0.99, lr=0.001, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)
rewards = agent.train(num_episodes=1000)

# 학습 결과 시각화
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DQN Training Rewards')
plt.show()
```

### 준비 자료
- **강의 자료**: 심층 강화학습 개요, DQN 및 DDPG 슬라이드 (PDF)
- **참고 코드**: DQN 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 심층 강화학습, DQN 및 DDPG의 개념과 알고리즘 정리.
- **코드 실습**: DQN을 사용

하여 CartPole 문제 해결 및 결과 시각화.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 심층 강화학습의 개념과 원리를 이해하고, 주요 알고리즘을 학습하며, 실제 문제를 해결하는 경험을 쌓을 수 있도록 유도합니다.