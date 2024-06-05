### 10주차 강의 상세 계획: 정책 최적화

#### 강의 목표
- 정책 최적화 기법의 개념과 원리 이해
- A3C, PPO, TRPO 알고리즘 학습
- 정책 최적화 기법을 사용한 문제 해결 경험

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 A3C (Asynchronous Advantage Actor-Critic) (20분)

##### A3C의 목적
- **정의**: 비동기적으로 여러 에이전트를 동시에 학습하여 학습 속도를 향상시키는 방법.
- **주요 특징**:
  - 여러 에이전트가 비동기적으로 환경과 상호작용.
  - 중앙집중식 정책과 가치 함수를 학습.

##### A3C 알고리즘의 과정
1. **여러 에이전트**:
   - 각 에이전트가 독립적으로 환경과 상호작용.
2. **정책 및 가치 함수 업데이트**:
   - 모든 에이전트의 경험을 모아 중앙집중식 네트워크 업데이트.

#### 1.2 PPO (Proximal Policy Optimization) (20분)

##### PPO의 목적
- **정의**: 정책 갱신 시 큰 변화를 방지하여 안정적인 학습을 유도하는 방법.
- **주요 특징**:
  - 클리핑 기법을 사용하여 정책 갱신 시 변화 제한.

##### PPO 알고

리즘의 과정
1. **손실 함수**:
   \[
   L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
   \]
   - \( r_t(\theta) \): 현재 정책과 이전 정책의 비율.
   - \( \hat{A}_t \): 이점 함수.

#### 1.3 TRPO (Trust Region Policy Optimization) (20분)

##### TRPO의 목적
- **정의**: 정책 갱신 시 신뢰 구간을 설정하여 큰 변화를 방지하는 방법.
- **주요 특징**:
  - KL 발산을 사용하여 정책 갱신 시 변화 제한.

##### TRPO 알고리즘의 과정
1. **손실 함수**:
   \[
   L^{TRPO}(\theta) = \mathbb{E}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t \right]
   \]
   - KL 발산을 제한하여 정책 갱신.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 PPO 구현 실습

##### 필요 라이브러리 설치
```bash
pip install numpy matplotlib torch gym
```

##### PPO 구현 코드 (Python)
```python
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 신경망 정의
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return nn.Softmax(dim=-1)(x)

# 이점 신경망 정의
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# PPO 에이전트
class PPOAgent:
    def __init__(self, env, gamma, lr, clip_epsilon, update_steps):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_steps = update_steps
        
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim)
        self.value_net = ValueNetwork(self.state_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_net(state)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)
    
    def compute_returns(self, rewards, dones, next_value):
        returns = []
        R = next_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return returns
    
    def update(self, states, actions, log_probs, returns, advantages):
        for _ in range(self.update_steps):
            state_tensor = torch.FloatTensor(states)
            action_tensor = torch.LongTensor(actions)
            old_log_probs_tensor = torch.FloatTensor(log_probs)
            return_tensor = torch.FloatTensor(returns)
            advantage_tensor = torch.FloatTensor(advantages)
            
            new_action_probs = self.policy_net(state_tensor)
            new_action_dist = Categorical(new_action_probs)
            new_log_probs = new_action_dist.log_prob(action_tensor)
            
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            surr1 = ratio * advantage_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value = self.value_net(state_tensor)
            value_loss = (return_tensor - value).pow(2).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
    
    def train(self, num_episodes):
        all_rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            states = []
            actions = []
            log_probs = []
            rewards = []
            dones = []
            episode_reward = 0
            
            while True:
                action, log_prob = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(done)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    next_value = self.value_net(torch.FloatTensor(next_state).unsqueeze(0)).item()
                    returns = self.compute_returns(rewards, dones, next_value)
                    advantages = [r - v for r, v in zip(returns, self.value_net(torch.FloatTensor(states)).detach().numpy())]
                    self.update(states, actions, log_probs, returns, advantages)
                    all_rewards.append(episode_reward)
                    break
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Reward: {episode_reward}")
        
        return all_rewards

# 환경 설정 및 PPO 에이전트 학습
env = gym.make('CartPole-v1')
agent = PPOAgent(env, gamma=0.99, lr=0.001, clip_epsilon=0.2, update_steps=10)
rewards = agent.train(num_episodes=1000)

# 학습 결과 시각화
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('PPO Training Rewards')
plt.show()
```

### 준비 자료
- **강의 자료**: 정책 최적화, A3C, PPO, TRPO 슬라이드 (PDF)
- **참고 코드**: PPO 구현 예제 코드 (Python)

### 과제
- **이론 정리**: A3C, PPO, TRPO의 개념과 알고리즘 정리.
- **코드 실습**: PPO를 사용하여 CartPole 문제 해결 및 결과 시각화.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 정책 최적화 기법의 개념과 원리를 이해하고, 주요 알고리즘을 학습하며, 실제 문제를 해결하는 경험을 쌓을 수 있도록 유도합니다.

---
