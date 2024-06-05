### 15주차 강의 상세 계획: 강화학습 특강 - 멀티 에이전트 시스템

#### 강의 목표
- 멀티 에이전트 강화학습의 기본 개념과 원리 이해
- 주요 멀티 에이전트 강화학습 알고리즘 학습
- 멀티 에이전트 시스템에서의 문제 해결 경험

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 멀티 에이전트 강화학습의 기본 개념 (20분)

##### 멀티 에이전트 강화학습이란?
- **정의**: 여러 에이전트가 상호작용하며 학습하는 강화학습 방법.
- **주요 특징**:
  - 에이전트 간의 협력과 경쟁.
  - 환경의 복잡성 증가.
  - 분산 학습 가능.

##### 멀티 에이전트 강화학습의 응용
- **게임**: 팀 기반의 게임 AI.
- **로봇 제어**: 다수의 로봇이 협력하여 작업 수행.
- **자율 주행**: 여러 자율 주행 차량의 협력.

#### 1.2 주요 멀티 에이전트 강화학습 알고리즘 (40분)

##### 독립 Q-러닝 (Independent Q-Learning)
- **정의**: 각 에이전트가 독립적으로 Q-러닝을 수행.
- **주요 특징**: 단순 구현, 에이전트 간의 상호작용을 고려하지 않음.
- **알고리즘 과정**:
  1. 각 에이전트가 독립적으로 Q-러닝 알고리즘을 수행.
  2. 각 에이전트가 자신의 Q 테이블을 업데이트.

##### 협력 Q-러닝 (Cooperative Q-Learning)
- **정의**: 에이전트들이 협력하여 공동 목표를 달성하는 Q-러닝.
- **주요 특징**: 에이전트 간의 협력을 통해 학습 성과 향상.
- **알고리즘 과정**:
  1. 에이전트들이 협력하여 Q 테이블을 공유하거나 업데이트.
  2. 공동 목표를 달성하기 위해 에이전트 간의 정보 교환.

##### MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
- **정의**: 여러 에이전트가 각각의 정책을 학습하는 심층 강화학습 알고리즘.
- **주요 특징**: 연속적 행동 공간에서의 학습, 각 에이전트가 다른 에이전트의 정책을 고려.
- **알고리즘 과정**:
  1. 각 에이전트가 자신의 정책 네트워크와 Q 네트워크를 학습.
  2. 다른 에이전트의 행동을 입력으로 사용하여 Q 값을 예측.
  3. 각 에이전트가 자신의 정책을 업데이트.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 독립 Q-러닝 구현 실습

##### 필요 라이브러리 설치
```bash
pip install numpy matplotlib torch gym
```

##### 독립 Q-러닝 구현 코드 (Python)
```python
import gym
import numpy as np
import random

class IndependentQLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = np.zeros((state_size, action_size))
    
    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.choice(range(self.action_size))
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        target = reward + (1 - done) * self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
        if done:
            self.exploration_rate *= self.exploration_decay

def train_independent_q_learning(env, num_episodes, agents):
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        while True:
            actions = [agent.choose_action(state) for agent in agents]
            next_state, reward, done, _ = env.step(actions)
            for i, agent in enumerate(agents):
                agent.learn(state, actions[i], reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                rewards.append(total_reward)
                break
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
    return rewards

# 환경 설정 및 독립 Q-러닝 에이전트 학습
env = gym.make('CartPole-v1')
num_agents = 2
agents = [IndependentQLearningAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n) for _ in range(num_agents)]
rewards = train_independent_q_learning(env, num_episodes=1000, agents=agents)

# 학습 결과 시각화
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Independent Q-Learning Training Rewards')
plt.show()
```

---

### 3. 협력 Q-러닝 및 MADDPG 구현 실습

#### 3.1 협력 Q-러닝 구현 실습

##### 협력 Q-러닝 구현 코드 (Python)
```python
# 협력 Q-러닝 에이전트
class CooperativeQLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = np.zeros((state_size, action_size))
    
    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.choice(range(self.action_size))
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        target = reward + (1 - done) * self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
        if done:
            self.exploration_rate *= self.exploration_decay

def train_cooperative_q_learning(env, num_episodes, agents):
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        while True:
            actions = [agent.choose_action(state) for agent in agents]
            next_state, reward, done, _ = env.step(actions)
            for agent in agents:
                agent.learn(state, actions, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                rewards.append(total_reward)
                break
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
    return rewards

# 환경 설정 및 협력 Q-러닝 에이전트 학습
env = gym.make('CartPole-v1')
num_agents = 2
agents = [CooperativeQLearningAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n) for _ in range(num_agents)]
rewards = train_cooperative_q_learning(env, num_episodes=1000, agents=agents)

# 학습 결과 시각화
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Cooperative Q-Learning Training Rewards')
plt.show()
```

#### 3.2 MADDPG 구현 실습

##### MADDPG 구현 코드 (Python)
```python
import torch
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128,

 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, num_agents, actor_lr, critic_lr, gamma, tau):
        self.num_agents = num_agents
        self.actors = [Actor(state_dim, action_dim) for _ in range(num_agents)]
        self.critics = [Critic(state_dim * num_agents, action_dim * num_agents) for _ in range(num_agents)]
        self.target_actors = [Actor(state_dim, action_dim) for _ in range(num_agents)]
        self.target_critics = [Critic(state_dim * num_agents, action_dim * num_agents) for _ in range(num_agents)]
        
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critics]
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(1000000)
        
        for i in range(num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
    
    def act(self, states):
        actions = []
        for i, actor in enumerate(self.actors):
            state = torch.FloatTensor(states[i]).unsqueeze(0)
            action = actor(state).detach().numpy()[0]
            actions.append(action)
        return actions
    
    def step(self, states, actions, rewards, next_states, dones):
        self.memory.push(states, actions, rewards, next_states, dones)
        if len(self.memory) > 1000:
            self.learn()
    
    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(256)
        
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[:, i, :])
            action = torch.FloatTensor(actions[:, i, :])
            reward = torch.FloatFloatTensor(rewards[:, i])
            next_state = torch.FloatFloatTensor(next_states[:, i, :])
            done = torch.FloatFloatTensor(dones[:, i])
            
            next_actions = [self.target_actors[j](torch.FloatFloatTensor(next_states[:, j, :])) for j in range(self.num_agents)]
            next_actions = torch.cat(next_actions, 1)
            
            next_state = torch.cat(next_states, 1)
            q_target_next = self.target_critics[i](next_state, next_actions).squeeze(1)
            q_target = reward + self.gamma * q_target_next * (1 - done)
            
            q_expected = self.critics[i](torch.cat(states, 1), torch.cat(actions, 1)).squeeze(1)
            critic_loss = F.mse_loss(q_expected, q_target.detach())
            
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()
            
            actions_pred = [self.actors[j](torch.FloatFloatTensor(states[:, j, :])) if j == i else actions[:, j, :] for j in range(self.num_agents)]
            actions_pred = torch.cat(actions_pred, 1)
            actor_loss = -self.critics[i](torch.cat(states, 1), actions_pred).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
            
            self.soft_update(self.actors[i], self.target_actors[i])
            self.soft_update(self.critics[i], self.target_critics[i])
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)

# 환경 설정 및 MADDPG 에이전트 학습
env = gym.make('Pendulum-v0')
num_agents = 2
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = MADDPGAgent(state_dim, action_dim, num_agents, actor_lr=0.001, critic_lr=0.001, gamma=0.99, tau=0.01)

for episode in range(1000):
    states = env.reset()
    total_reward = 0
    while True:
        actions = agent.act(states)
        next_states, rewards, dones, _ = env.step(actions)
        agent.step(states, actions, rewards, next_states, dones)
        states = next_states
        total_reward += sum(rewards)
        if all(dones):
            break
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# 학습 결과 시각화
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('MADDPG Training Rewards')
plt.show()
```

### 준비 자료
- **강의 자료**: 멀티 에이전트 강화학습, 독립 Q-러닝, 협력 Q-러닝 및 MADDPG 슬라이드 (PDF)
- **참고 코드**: 독립 Q-러닝, 협력 Q-러닝 및 MADDPG 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 멀티 에이전트 강화학습, 독립 Q-러닝, 협력 Q-러닝 및 MADDPG의 개념과 알고리즘 정리.
- **코드 실습**: 독립 Q-러닝, 협력 Q-러닝 및 MADDPG를 사용하여 멀티 에이전트 문제 해결 및 결과 시각화.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 멀티 에이전트 강화학습의 개념과 원리를 이해하고, 주요 알고리즘을 학습하며, 실제 문제를 해결하는 경험을 쌓을 수 있도록 유도합니다.

---
