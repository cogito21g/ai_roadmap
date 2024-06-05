### 42주차: 강화 학습 (Reinforcement Learning) 기초

#### 강의 목표
- 강화 학습의 기본 개념 이해
- 강화 학습의 주요 알고리즘 학습
- 간단한 강화 학습 모델 구현

#### 강의 내용

##### 1. 강화 학습의 기본 개념
- **강화 학습 개요**
  - 정의: 에이전트가 환경과 상호작용하면서 보상을 최대화하는 행동을 학습하는 기계 학습 분야
  - 주요 용어: 에이전트(Agent), 환경(Environment), 상태(State), 행동(Action), 보상(Reward), 정책(Policy), 가치 함수(Value Function), Q-함수(Q-Function)

- **강화 학습의 유형**
  - 모델 기반 강화 학습 (Model-Based Reinforcement Learning)
  - 모델 프리 강화 학습 (Model-Free Reinforcement Learning)

- **강화 학습의 주요 알고리즘**
  - Q-러닝 (Q-Learning)
  - SARSA (State-Action-Reward-State-Action)
  - 딥 Q-네트워크 (Deep Q-Network, DQN)

##### 2. 강화 학습 알고리즘
- **Q-러닝 (Q-Learning)**
  - 설명: 모델 프리 강화 학습의 일종으로, 상태-행동 쌍에 대한 가치를 학습하여 최적의 정책을 찾는 알고리즘
  - 구현 예제:

```python
import numpy as np
import gym

# 환경 설정
env = gym.make('FrozenLake-v0')

# Q-테이블 초기화
Q = np.zeros([env.observation_space.n, env.action_space.n])

# 학습 파라미터
alpha = 0.8
gamma = 0.95
epsilon = 0.1
episodes = 1000

# Q-러닝 알고리즘
for episode in range(episodes):
    state = env.reset()
    done = False

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state, reward, done, _ = env.step(action)

        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
        state = next_state

print("Training finished.\n")

# Q-테이블 출력
print(Q)
```

##### 3. 딥 Q-네트워크 (DQN)
- **딥 Q-네트워크 개요**
  - 설명: 심층 신경망을 사용하여 Q-함수를 근사하는 모델 프리 강화 학습 알고리즘
  - 주요 구성 요소: 경험 재플레이(Experience Replay), 타겟 네트워크(Target Network)

- **DQN 구현 (TensorFlow와 Keras 사용)**

```python
import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random

# 환경 설정
env = gym.make('CartPole-v1')

# 하이퍼파라미터
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32
n_episodes = 1000
output_dir = 'model_output/cartpole/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

agent = DQNAgent(state_size, action_size)

done = False
for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"episode: {e}/{n_episodes}, score: {time}, e: {agent.epsilon:.2}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    if e % 50 == 0:
        agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")
```

#### 과제

1. **Q-러닝 알고리즘 구현**
   - 주어진 환경에서 Q-러닝 알고리즘을 구현하고, Q-테이블을 학습합니다.

2. **DQN 알고리즘 구현**
   - 주어진 환경에서 DQN 알고리즘을 구현하고, 모델을 훈련 및 평가합니다.

3. **강화 학습 실험**
   - 다양한 하이퍼파라미터를 시도해보고, 성능을 비교합니다.
   - 다른 환경에서 강화 학습 알고리즘을 적용해봅니다.

#### 퀴즈

1. **강화 학습에서 에이전트가 환경과 상호작용하며 학습하는 주요 요소가 아닌 것은?**
   - A) 상태(State)
   - B) 행동(Action)
   - C) 보상(Reward)
   - D) 손실(Loss)

2. **Q-러닝의 주요 목적은 무엇인가?**
   - A) 상태-행동 쌍에 대한 가치를 학습하여 최적의 정책을 찾기 위해
   - B) 신경망의 가중치를 조정하기 위해
   - C) 데이터를 군집화하기 위해
   - D) 텍스트 데이터를 분류하기 위해

3. **DQN에서 경험 재플레이(Experience Replay)의 주요 목적은 무엇인가?**
   - A) 에이전트가 더 많은 보상을 얻도록 하기 위해
   - B) 샘플 효율성을 높이고 데이터의 상관성을 줄이기 위해
   - C) 에이전트의 학습 속도를 높이기 위해
   - D) 에이전트의 정책을 변경하기 위해

4. **강화 학습에서 사용하는 주요 알고리즘이 아닌 것은?**
   - A) Q-러닝
   - B) SARSA
   - C) 딥 Q-네트워크(DQN)
   - D) 선형 회귀

#### 퀴즈 해설

1. **강화 학습에서 에이전트가 환경과 상호작용하며 학습하는 주요 요소가 아닌 것은?**
   - **정답: D) 손실(Loss)**
     - 해설: 강화 학습에서 에이전트가 학습하는 주요 요소는 상태(State), 행동(Action), 보상(Reward)입니다. 손실(Loss)은 지도 학습에서 주로 사용되는 개념입니다.

2. **Q-러닝의 주요 목적은 무엇인가?**
   - **정답: A) 상태-행동 쌍에 대한 가치를 학습하여 최적의 정책을 찾기 위해**
     - 해설: Q-러닝은 상태-행동 쌍에 대한 가치를 학습하여 최적의 정책을 찾기 위한 알고리즘입니다.

3. **DQN에서 경험 재플레이(Experience Replay)의 주요 목적은 무엇인가?**
   - **정답: B) 샘플 효율성을 높이고 데이터의 상관성을 줄이기 위해**
     - 해설: 경험 재플레이는 샘플 효율성을 높이고 데이터의 상

관성을 줄이기 위해 사용됩니다.

4. **강화 학습에서 사용하는 주요 알고리즘이 아닌 것은?**
   - **정답: D) 선형 회귀**
     - 해설: 강화 학습에서 사용하는 주요 알고리즘에는 Q-러닝, SARSA, 딥 Q-네트워크(DQN)가 있으며, 선형 회귀는 주로 회귀 문제에 사용되는 알고리즘입니다.

강화 학습을 포함한 모든 주제를 마쳤습니다. 추가적인 학습이 필요하거나 더 알고 싶은 주제가 있으면 언제든지 문의해 주세요.