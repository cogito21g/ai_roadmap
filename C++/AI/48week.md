### 48주차: 강화 학습 심화 (Advanced Reinforcement Learning)

#### 강의 목표
- 강화 학습의 고급 알고리즘 이해
- 정책 경사 방법 및 Actor-Critic 알고리즘 학습
- 강화 학습을 위한 고급 기법 구현

#### 강의 내용

##### 1. 정책 경사 (Policy Gradient) 방법
- **정책 경사 개요**
  - 설명: 정책을 직접 파라미터화하여 정책의 파라미터를 최적화하는 방법
  - 주요 알고리즘: REINFORCE, Actor-Critic

- **REINFORCE 알고리즘**
  - 설명: 정책 경사 기법 중 하나로, 정책 네트워크를 사용하여 행동을 선택하고 보상에 따라 정책을 업데이트

```python
import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 환경 설정
env = gym.make('CartPole-v1')

# 하이퍼파라미터
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
learning_rate = 0.01
n_episodes = 1000
gamma = 0.99

# 정책 신경망 구축
model = Sequential([
    Dense(24, input_dim=state_size, activation='relu'),
    Dense(24, activation='relu'),
    Dense(action_size, activation='softmax')
])
optimizer = Adam(lr=learning_rate)

# 정책 경사 함수
def policy_gradient():
    states = tf.placeholder(tf.float32, [None, state_size])
    actions = tf.placeholder(tf.int32, [None])
    advantages = tf.placeholder(tf.float32, [None])
    probs = model(states)
    indices = tf.range(0, tf.shape(probs)[0]) * tf.shape(probs)[1] + actions
    selected_probs = tf.gather(tf.reshape(probs, [-1]), indices)
    loss = -tf.reduce_mean(tf.log(selected_probs) * advantages)
    return states, actions, advantages, loss

states, actions, advantages, loss = policy_gradient()
train_op = optimizer.minimize(loss)

# REINFORCE 알고리즘
for e in range(n_episodes):
    state = env.reset()
    rewards = []
    states_memory = []
    actions_memory = []

    for time in range(500):
        state = np.reshape(state, [1, state_size])
        action_probs = model.predict(state)
        action = np.random.choice(action_size, p=action_probs[0])
        next_state, reward, done, _ = env.step(action)
        states_memory.append(state)
        actions_memory.append(action)
        rewards.append(reward)
        state = next_state
        if done:
            break

    # 할인 보상 계산
    discounted_rewards = []
    cumulative = 0
    for reward in rewards[::-1]:
        cumulative = reward + gamma * cumulative
        discounted_rewards.insert(0, cumulative)

    # 상태, 행동, 할인 보상을 사용하여 정책 신경망 업데이트
    for state, action, reward in zip(states_memory, actions_memory, discounted_rewards):
        _, loss_val = sess.run([train_op, loss], feed_dict={
            states: state, actions: [action], advantages: [reward]
        })
    print(f"Episode: {e}, Score: {sum(rewards)}")

print("Training finished.\n")
```

##### 2. Actor-Critic 알고리즘
- **Actor-Critic 개요**
  - 설명: 정책 함수와 가치 함수를 동시에 학습하는 강화 학습 알고리즘

```python
import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# 환경 설정
env = gym.make('CartPole-v1')

# 하이퍼파라미터
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
learning_rate = 0.001
gamma = 0.99
n_episodes = 1000

# Actor-Critic 신경망 구축
inputs = Input(shape=(state_size,))
dense1 = Dense(24, activation='relu')(inputs)
dense2 = Dense(24, activation='relu')(dense1)
policy = Dense(action_size, activation='softmax')(dense2)
value = Dense(1, activation='linear')(dense2)

actor = Model(inputs, policy)
critic = Model(inputs, value)

actor.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy')
critic.compile(optimizer=Adam(lr=learning_rate), loss='mse')

# Actor-Critic 알고리즘
for e in range(n_episodes):
    state = env.reset()
    states_memory = []
    actions_memory = []
    rewards_memory = []

    for time in range(500):
        state = np.reshape(state, [1, state_size])
        action_probs = actor.predict(state)
        action = np.random.choice(action_size, p=action_probs[0])
        next_state, reward, done, _ = env.step(action)
        states_memory.append(state)
        action_one_hot = np.zeros(action_size)
        action_one_hot[action] = 1
        actions_memory.append(action_one_hot)
        rewards_memory.append(reward)
        state = next_state
        if done:
            break

    # 할인 보상 및 가치 계산
    discounted_rewards = []
    cumulative = 0
    for reward in rewards_memory[::-1]:
        cumulative = reward + gamma * cumulative
        discounted_rewards.insert(0, cumulative)

    values = critic.predict(np.vstack(states_memory))
    advantages = np.array(discounted_rewards) - values.squeeze()

    # Actor와 Critic 업데이트
    actor.fit(np.vstack(states_memory), np.vstack(actions_memory), sample_weight=advantages, epochs=1, verbose=0)
    critic.fit(np.vstack(states_memory), np.array(discounted_rewards), epochs=1, verbose=0)

    print(f"Episode: {e}, Score: {sum(rewards_memory)}")

print("Training finished.\n")
```

##### 3. 심층 강화 학습 (Deep Reinforcement Learning)
- **심층 강화 학습 개요**
  - 설명: 심층 신경망을 사용하여 강화 학습을 수행하는 방법
  - 주요 알고리즘: DQN, Double DQN, DDPG, A3C

- **DQN (Deep Q-Network)**
  - 설명: 심층 신경망을 사용하여 Q-함수를 근사하는 모델 프리 강화 학습 알고리즘

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
        agent.remember(state, action,

 reward, next_state, done)
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

1. **정책 경사 알고리즘 구현**
   - 주어진 환경에서 REINFORCE 알고리즘을 구현하고, 정책 신경망을 학습합니다.

2. **Actor-Critic 알고리즘 구현**
   - 주어진 환경에서 Actor-Critic 알고리즘을 구현하고, 모델을 훈련 및 평가합니다.

3. **DQN 알고리즘 구현**
   - 주어진 환경에서 DQN 알고리즘을 구현하고, 모델을 훈련 및 평가합니다.

#### 퀴즈

1. **정책 경사 방법의 주요 목적은 무엇인가?**
   - A) Q-함수를 학습하여 최적의 정책을 찾기 위해
   - B) 정책을 직접 파라미터화하여 최적의 정책을 찾기 위해
   - C) 가치 함수를 학습하여 최적의 정책을 찾기 위해
   - D) 환경의 모델을 학습하기 위해

2. **REINFORCE 알고리즘의 주요 특징은 무엇인가?**
   - A) 정책과 가치를 동시에 학습한다.
   - B) Q-테이블을 사용하여 최적의 정책을 학습한다.
   - C) 정책 신경망을 사용하여 행동을 선택하고 보상에 따라 정책을 업데이트한다.
   - D) 환경의 모델을 사용하여 정책을 업데이트한다.

3. **Actor-Critic 알고리즘의 주요 특징은 무엇인가?**
   - A) 정책과 가치를 동시에 학습한다.
   - B) Q-테이블을 사용하여 최적의 정책을 학습한다.
   - C) 정책 신경망만을 사용하여 행동을 선택한다.
   - D) 환경의 모델을 사용하여 정책을 업데이트한다.

4. **심층 강화 학습의 주요 장점은 무엇인가?**
   - A) 정책을 직접 학습하여 높은 차원의 문제를 해결할 수 있다.
   - B) 가치 함수를 학습하여 보상을 극대화할 수 있다.
   - C) Q-테이블을 사용하여 학습이 간단하다.
   - D) 환경의 모델을 사용하여 높은 효율성을 가진다.

#### 퀴즈 해설

1. **정책 경사 방법의 주요 목적은 무엇인가?**
   - **정답: B) 정책을 직접 파라미터화하여 최적의 정책을 찾기 위해**
     - 해설: 정책 경사 방법은 정책을 직접 파라미터화하여 최적의 정책을 찾는 방법입니다.

2. **REINFORCE 알고리즘의 주요 특징은 무엇인가?**
   - **정답: C) 정책 신경망을 사용하여 행동을 선택하고 보상에 따라 정책을 업데이트한다.**
     - 해설: REINFORCE 알고리즘은 정책 신경망을 사용하여 행동을 선택하고 보상에 따라 정책을 업데이트합니다.

3. **Actor-Critic 알고리즘의 주요 특징은 무엇인가?**
   - **정답: A) 정책과 가치를 동시에 학습한다.**
     - 해설: Actor-Critic 알고리즘은 정책과 가치를 동시에 학습하는 강화 학습 알고리즘입니다.

4. **심층 강화 학습의 주요 장점은 무엇인가?**
   - **정답: A) 정책을 직접 학습하여 높은 차원의 문제를 해결할 수 있다.**
     - 해설: 심층 강화 학습은 정책을 직접 학습함으로써 높은 차원의 문제를 효과적으로 해결할 수 있는 장점이 있습니다.

더 깊이 있는 학습이 필요하면, 구체적인 주제나 알고 싶은 내용을 요청해 주세요.