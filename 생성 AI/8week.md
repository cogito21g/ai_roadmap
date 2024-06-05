### 8주차 강의 계획: 강화학습과 생성 모델

#### 강의 목표
- 강화학습의 기본 개념과 원리 이해
- 강화학습을 생성 모델에 적용하는 방법 학습
- 강화학습과 GAN의 결합 사례 학습 및 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 강화학습의 기본 개념 (20분)

##### 강화학습이란?
- **정의**: 강화학습(RL, Reinforcement Learning)은 에이전트가 환경과 상호작용하면서 보상을 최대화하도록 학습하는 방법입니다.
- **구성 요소**:
  - **에이전트(Agent)**: 행동을 수행하는 주체.
  - **환경(Environment)**: 에이전트가 상호작용하는 대상.
  - **상태(State)**: 환경의 현재 상태.
  - **행동(Action)**: 에이전트가 선택하는 행동.
  - **보상(Reward)**: 에이전트가 행동을 수행한 후 받는 값.

##### 강화학습의 주요 알고리즘
- **Q-러닝(Q-Learning)**: 상태-행동 값 함수를 학습하여 최적의 정책을 찾는 방법.
- **SARSA**: Q-러닝과 유사하지만, 에피소드의 모든 단계에서 학습하는 방법.
- **정책 그라디언트(Policy Gradient)**: 정책을 직접 최적화하는 방법.

#### 1.2 강화학습과 생성 모델의 결합 (20분)

##### 강화학습을 통한 생성 모델 학습
- **GAN에서의 강화학습**:
  - 강화학습의 원리를 사용하여 Generator와 Discriminator의 성능을 향상.
  - 예: 강화학습 기반의 GAN 최적화 기법.

##### 강화학습 기반의 생성 모델
- **Generative Adversarial Imitation Learning (GAIL)**:
  - 강화학습과 GAN의 결합을 통해 에이전트의 행동을 학습.
- **Policy Gradient 기반의 생성 모델**:
  - 정책 그라디언트 방법을 사용하여 생성 모델 학습.

##### 강화학습과 생성 모델의 응용
- **게임 AI**: 강화학습을 통해 게임에서의 전략 생성.
- **자율 주행**: 강화학습을 통해 자율 주행 차량의 경로 생성.
- **텍스트 생성**: 강화학습을 통해 자연어 생성을 최적화.

#### 1.3 강화학습과 GAN의 결합 사례 (20분)

##### 강화학습과 GAN의 융합 사례
- **Improving GAN Training with RL**:
  - 강화학습을 사용하여 GAN의 학습 안정성을 향상.
- **Reinforced GAN**:
  - 강화학습을 사용하여 생성 모델의 품질을 향상.

##### 사례 연구
- **논문 리뷰**: "Improving GAN Training with Reward Signals from Pretrained Classifiers"
  - GAN과 강화학습의 결합을 통한 성능 향상 사례.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 강화학습 기반 GAN 구현 실습

##### 필요 라이브러리 설치
```bash
pip install torch torchvision gym matplotlib
```

##### 강화학습 기반 GAN 구현 코드 (Python 3.10 및 PyTorch)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import gym
import matplotlib.pyplot as plt

# 하이퍼파라미터 설정
batch_size = 64
learning_rate = 0.0002
num_epochs = 10
latent_dim = 100
gamma = 0.99  # 할인율

# 데이터셋 로드 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Generator 모델 정의
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img

# Discriminator 모델 정의
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# 강화학습 환경 설정 (OpenAI Gym)
env = gym.make('CartPole-v1')

# 모델 초기화
generator = Generator()
discriminator = Discriminator()

# 손실 함수 및 최적화 기법 설정
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# 강화학습 함수 정의
def select_action(state, policy):
    state = torch.FloatTensor(state).unsqueeze(0)
    probs = policy(state)
    action = probs.multinomial(num_samples=1)
    return action.item()

def train_RL(policy, optimizer, gamma):
    state = env.reset()
    rewards = []
    log_probs = []
    for t in range(1000):
        action = select_action(state, policy)
        state, reward, done, _ = env.step(action)
        log_prob = torch.log(policy(torch.FloatTensor(state))[action])
        rewards.append(reward)
        log_probs.append(log_prob)
        if done:
            break
    R = 0
    policy_loss = []
    returns = []
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

# GAN 학습
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        
        # 진짜 및 가짜 라벨 설정
        real = torch.ones(imgs.size(0), 1)
        fake = torch.zeros(imgs.size(0), 1)
        
        # 진짜 이미지 학습
        real_imgs = imgs
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), real)
        
        # 가짜 이미지 학습
        z = torch.randn(imgs.size(0), latent_dim)
        gen_imgs = generator(z)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        
        # Discriminator 업데이트
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        # Generator 업데이트
        optimizer_G.zero_grad()
        g_loss = adversarial_loss(discriminator(gen_imgs), real)
        g_loss.backward()
        optimizer_G.step()
        
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(train_loader)} \
                  Loss D: {d_loss.item()}, loss G: {g_loss.item()}")

    # 강화학습을 통한 정책 업데이트
    policy = nn.Sequential(
        nn.Linear(4, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
        nn.Softmax(dim=1)
    )
    optimizer_policy = optim.Adam(policy.parameters(), lr=1e-2)
    train_RL(policy, optimizer_policy, gamma)

    # 생성된 이미지 시각화
    with torch.no_grad():
        z = torch.randn(16, latent_dim)
        gen_imgs = generator(z).detach().cpu()
        grid = torchvision.utils.make_grid(gen_imgs, nrow=4, normalize=True)
        plt.imshow(grid.permute(1, 2, 0))
        plt.title('Generated Images')
        plt.show()
```

### 준비 자료
- **강의 자료**: 강화학습과 생성 모델 (RL-GAN) 슬라이드 (PDF)
- **참고 코드**: 강화학습 기반 GAN 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 강화학습과 생성 모델의 결합 방법 정리.
- **코드 실습**: 강화학습 기반 GAN 구현 코드 실행 및 결과 분석.
- **과제 제출

**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획을 통해 학생들이 강화학습의 기본 개념과 원리를 이해하고, 강화학습을 생성 모델에 적용하는 방법을 학습하여 실제로 구현하는 경험을 쌓을 수 있도록 유도합니다.