### 1주차 강의: 생성 모델 개요 및 구현

#### 강의 목표
- 생성 모델의 기본 개념과 역사 이해
- 생성 모델의 주요 응용 분야 파악
- GAN 원본 논문 읽기 및 토론
- 간단한 GAN 구현 실습

#### 강의 구성
- **이론 강의**: 1시간
- **논문 읽기 및 토론**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 생성 모델의 정의와 기본 개념 (15분)

##### 생성 모델이란?
- **정의**: 생성 모델(Generative Models)은 주어진 데이터의 분포를 학습하여 새로운 데이터를 생성하는 모델입니다.
- **주요 목표**:
  - 데이터의 근본적인 구조와 분포를 이해하고 학습.
  - 학습된 분포를 바탕으로 새로운 데이터 포인트를 생성.

##### 생성 모델 vs. 분류 모델
- **Discriminative Models**:
  - 주어진 입력이 어떤 클래스에 속하는지를 학습합니다.
  - 예: 로지스틱 회귀, 서포트 벡터 머신(SVM).
- **Generative Models**:
  - 데이터의 근본적인 분포를 학습하고 새로운 데이터를 생성합니다.
  - 예: GMM, HMM, VAE, GAN.

#### 1.2 생성 모델의 역사 (10분)

##### 초기 생성 모델
- **Gaussian Mixture Models (GMM)**:
  - 여러 개의 가우시안 분포를 조합하여 데이터를 모델링.
- **Hidden Markov Models (HMM)**:
  - 관측 가능한 데이터 뒤에 숨겨진 상태를 모델링.

##### 현대 생성 모델
- **Variational Autoencoders (VAEs)**:
  - 확률적 그래픽 모델 기반의 생성 모델.
  - Latent Space를 통해 데이터의 잠재적 표현 학습.
- **Generative Adversarial Networks (GANs)**:
  - 두 개의 신경망 (Generator와 Discriminator)의 경쟁적 학습.
  - Goodfellow et al.이 2014년에 제안.

#### 1.3 주요 생성 모델 소개 (15분)
- **Variational Autoencoders (VAEs)**:
  - 잠재 공간(Latent Space)에서 데이터를 생성.
  - 데이터의 압축과 복원이 주요 메커니즘.
- **Generative Adversarial Networks (GANs)**:
  - Generator: 무작위 노이즈를 입력받아 데이터를 생성.
  - Discriminator: 생성된 데이터와 실제 데이터를 구분.
  - 경쟁적 학습을 통해 Generator의 성능을 향상.
- **Autoregressive Models**:
  - 데이터 포인트를 순차적으로 생성.
  - 예: PixelRNN, PixelCNN.
- **Flow-based Models**:
  - 데이터의 정확한 확률 분포를 학습.
  - 예: RealNVP, Glow.

#### 1.4 생성 모델의 응용 분야 (10분)
- **이미지 생성**:
  - 새로운 이미지를 생성하거나 기존 이미지를 변형.
- **텍스트 생성**:
  - 자동 글쓰기, 챗봇, 번역 등.
- **음성 생성**:
  - 음성 합성, 음악 생성.
- **데이터 증강**:
  - 부족한 데이터를 보완하여 머신러닝 모델의 성능 향상.

---

### 2. 논문 읽기 및 토론 (1시간)

#### 2.1 논문 소개
- **논문 제목**: Generative Adversarial Networks
- **저자**: Ian Goodfellow et al.
- **발표 연도**: 2014

#### 2.2 논문 핵심 내용
- **GAN의 구조**:
  - Generator: 무작위 노이즈를 입력받아 데이터를 생성.
  - Discriminator: 생성된 데이터와 실제 데이터를 구분.
- **경쟁적 학습**:
  - Generator는 Discriminator를 속이기 위해 학습.
  - Discriminator는 진짜와 가짜 데이터를 구분하기 위해 학습.
- **수학적 배경**:
  - Minimax 게임 이론.
  - Cost 함수와 최적화 방법.

#### 2.3 논문 토론
- **질문**:
  - GAN의 장점과 단점은 무엇인가?
  - GAN이 다른 생성 모델과 비교했을 때 가지는 독특한 점은 무엇인가?
  - 논문의 실험 결과와 한계점은 무엇인가?
- **토론 주제**:
  - GAN의 실용적인 응용 분야.
  - GAN의 발전 방향과 최근 연구 동향.

---

### 3. 코드 구현 실습 (1시간)

#### 3.1 간단한 GAN 구현

##### 필요 라이브러리 설치
```bash
pip install torch torchvision matplotlib
```

##### GAN 구현 코드 (Python 3.10 및 PyTorch)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 하이퍼파라미터 설정
batch_size = 64
learning_rate = 0.0002
num_epochs = 10
latent_dim = 100

# 데이터셋 로드
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
            nn.Linear(28*28, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
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

# 모델 초기화
generator = Generator()
discriminator = Discriminator()

# 손실 함수 및 최적화 기법 설정
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# 학습
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
        
        if i % 400 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(train_loader)} \
                  Loss D: {d_loss.item()}, loss G: {g_loss.item()}")

    # 생성된 이미지 시각화
    with torch.no_grad():
        z = torch.randn(16, latent_dim)
        gen_imgs = generator(z).detach().cpu()
        grid = torchvision.utils.make_grid(gen_imgs, nrow=4, normalize=True)
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()
```

#### 3.2 실습 결과
- 코드를 실행하고, 각 에포크마다 생성된 이미지를 시각화하여 확인.
- Generator와 Discriminator의 손실 값(Loss)이 학습 과정에서 어떻게 변화하는지 관찰.

### 준비 자료
- **강의 자료**: 생성 모델 개요 슬라이드 (PDF)
- **참고 논문**: "Generative Adversarial Networks" by Ian Goodfellow et al. (PDF)

### 과제
- **논문 읽기**: "Generative Adversarial Networks" 논문을 읽고 요약 작성.
- **코드

 실습**: GAN 구현 코드 실행 및 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획을 통해 학생들이 생성 모델의 기본 개념을 이해하고, GAN 논문을 통해 실질적인 학습을 할 수 있도록 유도하며, 간단한 GAN 구현을 통해 실습 경험을 쌓을 수 있도록 합니다.