### 7주차 강의 상세 계획: 이미지 생성

#### 강의 목표
- GAN(Generative Adversarial Networks)의 기본 개념과 구조 이해
- PyTorch를 이용한 DCGAN(Deep Convolutional GAN) 모델 구현 및 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 GAN의 기본 개념 (20분)

##### GAN이란?
- **정의**: Generative Adversarial Networks (GANs)는 두 개의 신경망(Generator와 Discriminator)이 경쟁적으로 학습하여 데이터를 생성하는 모델.
- **구성 요소**:
  - **Generator (G)**: 무작위 노이즈를 입력 받아 현실적인 데이터를 생성.
  - **Discriminator (D)**: 생성된 데이터와 실제 데이터를 구분.
- **경쟁적 학습**:
  - Generator는 Discriminator를 속이기 위해 학습.
  - Discriminator는 진짜 데이터와 가짜 데이터를 구분하기 위해 학습.

#### 1.2 DCGAN (Deep Convolutional GAN) (20분)

##### DCGAN의 기본 구조
- **Generator**:
  - 여러 층의 ConvTranspose2d 레이어를 통해 이미지를 점진적으로 업샘플링.
  - ReLU 활성화 함수와 BatchNorm2d 사용.
  - 최종 출력 레이어에서 Tanh 활성화 함수 사용.
- **Discriminator**:
  - 여러 층의 Conv2d 레이어를 통해 이미지를 다운샘플링.
  - LeakyReLU 활성화 함수와 BatchNorm2d 사용.
  - 최종 출력 레이어에서 Sigmoid 활성화 함수 사용.

##### DCGAN의 학습 과정
- **손실 함수**: Binary Cross-Entropy Loss 사용.
- **최적화 기법**: Adam Optimizer 사용.
- **학습 루프**:
  - Generator와 Discriminator를 번갈아가며 업데이트.
  - Generator는 Discriminator를 속이는 방향으로 학습.
  - Discriminator는 실제 데이터와 생성된 데이터를 구분하는 방향으로 학습.

#### 1.3 DCGAN의 응용 (20분)

##### 이미지 생성
- **예시**: 얼굴 생성, 풍경 이미지 생성.
- **적용 사례**: DeepArt, StyleGAN 등.

##### 데이터 증강
- **예시**: 의료 데이터, 소수 클래스 데이터 증강.
- **적용 사례**: 데이터 불균형 문제 해결.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 PyTorch를 이용한 DCGAN 구현

##### 필요 라이브러리 설치
```bash
pip install torch torchvision matplotlib
```

##### DCGAN 구현 코드 (Python 3.10 및 PyTorch)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 하이퍼파라미터 설정
batch_size = 128
learning_rate = 0.0002
num_epochs = 10
latent_dim = 100
image_size = 64
channels = 1

# 데이터셋 로드 및 전처리
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Generator 모델 정의
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator 모델 정의
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img).view(-1, 1).squeeze(1)

# 모델 초기화
generator = Generator()
discriminator = Discriminator()

# 손실 함수 및 최적화 기법 설정
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# 학습
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        
        # 진짜 및 가짜 라벨 설정
        real = torch.ones(imgs.size(0), 1)
        fake = torch.zeros(imgs.size(0), 1)
        
        # 진짜 이미지 학습
        real_imgs = imgs
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_imgs), real)
        
        # 가짜 이미지 학습
        z = torch.randn(imgs.size(0), latent_dim, 1, 1)
        gen_imgs = generator(z)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        
        # Discriminator 업데이트
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        # Generator 업데이트
        optimizer_G.zero_grad()
        g_loss = criterion(discriminator(gen_imgs), real)
        g_loss.backward()
        optimizer_G.step()
        
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(train_loader)} \
                  Loss D: {d_loss.item()}, loss G: {g_loss.item()}")

    # 생성된 이미지 시각화
    with torch.no_grad():
        z = torch.randn(64, latent_dim, 1, 1)
        gen_imgs = generator(z).detach().cpu()
        grid = torchvision.utils.make_grid(gen_imgs, nrow=8, normalize=True)
        plt.imshow(grid.permute(1, 2, 0))
        plt.title('Generated Images')
        plt.show()
```

### 준비 자료
- **강의 자료**: GAN 및 DCGAN 개요 슬라이드 (PDF)
- **참고 코드**: DCGAN 구현 예제 코드 (Python)

### 과제
- **이론 정리**: GAN과 DCGAN의 기본 개념과 학습 과정 정리.
- **코드 실습**: DCGAN 구현 코드 실행 및 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 GAN의 기본 개념과 DCGAN의 구조를 이해하고, PyTorch를 이용하여 DCGAN을 구현하며, 실제 데이터를 통해 모델을 학습시키고 예측 결과를 분석하는 경험을 쌓을 수 있도록 유도합니다.