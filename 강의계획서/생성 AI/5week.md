### 5주차 강의 계획: 기본 GAN 구현

#### 강의 목표
- GAN의 기본 구조와 구현 방법 이해
- PyTorch를 이용한 간단한 GAN 구현 및 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 GAN의 기본 구조 복습 (10분)

##### GAN의 구성 요소
- **Generator (G)**: 무작위 노이즈를 입력 받아 현실적인 데이터를 생성.
- **Discriminator (D)**: 생성된 데이터와 실제 데이터를 구분.

##### GAN의 학습 과정
- **경쟁적 학습**: Generator는 Discriminator를 속이기 위해 학습하고, Discriminator는 진짜와 가짜 데이터를 구분하기 위해 학습.
- **손실 함수**:
  - Generator: \( \log(1 - D(G(z))) \)
  - Discriminator: \( \log(D(x)) + \log(1 - D(G(z))) \)

#### 1.2 GAN 구현의 주요 단계 (30분)

##### 데이터 준비
- **데이터셋 선택**: MNIST 데이터셋 사용.
- **데이터 전처리**: 데이터 정규화 및 배치 처리.

##### 네트워크 설계
- **Generator**:
  - 입력: 무작위 노이즈 벡터.
  - 출력: 생성된 이미지.
- **Discriminator**:
  - 입력: 이미지 (진짜 또는 가짜).
  - 출력: 이미지가 진짜일 확률.

##### 손실 함수와 최적화
- **손실 함수**: Binary Cross-Entropy Loss 사용.
- **최적화 기법**: Adam Optimizer 사용.

##### 학습 과정
- **Generator 업데이트**: 가짜 데이터를 생성하여 Discriminator를 속이도록 학습.
- **Discriminator 업데이트**: 진짜 데이터와 가짜 데이터를 구분하도록 학습.

#### 1.3 GAN의 구현 및 실습 준비 (20분)

##### 구현 순서
1. 데이터 로딩 및 전처리.
2. Generator와 Discriminator 네트워크 설계.
3. 손실 함수 및 최적화 기법 설정.
4. 학습 루프 구성.
5. 학습된 모델을 이용한 데이터 생성 및 시각화.

##### 주요 코드 설명
- 데이터셋 로딩 및 전처리 코드.
- 네트워크 설계 코드.
- 학습 루프 및 모델 업데이트 코드.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 GAN 구현 실습

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

### 준비 자료
- **강의 자료**: GAN 기본 구조 및 구현 슬라이드 (PDF)
- **참고 코드**: GAN 구현 예제 코드 (Python)

### 과제
- **이론 정리**: GAN의 기본 구조 및 학습 과정 정리.
- **코드 실습**: GAN 구현 코드 실행 및 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획을 통해 학생들이 GAN의 기본 구조와 구현 방법을 이해하고, 실제로 간단한 GAN을 구현하여 학습하는 경험을 쌓을 수 있도록 유도합니다.