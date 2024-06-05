### 6주차 강의 계획: 다양한 GAN 구조

#### 강의 목표
- 다양한 GAN 구조 (cGAN, DCGAN 등)의 원리와 구현 방법 이해
- PyTorch를 이용한 DCGAN 구현 및 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 Conditional GAN (cGAN) (20분)

##### cGAN의 기본 개념
- **정의**: Conditional GAN (cGAN)은 조건부 정보를 활용하여 데이터를 생성하는 GAN 구조입니다.
- **구성 요소**:
  - **조건부 정보**: 클래스 레이블, 텍스트 설명 등.
  - **Generator와 Discriminator**: 입력에 조건부 정보를 결합하여 학습.

##### cGAN의 학습 과정
- **Generator**: 조건부 정보를 입력받아 해당 조건에 맞는 데이터를 생성.
- **Discriminator**: 생성된 데이터와 실제 데이터를 구분하며, 조건부 정보도 함께 입력으로 받음.

##### cGAN의 응용
- **이미지 생성**: 특정 클래스의 이미지를 생성.
- **텍스트-이미지 변환**: 텍스트 설명을 기반으로 이미지를 생성.

#### 1.2 Deep Convolutional GAN (DCGAN) (20분)

##### DCGAN의 기본 개념
- **정의**: DCGAN은 Convolutional Neural Network (CNN)을 기반으로 한 GAN 구조로, 이미지 생성에 특화되어 있습니다.
- **구성 요소**:
  - **Convolutional Layers**: 이미지의 공간적 특징을 효과적으로 학습.
  - **Batch Normalization**: 학습 안정성과 속도 향상.
  - **ReLU 및 LeakyReLU**: 활성화 함수로 사용.

##### DCGAN의 네트워크 구조
- **Generator**:
  - ConvTranspose2d 층을 사용하여 이미지를 업샘플링.
  - ReLU 활성화 함수와 Batch Normalization 사용.
- **Discriminator**:
  - Conv2d 층을 사용하여 이미지를 다운샘플링.
  - LeakyReLU 활성화 함수와 Batch Normalization 사용.

##### DCGAN의 응용
- **이미지 생성**: 고해상도 이미지 생성.
- **스타일 변환**: 입력 이미지의 스타일을 변환.

#### 1.3 최근 GAN 연구 (20분)

##### StyleGAN 및 BigGAN
- **StyleGAN**:
  - 스타일 트랜스퍼와 생성 이미지의 높은 품질.
  - 잠재 공간의 조작을 통한 다양한 이미지 생성.
- **BigGAN**:
  - 대규모 데이터셋에서의 고품질 이미지 생성.
  - 높은 해상도와 다양한 이미지 생성 능력.

##### GAN의 한계와 해결 방안
- **한계**:
  - 학습 불안정성.
  - 모드 붕괴(Mode Collapse).
- **해결 방안**:
  - 개선된 손실 함수.
  - 다양한 아키텍처 실험.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 DCGAN 구현 실습

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

# 하이퍼파라미터 설정
batch_size = 128
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
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img

# Discriminator 모델 정의
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1).squeeze(1)

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
        z = torch.randn(imgs.size(0), latent_dim, 1, 1)
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

    # 생성된 이미지 시각화
    with torch.no_grad():
        z = torch.randn(16, latent_dim, 1, 1)
        gen_imgs = generator(z).detach().cpu()
        grid = torchvision.utils.make_grid(gen_imgs, nrow=4, normalize=True)
        plt.imshow(grid.permute(1, 2, 0))
        plt.title('Generated Images')
        plt.show()
```

### 준비 자료
- **강의 자료**: 다양한 GAN 구조 (cGAN, DCGAN 등) 슬라이드 (PDF)
- **참고 코드**: DCGAN 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 다양한 GAN 구조 및 응용 분야 정리.
- **코드 실습**: DCGAN 구현 코드 실행 및 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획을 통해 학생들이 다양한 GAN 구조를 이해하고, DCGAN을 이용하여 실제 데이터를 생성하는 경험을 쌓을 수 있도록 유도합니다.