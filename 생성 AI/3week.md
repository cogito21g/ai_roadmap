### 3주차 강의 계획: Variational Autoencoder (VAE)

#### 강의 목표
- Variational Autoencoder (VAE)의 기본 개념과 구조 이해
- VAE의 Latent Space 개념 학습
- VAE를 이용한 MNIST 데이터셋 적용 및 구현 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 Variational Autoencoder (VAE) 기본 개념 (20분)

##### VAE란?
- **정의**: VAE는 확률적 그래픽 모델과 딥러닝을 결합하여 데이터의 잠재 공간(Latent Space)을 학습하고, 이를 바탕으로 새로운 데이터를 생성하는 모델입니다.
- **구성 요소**:
  - 인코더(Encoder): 입력 데이터를 잠재 공간으로 매핑.
  - 디코더(Decoder): 잠재 공간의 벡터를 원래 데이터 공간으로 변환.

##### VAE의 수학적 배경
- **Latent Variable Model**: 입력 데이터의 잠재 변수(latent variable)를 통해 데이터 생성.
- **Evidence Lower Bound (ELBO)**: VAE의 손실 함수로, 재구성 손실(reconstruction loss)과 정규화 손실(regularization loss)로 구성.
  - **재구성 손실**: 원본 데이터와 재구성된 데이터 간의 차이.
  - **정규화 손실**: 잠재 변수의 분포가 정규 분포를 따르도록 강제.

#### 1.2 VAE의 구조 (20분)

##### 인코더와 디코더
- **인코더**:
  - 입력 데이터를 잠재 변수의 분포 파라미터(평균과 분산)로 매핑.
  - 인코더 네트워크는 입력 데이터에서 잠재 변수의 평균(mean)과 로그 분산(log variance)을 출력.
- **디코더**:
  - 잠재 변수를 입력 받아 원래 데이터 공간으로 변환.
  - 디코더 네트워크는 잠재 변수에서 출력 데이터로 매핑.

##### 잠재 공간(Latent Space)
- **잠재 공간**: 데이터의 주요 특징을 학습한 공간.
- **재구성 및 생성**: 잠재 공간의 벡터를 샘플링하여 새로운 데이터를 생성.

#### 1.3 VAE의 응용 (20분)

##### 데이터 생성
- **이미지 생성**: 새로운 이미지를 생성하거나 기존 이미지를 변형.
- **데이터 증강**: 부족한 데이터를 보완하여 머신러닝 모델의 성능 향상.

##### 데이터 압축
- **차원 축소**: 고차원의 데이터를 저차원으로 압축하여 주요 특징을 추출.

##### 데이터 복원
- **결측 데이터 복원**: 누락된 데이터를 복원하여 데이터 완성도 향상.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 Variational Autoencoder (VAE) 실습

##### 필요 라이브러리 설치
```bash
pip install torch torchvision matplotlib
```

##### VAE 구현 코드 (Python 3.10 및 PyTorch)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 하이퍼파라미터 설정
batch_size = 128
learning_rate = 1e-3
num_epochs = 10
latent_dim = 20

# 데이터셋 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# VAE 모델 정의
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 400),
            nn.ReLU(),
            nn.Linear(400, 2*latent_dim)  # 평균과 로그 분산
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 28*28),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(-1, 28*28)
        h = self.encoder(x)
        mu, log_var = h[:, :latent_dim], h[:, latent_dim:]
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

# 손실 함수 정의
def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 28*28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

# 모델 초기화
vae = VAE()
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# 학습
for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {train_loss/len(train_loader.dataset)}')

# 잠재 공간에서 샘플링하여 이미지 생성
with torch.no_grad():
    z = torch.randn(64, latent_dim)
    sample = vae.decoder(z).cpu()
    sample = sample.view(64, 1, 28, 28)
    grid = torchvision.utils.make_grid(sample, nrow=8, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.title('Generated Images')
    plt.show()
```

### 준비 자료
- **강의 자료**: VAE 개요 슬라이드 (PDF)
- **참고 자료**: Variational Autoencoder 관련 논문 및 책 (PDF)

### 과제
- **이론 정리**: VAE의 원리와 응용 분야 정리.
- **코드 실습**: VAE 구현 코드 실행 및 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획을 통해 학생들이 VAE의 기본 개념과 구조를 이해하고, VAE를 이용한 데이터 생성 실습을 통해 실제 데이터를 모델링하는 경험을 쌓을 수 있도록 유도합니다.