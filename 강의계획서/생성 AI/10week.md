### 10주차 강의 계획: 변형 가능한 생성 모델

#### 강의 목표
- Flow-based Models의 기본 개념과 원리 이해
- RealNVP와 Glow 모델의 구조와 구현 방법 학습
- Flow-based 모델을 이용한 데이터 생성 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 Flow-based Models의 기본 개념 (20분)

##### Flow-based Models란?
- **정의**: Flow-based Models는 데이터의 정확한 확률 분포를 학습하고, 역함수를 통해 데이터를 생성하는 모델입니다.
- **주요 특징**:
  - 역변환 가능성 (Invertibility): 입력 데이터와 잠재 변수를 서로 변환 가능.
  - 확률 밀도 함수의 직접 계산 가능: 데이터의 확률 밀도를 명시적으로 계산 가능.

##### 대표적인 Flow-based Models
- **RealNVP**: Real-valued Non-Volume Preserving Transformation.
- **Glow**: Generalization of RealNVP with efficient sampling and training.

#### 1.2 RealNVP (20분)

##### RealNVP의 기본 구조
- **Affine Coupling Layer**: 입력 데이터를 두 부분으로 나누고, 한 부분은 변형 없이, 다른 부분은 Affine 변환 적용.
- **역변환 가능성**: 각 층이 역변환 가능하도록 설계되어 전체 모델도 역변환 가능.
- **변환 및 학습**: 변환된 데이터의 로그 확률 밀도를 최대화하도록 학습.

##### RealNVP의 학습 과정
- **학습 데이터**: 입력 데이터를 잠재 변수로 변환하여 학습.
- **손실 함수**: 변환된 데이터의 로그 확률 밀도와 입력 데이터의 확률 밀도를 비교하여 손실 계산.

##### RealNVP의 응용
- **이미지 생성**: 고해상도 이미지 생성.
- **확률 밀도 추정**: 데이터의 확률 밀도 함수 추정.

#### 1.3 Glow (20분)

##### Glow의 기본 구조
- **1x1 Convolution**: Affine Coupling Layer를 개선하여 학습 가능 매개변수를 추가.
- **ActNorm**: 데이터의 분산을 조정하는 정규화 층.
- **Multi-scale Architecture**: 여러 해상도에서 데이터를 처리하여 고해상도 데이터 생성.

##### Glow의 학습 과정
- **학습 데이터**: 입력 데이터를 잠재 변수로 변환하여 학습.
- **손실 함수**: 변환된 데이터의 로그 확률 밀도와 입력 데이터의 확률 밀도를 비교하여 손실 계산.

##### Glow의 응용
- **이미지 생성**: 고해상도 이미지 생성.
- **스타일 변환**: 다양한 스타일의 이미지 생성.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 RealNVP 구현 실습

##### 필요 라이브러리 설치
```bash
pip install torch torchvision matplotlib
```

##### RealNVP 구현 코드 (Python 3.10 및 PyTorch)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 하이퍼파라미터 설정
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# 데이터셋 로드 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Affine Coupling Layer 정의
class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AffineCouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2),
            nn.Tanh()
        )
        self.translation_net = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2)
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        s = self.scale_net(x1)
        t = self.translation_net(x1)
        y1 = x1
        y2 = x2 * torch.exp(s) + t
        return torch.cat([y1, y2], dim=1)

    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=1)
        s = self.scale_net(y1)
        t = self.translation_net(y1)
        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([x1, x2], dim=1)

# RealNVP 모델 정의
class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(RealNVP, self).__init__()
        self.layers = nn.ModuleList([AffineCouplingLayer(input_dim, hidden_dim) for _ in range(num_layers)])

    def forward(self, x):
        log_det_jacobian = 0
        for layer in self.layers:
            x = layer(x)
            log_det_jacobian += layer.scale_net(x.chunk(2, dim=1)[0]).sum()
        return x, log_det_jacobian

    def inverse(self, y):
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y

# 모델 초기화
input_dim = 28 * 28
hidden_dim = 256
num_layers = 5
model = RealNVP(input_dim, hidden_dim, num_layers)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 학습
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        imgs = imgs.view(-1, input_dim).to(torch.float32)
        optimizer.zero_grad()
        output, log_det_jacobian = model(imgs)
        loss = criterion(output, imgs) - log_det_jacobian.mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}] Batch {i}/{len(train_loader)} Loss: {loss.item()}')

# 생성된 이미지 시각화
with torch.no_grad():
    sample = torch.randn(64, input_dim)
    sample = model.inverse(sample).view(-1, 1, 28, 28)
    grid = torchvision.utils.make_grid(sample, nrow=8, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.title('Generated Images')
    plt.show()
```

### 준비 자료
- **강의 자료**: Flow-based Models (RealNVP, Glow) 슬라이드 (PDF)
- **참고 코드**: RealNVP 구현 예제 코드 (Python)

### 과제
- **이론 정리**: Flow-based Models의 원리와 응용 분야 정리.
- **코드 실습**: RealNVP 구현 코드 실행 및 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획을 통해 학생들이 Flow-based Models의 기본 개념과 원리를 이해하고, RealNVP을 이용하여 실제 데이터를 생성하는 경험을 쌓을 수 있도록 유도합니다.