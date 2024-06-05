### 9주차 강의 계획: 자율적 생성 모델

#### 강의 목표
- Autoregressive Models의 기본 개념과 원리 이해
- PixelRNN과 PixelCNN의 구조와 구현 방법 학습
- PixelCNN을 이용한 이미지 생성 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 Autoregressive Models의 기본 개념 (20분)

##### Autoregressive Models란?
- **정의**: Autoregressive Models는 시계열 데이터나 이미지를 순차적으로 생성하는 모델입니다. 이전에 생성된 데이터를 바탕으로 다음 데이터를 생성하는 방식입니다.
- **주요 특징**:
  - 데이터의 순차적 생성.
  - 과거의 데이터를 이용해 현재 데이터를 예측.

##### 대표적인 Autoregressive Models
- **PixelRNN**: 이미지의 픽셀을 순차적으로 생성하는 모델.
- **PixelCNN**: CNN을 사용하여 픽셀을 순차적으로 생성하는 모델.

#### 1.2 PixelRNN (20분)

##### PixelRNN의 기본 구조
- **구조**: 이미지의 픽셀을 왼쪽 위에서 오른쪽 아래로 순차적으로 생성.
- **RNN 사용**: 각 픽셀을 생성하기 위해 이전 픽셀의 정보를 RNN으로 전달.
- **Masked RNN**: 생성할 픽셀의 정보만을 사용하기 위해 마스크 적용.

##### PixelRNN의 학습 과정
- **학습 데이터**: 이미지를 순차적으로 생성하는 데이터셋.
- **손실 함수**: 각 픽셀의 로그 우도(log likelihood)를 최대화.

##### PixelRNN의 응용
- **이미지 생성**: 고해상도 이미지 생성.
- **이미지 복원**: 손상된 이미지의 복원.

#### 1.3 PixelCNN (20분)

##### PixelCNN의 기본 구조
- **구조**: CNN을 사용하여 픽셀을 순차적으로 생성.
- **Masked Convolution**: 생성할 픽셀의 정보만을 사용하기 위해 마스크 적용.
- **Residual Connections**: 학습의 안정성과 성능 향상을 위한 잔차 연결.

##### PixelCNN의 학습 과정
- **학습 데이터**: 이미지를 순차적으로 생성하는 데이터셋.
- **손실 함수**: 각 픽셀의 로그 우도(log likelihood)를 최대화.

##### PixelCNN의 응용
- **이미지 생성**: 고해상도 이미지 생성.
- **이미지 복원**: 손상된 이미지의 복원.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 PixelCNN 구현 실습

##### 필요 라이브러리 설치
```bash
pip install torch torchvision matplotlib
```

##### PixelCNN 구현 코드 (Python 3.10 및 PyTorch)
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

# PixelCNN 모델 정의
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, height, width = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, height // 2, width // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, height // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class PixelCNN(nn.Module):
    def __init__(self):
        super(PixelCNN, self).__init__()
        self.net = nn.Sequential(
            MaskedConv2d('A', 1, 64, 7, 1, 3, bias=False),
            nn.ReLU(),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False),
            nn.ReLU(),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False),
            nn.ReLU(),
            MaskedConv2d('B', 64, 1, 7, 1, 3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 모델 초기화
model = PixelCNN()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

# 학습
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(torch.float32)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, imgs)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}] Batch {i}/{len(train_loader)} Loss: {loss.item()}')

# 생성된 이미지 시각화
with torch.no_grad():
    sample = torch.zeros(64, 1, 28, 28)
    for i in range(28):
        for j in range(28):
            out = model(sample)
            sample[:, :, i, j] = out[:, :, i, j]
    grid = torchvision.utils.make_grid(sample, nrow=8, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.title('Generated Images')
    plt.show()
```

### 준비 자료
- **강의 자료**: Autoregressive Models (PixelRNN, PixelCNN) 슬라이드 (PDF)
- **참고 코드**: PixelCNN 구현 예제 코드 (Python)

### 과제
- **이론 정리**: Autoregressive Models의 원리와 응용 분야 정리.
- **코드 실습**: PixelCNN 구현 코드 실행 및 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획을 통해 학생들이 Autoregressive Models의 기본 개념과 원리를 이해하고, PixelCNN을 이용하여 실제 데이터를 생성하는 경험을 쌓을 수 있도록 유도합니다.