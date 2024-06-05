### 6주차 강의 상세 계획: 시맨틱 세그멘테이션

#### 강의 목표
- 시맨틱 세그멘테이션의 기본 개념과 주요 알고리즘 이해
- PyTorch를 이용한 U-Net 모델 구현 및 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 시맨틱 세그멘테이션의 기본 개념 (20분)

##### 시맨틱 세그멘테이션이란?
- **정의**: 시맨틱 세그멘테이션은 이미지의 각 픽셀을 특정 클래스에 할당하는 작업.
- **주요 작업**: 각 픽셀에 클래스 레이블을 할당하여 의미 있는 영역을 분할.

##### 시맨틱 세그멘테이션의 주요 응용 분야
- **의료**: 의료 영상에서 장기, 병변 등 구역화.
- **자동차**: 자율 주행 차량에서 도로, 보행자, 차량 인식.
- **위성 이미지 분석**: 농지, 도시, 자연 지형 분할.
- **AR/VR**: 증강 현실 및 가상 현실에서 객체 인식 및 환경 분석.

#### 1.2 주요 시맨틱 세그멘테이션 알고리즘 (40분)

##### Fully Convolutional Networks (FCN)
- **구조**: 전통적인 CNN을 변형하여 마지막 레이어를 컨볼루션 레이어로 교체.
- **특징**:
  - 입력 이미지의 크기와 동일한 크기의 출력 생성.
  - 중간 레이어의 정보를 합성하여 더 나은 세그멘테이션 결과 도출.

##### U-Net
- **구조**: U자형 네트워크로, 다운샘플링과 업샘플링 경로로 구성.
- **특징**:
  - 다운샘플링 경로에서 추출된 정보를 업샘플링 경로에 전달.
  - 원래 이미지와 동일한 크기의 세그멘테이션 맵 생성.
  - 주로 바이오메디컬 이미지 세그멘테이션에서 사용.

##### DeepLab
- **구조**: 다양한 스케일에서 특징을 추출하기 위해 아트리우스 컨볼루션(Atrous Convolution) 사용.
- **특징**:
  - 다양한 크기의 객체를 효과적으로 인식.
  - 공간적 해상도를 유지하면서도 계산 효율성 확보.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 PyTorch를 이용한 U-Net 시맨틱 세그멘테이션 모델 구현

##### 필요 라이브러리 설치
```bash
pip install torch torchvision matplotlib
```

##### U-Net 모델 구현 코드 (Python 3.10 및 PyTorch)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# U-Net 모델 정의
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x = self.decoder(x2)
        return x

# 데이터셋 정의 (예시)
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

# 하이퍼파라미터 설정
batch_size = 4
learning_rate = 0.001
num_epochs = 10

# 데이터셋 로드 및 전처리
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = SegmentationDataset(image_dir='./data/train_images', mask_dir='./data/train_masks', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 모델 초기화
model = UNet()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

# 학습
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, masks) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

# 샘플 이미지 시각화
def visualize_sample(image, mask, prediction):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image.squeeze(), cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(mask.squeeze(), cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[2].imshow(prediction.squeeze(), cmap='gray')
    axes[2].set_title('Prediction')
    plt.show()

# 예측 결과 시각화
model.eval()
with torch.no_grad():
    sample_image, sample_mask = train_dataset[0]
    sample_image = sample_image.unsqueeze(0)
    prediction = model(sample_image)
    visualize_sample(sample_image.squeeze(), sample_mask, prediction.squeeze())
```

### 준비 자료
- **강의 자료**: 시맨틱 세그멘테이션 (FCN, U-Net, DeepLab) 슬라이드 (PDF)
- **참고 코드**: U-Net 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 시맨틱 세그멘테이션의 기본 개념과 주요 알고리즘 정리.
- **코드 실습**: U-Net 모델 구현 및 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 시맨틱 세그멘테이션의 기본 개념과 주요 알고리즘을 이해하고, PyTorch를 이용하여 U-Net 모델을 구현하며, 실제 데이터를 통해 모델을 학습시키고 예측 결과를 분석하는 경험을 쌓을 수 있도록 유도합니다.