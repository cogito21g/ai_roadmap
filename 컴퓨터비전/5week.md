### 5주차 강의 상세 계획: 객체 탐지

#### 강의 목표
- 객체 탐지의 기본 개념과 주요 알고리즘 이해
- PyTorch를 이용한 YOLO 객체 탐지 모델 구현 및 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 객체 탐지의 기본 개념 (20분)

##### 객체 탐지란?
- **정의**: 객체 탐지는 이미지 또는 비디오에서 사물의 위치와 클래스를 식별하는 기술.
- **주요 작업**: 바운딩 박스(Bounding Box) 생성 및 클래스 레이블 할당.

##### 객체 탐지의 주요 응용 분야
- **자동차**: 자율 주행 차량에서 보행자, 차량, 도로 표지판 인식.
- **보안**: 감시 카메라에서 침입자 감지.
- **소매업**: 매장 내 상품 추적 및 관리.
- **의료**: 의료 영상에서 병변 탐지.

#### 1.2 주요 객체 탐지 알고리즘 (40분)

##### R-CNN (Region-Based Convolutional Neural Networks)
- **R-CNN**: 이미지에서 제안된 영역을 CNN을 통해 특징 추출 후 분류.
- **Fast R-CNN**: R-CNN의 속도 개선, ROI 풀링을 도입하여 단일 네트워크로 학습.
- **Faster R-CNN**: 제안된 영역 생성을 위한 RPN(Region Proposal Network) 추가.

##### YOLO (You Only Look Once)
- **YOLO**: 단일 신경망을 통해 이미지 전체를 한 번에 처리하여 객체 탐지.
- **구성 요소**:
  - **그리드 셀**: 이미지를 S x S 그리드로 분할.
  - **바운딩 박스**: 각 셀이 예측하는 바운딩 박스.
  - **신뢰도 점수**: 객체가 있을 확률과 바운딩 박스의 정확도.
- **특징**:
  - 실시간 객체 탐지 가능.
  - 단일 패스 방식으로 속도 향상.

##### SSD (Single Shot MultiBox Detector)
- **SSD**: 다중 스케일 특징 맵을 사용하여 객체 탐지.
- **구성 요소**:
  - **다중 스케일**: 다양한 크기의 객체를 탐지하기 위한 다중 스케일 예측.
  - **디폴트 박스**: 다양한 비율의 디폴트 박스를 사용하여 다양한 크기의 객체 탐지.
- **특징**:
  - 빠르고 정확한 객체 탐지.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 PyTorch를 이용한 YOLO 객체 탐지 모델 구현

##### 필요 라이브러리 설치
```bash
pip install torch torchvision matplotlib
```

##### YOLO 객체 탐지 코드 (Python 3.10 및 PyTorch)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# YOLO 모델 정의
class YOLO(nn.Module):
    def __init__(self, num_classes=20, grid_size=7, num_boxes=2):
        super(YOLO, self).__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_boxes = num_boxes

        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        
        self.fc = nn.Sequential(
            nn.Linear(512 * grid_size * grid_size, 4096),
            nn.ReLU(True),
            nn.Linear(4096, grid_size * grid_size * (num_classes + num_boxes * 5))
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, self.grid_size, self.grid_size, self.num_classes + self.num_boxes * 5)
        return x

# 학습 파라미터 설정
num_classes = 20
grid_size = 7
num_boxes = 2
batch_size = 16
learning_rate = 0.001
num_epochs = 10

# 데이터셋 로드 및 전처리
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.VOCDetection(root='./data', year='2012', image_set='train', download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 모델 초기화
model = YOLO(num_classes=num_classes, grid_size=grid_size, num_boxes=num_boxes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 학습
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs['image']
        targets = targets['annotation']

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

# 객체 탐지 결과 시각화
def plot_boxes(image, boxes):
    image = np.array(image)
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    for box in boxes:
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        plt.text(box[0], box[1], s=box[4], color='white', verticalalignment='top', bbox={'color': 'red', 'pad': 0})
    plt.show()

# 샘플 이미지 시각화
sample_image, _ = train_dataset[0]
sample_image = Image.fromarray((sample_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
model.eval()
with torch.no_grad():
    output = model(sample_image.unsqueeze(0))
    # Placeholder for detected boxes - this needs to be replaced with actual box extraction code
    detected_boxes = [[100, 100, 200, 200, 'class1'], [150, 150, 250, 250, 'class2']]
    plot_boxes(sample_image, detected_boxes)
```

### 준비 자료
- **강의 자료**: 객체 탐지 (R-CNN, YOLO, SSD) 슬라이드 (PDF)
- **참고 코드**: YOLO 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 객체 탐지의 기본 개념과 주요 알고리즘 정리.
- **코드 실습**: YOLO 객체 탐지 모델 구현 및 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 객체 탐지의 기본 개념과 주요 알고리즘을 이해하고, PyTorch를 이용하여 YOLO 객체 탐지 모델을 구현하며, 실제 데이터를 통해 모델을 학습시키는 경험을 쌓을 수 있도록 유도합니다.