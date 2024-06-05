### 7주차 강의 상세 계획: 합성곱 신경망 (CNN)

#### 강의 목표
- 합성곱 신경망(Convolutional Neural Network, CNN)의 구조와 원리 이해
- CNN의 주요 구성 요소 및 적용 사례 학습
- CNN을 사용하여 이미지 분류 모델 구현 경험

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 합성곱 신경망의 기본 구조 (20분)

##### CNN이란?
- **정의**: 합성곱 연산을 통해 입력 이미지의 특징을 추출하는 신경망.
- **구조**: 입력층, 합성곱층, 풀링층, 완전 연결층으로 구성.

##### 주요 구성 요소
- **합성곱층(Convolutional Layer)**: 필터를 사용하여 이미지의 특징 맵을 생성.
  - **필터(Filter)**: 학습 가능한 파라미터로, 이미지의 특정 패턴을 인식.
  - **활성화 함수**: ReLU와 같은 활성화 함수를 사용하여 비선형성 도입.
- **풀링층(Pooling Layer)**: 특징 맵의 크기를 줄이고 중요한 특징을 추출.
  - **최대 풀링(Max Pooling)**: 필터 내의 최대 값을 선택하여 다운샘플링.
  - **평균 풀링(Average Pooling)**: 필터 내의 평균 값을 선택하여 다운샘플링.
- **완전 연결층(Fully Connected Layer)**: 추출된 특징을 기반으로 분류를 수행.

#### 1.2 CNN의 학습 과정 (20분)

##### CNN의 학습 단계
1. **입력 이미지 처리**: 입력 이미지를 여러 개의 필터를 통해 합성곱 연산 수행.
2. **특징 맵 생성**: 각 필터를 통해 추출된 특징 맵 생성.
3. **풀링 연산**: 특징 맵의 크기를 줄이고 중요한 특징을 추출.
4. **완전 연결층**: 추출된 특징을 기반으로 최종 분류 수행.

##### CNN의 장점
- **공간 불변성(Spatial Invariance)**: 이미지의 위치 변화에 대한 강인성.
- **파라미터 공유(Parameter Sharing)**: 필터를 공유하여 학습 파라미터 수 감소.
- **지역 수용장(Receptive Field)**: 국소적인 특징을 학습하여 효율적인 패턴 인식.

#### 1.3 CNN의 응용 (20분)

##### 주요 응용 분야
- **이미지 분류**: 입력 이미지를 사전 정의된 클래스 중 하나로 분류.
  - **예시**: ImageNet 대회에서의 우수한 성능.
- **객체 탐지**: 이미지 내 객체의 위치와 클래스를 식별.
  - **예시**: YOLO, Faster R-CNN.
- **시맨틱 세그멘테이션**: 이미지의 각 픽셀을 클래스별로 분류.
  - **예시**: FCN, U-Net.

##### 사례 연구
- **LeNet**: 최초의 합성곱 신경망 중 하나로, 손글씨 숫자 인식(MNIST)에서 사용.
- **AlexNet**: ImageNet 대회에서 우승하며 딥러닝의 발전을 이끈 모델.
- **VGGNet**: 더 깊은 네트워크 구조를 통해 성능 향상.
- **ResNet**: 잔차 연결을 통해 매우 깊은 네트워크 학습 가능.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 합성곱 신경망 실습

##### 필요 라이브러리 설치
```bash
pip install torch torchvision matplotlib
```

##### 합성곱 신경망 구현 코드 (Python 3.10 및 PyTorch)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 하이퍼파라미터 설정
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# 데이터셋 로드 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(7*7*64, 1000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, 10)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = SimpleCNN()

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# 테스트
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 샘플 결과 시각화
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

with torch.no_grad():
    output = model(example_data)

fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title(f'Prediction: {output.data.max(1, keepdim=True)[1][i].item()}')
    plt.xticks([])
    plt.yticks([])
plt.show()
```

### 준비 자료
- **강의 자료**: 합성곱 신경망 슬라이드 (PDF)
- **참고 코드**: CNN 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 합성곱 신경망의 구조와 학습 과정, 주요 구성 요소 정리.
- **코드 실습**: 합성곱 신경망 모델 구현 및 학습, 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 합성곱 신경망의 구조와 학습 방법을 이해하고, 이를 활용하여 CNN을 학습시키는 경험을 쌓을 수 있도록 유도합니다.