### 3주차 강의 상세 계획: 기초 딥러닝

#### 강의 목표
- 인공신경망(ANN)과 컨볼루션 신경망(CNN)의 기본 구조와 원리 이해
- PyTorch를 이용한 간단한 CNN 구현 및 MNIST 데이터셋 적용

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 인공신경망(ANN) 기본 개념 (20분)

##### 신경망의 기본 구성 요소
- **뉴런**: 입력 값을 받아 가중치와 곱한 후 활성화 함수를 통해 출력.
- **레이어**: 입력층, 은닉층, 출력층으로 구성.
- **활성화 함수**: Sigmoid, ReLU, Tanh 등.

##### 신경망의 학습 과정
- **순전파**: 입력 데이터를 뉴런을 통해 전달하여 최종 출력 계산.
- **손실 함수**: 예측 값과 실제 값의 차이를 계산 (예: MSE, Cross-Entropy).
- **역전파**: 손실을 최소화하기 위해 가중치를 업데이트.

#### 1.2 컨볼루션 신경망(CNN) 기본 개념 (20분)

##### CNN의 주요 구성 요소
- **컨볼루션 레이어**: 필터를 사용하여 입력 데이터의 특징을 추출.
- **풀링 레이어**: 특징 맵을 다운샘플링하여 계산량 감소.
- **완전 연결층**: 추출된 특징을 바탕으로 최종 분류 수행.

##### CNN의 학습 과정
- **특징 추출**: 컨볼루션과 풀링을 통해 입력 데이터의 특징 추출.
- **분류**: 추출된 특징을 바탕으로 최종 분류 수행.

#### 1.3 CNN의 응용 (20분)

##### CNN의 주요 응용 분야
- **이미지 분류**: 입력 이미지를 미리 정의된 클래스 중 하나로 분류.
- **객체 탐지**: 이미지 내 객체의 위치와 클래스 식별.
- **시맨틱 세그멘테이션**: 이미지의 각 픽셀을 클래스별로 분류.

##### 사례 연구
- **LeNet**: 최초의 CNN 중 하나로, 손글씨 숫자 인식을 위해 설계.
- **AlexNet**: 2012년 ImageNet 대회에서 우승한 모델로, 딥러닝의 발전을 이끈 모델.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 PyTorch를 이용한 간단한 CNN 구현 및 MNIST 데이터셋 적용

##### 필요 라이브러리 설치
```bash
pip install torch torchvision matplotlib
```

##### CNN 구현 코드 (Python 3.10 및 PyTorch)
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

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 초기화
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 학습
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

# 테스트
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 샘플 이미지 시각화
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
- **강의 자료**: 기초 딥러닝 (ANN, CNN) 슬라이드 (PDF)
- **참고 코드**: 간단한 CNN 구현 예제 코드 (Python)

### 과제
- **이론 정리**: ANN과 CNN의 구조와 원리 정리.
- **코드 실습**: 간단한 CNN 구현 코드 실행 및 MNIST 데이터셋 적용 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 인공신경망과 컨볼루션 신경망의 기본 개념을 이해하고, PyTorch를 이용하여 간단한 CNN을 구현하며, MNIST 데이터셋을 통해 실제로 모델을 학습시키는 경험을 쌓을 수 있도록 유도합니다.