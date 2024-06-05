### 2주차 강의 상세 계획: 신경망의 기초

#### 강의 목표
- 인공신경망의 기본 구조와 원리 이해
- 간단한 신경망 모델 구현 및 학습 경험

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 인공신경망의 기본 구조 (20분)
- **뉴런**: 입력 값과 가중치의 곱을 활성화 함수를 통해 출력.
- **레이어**: 입력층, 은닉층, 출력층.
- **전방 전달 (Forward Propagation)**: 입력 데이터를 각 레이어를 거쳐 출력으로 전달.

#### 1.2 활성화 함수 (20분)
- **Sigmoid**: 출력 값을 0과 1 사이로 압축.
- **ReLU (Rectified Linear Unit)**: 음수를 0으로 변환하여 비선형성 도입.
- **Tanh**: 출력 값을 -1과 1 사이로 압축.

#### 1.3 손실 함수 (20분)
- **MSE (Mean Squared Error)**: 예측 값과 실제 값의 차이의 제곱 평균.
- **Cross-Entropy Loss**: 분류 문제에서 사용되는 손실 함수로, 예측 값과 실제 값의 확률 분포 차이 계산.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 간단한 신경망 모델 구현

##### 필요 라이브러리 설치
```bash
pip install torch torchvision matplotlib
```

##### 신경망 모델 구현 코드 (Python 3.10 및 PyTorch)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 하이퍼파라미터 설정
input_size = 784  # 28x28 이미지 크기
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST 데이터셋 로드 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 간단한 신경망 모델 정의
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = SimpleNN(input_size, hidden_size, num_classes)

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28)
        
        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 역전파 및 최적화
        optimizer

.zero_grad()
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
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 샘플 결과 시각화
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

with torch.no_grad():
    output = model(example_data.reshape(-1, 28*28))

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
- **강의 자료**: 신경망의 기초 슬라이드 (PDF)
- **참고 코드**: 간단한 신경망 모델 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 신경망의 기본 구조와 원리, 주요 활성화 함수와 손실 함수 정리.
- **코드 실습**: 간단한 신경망 모델 구현 및 MNIST 데이터셋 적용 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 인공신경망의 기본 개념을 이해하고, 간단한 신경망 모델을 구현하여 학습시키는 경험을 쌓을 수 있도록 유도합니다.