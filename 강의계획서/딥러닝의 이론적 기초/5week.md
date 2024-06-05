### 5주차 강의 상세 계획: 신경망 학습 방법

#### 강의 목표
- 신경망 학습 방법 이해
- 경사 하강법, 학습률 조정, 과적합 방지 기법 학습
- 다양한 학습 방법을 사용한 신경망 학습 경험

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 경사 하강법 (30분)

##### 경사 하강법의 기본 개념
- **정의**: 경사 하강법(Gradient Descent)은 손실 함수를 최소화하기 위해 기울기의 반대 방향으로 파라미터를 업데이트하는 최적화 알고리즘.
- **기본 원리**:
  - 손실 함수 \( L(\theta) \)와 파라미터 \( \theta \)가 주어졌을 때, 기울기 \( \nabla L(\theta) \)를 계산하고, 이를 이용하여 파라미터를 업데이트.
  - 업데이트 공식: \( \theta = \theta - \eta \nabla L(\theta) \)
  - 여기서 \( \eta \)는 학습률(learning rate).

##### 경사 하강법의 변형
- **배치 경사 하강법**: 전체 데이터셋을 사용하여 기울기를 계산.
- **확률적 경사 하강법 (SGD)**: 한 개의 샘플을 사용하여 기울기를 계산.
- **미니배치 경사 하강법**: 소규모 배치를 사용하여 기울기를 계산.

#### 1.2 학습률 조정 (15분)

##### 학습률의 중요성
- **학습률이 너무 크면**: 최적점 주변에서 진동하며 수렴하지 않을 수 있음.
- **학습률이 너무 작으면**: 수렴 속도가 매우 느려질 수 있음.

##### 학습률 조정 기법
- **고정 학습률**: 학습률을 일정하게 유지.
- **적응형 학습률**: 학습률을 점진적으로 감소시키거나 증가시킴.
  - **Learning Rate Decay**: 학습이 진행됨에 따라 학습률을 감소.
  - **Adaptive Learning Rates**: Adagrad, RMSprop, Adam 등.

#### 1.3 과적합 방지 기법 (15분)

##### 과적합의 개념
- **정의**: 모델이 학습 데이터에 과도하게 적응하여 새로운 데이터에 대한 일반화 성능이 떨어지는 현상.

##### 과적합 방지 기법
- **정규화 (Regularization)**:
  - **L2 정규화**: 가중치 제곱합을 손실 함수에 추가.
  - **L1 정규화**: 가중치 절대값 합을 손실 함수에 추가.
- **드롭아웃 (Dropout)**: 학습 중 무작위로 뉴런을 제거하여 과적합 방지.
- **조기 종료 (Early Stopping)**: 검증 손실이 증가하기 시작하면 학습을 중단.
- **데이터 증강 (Data Augmentation)**: 학습 데이터에 다양한 변형을 적용하여 데이터셋 크기 증가.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 다양한 학습 방법 실습

##### 필요 라이브러리 설치
```bash
pip install torch torchvision matplotlib
```

##### 다양한 학습 방법 구현 코드 (Python 3.10 및 PyTorch)
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

# 신경망 모델 정의
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

# 모델 초기화
model = SimpleNN(input_size, hidden_size, num_classes)

# 손실 함수와 다양한 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizers = {
    'SGD': optim.SGD(model.parameters(), lr=learning_rate),
    'Momentum': optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9),
    'RMSprop': optim.RMSprop(model.parameters(), lr=learning_rate),
    'Adam': optim.Adam(model.parameters(), lr=learning_rate)
}

# 학습 및 테스트 함수
def train_and_test(optimizer_name):
    optimizer = optimizers[optimizer_name]
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28*28)
            
            # 순전파
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
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

    accuracy = 100 * correct / total
    print(f'Optimizer: {optimizer_name}, Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 다양한 옵티마이저를 사용한 학습 및 결과 비교
results = {}
for optimizer_name in optimizers.keys():
    accuracy = train_and_test(optimizer_name)
    results[optimizer_name] = accuracy

# 결과 시각화
plt.bar(results.keys(), results.values())
plt.xlabel('Optimizer')
plt.ylabel('Test Accuracy')
plt.title('Optimizer Comparison on MNIST')
plt.show()
```

### 준비 자료
- **강의 자료**: 신경망 학습 방법 슬라이드 (PDF)
- **참고 코드**: 다양한 학습 방법 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 경사 하강법, 학습률 조정, 과적합 방지 기법의 원리와 특징 정리.
- **코드 실습**: 다양한 학습 방법을 사용하여 신경망 학습 및 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 신경망 학습 방법의 다양한 기법을 이해하고, 이를 활용하여 신경망을 학습시키는 경험을 쌓을 수 있도록 유도합니다.