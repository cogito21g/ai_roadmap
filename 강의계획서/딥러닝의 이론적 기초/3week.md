### 3주차 강의 상세 계획: 활성화 함수와 손실 함수

#### 강의 목표
- 활성화 함수의 종류와 역할 이해
- 손실 함수의 종류와 역할 이해
- 활성화 함수와 손실 함수를 활용한 신경망 학습 경험

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 활성화 함수 (30분)

##### 활성화 함수의 역할
- **비선형성 도입**: 신경망에 비선형성을 도입하여 복잡한 패턴 학습 가능.
- **출력 값 스케일링**: 뉴런의 출력 값을 특정 범위로 스케일링.

##### 주요 활성화 함수
- **Sigmoid**: 출력 값을 0과 1 사이로 압축.
  - \( \sigma(x) = \frac{1}{1 + e^{-x}} \)
  - **장점**: 출력이 항상 0과 1 사이에 존재, 확률 해석 가능.
  - **단점**: 기울기 소실 문제 (Gradient Vanishing).

- **Tanh**: 출력 값을 -1과 1 사이로 압축.
  - \( \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \)
  - **장점**: 출력이 -1과 1 사이에 존재, Sigmoid보다 중심에 가깝게 수렴.
  - **단점**: 기울기 소실 문제.

- **ReLU (Rectified Linear Unit)**: 음수를 0으로 변환.
  - \( \text{ReLU}(x) = \max(0, x) \)
  - **장점**: 계산이 간단하고 빠르며 기울기 소실 문제 해결.
  - **단점**: 음수 입력에서 뉴런이 죽는 문제 (Dead Neurons).

- **Leaky ReLU**: 음수를 작은 기울기로 변환.
  - \( \text{Leaky ReLU}(x) = \max(0.01x, x) \)
  - **장점**: ReLU의 단점인 죽은 뉴런 문제를 해결.

#### 1.2 손실 함수 (30분)

##### 손실 함수의 역할
- **오차 측정**: 예측 값과 실제 값의 차이를 측정.
- **모델 최적화**: 손실 값을 최소화하여 모델 성능 향상.

##### 주요 손실 함수
- **MSE (Mean Squared Error)**: 회귀 문제에서 주로 사용.
  - \( \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \)
  - **장점**: 큰 오차에 대해 더 큰 패널티를 부여.
  - **단점**: 이상치에 민감.

- **Cross-Entropy Loss**: 분류 문제에서 주로 사용.
  - \( \text{Cross-Entropy Loss} = -\sum_{i} y_i \log(\hat{y}_i) \)
  - **장점**: 확률 분포 간의 차이를 측정하여 분류 성능 향상.
  - **단점**: 출력 값이 확률 분포를 따르지 않으면 성능 저하.

- **Hinge Loss**: SVM에서 주로 사용.
  - \( \text{Hinge Loss} = \max(0, 1 - y_i \hat{y}_i) \)
  - **장점**: 마진을 최대화하여 분류 성능 향상.
  - **단점**: 확률 해석이 어렵고, 이상치에 민감.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 활성화 함수 실습

##### 필요 라이브러리 설치
```bash
pip install torch torchvision matplotlib
```

##### 활성화 함수 구현 코드 (Python 3.10 및 PyTorch)
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

# 활성화 함수 비교 모델 정의
class ActivationComparisonNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation_fn):
        super(ActivationComparisonNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = activation_fn
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out

# 활성화 함수 실험
activations = {
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'ReLU': nn.ReLU(),
    'Leaky ReLU': nn.LeakyReLU()
}

results = {}

for name, activation in activations.items():
    model = ActivationComparisonNN(input_size, hidden_size, num_classes, activation)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
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

    accuracy = 100 * correct / total
    results[name] = accuracy
    print(f'Activation: {name}, Test Accuracy: {accuracy:.2f}%')

# 결과 시각화
plt.bar(results.keys(), results.values())
plt.xlabel('Activation Function')
plt.ylabel('Test Accuracy')
plt.title('Activation Function Comparison on MNIST')
plt.show()
```

### 준비 자료
- **강의 자료**: 활성화 함수 및 손실 함수 슬라이드 (PDF)
- **참고 코드**: 활성화 함수 비교 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 주요 활성화 함수와 손실 함수의 역할과 특징 정리.
- **코드 실습**: 활성화 함수와 손실 함수를 변경하면서 신경망 학습 및 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 활성화 함수와 손실 함수의 역할과 특징을 이해하고, 이를 활용하여 신경망을 학습시키는 경험을 쌓을 수 있도록 유도합니다.