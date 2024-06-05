### 9주차 강의 상세 계획: 최적화 기법

#### 강의 목표
- 신경망 학습을 위한 다양한 최적화 기법 이해
- 주요 최적화 알고리즘의 원리와 적용 방법 학습
- 최적화 기법을 사용하여 신경망 모델 성능 향상 경험

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 최적화 기법 개요 (15분)

##### 최적화의 정의
- **정의**: 손실 함수를 최소화하기 위해 모델의 파라미터를 조정하는 과정.
- **목표**: 학습 과정에서 손실을 최소화하여 모델의 성능을 최적화.

##### 최적화 기법의 역할
- **학습 속도 향상**: 효율적인 최적화 알고리즘을 통해 빠른 수렴.
- **안정적인 학습**: 기울기 폭발이나 기울기 소실 문제 해결.
- **일반화 성능 향상**: 과적합 방지 및 모델의 일반화 성능 개선.

#### 1.2 주요 최적화 기법 (45분)

##### 확률적 경사 하강법 (Stochastic Gradient Descent, SGD)
- **기본 원리**: 전체 데이터셋 대신 미니배치를 사용하여 기울기를 계산하고 파라미터 업데이트.
- **수식**: \( \theta = \theta - \eta \nabla L(\theta) \)
- **장점**: 계산 비용 감소, 빠른 수렴 가능.
- **단점**: 진동 및 불안정성 문제.

##### 모멘텀 (Momentum)
- **기본 원리**: 이전 기울기의 누적된 값을 사용하여 관성을 추가.
- **수식**: \( v_t = \beta v_{t-1} + (1 - \beta) \nabla L(\theta) \), \( \theta = \theta - \eta v_t \)
- **장점**: 진동 감소, 수렴 속도 향상.

##### RMSprop (Root Mean Square Propagation)
- **기본 원리**: 기울기의 제곱 평균을 사용하여 학습률 조정.
- **수식**: \( E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma)g_t^2 \), \( \theta = \theta - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t \)
- **장점**: 학습률의 자동 조정, 안정적인 학습.

##### Adam (Adaptive Moment Estimation)
- **기본 원리**: 모멘텀과 RMSprop을 결합하여 학습률을 적응적으로 조정.
- **수식**: \( m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \), \( v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \), \( \hat{m}_t = \frac{m_t}{1 - \beta_1^t} \), \( \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \), \( \theta = \theta - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t \)
- **장점**: 빠른 수렴, 효율적인 학습률 조정.

##### AdaGrad (Adaptive Gradient Algorithm)
- **기본 원리**: 학습률을 각 파라미터에 대해 개별적으로 조정.
- **수식**: \( G_t = \sum_{i=1}^t g_i^2 \), \( \theta = \theta - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t \)
- **장점**: 드문 특징의 학습률 증가, 계산의 간단함.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 다양한 최적화 기법 실습

##### 필요 라이브러리 설치
```bash
pip install torch torchvision matplotlib
```

##### 다양한 최적화 기법 구현 코드 (Python 3.10 및 PyTorch)
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
- **강의 자료**: 최적화 기법 슬라이드 (PDF)
- **참고 코드**: 다양한 최적화 기법 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 주요 최적화 기법의 원리와 특징 정리.
- **코드 실습**: 다양한 최적화 기법을 사용하여 신경망 학습 및 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 다양한 최적화 기법의 원리와 특징을 이해하고, 이를 활용하여 신경망 모델의 성능을 향상시키는 경험을 쌓을 수 있도록 유도합니다.