### 8주차 강의 상세 계획: 순환 신경망 (RNN)

#### 강의 목표
- 순환 신경망(Recurrent Neural Network, RNN)의 구조와 원리 이해
- RNN의 주요 구성 요소 및 문제점 해결 방법 학습
- RNN을 사용하여 시계열 데이터 처리 모델 구현 경험

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 순환 신경망의 기본 구조 (20분)

##### RNN이란?
- **정의**: 시계열 데이터나 순차적인 데이터의 패턴을 학습하기 위해 설계된 신경망.
- **구조**: 입력층, 은닉층, 출력층으로 구성되며, 은닉층의 출력이 다음 시점의 입력으로 사용.

##### 주요 구성 요소
- **순환 연결**: 이전 시점의 은닉층 출력을 현재 시점의 입력으로 사용.
- **은닉 상태(Hidden State)**: 시퀀스 데이터의 정보가 축적되는 내부 상태.
- **출력**: 각 시점에서 은닉층의 출력을 사용하여 최종 출력 계산.

#### 1.2 RNN의 학습 과정 (20분)

##### 학습 단계
1. **순전파(forward propagation)**: 입력 데이터를 시퀀스의 각 시점에서 처리.
2. **손실 함수 계산**: 출력 값과 실제 값의 차이를 계산하여 손실(loss)을 구함.
3. **역전파(backward propagation through time, BPTT)**: 시간에 따라 펼쳐진 네트워크에서 각 시점의 기울기를 계산하여 가중치 업데이트.

##### RNN의 문제점과 해결 방법
- **기울기 소실 문제 (Gradient Vanishing)**: 시간이 지남에 따라 기울기가 점점 작아져 학습이 어려워짐.
  - **해결 방법**: LSTM(Long Short-Term Memory), GRU(Gated Recurrent Unit)와 같은 개선된 아키텍처 사용.
- **기울기 폭발 문제 (Gradient Exploding)**: 시간이 지남에 따라 기울기가 너무 커져 학습이 불안정해짐.
  - **해결 방법**: 기울기 클리핑(Gradient Clipping) 사용.

#### 1.3 LSTM과 GRU (20분)

##### LSTM(Long Short-Term Memory)
- **구조**: 입력 게이트, 망각 게이트, 출력 게이트를 통해 장기 의존성 문제 해결.
- **특징**: 셀 상태(Cell State)를 통해 중요한 정보를 장기간 유지.
- **장점**: 장기 의존성 문제 해결, 기울기 소실 문제 완화.

##### GRU(Gated Recurrent Unit)
- **구조**: 업데이트 게이트와 리셋 게이트를 통해 단순화된 구조.
- **특징**: LSTM보다 간단한 구조로 계산 효율성 향상.
- **장점**: 장기 의존성 문제 해결, 계산 효율성 향상.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 순환 신경망 실습

##### 필요 라이브러리 설치
```bash
pip install torch torchvision matplotlib
```

##### 순환 신경망 구현 코드 (Python 3.10 및 PyTorch)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 하이퍼파라미터 설정
input_size = 28  # 각 시점에서 입력 크기 (MNIST 이미지의 가로 길이)
hidden_size = 128
num_layers = 2
num_classes = 10
num_epochs = 2
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

# RNN 모델 정의
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28, 28).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        labels = labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
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
        images = images.reshape(-1, 28, 28).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        labels = labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 샘플 결과 시각화
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

with torch.no_grad():
    output = model(example_data.reshape(-1, 28, 28).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

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
- **강의 자료**: 순환 신경망 슬라이드 (PDF)
- **참고 코드**: RNN 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 순환 신경망의 구조와 학습 과정, 주요 구성 요소 정리.
- **코드 실습**: 순환 신경망 모델 구현 및 학습, 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 순환 신경망의 구조와 학습 방법을 이해하고, 이를 활용하여 RNN을 학습시키는 경험을 쌓을 수 있도록 유도합니다.