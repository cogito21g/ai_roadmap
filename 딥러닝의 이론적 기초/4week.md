### 4주차 강의 상세 계획: 역전파 알고리즘

#### 강의 목표
- 역전파 알고리즘의 원리와 수학적 배경 이해
- 역전파 알고리즘을 사용하여 신경망을 학습시키는 방법 학습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 역전파 알고리즘의 원리 (20분)

##### 역전파 알고리즘이란?
- **정의**: 역전파(backpropagation)는 신경망의 가중치를 업데이트하기 위해 사용되는 알고리즘으로, 출력에서 입력으로 오차를 전파하며 가중치를 조정.
- **핵심 개념**:
  - **오차 역전파**: 출력 레이어에서 계산된 오차를 이전 레이어로 전파.
  - **체인 룰**: 미분의 체인 룰을 사용하여 각 가중치에 대한 오차 기울기 계산.
  - **가중치 업데이트**: 경사 하강법을 사용하여 가중치를 업데이트.

##### 역전파 알고리즘의 단계
1. **순전파(forward pass)**: 입력 데이터가 신경망을 통과하여 출력을 계산.
2. **오차 계산**: 출력과 실제 값의 차이를 계산하여 손실(loss)을 구함.
3. **오차 역전파(backward pass)**: 출력에서 입력으로 오차를 전파하며 각 가중치에 대한 기울기(gradient)를 계산.
4. **가중치 업데이트**: 경사 하강법을 사용하여 가중치를 업데이트.

#### 1.2 역전파 알고리즘의 수학적 배경 (40분)

##### 체인 룰
- **정의**: 복합 함수의 미분을 계산하기 위한 방법.
- **적용**: 역전파 알고리즘에서 각 레이어의 기울기를 계산할 때 사용.

##### 손실 함수의 기울기 계산
- **정의**: 손실 함수를 각 가중치에 대해 미분하여 기울기를 계산.
- **예제**: Mean Squared Error (MSE)의 기울기 계산.
  - \( L = \frac{1}{2} (y - \hat{y})^2 \)
  - \( \frac{\partial L}{\partial \hat{y}} = \hat{y} - y \)

##### 가중치 업데이트
- **경사 하강법**: 가중치를 기울기의 반대 방향으로 업데이트.
  - \( w = w - \eta \frac{\partial L}{\partial w} \)
  - 여기서 \( \eta \)는 학습률(learning rate).

##### 역전파 알고리즘의 수학적 단계
1. **순전파**: 각 레이어의 출력을 계산.
   - \( z_i = w_i \cdot a_{i-1} + b_i \)
   - \( a_i = \sigma(z_i) \)
2. **오차 계산**: 출력과 실제 값의 차이를 계산.
   - \( \delta_L = \frac{\partial L}{\partial a_L} \cdot \sigma'(z_L) \)
3. **오차 역전파**: 각 레이어의 기울기를 계산.
   - \( \delta_i = (\delta_{i+1} \cdot w_{i+1}) \cdot \sigma'(z_i) \)
4. **가중치 업데이트**: 기울기를 사용하여 가중치를 업데이트.
   - \( w_i = w_i - \eta \delta_i \cdot a_{i-1} \)
   - \( b_i = b_i - \eta \delta_i \)

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 역전파 알고리즘 실습

##### 필요 라이브러리 설치
```bash
pip install torch torchvision matplotlib
```

##### 역전파 알고리즘 구현 코드 (Python 3.10 및 PyTorch)
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
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 학습
total_step = len(train_loader)
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
- **강의 자료**: 역전파 알고리즘 슬라이드 (PDF)
- **참고 코드**: 역전파 알고리즘 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 역전파 알고리즘의 원리와 수학적 배경 정리.
- **코드 실습**: 역전파 알고리즘을 사용하여 신경망 학습 및 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 역전파 알고리즘의 원리와 수학적 배경을 이해하고, 이를 활용하여 신경망을 학습시키는 경험을 쌓을 수 있도록 유도합니다.