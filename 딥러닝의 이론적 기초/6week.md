### 6주차 강의 상세 계획: 심층 신경망

#### 강의 목표
- 심층 신경망(Deep Neural Networks, DNN)의 구조와 학습 이해
- 심층 신경망의 학습 과정과 문제점 해결 방법 학습
- 심층 신경망을 사용하여 신경망 학습 경험

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 심층 신경망의 기본 구조 (20분)

##### 심층 신경망이란?
- **정의**: 여러 개의 은닉층을 가지는 인공신경망.
- **구조**: 입력층, 다수의 은닉층, 출력층으로 구성.
- **특징**: 더 많은 은닉층을 통해 복잡한 데이터의 패턴 학습 가능.

##### 주요 구성 요소
- **뉴런**: 각 뉴런은 가중치와 활성화 함수를 통해 입력 값을 처리.
- **레이어**: 입력 데이터가 여러 층을 통과하며 처리됨.
- **활성화 함수**: 각 층의 뉴런에서 비선형성을 도입.

#### 1.2 심층 신경망의 학습 (20분)

##### 학습 과정
- **순전파(forward propagation)**: 입력 데이터가 신경망을 통과하며 출력 값 계산.
- **손실 함수 계산**: 출력 값과 실제 값의 차이를 계산하여 손실(loss)을 구함.
- **역전파(backward propagation)**: 손실을 기반으로 각 가중치에 대한 기울기 계산.
- **가중치 업데이트**: 경사 하강법을 사용하여 가중치 조정.

##### 학습의 문제점과 해결 방법
- **기울기 소실 문제 (Gradient Vanishing)**: 기울기가 층을 거치며 점점 작아져 학습이 어려워짐.
  - **해결 방법**: ReLU와 같은 활성화 함수 사용, 가중치 초기화 방법 개선, 배치 정규화 사용.
- **과적합 (Overfitting)**: 모델이 학습 데이터에 과도하게 적응하여 새로운 데이터에 대한 일반화 성능 저하.
  - **해결 방법**: 정규화, 드롭아웃, 데이터 증강, 조기 종료.
- **학습 속도 문제**: 심층 신경망은 학습 시간이 오래 걸림.
  - **해결 방법**: 적응형 학습률 방법 사용, GPU를 통한 병렬 처리.

#### 1.3 심층 신경망의 응용 (20분)

##### 주요 응용 분야
- **컴퓨터 비전**: 이미지 분류, 객체 탐지, 시맨틱 세그멘테이션.
- **자연어 처리**: 번역, 텍스트 생성, 감정 분석.
- **의료**: 진단 보조, 의료 영상 분석.
- **자율 주행**: 객체 인식, 경로 계획.

##### 사례 연구
- **ImageNet 대회**: 심층 신경망을 활용한 이미지 분류 성능 개선.
- **자율 주행 자동차**: 심층 신경망을 활용한 실시간 객체 인식 및 경로 계획.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 심층 신경망 실습

##### 필요 라이브러리 설치
```bash
pip install torch torchvision matplotlib
```

##### 심층 신경망 구현 코드 (Python 3.10 및 PyTorch)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 하이퍼파라미터 설정
input_size = 784  # 28x28 이미지 크기
hidden_sizes = [512, 256, 128]
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

# 심층 신경망 모델 정의
class DeepNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_sizes[2], num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out

model = DeepNN(input_size, hidden_sizes, num_classes)

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
- **강의 자료**: 심층 신경망 슬라이드 (PDF)
- **참고 코드**: 심층 신경망 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 심층 신경망의 구조와 학습 과정, 문제점 해결 방법 정리.
- **코드 실습**: 심층 신경망 모델 구현 및 학습, 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 심층 신경망의 구조와 학습 방법을 이해하고, 이를 활용하여 심층 신경망을 학습시키는 경험을 쌓을 수 있도록 유도합니다.