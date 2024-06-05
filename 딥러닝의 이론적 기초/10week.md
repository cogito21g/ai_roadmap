### 10주차 강의 상세 계획: 딥러닝의 설명 가능성

#### 강의 목표
- 딥러닝 모델의 설명 가능성 (Explainability) 이해
- 주요 설명 기법 및 적용 사례 학습
- 딥러닝 모델의 설명 가능성을 높이는 기법 구현 경험

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 딥러닝의 설명 가능성 개요 (20분)

##### 설명 가능성의 중요성
- **정의**: 모델이 내리는 예측의 이유를 이해하고 설명할 수 있는 능력.
- **필요성**: 신뢰성 확보, 디버깅 용이, 법적 요구사항 준수, 도메인 전문가와의 협업.

##### 주요 용어
- **모델 해석**: 모델의 동작을 이해하는 것.
- **설명 가능 인공지능 (Explainable AI, XAI)**: 모델의 투명성과 이해 가능성을 높이기 위한 기법.

#### 1.2 주요 설명 기법 (40분)

##### 모델 불가지론적 기법 (Model-Agnostic Methods)
- **LIME (Local Interpretable Model-agnostic Explanations)**:
  - **원리**: 모델의 출력과 입력 간의 관계를 선형 모델로 근사하여 설명.
  - **적용**: 특정 예측에 대한 로컬 설명 제공.
- **SHAP (SHapley Additive exPlanations)**:
  - **원리**: 게임 이론의 샤플리 값을 사용하여 각 특징이 예측에 기여하는 정도를 계산.
  - **적용**: 전체 모델의 글로벌 설명 제공.

##### 모델 고유 기법 (Model-Specific Methods)
- **특징 시각화 (Feature Visualization)**:
  - **원리**: 신경망의 특정 뉴런이 활성화되는 입력을 시각화하여 설명.
  - **적용**: CNN의 필터, RNN의 상태 등을 시각화.
- **층별 중요도 전파 (Layer-wise Relevance Propagation, LRP)**:
  - **원리**: 예측 결과를 입력 특징에 따라 역전파하여 설명.
  - **적용**: 모델의 각 층에서 중요도를 전파하여 특정 예측에 대한 글로벌 설명 제공.

##### 설명 기법의 비교
- **LIME vs SHAP**:
  - **LIME**: 로컬 설명에 적합, 계산이 상대적으로 빠름.
  - **SHAP**: 글로벌 설명에 적합, 계산이 복잡하지만 더 정교함.
- **특징 시각화 vs LRP**:
  - **특징 시각화**: 모델의 내부 동작을 이해하는 데 유용.
  - **LRP**: 예측 결과를 설명하는 데 유용.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 설명 기법 실습

##### 필요 라이브러리 설치
```bash
pip install torch torchvision matplotlib lime shap
```

##### LIME을 이용한 설명 기법 구현 코드 (Python 3.10 및 PyTorch)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries

# 하이퍼파라미터 설정
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
model.eval()

# LIME 설명 기법 적용 함수
def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    return transf

def batch_predict(images):
    model.eval()
    images = torch.stack([get_pil_transform()(i) for i in images], dim=0)
    images = images.reshape(-1, 1, 28, 28)
    outputs = model(images)
    return outputs.detach().numpy()

explainer = lime_image.LimeImageExplainer()

# 테스트 이미지 선택 및 설명 생성
test_images, test_labels = next(iter(test_loader))
idx = 0
image = test_images[idx].numpy().transpose(1, 2, 0)
explanation = explainer.explain_instance(image, batch_predict, top_labels=1, hide_color=0, num_samples=1000)

# 설명 결과 시각화
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.title(f'Prediction: {explanation.top_labels[0]}')
plt.show()
```

##### SHAP을 이용한 설명 기법 구현 코드
```python
import shap

# SHAP 설명 기법 적용
background = train_dataset.data[:100].reshape(100, 1, 28, 28).float()
test_images = test_dataset.data[:10].reshape(10, 1, 28, 28).float()

e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(test_images)

# 설명 결과 시각화
shap.image_plot(shap_values, test_images.numpy())
```

### 준비 자료
- **강의 자료**: 딥러닝의 설명 가능성 슬라이드 (PDF)
- **참고 코드**: LIME 및 SHAP 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 주요 설명 기법의 원리와 특징 정리.
- **코드 실습**: LIME 및 SHAP을 사용하여 신경망 모델의 설명 가능성 높이기.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 딥러닝 모델의 설명 가능성의 중요성을 이해하고, 이를 높이는 다양한 기법을 학습하고 실습할 수 있도록 유도합니다.