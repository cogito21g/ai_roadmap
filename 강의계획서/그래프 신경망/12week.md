### 12주차 강의 상세 계획: GNN 실습 프로젝트 2

#### 강의 목표
- 모델 학습 및 하이퍼파라미터 튜닝
- GNN 실습 프로젝트의 모델 학습 과정 이해

#### 강의 구성
- **모델 학습**: 1시간
- **하이퍼파라미터 튜닝**: 1시간

---

### 1. 모델 학습 (1시간)

#### 1.1 GNN 모델 학습

##### 필요 라이브러리 설치
```bash
pip install torch torch-geometric
```

##### GNN 모델 학습 코드 (Python)
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader

# GNN 모델 정의
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 데이터셋 로드
train_data_list = [train_data]  # 여기에 DataLoader를 사용할 수 있습니다.
test_data_list = [test_data]
train_loader = DataLoader(train_data_list, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=1, shuffle=False)

# 모델 학습 및 평가
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(in_channels=1000, hidden_channels=64, out_channels=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}')

model.eval()
correct = 0
for data in test_loader:
    data = data.to(device)
    out = model(data)
    _, pred = out.max(dim=1)
    correct += int((pred == data.y).sum().item())

accuracy = correct / len(test_loader.dataset)
print(f'Test Accuracy: {accuracy:.4f}')
```

---

### 2. 하이퍼파라미터 튜닝 (1시간)

#### 2.1 하이퍼파라미터 튜닝의 중요성

##### 하이퍼파라미터 튜닝이란?
- **정의**: 모델의 성능을 최적화하기 위해 하이퍼파라미터 값을 조정하는 과정.
- **중요성**: 적절한 하이퍼파라미터 설정은 모델의 성능에 큰 영향을 미침.

#### 2.2 하이퍼파라미터 튜닝 방법

##### 그리드 서치 (Grid Search)
- **정의**: 모든 가능한 하이퍼파라미터 조합을 시도하여 최적의 조합을 찾는 방법.
- **장점**: 모든 조합을 시도하므로 최적 해를 찾을 가능성이 높음.
- **단점**: 계산 비용이 매우 높음.

##### 랜덤 서치 (Random Search)
- **정의**: 하이퍼파라미터 공간에서 무작위로 조합을 선택하여 최적의 조합을 찾는 방법.
- **장점**: 계산 비용이 그리드 서치보다 낮음, 다양한 조합 시도 가능.
- **단점**: 최적 해를 놓칠 가능성이 있음.

#### 2.3 하이퍼파라미터 튜닝 실습

##### 필요 라이브러리 설치
```bash
pip install scikit-learn
```

##### 하이퍼파라미터 튜닝 코드 (Python)
```python
from sklearn.model_selection import ParameterGrid

# 하이퍼파라미터 그리드 설정
param_grid = {
    'hidden_channels': [32, 64, 128],
    'learning_rate': [0.001, 0.01, 0.1],
    'weight_decay': [5e-4, 1e-4, 1e-5]
}

# 그리드 서치
best_accuracy = 0
best_params = None

for params in ParameterGrid(param_grid):
    model = GCN(in_channels=1000, hidden_channels=params['hidden_channels'], out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    model.train()
    for epoch in range(50):  # 간단히 50 에포크만 실행
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        _, pred = out.max(dim=1)
        correct += int((pred == data.y).sum().item())

    accuracy = correct / len(test_loader.dataset)
    print(f'Params: {params}, Accuracy: {accuracy:.4f}')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

print(f'Best Params: {best_params}, Best Accuracy: {best_accuracy:.4f}')
```

### 준비 자료
- **강의 자료**: 모델 학습 및 하이퍼파라미터 튜닝 슬라이드 (PDF)
- **참고 코드**: GNN 모델 학습 및 하이퍼파라미터 튜닝 예제 코드 (Python)

### 과제
- **모델 학습 및 평가**: 제공된 코드 예제를 실행하고, 모델 학습 및 평가 결과 요약.
- **하이퍼파라미터 튜닝**: 하이퍼파라미터 튜닝을 통해 모델 성능을 최적화하고, 결과 요약.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 모델 학습과 하이퍼파라미터 튜닝의 중요성을 이해하고, GNN을 사용하여 실습 프로젝트의 모델 학습 과정을 경험할 수 있도록 유도합니다.

---
