### 3주차 강의 상세 계획: 그래프 신경망 기본 모델

#### 강의 목표
- 그래프 신경망의 기본 모델 이해
- 주요 그래프 신경망 모델 학습
- 그래프 신경망 구현 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 그래프 신경망의 기본 개념 (20분)

##### 그래프 신경망이란?
- **정의**: 그래프 구조 데이터를 학습하기 위한 신경망 모델.
- **목적**: 그래프 데이터의 노드, 엣지, 전체 그래프의 특징을 학습.

#### 1.2 주요 그래프 신경망 모델 (40분)

##### 그래프 합성곱 네트워크 (GCN)
- **정의**: 그래프 데이터에 합성곱 연산을 적용한 모델.
- **특징**: 로컬 그래프 구조를 효과적으로 학습.

##### GraphSAGE
- **정의**: 샘플링과 집계를 통해 그래프 임베딩을 학습하는 모델.
- **특징**: 대규모 그래프에서도 효율적으로 작동.

##### Message Passing Neural Network (MPNN)
- **정의**: 노드 간의 메시지 전달을 통해 특징을 학습하는 모델.
- **특징**: 유연성과 강력한 표현력.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 그래프 합성곱 네트워크 (GCN) 구현 실습

##### 필요 라이브러리 설치
```bash
pip install torch torch-geometric
```

##### GCN 구현 코드 (Python)
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

# 데이터셋 로드
dataset = Planetoid(root='/tmp/Cora', name='Cora')

# GCN 모델 정의
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 모델 학습 및 평가
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
accuracy = correct / int(data.test_mask.sum())
print(f'Accuracy: {accuracy:.4f}')
```

### 준비 자료
- **강의 자료**: 그래프 신경망 기본 모델 슬라이드 (PDF)
- **참고 코드**: GCN 구현 예제 코드 (Python)

### 과제
- **이론 정리**: GCN, GraphSAGE, MPNN의 개념과 특징을 요약.
- **코드 실습**: 제공된 GCN 코드를 실행하고, 다른 그래프 데이터로 실습.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 그래프 신경망의 기본 모델을 이해하고, 주요 그래프 신경망 모델을 학습하며, 실제 데이터를 사용해 그래프 신경망을 구현하는 경험을 쌓을 수 있도록 유도합니다.

---
