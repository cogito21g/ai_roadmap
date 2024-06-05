### 10주차 강의 상세 계획: GNN 응용 2 - 추천 시스템, 지식 그래프

#### 강의 목표
- 추천 시스템과 지식 그래프의 개념과 응용 이해
- GNN을 사용한 추천 시스템과 지식 그래프 분석 실습

#### 강의 구성
- **이론 강의**: 1시간
- **응용 사례 분석 및 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 추천 시스템 (30분)

##### 추천 시스템이란?
- **정의**: 사용자에게 맞춤형 콘텐츠를 추천하는 시스템.
- **응용 분야**: 전자 상거래, 스트리밍 서비스, 소셜 미디어 등.

##### GNN을 사용한 추천 시스템
- **장점**: 사용자와 아이템 간의 복잡한 관계와 패턴을 학습.
- **모델 예시**: NGCF (Neural Graph Collaborative Filtering), GCMC (Graph Convolutional Matrix Completion) 등.

##### 주요 개념
- **사용자-아이템 상호작용 그래프**: 사용자와 아이템을 노드로, 상호작용을 엣지로 표현.
- **노드 임베딩**: 사용자의 선호도와 아이템의 특성을 벡터로 표현.

#### 1.2 지식 그래프 (30분)

##### 지식 그래프란?
- **정의**: 실세계의 지식과 관계를 그래프로 표현한 것.
- **응용 분야**: 검색 엔진, 질문 답변 시스템, 지식 관리 등.

##### GNN을 사용한 지식 그래프
- **장점**: 복잡한 관계와 구조를 학습하여 정확한 예측 및 추론 가능.
- **모델 예시**: R-GCN (Relational Graph Convolutional Networks), KBGAT (Knowledge Base Graph Attention Network) 등.

##### 주요 개념
- **노드와 엣지 타입**: 다양한 유형의 노드와 엣지가 존재.
- **관계 추론**: 노드 간의 관계를 예측하고 새로운 지식을 추론.

---

### 2. 응용 사례 분석 및 실습 (1시간)

#### 2.1 추천 시스템 실습

##### 필요 라이브러리 설치
```bash
pip install torch-geometric
```

##### 추천 시스템 구현 코드 (Python)
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# 사용자-아이템 상호작용 그래프 생성
edge_index = torch.tensor([[0, 1, 2, 0, 3], [1, 2, 0, 3, 0]], dtype=torch.long)
x = torch.eye(4, dtype=torch.float)  # 노드 특성 (사용자 2명, 아이템 2개)
data = Data(x=x, edge_index=edge_index)

# GCN 모델 정의
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
        return x

# 모델 학습 및 예측
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(4, 8, 4).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, data.x)
    loss.backward()
    optimizer.step()

print("Node embeddings after training:\n", model(data))
```

#### 2.2 지식 그래프 실습

##### 필요 라이브러리 설치
```bash
pip install torch-geometric
```

##### 지식 그래프 구현 코드 (Python)
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data

# 지식 그래프 데이터 생성
edge_index = torch.tensor([[0, 1, 2, 0, 3], [1, 2, 0, 3, 0]], dtype=torch.long)
edge_type = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
x = torch.eye(4, dtype=torch.float)  # 노드 특성
data = Data(x=x, edge_index=edge_index, edge_type=edge_type)

# R-GCN 모델 정의
class RGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(in_channels, out_channels, num_relations)
        self.conv2 = RGCNConv(out_channels, out_channels, num_relations)

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return x

# 모델 학습 및 예측
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RGCN(4, 8, 3).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, data.x)
    loss.backward()
    optimizer.step()

print("Node embeddings after training:\n", model(data))
```

### 준비 자료
- **강의 자료**: 추천 시스템과 지식 그래프 슬라이드 (PDF)
- **참고 코드**: 추천 시스템 및 지식 그래프 분석 예제 코드 (Python)

### 과제
- **이론 정리**: 추천 시스템과 지식 그래프의 개념과 응용을 요약.
- **코드 실습**: 제공된 추천 시스템 및 지식 그래프 분석 코드를 실행하고, 다른 데이터를 사용해 실습.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 추천 시스템과 지식 그래프의 개념과 응용을 이해하고, GNN을 사용한 실습을 통해 실제 데이터를 분석하는 경험을 쌓을 수 있도록 유도합니다.

---
