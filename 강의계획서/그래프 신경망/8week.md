### 8주차 강의 상세 계획: 고급 GNN 모델 2 - Graph Isomorphism Networks (GIN)

#### 강의 목표
- Graph Isomorphism Networks (GIN)의 개념과 원리 이해
- GIN의 주요 구성 요소 학습
- GIN 구현 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 Graph Isomorphism Networks (GIN)의 개념 (20분)

##### GIN이란?
- **정의**: 그래프 동형성 문제를 해결하기 위해 설계된 그래프 신경망 모델.
- **목적**: 그래프 동형성 문제에서 최고 성능을 발휘하도록 설계됨.

##### 그래프 동형성(Graph Isomorphism)
- **정의**: 두 그래프가 구조적으로 동일한지를 판단하는 문제.
- **중요성**: 그래프 신경망 모델의 표현력 평가 기준 중 하나.

#### 1.2 GIN의 주요 구성 요소 (40분)

##### GIN 레이어
- **정의**: GIN 레이어는 특정 형태의 집계 함수를 사용하여 노드의 특징을 업데이트.
- **수식**: \( h_v^{(k)} = \text{MLP}^{(k)} \left( (1 + \epsilon^{(k)}) \cdot h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)} \right) \)
- **특징**: 각 노드의 현재 특징과 이웃 노드들의 특징을 합산하여 업데이트.

##### 하이퍼파라미터 \( \epsilon \)
- **역할**: 노드 자신의 특징을 얼마나 반영할지 조절.
- **설정 방법**: 학습을 통해 최적값을 찾거나 고정된 값으로 설정.

##### MLP (Multilayer Perceptron)
- **정의**: 특징 업데이트를 위한 다층 퍼셉트론.
- **구성**: 입력층, 은닉층, 출력층으로 구성.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 Graph Isomorphism Networks (GIN) 구현 실습

##### 필요 라이브러리 설치
```bash
pip install torch torch-geometric
```

##### GIN 구현 코드 (Python)
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

# 데이터셋 로드
dataset = Planetoid(root='/tmp/Cora', name='Cora')

# MLP 정의
class MLP(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)])

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.bns[i](x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x

# GIN 모델 정의
class GIN(torch.nn.Module):
    def __init__(self, num_layers, hidden_dim, num_classes):
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(MLP(2, dataset.num_node_features, hidden_dim, hidden_dim)))
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(MLP(2, hidden_dim, hidden_dim, hidden_dim)))
        self.convs.append(GINConv(MLP(2, hidden_dim, hidden_dim, num_classes)))
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)])

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        x = global_add_pool(x, batch)
        return F.log_softmax(x, dim=1)

# 모델 학습 및 평가
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(num_layers=3, hidden_dim=64, num_classes=dataset.num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.batch)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data.x, data.edge_index, data.batch).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
accuracy = correct / int(data.test_mask.sum())
print(f'Accuracy: {accuracy:.4f}')
```

### 준비 자료
- **강의 자료**: Graph Isomorphism Networks (GIN)의 개념과 원리 슬라이드 (PDF)
- **참고 코드**: GIN 구현 예제 코드 (Python)

### 과제
- **이론 정리**: GIN의 개념과 주요 구성 요소를 요약.
- **코드 실습**: 제공된 GIN 코드를 실행하고, 다른 그래프 데이터로 실습.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 Graph Isomorphism Networks (GIN)의 개념과 원리를 이해하고, 주요 구성 요소를 학습하며, 실제 데이터를 사용해 GIN을 구현하는 경험을 쌓을 수 있도록 유도합니다.

---
