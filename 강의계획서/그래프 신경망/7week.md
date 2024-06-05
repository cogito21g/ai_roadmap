### 7주차 강의 상세 계획: 고급 GNN 모델 1 - Graph Attention Networks (GAT)

#### 강의 목표
- Graph Attention Networks (GAT)의 개념과 원리 이해
- GAT의 주요 구성 요소 학습
- GAT 구현 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 Graph Attention Networks (GAT)의 개념 (20분)

##### GAT란?
- **정의**: 그래프 구조 데이터에서 주어진 노드에 대해 중요한 이웃 노드에 가중치를 두어 정보를 합성하는 신경망 모델.
- **목적**: 다양한 이웃 노드의 중요도를 학습하여 더 정교한 노드 표현을 생성.

#### 1.2 GAT의 주요 구성 요소 (40분)

##### 어텐션 메커니즘
- **정의**: 노드 간의 중요도를 학습하는 메커니즘.
- **수식**: \( e_{ij} = \text{LeakyReLU}(\mathbf{a}^T [\mathbf{W} \mathbf{h}_i || \mathbf{W} \mathbf{h}_j]) \)
- **소프트맥스**: \( \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})} \)

##### GAT 레이어
- **정의**: 어텐션 메커니즘을 적용한 후, 이웃 노드의 정보를 집계하여 노드 표현을 업데이트.
- **수식**: \( \mathbf{h}_i' = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j \right) \)

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 Graph Attention Networks (GAT) 구현 실습

##### 필요 라이브러리 설치
```bash
pip install torch torch-geometric
```

##### GAT 구현 코드 (Python)
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid

# 데이터셋 로드
dataset = Planetoid(root='/tmp/Cora', name='Cora')

# GAT 모델 정의
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 모델 학습 및 평가
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(dataset.num_node_features, dataset.num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model

(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
accuracy = correct / int(data.test_mask.sum())
print(f'Accuracy: {accuracy:.4f}')
```

### 준비 자료
- **강의 자료**: Graph Attention Networks (GAT)의 개념과 원리 슬라이드 (PDF)
- **참고 코드**: GAT 구현 예제 코드 (Python)

### 과제
- **이론 정리**: GAT의 개념과 주요 구성 요소를 요약.
- **코드 실습**: 제공된 GAT 코드를 실행하고, 다른 그래프 데이터로 실습.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 Graph Attention Networks (GAT)의 개념과 원리를 이해하고, 주요 구성 요소를 학습하며, 실제 데이터를 사용해 GAT을 구현하는 경험을 쌓을 수 있도록 유도합니다.

---

