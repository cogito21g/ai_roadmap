### 5주차 강의 상세 계획: 메시지 패싱 신경망 (MPNN)

#### 강의 목표
- 메시지 패싱 신경망(MPNN)의 개념과 원리 이해
- MPNN의 주요 구성 요소 학습
- MPNN 구현 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 메시지 패싱 신경망의 개념 (20분)

##### 메시지 패싱 신경망(MPNN)이란?
- **정의**: 그래프 구조 데이터를 학습하기 위해 노드 간의 메시지를 전달하고 업데이트하는 신경망 모델.
- **특징**: 노드와 엣지의 특징을 동시에 학습할 수 있음.

##### MPNN의 주요 구성 요소
- **메시지 함수**: 노드 간의 정보를 전달하는 함수.
- **노드 업데이트 함수**: 전달받은 메시지를 사용하여 노드의 상태를 업데이트하는 함수.
- **읽기 함수**: 그래프의 전체적인 상태를 요약하는 함수.

#### 1.2 MPNN의 작동 원리 (40분)

##### 메시지 전달 단계
- **정의**: 각 노드가 인접한 노드로부터 메시지를 수신하고 이를 집계.
- **수식**: \( m_{i}^{(t)} = \sum_{j \in \mathcal{N}(i)} M(h_{i}^{(t-1)}, h_{j}^{(t-1)}, e_{ij}) \)

##### 노드 업데이트 단계
- **정의**: 수신한 메시지를 바탕으로 노드의 상태를 업데이트.
- **수식**: \( h_{i}^{(t)} = U(h_{i}^{(t-1)}, m_{i}^{(t)}) \)

##### 읽기 단계
- **정의**: 그래프의 전체적인 정보를 요약하여 출력.
- **수식**: \( \hat{y} = R(\{h_{i}^{(T)} | i \in V\}) \)

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 메시지 패싱 신경망 구현 실습

##### 필요 라이브러리 설치
```bash
pip install torch torch-geometric
```

##### 메시지 패싱 신경망 구현 코드 (Python)
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import Planetoid

# 데이터셋 로드
dataset = Planetoid(root='/tmp/Cora', name='Cora')

# MPNN 모델 정의
class MPNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNN, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # 메시지 전달 단계
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_j, edge_index, size):
        # 메시지 전달 함수
        return x_j

    def update(self, aggr_out):
        # 노드 업데이트 함수
        return aggr_out

# 모델 학습 및 평가
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MPNN(dataset.num_node_features, dataset.num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-

4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data.x, data.edge_index).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
accuracy = correct / int(data.test_mask.sum())
print(f'Accuracy: {accuracy:.4f}')
```

### 준비 자료
- **강의 자료**: 메시지 패싱 신경망의 개념과 원리 슬라이드 (PDF)
- **참고 코드**: MPNN 구현 예제 코드 (Python)

### 과제
- **이론 정리**: MPNN의 개념과 주요 구성 요소를 요약.
- **코드 실습**: 제공된 MPNN 코드를 실행하고, 다른 그래프 데이터로 실습.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 메시지 패싱 신경망의 개념과 원리를 이해하고, 주요 구성 요소를 학습하며, 실제 데이터를 사용해 MPNN을 구현하는 경험을 쌓을 수 있도록 유도합니다.

---
