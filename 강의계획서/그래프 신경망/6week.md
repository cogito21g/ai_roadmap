### 6주차 강의 상세 계획: GNN을 위한 데이터 전처리

#### 강의 목표
- 그래프 신경망(GNN)에서의 데이터 전처리 중요성 이해
- 노드 특성, 엣지 특성, 그래프 정규화 등 전처리 기법 학습
- 실제 그래프 데이터를 전처리하는 실습

#### 강의 구성
- **이론 강의**: 1시간
- **데이터 전처리 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 그래프 데이터 전처리의 중요성 (20분)

##### 데이터 전처리란?
- **정의**: 원본 데이터를 모델 학습에 적합한 형식으로 변환하는 과정.
- **목적**: 데이터의 품질을 높여 모델의 성능을 최적화.

##### 그래프 데이터 전처리의 중요성
- **특징**: 그래프 데이터는 복잡하고, 노드와 엣지의 다양한 특성을 고려해야 함.
- **필요성**: 노드와 엣지의 특성 추출, 그래프 정규화 등을 통해 모델 학습 효과 증대.

#### 1.2 노드 특성과 엣지 특성 (20분)

##### 노드 특성 (Node Features)
- **정의**: 각 노드가 가지는 속성 값들.
- **예시**: 노드의 속성 값, 노드의 중심성, 노드의 연결도 등.
- **추출 방법**: 원본 데이터에서 직접 추출하거나, 추가 연산을 통해 생성.

##### 엣지 특성 (Edge Features)
- **정의**: 각 엣지가 가지는 속성 값들.
- **예시**: 엣지의 가중치, 엣지의 유형 등.
- **추출 방법**: 원본 데이터에서 직접 추출하거나, 추가 연산을 통해 생성.

#### 1.3 그래프 정규화 (20분)

##### 그래프 정규화란?
- **정의**: 그래프 데이터를 일정한 범위로 스케일링하거나, 특정 기준에 맞게 변환하는 과정.
- **목적**: 모델 학습의 안정성과 성능 향상을 위해.

##### 정규화 기법
- **노드 특성 정규화**: 노드의 특성을 일정한 범위로 변환.
- **엣지 특성 정규화**: 엣지의 특성을 일정한 범위로 변환.
- **그래프 정규화**: 그래프 전체를 일정한 기준에 맞게 변환.

---

### 2. 데이터 전처리 실습 (1시간)

#### 2.1 노드 특성과 엣지 특성 추출 실습

##### 필요 라이브러리 설치
```bash
pip install networkx torch-geometric
```

##### 노드 특성과 엣지 특성 추출 코드 (Python)
```python
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

# 데이터셋 로드
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# NetworkX 그래프 생성
G = nx.Graph()
G.add_edges_from(data.edge_index.t().tolist())

# 노드 특성 추출 예제: 노드의 차수
node_degrees = dict(G.degree())
node_features = torch.tensor([[node_degrees[i]] for i in range(len(G.nodes))], dtype=torch.float)

# 엣지 특성 추출 예제: 엣지의 가중치 (모두 1로 설정)
edge_weights = torch.ones(data.edge_index.size(1), dtype=torch.float)

print("Node Features:\n", node_features)
print("Edge Weights:\n", edge_weights)
```

#### 2.2 그래프 정규화 실습

##### 그래프 정규화 코드 (Python)
```python
import torch_geometric.transforms as T

# 데이터 정규화 예제: 노드 특성 정규화
data.x = node_features
transform = T.NormalizeFeatures()
data = transform(data)

print("Normalized Node Features:\n", data.x)

# 데이터 정규화 예제: 엣지 가중치 정규화
data.edge_attr = edge_weights.view(-1, 1)
transform = T.NormalizeFeatures()
data = transform(data)

print("Normalized Edge Weights:\n", data.edge_attr)
```

### 준비 자료
- **강의 자료**: GNN 데이터 전처리의 중요성 및 기법 슬라이드 (PDF)
- **참고 코드**: 노드 특성, 엣지 특성 추출 및 그래프 정규화 예제 코드 (Python)

### 과제
- **이론 정리**: 노드 특성, 엣지 특성, 그래프 정규화의 개념과 중요성을 요약.
- **코드 실습**: 제공된 전처리 코드를 실행하고, 다른 그래프 데이터로 실습.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 그래프 신경망에서 데이터 전처리의 중요성을 이해하고, 노드와 엣지 특성을 추출하고 정규화하는 방법을 학습하며, 실제 데이터를 사용해 데이터 전처리를 수행하는 경험을 쌓을 수 있도록 유도합니다.

---
