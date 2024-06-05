### 9주차 강의 상세 계획: GNN 응용 1 - 분자 그래프, 소셜 네트워크 분석

#### 강의 목표
- 분자 그래프와 소셜 네트워크 분석의 개념과 응용 이해
- GNN을 사용한 분자 그래프와 소셜 네트워크 분석 실습

#### 강의 구성
- **이론 강의**: 1시간
- **응용 사례 분석 및 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 분자 그래프 분석 (30분)

##### 분자 그래프란?
- **정의**: 분자를 그래프로 표현한 것. 원자는 노드, 화학 결합은 엣지로 표현.
- **응용 분야**: 약물 발견, 화학 반응 예측, 물질의 물리적 특성 예측.

##### GNN을 사용한 분자 그래프 분석
- **장점**: 분자의 구조적 정보를 효과적으로 학습.
- **모델 예시**: MPNN, GCN, GAT 등.

#### 1.2 소셜 네트워크 분석 (30분)

##### 소셜 네트워크란?
- **정의**: 사람들 간의 관계를 그래프로 표현한 것.
- **응용 분야**: 영향력 분석, 커뮤니티 탐지, 정보 확산 모델링.

##### GNN을 사용한 소셜 네트워크 분석
- **장점**: 소셜 네트워크의 복잡한 관계와 패턴을 효과적으로 학습.
- **모델 예시**: GCN, GraphSAGE, GAT 등.

---

### 2. 응용 사례 분석 및 실습 (1시간)

#### 2.1 분자 그래프 분석 실습

##### 필요 라이브러리 설치
```bash
pip install torch-geometric rdkit
```

##### 분자 그래프 분석 코드 (Python)
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem

# 분자 데이터를 그래프로 변환하는 함수
def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    AllChem.Compute2DCoords(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    x = torch.tensor([atom.GetAtomicNum() for atom in atoms], dtype=torch.float).view(-1, 1)
    edge_index = torch.tensor([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in bonds], dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    return data

# 예제 분자 데이터 (SMILES 형식)
smiles_list = ['CCO', 'CCN', '

CCC']
graphs = [mol_to_graph(smiles) for smiles in smiles_list]

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
model = GCN(1, 16, 1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.train()
for epoch in range(100):
    for data in graphs:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.x)
        loss.backward()
        optimizer.step()

print("Model training completed for molecular graphs")
```

#### 2.2 소셜 네트워크 분석 실습

##### 필요 라이브러리 설치
```bash
pip install torch-geometric networkx
```

##### 소셜 네트워크 분석 코드 (Python)
```python
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# 소셜 네트워크 데이터 생성
G = nx.karate_club_graph()
edge_index = torch.tensor(list(G.edges)).t().contiguous()
x = torch.eye(G.number_of_nodes(), dtype=torch.float)
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
model = GCN(G.number_of_nodes(), 16, 2).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
accuracy = correct / int(data.test_mask.sum())
print(f'Accuracy: {accuracy:.4f}')
```

### 준비 자료
- **강의 자료**: 분자 그래프와 소셜 네트워크 분석 슬라이드 (PDF)
- **참고 코드**: 분자 그래프 및 소셜 네트워크 분석 예제 코드 (Python)

### 과제
- **이론 정리**: 분자 그래프와 소셜 네트워크 분석의 개념과 응용을 요약.
- **코드 실습**: 제공된 분자 그래프 및 소셜 네트워크 분석 코드를 실행하고, 다른 데이터를 사용해 실습.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 분자 그래프와 소셜 네트워크 분석의 개념과 응용을 이해하고, GNN을 사용한 실습을 통해 실제 데이터를 분석하는 경험을 쌓을 수 있도록 유도합니다.

---
