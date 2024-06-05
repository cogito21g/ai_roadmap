### 8주차 강의 상세 계획: 시퀀스-투-시퀀스 모델

#### 강의 목표
- 시퀀스-투-시퀀스 (Seq2Seq) 모델의 개념과 구조 이해
- 어텐션 메커니즘 학습
- Seq2Seq 모델 구현 및 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 시퀀스-투-시퀀스 모델의 개념 (20분)

##### Seq2Seq 모델이란?
- **정의**: 입력 시퀀스를 다른 시퀀스로 변환하는 모델.
- **응용 분야**: 기계 번역, 텍스트 요약, 대화 시스템 등.

##### Seq2Seq 모델의 구조
- **인코더-디코더 구조**:
  - **인코더**: 입력 시퀀스를 고정된 길이의 컨텍스트 벡터로 변환.
  - **디코더**: 컨텍스트 벡터를 사용하여 출력 시퀀스를 생성.

#### 1.2 어텐션 메커니즘 (20분)

##### 어텐션 메커니즘의 개념
- **정의**: 디코더가 입력 시퀀스의 모든 위치를 참고하여 더 나은 예측을 수행하도록 하는 메커니즘.
- **목적**: 긴 시퀀스의 정보를 효율적으로 처리.

##### 어텐션의 수식
- **어텐션 가중치**:
  \[
  \alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^T \exp(e_{t,j})}
  \]
- **컨텍스트 벡터**:
  \[
  c_t = \sum_{i=1}^T \alpha_{t,i} h_i
  \]
- **디코더 출력**:
  \[
  s_t = f(s_{t-1}, y_{t-1}, c_t)
  \]

#### 1.3 Seq2Seq 모델의 응용 (20분)

##### 기계 번역
- **정의**: 한 언어의 텍스트를 다른 언어로 번역.
- **모델 구조**: Seq2Seq 모델 + 어텐션.

##### 텍스트 요약
- **정의**: 긴 문서를 짧은 요약으로 변환.
- **모델 구조**: Seq2Seq 모델 + 어텐션.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 Seq2Seq 모델 구현 실습

##### 필요 라이브러리 설치
```bash
pip install numpy torch
```

##### Seq2Seq 구현 코드 (Python)
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 하이퍼파라미터 설정
input_size = 10
output_size = 10
hidden_size = 128
num_layers = 1
learning_rate = 0.01
num_epochs = 1000

# Seq2Seq 모델 정의
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        src_len = src.size(1)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.fc.out_features
        
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)
        
        _, (hidden, cell) = self.encoder(src)
        
        input = tgt[:, 0, :]
        for t in range(1, tgt_len):
            output, (hidden, cell) = self.decoder(input.unsqueeze(1), (hidden, cell))
            output = self.fc(output.squeeze(1))
            outputs[:, t, :] = output
            
            teacher_force = np.random.random() < teacher_forcing_ratio
            input = tgt[:, t, :] if teacher_force else output
        
        return outputs

# 데이터 준비
src_seq = torch.randn(2, 5, input_size)  # (batch_size, seq_length, input_size)
tgt_seq = torch.randn(2, 5, output_size) # (batch_size, seq_length, output_size)

# 모델 초기화
model = Seq2Seq(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
for epoch in range(num_epochs):
    output = model(src_seq, tgt_seq)
    loss = criterion(output, tgt_seq)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 예측
with torch.no_grad():
    output = model(src_seq, tgt_seq, teacher_forcing_ratio=0)
    print("Predicted output:", output)
```

### 준비 자료
- **강의 자료**: Seq2Seq 모델, 어텐션 메커니즘 슬라이드 (PDF)
- **참고 코드**: Seq2Seq 모델 구현 예제 코드 (Python)

### 과제
- **이론 정리**: Seq2Seq 모델의 구조와 어텐션 메커니즘의 개념 요약.
- **코드 실습**: 제공된 Seq2Seq 모델 코드를 실행하고, 다른 텍스트 데이터로 실습.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 시퀀스-투-시퀀스 모델의 개념과 구조를 이해하고, 어텐션 메커니즘을 학습하며, 실제 데이터를 사용해 Seq2Seq 모델을 구현하는 경험을 쌓을 수 있도록 유도합니다.

---
