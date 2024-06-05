### 7주차 강의 상세 계획: 순환 신경망 (RNN)

#### 강의 목표
- 순환 신경망 (RNN)의 기본 개념과 구조 이해
- RNN의 주요 변형 (LSTM, GRU) 학습
- RNN을 사용한 텍스트 생성 모델 구현

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 순환 신경망 (RNN)의 기본 개념 (20분)

##### RNN이란?
- **정의**: 순차적 데이터에서 패턴을 학습하는 신경망 구조.
- **특징**: 순환 구조를 통해 이전 상태 정보를 유지하며 학습.

##### RNN의 구조
- **입력 및 출력**: 시퀀스 입력과 시퀀스 출력.
- **순환 구조**: 은닉 상태를 사용하여 이전 시점의 정보를 다음 시점으로 전달.

##### RNN의 수식
- **은닉 상태 업데이트**:
  \[
  h_t = \tanh(W_h \cdot x_t + U_h \cdot h_{t-1} + b_h)
  \]
- **출력**:
  \[
  y_t = W_y \cdot h_t + b_y
  \]

#### 1.2 RNN의 한계와 LSTM, GRU (20분)

##### RNN의 한계
- **기울기 소실 및 폭발**: 장기 의존성을 학습하는 데 어려움.
- **해결 방안**: LSTM, GRU 등 변형 RNN 구조.

##### LSTM (Long Short-Term Memory)
- **정의**: 장기 의존성 문제를 해결하기 위해 고안된 RNN 구조.
- **구조**: 입력 게이트, 출력 게이트, 망각 게이트로 구성.
- **장점**: 장기 의존성 학습 가능.

##### GRU (Gated Recurrent Unit)
- **정의**: LSTM의 변형으로, 구조가 단순화된 RNN.
- **구조**: 업데이트 게이트, 리셋 게이트로 구성.
- **장점**: 계산 비용 절감, 성능 유지.

#### 1.3 RNN의 응용 (20분)

##### 텍스트 생성
- **정의**: 학습된 모델을 사용하여 새로운 텍스트 시퀀스 생성.
- **적용 사례**: 시, 소설, 코드 등 텍스트 생성.

##### 감정 분석
- **정의**: RNN을 사용하여 텍스트의 감정을 예측.
- **적용 사례**: 리뷰 분석, 소셜 미디어 분석.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 RNN을 사용한 텍스트 생성 모델 구현 실습

##### 필요 라이브러리 설치
```bash
pip install numpy torch
```

##### RNN 텍스트 생성 구현 코드 (Python)
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 텍스트 예제 데이터
text = "hello world"

# 문자 집합 생성
chars = sorted(list

(set(text)))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

# 하이퍼파라미터 설정
input_size = len(chars)
hidden_size = 128
output_size = len(chars)
sequence_length = 10
learning_rate = 0.01

# 데이터 전처리
def char_to_tensor(char):
    tensor = torch.zeros(input_size)
    tensor[char_to_idx[char]] = 1
    return tensor

def string_to_tensor(string):
    tensor = torch.zeros(len(string), input_size)
    for li, letter in enumerate(string):
        tensor[li][char_to_idx[letter]] = 1
    return tensor

# RNN 모델 정의
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

# 모델 초기화
model = RNN(input_size, hidden_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

# 학습 함수
def train(input_tensor, target_tensor):
    hidden = model.init_hidden()
    model.zero_grad()
    loss = 0
    
    for i in range(input_tensor.size(0)):
        output, hidden = model(input_tensor[i], hidden)
        loss += criterion(output, target_tensor[i].unsqueeze(0))
    
    loss.backward()
    optimizer.step()
    
    return loss.item() / input_tensor.size(0)

# 텍스트 생성 함수
def generate(start_char='h', predict_len=100, temperature=0.8):
    hidden = model.init_hidden()
    input = char_to_tensor(start_char)
    predicted = start_char
    
    for p in range(predict_len):
        output, hidden = model(input.unsqueeze(0), hidden)
        
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        predicted_char = idx_to_char[top_i.item()]
        predicted += predicted_char
        input = char_to_tensor(predicted_char)
    
    return predicted

# 데이터 준비
input_seq = string_to_tensor(text[:-1])
target_seq = torch.tensor([char_to_idx[char] for char in text[1:]])

# 모델 학습
n_iters = 1000
print_every = 100

for iter in range(1, n_iters + 1):
    loss = train(input_seq, target_seq)
    if iter % print_every == 0:
        print(f"Iteration {iter}, Loss: {loss}")

# 텍스트 생성
generated_text = generate()
print(f"Generated text: {generated_text}")
```

### 준비 자료
- **강의 자료**: RNN, LSTM, GRU 슬라이드 (PDF)
- **참고 코드**: RNN 텍스트 생성 구현 예제 코드 (Python)

### 과제
- **이론 정리**: RNN의 구조와 LSTM, GRU의 개념 요약.
- **코드 실습**: 제공된 RNN 텍스트 생성 코드를 실행하고, 다른 텍스트 데이터로 실습.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 순환 신경망 (RNN)의 개념과 주요 변형 (LSTM, GRU)을 이해하고, RNN을 사용해 텍스트 생성 모델을 구현하는 경험을 쌓을 수 있도록 유도합니다.

---
