### 3주차 강의 상세 계획: 언어 모델

#### 강의 목표
- 언어 모델의 기본 개념과 원리 이해
- N-그램 모델 학습
- 언어 모델 평가 방법 학습 및 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 언어 모델의 기본 개념 (20분)

##### 언어 모델이란?
- **정의**: 단어 시퀀스의 확률을 추정하는 모델.
- **목적**: 문법적이고 자연스러운 문장을 생성하거나 예측.

##### 언어 모델의 유형
- **통계적 언어 모델**: N-그램 모델.
- **신경망 언어 모델**: RNN, LSTM, Transformer.

#### 1.2 N-그램 모델 (20분)

##### N-그램 모델의 정의
- **정의**: N개의 연속된 단어들의 빈도에 기반하여 다음 단어의 확률을 추정.
- **N의 값에 따른 모델**:
  - Unigram 모델 (N=1): 단어 하나의 확률.
  - Bigram 모델 (N=2): 두 단어의 연속 확률.
  - Trigram 모델 (N=3): 세 단어의 연속 확률.

##### N-그램 모델의 수식
- **확률 계산**:
  \[
  P(w_1, w_2, ..., w_n) = \prod_{i=1}^n P(w_i | w_{i-(N-1)}, ..., w_{i-1})
  \]

##### N-그램 모델의 한계
- **데이터 희소성 문제**: 드문 단어 조합에 대한 확률 예측 어려움.
- **문맥의 한계**: 긴 문맥을 반영하기 어려움.

#### 1.3 언어 모델 평가 (20분)

##### 평가 방법
- **퍼플렉서티 (Perplexity)**:
  - **정의**: 모델이 예측한 확률의 불확실성을 측정.
  - **수식**:
    \[
    \text{Perplexity}(P) = 2^{-\frac{1}{N} \sum_{i=1}^N \log_2 P(w_i | w_{1}, w_{2}, ..., w_{i-1})}
    \]

##### 퍼플렉서티 계산 방법
- **퍼플렉서티의 의미**: 낮을수록 좋은 성능, 모델의 예측이 정확함을 의미.
- **퍼플렉서티 계산 예제**: 주어진 텍스트 시퀀스에 대해 퍼플렉서티 계산.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 N-그램 모델 구현 실습

##### 필요 라이브러리 설치
```bash
pip install nltk
```

##### N-그램 모델 구현 코드 (Python)
```python
import nltk
from collections import defaultdict, Counter
import math

# NLTK 데이터 다운로드
nltk.download('punkt')

class NgramModel:
    def __init__(self, n):
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.context_counts = Counter()
    
    def train(self, text):
        tokens = nltk.word_tokenize(text)
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i+self.n])
            context = ngram[:-1]
            word = ngram[-1]
            self.ngrams[context][word] += 1
            self.context_counts[context] += 1
    
    def predict(self, context):
        context = tuple(context[-(self.n-1):])
        if context in self.ngrams:
            return self.ngrams[context].most_common(1)[0][0]
        else:
            return None
    
    def perplexity(self, text):
        tokens = nltk.word_tokenize(text)
        log_prob = 0
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i+self.n])
            context = ngram[:-1]
            word = ngram[-1]
            if self.context_counts[context] > 0:
                prob = self.ngrams[context][word] / self.context_counts[context]
                log_prob += math.log(prob, 2)
            else:
                log_prob += float('-inf')
        return 2 ** (-log_prob / len(tokens))

# 텍스트 예제
text = "This is a test text for n-gram model implementation. The n-gram model predicts the next word based on previous words."

# N-그램 모델 학습
ngram_model = NgramModel(n=2)
ngram_model.train(text)

# 다음 단어 예측
context = ["This", "is"]
print(f"Next word prediction for context '{context}': {ngram_model.predict(context)}")

# 퍼플렉서티 계산
perplexity = ngram_model.perplexity(text)
print(f"Perplexity of the model on the given text: {perplexity}")
```

### 준비 자료
- **강의 자료**: 언어 모델 및 N-그램 모델 슬라이드 (PDF)
- **참고 코드**: N-그램 모델 구현 예제 코드 (Python)

### 과제
- **이론 정리**: N-그램 모델의 개념과 퍼플렉서티 계산 방법 요약.
- **코드 실습**: 제공된 N-그램 모델 코드를 실행하고, 다른 텍스트에 적용하여 퍼플렉서티 계산.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 언어 모델의 개념과 N-그램 모델의 원리를 이해하고, 언어 모델을 평가하는 방법을 학습하며, 실제 데이터를 사용해 N-그램 모델을 구현하는 경험을 쌓을 수 있도록 유도합니다.

---
