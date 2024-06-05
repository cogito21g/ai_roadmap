### 10주차 강의 상세 계획: 텍스트 요약

#### 강의 목표
- 텍스트 요약의 개념과 주요 기법 이해
- 추출적 요약과 생성적 요약 학습
- 텍스트 요약 모델 구현 및 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 텍스트 요약의 개념 (20분)

##### 텍스트 요약이란?
- **정의**: 긴 텍스트를 짧게 요약하는 작업.
- **목적**: 핵심 정보 추출, 문서 요약, 뉴스 요약 등.

##### 요약의 유형
- **추출적 요약**: 원문에서 중요한 문장을 추출하여 요약.
- **생성적 요약**: 원문을 이해하고 새로운 문장으로 요약 생성.

#### 1.2 추출적 요약 (20분)

##### 추출적 요약의 개념
- **정의**: 원문에서 중요한 문장을 추출하여 요약하는 방법.
- **기법**:
  - **빈도 기반 접근**: 빈도가 높은 단어를 포함하는 문장 추출.
  - **중요도 기반 접근**: 문서의 중요도를 평가하여 중요한 문장 추출.

##### 추출적 요약의 평가
- **ROUGE 지표**: 요약의 질을 평가하는 지표.
  - **ROUGE-N**: N-그램 겹침 정도.
  - **ROUGE-L**: 최장 공통 부분 수열 겹침 정도.

#### 1.3 생성적 요약 (20분)

##### 생성적 요약의 개념
- **정의**: 원문을 이해하고 새로운 문장으로 요약을 생성하는 방법.
- **모델**: Seq2Seq 모델 + 어텐션, 트랜스포머 모델 등.

##### 생성적 요약의 평가
- **BLEU 지표**: 생성된 텍스트의 질을 평가하는 지표.
  - **BLEU-N**: N-그램 겹침 정도.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 추출적 요약 구현 실습

##### 필요 라이브러리 설치
```bash
pip install nltk scikit-learn
```

##### 추출적 요약 구현 코드 (Python)
```python
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# NLTK 데이터 다운로드
nltk.download('punkt')

# 입력 텍스트
text = "Natural language processing (NLP) is a field of artificial intelligence (AI) that focuses on the interaction between computers and humans. NLP combines computational linguistics with statistical, machine learning, and deep learning models. It aims to enable computers to understand, interpret, and generate human language."

# 문장 토큰화
sentences = sent_tokenize(text)

# TF-IDF 벡터화
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

# 중요도 점수 계산 (TF-IDF 합계)
scores = np.array(X.sum(axis=1)).flatten()
ranked_sentences = [sentences[i] for i in scores.argsort()[::-1]]

# 상위 N개의 문장 선택
N = 2
summary = " ".join(ranked_sentences[:N])
print("Extractive Summary:", summary)
```

#### 2.2 생성적 요약 구현 실습

##### 필요 라이브러리 설치
```bash
pip install torch transformers
```

##### 생성적 요약 구현 코드 (Python)
```python
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# 모델과 토크나이저 로드
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# 입력 텍스트
text = "Natural language processing (NLP) is a field of artificial intelligence (AI) that focuses on the interaction between computers and humans. NLP combines computational linguistics with statistical, machine learning, and deep learning models. It aims to enable computers to understand, interpret, and generate human language."

# 토크나이징
inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)

# 요약 생성
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=50, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Generative Summary:", summary)
```

### 준비 자료
- **강의 자료**: 텍스트 요약, 추출적 요약, 생성적 요약 슬라이드 (PDF)
- **참고 코드**: 추출적 요약, 생성적 요약 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 텍스트 요약의 개념과 추출적 요약, 생성적 요약의 기법 요약.
- **코드 실습**: 제공된 텍스트 요약 코드를 실행하고, 다른 텍스트 데이터로 실습.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 텍스트 요약의 개념과 주요 기법을 이해하고, 추출적 요약과 생성적 요약 모델을 학습하며, 실제 데이터를 사용해 텍스트 요약 모델을 구현하는 경험을 쌓을 수 있도록 유도합니다.

---

