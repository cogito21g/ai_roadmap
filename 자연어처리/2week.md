### 2주차 강의 상세 계획: 텍스트 전처리

#### 강의 목표
- 텍스트 전처리의 개념과 중요성 이해
- 주요 텍스트 전처리 기법 학습
- 텍스트 전처리 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 텍스트 전처리의 개념 (20분)

##### 텍스트 전처리란?
- **정의**: 텍스트 데이터를 분석 가능하게 준비하는 과정.
- **주요 기법**:
  - **토큰화**: 문장을 단어 또는 서브워드 단위로 분할.
  - **정규화**: 대소문자 변환, 숫자 처리 등.
  - **정제**: 불필요한 문자, 특수 문자 제거.

#### 1.2 텍스트 전처리 기법 (40분)

##### 토큰화 (Tokenization)
- **정의**: 텍스트를 최소 의미 단위로 분할.
- **기법**:
  - **단어 토큰화**: 문장을 단어 단위로 분할.
  - **문장 토큰화**: 텍스트를 문장 단위로 분할.
  - **서브워드 토큰화**: BPE, SentencePiece 등.

##### 정규화 (Normalization)
- **정의**: 텍스트를 일관된 형태로 변환.
- **기법**:
  - **대소문자 변환**: 모든 텍스트를 소문자로 변환.
  - **숫자 처리**: 숫자를 텍스트로 변환.
  - **표제어 추출**: 단어의 기본형 추출.

##### 정제 (Cleaning)
- **정의**: 불필요한 문자나 특수 문자 제거.
- **기법**:
  - **불용어 제거**: 의미 없는 단어 제거.
  - **특수 문자 제거**: 구두점, 특수 문자 제거.
  - **중복 제거**: 중복 단어 제거.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 텍스트 전처리 실습

##### 필요 라이브러리 설치
```bash
pip install nltk spacy
```

##### 텍스트 전처리 구현 코드 (Python)
```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
import re

# NLTK 데이터 다운로드
nltk.download('punkt')
nltk.download('stopwords')

# 텍스트 예제
text = "Hello world! This is a test text for NLP preprocessing. Let's see how it works."

# 토큰화
sentences = sent_tokenize(text)
words = word_tokenize(text)
print("Sentences:", sentences)
print("Words:", words)

# 정규화
text = text.lower()
print("Lowercased Text:", text)

# 정제
stop_words = set(stopwords.words('english'))
words = [word for word in words if word.isalnum() and word not in stop_words]
print("Cleaned Words:", words)

# 표제어 추출
nlp = spacy.load("en_core_web_sm")
doc = nlp(" ".join(words))
lemmas = [token.lemma_ for token in doc]
print("Lemmas:", lemmas)
```

### 준비 자료
- **강의 자료**: 텍스트 전처리 슬라이드 (PDF)
- **참고 코드**: 텍스트 전처리 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 텍스트 전처리 기법 요약.
- **

코드 실습**: 제공된 텍스트 전처리 코드를 실행하고, 다른 텍스트에 적용.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 텍스트 전처리의 개념과 중요성을 이해하고, 주요 텍스트 전처리 기법을 학습하며, 실제 데이터를 처리하는 경험을 쌓을 수 있도록 유도합니다.

---

이와 같은 방식으로 NLP 기초 강의 계획을 주차별로 상세하게 구성해 나갈 수 있습니다. 다음 주차의 상세 계획이 필요하거나, 다른 주제에 대한 강의 계획이 필요하면 알려주세요.