### 5주차 강의 상세 계획: 문서 분류

#### 강의 목표
- 문서 분류의 개념과 주요 기법 이해
- TF-IDF와 나이브 베이즈 분류 알고리즘 학습
- 문서 분류 모델 구현 및 평가

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 문서 분류의 기본 개념 (20분)

##### 문서 분류란?
- **정의**: 주어진 문서를 사전 정의된 카테고리로 자동 분류하는 작업.
- **목적**: 이메일 스팸 필터링, 뉴스 기사 분류, 감정 분석 등.

##### 문서 분류의 주요 단계
- **텍스트 전처리**: 토큰화, 정규화, 정제.
- **특징 추출**: TF-IDF, 단어 임베딩 등.
- **모델 학습**: 분류 알고리즘을 사용하여 모델 학습.
- **모델 평가**: 분류 성능 평가.

#### 1.2 TF-IDF (Term Frequency-Inverse Document Frequency) (20분)

##### TF-IDF의 개념
- **정의**: 단어의 빈도와 역 문서 빈도를 결합하여 단어의 중요도를 측정하는 방법.
- **TF (Term Frequency)**: 단어의 빈도.
- **IDF (Inverse Document Frequency)**: 단어가 문서 전체에서 나타나는 빈도의 역수.

##### TF-IDF의 수식
- **TF**:
  \[
  \text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}
  \]
- **IDF**:
  \[
  \text{IDF}(t) = \log \frac{\text{Total number of documents}}{\text{Number of documents with term } t}
  \]
- **TF-IDF**:
  \[
  \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
  \]

#### 1.3 나이브 베이즈 분류기 (20분)

##### 나이브 베이즈 분류기의 개념
- **정의**: 베이즈 정리를 기반으로 독립 가정 하에 분류를 수행하는 알고리즘.
- **장점**: 간단하고 빠르며, 텍스트 분류에서 좋은 성능.

##### 나이브 베이즈의 수식
- **베이즈 정리**:
  \[
  P(c|d) = \frac{P(d|c) \cdot P(c)}{P(d)}
  \]
- **나이브 가정**: 단어의 독립성 가정.
- **최종 분류**:
  \[
  c_{\text{NB}} = \arg\max_c P(c) \prod_{i=1}^n P(w_i|c)
  \]

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 TF-IDF와 나이브 베이즈 분류기 구현 실습

##### 필요 라이브러리 설치
```bash
pip install nltk scikit-learn
```

##### 문서 분류 구현 코드 (Python)
```python
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# NLTK 데이터 다운로드
nltk.download('punkt')

# 텍스트 예제 데이터
texts = [
    "This is a good movie",
    "I did not like this movie",
    "Amazing film, really enjoyed it",
    "Worst movie ever",
    "It was a fantastic film",
    "Not my cup of tea",
    "An excellent movie",
    "Terrible film, not recommended"
]

# 레이블 (0: 부정, 1: 긍정)
labels = [1, 0, 1, 0, 1, 0, 1, 0]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42)

# TF-IDF와 나이브 베이즈 파이프라인
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# 모델 학습
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 새로운 텍스트 예측
new_text = ["I really liked this movie"]
predicted_label = model.predict(new_text)
print(f"Predicted label for '{new_text[0]}': {predicted_label[0]}")
```

### 준비 자료
- **강의 자료**: 문서 분류, TF-IDF, 나이브 베이즈 슬라이드 (PDF)
- **참고 코드**: 문서 분류 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 문서 분류의 주요 단계와 TF-IDF, 나이브 베이즈의 개념 요약.
- **코드 실습**: 제공된 문서 분류 코드를 실행하고, 다른 텍스트 데이터로 실습.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 문서 분류의 개념과 주요 기법을 이해하고, TF-IDF와 나이브 베이즈 분류기를 학습하며, 실제 데이터를 사용해 문서 분류 모델을 구현하는 경험을 쌓을 수 있도록 유도합니다.

---
