### 6주차 강의 상세 계획: 감정 분석

#### 강의 목표
- 감정 분석의 개념과 주요 기법 이해
- 감정 분석을 위한 데이터 전처리 및 특징 추출 방법 학습
- 감정 분석 모델 구현 및 평가

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 감정 분석의 기본 개념 (20분)

##### 감정 분석이란?
- **정의**: 텍스트 데이터를 분석하여 긍정적, 부정적, 중립적 감정을 분류하는 작업.
- **목적**: 소셜 미디어 의견 분석, 제품 리뷰 분석, 고객 만족도 평가 등.

##### 감정 분석의 주요 단계
- **텍스트 전처리**: 토큰화, 정규화, 정제.
- **특징 추출**: TF-IDF, 단어 임베딩 등.
- **모델 학습**: 분류 알고리즘을 사용하여 감정 예측 모델 학습.
- **모델 평가**: 분류 성능 평가.

#### 1.2 감정 분석을 위한 데이터 전처리 (20분)

##### 데이터 전처리 기법
- **토큰화**: 문장을 단어 단위로 분할.
- **정규화**: 대소문자 변환, 숫자 처리 등.
- **정제**: 불필요한 문자, 특수 문자 제거.

##### 특징 추출 방법
- **TF-IDF**: 단어의 빈도와 역 문서 빈도를 결합하여 단어의 중요도를 측정.
- **단어 임베딩**: Word2Vec, GloVe 등을 사용하여 단어 벡터를 학습.

#### 1.3 감정 분석 모델 (20분)

##### 나이브 베이즈 분류기
- **정의**: 베이즈 정리를 기반으로 독립 가정 하에 분류를 수행하는 알고리즘.
- **장점**: 간단하고 빠르며, 텍스트 분류에서 좋은 성능.

##### 로지스틱 회귀
- **정의**: 입력 특징의 선형 결합을 통해 이진 분류를 수행하는 알고리즘.
- **장점**: 해석이 용이하고, 비교적 간단한 모델.

##### 서포트 벡터 머신 (SVM)
- **정의**: 고차원 공간에서 최적의 초평면을 찾아 분류를 수행하는 알고리즘.
- **장점**: 높은 분류 성능, 과적합 방지.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 감정 분석 모델 구현 실습

##### 필요 라이브러리 설치
```bash
pip install nltk scikit-learn
```

##### 감정 분석 구현 코드 (Python)
```python
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# NLTK 데이터 다운로드
nltk.download('punkt')

# 텍스트 예제 데이터
texts = [
    "I love this product!", "This is the worst thing I've ever bought.",
    "Absolutely fantastic service!", "Not good, very disappointed.",
    "Amazing quality and fast shipping!", "Terrible customer support.",
    "I'm very happy with my purchase.", "Will never buy again from this store."
]

# 레이블 (0: 부정, 1: 긍정)
labels = [1, 0, 1, 0, 1, 0, 1, 0]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42)

# TF-IDF 벡터화
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 나이브 베이즈 분류기
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)
print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))

# 로지스틱 회귀
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

# 서포트 벡터 머신 (SVM)
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
```

### 준비 자료
- **강의 자료**: 감정 분석, 데이터 전처리, 나이브 베이즈, 로지스틱 회귀, SVM 슬라이드 (PDF)
- **참고 코드**: 감정 분석 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 감정 분석의 주요 단계와 나이브 베이즈, 로지스틱 회귀, SVM의 개념 요약.
- **코드 실습**: 제공된 감정 분석 코드를 실행하고, 다른 텍스트 데이터로 실습.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 감정 분석의 개념과 주요 기법을 이해하고, 나이브 베이즈, 로지스틱 회귀, SVM을 학습하며, 실제 데이터를 사용해 감정 분석 모델을 구현하는 경험을 쌓을 수 있도록 유도합니다.

---
