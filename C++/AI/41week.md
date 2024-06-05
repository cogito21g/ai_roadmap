### 41주차: 자연어 처리 (NLP) 기초

#### 강의 목표
- 자연어 처리의 기본 개념 이해
- 텍스트 전처리 기법 학습
- 기본적인 NLP 모델 구현

#### 강의 내용

##### 1. 자연어 처리의 기본 개념
- **자연어 처리(NLP) 개요**
  - 정의: 인간의 언어를 이해하고 생성하는 기계 학습 분야
  - 주요 응용 분야: 텍스트 분류, 감정 분석, 기계 번역, 질의 응답 시스템

- **NLP의 주요 기법**
  - 토큰화(Tokenization)
  - 어휘 정규화(Lemmatization and Stemming)
  - 불용어 제거(Stop Words Removal)
  - Bag of Words, TF-IDF

##### 2. 텍스트 전처리 기법
- **토큰화(Tokenization)**
  - 설명: 문장을 단어 또는 문장으로 분리하는 과정
  - 구현 예제:

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = "Hello, how are you?"
tokens = word_tokenize(text)
print(tokens)
```

- **어휘 정규화(Lemmatization and Stemming)**
  - 설명: 단어의 원형을 찾는 과정
  - 구현 예제 (Lemmatization):

```python
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running"))
```

- **불용어 제거(Stop Words Removal)**
  - 설명: 의미가 적은 일반적인 단어를 제거하는 과정
  - 구현 예제:

```python
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
words = word_tokenize("This is a sample sentence.")
filtered_words = [word for word in words if word.lower() not in stop_words]
print(filtered_words)
```

- **Bag of Words 및 TF-IDF**
  - 설명: 텍스트 데이터를 수치 데이터로 변환하는 방법
  - 구현 예제 (TF-IDF):

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = ["This is a sample sentence.", "This is another example."]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
print(tfidf_matrix.toarray())
```

##### 3. 기본적인 NLP 모델 구현
- **감정 분석 모델 구현**
  - 데이터셋: 영화 리뷰 데이터셋 (예: IMDb)
  - 모델: 로지스틱 회귀

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 데이터 로드
# 데이터셋을 직접 다운로드하거나 데이터셋의 경로를 지정하세요
# 예제 데이터셋은 https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz 에서 다운로드 가능합니다.
data = pd.read_csv('imdb_reviews.csv')  # 예제용 데이터셋 경로

# 데이터 전처리
X = data['review']
y = data['sentiment'].map({'positive': 1, 'negative': 0})

# 텍스트 데이터 벡터화
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

- **텍스트 분류 모델 구현**
  - 데이터셋: 뉴스 기사 데이터셋 (예: 20 Newsgroups)
  - 모델: 나이브 베이즈 분류기

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 데이터 로드
newsgroups = fetch_20newsgroups(subset='train')
X_train, y_train = newsgroups.data, newsgroups.target

newsgroups_test = fetch_20newsgroups(subset='test')
X_test, y_test = newsgroups_test.data, newsgroups_test.target

# 텍스트 데이터 벡터화
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# 모델 학습
model = MultinomialNB()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

#### 과제

1. **텍스트 전처리**
   - 주어진 텍스트 데이터에 대해 토큰화, 어휘 정규화, 불용어 제거를 수행합니다.
   - Bag of Words 또는 TF-IDF를 사용하여 텍스트 데이터를 수치 데이터로 변환합니다.

2. **감정 분석 모델 구현**
   - 주어진 영화 리뷰 데이터셋을 사용하여 감정 분석 모델을 구현하고, 모델을 훈련 및 평가합니다.

3. **텍스트 분류 모델 구현**
   - 주어진 뉴스 기사 데이터셋을 사용하여 텍스트 분류 모델을 구현하고, 모델을 훈련 및 평가합니다.

#### 퀴즈

1. **자연어 처리(NLP)의 주요 응용 분야가 아닌 것은?**
   - A) 텍스트 분류
   - B) 감정 분석
   - C) 기계 번역
   - D) 이미지 인식

2. **토큰화(Tokenization)의 주요 목적은 무엇인가?**
   - A) 텍스트를 단어 또는 문장으로 분리하기 위해
   - B) 텍스트의 불필요한 부분을 제거하기 위해
   - C) 텍스트의 중요도를 계산하기 위해
   - D) 텍스트의 차원을 축소하기 위해

3. **TF-IDF의 주요 목적은 무엇인가?**
   - A) 단어의 빈도를 계산하기 위해
   - B) 단어의 중요도를 계산하기 위해
   - C) 텍스트를 요약하기 위해
   - D) 텍스트의 차원을 축소하기 위해

4. **감정 분석에 사용될 수 있는 기계 학습 모델은?**
   - A) 선형 회귀
   - B) 로지스틱 회귀
   - C) K-평균 군집화
   - D) 주성분 분석

#### 퀴즈 해설

1. **자연어 처리(NLP)의 주요 응용 분야가 아닌 것은?**
   - **정답: D) 이미지 인식**
     - 해설: 자연어 처리의 주요 응용 분야에는 텍스트 분류, 감정 분석, 기계 번역 등이 있으며, 이미지 인식은 컴퓨터 비전의 응용 분야입니다.

2. **토큰화(Tokenization)의 주요 목적은 무엇인가?**
   - **정답: A) 텍스트를 단어 또는 문장으로 분리하기 위해**
     - 해설: 토큰화는 텍스트를 단어 또는 문장으로 분리하는 과정입니다.

3. **TF-IDF의 주요 목적은 무엇인가?**
   - **정답: B) 단어의 중요도를 계산하기 위해**
     - 해설: TF-IDF는 문서에서 단어의 중요도를 계산하는 방법입니다.

4. **감정 분석에 사용될 수 있는 기계 학습 모델은?**
   - **정답: B) 로지스틱 회귀**
     - 해설: 로지스틱 회귀는 이진 분류 문제에 사용될 수 있으며, 감정 분석에 적합한 모델입니다.

다음 주차 강의 내용을 요청하시면, 42주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.