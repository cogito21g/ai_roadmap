### 37주차: 기계 학습 (Machine Learning) 기초

#### 강의 목표
- 기계 학습의 기본 개념 이해
- 데이터 전처리 및 탐색적 데이터 분석
- 기본적인 기계 학습 모델 구현

#### 강의 내용

##### 1. 기계 학습의 기본 개념
- **기계 학습 개요**
  - 정의: 데이터를 통해 학습하고 예측 또는 결정을 내리는 알고리즘
  - 주요 유형: 지도 학습, 비지도 학습, 강화 학습

- **지도 학습 (Supervised Learning)**
  - 정의: 입력 데이터와 그에 대응하는 출력 데이터를 통해 모델을 학습
  - 주요 알고리즘: 선형 회귀, 로지스틱 회귀, 의사결정 나무, 서포트 벡터 머신, k-최근접 이웃

- **비지도 학습 (Unsupervised Learning)**
  - 정의: 입력 데이터만을 사용하여 데이터의 구조나 패턴을 학습
  - 주요 알고리즘: k-평균 군집화, 주성분 분석(PCA), 연관 규칙 학습

##### 2. 데이터 전처리 및 탐색적 데이터 분석 (EDA)
- **데이터 전처리**
  - 결측값 처리
  - 이상값 제거
  - 데이터 정규화 및 표준화

- **탐색적 데이터 분석 (EDA)**
  - 데이터 분포 확인
  - 상관 관계 분석
  - 시각화 도구: Matplotlib, Seaborn

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 로드
data = pd.read_csv('data.csv')

# 결측값 처리
data = data.dropna()

# 데이터 분포 확인
print(data.describe())

# 상관 관계 분석
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()
```

##### 3. 기본적인 기계 학습 모델 구현
- **선형 회귀 (Linear Regression)**
  - 목적: 연속형 변수 예측
  - 예제: 주택 가격 예측

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 데이터 로드 및 전처리
data = pd.read_csv('housing.csv')
X = data[['square_feet', 'num_rooms']]
y = data['price']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
```

- **로지스틱 회귀 (Logistic Regression)**
  - 목적: 이진 분류
  - 예제: 이메일 스팸 분류

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 데이터 로드 및 전처리
data = pd.read_csv('emails.csv')
X = data[['feature1', 'feature2']]
y = data['is_spam']

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

#### 과제

1. **데이터 전처리 및 탐색적 데이터 분석**
   - 주어진 데이터셋을 로드하고 결측값을 처리합니다.
   - 데이터의 분포를 확인하고, 상관 관계를 분석합니다.

2. **선형 회귀 모델 구현**
   - 주어진 데이터셋을 사용하여 선형 회귀 모델을 학습하고, 예측 및 평가를 수행합니다.

3. **로지스틱 회귀 모델 구현**
   - 주어진 데이터셋을 사용하여 로지스틱 회귀 모델을 학습하고, 예측 및 평가를 수행합니다.

#### 퀴즈

1. **기계 학습의 주요 유형이 아닌 것은?**
   - A) 지도 학습
   - B) 비지도 학습
   - C) 강화 학습
   - D) 감시 학습

2. **탐색적 데이터 분석(EDA)의 주요 목적은 무엇인가?**
   - A) 데이터의 구조와 패턴을 파악하기 위해
   - B) 모델을 학습하기 위해
   - C) 데이터의 차원을 줄이기 위해
   - D) 데이터의 이상값을 제거하기 위해

3. **선형 회귀의 주요 목적은 무엇인가?**
   - A) 범주형 변수 예측
   - B) 연속형 변수 예측
   - C) 데이터의 차원 축소
   - D) 데이터 클러스터링

4. **로지스틱 회귀의 주요 목적은 무엇인가?**
   - A) 연속형 변수 예측
   - B) 이진 분류
   - C) 데이터 클러스터링
   - D) 데이터 전처리

#### 퀴즈 해설

1. **기계 학습의 주요 유형이 아닌 것은?**
   - **정답: D) 감시 학습**
     - 해설: 기계 학습의 주요 유형에는 지도 학습, 비지도 학습, 강화 학습이 있으며, 감시 학습은 포함되지 않습니다.

2. **탐색적 데이터 분석(EDA)의 주요 목적은 무엇인가?**
   - **정답: A) 데이터의 구조와 패턴을 파악하기 위해**
     - 해설: EDA는 데이터의 구조와 패턴을 파악하여 이후 모델링 과정에서 유용한 인사이트를 얻기 위해 수행됩니다.

3. **선형 회귀의 주요 목적은 무엇인가?**
   - **정답: B) 연속형 변수 예측**
     - 해설: 선형 회귀는 연속형 변수를 예측하는 데 사용됩니다.

4. **로지스틱 회귀의 주요 목적은 무엇인가?**
   - **정답: B) 이진 분류**
     - 해설: 로지스틱 회귀는 이진 분류 문제를 해결하는 데 사용됩니다.

다음 주차 강의 내용을 요청하시면, 38주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.