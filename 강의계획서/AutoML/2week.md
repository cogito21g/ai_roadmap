### 2주차 강의 상세 계획: 데이터 준비

#### 강의 목표
- 데이터 전처리의 중요성과 주요 기법 이해
- 특징 추출과 데이터 분할 방법 학습
- AutoML을 위한 데이터 준비 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 데이터 전처리의 중요성 (20분)

##### 데이터 전처리란?
- **정의**: 머신 러닝 모델을 훈련시키기 전에 데이터를 정리하고 변환하는 과정.
- **목적**: 데이터의 품질을 높이고 모델 성능을 향상시키기 위해.

##### 데이터 전처리의 주요 단계
- **결측값 처리**: 결측값 대체 또는 제거.
- **정규화**: 데이터의 스케일을 조정.
- **범주형 데이터 인코딩**: 범주형 데이터를 숫자형으로 변환.

#### 1.2 특징 추출과 데이터 분할 (40분)

##### 특징 추출
- **정의**: 원본 데이터에서 유용한 정보를 추출하는 과정.
- **기법**: PCA, LDA, 특징 선택, 특징 생성.

##### 데이터 분할
- **훈련/검증/테스트 분할**: 데이터셋을 훈련, 검증, 테스트 셋으로 나누는 방법.
- **교차 검증**: 데이터셋을 여러 부분으로 나누어 여러 번 훈련하고 검증.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 데이터 준비 실습

##### 필요 라이브러리 설치
```bash
pip install pandas scikit-learn
```

##### 데이터 전처리 및 특징 추출 구현 코드 (Python)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 예제 데이터 로드
data = pd.DataFrame({
    'age': [25, 30, 35, 40, None, 50, 55],
    'salary': [50000, 60000, None, 80000, 90000, None, 150000],
    'gender': ['male', 'female', 'female', 'male', 'male', 'female', None],
    'purchased': ['no', 'yes', 'no', 'no', 'yes', 'yes', 'no']
})

# 결측값 처리 및 범주형 데이터 인코딩
numeric_features = ['age', 'salary']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_features = ['gender']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# 데이터 전처리 파이프라인
X = data.drop('purchased', axis=1)
y = data['purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

print("Processed training data:\n", X_train)
print("Processed testing data:\n", X_test)
```

### 준비 자료
- **강의 자료**: 데이터 전처리, 특징 추출, 데이터 분할 슬라이드 (PDF)
- **참고 코드**: 데이터 전처리 및 특징 추출 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 데이터 전처리의 중요성과 주요 기법, 특징 추출 및 데이터 분할 방법 요약.
- **코드 실습**: 제공된 데이터 전처리 코드를 실행하고, 다른 데이터셋에 적용.
- **과제 제출**: 다음 주차 강

의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 데이터 전처리의 중요성과 주요 기법을 이해하고, 특징 추출과 데이터 분할 방법을 학습하며, 실제 데이터를 사용해 데이터 준비 과정을 실습할 수 있도록 유도합니다.

---
