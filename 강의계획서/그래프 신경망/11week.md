
### 11주차 강의 상세 계획: GNN 실습 프로젝트 1

#### 강의 목표
- 실습 프로젝트 소개 및 데이터 준비
- GNN 실습 프로젝트 데이터 전처리 및 특징 추출

#### 강의 구성
- **프로젝트 소개**: 30분
- **데이터 전처리 실습**: 1시간 30분

---

### 1. 프로젝트 소개 (30분)

#### 1.1 실습 프로젝트 개요
- **프로젝트 주제**: 예시 - 소셜 미디어 감정 분석.
- **목표**: GNN을 사용하여 소셜 미디어 데이터의 감정 분석 모델 개발.
- **데이터셋**: 소셜 미디어 댓글 데이터셋.

#### 1.2 프로젝트 계획
- **주차별 목표**:
  - 11주차: 데이터 전처리 및 특징 추출.
  - 12주차: 모델 학습 및 튜닝.
  - 13주차: 모델 평가 및 최종 모델 선택.
  - 14주차: 프로젝트 발표 준비.
  - 15주차: 프로젝트 발표 및 피드백.

#### 1.3 데이터셋 소개
- **데이터셋 구성**: 댓글 텍스트, 감정 레이블(긍정, 부정, 중립).
- **데이터셋 출처**: Kaggle, 데이터 공개 저장소 등.

---

### 2. 데이터 전처리 실습 (1시간 30분)

#### 2.1 데이터 로드 및 확인

##### 필요 라이브러리 설치
```bash
pip install pandas scikit-learn nltk torch-geometric
```

##### 데이터 로드 및 확인 코드 (Python)
```python
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split

# NLTK 데이터 다운로드
nltk.download('punkt')

# 데이터 로드
data = pd.read_csv('social_media_comments.csv')
print(data.head())

# 데이터 요약
print(data.info())
print(data.describe())
```

#### 2.2 데이터 전처리 및 특징 추출

##### 데이터 전처리 및 특징 추출 코드 (Python)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import Data

# 결측값 처리
data = data.dropna()

# 텍스트 토큰화 및 TF-IDF 벡터화
vectorizer = TfidfVectorizer(stop_words='english', max_features=

1000)
X = vectorizer.fit_transform(data['comment']).toarray()

# 레이블 인코딩
y = data['sentiment'].factorize()[0]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 그래프 데이터 생성
train_data = Data(x=torch.tensor(X_train, dtype=torch.float), y=torch.tensor(y_train, dtype=torch.long))
test_data = Data(x=torch.tensor(X_test, dtype=torch.float), y=torch.tensor(y_test, dtype=torch.long))

print("Training Data:\n", train_data)
print("Testing Data:\n", test_data)
```

### 준비 자료
- **프로젝트 소개 자료**: 실습 프로젝트 계획 및 데이터셋 설명 슬라이드 (PDF)
- **참고 코드**: 데이터 전처리 및 특징 추출 예제 코드 (Python)

### 과제
- **데이터 전처리 및 특징 추출**: 제공된 코드 예제를 실행하고, 데이터 전처리 및 특징 추출 결과 요약.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 실습 프로젝트의 목표와 계획을 이해하고, 데이터 전처리 및 특징 추출 과정을 학습하며, 실제 데이터를 사용해 실습 프로젝트를 시작할 수 있도록 유도합니다.

---