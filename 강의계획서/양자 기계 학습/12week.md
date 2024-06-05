### 12주차 강의 상세 계획: 양자 기계 학습 실습 프로젝트 1 - 프로젝트 소개 및 데이터 준비

#### 강의 목표
- 실습 프로젝트 소개 및 데이터 준비
- 양자 기계 학습 실습 프로젝트 데이터 전처리 및 특징 추출

#### 강의 구성
- **프로젝트 소개**: 30분
- **데이터 전처리 실습**: 1시간 30분

---

### 1. 프로젝트 소개 (30분)

#### 1.1 실습 프로젝트 개요
- **프로젝트 주제**: 예시 - 금융 데이터 분석을 통한 주식 시장 예측.
- **목표**: 양자 기계 학습 모델을 사용하여 주식 시장 데이터를 분석하고 예측.
- **데이터셋**: 주식 시장 데이터셋 (예: Yahoo Finance 데이터셋).

#### 1.2 프로젝트 계획
- **주차별 목표**:
  - 12주차: 데이터 전처리 및 특징 추출.
  - 13주차: 모델 학습 및 하이퍼파라미터 튜닝.
  - 14주차: 모델 평가 및 최종 모델 선택.
  - 15주차: 프로젝트 발표 준비.
  - 16주차: 프로젝트 발표 및 피드백.

#### 1.3 데이터셋 소개
- **데이터셋 구성**: 주식 가격, 거래량, 기술적 지표 등.
- **데이터셋 출처**: Yahoo Finance, Kaggle 등.

---

### 2. 데이터 전처리 실습 (1시간 30분)

#### 2.1 데이터 로드 및 확인

##### 필요 라이브러리 설치
```bash
pip install pandas yfinance scikit-learn
```

##### 데이터 로드 및 확인 코드 (Python)
```python
import pandas as pd
import yfinance as yf

# 데이터 로드
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2021-01-01')
print(data.head())

# 데이터 요약
print(data.info())
print(data.describe())
```

#### 2.2 데이터 전처리 및 특징 추출

##### 데이터 전처리 및 특징 추출 코드 (Python)
```python
from sklearn.preprocessing import StandardScaler

# 결측값 처리
data = data.dropna()

# 특징 추출 (예: 종가와 거래량)
features = data[['Close', 'Volume']]

# 데이터 정규화
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 데이터 분할
train_size = int(len(features_scaled) * 0.8)
train_data = features_scaled[:train_size]
test_data = features_scaled[train_size:]

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
