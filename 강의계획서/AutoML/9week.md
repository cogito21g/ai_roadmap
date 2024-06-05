### 9주차 강의 상세 계획: AutoML 실습 2

#### 강의 목표
- 모델 학습 및 하이퍼파라미터 튜닝
- AutoML 도구를 사용하여 최적의 모델 찾기

#### 강의 구성
- **모델 학습**: 1시간
- **하이퍼파라미터 튜닝**: 1시간

---

### 1. 모델 학습 (1시간)

#### 1.1 Auto-sklearn을 사용한 모델 학습

##### 필요 라이브러리 설치
```bash
pip install auto-sklearn
```

##### Auto-sklearn을 사용한 모델 학습 코드 (Python)
```python
import autosklearn.classification
from sklearn.metrics import accuracy_score

# Auto-sklearn 분류기 설정
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60, per_run_time_limit=30)

# 모델 학습
automl.fit(X_train_tfidf, y_train)

# 예측
y_pred = automl.predict(X_test_tfidf)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"Auto-sklearn Accuracy: {accuracy:.4f}")

# 최적 모델 출력
print(automl.show_models())
```

#### 1.2 TPOT을 사용한 모델 학습

##### 필요 라이브러리 설치
```bash
pip install tpot
```

##### TPOT을 사용한 모델 학습 코드 (Python)
```python
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score

# TPOT 분류기 설정
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)

# 모델 학습
tpot.fit(X_train_tfidf, y_train)

# 예측
y_pred = tpot.predict(X_test_tfidf)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"TPOT Accuracy: {accuracy:.4f}")

# 최적 파이프라인 출력 및 저장
tpot.export('tpot_sentiment_analysis_pipeline.py')
```

---

### 2. 하이퍼파라미터 튜닝 (1시간)

#### 2.1 Auto-sklearn 하이퍼파라미터 튜닝

##### Auto-sklearn 하이퍼파라미터 튜닝 코드 (Python)
```python
# 하이퍼파라미터 튜닝 설정
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,  # 전체 작업 시간
    per_run_time_limit=30,  # 개별 모델 훈련 시간
    initial_configurations_via_metalearning=0,  # 메타 러닝 비활성화
    ensemble_size=50  # 앙상블 크기
)

# 모델 학습
automl.fit(X_train_tfidf, y_train)

# 예측
y_pred = automl.predict(X_test_tfidf)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"Auto-sklearn (Tuned) Accuracy: {accuracy:.4f}")

# 최적 모델 출력
print(automl.show_models())
```

#### 2.2 TPOT 하이퍼파라미터 튜닝

##### TPOT 하이퍼파라미터 튜닝 코드 (Python)
```python
# 하이퍼파라미터 튜닝 설정
tpot = TPOTClassifier(
    generations=10,  # 유전자 알고리즘 세대 수 증가
    population_size=50,  # 인구 크기 증가
    verbosity=3
)

# 모델 학습
tpot.fit(X_train_tfidf, y_train)

# 예측
y_pred = tpot.predict(X_test_tfidf)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"TPOT (Tuned) Accuracy: {accuracy:.4f}")

# 최적 파이프라인 출력 및 저장
tpot.export('tpot_sentiment_analysis_pipeline_tuned.py')
```

### 준비 자료
- **강의 자료**: 모델 학습 및 하이퍼파라미터 튜닝 슬라이드 (PDF)
- **참고 코드**: Auto-sklearn 및 TPOT 하이퍼파라미터 튜닝 예제 코드 (Python)

### 과제
- **모델 학습 및 평가**: 제공된 코드 예제를 실행하고, 모델 학습 및 평가 결과 요약.
- **하이퍼파라미터 튜닝**: 하이퍼파라미터 튜닝을 통해 모델 성능을 최적화하고, 결과 요약.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 모델 학습과 하이퍼파라미터 튜닝의 중요성을 이해하고, AutoML 도구를 사용하여 최적의 모델을 찾는 경험을 쌓을 수 있도록 유도합니다.

---
