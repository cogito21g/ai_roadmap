### 10주차 강의 상세 계획: AutoML 실습 3

#### 강의 목표
- 모델 평가 및 최종 모델 선택
- AutoML 모델 성능 비교 및 분석

#### 강의 구성
- **모델 평가**: 1시간
- **최종 모델 선택 및 분석**: 1시간

---

### 1. 모델 평가 (1시간)

#### 1.1 모델 평가 지표

##### 모델 평가 지표의 중요성
- **정의**: 모델의 성능을 평가하기 위한 다양한 지표.
- **목적**: 모델의 강점과 약점을 파악하고, 최적의 모델 선택.

##### 주요 평가 지표
- **정확도 (Accuracy)**: 전체 예측 중 올바르게 예측한 비율.
- **정밀도 (Precision)**: 양성 예측 중 실제 양성의 비율.
- **재현율 (Recall)**: 실제 양성 중 올바르게 예측한 비율.
- **F1 점수**: 정밀도와 재현율의 조화 평균.
- **AUC-ROC**: 분류 모델의 성능을 평가하는 곡선 아래 면적.

#### 1.2 모델 평가 코드 (Python)

##### 필요 라이브러리 설치
```bash
pip install scikit-learn
```

##### 모델 평가 코드 (Python)
```python
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Auto-sklearn 모델 평가
y_pred_proba = automl.predict_proba(X_test_tfidf)[:, 1]
print("Auto-sklearn Classification Report:")
print(classification_report(y_test, y_pred))
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Auto-sklearn AUC-ROC: {auc:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'Auto-sklearn (AUC = {auc:.4f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

# TPOT 모델 평가
y_pred_proba = tpot.predict_proba(X_test_tfidf)[:, 1]
print("TPOT Classification Report:")
print(classification_report(y_test, y_pred))
auc = roc_auc_score(y_test, y_pred_proba)
print(f"TPOT AUC-ROC: {auc:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'TPOT (AUC = {auc:.4f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()
```

---

### 2. 최종 모델 선택 및 분석 (1시간)

#### 2.1 모델 성능 비교

##### Auto-sklearn vs TPOT
- **성능 비교**: 정확도, 정밀도, 재현율, F1 점수, AUC-ROC 등을 비교.
- **모델 강점 및 약점 분석**: 각 모델의 성능과 한계 파악.

#### 2.2 최종 모델 선택

##### 최적 모델 선택 기준
- **성능 지표**: 다양한 성능 지표를 종합하여 최적의 모델 선택.
- **사용 용이성**: 모델의 사용 및 해석 용이성 고려.
- **계산 자원 및 시간**: 모델 훈련 및 예측에 소요되는 자원과 시간 고려.

##### 최종 모델 선택 및 분석 코드 (Python)
```python
# 최종 모델 선택
if auc_sklearn > auc_tpot:
    final_model = automl
    final_model_name = "Auto-sklearn"
else:
    final_model = tpot
    final_model_name = "TPOT"

# 최종 모델 분석
print(f"최종 선택된 모델: {final_model_name}")
print(final_model)
```

### 준비 자료
- **강의 자료**: 모델 평가 지표 및 최종 모델 선택 슬라이드 (PDF)
- **참고 코드**: 모델 평가 및 최종 모델 선택 예제 코드 (Python)

### 과제
- **모델 평가 및 비교**: 제공된 코드 예제를 실행하고, Auto-s

klearn 및 TPOT 모델의 성능을 비교.
- **최종 모델 선택 및 분석**: 최종 모델을 선택하고, 선택 이유를 요약.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 모델 평가 지표의 중요성을 이해하고, 다양한 평가 지표를 사용해 모델 성능을 평가하며, 최적의 모델을 선택하는 경험을 쌓을 수 있도록 유도합니다.

---
