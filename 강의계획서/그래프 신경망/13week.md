### 13주차 강의 상세 계획: GNN 실습 프로젝트 3 - 모델 평가 및 최종 모델 선택

#### 강의 목표
- 모델의 성능 평가 방법과 최종 모델 선택 과정 이해
- 다양한 평가 지표를 활용하여 모델을 비교하고 최적의 모델 선택

#### 강의 구성
- **모델 평가**: 1시간
- **최종 모델 선택**: 1시간

---

### 1. 모델 평가 (1시간)

#### 1.1 모델 평가 지표

##### 주요 평가 지표
- **정확도 (Accuracy)**: 전체 예측 중 올바르게 예측한 비율.
- **정밀도 (Precision)**: 양성 예측 중 실제 양성의 비율.
- **재현율 (Recall)**: 실제 양성 중 올바르게 예측한 비율.
- **F1 점수**: 정밀도와 재현율의 조화 평균.
- **AUC-ROC**: 분류 모델의 성능을 평가하는 곡선 아래 면적.

#### 1.2 모델 평가 코드 (Python)

##### 필요 라이브러리 설치
```bash
pip install scikit-learn matplotlib
```

##### 모델 평가 코드 (Python)
```python
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 모델 평가 함수
def evaluate_model(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    y_pred_proba = []
    for data in loader:
        data = data.to(device)
        out = model(data)
        _, pred = out.max(dim=1)
        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_pred_proba.extend(out.softmax(dim=1).cpu().numpy()[:, 1])
    return y_true, y_pred, y_pred_proba

# 모델 평가 실행
y_true, y_pred, y_pred_proba = evaluate_model(model, test_loader)

# 평가 지표 출력
print("Classification Report:")
print(classification_report(y_true, y_pred))

# ROC Curve 및 AUC
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
auc = roc_auc_score(y_true, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()
```

---

### 2. 최종 모델 선택 (1시간)

#### 2.1 최종 모델 선택 기준

##### 모델 성능 비교
- **성능 지표**: 다양한 성능 지표를 종합하여 최적의 모델 선택.
- **사용 용이성**: 모델의 사용 및 해석 용이성 고려.
- **계산 자원 및 시간**: 모델 훈련 및 예측에 소요되는 자원과 시간 고려.

#### 2.2 최종 모델 선택 및 분석 코드 (Python)
```python
# 최종 모델 선택
if auc > 0.8:  # 예시 기준
    final_model = model
    final_model_name = "GCN"

# 최종 모델 분석
print(f"최종 선택된 모델: {final_model_name}")
print(final_model)
```

### 준비 자료
- **강의 자료**: 모델 평가 지표 및 최종 모델 선택 슬라이드 (PDF)
- **참고 코드**: 모델 평가 및 최종 모델 선택 예제 코드 (Python)

### 과제
- **모델 평가 및 비교**: 제공된 코드 예제를 실행하고, 다양한 평가 지표를 사용해 모델 성능을 비교.
- **최종 모델 선택 및 분석**: 최종 모델을 선택하고, 선택 이유를 요약.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 모델 평가 지표의 중요성을 이해하고, 다양한 평가 지표를 사용해 모델 성능을 평가하며, 최적의 모델을 선택하는 경험을 쌓을 수 있도록 유도합니다.

---

