### 38주차: 기계 학습 심화

#### 강의 목표
- 기계 학습 모델 평가 및 성능 향상 기법 이해
- 고급 기계 학습 알고리즘 이해 및 구현
- 기계 학습 프로젝트 관리 및 배포

#### 강의 내용

##### 1. 기계 학습 모델 평가 및 성능 향상 기법
- **모델 평가 지표**
  - 정확도 (Accuracy), 정밀도 (Precision), 재현율 (Recall), F1 점수 (F1 Score), ROC-AUC

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 모델 예측
y_pred = model.predict(X_test)

# 평가 지표 계산
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC-AUC: {roc_auc}')
```

- **교차 검증 (Cross-Validation)**
  - K-겹 교차 검증 (K-Fold Cross-Validation)

```python
from sklearn.model_selection import cross_val_score

# K-겹 교차 검증
scores = cross_val_score(model, X, y, cv=5)

print(f'Cross-Validation Scores: {scores}')
print(f'Mean Score: {scores.mean()}')
```

- **하이퍼파라미터 튜닝 (Hyperparameter Tuning)**
  - 그리드 서치 (Grid Search), 랜덤 서치 (Random Search)

```python
from sklearn.model_selection import GridSearchCV

# 하이퍼파라미터 그리드 정의
param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}

# 그리드 서치
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score: {grid_search.best_score_}')
```

##### 2. 고급 기계 학습 알고리즘
- **의사결정 나무 (Decision Tree)**
  - 설명: 결정 규칙을 기반으로 데이터를 분할하여 예측
  - 예제 구현:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 모델 학습
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

- **랜덤 포레스트 (Random Forest)**
  - 설명: 여러 개의 의사결정 나무를 앙상블하여 예측 성능 향상
  - 예제 구현:

```python
from sklearn.ensemble import RandomForestClassifier

# 모델 학습
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

- **서포트 벡터 머신 (SVM)**
  - 설명: 데이터의 경계를 최적화하여 분류
  - 예제 구현:

```python
from sklearn.svm import SVC

# 모델 학습
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

##### 3. 기계 학습 프로젝트 관리 및 배포
- **모델 저장 및 불러오기**
  - 모델 저장: `joblib` 또는 `pickle` 사용
  - 모델 불러오기

```python
import joblib

# 모델 저장
joblib.dump(model, 'model.pkl')

# 모델 불러오기
loaded_model = joblib.load('model.pkl')
```

- **기계 학습 모델 배포**
  - Flask를 사용하여 RESTful API 서버 구현
  - 예제:

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# 모델 로드
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
```

#### 과제

1. **모델 평가 및 성능 향상**
   - 주어진 데이터셋을 사용하여 모델을 학습하고, 다양한 평가 지표를 계산합니다.
   - 교차 검증을 통해 모델의 성능을 평가하고, 하이퍼파라미터 튜닝을 수행합니다.

2. **고급 기계 학습 알고리즘 구현**
   - 의사결정 나무, 랜덤 포레스트, 서포트 벡터 머신을 사용하여 모델을 학습하고, 예측 및 평가를 수행합니다.

3. **모델 저장 및 배포**
   - 학습된 모델을 저장하고, Flask를 사용하여 RESTful API 서버를 구현합니다.
   - API 서버를 통해 모델 예측을 수행하는 엔드포인트를 만듭니다.

#### 퀴즈

1. **기계 학습 모델의 성능을 평가하기 위한 지표가 아닌 것은?**
   - A) 정확도 (Accuracy)
   - B) 정밀도 (Precision)
   - C) 재현율 (Recall)
   - D) 이익률 (Profit Margin)

2. **K-겹 교차 검증의 주요 목적은 무엇인가?**
   - A) 데이터의 차원을 축소하기 위해
   - B) 모델의 하이퍼파라미터를 튜닝하기 위해
   - C) 모델의 일반화 성능을 평가하기 위해
   - D) 모델의 학습 속도를 높이기 위해

3. **랜덤 포레스트 모델의 주요 특징은 무엇인가?**
   - A) 하나의 의사결정 나무를 사용하여 예측
   - B) 여러 개의 의사결정 나무를 앙상블하여 예측
   - C) 데이터의 경계를 최적화하여 분류
   - D) 주성분을 사용하여 데이터 차원을 축소

4. **Flask를 사용하여 기계 학습 모델을 배포할 때 적절한 방법은?**
   - A) HTML 파일을 반환하는 엔드포인트를 만들기
   - B) 모델을 로드하고 예측 결과를 반환하는 RESTful API 엔드포인트를 만들기
   - C) 데이터베이스 연결을 설정하는 엔드포인트를 만들기
   - D) 정적 파일을 제공하는 엔드포인트를 만들기

#### 퀴즈 해설

1. **기계 학습 모델의 성능을 평가하기 위한 지표가 아닌 것은?**
   - **정답: D) 이익률 (Profit Margin)**
     - 해설: 이익률은 비즈니스 성과 지표로, 기계 학습 모델의 성능을 평가하는 지표는 아닙니다. 성능 평가는 정확도, 정밀도, 재현율, F1 점수, ROC-AUC 등을 사용합니다.

2. **K-겹 교차 검증의 주요 목적은 무엇인가?**
   - **정답: C) 모델의 일반화 성능을 평가하기 위해**
     - 해설: K-겹 교차 검증은 모델이 새로운 데이터에 대해 얼마나 잘 일반화되는지 평가하기 위해 사용됩니다.

3. **랜덤 포레스트 모델의 주요 특징은 무엇인가?**
   - **정답: B) 여러 개의 의사결정 나무를 앙상블하여 예측**
     - 해설: 랜덤 포레스트는 여러 개의 의사결정 나무를 앙상블하여 예측 성능을 향상시키는 모델입니다.

4. **Flask를 사용하여 기계 학습 모델을 배포할 때 적절한 방법은?**
   - **정답: B) 모델을 로드하고 예측 결과를 반환하는 RESTful API 엔드포인트를 만들기**
     - 해설: Flask를 사용하여 기계 학습 모델을 배포할 때는 모델을 로드하고 예측 결과를 반환하는 RESTful API 엔드포인트를 만드는 것이 적절한 방법입니다.

다음 주차 강의 내용을 요청하시면, 39주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.