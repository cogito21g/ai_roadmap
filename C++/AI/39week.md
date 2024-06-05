### 39주차: 딥 러닝 (Deep Learning) 기초

#### 강의 목표
- 딥 러닝의 기본 개념 이해
- 인공 신경망(Artificial Neural Networks) 구조 이해 및 구현
- 기본적인 딥 러닝 모델 구현 및 훈련

#### 강의 내용

##### 1. 딥 러닝의 기본 개념
- **딥 러닝 개요**
  - 정의: 여러 층의 인공 신경망을 통해 데이터를 학습하는 기계 학습의 한 분야
  - 주요 용어: 뉴런, 활성화 함수, 가중치, 편향, 층(레이어)

- **인공 신경망(ANN) 구조**
  - 입력층(Input Layer)
  - 은닉층(Hidden Layer)
  - 출력층(Output Layer)
  - 활성화 함수(Activation Function): ReLU, Sigmoid, Tanh 등

##### 2. 기본적인 인공 신경망 구현


#### 2. 기본적인 인공 신경망 구현

- **인공 신경망 구현 (TensorFlow와 Keras 사용)**

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터 준비 (예제: MNIST 손글씨 데이터셋)
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# 모델 생성
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(X_train.reshape(-1, 784), y_train, epochs=5)

# 모델 평가
loss, accuracy = model.evaluate(X_test.reshape(-1, 784), y_test)
print(f'Accuracy: {accuracy}')
```

##### 3. 기본적인 딥 러닝 모델 훈련 및 평가
- **데이터 전처리**
  - 데이터 정규화 및 전처리
  - 학습용 데이터와 검증용 데이터 분리

```python
from sklearn.model_selection import train_test_split

# 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
```

- **모델 훈련**
  - 학습 과정 모니터링을 위한 콜백(callback) 설정
  - 예시: 조기 종료(Early Stopping)

```python
from tensorflow.keras.callbacks import EarlyStopping

# 조기 종료 콜백
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# 모델 훈련
model.fit(X_train.reshape(-1, 784), y_train, epochs=50, validation_data=(X_val.reshape(-1, 784), y_val), callbacks=[early_stopping])
```

- **모델 평가 및 예측**
  - 테스트 데이터셋을 사용하여 모델 성능 평가

```python
# 모델 평가
loss, accuracy = model.evaluate(X_test.reshape(-1, 784), y_test)
print(f'Test Accuracy: {accuracy}')

# 예측
predictions = model.predict(X_test.reshape(-1, 784))
predicted_labels = predictions.argmax(axis=1)
```

#### 과제

1. **인공 신경망 구현**
   - TensorFlow와 Keras를 사용하여 기본적인 인공 신경망을 구현하고, 주어진 데이터셋에 대해 모델을 훈련합니다.

2. **데이터 전처리 및 모델 훈련**
   - 주어진 데이터셋을 정규화하고, 학습용 데이터와 검증용 데이터를 분리합니다.
   - 모델을 훈련하고, 조기 종료 콜백을 사용하여 학습 과정을 모니터링합니다.

3. **모델 평가 및 예측**
   - 테스트 데이터셋을 사용하여 모델의 성능을 평가합니다.
   - 모델을 사용하여 예측을 수행하고, 예측 결과를 출력합니다.

#### 퀴즈

1. **딥 러닝의 주요 용어가 아닌 것은?**
   - A) 뉴런
   - B) 활성화 함수
   - C) 가중치
   - D) 회귀 계수

2. **인공 신경망에서 사용되는 활성화 함수가 아닌 것은?**
   - A) ReLU
   - B) Sigmoid
   - C) Tanh
   - D) 선형 회귀

3. **TensorFlow와 Keras를 사용하여 모델을 훈련할 때 사용하는 메서드는?**
   - A) fit
   - B) train
   - C) run
   - D) execute

4. **조기 종료 콜백의 주요 목적은 무엇인가?**
   - A) 모델의 학습 속도를 높이기 위해
   - B) 모델이 과적합되는 것을 방지하기 위해
   - C) 모델의 예측 정확도를 높이기 위해
   - D) 모델의 가중치를 초기화하기 위해

#### 퀴즈 해설

1. **딥 러닝의 주요 용어가 아닌 것은?**
   - **정답: D) 회귀 계수**
     - 해설: 딥 러닝의 주요 용어에는 뉴런, 활성화 함수, 가중치 등이 있으며, 회귀 계수는 선형 회귀에서 주로 사용되는 용어입니다.

2. **인공 신경망에서 사용되는 활성화 함수가 아닌 것은?**
   - **정답: D) 선형 회귀**
     - 해설: ReLU, Sigmoid, Tanh는 활성화 함수로 사용되지만, 선형 회귀는 활성화 함수가 아닙니다.

3. **TensorFlow와 Keras를 사용하여 모델을 훈련할 때 사용하는 메서드는?**
   - **정답: A) fit**
     - 해설: TensorFlow와 Keras에서는 모델을 훈련할 때 `fit` 메서드를 사용합니다.

4. **조기 종료 콜백의 주요 목적은 무엇인가?**
   - **정답: B) 모델이 과적합되는 것을 방지하기 위해**
     - 해설: 조기 종료 콜백은 모델이 과적합되는 것을 방지하기 위해 사용됩니다.

다음 주차 강의 내용을 요청하시면, 40주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.