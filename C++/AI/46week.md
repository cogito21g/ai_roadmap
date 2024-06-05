### 46주차: 컴퓨터 비전 (Computer Vision) 기초

#### 강의 목표
- 컴퓨터 비전의 기본 개념 이해
- 이미지 전처리 및 데이터 증강 기법 학습
- 기본적인 이미지 분류 모델 구현

#### 강의 내용

##### 1. 컴퓨터 비전의 기본 개념
- **컴퓨터 비전 개요**
  - 정의: 컴퓨터가 이미지나 비디오를 이해하고 해석할 수 있게 하는 학문
  - 주요 응용 분야: 이미지 분류, 객체 탐지, 이미지 생성, 영상 인식

- **이미지 데이터의 특징**
  - 픽셀 값, 채널, 해상도
  - 컬러 이미지와 흑백 이미지

##### 2. 이미지 전처리 및 데이터 증강
- **이미지 전처리**
  - 이미지 리사이징, 정규화, 그레이스케일 변환

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 이미지 로드
image = cv2.imread('path_to_your_image.jpg')

# 그레이스케일 변환
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 이미지 리사이징
resized_image = cv2.resize(image, (128, 128))

# 이미지 정규화
normalized_image = resized_image / 255.0

# 이미지 시각화
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.subplot(1, 3, 3)
plt.imshow(normalized_image)
plt.title('Normalized Image')
plt.show()
```

- **데이터 증강**
  - 회전, 이동, 반전, 노이즈 추가 등

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 증강 설정
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 예제 이미지 로드
image = np.expand_dims(resized_image, 0)

# 데이터 증강 시각화
i = 0
for batch in datagen.flow(image, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 4 == 0:
        break
plt.show()
```

##### 3. 기본적인 이미지 분류 모델 구현
- **CNN 모델**
  - 설명: 합성곱 신경망 (Convolutional Neural Network)을 사용하여 이미지의 특징을 추출하고 분류

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# CNN 모델 생성
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 데이터 로드 및 전처리 (예제: CIFAR-10 데이터셋)
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# 모델 훈련
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy}')
```

#### 과제

1. **이미지 전처리 및 데이터 증강**
   - 주어진 이미지 데이터를 리사이징, 정규화, 그레이스케일 변환하고, 다양한 데이터 증강 기법을 적용하여 시각화합니다.

2. **CNN 모델 구현**
   - 주어진 이미지 데이터셋을 사용하여 CNN 모델을 구현하고, 모델을 훈련 및 평가합니다.

3. **전이 학습 (Transfer Learning)**
   - 사전 학습된 모델 (예: VGG16, ResNet)을 사용하여 전이 학습을 수행하고, 주어진 이미지 데이터셋에 대해 모델을 훈련 및 평가합니다.

#### 퀴즈

1. **컴퓨터 비전의 주요 응용 분야가 아닌 것은?**
   - A) 이미지 분류
   - B) 객체 탐지
   - C) 자연어 처리
   - D) 영상 인식

2. **이미지 전처리의 주요 목적은 무엇인가?**
   - A) 데이터의 양을 늘리기 위해
   - B) 데이터의 품질을 향상시키고 모델의 성능을 높이기 위해
   - C) 데이터의 차원을 줄이기 위해
   - D) 데이터의 중요도를 계산하기 위해

3. **CNN 모델의 주요 구성 요소가 아닌 것은?**
   - A) 합성곱 층
   - B) 풀링 층
   - C) 순환 층
   - D) 완전 연결 층

4. **데이터 증강의 주요 목적은 무엇인가?**
   - A) 모델의 복잡성을 줄이기 위해
   - B) 데이터의 양을 늘리고 모델의 일반화 성능을 높이기 위해
   - C) 데이터의 차원을 줄이기 위해
   - D) 데이터의 중요도를 계산하기 위해

#### 퀴즈 해설

1. **컴퓨터 비전의 주요 응용 분야가 아닌 것은?**
   - **정답: C) 자연어 처리**
     - 해설: 컴퓨터 비전의 주요 응용 분야에는 이미지 분류, 객체 탐지, 영상 인식 등이 있으며, 자연어 처리는 NLP의 응용 분야입니다.

2. **이미지 전처리의 주요 목적은 무엇인가?**
   - **정답: B) 데이터의 품질을 향상시키고 모델의 성능을 높이기 위해**
     - 해설: 이미지 전처리는 데이터의 품질을 향상시키고 모델의 성능을 높이기 위해 수행됩니다.

3. **CNN 모델의 주요 구성 요소가 아닌 것은?**
   - **정답: C) 순환 층**
     - 해설: CNN 모델의 주요 구성 요소에는 합성곱 층, 풀링 층, 완전 연결 층이 있으며, 순환 층은 포함되지 않습니다.

4. **데이터 증강의 주요 목적은 무엇인가?**
   - **정답: B) 데이터의 양을 늘리고 모델의 일반화 성능을 높이기 위해**
     - 해설: 데이터 증강은 데이터의 양을 늘리고 모델의 일반화 성능을 높이기 위해 수행됩니다.

다음 주차 강의 내용을 요청하시면, 47주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.