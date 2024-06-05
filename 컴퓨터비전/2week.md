### 2. 코드 구현 실습 (1시간)

#### 2.1 OpenCV를 이용한 이미지 처리 실습

##### 필요 라이브러리 설치
```bash
pip install opencv-python matplotlib
```

##### 이미지 필터링 코드 (Python 3.10)
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# 평균 필터 적용
average_filter = cv2.blur(image, (5, 5))

# 가우시안 필터 적용
gaussian_filter = cv2.GaussianBlur(image, (5, 5), 0)

# 샤프닝 필터 적용
kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
sharpen_filter = cv2.filter2D(image, -1, kernel)

# 결과 시각화
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(2, 2, 2), plt.imshow(average_filter, cmap='gray'), plt.title('Average Filter')
plt.subplot(2, 2, 3), plt.imshow(gaussian_filter, cmap='gray'), plt.title('Gaussian Filter')
plt.subplot(2, 2, 4), plt.imshow(shar

pen_filter, cmap='gray'), plt.title('Sharpen Filter')
plt.show()
```

##### 엣지 검출 코드
```python
# 소벨 필터 적용
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# 캐니 엣지 검출기 적용
canny_edges = cv2.Canny(image, 100, 200)

# 결과 시각화
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(2, 2, 2), plt.imshow(sobel_x, cmap='gray'), plt.title('Sobel X')
plt.subplot(2, 2, 3), plt.imshow(sobel_y, cmap='gray'), plt.title('Sobel Y')
plt.subplot(2, 2, 4), plt.imshow(canny_edges, cmap='gray'), plt.title('Canny Edges')
plt.show()
```

##### 히스토그램 평활화 코드
```python
# 히스토그램 평활화 적용
equalized_image = cv2.equalizeHist(image)

# 결과 시각화
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 2, 2), plt.imshow(equalized_image, cmap='gray'), plt.title('Equalized Image')
plt.show()
```

### 준비 자료
- **강의 자료**: 이미지 처리 기본 슬라이드 (PDF)
- **예제 이미지**: 실습에 사용할 이미지 파일 (example.jpg)

### 과제
- **이론 정리**: 이미지 필터링, 엣지 검출, 히스토그램 평활화의 원리와 응용 사례 정리.
- **코드 실습**: OpenCV를 이용한 이미지 처리 실습 코드 실행 및 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 이미지 처리의 기본 개념을 이해하고, OpenCV를 활용하여 실제 이미지 처리 기술을 실습할 수 있도록 유도합니다.