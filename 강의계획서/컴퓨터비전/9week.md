### 9주차 강의 상세 계획: 비디오 처리

#### 강의 목표
- 비디오 처리의 기본 개념과 주요 기술 이해
- OpenCV를 이용한 비디오 프레임 추출 및 동작 인식 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 비디오 처리의 기본 개념 (20분)

##### 비디오 처리란?
- **정의**: 비디오 처리란 연속된 프레임으로 구성된 비디오 데이터를 분석하고 조작하는 기술.
- **주요 작업**: 비디오 데이터의 프레임 추출, 동작 인식, 객체 추적 등.

##### 비디오 처리의 주요 응용 분야
- **보안**: CCTV 영상 분석을 통한 이상 행위 감지.
- **의료**: 의료 영상 분석을 통한 진단 및 수술 보조.
- **스포츠**: 스포츠 경기 분석을 통한 선수 성과 평가.
- **자율 주행**: 차량 주변 환경 인식을 통한 자율 주행 구현.

#### 1.2 비디오 프레임 추출 (20분)

##### 프레임 추출이란?
- **정의**: 비디오 데이터를 구성하는 개별 이미지를 추출하는 과정.
- **기술적 고려 사항**:
  - **프레임 속도**: 초당 프레임 수 (FPS, Frames Per Second).
  - **해상도**: 각 프레임의 픽셀 수.
  - **압축**: 비디오 코덱을 통해 압축된 데이터를 디코딩.

#### 1.3 동작 인식 (20분)

##### 동작 인식이란?
- **정의**: 비디오에서 사람이나 객체의 동작을 분석하고 인식하는 기술.
- **주요 기술**:
  - **광학 흐름 (Optical Flow)**: 연속된 프레임 간의 픽셀 이동을 추적하여 동작을 인식.
  - **움직임 기반 세분화 (Motion Segmentation)**: 움직이는 객체와 배경을 분리.
  - **딥러닝 기반 방법**: RNN, LSTM, 3D-CNN을 사용하여 동작 인식.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 OpenCV를 이용한 비디오 프레임 추출 및 동작 인식

##### 필요 라이브러리 설치
```bash
pip install opencv-python matplotlib
```

##### 비디오 프레임 추출 코드 (Python 3.10 및 OpenCV)
```python
import cv2
import matplotlib.pyplot as plt

# 비디오 파일 경로
video_path = 'path_to_your_video.mp4'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 프레임 추출 및 저장
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    # 프레임 저장 (예: 30번째마다 저장)
    if frame_count % 30 == 0:
        frame_path = f'frame_{frame_count}.jpg'
        cv2.imwrite(frame_path, frame)
        print(f'Saved: {frame_path}')
        # 프레임 시각화
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f'Frame {frame_count}')
        plt.show()

cap.release()
cv2.destroyAllWindows()
```

##### Optical Flow를 이용한 동작 인식 코드
```python
# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 첫 번째 프레임 읽기
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Optical Flow 파라미터 설정
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 특징점 검출기 생성
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# 동작 인식
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_points, None, **lk_params)

    # 좋은 점들만 선택
    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]

    # 움직임 그리기
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame = cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

    # 결과 시각화
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Optical Flow')
    plt.show()

    prev_gray = frame_gray.copy()
    prev_points = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
```

### 준비 자료
- **강의 자료**: 비디오 처리 개요 및 동작 인식 슬라이드 (PDF)
- **참고 코드**: 비디오 프레임 추출 및 Optical Flow 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 비디오 처리의 기본 개념과 주요 기술 정리.
- **코드 실습**: 비디오 프레임 추출 및 Optical Flow 구현 코드 실행 및 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 비디오 처리의 기본 개념과 주요 기술을 이해하고, OpenCV를 이용하여 비디오 프레임을 추출하고 동작을 인식하는 경험을 쌓을 수 있도록 유도합니다.