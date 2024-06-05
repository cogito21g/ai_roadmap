### 4주차 강의 상세 계획: CNN 심화

#### 강의 목표
- 다양한 CNN 구조의 원리와 구현 방법 이해
- 유명 CNN 모델 (LeNet, AlexNet, VGG, ResNet) 리뷰 및 토론

#### 강의 구성
- **이론 강의**: 1시간
- **모델 리뷰 및 토론**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 LeNet (15분)
- **LeNet의 구조**:
  - Yann LeCun이 1998년에 제안한 최초의 CNN 중 하나.
  - 주로 손글씨 숫자 인식(MNIST 데이터셋)에서 사용.
- **구성 요소**:
  - 두 개의 컨볼루션 레이어 + 풀링 레이어.
  - 두 개의 완전 연결층.
- **특징**:
  - 간단한 구조, 낮은 연산 비용.

#### 1.2 AlexNet (15분)
- **AlexNet의 구조**:
  - Alex Krizhevsky가 2012년에 제안한 CNN.
  - ImageNet 대회에서 우승하여 딥러닝의 발전을 이끌음.
- **구성 요소**:
  - 다섯 개의 컨볼루션 레이어 + 풀링 레이어.
  - 세 개의 완전 연결층.
  - ReLU 활성화 함수와 Dropout 사용.
- **특징**:
  - GPU를 활용한 대규모 데이터셋 학습.
  - 데이터 증강 기법 사용.

#### 1.3 VGG (15분)
- **VGG의 구조**:
  - 2014년에 Visual Geometry Group이 제안한 CNN.
  - ImageNet 대회에서 우수한 성능을 기록.
- **구성 요소**:
  - 작은 필터(3x3)를 사용한 다수의 컨볼루션 레이어.
  - 다섯 개의 컨볼루션 블록 + 풀링 레이어.
  - 세 개의 완전 연결층.
- **특징**:
  - 깊은 구조, 높은 연산 비용.
  - 더 작은 필터를 사용하여 복잡한 특징 추출.

#### 1.4 ResNet (15분)
- **ResNet의 구조**:
  - Kaiming He가 2015년에 제안한 ResNet(Residual Network).
  - ImageNet 대회에서 우수한 성능을 기록.
- **구성 요소**:
  - 잔차 연결(Residual Connections)을 사용하여 깊은 네트워크 학습.
  - 기본 블록과 병렬로 연결된 단순화된 경로.
- **특징**:
  - 매우 깊은 네트워크 학습 가능 (50, 101, 152 레이어).
  - 기울기 소실 문제 해결.

---

### 2. 모델 리뷰 및 토론 (1시간)

#### 2.1 모델 리뷰 (30분)
- **LeNet, AlexNet, VGG, ResNet 모델 리뷰**:
  - 각 모델의 구조, 장단점, 적용 사례.
- **비교 분석**:
  - 모델 간의 차이점과 발전 방향.
  - 각 모델의 특징 및 성능 비교.

##### 리뷰 자료
- LeNet 논문: "Gradient-Based Learning Applied to Document Recognition" by Yann LeCun et al.
- AlexNet 논문: "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.
- VGG 논문: "Very Deep Convolutional Networks for Large-Scale Image Recognition" by Karen Simonyan and Andrew Zisserman.
- ResNet 논문: "Deep Residual Learning for Image Recognition" by Kaiming He et al.

#### 2.2 토론 (30분)
- **질문 및 토론 주제**:
  - 각 모델의 주요 혁신점은 무엇인가?
  - 각 모델의 단점과 한계는 무엇인가?
  - 실생활 응용에서 어떤 모델이 가장 적합한가?
  - 앞으로의 CNN 연구 방향은 무엇인가?

### 준비 자료
- **강의 자료**: CNN 심화 (LeNet, AlexNet, VGG, ResNet) 슬라이드 (PDF)
- **참고 논문**: 각 모델의 원본 논문 (PDF)

### 과제
- **이론 정리**: 각 CNN 모델의 구조와 원리 정리.
- **논문 읽기**: LeNet, AlexNet, VGG, ResNet 논문 읽고 요약 작성.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 다양한 CNN 모델의 구조와 원리를 이해하고, 각 모델의 발전 방향과 실생활 응용에 대해 토론할 수 있도록 유도합니다.