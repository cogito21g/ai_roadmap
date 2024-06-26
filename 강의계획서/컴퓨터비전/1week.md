### 1주차 강의 상세 계획: 컴퓨터 비전 개요

#### 강의 목표
- 컴퓨터 비전의 기본 개념과 역사 이해
- 컴퓨터 비전의 주요 응용 분야 파악
- ImageNet Classification 논문 읽기 및 토론

#### 강의 구성
- **이론 강의**: 1시간
- **논문 읽기 및 토론**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 컴퓨터 비전의 정의와 기본 개념 (15분)
- **컴퓨터 비전이란?**:
  - 컴퓨터가 이미지나 비디오에서 의미 있는 정보를 추출하고 해석하는 기술.
  - 인간의 시각적 인지 능력을 컴퓨터가 모방하는 기술.

#### 1.2 컴퓨터 비전의 역사 (15분)
- **역사적 배경**:
  - 1960년대: 초기 이미지 분석 연구 시작.
  - 1990년대: 기계 학습 기법 도입.
  - 2010년대: 딥러닝의 발전과 함께 컴퓨터 비전 기술의 급속한 발전.

#### 1.3 컴퓨터 비전의 주요 응용 분야 (30분)
- **응용 분야**:
  - **의료**: 의료 영상 분석, 암 진단.
  - **자동차**: 자율 주행, 교통 신호 인식.
  - **보안**: 얼굴 인식, 영상 감시.
  - **엔터테인먼트**: 증강 현실(AR), 가상 현실(VR).
  - **산업**: 공정 자동화, 품질 검사.
- **사례 연구**:
  - ImageNet 대회와 딥러닝의 발전.
  - 자율 주행 자동차의 비전 시스템.

---

### 2. 논문 읽기 및 토론 (1시간)

#### 2.1 논문 소개
- **논문 제목**: ImageNet Classification with Deep Convolutional Neural Networks
- **저자**: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
- **발표 연도**: 2012

#### 2.2 논문 핵심 내용
- **CNN 구조**:
  - AlexNet의 구조와 주요 구성 요소.
  - ReLU 활성화 함수, Dropout, 데이터 증강.
- **학습 방법**:
  - 대규모 데이터셋(ImageNet) 사용.
  - GPU 가속을 통한 효율적인 학습.
- **실험 결과**:
  - ImageNet 대회에서의 성능.
  - 기존 기술 대비 성능 향상.

#### 2.3 논문 토론
- **질문**:
  - AlexNet의 주요 혁신점은 무엇인가?
  - AlexNet이 기존 모델과 비교했을 때 어떤 점에서 우수한가?
  - 논문의 한계점과 개선 가능성은 무엇인가?
- **토론 주제**:
  - CNN의 발전 방향과 최신 연구 동향.
  - AlexNet 이후의 주요 CNN 모델 (VGG, ResNet 등).

### 준비 자료
- **강의 자료**: 컴퓨터 비전 개요 슬라이드 (PDF)
- **참고 논문**: "ImageNet Classification with Deep Convolutional Neural Networks" (PDF)

### 과제
- **논문 읽기**: "ImageNet Classification with Deep Convolutional Neural Networks" 논문을 읽고 요약 작성.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

---

### 2주차 강의 상세 계획: 이미지 처리 기본

#### 강의 목표
- 이미지 처리의 기본 개념과 주요 기술 이해
- OpenCV를 이용한 이미지 처리 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 이미지 필터링 (20분)
- **정의**: 이미지 필터링은 이미지의 각 픽셀 값을 주변 픽셀 값과 결합하여 새로운 이미지로 변환하는 과정.
- **필터 종류**:
  - **평균 필터**: 이미지의 노이즈를 제거하는 데 사용.
  - **가우시안 필터**: 이미지의 블러링 효과를 위해 사용.
  - **샤프닝 필터**: 이미지의 경계를 더 뚜렷하게 만드는 필터.

#### 1.2 엣지 검출 (20분)
- **정의**: 이미지에서 경계선을 검출하는 과정.
- **대표적 알고리즘**:
  - **소벨 필터**: 엣지를 강조하기 위한 필터.
  - **캐니 엣지 검출기**: 멀티스테이지 엣지 검출 알고리즘.

#### 1.3 히스토그램 평활화 (20분)
- **정의**: 이미지의 명암 대비를 향상시키기 위해 히스토그램을 균일하게 분포시키는 과정.
- **적용 사례**:
  - 저조도 이미지의 명암 대비 향상.
  - 의료 영상의 명확성 향상.

---
